"""
Omni core engine utilities for single-stage deployment mode.

Provides:
- OmniMasterServer: listens for engine core registrations, handles the
  vLLM HELLO/READY handshake and signals per-stage readiness events.
- run_omni_engine_core: target function for CoreEngineProcManager that
  first registers with the master to obtain a handshake address, then
  delegates to EngineCoreProc.run_engine_core.
- launch_omni_core_engines: context-manager replacing launch_core_engines
  in single-stage mode; starts local engine processes and waits for the
  master server to confirm all local engines are READY.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port, zmq_socket_ctx
from vllm.v1.engine.utils import (
    STARTUP_POLL_PERIOD_MS,
    CoreEngine,
    CoreEngineProcManager,
    CoreEngineState,
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    wait_for_engine_startup,
)
from vllm.v1.executor import Executor

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# Poll period (ms) used by the registration/handshake loop.
_POLL_PERIOD_MS = 5_000
# Default timeout (s) for a stage to send READY.
_DEFAULT_STARTUP_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Per-stage address allocation
# ---------------------------------------------------------------------------


@dataclass
class StageAllocation:
    """ZMQ address allocation for one remote (or locally-registered) stage.

    *_bind_address   – address that the client-side socket (StageEngineCoreClient,
                       or OmniMasterServer handshake socket) binds to.
    *_connect_address – address that the engine-core DEALER/PUSH socket
                       connects to; must be reachable from the engine host.
    """

    # Per-stage handshake socket (OmniMasterServer binds, engine connects)
    handshake_bind_address: str
    handshake_connect_address: str
    # Input channel: client binds ROUTER, engine connects DEALER
    input_bind_address: str
    input_connect_address: str
    # Output channel: client binds PULL, engine connects PUSH
    output_bind_address: str
    output_connect_address: str


# ---------------------------------------------------------------------------
# OmniMasterServer
# ---------------------------------------------------------------------------


class OmniMasterServer:
    """Master server that orchestrates engine startup in single-stage mode.

    Lifecycle
    ---------
    1. Created with the master's reachable address/port and the list of all
       stage IDs it will manage (both the locally-launched stage and any
       remote stages).
    2. ``start()`` launches a background thread that:
       a. Binds a ROUTER registration socket at
          ``tcp://<master_address>:<master_port>``.
       b. Binds per-stage handshake ROUTER sockets at
          ``tcp://<master_address>:<allocated_handshake_port>``.
       c. When an engine core registers (sends ``{"stage_id": N}``), it
          responds with ``{"handshake_address": ...}``.
       d. When the engine connects to the handshake socket and sends HELLO,
          it responds with ``EngineHandshakeMetadata`` containing the
          pre-allocated engine connect addresses.
       e. When the engine sends READY, it updates the CacheConfig and sets
          the per-stage ``threading.Event``.
    3. Callers use ``wait_for_stage_ready(stage_id, timeout)`` to block
       until the engine is fully initialised.
    """

    def __init__(
        self,
        master_address: str,
        master_port: int,
        stage_ids: list[int],
    ) -> None:
        self._address = master_address
        self._port = master_port
        self._allocations: dict[int, StageAllocation] = {}
        self._cache_configs: dict[int, CacheConfig] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        for sid in stage_ids:
            hs_port = get_open_port()
            inp_port = get_open_port()
            out_port = get_open_port()
            self._allocations[sid] = StageAllocation(
                handshake_bind_address=f"tcp://{master_address}:{hs_port}",
                handshake_connect_address=f"tcp://{master_address}:{hs_port}",
                input_bind_address=f"tcp://{master_address}:{inp_port}",
                input_connect_address=f"tcp://{master_address}:{inp_port}",
                output_bind_address=f"tcp://{master_address}:{out_port}",
                output_connect_address=f"tcp://{master_address}:{out_port}",
            )

        logger.info(
            "[OmniMasterServer] Pre-allocated addresses for stages %s (master=%s:%d)",
            list(stage_ids),
            master_address,
            master_port,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register_cache_config(self, stage_id: int, cache_config: CacheConfig) -> None:
        """Register a CacheConfig whose num_gpu_blocks will be updated on READY."""
        self._cache_configs[stage_id] = cache_config

    def get_allocation(self, stage_id: int) -> StageAllocation:
        """Return the full address allocation for *stage_id*."""
        return self._allocations[stage_id]

    def get_client_addresses(self, stage_id: int) -> dict[str, str]:
        """Return the addresses the client-side sockets should *bind* to."""
        alloc = self._allocations[stage_id]
        return {
            "input_address": alloc.input_bind_address,
            "output_address": alloc.output_bind_address,
        }

    def get_zmq_addresses(self, stage_id: int) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *bind* (client) side addresses."""
        alloc = self._allocations[stage_id]
        return EngineZmqAddresses(
            inputs=[alloc.input_bind_address],
            outputs=[alloc.output_bind_address],
        )

    def get_engine_zmq_addresses(self, stage_id: int) -> EngineZmqAddresses:
        """Return EngineZmqAddresses using the *connect* (engine) addresses."""
        alloc = self._allocations[stage_id]
        return EngineZmqAddresses(
            inputs=[alloc.input_connect_address],
            outputs=[alloc.output_connect_address],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background server thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="OmniMasterServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[OmniMasterServer] Listening on tcp://%s:%d",
            self._address,
            self._port,
        )

    def stop(self) -> None:
        """Signal stop and join the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------
    # Internal server logic
    # ------------------------------------------------------------------

    def _run(self) -> None:
        ctx = zmq.Context()
        try:
            self._serve(ctx)
        except Exception:
            logger.exception("[OmniMasterServer] Server thread crashed")
        finally:
            ctx.term()

    def _serve(self, ctx: zmq.Context) -> None:  # type: ignore[type-arg]
        # Registration socket – all engines hit this first.
        # Per-stage handshake sockets are NOT created here; they are already
        # bound by the stage core client (vLLM's MPClient) and their connect
        # addresses are simply forwarded in the registration reply.
        reg_socket: zmq.Socket = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        reg_socket.bind(f"tcp://{self._address}:{self._port}")

        poller = zmq.Poller()
        poller.register(reg_socket, zmq.POLLIN)

        pending: set[int] = set(self._allocations.keys())

        while pending and not self._stop_event.is_set():
            events: list[tuple[zmq.Socket, int]] = poller.poll(_POLL_PERIOD_MS)  # type: ignore[assignment]
            if not events:
                logger.debug("[OmniMasterServer] Still waiting for registration from stages: %s", pending)
                continue

            for sock, _ in events:
                if sock is reg_socket:
                    sid = self._handle_registration(reg_socket)
                    if sid is not None:
                        pending.discard(sid)

        # Cleanup
        reg_socket.close(linger=0)
        logger.info("[OmniMasterServer] All stages registered; server thread exiting.")

    def _handle_registration(self, reg_socket: zmq.Socket) -> int | None:  # type: ignore[type-arg]
        """Receive a stage registration and reply with the handshake address.

        Returns the registered stage_id on success, or None on failure.
        """
        frames = reg_socket.recv_multipart()
        if len(frames) < 2:
            logger.warning(
                "[OmniMasterServer] Unexpected registration frame count: %d",
                len(frames),
            )
            return None
        identity = frames[0]
        msg_bytes = frames[-1]
        try:
            msg = msgspec.msgpack.decode(msg_bytes)
        except Exception as exc:
            logger.warning("[OmniMasterServer] Failed to decode registration message: %s", exc)
            return None

        stage_id: int | None = msg.get("stage_id")
        if stage_id not in self._allocations:
            logger.warning(
                "[OmniMasterServer] Received registration for unknown stage_id=%s",
                stage_id,
            )
            return None

        alloc = self._allocations[stage_id]
        response = msgspec.msgpack.encode({"handshake_address": alloc.handshake_connect_address})
        # ROUTER-DEALER: reply is [identity, payload] (no empty delimiter).
        reg_socket.send_multipart([identity, response])
        logger.info(
            "[OmniMasterServer] Stage %d registered; assigned handshake=%s",
            stage_id,
            alloc.handshake_connect_address,
        )
        return stage_id


def run_omni_engine_core(
    *args: Any,
    omni_master_address: str,
    omni_master_port: int,
    omni_stage_id: int,
    dp_rank: int = 0,
    local_dp_rank: int = 0,
    **kwargs: Any,
) -> None:
    """Engine-core target function for single-stage (omni) mode.

    Steps
    -----
    1. Connect a ZMQ DEALER socket to the OmniMasterServer registration
       socket at ``tcp://<omni_master_address>:<omni_master_port>``.
    2. Send ``{"stage_id": omni_stage_id}`` and receive
       ``{"handshake_address": <addr>}`` in reply.
    3. Override ``kwargs["handshake_address"]`` with the received address.
    4. Delegate to ``EngineCoreProc.run_engine_core`` with the updated
       kwargs so that the normal vLLM HELLO/READY handshake proceeds
       against the OmniMasterServer's per-stage handshake ROUTER socket.
    """
    from vllm.v1.engine.core import EngineCoreProc

    # --- Step 1 & 2: register with master, obtain handshake address ---
    reg_ctx = zmq.Context()
    try:
        reg_sock: zmq.Socket = reg_ctx.socket(zmq.DEALER)  # type: ignore[attr-defined]
        try:
            reg_sock.connect(f"tcp://{omni_master_address}:{omni_master_port}")
            reg_sock.send(msgspec.msgpack.encode({"stage_id": omni_stage_id}))
            timeout_ms = _DEFAULT_STARTUP_TIMEOUT_S * 1_000
            if not reg_sock.poll(timeout=timeout_ms):
                raise RuntimeError(
                    f"[run_omni_engine_core] Timed out waiting for registration "
                    f"response from OmniMasterServer "
                    f"({omni_master_address}:{omni_master_port}) "
                    f"for stage {omni_stage_id}."
                )
            response_bytes = reg_sock.recv()
            response = msgspec.msgpack.decode(response_bytes)
            handshake_address: str = response["handshake_address"]
            logger.info(
                "[run_omni_engine_core] Stage %d registered; handshake_address=%s",
                omni_stage_id,
                handshake_address,
            )
        finally:
            reg_sock.close(linger=0)
    finally:
        reg_ctx.term()

    # --- Step 3: override handshake address ---
    kwargs["handshake_address"] = handshake_address

    # --- Step 4: run standard engine core ---
    EngineCoreProc.run_engine_core(
        *args,
        dp_rank=dp_rank,
        local_dp_rank=local_dp_rank,
        **kwargs,
    )


def _wait_for_omni_engine_startup(
    handshake_socket: zmq.Socket,
    engine_addresses: EngineZmqAddresses,
    engines: list[CoreEngine],
    cache_config: CacheConfig,
) -> None:
    """HELLO/READY handshake loop for omni-managed engine cores.

    Unlike :func:`~vllm.v1.engine.utils.wait_for_engine_startup`, this
    function does **not** validate the ``local`` / ``headless`` fields that
    vLLM uses to distinguish co-located from remote engines, because in omni
    single-stage mode the remote engines are launched by a separate command and
    their reported locality has no meaning to this host.
    """
    conn_pending = len(engines)
    start_pending = 0

    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    while conn_pending or start_pending:
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            logger.debug(
                "[omni] Waiting for %d engine(s) to connect, %d to start.",
                conn_pending,
                start_pending,
            )
            continue

        eng_identity, msg_bytes = handshake_socket.recv_multipart()
        eng_index = int.from_bytes(eng_identity, "little")
        engine = next((e for e in engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(f"[omni] Handshake message from unexpected engine rank: {eng_index}")

        msg = msgspec.msgpack.decode(msg_bytes)
        status: str = msg["status"]

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(addresses=engine_addresses, parallel_config={})
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            conn_pending -= 1
            start_pending += 1
            engine.state = CoreEngineState.CONNECTED
            logger.debug("[omni] HELLO from engine %d", eng_index)

        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            num_gpu_blocks = (cache_config.num_gpu_blocks or 0) + msg["num_gpu_blocks"]
            cache_config.num_gpu_blocks = num_gpu_blocks
            if engine_addresses.frontend_stats_publish_address is None:
                engine_addresses.frontend_stats_publish_address = msg.get("dp_stats_address")
            start_pending -= 1
            engine.state = CoreEngineState.READY
            logger.debug("[omni] READY from engine %d (num_gpu_blocks=%d)", eng_index, msg["num_gpu_blocks"])

        else:
            raise RuntimeError(f"[omni] Unexpected status '{status}' from engine {eng_index} in state {engine.state}.")


@contextlib.contextmanager
def connect_remote_engine_cores(
    vllm_config: VllmConfig,
    addresses: EngineZmqAddresses,
    omni_master_server: OmniMasterServer,
    stage_id: int,
) -> Iterator[tuple[None, None, EngineZmqAddresses]]:
    """Context manager that waits for a remotely-launched engine-core to
    complete the standard vLLM HELLO/READY handshake.

    Unlike :func:`launch_omni_core_engines`, this function does **not** start
    any subprocesses.  It is used in single-stage mode for stages whose
    processes are launched by a remote host; this side only needs to bind the
    pre-allocated handshake ROUTER socket and wait for the engine to connect.

    Parameters
    ----------
    vllm_config:
        Configuration for this stage (used to derive engine count and ranks).
    addresses:
        ``EngineZmqAddresses`` with client-bind addresses for this stage.
    omni_master_server:
        The running :class:`OmniMasterServer` instance.
    stage_id:
        The stage ID of the remote engine being awaited.

    Yields
    ------
    ``(None, None, addresses)`` — mirrors the return signature of
    :func:`~vllm.v1.engine.utils.launch_core_engines` with no process manager.
    """
    parallel_config = vllm_config.parallel_config
    # Mirror the engine-count logic from launch_omni_core_engines.
    remote_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, parallel_config.data_parallel_size)
    )
    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    start_index = dp_rank

    # Register the stage's cache config so OmniMasterServer can update it.
    omni_master_server.register_cache_config(stage_id, vllm_config.cache_config)

    engines_to_handshake = [CoreEngine(index=start_index + i, local=True) for i in range(remote_engine_count)]

    logger.info(
        "[connect_remote_engine_cores] Waiting for %d remote engine(s) for stage %d (dp_rank=%d)",
        remote_engine_count,
        stage_id,
        dp_rank,
    )

    handshake_bind_address = omni_master_server.get_allocation(stage_id).handshake_bind_address
    engine_addresses = omni_master_server.get_engine_zmq_addresses(stage_id)

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        yield None, None, addresses

        _wait_for_omni_engine_startup(
            handshake_socket,
            engine_addresses,
            engines_to_handshake,
            vllm_config.cache_config,
        )
        if addresses.frontend_stats_publish_address is None:
            addresses.frontend_stats_publish_address = engine_addresses.frontend_stats_publish_address


@contextlib.contextmanager
def launch_omni_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    omni_master_server: OmniMasterServer,
    stage_id: int,
) -> Iterator[tuple[CoreEngineProcManager, None, EngineZmqAddresses]]:
    """Context manager that launches local engine-core processes using the
    omni registration protocol instead of the standard vLLM handshake.

    Parameters
    ----------
    vllm_config:
        Configuration for this stage.
    executor_class:
        Executor class to pass through to the engine core.
    log_stats:
        Whether to log stats.
    addresses:
        ``EngineZmqAddresses`` whose *inputs* / *outputs* contain the
        **client-bind** addresses (``tcp://<master_address>:<port>``).  These are
        retrieved from ``omni_master_server.get_zmq_addresses(stage_id)``.
    omni_master_server:
        The running :class:`OmniMasterServer` instance.
    stage_id:
        The stage ID of the engines being launched.

    Yields
    ------
    ``(engine_manager, None, addresses)`` — mirrors the return signature of
    :func:`~vllm.v1.engine.utils.launch_core_engines`.
    """
    parallel_config = vllm_config.parallel_config
    # Determine the number of local engines and their ranks.
    local_engine_count = (
        parallel_config.data_parallel_size_local
        if parallel_config.data_parallel_size_local is not None and parallel_config.data_parallel_size_local > 0
        else max(1, parallel_config.data_parallel_size)
    )
    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    local_start_index = 0
    start_index = dp_rank

    # Register the stage's cache config so OmniMasterServer can update it.
    omni_master_server.register_cache_config(stage_id, vllm_config.cache_config)

    # Build the partial target function with the extra omni kwargs baked in.
    target_fn = functools.partial(
        run_omni_engine_core,
        omni_master_address=omni_master_server._address,
        omni_master_port=omni_master_server._port,
        omni_stage_id=stage_id,
    )

    logger.info(
        "[launch_omni_core_engines] Starting %d local engine(s) for stage %d (dp_rank=%d)",
        local_engine_count,
        stage_id,
        dp_rank,
    )

    # ``CoreEngineProcManager`` requires a ``handshake_address`` in common_kwargs.
    # We pass a placeholder; ``run_omni_engine_core`` will override it after
    # registering with the master.
    placeholder_handshake = "tcp://0.0.0.0:0"

    # One CoreEngine entry per local engine so wait_for_engine_startup can
    # track the HELLO/READY handshake for each of them.
    engines_to_handshake = [CoreEngine(index=start_index + i, local=True) for i in range(local_engine_count)]

    # Bind the per-stage handshake ROUTER socket that OmniMasterServer
    # already allocated.  ``run_omni_engine_core`` will obtain the
    # corresponding connect-address from the master and hand it to
    # ``EngineCoreProc.run_engine_core`` so the standard HELLO/READY
    # exchange hits this socket.
    handshake_bind_address = omni_master_server.get_allocation(stage_id).handshake_bind_address
    engine_addresses = omni_master_server.get_engine_zmq_addresses(stage_id)

    with zmq_socket_ctx(handshake_bind_address, zmq.ROUTER, bind=True) as handshake_socket:
        local_engine_manager = CoreEngineProcManager(
            target_fn=target_fn,
            local_engine_count=local_engine_count,
            start_index=start_index,
            local_start_index=local_start_index,
            vllm_config=vllm_config,
            local_client=True,
            handshake_address=placeholder_handshake,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield local_engine_manager, None, addresses

        # Wait for all local engine-core processes to complete the
        # standard HELLO/READY handshake — mirrors launch_core_engines.
        wait_for_engine_startup(
            handshake_socket,
            engine_addresses,
            engines_to_handshake,
            parallel_config,
            False,  # coordinated_dp: no DP coordinator in omni mode
            vllm_config.cache_config,
            local_engine_manager,
            None,  # coord_process
        )
        if addresses.frontend_stats_publish_address is None:
            addresses.frontend_stats_publish_address = engine_addresses.frontend_stats_publish_address
