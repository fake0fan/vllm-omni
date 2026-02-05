# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage engine launcher - maximally reuses vLLM's launch infrastructure.

For LLM stages: directly use vLLM's launch_core_engines() and EngineCoreClient.
For non-LLM stages: use vLLM's infrastructure with custom stage worker.

This module reuses:
- vLLM's EngineZmqAddresses, EngineHandshakeMetadata
- vLLM's CoreEngineProcManager pattern
- vLLM's wait_for_engine_startup pattern
- vLLM's get_engine_client_zmq_addr for address generation
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING

import msgspec
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
)
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

STARTUP_POLL_PERIOD_MS = 10000


class StageState(Enum):
    """State of a stage during handshake."""
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


@dataclass
class StageInfo:
    """Information about a stage for launching.

    Attributes:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("llm", "diffusion", "audio")
        vllm_config: VllmConfig for LLM stages (None for non-LLM stages)
    """
    stage_id: int
    stage_type: str
    vllm_config: VllmConfig | None = None


class StageHandle:
    """Handle for a stage during handshake, tracks state."""

    def __init__(self, stage_id: int, stage_type: str):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.identity = stage_id.to_bytes(2, "little")
        self.state = StageState.NEW


class StageProcManager:
    """Manages non-LLM stage worker processes.

    Reuses vLLM's CoreEngineProcManager pattern.
    For LLM stages, use vLLM's launch_core_engines() directly.
    """

    def __init__(
        self,
        target_fn,
        stage_infos: list[StageInfo],
        handshake_address: str,
        executor_class: type["Executor"] | None,
        log_stats: bool,
    ):
        """Initialize and start stage worker processes.

        Args:
            target_fn: Function to run in each worker process
            stage_infos: List of StageInfo for each stage
            handshake_address: ZMQ address for handshake communication
            executor_class: Executor class for stages
            log_stats: Whether to log statistics
        """
        context = get_mp_context()

        self.processes: list[BaseProcess] = []
        for info in stage_infos:
            proc = context.Process(
                target=target_fn,
                name=f"Stage_{info.stage_id}_{info.stage_type}",
                kwargs={
                    "stage_id": info.stage_id,
                    "stage_type": info.stage_type,
                    "vllm_config": info.vllm_config,
                    "handshake_address": handshake_address,
                    "executor_class": executor_class,
                    "log_stats": log_stats,
                },
            )
            self.processes.append(proc)

        self._finalizer = weakref.finalize(self, shutdown, self.processes)

        try:
            for proc in self.processes:
                proc.start()
        finally:
            if self.finished_procs():
                self.close()

    def close(self):
        """Shutdown all processes."""
        self._finalizer()

    def sentinels(self) -> list:
        """Get process sentinels for polling."""
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


@contextlib.contextmanager
def launch_non_llm_stages(
    stage_infos: list[StageInfo],
    executor_class: type["Executor"] | None,
    log_stats: bool,
    local_only: bool = True,
    host: str = "localhost",
) -> Iterator[tuple[StageProcManager, dict[int, EngineZmqAddresses]]]:
    """Launch non-LLM stage engine processes with handshake.

    For LLM stages, use vLLM's launch_core_engines() directly instead.

    This function reuses vLLM's EngineZmqAddresses and handshake pattern.

    Args:
        stage_infos: List of StageInfo for each non-LLM stage
        executor_class: Executor class for stages
        log_stats: Whether to log statistics
        local_only: If True, use IPC addresses; otherwise TCP
        host: Host for TCP addresses

    Yields:
        Tuple of (StageProcManager, dict mapping stage_id to EngineZmqAddresses)
    """
    from vllm_omni.entrypoints.stage_engine.stage_core_proc import (
        StageEngineCoreProc,
    )

    # Generate ZMQ addresses for each stage using vLLM's utility
    stage_addresses: dict[int, EngineZmqAddresses] = {}
    for info in stage_infos:
        stage_addresses[info.stage_id] = EngineZmqAddresses(
            inputs=[get_engine_client_zmq_addr(local_only, host)],
            outputs=[get_engine_client_zmq_addr(local_only, host)],
        )

    # Create handshake address
    handshake_address = get_open_zmq_ipc_path()

    # Create stage handles for tracking handshake state
    stage_handles = [
        StageHandle(info.stage_id, info.stage_type)
        for info in stage_infos
    ]

    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        # Start stage worker processes
        proc_manager = StageProcManager(
            target_fn=StageEngineCoreProc.run_stage_worker,
            stage_infos=stage_infos,
            handshake_address=handshake_address,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        # Wait for all stages to complete handshake
        wait_for_stage_startup(
            handshake_socket=handshake_socket,
            stage_handles=stage_handles,
            stage_addresses=stage_addresses,
            proc_manager=proc_manager,
        )

        yield proc_manager, stage_addresses


def wait_for_stage_startup(
    handshake_socket: zmq.Socket,
    stage_handles: list[StageHandle],
    stage_addresses: dict[int, EngineZmqAddresses],
    proc_manager: StageProcManager,
):
    """Wait for stage processes to complete startup handshake.

    Reuses vLLM's wait_for_engine_startup pattern with EngineHandshakeMetadata.

    Args:
        handshake_socket: ZMQ ROUTER socket for handshake
        stage_handles: List of StageHandle for tracking state
        stage_addresses: Dict mapping stage_id to EngineZmqAddresses
        proc_manager: Process manager for checking process health
    """
    conn_pending = len(stage_handles)
    start_pending = 0

    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    for sentinel in proc_manager.sentinels():
        poller.register(sentinel, zmq.POLLIN)

    while conn_pending > 0 or start_pending > 0:
        events = poller.poll(STARTUP_POLL_PERIOD_MS)
        if not events:
            if conn_pending > 0:
                logger.debug(
                    "Waiting for %d stage proc(s) to connect.",
                    conn_pending,
                )
            if start_pending > 0:
                logger.debug(
                    "Waiting for %d stage proc(s) to start.",
                    start_pending,
                )
            continue

        if len(events) > 1 or events[0][0] != handshake_socket:
            finished = proc_manager.finished_procs()
            raise RuntimeError(
                "Stage engine initialization failed. "
                f"Failed stage proc(s): {finished}"
            )

        # Receive HELLO and READY messages
        stage_identity, msg_bytes = handshake_socket.recv_multipart()
        stage_id = int.from_bytes(stage_identity, "little")

        handle = next(
            (h for h in stage_handles if h.identity == stage_identity),
            None,
        )
        if handle is None:
            raise RuntimeError(
                f"Message from stage with unexpected id: {stage_id}"
            )

        msg = msgspec.msgpack.decode(msg_bytes)
        status = msg["status"]

        if status == "HELLO" and handle.state == StageState.NEW:
            # Send init message using vLLM's EngineHandshakeMetadata
            addresses = stage_addresses[stage_id]
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={},  # No DP config for non-LLM stages
                )
            )
            handshake_socket.send_multipart(
                (stage_identity, init_message),
                copy=False,
            )
            conn_pending -= 1
            start_pending += 1
            handle.state = StageState.CONNECTED
            logger.debug(
                "Stage %d (%s) connected, sent addresses",
                stage_id,
                handle.stage_type,
            )

        elif status == "READY" and handle.state == StageState.CONNECTED:
            start_pending -= 1
            handle.state = StageState.READY
            logger.info(
                "Stage %d (%s) ready",
                stage_id,
                handle.stage_type,
            )

        else:
            raise RuntimeError(
                f"Unexpected {status} message for stage {stage_id} "
                f"in {handle.state} state."
            )
