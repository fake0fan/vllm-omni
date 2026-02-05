"""ZMQ-based stage engine process for vLLM-Omni multi-stage pipelines.

This module provides StageEngineCoreProc, a ZMQ-based server that wraps
AsyncOmniLLM or AsyncOmniDiffusion engines and handles request processing
in a background process.

Based on vLLM's EngineCoreProc pattern (vllm/v1/engine/core.py).

Reuses vLLM's:
- EngineHandshakeMetadata, EngineZmqAddresses for handshake
- startup_handshake pattern from EngineCoreProc

Startup flow (handshake):
    1. Worker starts and connects to handshake_address
    2. Worker sends HELLO message
    3. Parent sends EngineHandshakeMetadata with ZMQ addresses
    4. Worker connects to input/output addresses
    5. Worker sends READY message
    6. Worker enters main processing loop
"""

import asyncio
import queue
import signal
import threading
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
)

from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.stage_engine.stage_serialization import (
    StageMsgpackDecoder,
    StageMsgpackEncoder,
)
from vllm_omni.engine.input_processor import NonLLMInputProcessor
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

# Request types for stage communication
REQUEST_TYPE_GENERATE = b"GENERATE"
REQUEST_TYPE_ABORT = b"ABORT"
REQUEST_TYPE_SHUTDOWN = b"SHUTDOWN"
REQUEST_TYPE_HEALTH_CHECK = b"HEALTH_CHECK"

# Response types
RESPONSE_TYPE_OUTPUT = b"OUTPUT"
RESPONSE_TYPE_HEALTH = b"HEALTH"
RESPONSE_TYPE_ERROR = b"ERROR"
RESPONSE_TYPE_DEAD = b"DEAD"


class StageEngineCoreProc:
    """ZMQ-based server wrapping a stage engine (AsyncOmniLLM or AsyncOmniDiffusion).

    Runs in a background process and communicates with StageEngineCoreClient via ZMQ:
    - DEALER socket for receiving requests from orchestrator
    - PUSH socket for sending outputs to orchestrator

    This follows vLLM's EngineCoreProc pattern (vllm/v1/engine/core.py).

    Attributes:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("llm", "diffusion", "audio")
        input_address: ZMQ address for receiving requests (DEALER)
        output_address: ZMQ address for sending outputs (PUSH)
        vllm_config: VllmConfig for LLM stages
        executor_class: Executor class for the stage process
        log_stats: Whether to log statistics
        engine: The underlying AsyncOmniLLM or AsyncOmniDiffusion engine
    """

    def __init__(
        self,
        stage_id: int,
        stage_type: str,
        input_address: str,
        output_address: str,
        vllm_config: VllmConfig | None = None,
        executor_class: type["Executor"] | None = None,
        log_stats: bool = False,
    ):
        """Initialize the stage engine process.

        Args:
            stage_id: Unique identifier for this stage
            stage_type: Type of stage ("llm", "diffusion", "audio")
            input_address: ZMQ address for receiving requests
            output_address: ZMQ address for sending outputs
            vllm_config: VllmConfig for LLM stages (None for non-LLM stages)
            executor_class: Executor class for the stage process
            log_stats: Whether to log statistics
        """
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.input_address = input_address
        self.output_address = output_address
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.log_stats = log_stats

        # Queues for request/output handling
        self.input_queue: queue.Queue[tuple[bytes, Any]] = queue.Queue()
        self.output_queue: queue.Queue[tuple[bytes, bytes, list[bytes]]] = queue.Queue()

        # Engine instance (initialized in run_stage_loop)
        self.engine: AsyncOmniLLM | AsyncOmniDiffusion | None = None

        # Input processor for non-LLM stages (validates and normalizes inputs)
        self.input_processor: NonLLMInputProcessor | None = None
        if stage_type != "llm":
            self.input_processor = NonLLMInputProcessor(stage_type, config=vllm_config)

        # Serialization
        self.encoder = StageMsgpackEncoder()
        self.decoder = StageMsgpackDecoder()

        # Shutdown flag
        self.shutdown_requested = False

    @staticmethod
    def run_stage_worker(
        stage_id: int,
        stage_type: str,
        handshake_address: str,
        vllm_config: VllmConfig | None = None,
        executor_class: type["Executor"] | None = None,
        log_stats: bool = False,
    ):
        """Entry point for running stage worker in background process.

        This is called by StageProcManager to start the background process.
        Similar to vLLM's EngineCoreProc.run_engine_core().

        The worker performs handshake to receive ZMQ addresses from parent:
        1. Connect to handshake_address
        2. Send HELLO message
        3. Receive EngineHandshakeMetadata with input/output addresses
        4. Connect to those addresses
        5. Send READY message
        6. Enter main processing loop

        Args:
            stage_id: Unique identifier for this stage
            stage_type: Type of stage ("llm", "diffusion", "audio")
            handshake_address: ZMQ address for handshake with parent
            vllm_config: VllmConfig for LLM stages (None for non-LLM stages)
            executor_class: Executor class for the stage process
            log_stats: Whether to log statistics
        """
        # Signal handler for graceful shutdown
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = None
        try:
            logger.info(f"Starting stage {stage_id} worker process (type={stage_type})")

            # Perform handshake to get ZMQ addresses (returns EngineZmqAddresses)
            addresses = StageEngineCoreProc._do_handshake(
                stage_id=stage_id,
                stage_type=stage_type,
                handshake_address=handshake_address,
            )

            # EngineZmqAddresses has inputs/outputs as lists
            input_address = addresses.inputs[0]
            output_address = addresses.outputs[0]

            logger.info(
                f"Stage {stage_id} received addresses: "
                f"input={input_address}, output={output_address}"
            )

            proc = StageEngineCoreProc(
                stage_id=stage_id,
                stage_type=stage_type,
                input_address=input_address,
                output_address=output_address,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
            )
            proc.run_stage_loop(handshake_address)
        except SystemExit:
            logger.info(f"Stage {stage_id} worker exiting gracefully")
            raise
        except Exception as e:
            logger.exception(f"Stage {stage_id} worker encountered fatal error")
            if proc is not None:
                proc._send_dead_signal()
            raise e
        finally:
            if proc is not None:
                proc.shutdown()

    @staticmethod
    def _do_handshake(
        stage_id: int,
        stage_type: str,
        handshake_address: str,
    ) -> EngineZmqAddresses:
        """Perform handshake with parent to receive ZMQ addresses.

        Reuses vLLM's EngineHandshakeMetadata for the init message.

        Args:
            stage_id: Stage identifier
            stage_type: Type of stage
            handshake_address: ZMQ address for handshake

        Returns:
            EngineZmqAddresses with input/output addresses
        """
        identity = stage_id.to_bytes(2, "little")

        ctx = zmq.Context()
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            bind=False,
        ) as socket:
            # Send HELLO message (same format as vLLM's EngineCoreProc)
            hello_msg = msgspec.msgpack.encode({
                "status": "HELLO",
                "local": True,
                "headless": False,
            })
            socket.send(hello_msg)
            logger.debug(f"Stage {stage_id} sent HELLO")

            # Receive init message with addresses (vLLM's EngineHandshakeMetadata)
            init_msg_bytes = socket.recv()
            metadata: EngineHandshakeMetadata = msgspec.msgpack.decode(
                init_msg_bytes,
                type=EngineHandshakeMetadata,
            )
            logger.debug(f"Stage {stage_id} received handshake metadata")

            return metadata.addresses

    def run_stage_loop(self, handshake_address: str | None = None):
        """Main loop for stage worker process.

        Initializes the engine, sends READY, starts IO threads, and processes requests.

        Args:
            handshake_address: If provided, send READY message after init
        """
        # Initialize the engine
        logger.info(f"Initializing stage {self.stage_id} engine")
        self._init_engine()

        # Send READY message if handshake address provided
        if handshake_address:
            self._send_ready(handshake_address)

        # Start IO threads
        with ExitStack() as stack:
            # Input thread: ZMQ DEALER -> input_queue
            input_thread = threading.Thread(
                target=self._process_input_socket,
                daemon=True,
            )
            input_thread.start()
            stack.callback(lambda: input_thread.join(timeout=5.0))

            # Output thread: output_queue -> ZMQ PUSH
            output_thread = threading.Thread(
                target=self._process_output_socket,
                daemon=True,
            )
            output_thread.start()
            stack.callback(lambda: output_thread.join(timeout=5.0))

            # Main processing loop
            logger.info(f"Stage {self.stage_id} entering main loop")
            asyncio.run(self._async_main_loop())

    def _send_ready(self, handshake_address: str):
        """Send READY message to parent after engine initialization.

        Uses same format as vLLM's EngineCoreProc ready message.

        Args:
            handshake_address: ZMQ address for handshake
        """
        identity = self.stage_id.to_bytes(2, "little")

        ctx = zmq.Context()
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            bind=False,
        ) as socket:
            ready_msg = msgspec.msgpack.encode({
                "status": "READY",
                "local": True,
                "headless": False,
                "num_gpu_blocks": 0,  # Non-LLM stages don't have KV cache
            })
            socket.send(ready_msg)
            logger.debug(f"Stage {self.stage_id} sent READY")

    def _init_engine(self):
        """Initialize the stage engine (AsyncOmniLLM or AsyncOmniDiffusion).

        Uses vllm_config and executor_class passed from StageEngineCoreClient.
        """
        if self.stage_type == "llm":
            if self.vllm_config is None:
                raise ValueError("vllm_config is required for LLM stage")
            # Initialize AsyncOmniLLM with vllm_config
            # Use the executor_class if provided, otherwise get from vllm_config
            executor_cls = self.executor_class
            if executor_cls is None:
                from vllm.v1.executor import Executor
                executor_cls = Executor.get_class(self.vllm_config)
            self.engine = AsyncOmniLLM(
                vllm_config=self.vllm_config,
                executor_class=executor_cls,
                log_stats=self.log_stats,
            )
        elif self.stage_type == "diffusion":
            # Initialize AsyncOmniDiffusion
            # AsyncOmniDiffusion uses od_config, not vllm_config
            self.engine = AsyncOmniDiffusion(
                od_config=self.vllm_config,  # Pass as od_config
            )
        elif self.stage_type == "audio":
            # TODO: Initialize audio stage engine
            raise NotImplementedError("Audio stage not yet implemented")
        else:
            raise ValueError(f"Unknown stage type: {self.stage_type}")

        logger.info(f"Stage {self.stage_id} engine initialized (type={self.stage_type})")

    async def _async_main_loop(self):
        """Async main loop for processing requests."""
        while not self.shutdown_requested:
            # Process input queue
            try:
                request_type, request_data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            # Handle request
            if request_type == REQUEST_TYPE_GENERATE:
                await self._handle_generate_request(request_data)
            elif request_type == REQUEST_TYPE_ABORT:
                await self._handle_abort_request(request_data)
            elif request_type == REQUEST_TYPE_HEALTH_CHECK:
                self._handle_health_check()
            elif request_type == REQUEST_TYPE_SHUTDOWN:
                logger.info(f"Stage {self.stage_id} received shutdown request")
                self.shutdown_requested = True
                break
            else:
                logger.error(f"Unknown request type: {request_type}")

    async def _handle_generate_request(self, request_data: dict[str, Any]):
        """Handle a generate request.

        Args:
            request_data: Dictionary containing:
                - request_id: Unique request identifier
                - prompt: OmniPromptType
                - sampling_params: OmniSamplingParams
        """
        request_id = request_data["request_id"]
        prompt = request_data["prompt"]
        sampling_params = request_data["sampling_params"]

        try:
            # For non-LLM stages, use input processor to validate and normalize
            if self.input_processor is not None:
                processed = self.input_processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=sampling_params,
                )
                # Use processed values (may extract embeddings, etc.)
                prompt = processed.prompt
                sampling_params = processed.sampling_params

            # Generate outputs
            async for output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                # Serialize and send output
                self._send_output(request_id, output)

        except Exception as e:
            logger.exception(f"Error generating for request {request_id}")
            self._send_error(request_id, str(e))

    async def _handle_abort_request(self, request_data: dict[str, Any]):
        """Handle an abort request.

        Args:
            request_data: Dictionary containing:
                - request_id: Request ID to abort
        """
        request_id = request_data["request_id"]
        try:
            await self.engine.abort(request_id)
            logger.info(f"Aborted request {request_id}")
        except Exception as e:
            logger.exception(f"Error aborting request {request_id}")

    def _handle_health_check(self):
        """Handle a health check request."""
        self.output_queue.put_nowait((RESPONSE_TYPE_HEALTH, b"", []))

    def _send_output(self, request_id: str, output: OmniRequestOutput):
        """Send an output to the orchestrator.

        Args:
            request_id: Request identifier
            output: OmniRequestOutput to send
        """
        try:
            # Serialize output
            bufs = self.encoder.encode(output)
            # Put in output queue: (type, request_id, buffers)
            self.output_queue.put_nowait(
                (RESPONSE_TYPE_OUTPUT, request_id.encode("utf-8"), bufs)
            )
        except Exception as e:
            logger.exception(f"Error serializing output for {request_id}")
            self._send_error(request_id, f"Serialization error: {e}")

    def _send_error(self, request_id: str, error_msg: str):
        """Send an error response.

        Args:
            request_id: Request identifier
            error_msg: Error message
        """
        self.output_queue.put_nowait(
            (RESPONSE_TYPE_ERROR, request_id.encode("utf-8"), [error_msg.encode("utf-8")])
        )

    def _send_dead_signal(self):
        """Send a DEAD signal to indicate engine failure."""
        self.output_queue.put_nowait((RESPONSE_TYPE_DEAD, b"", []))
        # Wait for output thread to send
        time.sleep(1.0)

    def _process_input_socket(self):
        """Input socket IO thread: DEALER -> input_queue."""
        ctx = zmq.Context()
        with ctx.socket(zmq.DEALER) as socket:
            socket.connect(self.input_address)
            logger.info(f"Stage {self.stage_id} connected to input socket: {self.input_address}")

            while not self.shutdown_requested:
                try:
                    # Poll with timeout
                    if not socket.poll(timeout=100):
                        continue

                    # Receive multipart message: [type, data_frames...]
                    frames = socket.recv_multipart(copy=False)
                    if not frames:
                        continue

                    request_type = bytes(frames[0].buffer)
                    data_frames = frames[1:]

                    # Deserialize request data
                    if request_type == REQUEST_TYPE_GENERATE:
                        # Decode generate request
                        request_data = self.decoder.decode([bytes(f.buffer) for f in data_frames])
                        self.input_queue.put_nowait((request_type, request_data))
                    elif request_type in (REQUEST_TYPE_ABORT, REQUEST_TYPE_HEALTH_CHECK, REQUEST_TYPE_SHUTDOWN):
                        # Simple requests
                        request_data = {}
                        if data_frames:
                            request_data = self.decoder.decode([bytes(f.buffer) for f in data_frames])
                        self.input_queue.put_nowait((request_type, request_data))
                    else:
                        logger.error(f"Unknown request type: {request_type}")

                except Exception as e:
                    if not self.shutdown_requested:
                        logger.exception("Error in input socket thread")

    def _process_output_socket(self):
        """Output socket IO thread: output_queue -> PUSH."""
        ctx = zmq.Context()
        with ctx.socket(zmq.PUSH) as socket:
            socket.connect(self.output_address)
            logger.info(f"Stage {self.stage_id} connected to output socket: {self.output_address}")

            while not self.shutdown_requested:
                try:
                    # Get from output queue with timeout
                    response_type, request_id, bufs = self.output_queue.get(timeout=0.1)

                    # Send multipart message: [type, request_id, data_frames...]
                    frames = [response_type, request_id] + bufs
                    socket.send_multipart(frames)

                except queue.Empty:
                    continue
                except Exception as e:
                    if not self.shutdown_requested:
                        logger.exception("Error in output socket thread")

    def shutdown(self):
        """Shutdown the stage worker."""
        logger.info(f"Shutting down stage {self.stage_id} worker")
        self.shutdown_requested = True

        # Close engine
        if self.engine is not None:
            try:
                # AsyncOmniLLM/AsyncOmniDiffusion may have cleanup methods
                if hasattr(self.engine, "shutdown"):
                    self.engine.shutdown()
            except Exception as e:
                logger.exception(f"Error shutting down engine: {e}")
