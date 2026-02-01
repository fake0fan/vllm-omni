"""ZMQ-based stage engine process for vLLM-Omni multi-stage pipelines.

This module provides StageEngineCoreProc, a ZMQ-based server that wraps
AsyncOmniLLM or AsyncOmniDiffusion engines and handles request processing
in a background process.

Based on vLLM's EngineCoreProc pattern (vllm/v1/engine/core.py).
"""

import asyncio
import queue
import signal
import threading
import time
from contextlib import ExitStack
from typing import Any

import zmq
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.stage_engine.stage_serialization import (
    StageMsgpackDecoder,
    StageMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

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

    Attributes:
        stage_id: Unique identifier for this stage
        stage_config: Configuration for this stage
        input_address: ZMQ address for receiving requests (DEALER)
        output_address: ZMQ address for sending outputs (PUSH)
        engine: The underlying AsyncOmniLLM or AsyncOmniDiffusion engine
    """

    def __init__(
        self,
        stage_id: int,
        stage_config: Any,
        input_address: str,
        output_address: str,
        stage_type: str = "llm",
    ):
        """Initialize the stage engine process.

        Args:
            stage_id: Unique identifier for this stage
            stage_config: Configuration for this stage
            input_address: ZMQ address for receiving requests
            output_address: ZMQ address for sending outputs
            stage_type: Type of stage ("llm" or "diffusion")
        """
        self.stage_id = stage_id
        self.stage_config = stage_config
        self.input_address = input_address
        self.output_address = output_address
        self.stage_type = stage_type

        # Queues for request/output handling
        self.input_queue: queue.Queue[tuple[bytes, Any]] = queue.Queue()
        self.output_queue: queue.Queue[tuple[bytes, bytes, list[bytes]]] = queue.Queue()

        # Engine instance (initialized in run_stage_loop)
        self.engine: AsyncOmniLLM | AsyncOmniDiffusion | None = None

        # Serialization
        self.encoder = StageMsgpackEncoder()
        self.decoder = StageMsgpackDecoder()

        # Shutdown flag
        self.shutdown_requested = False

    @staticmethod
    def run_stage_worker(
        stage_id: int,
        stage_config: Any,
        input_address: str,
        output_address: str,
        stage_type: str = "llm",
    ):
        """Entry point for running stage worker in background process.

        Args:
            stage_id: Unique identifier for this stage
            stage_config: Configuration for this stage
            input_address: ZMQ address for receiving requests
            output_address: ZMQ address for sending outputs
            stage_type: Type of stage ("llm" or "diffusion")
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
            proc = StageEngineCoreProc(
                stage_id, stage_config, input_address, output_address, stage_type
            )
            proc.run_stage_loop()
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

    def run_stage_loop(self):
        """Main loop for stage worker process.

        Initializes the engine, starts IO threads, and processes requests.
        """
        # Initialize the engine
        logger.info(f"Initializing stage {self.stage_id} engine")
        self._init_engine()

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

    def _init_engine(self):
        """Initialize the stage engine (AsyncOmniLLM or AsyncOmniDiffusion)."""
        engine_args = self.stage_config.engine_args

        if self.stage_type == "llm":
            # Initialize AsyncOmniLLM
            self.engine = AsyncOmniLLM(
                model=engine_args.model,
                **engine_args.to_dict(),
            )
        elif self.stage_type == "diffusion":
            # Initialize AsyncOmniDiffusion
            self.engine = AsyncOmniDiffusion(
                model=engine_args.model,
                **engine_args.to_dict(),
            )
        else:
            raise ValueError(f"Unknown stage type: {self.stage_type}")

        logger.info(f"Stage {self.stage_id} engine initialized")

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

            # Send initial empty message to establish connection
            socket.send(b"")

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
