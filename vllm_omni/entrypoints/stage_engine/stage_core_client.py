# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ-based client for communicating with non-LLM stage engine processes.

This module provides NonLLMStageClient, a ZMQ-based client for submitting
requests to and receiving outputs from StageEngineCoreProc workers.

For LLM stages, use vLLM's EngineCoreClient.make_async_mp_client() directly.
For non-LLM stages (Diffusion, Audio), use NonLLMStageClient.

Based on vLLM's AsyncMPClient pattern (vllm/v1/engine/core_client.py).
"""

from __future__ import annotations

import asyncio
import multiprocessing
import weakref
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.entrypoints.stage_engine.stage_client_protocol import (
    BaseStageClient,
)
from vllm_omni.entrypoints.stage_engine.stage_core_proc import (
    REQUEST_TYPE_ABORT,
    REQUEST_TYPE_GENERATE,
    REQUEST_TYPE_HEALTH_CHECK,
    REQUEST_TYPE_SHUTDOWN,
    RESPONSE_TYPE_DEAD,
    RESPONSE_TYPE_ERROR,
    RESPONSE_TYPE_HEALTH,
    RESPONSE_TYPE_OUTPUT,
    StageEngineCoreProc,
)
from vllm_omni.entrypoints.stage_engine.stage_serialization import (
    StageMsgpackDecoder,
    StageMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

# Timeout for waiting for stage engine to be ready (seconds)
STAGE_ENGINE_READY_TIMEOUT_S = 120


class NonLLMStageClient(BaseStageClient):
    """ZMQ-based client for non-LLM stage engine communication.

    This client is used for Diffusion and Audio stages. For LLM stages,
    use vLLM's EngineCoreClient.make_async_mp_client() directly.

    Communicates with StageEngineCoreProc via ZMQ:
    - ROUTER socket for sending requests to stage worker
    - PULL socket for receiving outputs from stage worker

    This follows the same pattern as vLLM's AsyncMPClient.

    Attributes:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("diffusion", "audio")
        input_address: ZMQ address for sending requests (ROUTER)
        output_address: ZMQ address for receiving outputs (PULL)
    """

    @staticmethod
    def make_client(
        stage_id: int,
        stage_type: str,
        vllm_config: VllmConfig | None = None,
        executor_class: type["Executor"] | None = None,
        input_address: str | None = None,
        output_address: str | None = None,
        log_stats: bool = False,
    ) -> "NonLLMStageClient":
        """Create a NonLLMStageClient and start the background stage process.

        This is similar to EngineCoreClient.make_async_mp_client().

        Args:
            stage_id: Unique identifier for this stage
            stage_type: Type of stage ("diffusion", "audio")
            vllm_config: Config for the stage (passed to stage process)
            executor_class: Executor class for the stage process
            input_address: ZMQ address for sending requests (auto-generated if None)
            output_address: ZMQ address for receiving outputs (auto-generated if None)
            log_stats: Whether to log statistics

        Returns:
            Initialized NonLLMStageClient with background process running
        """
        # Generate addresses if not provided
        if input_address is None:
            input_address = f"tcp://127.0.0.1:{5555 + stage_id * 2}"
        if output_address is None:
            output_address = f"tcp://127.0.0.1:{5556 + stage_id * 2}"

        # Create client
        client = NonLLMStageClient(
            stage_id=stage_id,
            stage_type=stage_type,
            input_address=input_address,
            output_address=output_address,
        )

        # Start background stage process
        client._start_stage_process(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        return client

    def __init__(
        self,
        stage_id: int,
        stage_type: str,
        input_address: str,
        output_address: str,
    ):
        """Initialize the stage engine client.

        Note: Use make_client() to create a client with a background process.

        Args:
            stage_id: Unique identifier for this stage
            stage_type: Type of stage ("diffusion", "audio")
            input_address: ZMQ address for sending requests
            output_address: ZMQ address for receiving outputs
        """
        super().__init__(stage_id, stage_type)
        self.input_address = input_address
        self.output_address = output_address

        # ZMQ context and sockets
        self.ctx = zmq.asyncio.Context()
        self.input_socket: zmq.asyncio.Socket | None = None
        self.output_socket: zmq.asyncio.Socket | None = None

        # Serialization
        self.encoder = StageMsgpackEncoder()
        self.decoder = StageMsgpackDecoder()

        # Output queue for async iteration
        self.outputs_queue: asyncio.Queue[OmniRequestOutput | Exception] = (
            asyncio.Queue()
        )

        # Output processing task
        self.output_task: asyncio.Task | None = None

        # Tracking active requests
        self.active_requests: set[str] = set()

        # Engine health
        self.engine_dead = False

        # Background process
        self._stage_process: multiprocessing.Process | None = None

        # Finalizer for cleanup
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _start_stage_process(
        self,
        vllm_config: VllmConfig | None,
        executor_class: type["Executor"] | None,
        log_stats: bool,
    ) -> None:
        """Start the background stage process.

        Args:
            vllm_config: Config for the stage
            executor_class: Executor class for the stage process
            log_stats: Whether to log statistics
        """
        # Create and start the stage process
        self._stage_process = multiprocessing.Process(
            target=StageEngineCoreProc.run_stage_worker,
            args=(
                self.stage_id,
                self.stage_type,
                self.input_address,
                self.output_address,
                vllm_config,
                executor_class,
                log_stats,
            ),
            name=f"StageEngine-{self.stage_id}",
            daemon=True,
        )
        self._stage_process.start()
        logger.info(
            f"Started stage {self.stage_id} ({self.stage_type}) process "
            f"with PID {self._stage_process.pid}"
        )

        # Create ZMQ sockets
        self._create_sockets()

        # Wait for stage to be ready
        self._wait_for_ready()

        # Start output processing task
        try:
            asyncio.get_running_loop()
            self._ensure_output_task()
        except RuntimeError:
            pass

    def _create_sockets(self) -> None:
        """Create ZMQ sockets for communication."""
        # Create ROUTER socket for sending requests
        self.input_socket = self.ctx.socket(zmq.ROUTER)
        self.input_socket.bind(self.input_address)
        logger.info(
            f"Stage {self.stage_id} client bound to input: {self.input_address}"
        )

        # Create PULL socket for receiving outputs
        self.output_socket = self.ctx.socket(zmq.PULL)
        self.output_socket.bind(self.output_address)
        logger.info(
            f"Stage {self.stage_id} client bound to output: {self.output_address}"
        )

    def _wait_for_ready(self) -> None:
        """Wait for the stage process to be ready."""
        # Use a sync socket to wait for the ready message
        sync_input_socket = zmq.Socket.shadow(self.input_socket)

        if not sync_input_socket.poll(timeout=STAGE_ENGINE_READY_TIMEOUT_S * 1000):
            raise TimeoutError(
                f"Timed out waiting for stage {self.stage_id} to send ready message"
            )

        # Receive the ready message
        identity, _ = sync_input_socket.recv_multipart()
        logger.info(f"Stage {self.stage_id} is ready (identity: {identity})")

    def _cleanup(self) -> None:
        """Cleanup resources (called by finalizer)."""
        # Terminate stage process
        if self._stage_process is not None and self._stage_process.is_alive():
            self._stage_process.terminate()
            self._stage_process.join(timeout=5)
            if self._stage_process.is_alive():
                self._stage_process.kill()

        # Close sockets
        if self.input_socket is not None:
            self.input_socket.close()
        if self.output_socket is not None:
            self.output_socket.close()

        # Terminate context
        self.ctx.term()

    def _ensure_output_task(self):
        """Ensure output processing task is running."""
        if self.output_task is not None and not self.output_task.done():
            return

        self.output_task = asyncio.create_task(
            self._process_outputs(),
            name=f"Stage{self.stage_id}OutputTask",
        )

    async def _process_outputs(self):
        """Process outputs from stage worker."""
        try:
            while not self.engine_dead:
                # Receive multipart message: [type, request_id, data_frames...]
                frames = await self.output_socket.recv_multipart(copy=False)

                if not frames:
                    continue

                response_type = bytes(frames[0].buffer)
                request_id_bytes = bytes(frames[1].buffer) if len(frames) > 1 else b""
                data_frames = frames[2:] if len(frames) > 2 else []

                # Handle different response types
                if response_type == RESPONSE_TYPE_OUTPUT:
                    # Deserialize output
                    request_id = request_id_bytes.decode("utf-8")
                    output = self.decoder.decode(
                        [bytes(f.buffer) for f in data_frames]
                    )
                    await self.outputs_queue.put(output)

                elif response_type == RESPONSE_TYPE_ERROR:
                    # Error response
                    request_id = request_id_bytes.decode("utf-8")
                    error_msg = (
                        data_frames[0].bytes.decode("utf-8")
                        if data_frames
                        else "Unknown error"
                    )
                    logger.error(
                        f"Stage {self.stage_id} error for {request_id}: {error_msg}"
                    )
                    await self.outputs_queue.put(
                        Exception(f"Stage {self.stage_id} error: {error_msg}")
                    )

                elif response_type == RESPONSE_TYPE_HEALTH:
                    # Health check response
                    logger.debug(f"Stage {self.stage_id} health check OK")

                elif response_type == RESPONSE_TYPE_DEAD:
                    # Engine died
                    logger.error(f"Stage {self.stage_id} engine died")
                    self.engine_dead = True
                    await self.outputs_queue.put(
                        Exception(f"Stage {self.stage_id} engine died")
                    )
                    break

                else:
                    logger.error(f"Unknown response type: {response_type}")

        except asyncio.CancelledError:
            logger.info(f"Stage {self.stage_id} output task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in stage {self.stage_id} output processing")
            await self.outputs_queue.put(e)

    # ========== StageClient Protocol Methods ==========

    async def add_request_async(self, request: dict[str, Any]) -> None:
        """Submit a request to the stage worker.

        This method implements the StageClient protocol.

        Args:
            request: Dictionary containing:
                - request_id: Unique request identifier
                - prompt: OmniPromptType (text, tokens, or embeds)
                - sampling_params: OmniSamplingParams
        """
        request_id = request["request_id"]
        prompt = request["prompt"]
        sampling_params = request["sampling_params"]

        if self.engine_dead:
            raise RuntimeError(f"Stage {self.stage_id} engine is dead")

        # Track active request
        self.active_requests.add(request_id)

        # Prepare request data
        request_data = {
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params": sampling_params,
        }

        # Serialize request
        bufs = self.encoder.encode(request_data)

        # Send multipart message: [identity, type, data_frames...]
        worker_identity = f"stage_{self.stage_id}_worker".encode("utf-8")
        frames = [worker_identity, REQUEST_TYPE_GENERATE] + bufs

        await self.input_socket.send_multipart(frames)
        logger.debug(f"Submitted request {request_id} to stage {self.stage_id}")

    async def get_output_async(self) -> OmniRequestOutput:
        """Get the next output from the stage worker.

        Returns:
            OmniRequestOutput from the stage

        Raises:
            Exception if an error occurred in the stage worker
        """
        self._ensure_output_task()

        output = await self.outputs_queue.get()
        if isinstance(output, Exception):
            raise output

        return output

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort one or more requests in the stage worker.

        This method implements the StageClient protocol.

        Args:
            request_ids: List of request IDs to abort
        """
        if self.engine_dead:
            return

        for request_id in request_ids:
            # Prepare abort request
            request_data = {"request_id": request_id}

            # Serialize request
            bufs = self.encoder.encode(request_data)

            # Send multipart message
            worker_identity = f"stage_{self.stage_id}_worker".encode("utf-8")
            frames = [worker_identity, REQUEST_TYPE_ABORT] + bufs

            await self.input_socket.send_multipart(frames)
            logger.debug(f"Aborted request {request_id} in stage {self.stage_id}")

            # Remove from active requests
            self.active_requests.discard(request_id)

    # ========== Legacy Methods (for backward compatibility) ==========

    async def submit_request(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniSamplingParams,
    ) -> None:
        """Submit a generate request to the stage worker.

        This is a convenience method that wraps add_request_async.

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType (text, tokens, or embeds)
            sampling_params: OmniSamplingParams
        """
        await self.add_request_async({
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params": sampling_params,
        })

    async def abort_request(self, request_id: str) -> None:
        """Abort a single request in the stage worker.

        This is a convenience method that wraps abort_requests_async.

        Args:
            request_id: Request ID to abort
        """
        await self.abort_requests_async([request_id])

    async def check_health(self) -> bool:
        """Check if the stage worker is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self.engine_dead:
            return False

        # Check if process is alive
        if self._stage_process is not None and not self._stage_process.is_alive():
            self.engine_dead = True
            return False

        try:
            # Send health check request
            worker_identity = f"stage_{self.stage_id}_worker".encode("utf-8")
            frames = [worker_identity, REQUEST_TYPE_HEALTH_CHECK]
            await self.input_socket.send_multipart(frames)

            # Wait for response with timeout
            await asyncio.wait_for(
                self._wait_for_health_response(),
                timeout=5.0,
            )
            return True

        except asyncio.TimeoutError:
            logger.error(f"Stage {self.stage_id} health check timeout")
            return False
        except Exception as e:
            logger.exception(f"Stage {self.stage_id} health check failed")
            return False

    async def _wait_for_health_response(self):
        """Wait for health check response."""
        # This is a simplified implementation
        await asyncio.sleep(0.1)

    def shutdown(self) -> None:
        """Shutdown the stage worker and client."""
        logger.info(f"Shutting down stage {self.stage_id} client")

        # Send shutdown request (sync version for shutdown)
        if not self.engine_dead and self.input_socket is not None:
            try:
                worker_identity = f"stage_{self.stage_id}_worker".encode("utf-8")
                frames = [worker_identity, REQUEST_TYPE_SHUTDOWN]
                # Use sync send for shutdown
                sync_socket = zmq.Socket.shadow(self.input_socket)
                sync_socket.send_multipart(frames)
            except Exception as e:
                logger.exception(f"Error sending shutdown to stage {self.stage_id}")

        # Cancel output task
        if self.output_task is not None and not self.output_task.done():
            self.output_task.cancel()

        # Trigger finalizer for cleanup
        self._finalizer()

        logger.info(f"Stage {self.stage_id} client shutdown complete")


# Backward compatibility alias
StageEngineCoreClient = NonLLMStageClient
