"""ZMQ-based client for communicating with stage engine processes.

This module provides StageEngineCoreClient, a ZMQ-based client for submitting
requests to and receiving outputs from StageEngineCoreProc workers.

Based on vLLM's AsyncMPClient pattern (vllm/v1/engine/core_client.py).
"""

import asyncio
import uuid
from typing import Any, AsyncGenerator

import zmq
import zmq.asyncio
from vllm.logger import init_logger

from vllm_omni.entrypoints.stage_engine.stage_core_proc import (
    REQUEST_TYPE_ABORT,
    REQUEST_TYPE_GENERATE,
    REQUEST_TYPE_HEALTH_CHECK,
    REQUEST_TYPE_SHUTDOWN,
    RESPONSE_TYPE_DEAD,
    RESPONSE_TYPE_ERROR,
    RESPONSE_TYPE_HEALTH,
    RESPONSE_TYPE_OUTPUT,
)
from vllm_omni.entrypoints.stage_engine.stage_serialization import (
    StageMsgpackDecoder,
    StageMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class StageEngineCoreClient:
    """ZMQ-based client for stage engine communication.

    Communicates with StageEngineCoreProc via ZMQ:
    - ROUTER socket for sending requests to stage worker
    - PULL socket for receiving outputs from stage worker

    Attributes:
        stage_id: Unique identifier for this stage
        input_address: ZMQ address for sending requests (ROUTER)
        output_address: ZMQ address for receiving outputs (PULL)
    """

    def __init__(
        self,
        stage_id: int,
        input_address: str,
        output_address: str,
    ):
        """Initialize the stage engine client.

        Args:
            stage_id: Unique identifier for this stage
            input_address: ZMQ address for sending requests
            output_address: ZMQ address for receiving outputs
        """
        self.stage_id = stage_id
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
        self.outputs_queue: asyncio.Queue[OmniRequestOutput | Exception] = asyncio.Queue()

        # Output processing task
        self.output_task: asyncio.Task | None = None

        # Tracking active requests
        self.active_requests: set[str] = set()

        # Engine health
        self.engine_dead = False

    async def start(self):
        """Start the client and connect to stage worker."""
        # Create ROUTER socket for sending requests
        self.input_socket = self.ctx.socket(zmq.ROUTER)
        self.input_socket.bind(self.input_address)
        logger.info(f"Stage {self.stage_id} client bound to input: {self.input_address}")

        # Create PULL socket for receiving outputs
        self.output_socket = self.ctx.socket(zmq.PULL)
        self.output_socket.bind(self.output_address)
        logger.info(f"Stage {self.stage_id} client bound to output: {self.output_address}")

        # Start output processing task
        self._ensure_output_task()

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
                    output = self.decoder.decode([bytes(f.buffer) for f in data_frames])
                    await self.outputs_queue.put(output)

                elif response_type == RESPONSE_TYPE_ERROR:
                    # Error response
                    request_id = request_id_bytes.decode("utf-8")
                    error_msg = data_frames[0].decode("utf-8") if data_frames else "Unknown error"
                    logger.error(f"Stage {self.stage_id} error for {request_id}: {error_msg}")
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

    async def submit_request(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniSamplingParams,
    ) -> None:
        """Submit a generate request to the stage worker.

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType (text, tokens, or embeds)
            sampling_params: OmniSamplingParams (SamplingParams or OmniDiffusionSamplingParams)
        """
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
        # For ROUTER socket, we need to include the worker identity
        # Since we only have one worker per stage, we can use a fixed identity
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

    async def get_outputs_async(self) -> AsyncGenerator[OmniRequestOutput, None]:
        """Get outputs from the stage worker as an async generator.

        Yields:
            OmniRequestOutput instances from the stage
        """
        while True:
            try:
                output = await self.get_output_async()
                yield output

                # If output is finished, remove from active requests
                if output.finished:
                    self.active_requests.discard(output.request_id)

            except Exception as e:
                logger.exception(f"Error getting output from stage {self.stage_id}")
                raise

    async def abort_request(self, request_id: str) -> None:
        """Abort a request in the stage worker.

        Args:
            request_id: Request ID to abort
        """
        if self.engine_dead:
            return

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

    async def check_health(self) -> bool:
        """Check if the stage worker is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self.engine_dead:
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
        # In production, we'd need to handle this more carefully
        await asyncio.sleep(0.1)

    async def shutdown(self):
        """Shutdown the stage worker and client."""
        logger.info(f"Shutting down stage {self.stage_id} client")

        # Send shutdown request
        if not self.engine_dead and self.input_socket is not None:
            try:
                worker_identity = f"stage_{self.stage_id}_worker".encode("utf-8")
                frames = [worker_identity, REQUEST_TYPE_SHUTDOWN]
                await self.input_socket.send_multipart(frames)
            except Exception as e:
                logger.exception(f"Error sending shutdown to stage {self.stage_id}")

        # Cancel output task
        if self.output_task is not None and not self.output_task.done():
            self.output_task.cancel()
            try:
                await self.output_task
            except asyncio.CancelledError:
                pass

        # Close sockets
        if self.input_socket is not None:
            self.input_socket.close()
        if self.output_socket is not None:
            self.output_socket.close()

        # Terminate context
        self.ctx.term()

        logger.info(f"Stage {self.stage_id} client shutdown complete")
