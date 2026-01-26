# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
StageCoreClient: Client for communicating with StageCore processes.

This module extends vLLM's EngineCoreClient/AsyncMPClient to support
multi-stage pipelines, reusing as much of vLLM's infrastructure as possible.
"""

import asyncio
import os
import weakref
from typing import Any

import zmq
import zmq.asyncio
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from vllm.v1.engine.core_client import (
    AsyncMPClient,
    BackgroundResources,
)
from vllm.v1.executor import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

logger = init_logger(__name__)


class StageAsyncMPClient(AsyncMPClient):
    """
    Async client for a single Stage, extending vLLM's AsyncMPClient.

    This class reuses vLLM's ZMQ infrastructure while adding:
    - Stage-specific identification (stage_id)
    - Integration with multi-stage pipeline
    - Custom output handling for stage outputs

    Key reused components from vLLM:
    - BackgroundResources for resource management
    - ZMQ socket setup (ROUTER/PULL)
    - MsgpackEncoder/Decoder for serialization
    - Output queue processing task
    """

    def __init__(
        self,
        stage_id: int,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        input_address: str,
        output_address: str,
    ):
        """
        Initialize StageAsyncMPClient.

        Unlike the parent class which launches engines, this client connects
        to an already-running StageCore process at the given addresses.

        Args:
            stage_id: Unique identifier for this stage
            vllm_config: vLLM configuration
            executor_class: Executor class (may not be used for stages)
            log_stats: Whether to log statistics
            input_address: ZMQ address for sending requests
            output_address: ZMQ address for receiving outputs
        """
        self.stage_id = stage_id
        self.vllm_config = vllm_config

        # Serialization setup (reused from vLLM)
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup (reused from vLLM)
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx)

        # Resource management (reused from vLLM)
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)

        success = False
        try:
            self.engines_running = False

            # Create input and output sockets (similar to MPClient)
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True
            )
            self.resources.output_socket = make_zmq_socket(self.ctx, output_address, zmq.PULL)

            # Stage identity
            self.core_engine = stage_id.to_bytes(2, "little")
            self.core_engines = [self.core_engine]
            self.engine_ranks_managed = [stage_id]

            # Wait for ready message from stage
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            timeout_ms = 300 * 1000  # 5 minutes
            if not sync_input_socket.poll(timeout=timeout_ms):
                raise TimeoutError(f"Timed out waiting for Stage-{stage_id} to send ready message")
            identity, _ = sync_input_socket.recv_multipart()
            logger.info(f"[StageClient-{stage_id}] Connected to stage")

            # Utility results tracking (reused from vLLM)
            self.utility_results: dict[int, asyncio.Future[Any]] = {}

            # Pending messages for tensor references (reused from vLLM)
            from collections import deque

            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # Output queue (reused from vLLM)
            self.client_count = 1
            self.client_index = 0
            self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

            try:
                asyncio.get_running_loop()
                self._ensure_output_queue_task()
            except RuntimeError:
                pass

            success = True
        finally:
            if not success:
                self._finalizer()

    # Override to customize output handling for stages if needed
    async def get_stage_output_async(self) -> EngineCoreOutputs:
        """Get output from this stage (alias for get_output_async)."""
        return await self.get_output_async()

    # Stage-specific methods

    async def add_stage_request_async(
        self,
        request: EngineCoreRequest,
        from_stage: int | None = None,
        connector_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a request to this stage.

        Extends the base add_request_async with stage-specific metadata.
        """
        # Store stage metadata in request if supported
        if hasattr(request, "from_stage"):
            request.from_stage = from_stage
        if hasattr(request, "connector_metadata"):
            request.connector_metadata = connector_metadata

        await self.add_request_async(request)


class MultiStageCoreClient:
    """
    Client for managing multiple Stage processes.

    This class orchestrates multiple StageAsyncMPClient instances,
    providing a unified interface for multi-stage pipelines.
    """

    def __init__(self, base_dir: str = "/tmp/vllm_omni"):
        self.base_dir = base_dir
        self.stage_clients: dict[int, StageAsyncMPClient] = {}

        # Shared ZMQ context
        self._sync_ctx = zmq.Context(io_threads=2)
        self._ctx = zmq.asyncio.Context(self._sync_ctx)

        # Track if any stage is dead
        self._any_stage_dead = False

    def add_stage_client(
        self,
        stage_id: int,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        input_address: str | None = None,
        output_address: str | None = None,
    ) -> StageAsyncMPClient:
        """
        Add a client for a stage.

        Args:
            stage_id: Unique stage identifier
            vllm_config: vLLM configuration
            executor_class: Executor class
            log_stats: Whether to log statistics
            input_address: Optional custom input address
            output_address: Optional custom output address

        Returns:
            StageAsyncMPClient for this stage
        """
        os.makedirs(self.base_dir, exist_ok=True)

        # Default addresses if not provided
        if input_address is None:
            input_address = f"ipc://{self.base_dir}/stage_{stage_id}_input.ipc"
        if output_address is None:
            output_address = f"ipc://{self.base_dir}/stage_{stage_id}_output.ipc"

        client = StageAsyncMPClient(
            stage_id=stage_id,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            input_address=input_address,
            output_address=output_address,
        )
        self.stage_clients[stage_id] = client
        return client

    def get_stage_client(self, stage_id: int) -> StageAsyncMPClient | None:
        """Get a stage client by ID."""
        return self.stage_clients.get(stage_id)

    async def add_request_to_stage(
        self,
        stage_id: int,
        request: EngineCoreRequest,
        from_stage: int | None = None,
        connector_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a request to a specific stage."""
        client = self.stage_clients.get(stage_id)
        if client is None:
            raise ValueError(f"Stage {stage_id} not found")
        await client.add_stage_request_async(request, from_stage, connector_metadata)

    async def get_output_from_stage(self, stage_id: int) -> EngineCoreOutputs:
        """Get output from a specific stage."""
        client = self.stage_clients.get(stage_id)
        if client is None:
            raise ValueError(f"Stage {stage_id} not found")
        return await client.get_stage_output_async()

    async def abort_requests_on_stage(self, stage_id: int, request_ids: list[str]) -> None:
        """Abort requests on a specific stage."""
        client = self.stage_clients.get(stage_id)
        if client is not None:
            await client.abort_requests_async(request_ids)

    async def abort_requests_all_stages(self, request_ids: list[str]) -> None:
        """Abort requests across all stages."""
        for client in self.stage_clients.values():
            await client.abort_requests_async(request_ids)

    def shutdown_stage(self, stage_id: int) -> None:
        """Shutdown a specific stage client."""
        client = self.stage_clients.pop(stage_id, None)
        if client is not None:
            client.shutdown()

    def shutdown_all(self) -> None:
        """Shutdown all stage clients."""
        for client in self.stage_clients.values():
            client.shutdown()
        self.stage_clients.clear()

    @property
    def is_running(self) -> bool:
        """Check if all stages are running."""
        return all(not c.resources.engine_dead for c in self.stage_clients.values())

    @property
    def errored(self) -> bool:
        """Check if any stage has errored."""
        return any(c.resources.engine_dead for c in self.stage_clients.values())
