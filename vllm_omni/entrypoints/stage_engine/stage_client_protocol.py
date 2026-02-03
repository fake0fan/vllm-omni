# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage client protocol for multi-stage pipelines.

This module defines the StageClient protocol that unifies:
- vLLM's EngineCoreClient (for LLM stages)
- Custom stage clients (for Diffusion/Audio stages)

This allows AsyncOmni to work with different stage types through a common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest

    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
    from vllm_omni.outputs import OmniRequestOutput


@runtime_checkable
class StageClient(Protocol):
    """Protocol for stage clients in multi-stage pipelines.

    This protocol defines the common interface that both vLLM's EngineCoreClient
    and custom stage clients (Diffusion, Audio) must implement.

    For LLM stages, this is satisfied by vLLM's AsyncMPClient.
    For non-LLM stages, this is satisfied by custom implementations.
    """

    @property
    def stage_id(self) -> int:
        """Unique identifier for this stage."""
        ...

    @property
    def stage_type(self) -> str:
        """Type of stage ('llm', 'diffusion', 'audio')."""
        ...

    async def add_request_async(self, request: Any) -> None:
        """Submit a request to the stage.

        For LLM stages: request is EngineCoreRequest
        For non-LLM stages: request is a dict with prompt and sampling_params
        """
        ...

    async def get_output_async(self) -> Any:
        """Get the next output from the stage.

        For LLM stages: returns EngineCoreOutputs
        For non-LLM stages: returns OmniRequestOutput
        """
        ...

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort one or more requests."""
        ...

    def shutdown(self) -> None:
        """Shutdown the stage client and cleanup resources."""
        ...


class BaseStageClient(ABC):
    """Abstract base class for non-LLM stage clients.

    This provides a common base for Diffusion and Audio stage clients,
    implementing the StageClient protocol.
    """

    def __init__(self, stage_id: int, stage_type: str):
        self._stage_id = stage_id
        self._stage_type = stage_type

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def stage_type(self) -> str:
        return self._stage_type

    @abstractmethod
    async def add_request_async(self, request: Any) -> None:
        """Submit a request to the stage."""
        ...

    @abstractmethod
    async def get_output_async(self) -> Any:
        """Get the next output from the stage."""
        ...

    @abstractmethod
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort one or more requests."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the stage client."""
        ...


class LLMStageClientWrapper:
    """Wrapper around vLLM's EngineCoreClient to add stage metadata.

    This wraps vLLM's AsyncMPClient to add stage_id and stage_type properties,
    making it compatible with the StageClient protocol.
    """

    def __init__(
        self,
        engine_core_client: Any,  # vLLM's EngineCoreClient
        stage_id: int,
    ):
        """Initialize the wrapper.

        Args:
            engine_core_client: vLLM's EngineCoreClient (AsyncMPClient)
            stage_id: Unique identifier for this stage
        """
        self._client = engine_core_client
        self._stage_id = stage_id
        self._stage_type = "llm"

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def stage_type(self) -> str:
        return self._stage_type

    @property
    def inner_client(self) -> Any:
        """Access the underlying vLLM EngineCoreClient."""
        return self._client

    # Delegate all methods to the inner client
    async def add_request_async(self, request: "EngineCoreRequest") -> None:
        """Submit a request to the LLM stage."""
        await self._client.add_request_async(request)

    async def get_output_async(self) -> "EngineCoreOutputs":
        """Get the next output from the LLM stage."""
        return await self._client.get_output_async()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort one or more requests."""
        await self._client.abort_requests_async(request_ids)

    def shutdown(self) -> None:
        """Shutdown the stage client."""
        self._client.shutdown()

    # Additional methods delegated from EngineCoreClient
    async def reset_mm_cache_async(self) -> None:
        """Reset multimodal cache."""
        await self._client.reset_mm_cache_async()

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset prefix cache."""
        return await self._client.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def profile_async(self, is_start: bool = True) -> None:
        """Start or stop profiling."""
        await self._client.profile_async(is_start)

    async def sleep_async(self, level: int = 1) -> None:
        """Put engine to sleep."""
        await self._client.sleep_async(level)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        """Wake up engine."""
        await self._client.wake_up_async(tags)

    async def is_sleeping_async(self) -> bool:
        """Check if engine is sleeping."""
        return await self._client.is_sleeping_async()

    async def add_lora_async(self, lora_request: Any) -> bool:
        """Add LoRA adapter."""
        return await self._client.add_lora_async(lora_request)

    async def get_supported_tasks_async(self) -> tuple:
        """Get supported tasks."""
        return await self._client.get_supported_tasks_async()
