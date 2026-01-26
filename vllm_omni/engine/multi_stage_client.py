# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MultiStageEngineClient: Base class for multi-stage pipeline engine clients.

This module provides a base implementation of the EngineClient protocol
for multi-stage pipelines, similar to how vLLM's AsyncLLM works with
a single EngineCore but extended to support multiple stages.

The architecture follows vLLM's design patterns:
- StageCore (similar to EngineCore): Handles engine execution in subprocess
- StageCoreClient (similar to EngineCoreClient): Manages ZMQ communication
- MultiStageEngineClient: Orchestrates multiple stages

Communication modes:
- ZMQ mode (recommended): High-performance ZMQ-based communication
- Queue mode (legacy): mp.Queue/Ray Queue for backward compatibility
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import RendererLike
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.exceptions import EngineDeadError

if TYPE_CHECKING:
    from vllm_omni.engine.stage_core_client import MultiStageCoreClient
    from vllm_omni.entrypoints.omni_stage import OmniStage
    from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class CommunicationMode(str, Enum):
    """Communication mode for stage-to-stage communication."""

    QUEUE = "queue"  # Legacy mp.Queue/Ray Queue mode
    ZMQ = "zmq"  # ZMQ-based communication (recommended)


class MultiStageEngineClient(EngineClient, ABC):
    """
    Base class for multi-stage pipeline engine clients.

    This class provides a common implementation of the EngineClient protocol
    for orchestrating multiple stages (e.g., LLM + Audio + Diffusion).

    Subclasses:
        - AsyncOmni: Asynchronous multi-stage client

    Key Design Principles (following vLLM's architecture):
        1. Each stage has its own StageCore process (similar to vLLM's EngineCore)
        2. Communication with stages uses ZMQ for high-performance IPC:
           - ZMQ DEALER/ROUTER for request routing
           - ZMQ PUSH/PULL for output streaming
           - Background IO threads for non-blocking communication
        3. Stage-to-stage data transfer is handled via connectors:
           - ZMQConnector: High-performance direct transfer
           - SharedMemoryConnector: Zero-copy for large payloads
        4. Unified message format (StageCoreRequest/StageCoreOutput)

    Attributes:
        stage_list: List of OmniStage objects representing each stage
        stage_configs: Raw stage configurations from YAML
        output_modalities: List of output modalities for each final_output stage
        default_sampling_params_list: Default parameters for each stage
        connectors: Stage-to-stage connectors for data transfer
        communication_mode: Mode for stage communication (zmq/queue)
        stage_core_client: ZMQ-based client for StageCore communication
    """

    # Stage management (set by subclass or _initialize_stages)
    stage_list: list["OmniStage"]
    stage_configs: list[dict[str, Any]]
    output_modalities: list[str]
    default_sampling_params_list: list[Any]
    connectors: dict[tuple[str, str], Any]

    # Communication mode and ZMQ client
    communication_mode: CommunicationMode = CommunicationMode.QUEUE
    stage_core_client: "MultiStageCoreClient | None" = None

    # Configuration from primary LLM stage (may be None for pure diffusion)
    _vllm_config: VllmConfig | None = None
    _model_config: ModelConfig | None = None
    _tokenizer: TokenizerLike | None = None
    _input_processor: Any | None = None
    _io_processor: IOProcessor | None = None

    # ========== EngineClient Protocol: Properties ==========

    @property
    def vllm_config(self) -> VllmConfig | None:
        """Return vllm_config from the primary LLM stage."""
        return self._vllm_config

    @property
    def model_config(self) -> ModelConfig | None:
        """Return model_config from the primary LLM stage."""
        return self._model_config

    @property
    def input_processor(self) -> Any | None:
        """Return input_processor from the primary LLM stage."""
        return self._input_processor

    @property
    def io_processor(self) -> IOProcessor | None:
        """Return io_processor from the primary LLM stage."""
        return self._io_processor

    @property
    def renderer(self) -> RendererLike | None:
        """Return renderer from input_processor if available."""
        if self._input_processor is not None:
            return getattr(self._input_processor, "renderer", None)
        return None

    @property
    def is_running(self) -> bool:
        """Check if any stage is running."""
        return len(getattr(self, "stage_list", [])) > 0

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self.errored

    @property
    def errored(self) -> bool:
        """Check if any stage has errored."""
        return not self.is_running

    @property
    def dead_error(self) -> BaseException:
        """Return the error to raise when the engine is dead."""
        return EngineDeadError()

    # ========== Abstract Methods (must be implemented by subclass) ==========

    @abstractmethod
    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        # Multi-stage specific parameters
        sampling_params_list: list[Any] | None = None,
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator["OmniRequestOutput", None]:
        """
        Generate outputs for a request through the multi-stage pipeline.

        This method orchestrates the execution across multiple stages.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters (used for first LLM stage)
            request_id: Unique request identifier
            sampling_params_list: Optional list of parameters for each stage
            output_modalities: Optional list of desired output modalities

        Yields:
            OmniRequestOutput objects from each final_output stage
        """
        ...

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request across all stages."""
        ...

    # ========== Default Implementations ==========

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Pooling/encoding is not supported for multi-stage pipelines."""
        raise NotImplementedError(
            "encode() is not implemented for MultiStageEngineClient. "
            "Multi-stage pipelines typically use generate() for end-to-end processing."
        )

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled on any LLM stage."""
        for stage in getattr(self, "stage_list", []):
            if getattr(stage, "is_tracing_enabled", False):
                return True
        return False

    async def do_log_stats(self) -> None:
        """Log statistics (delegate to individual stages if needed)."""
        pass

    async def check_health(self) -> None:
        """Check health of all stages."""
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        """Start profiling on all stages."""
        for stage in getattr(self, "stage_list", []):
            if hasattr(stage, "start_profile"):
                stage.start_profile()

    async def stop_profile(self) -> None:
        """Stop profiling on all stages."""
        for stage in getattr(self, "stage_list", []):
            if hasattr(stage, "stop_profile"):
                stage.stop_profile()

    async def reset_mm_cache(self) -> None:
        """Reset multi-modal cache on LLM stages."""
        # Delegate to stages that support it
        pass

    async def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        """Reset prefix cache on LLM stages."""
        # Delegate to stages that support it
        return True

    async def sleep(self, level: int = 1) -> None:
        """Put the engine to sleep (not typically used in multi-stage)."""
        pass

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine."""
        pass

    async def is_sleeping(self) -> bool:
        """Check if the engine is sleeping."""
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add LoRA adapter to LLM stages."""
        # Delegate to LLM stages that support LoRA
        return False

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation (implemented by subclass)."""
        pass

    async def resume_generation(self) -> None:
        """Resume generation (implemented by subclass)."""
        pass

    async def is_paused(self) -> bool:
        """Check if generation is paused (implemented by subclass)."""
        return False

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get supported tasks based on stage types.

        Returns:
            Tuple of supported tasks based on configured stages
        """
        tasks: set[str] = set()
        for stage in getattr(self, "stage_list", []):
            stage_type = getattr(stage, "stage_type", "llm")
            if stage_type == "llm":
                tasks.add("generate")
            elif stage_type == "diffusion":
                tasks.add("generate")  # Image generation via chat
        return tuple(tasks)  # type: ignore

    # ========== Helper Methods ==========

    def get_stage_by_type(self, stage_type: str) -> "OmniStage | None":
        """Get the first stage of a specific type."""
        for stage in getattr(self, "stage_list", []):
            if getattr(stage, "stage_type", None) == stage_type:
                return stage
        return None

    def get_llm_stage(self) -> "OmniStage | None":
        """Get the primary LLM stage (usually the first one with is_comprehension=True)."""
        for stage in getattr(self, "stage_list", []):
            if getattr(stage, "is_comprehension", False):
                return stage
        # Fallback: find first LLM stage
        return self.get_stage_by_type("llm")

    async def get_vllm_config(self) -> VllmConfig | None:
        """Get vllm_config from the primary LLM stage."""
        llm_stage = self.get_llm_stage()
        if llm_stage is not None:
            return getattr(llm_stage, "vllm_config", None)
        return None

    async def get_model_config(self) -> ModelConfig | None:
        """Get model_config from the primary LLM stage."""
        vllm_config = await self.get_vllm_config()
        if vllm_config is not None:
            return vllm_config.model_config
        return None

    async def get_tokenizer(self) -> TokenizerLike | None:
        """Get tokenizer from the primary LLM stage."""
        llm_stage = self.get_llm_stage()
        if llm_stage is not None:
            return getattr(llm_stage, "tokenizer", None)
        return None

    def _init_from_llm_stage(self) -> None:
        """
        Initialize client attributes from the primary LLM stage.

        This should be called after stages are ready to populate
        vllm_config, model_config, tokenizer, etc.
        """
        llm_stage = self.get_llm_stage()
        if llm_stage is None:
            logger.warning(
                "No LLM stage found, some EngineClient attributes will be None. "
                "This may cause issues with OpenAI-compatible serving."
            )
            return

        # Get vllm_config
        vllm_config = getattr(llm_stage, "vllm_config", None)
        if vllm_config is not None:
            self._vllm_config = vllm_config
            self._model_config = vllm_config.model_config

        # Get tokenizer
        self._tokenizer = getattr(llm_stage, "tokenizer", None)

        logger.debug(
            "Initialized MultiStageEngineClient from LLM stage: "
            f"vllm_config={self._vllm_config is not None}, "
            f"tokenizer={self._tokenizer is not None}"
        )

    # ========== ZMQ Communication Mode Methods ==========

    def _init_zmq_mode(self, base_dir: str = "/tmp/vllm_omni") -> None:
        """
        Initialize ZMQ communication mode.

        This method sets up the MultiStageCoreClient for ZMQ-based
        communication with StageCore processes. Call this instead of
        the traditional queue-based initialization for better performance.

        Args:
            base_dir: Base directory for IPC socket files
        """
        from vllm_omni.engine.stage_core_client import MultiStageCoreClient

        self.communication_mode = CommunicationMode.ZMQ
        self.stage_core_client = MultiStageCoreClient(base_dir=base_dir)
        logger.info(f"Initialized MultiStageEngineClient in ZMQ mode (base_dir={base_dir})")

    def _shutdown_zmq_mode(self) -> None:
        """Shutdown ZMQ communication mode and cleanup resources."""
        if self.stage_core_client is not None:
            self.stage_core_client.shutdown_all()
            self.stage_core_client = None

    @property
    def use_zmq_mode(self) -> bool:
        """Check if ZMQ communication mode is enabled."""
        return self.communication_mode == CommunicationMode.ZMQ

    # ========== Stage Communication Methods ==========

    async def _send_to_stage_zmq(
        self,
        stage_id: int,
        request_id: str,
        engine_inputs: Any,
        sampling_params: Any,
        **kwargs: Any,
    ) -> None:
        """Send request to a stage using ZMQ mode.

        Args:
            stage_id: Target stage ID
            request_id: Unique request identifier
            engine_inputs: Input data for the engine
            sampling_params: Sampling parameters
            **kwargs: Additional metadata (from_stage, connector_metadata, etc.)
        """
        if self.stage_core_client is None:
            raise RuntimeError("ZMQ mode not initialized")

        from vllm_omni.engine.stage_core import StageCoreRequest

        request = StageCoreRequest(
            request_id=request_id,
            engine_inputs=engine_inputs,
            sampling_params=sampling_params,
            from_stage=kwargs.get("from_stage"),
            to_stage=stage_id,
            connector_metadata=kwargs.get("connector_metadata"),
        )

        await self.stage_core_client.add_request_to_stage(stage_id, request)

    async def _get_output_from_stage_zmq(self, stage_id: int) -> Any:
        """Get output from a stage using ZMQ mode.

        Args:
            stage_id: Source stage ID

        Returns:
            StageCoreOutputs containing the stage results
        """
        if self.stage_core_client is None:
            raise RuntimeError("ZMQ mode not initialized")

        return await self.stage_core_client.get_output_from_stage(stage_id)

    async def _abort_all_stages_zmq(self, request_ids: list[str]) -> None:
        """Abort requests across all stages using ZMQ mode."""
        if self.stage_core_client is not None:
            await self.stage_core_client.abort_all_stages(request_ids)
