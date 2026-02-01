"""Multi-stage engine client implementing vLLM's EngineClient protocol.

This module provides MultiStageEngineClient, which implements the EngineClient
protocol for multi-stage vLLM-Omni pipelines. It delegates orchestration to
PipelineOrchestrator and provides a unified interface for multi-stage execution.

Based on vLLM's AsyncLLM (vllm/v1/engine/async_llm.py).
"""

import asyncio
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs.data import PromptType, StreamingInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.pipeline_orchestrator import PipelineOrchestrator
from vllm_omni.entrypoints.stage_context import StageContext
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class MultiStageEngineClient(EngineClient):
    """Multi-stage engine client implementing EngineClient protocol.

    Provides a unified entry point for multi-stage vLLM-Omni pipelines,
    delegating orchestration to PipelineOrchestrator while maintaining
    compatibility with vLLM's EngineClient interface.

    Attributes:
        orchestrator: PipelineOrchestrator for managing stage execution
        vllm_config: VllmConfig from stage 0
        model_config: ModelConfig from stage 0
        input_processor: InputProcessor from stage 0
        io_processor: IOProcessor from stage 0
    """

    def __init__(
        self,
        stages: list[StageContext],
        execution_mode: str = "sequential",
    ):
        """Initialize the multi-stage engine client.

        Args:
            stages: List of StageContext instances
            execution_mode: "sequential" or "async_chunk"
        """
        self.orchestrator = PipelineOrchestrator(stages, execution_mode)
        self.stages = stages

        # Get config from stage 0
        stage_0 = stages[0]
        self.vllm_config = stage_0.stage_config.vllm_config
        self.model_config = stage_0.stage_config.vllm_config.model_config if stage_0.stage_config.vllm_config else None
        self.input_processor = stage_0.stage_config.input_preprocessor
        self.io_processor = None  # TODO: Get from stage 0 if available

        # State tracking
        self._is_running = True
        self._is_stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None
        self._is_paused = False

        logger.info(f"MultiStageEngineClient initialized with {len(stages)} stages")

    @property
    def renderer(self) -> BaseRenderer:
        """Get the renderer from stage 0."""
        # TODO: Implement renderer support
        raise NotImplementedError("Renderer not yet supported for multi-stage pipelines")

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self._is_running

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self._is_stopped

    @property
    def errored(self) -> bool:
        """Check if the engine has errored."""
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        """Get the error that caused the engine to die."""
        if self._dead_error is None:
            raise RuntimeError("Engine has not errored")
        return self._dead_error

    def generate(
        self,
        prompt: EngineCoreRequest | PromptType | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Args:
            prompt: Input prompt (will be converted to OmniPromptType)
            sampling_params: Sampling parameters for stage 0
            request_id: Unique request identifier
            prompt_text: Optional prompt text
            lora_request: Optional LoRA request
            tokenization_kwargs: Optional tokenization kwargs
            trace_headers: Optional trace headers
            priority: Request priority
            data_parallel_rank: Optional data parallel rank

        Yields:
            RequestOutput instances (wrapped OmniRequestOutput)
        """
        return self._generate_impl(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_text=prompt_text,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )

    async def _generate_impl(
        self,
        prompt: EngineCoreRequest | PromptType | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Internal generate implementation."""
        if self._is_paused:
            raise RuntimeError("Engine is paused")

        if self._is_stopped:
            raise RuntimeError("Engine is stopped")

        # Convert prompt to OmniPromptType
        omni_prompt = self._convert_to_omni_prompt(prompt)

        # Create sampling params list (one per stage)
        # For now, use the same sampling params for all stages
        # TODO: Support per-stage sampling params
        sampling_params_list = [sampling_params] * len(self.stages)

        # Create metrics tracker
        metrics = OrchestratorMetrics()

        # Get final stage ID
        final_stage_id = len(self.stages) - 1
        for i in range(len(self.stages) - 1, -1, -1):
            if self.stages[i].is_final_output:
                final_stage_id = i
                break

        # Submit request to orchestrator
        await self.orchestrator.submit_request(
            request_id=request_id,
            prompt=omni_prompt,
            sampling_params_list=sampling_params_list,
        )

        # Process pipeline and yield outputs
        try:
            async for omni_output in self.orchestrator.process_pipeline(
                request_id=request_id,
                prompt=omni_prompt,
                sampling_params_list=sampling_params_list,
                metrics=metrics,
                final_stage_id=final_stage_id,
            ):
                # Convert OmniRequestOutput to RequestOutput
                # For compatibility, we yield the wrapped request_output
                if omni_output.request_output is not None:
                    yield omni_output.request_output
                else:
                    # For diffusion outputs, we need to create a RequestOutput-like object
                    # This is a simplified implementation
                    logger.warning(
                        f"Diffusion output for request {request_id} - "
                        "RequestOutput conversion not fully implemented"
                    )

        except Exception as e:
            logger.exception(f"Error generating for request {request_id}")
            self._errored = True
            self._dead_error = e
            raise

    def _convert_to_omni_prompt(
        self, prompt: EngineCoreRequest | PromptType | AsyncGenerator[StreamingInput, None]
    ) -> OmniPromptType:
        """Convert vLLM prompt to OmniPromptType.

        Args:
            prompt: vLLM prompt

        Returns:
            OmniPromptType
        """
        # For now, pass through as-is
        # TODO: Implement proper conversion if needed
        return prompt  # type: ignore

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model.

        Not supported for multi-stage pipelines.
        """
        raise NotImplementedError("Pooling not supported for multi-stage pipelines")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request, or an iterable of such ids.
        """
        if isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = list(request_id)

        for rid in request_ids:
            await self.orchestrator.abort_request(rid)

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        # Check stage 0
        return self.stages[0].stage_config.is_tracing_enabled

    async def do_log_stats(self) -> None:
        """Log statistics."""
        # TODO: Implement stats logging for multi-stage pipelines
        logger.debug("Stats logging not yet implemented for multi-stage pipelines")

    async def check_health(self) -> None:
        """Raise if unhealthy."""
        # Check health of all stages
        for stage_ctx in self.stages:
            is_healthy = await stage_ctx.client.check_health()
            if not is_healthy:
                raise RuntimeError(f"Stage {stage_ctx.stage_id} is unhealthy")

    async def start_profile(self) -> None:
        """Start profiling the engine."""
        # TODO: Implement profiling for multi-stage pipelines
        logger.warning("Profiling not yet implemented for multi-stage pipelines")

    async def stop_profile(self) -> None:
        """Stop profiling the engine."""
        # TODO: Implement profiling for multi-stage pipelines
        logger.warning("Profiling not yet implemented for multi-stage pipelines")

    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache."""
        # TODO: Implement cache reset for multi-stage pipelines
        logger.warning("MM cache reset not yet implemented for multi-stage pipelines")

    async def reset_encoder_cache(self) -> None:
        """Reset the encoder cache."""
        # TODO: Implement cache reset for multi-stage pipelines
        logger.warning("Encoder cache reset not yet implemented for multi-stage pipelines")

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache and optionally any configured connector cache."""
        # TODO: Implement cache reset for multi-stage pipelines
        logger.warning("Prefix cache reset not yet implemented for multi-stage pipelines")
        return False

    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine."""
        # TODO: Implement sleep for multi-stage pipelines
        logger.warning("Sleep not yet implemented for multi-stage pipelines")

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the engine."""
        # TODO: Implement wake up for multi-stage pipelines
        logger.warning("Wake up not yet implemented for multi-stage pipelines")

    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping."""
        # TODO: Implement sleep check for multi-stage pipelines
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # TODO: Implement LoRA support for multi-stage pipelines
        logger.warning("LoRA not yet implemented for multi-stage pipelines")
        return False

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause new generation/encoding requests.

        Args:
            wait_for_inflight_requests: When True waits for in-flight requests
                to finish before pausing.
            clear_cache: Whether to clear KV and prefix caches after draining.
        """
        self._is_paused = True
        logger.info("Multi-stage engine paused")

    async def resume_generation(self) -> None:
        """Resume accepting generation/encoding requests."""
        self._is_paused = False
        logger.info("Multi-stage engine resumed")

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        return self._is_paused

    async def shutdown(self):
        """Shutdown the multi-stage engine."""
        logger.info("Shutting down multi-stage engine")
        self._is_stopped = True
        self._is_running = False

        await self.orchestrator.shutdown()

        logger.info("Multi-stage engine shutdown complete")
