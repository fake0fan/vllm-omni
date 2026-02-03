# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsyncOmni: Multi-stage async engine client.

This module provides AsyncOmni, which implements the EngineClient protocol
for multi-stage vLLM-Omni pipelines. It follows the same pattern as vLLM's
AsyncLLM, using EngineCoreClient for each stage.

Architecture:
    AsyncOmni (EngineClient)
        ├── input_processor: InputProcessor (from vLLM)
        ├── output_processor: MultimodalOutputProcessor
        ├── stage_clients: list[StageClient] (one per stage)
        │   ├── LLM stages: LLMStageClientWrapper (wraps vLLM's AsyncMPClient)
        │   └── Non-LLM stages: NonLLMStageClient (custom ZMQ client)
        └── output_handler: asyncio.Task (background output routing)

Each stage runs in a background process with ZMQ communication,
similar to vLLM's EngineCoreProc/EngineCoreClient pattern.

For LLM stages, we directly use vLLM's EngineCoreClient.make_async_mp_client()
to leverage all vLLM optimizations (scheduler, KV cache, etc.).

For non-LLM stages (Diffusion, Audio), we use custom NonLLMStageClient.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import IOProcessor, get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import RequestOutputCollector

from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.stage_engine.stage_client_factory import (
    StageClientType,
    create_stage_client,
)
from vllm_omni.entrypoints.stage_engine.stage_client_protocol import (
    LLMStageClientWrapper,
)
from vllm_omni.entrypoints.stage_engine.stage_core_client import NonLLMStageClient
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.inputs.data import StreamingInput

    from vllm_omni.entrypoints.omni_stage import OmniStage

logger = init_logger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single stage.

    Attributes:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("llm", "diffusion", "audio")
        vllm_config: VllmConfig for this stage (None for non-LLM stages)
        model_config: ModelConfig for this stage
        tokenizer: Tokenizer for this stage
        is_final_output: Whether this stage produces final output
        final_output_type: Type of final output ("text", "image", "audio")
        input_address: ZMQ address for sending requests
        output_address: ZMQ address for receiving outputs
    """

    stage_id: int
    stage_type: str  # "llm", "diffusion", "audio"
    vllm_config: VllmConfig | None = None
    model_config: ModelConfig | None = None
    tokenizer: Any = None
    is_final_output: bool = False
    final_output_type: str | None = None
    input_address: str = ""
    output_address: str = ""


@dataclass
class RequestState:
    """Tracks request state across stages."""

    request_id: str
    current_stage: int = 0
    prompt: OmniPromptType | None = None
    sampling_params_list: list[OmniSamplingParams] = field(default_factory=list)
    queue: RequestOutputCollector | None = None
    finished_stages: set[int] = field(default_factory=set)
    final_stage_id: int = 0


class AsyncOmni(EngineClient):
    """Asynchronous multi-stage engine client implementing EngineClient protocol.

    This class follows the same pattern as vLLM's AsyncLLM:
    - Uses StageClient (via create_stage_client factory) for each stage
    - InputProcessor for request preprocessing
    - OutputProcessor for output handling
    - Background output_handler task for streaming

    Each stage has its own configuration (vllm_config, model_config, tokenizer).
    For EngineClient protocol compatibility, stage 0's config is exposed as primary.

    Example:
        >>> async_omni = AsyncOmni(
        ...     stage_configs=[stage_0_config, stage_1_config],
        ...     executor_class=MultiprocExecutor,
        ... )
        >>> async for output in async_omni.generate(prompt, sampling_params, request_id):
        ...     print(output)
    """

    def __init__(
        self,
        stage_configs: list[StageConfig],
        executor_class: type | None = None,
        log_stats: bool = False,
        log_requests: bool = True,
    ) -> None:
        """Initialize AsyncOmni.

        Args:
            stage_configs: List of StageConfig for each stage
            executor_class: Executor class for stage processes
            log_stats: Whether to log statistics
            log_requests: Whether to log requests
        """
        self.stage_configs = stage_configs
        self.log_stats = log_stats
        self.log_requests = log_requests

        # Get primary config from stage 0 (for EngineClient protocol)
        stage_0 = stage_configs[0]
        self._vllm_config = stage_0.vllm_config
        self._model_config = stage_0.model_config or (
            stage_0.vllm_config.model_config if stage_0.vllm_config else None
        )

        # Initialize InputProcessor from stage 0
        if self._vllm_config:
            self._input_processor = InputProcessor(self._vllm_config)
            self._io_processor = get_io_processor(
                self._vllm_config,
                self._model_config.io_processor_plugin if self._model_config else None,
            )
        else:
            self._input_processor = None
            self._io_processor = None

        # Initialize OutputProcessor with multimodal support
        tokenizer = stage_0.tokenizer or (
            self._input_processor.tokenizer if self._input_processor else None
        )
        self._output_processor = MultimodalOutputProcessor(
            tokenizer=tokenizer,
            log_stats=log_stats,
        )

        # Create stage clients for each stage using the factory
        # For LLM stages: uses vLLM's EngineCoreClient.make_async_mp_client()
        # For non-LLM stages: uses NonLLMStageClient.make_client()
        self.stage_clients: list[StageClientType] = []
        for cfg in stage_configs:
            client = create_stage_client(
                stage_id=cfg.stage_id,
                stage_type=cfg.stage_type,
                vllm_config=cfg.vllm_config,
                executor_class=executor_class,
                input_address=cfg.input_address,
                output_address=cfg.output_address,
                log_stats=log_stats,
            )
            self.stage_clients.append(client)

        # Request state tracking
        self._request_states: dict[str, RequestState] = {}

        # Engine state
        self._is_running = True
        self._is_stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None

        # Pause/resume control
        self._pause_cond = asyncio.Condition()
        self._paused = False

        # Background output handler (similar to AsyncLLM.output_handler)
        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        logger.info(
            f"AsyncOmni initialized with {len(stage_configs)} stages, "
            f"log_stats={log_stats}, log_requests={log_requests}"
        )

    # ========== Per-Stage Config Access ==========

    def get_stage_config(self, stage_id: int) -> StageConfig:
        """Get configuration for a specific stage."""
        if stage_id < 0 or stage_id >= len(self.stage_configs):
            raise ValueError(f"Invalid stage_id: {stage_id}")
        return self.stage_configs[stage_id]

    def get_stage_vllm_config(self, stage_id: int) -> VllmConfig | None:
        """Get VllmConfig for a specific stage."""
        return self.get_stage_config(stage_id).vllm_config

    def get_stage_model_config(self, stage_id: int) -> ModelConfig | None:
        """Get ModelConfig for a specific stage."""
        return self.get_stage_config(stage_id).model_config

    def get_stage_tokenizer(self, stage_id: int) -> Any:
        """Get tokenizer for a specific stage."""
        return self.get_stage_config(stage_id).tokenizer

    def get_stage_type(self, stage_id: int) -> str:
        """Get stage type ("llm", "diffusion", "audio")."""
        return self.get_stage_config(stage_id).stage_type

    @property
    def num_stages(self) -> int:
        """Number of stages in the pipeline."""
        return len(self.stage_configs)

    # ========== EngineClient Protocol Properties (from stage 0) ==========

    @property
    def vllm_config(self) -> VllmConfig:
        """VllmConfig from stage 0 (primary LLM stage)."""
        if self._vllm_config is None:
            raise RuntimeError(
                "No VllmConfig available. Stage 0 may not be an LLM stage. "
                "Use get_stage_vllm_config(stage_id) for per-stage access."
            )
        return self._vllm_config

    @property
    def model_config(self) -> ModelConfig:
        """ModelConfig from stage 0 (primary LLM stage)."""
        if self._model_config is None:
            raise RuntimeError(
                "No ModelConfig available. Stage 0 may not be an LLM stage. "
                "Use get_stage_model_config(stage_id) for per-stage access."
            )
        return self._model_config

    @property
    def input_processor(self) -> InputProcessor:
        """InputProcessor from stage 0."""
        if self._input_processor is None:
            raise RuntimeError("No InputProcessor available.")
        return self._input_processor

    @property
    def io_processor(self) -> IOProcessor | None:
        """IOProcessor from stage 0."""
        return self._io_processor

    @property
    def renderer(self) -> BaseRenderer:
        """Renderer from InputProcessor."""
        return self.input_processor.renderer

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._is_running and not self._errored

    @property
    def is_stopped(self) -> bool:
        """Check if engine is stopped."""
        return self._is_stopped

    @property
    def errored(self) -> bool:
        """Check if engine has errored."""
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        """Get the error that caused engine death."""
        if self._dead_error is None:
            return EngineDeadError()
        return self._dead_error

    # ========== Output Handler (similar to AsyncLLM._run_output_handler) ==========

    def _run_output_handler(self) -> None:
        """Start the background output handler task."""
        if self.output_handler is not None:
            return

        stage_clients = self.stage_clients
        stage_configs = self.stage_configs
        request_states = self._request_states
        output_processor = self._output_processor

        async def handler():
            """Background task: pull outputs from all stages and route them."""
            try:
                while True:
                    # Poll all stage clients for outputs
                    for stage_id, client in enumerate(stage_clients):
                        try:
                            # Non-blocking check for outputs
                            output = await asyncio.wait_for(
                                client.get_output_async(),
                                timeout=0.001,
                            )
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.exception(
                                f"Error getting output from stage {stage_id}: {e}"
                            )
                            continue

                        if output is None:
                            continue

                        req_id = output.request_id
                        req_state = request_states.get(req_id)
                        if req_state is None:
                            logger.debug(
                                f"Request {req_id} not found, may have been aborted"
                            )
                            continue

                        # Check if this stage is finished
                        if output.finished:
                            req_state.finished_stages.add(stage_id)

                            # Route to next stage if not final
                            if stage_id < req_state.final_stage_id:
                                await self._route_to_next_stage(
                                    req_state, stage_id, output
                                )
                            else:
                                # Final stage - put output in queue
                                if req_state.queue is not None:
                                    req_state.queue.put(output)
                        else:
                            # Intermediate output - put in queue for streaming
                            if req_state.queue is not None:
                                req_state.queue.put(output)

                    # Small sleep to avoid busy loop
                    await asyncio.sleep(0.001)

            except Exception as e:
                logger.exception("Output handler failed")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(handler())

    async def _route_to_next_stage(
        self,
        req_state: RequestState,
        current_stage_id: int,
        output: Any,
    ) -> None:
        """Route request output to the next stage."""
        next_stage_id = current_stage_id + 1
        if next_stage_id >= len(self.stage_configs):
            return

        next_stage_cfg = self.stage_configs[next_stage_id]
        next_client = self.stage_clients[next_stage_id]

        # Get sampling params for next stage
        if next_stage_id < len(req_state.sampling_params_list):
            next_sampling_params = req_state.sampling_params_list[next_stage_id]
        else:
            next_sampling_params = req_state.sampling_params_list[0]

        # Prepare input for next stage based on current output
        # TODO: Use connectors for more complex transformations
        next_prompt = self._prepare_next_stage_input(
            current_stage_id, next_stage_id, output, req_state.prompt
        )

        # Submit to next stage
        await next_client.submit_request(
            request_id=req_state.request_id,
            prompt=next_prompt,
            sampling_params=next_sampling_params,
        )

        logger.debug(
            f"Routed request {req_state.request_id} from stage "
            f"{current_stage_id} to {next_stage_id}"
        )

    def _prepare_next_stage_input(
        self,
        current_stage_id: int,
        next_stage_id: int,
        output: Any,
        original_prompt: Any,
    ) -> Any:
        """Prepare input for the next stage based on current output.

        This is a simplified version. For complex transformations,
        use OmniConnectors.
        """
        # For now, pass the output directly
        # TODO: Implement proper connector-based transformation
        return output

    def _get_final_stage_id(self, output_modalities: list[str] | None = None) -> int:
        """Get the final stage ID based on output modalities."""
        for i in range(len(self.stage_configs) - 1, -1, -1):
            if self.stage_configs[i].is_final_output:
                return i
        return len(self.stage_configs) - 1

    # ========== Core Generation Method ==========

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType | AsyncGenerator["StreamingInput", None],
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
        """Generate outputs for a request through the multi-stage pipeline.

        Args:
            prompt: Input prompt
            sampling_params: Sampling parameters for stage 0
            request_id: Unique request identifier
            prompt_text: Optional prompt text
            lora_request: Optional LoRA request
            tokenization_kwargs: Optional tokenization kwargs
            trace_headers: Optional trace headers
            priority: Request priority
            data_parallel_rank: Optional data parallel rank

        Yields:
            RequestOutput instances as they are produced
        """
        if self._errored:
            raise EngineDeadError()

        # Start output handler on first request
        self._run_output_handler()

        # Wait for pause to be released
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        queue: RequestOutputCollector | None = None
        try:
            # Process input through InputProcessor
            if self._input_processor and not isinstance(prompt, EngineCoreRequest):
                request = self._input_processor.process_inputs(
                    request_id,
                    prompt,
                    sampling_params,
                    arrival_time=None,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                )
                self._input_processor.assign_request_id(request)
                internal_request_id = request.request_id
            else:
                internal_request_id = request_id

            # Create output collector
            queue = RequestOutputCollector(
                sampling_params.output_kind,
                internal_request_id,
            )

            # Determine final stage
            final_stage_id = self._get_final_stage_id()

            # Create request state
            req_state = RequestState(
                request_id=internal_request_id,
                current_stage=0,
                prompt=prompt,
                sampling_params_list=[sampling_params],
                queue=queue,
                final_stage_id=final_stage_id,
            )
            self._request_states[internal_request_id] = req_state

            # Add to output processor
            if self._input_processor and not isinstance(prompt, EngineCoreRequest):
                self._output_processor.add_request(request, prompt_text, queue=queue)

            # Submit to stage 0
            await self.stage_clients[0].submit_request(
                request_id=internal_request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )

            if self.log_requests:
                logger.info(f"Added request {request_id} to stage 0")

            # Yield outputs from queue
            finished = False
            while not finished:
                out = queue.get_nowait() or await queue.get()
                assert isinstance(out, (RequestOutput, OmniRequestOutput))
                finished = out.finished
                yield out

        except asyncio.CancelledError:
            if queue is not None:
                await self.abort(queue.request_id)
            if self.log_requests:
                logger.info(f"Request {request_id} cancelled")
            raise

        except EngineDeadError:
            if self.log_requests:
                logger.info(f"Request {request_id} failed (engine dead)")
            raise

        except Exception as e:
            if queue is not None:
                await self.abort(queue.request_id)
            if self.log_requests:
                logger.exception(f"Request {request_id} failed: {e}")
            self._errored = True
            self._dead_error = e
            raise

        finally:
            if queue is not None:
                queue.close()
            self._request_states.pop(request_id, None)

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a pooling request."""
        raise NotImplementedError("Pooling not supported for multi-stage pipelines")

    # ========== Request Management ==========

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort one or more requests."""
        if isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = list(request_id)

        # Abort in output processor
        all_ids = self._output_processor.abort_requests(request_ids, internal=False)

        # Abort in all stage clients
        for client in self.stage_clients:
            for rid in all_ids:
                await client.abort_request(rid)

        # Clean up request states
        for rid in request_ids:
            self._request_states.pop(rid, None)

        if self.log_requests:
            logger.info(f"Aborted request(s): {request_ids}")

    # ========== Engine Control ==========

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if not wait_for_inflight_requests:
            request_ids = list(self._request_states.keys())
            if request_ids:
                await self.abort(request_ids)

        if self._output_processor.has_unfinished_requests():
            await self._output_processor.wait_for_requests_to_drain()

        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

        logger.info("AsyncOmni paused")

    async def resume_generation(self) -> None:
        """Resume generation after pause."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()
        logger.info("AsyncOmni resumed")

    async def is_paused(self) -> bool:
        """Check if engine is paused."""
        async with self._pause_cond:
            return self._paused

    # ========== Monitoring & Health ==========

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return False

    async def do_log_stats(self) -> None:
        """Log statistics."""
        pass

    async def check_health(self) -> None:
        """Check engine health."""
        if self._errored:
            raise self.dead_error

        for i, client in enumerate(self.stage_clients):
            is_healthy = await client.check_health()
            if not is_healthy:
                raise RuntimeError(f"Stage {i} is unhealthy")

    async def start_profile(self) -> None:
        """Start profiling."""
        logger.warning("Profiling not yet implemented for multi-stage pipelines")

    async def stop_profile(self) -> None:
        """Stop profiling."""
        logger.warning("Profiling not yet implemented for multi-stage pipelines")

    # ========== Cache Management ==========

    async def reset_mm_cache(self) -> None:
        """Reset multimodal cache."""
        if self._input_processor:
            self._input_processor.clear_mm_cache()

    async def reset_encoder_cache(self) -> None:
        """Reset encoder cache."""
        pass

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        """Reset prefix cache."""
        return True

    # ========== Sleep/Wake ==========

    async def sleep(self, level: int = 1) -> None:
        """Put engine to sleep."""
        await self.reset_prefix_cache()

    async def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up engine."""
        pass

    async def is_sleeping(self) -> bool:
        """Check if engine is sleeping."""
        return False

    # ========== LoRA ==========

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add LoRA adapter."""
        logger.warning("LoRA not yet implemented for multi-stage pipelines")
        return False

    # ========== Shutdown ==========

    def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("Shutting down AsyncOmni")

        self._is_stopped = True
        self._is_running = False

        # Cancel output handler
        if self.output_handler is not None:
            self.output_handler.cancel()
            self.output_handler = None

        # Shutdown all stage clients
        for client in self.stage_clients:
            try:
                client.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down stage client: {e}")

        # Close input processor
        if self._input_processor and hasattr(self._input_processor, "close"):
            self._input_processor.close()

        logger.info("AsyncOmni shutdown complete")

    def __del__(self):
        """Destructor."""
        self.shutdown()

    # ========== Factory Methods ==========

    @classmethod
    def from_omni_stages(
        cls,
        stages: list["OmniStage"],
        executor_class: type | None = None,
        log_stats: bool = False,
        log_requests: bool = True,
    ) -> "AsyncOmni":
        """Create AsyncOmni from OmniStage instances.

        Args:
            stages: List of OmniStage instances
            executor_class: Executor class for stage processes
            log_stats: Whether to log statistics
            log_requests: Whether to log requests

        Returns:
            Initialized AsyncOmni instance
        """
        stage_configs = []
        for stage in stages:
            cfg = StageConfig(
                stage_id=stage.stage_id,
                stage_type=stage.stage_type,
                vllm_config=getattr(stage, "vllm_config", None),
                model_config=getattr(stage, "model_config", None),
                tokenizer=getattr(stage, "tokenizer", None),
                is_final_output=getattr(stage, "final_output", False),
                final_output_type=getattr(stage, "final_output_type", None),
                input_address=f"tcp://127.0.0.1:{5555 + stage.stage_id * 2}",
                output_address=f"tcp://127.0.0.1:{5556 + stage.stage_id * 2}",
            )
            stage_configs.append(cfg)

        return cls(
            stage_configs=stage_configs,
            executor_class=executor_class,
            log_stats=log_stats,
            log_requests=log_requests,
        )
