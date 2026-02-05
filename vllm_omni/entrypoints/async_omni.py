# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsyncOmni: Multi-stage async engine client implementing vLLM's EngineClient.

Architecture:
    AsyncOmni (EngineClient)
        ├── stage_clients: list[StageClientType]
        │   ├── LLM stages: vLLM's EngineCoreClient (via LLMStageClientWrapper)
        │   └── Non-LLM stages: NonLLMStageClient
        └── output_handler: asyncio.Task

For LLM stages, we use vLLM's EngineCoreClient.make_async_mp_client().
For non-LLM stages (Diffusion, Audio), we use NonLLMStageClient.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.utils import EngineZmqAddresses

from vllm_omni.engine.arg_utils import AsyncOmniEngineArgs
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM
from vllm_omni.entrypoints.omni_stage import _resolve_worker_cls
from vllm_omni.entrypoints.stage_engine.stage_client_factory import (
    StageClientType,
    create_stage_client,
)
from vllm_omni.entrypoints.stage_engine.stage_launcher import (
    StageInfo,
    StageProcManager,
    launch_non_llm_stages,
)
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.v1.executor import Executor

logger = init_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StageConfig:
    """Configuration for a single stage in the pipeline."""

    stage_id: int
    stage_type: str  # "llm", "diffusion", "audio"
    vllm_config: VllmConfig | None = None
    is_final_output: bool = False


@dataclass
class RequestState:
    """Tracks request state across pipeline stages."""

    request_id: str
    prompt: OmniPromptType | None = None
    sampling_params_list: list[OmniSamplingParams] = field(default_factory=list)
    output_queue: asyncio.Queue[OmniRequestOutput | Exception] = field(
        default_factory=asyncio.Queue
    )
    finished_stages: set[int] = field(default_factory=set)
    final_stage_id: int = 0


# =============================================================================
# AsyncOmni
# =============================================================================


class AsyncOmni(EngineClient):
    """Multi-stage async engine client implementing vLLM's EngineClient protocol.

    This class wraps AsyncOmniLLM for single-stage LLM pipelines and provides
    multi-stage pipeline support for LLM + Diffusion/Audio workflows.

    For single-stage LLM pipelines (the common case), it delegates to AsyncOmniLLM.
    For multi-stage pipelines, it orchestrates multiple stage clients.

    Example:
        >>> async_omni = AsyncOmni(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_omni.generate(prompt, params, req_id):
        ...     print(output)
    """

    def __init__(
        self,
        model: str,
        stage_configs: list[StageConfig] | None = None,
        executor_class: type["Executor"] | None = None,
        log_stats: bool = False,
        **kwargs,
    ) -> None:
        """Initialize AsyncOmni.

        Args:
            model: Model name or path
            stage_configs: Optional list of stage configurations for multi-stage
                pipelines. If None, creates a single LLM stage.
            executor_class: Optional executor class override
            log_stats: Whether to log statistics
            **kwargs: Additional arguments passed to AsyncOmniLLM
        """
        
        # Initialize state variables first to avoid AttributeError in __del__
        # if initialization fails partway through
        self._is_running = False
        self._is_stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None
        self._paused = False
        self._inner_llm = None
        self._is_single_stage = True  # Default to single-stage
        self._output_handler: asyncio.Task | None = None
        self._stage_engine_ctx = None
        self.stage_clients: list[StageClientType] = []

        self._model = model
        self._kwargs = kwargs
        self._log_stats = log_stats
        self._executor_class = executor_class

        # For single-stage LLM pipelines, delegate to AsyncOmniLLM
        # This is the common case for models like Qwen2.5-Omni
        if stage_configs is None or len(stage_configs) == 0:
            # Create single LLM stage using AsyncOmniLLM
            self._inner_llm = self._create_async_omni_llm(model, **kwargs)
            self.stage_configs = [StageConfig(stage_id=0, stage_type="llm", is_final_output=True)]
            self._is_single_stage = True
            self.stage_clients = []  # Empty for single-stage (uses _inner_llm)

            # Copy attributes from inner LLM for EngineClient protocol
            self.vllm_config = self._inner_llm.vllm_config
            self.model_config = self._inner_llm.model_config
            self.input_processor = self._inner_llm.input_processor
            self.output_processor = self._inner_llm.output_processor

            logger.info("AsyncOmni initialized with single LLM stage (delegating to AsyncOmniLLM)")
        else:
            # Multi-stage pipeline
            self.stage_configs = stage_configs
            self._is_single_stage = False

            # EngineClient protocol attributes (may be None for non-LLM pipelines)
            self.vllm_config = None
            self.model_config = None
            self.input_processor = None
            self.output_processor = None

            # Stage management (populated by _start())
            self._proc_manager: StageProcManager | None = None
            self._stage_addresses: dict[int, EngineZmqAddresses] = {}
            self._stage_engine_ctx = None
            self.stage_clients: list[StageClientType] = []

            # Request tracking
            self._request_states: dict[str, RequestState] = {}

            # Background task
            self._output_handler: asyncio.Task | None = None

            # Start multi-stage engines
            self._start()

            logger.info("AsyncOmni initialized with %d stages", len(stage_configs))

        # Mark as running after successful initialization
        self._is_running = True

    def _create_async_omni_llm(self, model: str, **kwargs) -> AsyncOmniLLM:
        """Create an AsyncOmniLLM instance from model path and kwargs.

        Args:
            model: Model name or path
            **kwargs: Additional arguments for engine configuration

        Returns:
            Initialized AsyncOmniLLM instance
        """
        # Filter kwargs to only include valid fields for AsyncOmniEngineArgs
        # This is necessary because kwargs may contain extra CLI args like 'subparser'
        valid_fields = {f.name for f in dataclasses.fields(AsyncOmniEngineArgs)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        # Set default worker_type for omni models if not specified
        # This ensures the correct OmniGPUModelRunner is used which can handle
        # OmniOutput (NamedTuple with text_hidden_states and multimodal_outputs)
        if "worker_type" not in filtered_kwargs and "worker_cls" not in filtered_kwargs:
            filtered_kwargs["worker_type"] = "ar"

        # Resolve worker_type to worker_cls (converts "ar" -> GPUARWorker class path)
        _resolve_worker_cls(filtered_kwargs)

        # Build engine args from model and filtered kwargs
        engine_args = AsyncOmniEngineArgs(model=model, **filtered_kwargs)
        vllm_config = engine_args.create_engine_config()

        # Get executor class
        from vllm.v1.executor.abstract import Executor
        executor_class = self._executor_class or Executor.get_class(vllm_config)

        return AsyncOmniLLM.from_vllm_config(
            vllm_config=vllm_config,
            engine_args=engine_args,
            start_engine_loop=True,
            enable_log_requests=kwargs.get("enable_log_requests", False),
            disable_log_stats=not self._log_stats,
        )

    # =========================================================================
    # EngineClient Protocol Properties
    # =========================================================================

    @property
    def renderer(self):
        if self.input_processor is None:
            return None
        return self.input_processor.renderer

    @property
    def is_running(self) -> bool:
        return self._is_running and not self._errored

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        if self._dead_error is None:
            return RuntimeError("Engine is dead")
        return self._dead_error

    # =========================================================================
    # Stage Access
    # =========================================================================

    @property
    def num_stages(self) -> int:
        return len(self.stage_configs)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _start(self) -> None:
        """Start stage engines and create clients."""
        if self._is_running:
            raise RuntimeError("AsyncOmni already started")

        # Separate LLM and non-LLM stages
        non_llm_configs = [c for c in self.stage_configs if c.stage_type != "llm"]

        # Launch non-LLM stages with handshake
        if non_llm_configs:
            non_llm_infos = [
                StageInfo(c.stage_id, c.stage_type, c.vllm_config)
                for c in non_llm_configs
            ]
            self._stage_engine_ctx = launch_non_llm_stages(
                stage_infos=non_llm_infos,
                executor_class=self._executor_class,
                log_stats=self._log_stats,
            )
            self._proc_manager, self._stage_addresses = (
                self._stage_engine_ctx.__enter__()
            )

        # Create stage clients
        for cfg in self.stage_configs:
            if cfg.stage_type == "llm":
                client = create_stage_client(
                    stage_id=cfg.stage_id,
                    stage_type=cfg.stage_type,
                    vllm_config=cfg.vllm_config,
                    executor_class=self._executor_class,
                    input_address=None,
                    output_address=None,
                    log_stats=self._log_stats,
                )
            else:
                addrs = self._stage_addresses[cfg.stage_id]
                client = create_stage_client(
                    stage_id=cfg.stage_id,
                    stage_type=cfg.stage_type,
                    vllm_config=cfg.vllm_config,
                    executor_class=self._executor_class,
                    input_address=addrs.inputs[0],
                    output_address=addrs.outputs[0],
                    log_stats=self._log_stats,
                )
            self.stage_clients.append(client)

        self._is_running = True
        logger.info("AsyncOmni started with %d stage clients", len(self.stage_clients))

    def shutdown(self) -> None:
        """Shutdown the engine."""
        if self._is_stopped:
            return

        logger.info("Shutting down AsyncOmni")
        self._is_stopped = True
        self._is_running = False

        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            self._inner_llm.shutdown()
            logger.info("AsyncOmni shutdown complete (single-stage)")
            return

        # Multi-stage shutdown
        if self._output_handler is not None:
            self._output_handler.cancel()
            self._output_handler = None

        for client in self.stage_clients:
            try:
                client.shutdown()
            except Exception as e:
                logger.warning("Error shutting down stage client: %s", e)

        if self._stage_engine_ctx is not None:
            try:
                self._stage_engine_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing stage engine context: %s", e)

        logger.info("AsyncOmni shutdown complete")

    def __del__(self):
        # Only call shutdown if the object was fully initialized
        if hasattr(self, "_is_stopped"):
            self.shutdown()

    # =========================================================================
    # EngineClient Protocol Methods
    # =========================================================================

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs through the pipeline."""
        if self._errored:
            raise self.dead_error

        if not self._is_running:
            raise RuntimeError("Engine not started")

        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            async for output in self._inner_llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=priority,
                **kwargs,
            ):
                yield output
            return

        # Multi-stage pipeline
        # Create request state
        req_state = RequestState(
            request_id=request_id,
            prompt=prompt,
            sampling_params_list=[sampling_params],
            final_stage_id=self._get_final_stage_id(),
        )
        self._request_states[request_id] = req_state

        try:
            self._ensure_output_handler()

            # Submit to stage 0
            await self.stage_clients[0].submit_request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )

            logger.debug("Submitted request %s to stage 0", request_id)

            # Yield outputs
            while True:
                output = await req_state.output_queue.get()
                if isinstance(output, Exception):
                    raise output
                yield output
                if output.finished:
                    break

        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

        except Exception as e:
            self._errored = True
            self._dead_error = e
            raise

        finally:
            self._request_states.pop(request_id, None)

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        **kwargs,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Encode prompt using pooling."""
        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            async for output in self._inner_llm.encode(
                prompt=prompt,
                pooling_params=pooling_params,
                request_id=request_id,
                **kwargs,
            ):
                yield output
            return
        raise NotImplementedError("Pooling not supported for multi-stage pipelines")
        yield

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort one or more requests."""
        ids = [request_id] if isinstance(request_id, str) else list(request_id)

        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.abort(request_id)
            return

        # Multi-stage abort
        for client in self.stage_clients:
            for rid in ids:
                try:
                    await client.abort_request(rid)
                except Exception as e:
                    logger.debug("Error aborting request %s: %s", rid, e)

        for rid in ids:
            self._request_states.pop(rid, None)

    async def check_health(self) -> None:
        """Check engine health."""
        if self._errored:
            raise self.dead_error

        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.check_health()
            return

        # Multi-stage health check
        for i, client in enumerate(self.stage_clients):
            if not await client.check_health():
                raise RuntimeError(f"Stage {i} is unhealthy")

    async def get_vllm_config(self) -> VllmConfig | None:
        """Get the vLLM configuration."""
        return self.vllm_config

    async def is_tracing_enabled(self) -> bool:
        if self._is_single_stage and self._inner_llm is not None:
            return await self._inner_llm.is_tracing_enabled()
        return False

    async def do_log_stats(self) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.do_log_stats()

    async def start_profile(self) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.start_profile()

    async def stop_profile(self) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.stop_profile()

    async def reset_mm_cache(self) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.reset_mm_cache()

    async def reset_encoder_cache(self) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.reset_encoder_cache()

    async def reset_prefix_cache(self, **kwargs) -> bool:
        if self._is_single_stage and self._inner_llm is not None:
            return await self._inner_llm.reset_prefix_cache(**kwargs)
        return True

    async def sleep(self, level: int = 1) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.sleep(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        if self._is_single_stage and self._inner_llm is not None:
            await self._inner_llm.wake_up(tags)

    async def is_sleeping(self) -> bool:
        if self._is_single_stage and self._inner_llm is not None:
            return await self._inner_llm.is_sleeping()
        return False

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        if self._is_single_stage and self._inner_llm is not None:
            return await self._inner_llm.add_lora(lora_request)
        return False

    async def get_tokenizer(self, lora_request: LoRARequest | None = None):
        """Get the tokenizer for the engine."""
        # For single-stage, delegate to inner LLM
        if self._is_single_stage and self._inner_llm is not None:
            return await self._inner_llm.get_tokenizer(lora_request)

        # Multi-stage: try input processor first
        if self.input_processor is not None:
            return self.input_processor.tokenizer
        # Try to get tokenizer from first LLM stage client
        for client in self.stage_clients:
            if hasattr(client, "inner_client") and hasattr(client.inner_client, "get_tokenizer"):
                return await client.inner_client.get_tokenizer(lora_request)
        return None

    def is_paused(self) -> bool:
        """Check if generation is paused."""
        return self._paused

    async def pause_generation(self) -> None:
        """Pause generation (for async RL workflows)."""
        self._paused = True

    async def resume_generation(self) -> None:
        """Resume generation (for async RL workflows)."""
        self._paused = False

    # =========================================================================
    # Output Handler
    # =========================================================================

    def _ensure_output_handler(self) -> None:
        """Ensure output handler task is running."""
        if self._output_handler is not None and not self._output_handler.done():
            return
        self._output_handler = asyncio.create_task(self._run_output_handler())

    async def _run_output_handler(self) -> None:
        """Background task to route outputs between stages."""
        try:
            while self._is_running:
                await self._poll_stage_outputs()
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Output handler failed")
            self._errored = True
            self._dead_error = e

    async def _poll_stage_outputs(self) -> None:
        """Poll all stages for outputs."""
        for stage_id, client in enumerate(self.stage_clients):
            try:
                output = await asyncio.wait_for(
                    client.get_output_async(), timeout=0.001
                )
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue

            if output is None:
                continue

            req_state = self._request_states.get(output.request_id)
            if req_state is None:
                continue

            if output.finished:
                req_state.finished_stages.add(stage_id)
                if stage_id < req_state.final_stage_id:
                    await self._route_to_next_stage(req_state, stage_id, output)
                else:
                    await req_state.output_queue.put(output)
            else:
                await req_state.output_queue.put(output)

    async def _route_to_next_stage(
        self, req_state: RequestState, current_stage: int, output: Any
    ) -> None:
        """Route output to the next stage."""
        next_stage = current_stage + 1
        if next_stage >= self.num_stages:
            return

        params_idx = min(next_stage, len(req_state.sampling_params_list) - 1)
        next_params = req_state.sampling_params_list[params_idx]

        await self.stage_clients[next_stage].submit_request(
            request_id=req_state.request_id,
            prompt=output,
            sampling_params=next_params,
        )

        logger.debug(
            "Routed request %s: stage %d -> %d",
            req_state.request_id,
            current_stage,
            next_stage,
        )

    def _get_final_stage_id(self) -> int:
        """Get the ID of the final output stage."""
        for i in range(len(self.stage_configs) - 1, -1, -1):
            if self.stage_configs[i].is_final_output:
                return i
        return len(self.stage_configs) - 1
