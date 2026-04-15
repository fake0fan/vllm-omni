"""
Pipeline runtime for vLLM-Omni multi-stage execution.

Runs inside a background thread with its own asyncio event loop.
Owns all StageEngineCoreClient instances, input/output processors,
and handles stage-to-stage transfer logic.
"""

from __future__ import annotations

import asyncio
import copy
import time as _time
from typing import Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.pipeline_state import PipelineData, PipelineRequestState, RequestMeta
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_runtime import StageRuntime, build_stage_runtimes
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

logger = init_logger(__name__)


async def _accept_prebuilt_llm_entry_request(
    self: StageRuntime,
    *,
    meta: RequestMeta,
    data: PipelineData,
) -> Any:
    request = data.stage0_request if data.stage0_request is not None else data.raw_prompt
    data.stage0_request = request
    await self.submit(request=request, request_id=meta.request_id, params=meta.entry_params)
    return request


async def _accept_prebuilt_llm_streaming_update(
    self: StageRuntime,
    *,
    meta: RequestMeta,
    data: PipelineData,
) -> Any:
    return await _accept_prebuilt_llm_entry_request(self, meta=meta, data=data)


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt.

    Lightweight alternative to the full InputProcessor pipeline - skips
    tokenization, multimodal preprocessing, LoRA validation, and platform
    validation. Intended for stage 1+ where the upstream stage has already
    produced token IDs and optional embeddings.
    """
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        additional_information=additional_info_payload,
    )


class PipelineRuntime:
    """Runs inside a background thread's asyncio event loop.

    Owns all StageEngineCoreClient instances, input/output processors,
    and handles stage-to-stage transfer logic.
    """

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[dict[str, Any]],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_clients: list[Any],
        output_processors: list[Any],
        stage_vllm_configs: list[Any],
        *,
        async_chunk: bool = False,
        entry_input_processor: Any | None = None,
        supported_tasks: tuple[str, ...] | None = None,
        prompt_expand_func: Any | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.output_processors: list[Any] = output_processors
        self.stage_vllm_configs: list[Any] = stage_vllm_configs
        self.entry_stage_id = stage_clients[0].stage_id if stage_clients else None
        self.stage_runtimes: list[StageRuntime] = build_stage_runtimes(
            stage_clients=stage_clients,
            output_processors=output_processors,
            stage_vllm_configs=stage_vllm_configs,
            entry_stage_id=self.entry_stage_id,
            entry_input_processor=entry_input_processor,
            supported_tasks=supported_tasks,
        )
        self.entry_stage_pos = next(
            (idx for idx, runtime in enumerate(self.stage_runtimes) if runtime.stage_id == self.entry_stage_id),
            None,
        )
        self.entry_runtime = (
            self.stage_runtimes[self.entry_stage_pos] if self.entry_stage_pos is not None else None
        )
        self._entry_uses_prebuilt_request = bool(
            self.entry_runtime is not None
            and self.entry_runtime.stage_type == "llm"
            and getattr(self.entry_runtime, "input_processor", None) is None
        )
        if self._entry_uses_prebuilt_request:
            self.entry_runtime.accept_external_request = _accept_prebuilt_llm_entry_request.__get__(
                self.entry_runtime,
                type(self.entry_runtime),
            )
            self.entry_runtime.accept_streaming_update = _accept_prebuilt_llm_streaming_update.__get__(
                self.entry_runtime,
                type(self.entry_runtime),
            )
        self.prompt_expand_func = prompt_expand_func
        self.num_stages = len(self.stage_runtimes)
        self.stage_clients: list[Any] = [runtime.stage_client for runtime in self.stage_runtimes]

        self.request_states: dict[str, PipelineRequestState] = {}

        self._companion_map: dict[str, dict[str, str]] = {}
        self._companion_to_parent: dict[str, str] = {}
        self._companion_ids: set[str] = set()
        self._companion_done: dict[str, set[str]] = {}
        self._deferred_parents: dict[str, dict[str, Any]] = {}

        self._batch_seq: list[int] = [0] * self.num_stages
        self._agg_total_tokens: list[int] = [0] * self.num_stages
        self._agg_total_gen_time_ms: list[float] = [0.0] * self.num_stages

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False

    async def run(self) -> None:
        """Main entry point for the PipelineRuntime event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )

        try:
            await asyncio.gather(request_task, output_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in (request_task, output_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(request_task, output_task, return_exceptions=True)
            except Exception:
                pass

            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task() and not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.get("type")

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif msg_type == "shutdown":
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                self._shutdown_stages()
                break
            else:
                logger.warning(f"[Orchestrator] Unknown message type: {msg_type}")

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Inner loop for _orchestration_output_handler (clean cancellation)."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id, runtime in enumerate(self.stage_runtimes):
                if self._shutdown_event.is_set():
                    return

                try:
                    poll_result = await asyncio.wait_for(runtime.poll_processed_outputs(), timeout=0.001)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if self._shutdown_event.is_set():
                        return
                    logger.exception(
                        "[Orchestrator] poll_processed_outputs failed for stage-%s",
                        stage_id,
                    )
                    raise

                if not poll_result.request_outputs and not poll_result.kv_ready_outputs:
                    continue
                idle = False

                await self._handle_kv_ready_outputs(stage_id, poll_result.kv_ready_outputs)

                for output in poll_result.request_outputs:
                    req_state = self.request_states.get(output.request_id)
                    if req_state is None:
                        logger.warning(
                            "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                            output.request_id,
                            stage_id,
                            list(self.request_states.keys()),
                        )
                        continue
                    stage_metrics = None
                    if output.finished:
                        stage_metrics = self._build_stage_metrics(
                            stage_id,
                            output.request_id,
                            [output],
                            req_state,
                        )
                    await self._route_output(stage_id, output, req_state, stage_metrics)

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _route_output(
        self,
        stage_id: int,
        output: Any,
        req_state: PipelineRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to main thread and/or forward to next stage."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)
        runtime = self.stage_runtimes[stage_id]

        if finished:
            req_state.mark_stage_finished(stage_id)

        if finished and req_id in self._companion_ids:
            await self._handle_cfg_companion_ready(req_id)
            self.request_states.pop(req_id, None)
            return

        if runtime.final_output:
            await self.output_async_queue.put(
                {
                    "type": "output",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "engine_outputs": output,
                    "metrics": stage_metrics,
                    "finished": finished and stage_id == req_state.final_stage_id,
                    "stage_submit_ts": submit_ts,
                }
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                {
                    "type": "stage_metrics",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "metrics": stage_metrics,
                    "stage_submit_ts": submit_ts,
                }
            )

        if (
            finished
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and not req_state.next_stage_already_submitted(stage_id)
        ):
            if req_id in self._companion_map and not self._all_companions_done(req_id):
                self._deferred_parents[req_id] = {
                    "stage_id": stage_id,
                    "output": output,
                }
            else:
                await self._forward_to_next_stage(req_id, stage_id, output, req_state)

        if finished and stage_id == req_state.final_stage_id:
            self._cleanup_companion_state(req_id)
            self.request_states.pop(req_id, None)

    def _cleanup_companion_state(self, parent_id: str) -> None:
        """Remove all companion tracking state for a completed parent."""
        role_map = self._companion_map.pop(parent_id, {})
        for companion_id in role_map.values():
            self._companion_ids.discard(companion_id)
            self._companion_to_parent.pop(companion_id, None)
        self._companion_done.pop(parent_id, None)
        self._deferred_parents.pop(parent_id, None)

    def _all_companions_done(self, parent_id: str) -> bool:
        """Check whether all CFG companions for a parent request have finished."""
        role_map = self._companion_map.get(parent_id, {})
        if not role_map:
            return True
        done_set = self._companion_done.get(parent_id, set())
        return all(companion_id in done_set for companion_id in role_map.values())

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._companion_to_parent.get(req_id)
        if parent_id is None:
            return
        done_set = self._companion_done.setdefault(parent_id, set())
        if req_id in done_set:
            return
        done_set.add(req_id)
        if parent_id in self._deferred_parents and self._all_companions_done(parent_id):
            deferred = self._deferred_parents.pop(parent_id)
            parent_state = self.request_states.get(parent_id)
            if parent_state is not None and not parent_state.next_stage_already_submitted(deferred["stage_id"]):
                await self._forward_to_next_stage(
                    parent_id,
                    deferred["stage_id"],
                    deferred["output"],
                    parent_state,
                )

    async def _handle_kv_ready_outputs(self, stage_id: int, kv_ready_outputs: list[Any]) -> None:
        """Forward split requests once stage-0 KV is ready, not only when decode fully finishes."""
        if self.async_chunk:
            return
        for raw_output in kv_ready_outputs:
            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if req_id in self._companion_ids:
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if req_state.next_stage_already_submitted(stage_id):
                continue
            if req_id in self._companion_map and not self._all_companions_done(req_id):
                self._deferred_parents[req_id] = {
                    "stage_id": stage_id,
                    "output": raw_output,
                }
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_stage_metrics(
        self,
        stage_id: int,
        req_id: str,
        request_outputs: list[RequestOutput],
        req_state: PipelineRequestState,
    ) -> StageRequestMetrics:
        """Build StageRequestMetrics for a finished request at a stage."""
        now = _time.time()
        submit_ts = req_state.stage_submit_ts.get(stage_id, now)
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if stage_id == self.entry_stage_pos:
            for request_output in request_outputs:
                prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
                if prompt_token_ids is not None:
                    num_tokens_in += len(prompt_token_ids)

        self._batch_seq[stage_id] += 1
        batch_id = self._batch_seq[stage_id]

        self._agg_total_tokens[stage_id] += num_tokens_out
        self._agg_total_gen_time_ms[stage_id] += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=self._agg_total_tokens[stage_id],
                total_gen_time_ms=self._agg_total_gen_time_ms[stage_id],
            ),
        )

    def _build_kv_sender_info(self, sender_stage_ids: list[int]) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= self.num_stages:
                continue

            sender_stage = self.stage_clients[sender_stage_id]
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    async def _forward_to_next_stage(
        self,
        req_id: str,
        stage_id: int,
        output: Any,
        req_state: PipelineRequestState,
    ) -> None:
        """Forward output from current stage to the next stage."""
        next_stage_id = stage_id + 1
        current_runtime = self.stage_runtimes[stage_id]
        next_runtime = self.stage_runtimes[next_stage_id]
        params = req_state.sampling_params_list[next_stage_id]
        current_runtime.set_engine_outputs([output])

        if next_runtime.stage_type == "diffusion":
            if next_runtime.custom_process_input_func is not None:
                diffusion_prompt = next_runtime.custom_process_input_func(
                    self.stage_clients,
                    next_runtime.engine_input_source,
                    req_state.prompt,
                    False,
                )
            else:
                diffusion_prompt = req_state.prompt

            cfg_ids = self._companion_map.get(req_id)
            if cfg_ids:
                from vllm_omni.inputs.data import OmniDiffusionSamplingParams

                if isinstance(params, OmniDiffusionSamplingParams):
                    params = copy.deepcopy(params)
                    params.cfg_kv_request_ids = cfg_ids
                    logger.info(
                        "[Orchestrator] Attaching cfg_kv_request_ids=%s to req %s",
                        cfg_ids,
                        req_id,
                    )

            source_stage_ids = list(next_runtime.engine_input_source or [stage_id])
            kv_sender_info = self._build_kv_sender_info(sender_stage_ids=source_stage_ids)
            await next_runtime.submit(
                request=diffusion_prompt,
                request_id=req_id,
                params=params,
                kv_sender_info=kv_sender_info,
            )
            req_state.mark_stage_submitted(next_stage_id, _time.time())
            return

        try:
            next_inputs = next_runtime.process_engine_inputs(
                stage_list=self.stage_clients,
                prompt=req_state.prompt,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_stage_id,
            )
            raise

        for next_input in next_inputs:
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=next_runtime.stage_vllm_config.model_config,
            )

            request.external_req_id = request.request_id
            next_runtime.register_request(request=request, prompt=None)
            await next_runtime.submit(
                request=request,
                request_id=req_id,
                params=params,
            )

        req_state.mark_stage_submitted(next_stage_id, _time.time())

    async def _emit_request_error(self, request_id: str, error: Exception) -> None:
        stage_id = self.entry_stage_pos if self.entry_stage_pos is not None else 0
        await self.output_async_queue.put(
            {
                "type": "error",
                "request_id": request_id,
                "stage_id": stage_id,
                "error": str(error),
            }
        )

    async def _cleanup_failed_entry_request(self, request_id: str) -> None:
        companion_ids = list(self._companion_map.get(request_id, {}).values())
        all_request_ids = [request_id, *companion_ids]

        for runtime in self.stage_runtimes:
            try:
                await runtime.abort(all_request_ids)
            except Exception:
                logger.exception(
                    "[Orchestrator] abort failed while cleaning up req=%s after entry failure",
                    request_id,
                )

        for failed_request_id in all_request_ids:
            req_state = self.request_states.pop(failed_request_id, None)
            if req_state is not None:
                req_state.cancel()
        self._cleanup_companion_state(request_id)

    async def _handle_entry_request_error(self, request_id: str, error: Exception) -> None:
        logger.exception(
            "[Orchestrator] Entry ingress failed for req=%s at stage=%s",
            request_id,
            self.entry_stage_pos,
        )
        await self._cleanup_failed_entry_request(request_id)
        await self._emit_request_error(request_id, error)

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        if self.entry_runtime is None or self.entry_stage_pos is None:
            raise RuntimeError("PipelineRuntime requires at least one entry stage")

        request_id = msg["request_id"]
        prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", prompt)
        sampling_params_list = msg["sampling_params_list"]
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg["final_stage_id"]

        logger.info(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            self.entry_stage_pos,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        req_state = PipelineRequestState(
            meta=RequestMeta(
                request_id=request_id,
                final_stage_id=final_stage_id,
                sampling_params_list=sampling_params_list,
                prompt_text=msg.get("prompt_text"),
                arrival_time=msg.get("arrival_time"),
                lora_request=msg.get("lora_request"),
                tokenization_kwargs=msg.get("tokenization_kwargs"),
                trace_headers=msg.get("trace_headers"),
                priority=msg.get("priority", 0),
                data_parallel_rank=msg.get("data_parallel_rank"),
                reasoning_ended=msg.get("reasoning_ended"),
                resumable=msg.get("resumable", False),
            ),
            data=PipelineData(
                raw_prompt=original_prompt,
                stage0_request=prompt if self._entry_uses_prebuilt_request else None,
                terminal_outputs={},
            ),
        )
        req_state.mark_stage_submitted(self.entry_stage_pos, _time.time())
        self.request_states[request_id] = req_state

        try:
            stage0_request = await self.entry_runtime.accept_external_request(
                meta=req_state.meta,
                data=req_state.data,
            )
        except Exception as error:
            await self._handle_entry_request_error(request_id, error)
            return

        if self.async_chunk and final_stage_id > self.entry_stage_pos:
            await self._prewarm_async_chunk_stages(request_id, stage0_request, req_state)

    async def _handle_streaming_update(self, msg: dict[str, Any]) -> None:
        """Handle a streaming_update message for an existing request."""
        if self.entry_runtime is None or self.entry_stage_pos is None:
            raise RuntimeError("PipelineRuntime requires at least one entry stage")

        request_id = msg["request_id"]
        request = msg["prompt"]
        original_prompt = msg.get("original_prompt", request)

        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = dict(msg)
            fallback_msg["type"] = "add_request"
            await self._handle_add_request(fallback_msg)
            return

        req_state.data.raw_prompt = original_prompt
        if self._entry_uses_prebuilt_request:
            req_state.data.stage0_request = request
        if "sampling_params_list" in msg and msg["sampling_params_list"]:
            req_state.meta.sampling_params_list = msg["sampling_params_list"]

        req_state.mark_stage_submitted(self.entry_stage_pos, _time.time())
        try:
            await self.entry_runtime.accept_streaming_update(
                meta=req_state.meta,
                data=req_state.data,
            )
        except Exception as error:
            await self._handle_entry_request_error(request_id, error)

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: PipelineRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if self.entry_stage_pos is None or req_state.final_stage_id <= self.entry_stage_pos:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        try:
            next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
        except Exception:
            next_prompt_len = max(1, len(prompt_token_ids))
        original_prompt = req_state.prompt
        if isinstance(original_prompt, dict):
            base_input = copy.deepcopy(original_prompt)
        else:
            base_input = {}
        base_input["prompt_token_ids"] = [0] * next_prompt_len
        base_input["multi_modal_data"] = None
        base_input["mm_processor_kwargs"] = None

        for next_stage_id in range(self.entry_stage_pos + 1, req_state.final_stage_id + 1):
            next_runtime = self.stage_runtimes[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            if next_runtime.stage_type == "diffusion":
                source_stage_ids = list(next_runtime.engine_input_source or [next_stage_id - 1])
                kv_sender_info = self._build_kv_sender_info(sender_stage_ids=source_stage_ids)
                await next_runtime.submit(
                    request=req_state.prompt,
                    request_id=request_id,
                    params=params,
                    kv_sender_info=kv_sender_info,
                )
                req_state.mark_stage_submitted(next_stage_id, _time.time())
                continue

            request = build_engine_core_request_from_tokens(
                request_id=request_id,
                prompt=base_input,
                params=params,
                model_config=next_runtime.stage_vllm_config.model_config,
            )
            request.external_req_id = request.request_id

            next_runtime.register_request(request=request, prompt=None)
            await next_runtime.submit(
                request=request,
                request_id=request_id,
                params=params,
            )
            req_state.mark_stage_submitted(next_stage_id, _time.time())

    async def _handle_add_companion(self, msg: dict[str, Any]) -> None:
        """Handle an add_companion_request message via entry-stage ingress."""
        if self.entry_runtime is None or self.entry_stage_pos is None:
            raise RuntimeError("PipelineRuntime requires at least one entry stage")

        companion_id = msg["companion_id"]
        parent_id = msg["parent_id"]
        role = msg["role"]
        companion_prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", companion_prompt)
        sampling_params_list = msg["sampling_params_list"]

        if parent_id not in self.request_states:
            logger.warning(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent req %s is no longer live",
                companion_id,
                role,
                parent_id,
            )
            return

        if parent_id not in self._companion_map:
            self._companion_map[parent_id] = {}
        self._companion_map[parent_id][role] = companion_id
        self._companion_ids.add(companion_id)
        self._companion_to_parent[companion_id] = parent_id
        self._companion_done.setdefault(parent_id, set())

        companion_state = PipelineRequestState(
            meta=RequestMeta(
                request_id=companion_id,
                final_stage_id=self.entry_stage_pos,
                sampling_params_list=sampling_params_list,
                prompt_text=msg.get("prompt_text"),
                arrival_time=msg.get("arrival_time"),
                lora_request=msg.get("lora_request"),
                tokenization_kwargs=msg.get("tokenization_kwargs"),
                trace_headers=msg.get("trace_headers"),
                priority=msg.get("priority", 0),
                data_parallel_rank=msg.get("data_parallel_rank"),
                reasoning_ended=msg.get("reasoning_ended"),
                resumable=msg.get("resumable", False),
            ),
            data=PipelineData(
                raw_prompt=original_prompt,
                stage0_request=None,
                terminal_outputs={},
            ),
        )
        companion_state.mark_stage_submitted(self.entry_stage_pos, _time.time())
        self.request_states[companion_id] = companion_state

        try:
            await self.entry_runtime.accept_external_request(
                meta=companion_state.meta,
                data=companion_state.data,
            )
        except Exception as error:
            await self._handle_entry_request_error(parent_id, error)
            return

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s)",
            companion_id,
            role,
            parent_id,
        )

    async def _handle_abort(self, msg: dict[str, Any]) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg["request_ids"]
        companion_ids_to_abort: list[str] = []
        for req_id in request_ids:
            role_map = self._companion_map.pop(req_id, {})
            for companion_id in role_map.values():
                companion_ids_to_abort.append(companion_id)
                self._companion_ids.discard(companion_id)
                self._companion_to_parent.pop(companion_id, None)
            self._companion_done.pop(req_id, None)
            self._deferred_parents.pop(req_id, None)

        all_ids_to_abort = list(request_ids) + companion_ids_to_abort
        for runtime in self.stage_runtimes:
            await runtime.abort(all_ids_to_abort)
        for req_id in request_ids:
            req_state = self.request_states.get(req_id)
            if req_state is not None:
                req_state.cancel()
            self.request_states.pop(req_id, None)
        for companion_id in companion_ids_to_abort:
            companion_state = self.request_states.get(companion_id)
            if companion_state is not None:
                companion_state.cancel()
            self.request_states.pop(companion_id, None)
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _handle_collective_rpc(self, msg: dict[str, Any]) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg["rpc_id"]
        method = msg["method"]
        timeout = msg.get("timeout")
        args = tuple(msg.get("args", ()))
        kwargs = dict(msg.get("kwargs") or {})
        requested_stage_ids = msg.get("stage_ids")
        stage_ids = list(range(self.num_stages)) if requested_stage_ids is None else list(requested_stage_ids)

        results: list[Any] = []
        for stage_id in stage_ids:
            if stage_id < 0 or stage_id >= self.num_stages:
                results.append(
                    {
                        "supported": False,
                        "todo": True,
                        "error": f"Invalid stage id {stage_id}",
                    }
                )
                continue

            try:
                stage_result = await self.stage_runtimes[stage_id].collective_rpc(
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
            except Exception as exc:
                logger.exception(
                    "[Orchestrator] collective_rpc failed: stage=%s method=%s",
                    stage_id,
                    method,
                )
                stage_result = {
                    "supported": False,
                    "error": str(exc),
                }

            results.append(stage_result)

        await self.rpc_async_queue.put(
            {
                "type": "collective_rpc_result",
                "rpc_id": rpc_id,
                "method": method,
                "stage_ids": stage_ids,
                "results": results,
            }
        )

    def _shutdown_stages(self) -> None:
        """Shutdown all stage clients."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        logger.info("[Orchestrator] Shutting down all stages")
        for stage_id, runtime in enumerate(self.stage_runtimes):
            try:
                runtime.shutdown()
                logger.info(f"[Orchestrator] Stage {stage_id} shut down")
            except Exception as exc:
                logger.warning(f"[Orchestrator] Failed to shutdown stage {stage_id}: {exc}")
