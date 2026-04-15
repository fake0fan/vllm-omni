from __future__ import annotations

from typing import Any, Sequence

import torch
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.pipeline_state import PipelineData, RequestMeta
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)


def inject_global_request_id(target: Any, request_id: str) -> None:
    """Inject a global request ID into a prompt payload."""
    if isinstance(target, dict):
        if "additional_information" not in target:
            target["additional_information"] = {}
        if target["additional_information"] is None:
            target["additional_information"] = {}
        if isinstance(target["additional_information"], dict):
            target["additional_information"]["global_request_id"] = [str(request_id)]


def upgrade_to_omni_request(
    request: EngineCoreRequest,
    raw_prompt: Any,
    *,
    log_prefix: str = "stage0_processing",
) -> EngineCoreRequest:
    """Restore omni-only fields omitted by upstream InputProcessor."""
    prompt_embeds = request.prompt_embeds
    additional_information = None

    if isinstance(raw_prompt, dict):
        if prompt_embeds is None:
            raw_prompt_embeds = raw_prompt.get("prompt_embeds")
            if isinstance(raw_prompt_embeds, torch.Tensor):
                prompt_embeds = raw_prompt_embeds
        additional_information = serialize_additional_information(
            raw_prompt.get("additional_information"),
            log_prefix=log_prefix,
        )

    if prompt_embeds is None and additional_information is None:
        return request

    return OmniEngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        data_parallel_rank=request.data_parallel_rank,
        prompt_embeds=prompt_embeds,
        client_index=request.client_index,
        current_wave=request.current_wave,
        priority=request.priority,
        trace_headers=request.trace_headers,
        resumable=request.resumable,
        external_req_id=request.external_req_id,
        reasoning_ended=request.reasoning_ended,
        additional_information=additional_information,
    )


def apply_omni_final_stage_metadata(
    request: EngineCoreRequest,
    final_stage_id: int,
) -> EngineCoreRequest:
    """Attach Omni final-stage metadata to an EngineCoreRequest."""
    merged: dict[str, Any] = {}
    if isinstance(request, OmniEngineCoreRequest) and request.additional_information is not None:
        merged = deserialize_additional_information(request.additional_information)
    merged["omni_final_stage_id"] = final_stage_id
    payload = serialize_additional_information(merged)
    return OmniEngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        data_parallel_rank=request.data_parallel_rank,
        prompt_embeds=request.prompt_embeds,
        client_index=request.client_index,
        current_wave=request.current_wave,
        priority=request.priority,
        trace_headers=request.trace_headers,
        resumable=request.resumable,
        external_req_id=request.external_req_id,
        reasoning_ended=request.reasoning_ended,
        additional_information=payload,
    )


def register_stage0_output(
    output_processor: Any | None,
    *,
    request: Any,
    prompt_text: str | None,
    original_prompt: Any,
) -> None:
    """Register the stage-0 request with the output processor."""
    if output_processor is None:
        return
    output_prompt_text = prompt_text
    if output_prompt_text is None and isinstance(original_prompt, dict):
        output_prompt_text = original_prompt.get("prompt")
    output_processor.add_request(
        request=request,
        prompt=output_prompt_text,
        parent_req=None,
        request_index=0,
        queue=None,
    )


def prepare_stage0_llm_request(
    *,
    meta: RequestMeta,
    data: PipelineData,
    input_processor: InputProcessor,
    supported_tasks: Sequence[str] | None,
) -> EngineCoreRequest:
    """Prepare an entry-stage LLM request from request metadata and raw data."""
    raw_prompt = data.raw_prompt

    if isinstance(raw_prompt, dict):
        inject_global_request_id(raw_prompt, meta.request_id)
    elif isinstance(raw_prompt, list):
        for item in raw_prompt:
            inject_global_request_id(item, meta.request_id)

    request = input_processor.process_inputs(
        request_id=meta.request_id,
        prompt=raw_prompt,
        params=meta.entry_params,
        supported_tasks=tuple(supported_tasks) if supported_tasks is not None else ("generate",),
        arrival_time=meta.arrival_time,
        lora_request=meta.lora_request,
        tokenization_kwargs=meta.tokenization_kwargs,
        trace_headers=meta.trace_headers,
        priority=meta.priority,
        data_parallel_rank=meta.data_parallel_rank,
        resumable=meta.resumable,
    )
    request = upgrade_to_omni_request(request, raw_prompt)

    if meta.reasoning_ended is not None:
        request.reasoning_ended = meta.reasoning_ended

    request.external_req_id = meta.request_id
    return apply_omni_final_stage_metadata(request, meta.final_stage_id)
