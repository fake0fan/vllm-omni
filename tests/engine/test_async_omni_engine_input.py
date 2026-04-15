from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_add_request_message_keeps_raw_prompt_and_metadata(mocker: MockerFixture) -> None:
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    lora_request = object()
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("speech",)
    engine.input_processor = mocker.Mock()
    engine.output_processors = [mocker.Mock()]

    prompt = {
        "prompt": "hello world",
        "additional_information": {
            "speaker": ["vivian"],
        },
    }
    tokenization_kwargs = {"foo": "bar"}
    trace_headers = {"traceparent": "abc123"}

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        prompt_text="explicit prompt text",
        sampling_params_list=[params],
        final_stage_id=2,
        arrival_time=12.5,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        trace_headers=trace_headers,
        priority=7,
        data_parallel_rank=3,
        reasoning_ended=True,
        resumable=False,
    )

    assert msg == {
        "type": "add_request",
        "request_id": "req-1",
        "prompt": prompt,
        "original_prompt": prompt,
        "sampling_params_list": [params],
        "final_stage_id": 2,
        "prompt_text": "explicit prompt text",
        "arrival_time": 12.5,
        "lora_request": lora_request,
        "tokenization_kwargs": tokenization_kwargs,
        "trace_headers": trace_headers,
        "priority": 7,
        "data_parallel_rank": 3,
        "reasoning_ended": True,
        "resumable": False,
    }
    assert "global_request_id" not in prompt["additional_information"]
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()


def test_build_add_request_message_with_resumable_streaming_keeps_raw_prompt(
    mocker: MockerFixture,
) -> None:
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.output_processors = [mocker.Mock()]

    prompt = {"prompt": "partial transcript"}

    msg = engine._build_add_request_message(
        request_id="req-stream",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=4.2,
        resumable=True,
        message_type="streaming_update",
    )

    assert msg["type"] == "streaming_update"
    assert msg["prompt"] is prompt
    assert msg["original_prompt"] is prompt
    assert msg["arrival_time"] == 4.2
    assert msg["resumable"] is True
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()


def test_enqueue_cfg_companions_packs_raw_messages_with_overrides(mocker: MockerFixture) -> None:
    engine = object.__new__(AsyncOmniEngine)
    request_queue = mocker.Mock()
    request_queue.sync_q = mocker.Mock()
    engine.request_queue = request_queue
    engine.input_processor = mocker.Mock()
    engine.output_processors = [mocker.Mock()]

    original_prompt = {"prompt": "parent prompt"}
    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    sampling_params_list = [stage0_params, SamplingParams(max_tokens=16)]
    companion_params = SamplingParams(max_tokens=8, temperature=0.0)
    companion_spl = [companion_params, sampling_params_list[1]]
    expanded_prompt = {"prompt": "negative prompt"}
    expanded = SimpleNamespace(
        request_id_suffix="-neg",
        prompt=expanded_prompt,
        role="negative",
        apply_overrides=mocker.Mock(return_value=(companion_params, companion_spl)),
    )
    engine.prompt_expand_func = mocker.Mock(return_value=[expanded])

    engine._enqueue_cfg_companions(
        parent_id="req-parent",
        original_prompt=original_prompt,
        stage0_params=stage0_params,
        sampling_params_list=sampling_params_list,
    )

    engine.prompt_expand_func.assert_called_once_with(original_prompt, stage0_params)
    expanded.apply_overrides.assert_called_once_with(stage0_params, sampling_params_list)
    request_queue.sync_q.put_nowait.assert_called_once_with(
        {
            "type": "add_companion_request",
            "companion_id": "req-parent-neg",
            "parent_id": "req-parent",
            "role": "negative",
            "prompt": expanded_prompt,
            "original_prompt": expanded_prompt,
            "sampling_params_list": companion_spl,
        }
    )
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()
