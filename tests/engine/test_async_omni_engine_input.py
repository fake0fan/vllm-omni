from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage0_processing import inject_global_request_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_core_request(request_id: str = "req-1") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 1, 1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


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
        "prompt": msg["prompt"],
        "original_prompt": msg["original_prompt"],
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
    assert msg["prompt"] == prompt
    assert msg["prompt"] is not prompt
    assert msg["original_prompt"] is msg["prompt"]
    inject_global_request_id(msg["prompt"], "req-1")
    assert "global_request_id" not in prompt["additional_information"]
    assert msg["prompt"]["additional_information"]["global_request_id"] == ["req-1"]
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
    assert msg["prompt"] == prompt
    assert msg["prompt"] is not prompt
    assert msg["original_prompt"] is msg["prompt"]
    assert msg["arrival_time"] == 4.2
    assert msg["resumable"] is True
    inject_global_request_id(msg["prompt"], "req-stream")
    assert "additional_information" not in prompt
    assert msg["prompt"]["additional_information"]["global_request_id"] == ["req-stream"]
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()


def test_build_add_request_message_copies_list_prompts_for_async_ownership(mocker: MockerFixture) -> None:
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.output_processors = [mocker.Mock()]

    prompt = [{"prompt": "part one"}, {"prompt": "part two", "additional_information": {"speaker": ["a"]}}]

    msg = engine._build_add_request_message(
        request_id="req-list",
        prompt=prompt,
        sampling_params_list=[params],
    )

    assert msg["prompt"] == prompt
    assert msg["prompt"] is not prompt
    assert msg["prompt"][0] is not prompt[0]
    assert msg["prompt"][1] is not prompt[1]
    inject_global_request_id(msg["prompt"][0], "req-list")
    inject_global_request_id(msg["prompt"][1], "req-list")
    assert "additional_information" not in prompt[0]
    assert "global_request_id" not in prompt[1]["additional_information"]
    assert msg["prompt"][1]["additional_information"]["global_request_id"] == ["req-list"]
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()


def test_build_add_request_message_keeps_prebuilt_request_identity(mocker: MockerFixture) -> None:
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("generate",)
    engine.input_processor = mocker.Mock()
    engine.output_processors = [mocker.Mock()]
    request = _make_engine_core_request("req-prebuilt")

    msg = engine._build_add_request_message(
        request_id="req-prebuilt",
        prompt=request,
        sampling_params_list=[params],
        final_stage_id=0,
    )

    assert msg["prompt"] is request
    assert msg["original_prompt"] is request
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
            "prompt": request_queue.sync_q.put_nowait.call_args.args[0]["prompt"],
            "original_prompt": request_queue.sync_q.put_nowait.call_args.args[0]["original_prompt"],
            "sampling_params_list": companion_spl,
        }
    )
    enqueued = request_queue.sync_q.put_nowait.call_args.args[0]
    assert enqueued["prompt"] == expanded_prompt
    assert enqueued["prompt"] is not expanded_prompt
    assert enqueued["original_prompt"] is enqueued["prompt"]
    inject_global_request_id(enqueued["prompt"], "req-parent-neg")
    assert "additional_information" not in expanded_prompt
    assert enqueued["prompt"]["additional_information"]["global_request_id"] == ["req-parent-neg"]
    engine.input_processor.process_inputs.assert_not_called()
    engine.output_processors[0].add_request.assert_not_called()
