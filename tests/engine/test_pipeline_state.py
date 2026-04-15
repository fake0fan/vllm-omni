import pytest
from vllm import SamplingParams

from vllm_omni.engine.pipeline_state import PipelineData, PipelineRequestState, RequestMeta


def test_pipeline_request_state_tracks_stage_submission_and_completion() -> None:
    state = PipelineRequestState(
        meta=RequestMeta(
            request_id="req-1",
            final_stage_id=1,
            sampling_params_list=[SamplingParams(max_tokens=4)],
            prompt_text="hello",
            arrival_time=1.0,
            lora_request=None,
            tokenization_kwargs=None,
            trace_headers=None,
            priority=0,
            data_parallel_rank=None,
            reasoning_ended=None,
            resumable=False,
        ),
        data=PipelineData(
            raw_prompt={"prompt": "hello"},
            stage0_request={"prompt_token_ids": [1, 2, 3]},
            terminal_outputs=[],
        ),
    )

    assert state.meta.request_id == "req-1"
    assert state.data.raw_prompt == {"prompt": "hello"}
    assert len(state.meta.sampling_params_list) == 1
    assert state.meta.final_stage_id == 1
    assert state.request_id == "req-1"
    assert state.prompt == {"prompt": "hello"}
    assert state.original_prompt == {"prompt": "hello"}
    assert state.sampling_params_list == [SamplingParams(max_tokens=4)]
    assert state.final_stage_id == 1
    assert state.meta.entry_params == SamplingParams(max_tokens=4)
    assert state.entry_params == SamplingParams(max_tokens=4)
    assert state.active_stage_ids == set()

    state.mark_stage_submitted(stage_id=0, submitted_at=1.25)
    assert state.next_stage_already_submitted(0) is False
    state.mark_stage_submitted(stage_id=1, submitted_at=2.50)

    assert state.stage_submit_ts == {0: 1.25, 1: 2.50}
    assert state.active_stage_ids == {0, 1}
    assert state.next_stage_already_submitted(0) is True

    state.mark_stage_finished(0)
    assert state.active_stage_ids == {1}


def test_pipeline_request_state_can_be_cancelled_without_losing_submit_history() -> None:
    state = PipelineRequestState(
        meta=RequestMeta(
            request_id="req-cancel",
            final_stage_id=0,
            sampling_params_list=[SamplingParams(max_tokens=8)],
            prompt_text="bye",
            arrival_time=2.0,
            lora_request=None,
            tokenization_kwargs=None,
            trace_headers=None,
            priority=0,
            data_parallel_rank=None,
            reasoning_ended=None,
            resumable=False,
        ),
        data=PipelineData(
            raw_prompt={"prompt": "bye"},
            stage0_request={"prompt_token_ids": [9]},
            terminal_outputs=[],
        ),
    )

    state.mark_stage_submitted(stage_id=0, submitted_at=9.0)
    state.cancel()
    state.mark_stage_submitted(stage_id=1, submitted_at=10.0)

    assert state.cancelled is True
    assert state.stage_submit_ts == {0: 9.0, 1: 10.0}
    assert state.active_stage_ids == set()


def test_entry_params_requires_stage_zero_sampling_params() -> None:
    state = PipelineRequestState(
        meta=RequestMeta(
            request_id="req-empty",
            final_stage_id=0,
            sampling_params_list=[],
            prompt_text=None,
            arrival_time=3.0,
            lora_request=None,
            tokenization_kwargs=None,
            trace_headers=None,
            priority=0,
            data_parallel_rank=None,
            reasoning_ended=None,
            resumable=False,
        ),
        data=PipelineData(
            raw_prompt="hello",
            stage0_request=None,
            terminal_outputs=[],
        ),
    )

    with pytest.raises(ValueError, match="stage-0"):
        _ = state.meta.entry_params
