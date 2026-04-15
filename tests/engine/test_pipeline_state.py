from vllm import SamplingParams

from vllm_omni.engine.pipeline_state import PipelineRequestState


def test_pipeline_request_state_tracks_stage_submission_and_completion() -> None:
    state = PipelineRequestState(
        global_request_id="req-1",
        original_prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4)],
        final_stage_id=1,
    )

    assert state.global_request_id == "req-1"
    assert state.original_prompt == {"prompt": "hello"}
    assert len(state.sampling_params_list) == 1
    assert state.final_stage_id == 1
    assert state.request_id == "req-1"
    assert state.prompt == {"prompt": "hello"}
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
        global_request_id="req-cancel",
        original_prompt={"prompt": "bye"},
        sampling_params_list=[SamplingParams(max_tokens=8)],
        final_stage_id=0,
    )

    state.mark_stage_submitted(stage_id=0, submitted_at=9.0)
    state.cancel()
    state.mark_stage_submitted(stage_id=1, submitted_at=10.0)

    assert state.cancelled is True
    assert state.stage_submit_ts == {0: 9.0, 1: 10.0}
    assert state.active_stage_ids == set()
