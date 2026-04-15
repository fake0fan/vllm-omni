import asyncio
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    build_engine_core_request_from_tokens as orchestrator_build_engine_core_request_from_tokens,
)
from vllm_omni.engine.pipeline_runtime import (
    PipelineRequestState,
    PipelineRuntime,
    build_engine_core_request_from_tokens,
)
from vllm_omni.engine.pipeline_state import PipelineData, RequestMeta


def test_orchestrator_module_re_exports_pipeline_runtime_symbols() -> None:
    assert Orchestrator is PipelineRuntime
    assert OrchestratorRequestState is PipelineRequestState
    assert orchestrator_build_engine_core_request_from_tokens is build_engine_core_request_from_tokens


def test_streaming_update_preserves_original_prompt_for_prebuilt_entry_requests() -> None:
    runtime = object.__new__(PipelineRuntime)
    observed = {}

    async def _accept_streaming_update(*, meta, data):
        observed["meta"] = meta
        observed["data"] = data

    runtime.entry_runtime = SimpleNamespace(accept_streaming_update=_accept_streaming_update)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 3
    runtime._entry_uses_prebuilt_request = True
    runtime.request_states = {
        "req-prebuilt": PipelineRequestState(
            meta=RequestMeta(
                request_id="req-prebuilt",
                final_stage_id=0,
                sampling_params_list=[SamplingParams(max_tokens=4)],
            ),
            data=PipelineData(
                raw_prompt={"prompt": "old raw"},
                stage0_request=SimpleNamespace(request_id="old-prebuilt"),
                terminal_outputs={},
            ),
        )
    }

    new_prebuilt_request = SimpleNamespace(request_id="new-prebuilt")
    new_raw_prompt = {"prompt": "new raw"}

    asyncio.run(
        PipelineRuntime._handle_streaming_update(
            runtime,
            {
                "request_id": "req-prebuilt",
                "prompt": new_prebuilt_request,
                "original_prompt": new_raw_prompt,
                "sampling_params_list": [SamplingParams(max_tokens=8)],
            },
        )
    )

    req_state = runtime.request_states["req-prebuilt"]
    assert req_state.meta.sampling_params_list[0].max_tokens == 8
    assert req_state.data.raw_prompt == new_raw_prompt
    assert req_state.data.stage0_request is new_prebuilt_request
    assert req_state.stage_submit_ts[0] > 0
    assert observed["meta"] is req_state.meta
    assert observed["data"] is req_state.data
