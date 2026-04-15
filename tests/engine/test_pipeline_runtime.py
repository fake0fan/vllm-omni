import asyncio
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

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


def test_add_request_entry_failure_emits_request_scoped_error_and_keeps_runtime_alive() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    abort_calls: list[list[str]] = []

    async def _accept_external_request(*, meta, data):
        raise ValueError("bad entry request")

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    runtime.output_async_queue = output_queue
    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 2
    runtime._entry_uses_prebuilt_request = False
    runtime.request_states = {}
    runtime.stage_runtimes = [SimpleNamespace(abort=_abort)]
    runtime._companion_map = {}
    runtime._companion_ids = set()
    runtime._companion_to_parent = {}
    runtime._companion_done = {}
    runtime._deferred_parents = {}
    runtime.async_chunk = False

    asyncio.run(
        PipelineRuntime._handle_add_request(
            runtime,
            {
                "request_id": "req-fail",
                "prompt": {"prompt": "bad"},
                "original_prompt": {"prompt": "bad"},
                "sampling_params_list": [SamplingParams(max_tokens=4)],
                "final_stage_id": 0,
            },
        )
    )

    assert "req-fail" not in runtime.request_states
    assert abort_calls == [["req-fail"]]
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": "req-fail",
        "stage_id": 0,
        "error": "bad entry request",
    }


def test_add_request_preserves_prebuilt_entry_request_even_with_input_processor_configured() -> None:
    runtime = object.__new__(PipelineRuntime)
    observed = {}

    async def _accept_external_request(*, meta, data):
        observed["meta"] = meta
        observed["data"] = data
        return data.stage0_request

    request = _make_engine_core_request("req-prebuilt")
    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 2
    runtime._entry_uses_prebuilt_request = False
    runtime.request_states = {}
    runtime.stage_runtimes = []
    runtime._companion_map = {}
    runtime._companion_ids = set()
    runtime._companion_to_parent = {}
    runtime._companion_done = {}
    runtime._deferred_parents = {}
    runtime.async_chunk = False

    asyncio.run(
        PipelineRuntime._handle_add_request(
            runtime,
            {
                "request_id": "req-prebuilt",
                "prompt": request,
                "original_prompt": request,
                "sampling_params_list": [SamplingParams(max_tokens=4)],
                "final_stage_id": 0,
            },
        )
    )

    req_state = runtime.request_states["req-prebuilt"]
    assert observed["meta"] is req_state.meta
    assert observed["data"] is req_state.data
    assert observed["data"].stage0_request is request
    assert observed["data"].raw_prompt is request
    assert req_state.data.stage0_request is request
    assert req_state.stage_submit_ts[0] > 0


def test_add_companion_drops_message_when_parent_request_is_already_gone() -> None:
    runtime = object.__new__(PipelineRuntime)
    submit_calls: list[dict[str, object]] = []

    async def _submit(*, request, request_id, params):
        submit_calls.append(
            {
                "request": request,
                "request_id": request_id,
                "params": params,
            }
        )

    runtime.entry_runtime = SimpleNamespace(submit=_submit)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime.request_states = {}
    runtime._companion_map = {}
    runtime._companion_ids = set()
    runtime._companion_to_parent = {}
    runtime._companion_done = {}

    asyncio.run(
        PipelineRuntime._handle_add_companion(
            runtime,
            {
                "companion_id": "req-parent-neg",
                "parent_id": "req-parent",
                "role": "negative",
                "prompt": {"prompt": "companion"},
                "sampling_params_list": [SamplingParams(max_tokens=4)],
            },
        )
    )

    assert submit_calls == []
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}


def test_add_companion_routes_raw_prompt_through_entry_runtime_ingress() -> None:
    runtime = object.__new__(PipelineRuntime)
    observed = {}

    async def _accept_external_request(*, meta, data):
        observed["meta"] = meta
        observed["data"] = data
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime.request_states = {
        "req-parent": PipelineRequestState(
            meta=RequestMeta(
                request_id="req-parent",
                final_stage_id=1,
                sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=6)],
            ),
            data=PipelineData(raw_prompt={"prompt": "parent"}, stage0_request=None, terminal_outputs={}),
        )
    }
    runtime._companion_map = {}
    runtime._companion_ids = set()
    runtime._companion_to_parent = {}
    runtime._companion_done = {}

    prompt = {"prompt": "companion"}
    params = SamplingParams(max_tokens=4)
    asyncio.run(
        PipelineRuntime._handle_add_companion(
            runtime,
            {
                "companion_id": "req-parent-neg",
                "parent_id": "req-parent",
                "role": "negative",
                "prompt": prompt,
                "original_prompt": prompt,
                "sampling_params_list": [params],
                "prompt_text": "negative text",
                "arrival_time": 9.5,
                "priority": 3,
                "resumable": True,
            },
        )
    )

    companion_state = runtime.request_states["req-parent-neg"]
    assert observed["meta"] is companion_state.meta
    assert observed["data"] is companion_state.data
    assert observed["meta"].request_id == "req-parent-neg"
    assert observed["meta"].final_stage_id == 0
    assert observed["meta"].sampling_params_list == [params]
    assert observed["meta"].prompt_text == "negative text"
    assert observed["meta"].arrival_time == 9.5
    assert observed["meta"].priority == 3
    assert observed["meta"].resumable is True
    assert observed["data"].raw_prompt is prompt
    assert companion_state.data.stage0_request == {"request_id": "req-parent-neg"}
    assert runtime._companion_map == {"req-parent": {"negative": "req-parent-neg"}}
    assert runtime._companion_ids == {"req-parent-neg"}
    assert runtime._companion_to_parent == {"req-parent-neg": "req-parent"}
    assert runtime._companion_done == {"req-parent": set()}
    assert companion_state.stage_submit_ts[0] > 0


def test_add_companion_entry_failure_cleans_up_parent_and_companion() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    abort_calls: list[list[str]] = []

    async def _accept_external_request(*, meta, data):
        raise ValueError("bad companion entry request")

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    parent_id = "req-parent"
    companion_id = "req-parent-neg"
    runtime.output_async_queue = output_queue
    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime.request_states = {
        parent_id: PipelineRequestState(
            meta=RequestMeta(
                request_id=parent_id,
                final_stage_id=1,
                sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=6)],
            ),
            data=PipelineData(raw_prompt={"prompt": "parent"}, stage0_request=None, terminal_outputs={}),
        )
    }
    runtime.stage_runtimes = [SimpleNamespace(abort=_abort)]
    runtime._companion_map = {}
    runtime._companion_ids = set()
    runtime._companion_to_parent = {}
    runtime._companion_done = {}
    runtime._deferred_parents = {}

    asyncio.run(
        PipelineRuntime._handle_add_companion(
            runtime,
            {
                "companion_id": companion_id,
                "parent_id": parent_id,
                "role": "negative",
                "prompt": {"prompt": "companion"},
                "original_prompt": {"prompt": "companion"},
                "sampling_params_list": [SamplingParams(max_tokens=4)],
            },
        )
    )

    assert abort_calls == [[parent_id, companion_id]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert runtime._deferred_parents == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": parent_id,
        "stage_id": 0,
        "error": "bad companion entry request",
    }


def test_streaming_update_entry_failure_aborts_parent_and_live_companions() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    abort_calls: list[list[str]] = []

    async def _accept_streaming_update(*, meta, data):
        raise ValueError("bad streaming update")

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    parent_id = "req-parent"
    companion_ids = ["req-parent-pos", "req-parent-neg"]
    runtime.output_async_queue = output_queue
    runtime.entry_runtime = SimpleNamespace(accept_streaming_update=_accept_streaming_update)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 2
    runtime._entry_uses_prebuilt_request = False
    runtime.stage_runtimes = [SimpleNamespace(abort=_abort)]
    runtime.request_states = {
        parent_id: PipelineRequestState(
            meta=RequestMeta(
                request_id=parent_id,
                final_stage_id=0,
                sampling_params_list=[SamplingParams(max_tokens=4)],
            ),
            data=PipelineData(raw_prompt={"prompt": "parent"}, stage0_request=None, terminal_outputs={}),
        ),
        companion_ids[0]: PipelineRequestState(
            meta=RequestMeta(
                request_id=companion_ids[0],
                final_stage_id=0,
                sampling_params_list=[SamplingParams(max_tokens=4)],
            ),
            data=PipelineData(raw_prompt={"prompt": "pos"}, stage0_request=None, terminal_outputs={}),
        ),
        companion_ids[1]: PipelineRequestState(
            meta=RequestMeta(
                request_id=companion_ids[1],
                final_stage_id=0,
                sampling_params_list=[SamplingParams(max_tokens=4)],
            ),
            data=PipelineData(raw_prompt={"prompt": "neg"}, stage0_request=None, terminal_outputs={}),
        ),
    }
    runtime._companion_map = {
        parent_id: {
            "positive": companion_ids[0],
            "negative": companion_ids[1],
        }
    }
    runtime._companion_ids = set(companion_ids)
    runtime._companion_to_parent = {
        companion_ids[0]: parent_id,
        companion_ids[1]: parent_id,
    }
    runtime._companion_done = {parent_id: set()}
    runtime._deferred_parents = {parent_id: {"stage_id": 0, "output": object()}}

    asyncio.run(
        PipelineRuntime._handle_streaming_update(
            runtime,
            {
                "request_id": parent_id,
                "prompt": {"prompt": "updated parent"},
                "original_prompt": {"prompt": "updated parent"},
                "sampling_params_list": [SamplingParams(max_tokens=8)],
            },
        )
    )

    assert abort_calls == [[parent_id, *companion_ids]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert runtime._deferred_parents == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": parent_id,
        "stage_id": 0,
        "error": "bad streaming update",
    }
