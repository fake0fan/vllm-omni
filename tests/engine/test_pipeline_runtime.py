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
    _accept_prebuilt_llm_entry_request,
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

    new_prebuilt_request = _make_engine_core_request("new-prebuilt")
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


def test_add_request_rejects_raw_prompt_when_entry_runtime_requires_prebuilt_requests() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    submit_calls: list[dict[str, object]] = []
    abort_calls: list[list[str]] = []

    async def _submit(*, request, request_id, params):
        submit_calls.append(
            {
                "request": request,
                "request_id": request_id,
                "params": params,
            }
        )

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    entry_runtime = SimpleNamespace(
        stage_id=2,
        stage_type="llm",
        output_processor=None,
        submit=_submit,
        abort=_abort,
    )
    entry_runtime.accept_external_request = _accept_prebuilt_llm_entry_request.__get__(
        entry_runtime,
        type(entry_runtime),
    )

    runtime.output_async_queue = output_queue
    runtime.entry_runtime = entry_runtime
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 2
    runtime._entry_uses_prebuilt_request = True
    runtime.request_states = {}
    runtime.stage_runtimes = [entry_runtime]
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
                "request_id": "req-raw",
                "prompt": {"prompt": "raw prompt"},
                "original_prompt": {"prompt": "raw prompt"},
                "sampling_params_list": [SamplingParams(max_tokens=4)],
                "final_stage_id": 0,
            },
        )
    )

    assert submit_calls == []
    assert abort_calls == [["req-raw"]]
    assert runtime.request_states == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": "req-raw",
        "stage_id": 0,
        "error": "Prebuilt entry runtime requires EngineCoreRequest prompts, got dict",
    }


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


def test_add_request_cleans_up_parent_when_prebuilt_entry_cfg_companion_is_raw_prompt() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    submit_calls: list[dict[str, object]] = []
    abort_calls: list[list[str]] = []

    async def _submit(*, request, request_id, params):
        submit_calls.append(
            {
                "request": request,
                "request_id": request_id,
                "params": params,
            }
        )

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    entry_runtime = SimpleNamespace(
        stage_id=2,
        stage_type="llm",
        output_processor=None,
        submit=_submit,
        abort=_abort,
    )
    entry_runtime.accept_external_request = _accept_prebuilt_llm_entry_request.__get__(
        entry_runtime,
        type(entry_runtime),
    )

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)
    parent_request = _make_engine_core_request("req-parent")
    expansion = SimpleNamespace(
        request_id_suffix="-neg",
        prompt={"prompt": "negative prompt"},
        role="negative",
        apply_overrides=lambda entry_params, sampling_params_list: (entry_params, list(sampling_params_list)),
    )

    runtime.output_async_queue = output_queue
    runtime.entry_runtime = entry_runtime
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 2
    runtime._entry_uses_prebuilt_request = True
    runtime.prompt_expand_func = lambda prompt, entry_params: [expansion]
    runtime.request_states = {}
    runtime.stage_runtimes = [entry_runtime]
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
                "request_id": "req-parent",
                "prompt": parent_request,
                "original_prompt": {"prompt": "parent prompt"},
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert [call["request_id"] for call in submit_calls] == ["req-parent"]
    assert abort_calls == [["req-parent", "req-parent-neg"]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": "req-parent",
        "stage_id": 0,
        "error": "Prebuilt entry runtime requires EngineCoreRequest prompts, got dict",
    }


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


def test_add_request_expands_cfg_companions_via_background_entry_runtime() -> None:
    runtime = object.__new__(PipelineRuntime)
    observed_calls: list[tuple[str, object, object]] = []

    async def _accept_external_request(*, meta, data):
        observed_calls.append((meta.request_id, meta, data))
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)
    companion_params = SamplingParams(max_tokens=8, temperature=0.0)
    companion_spl = [companion_params, downstream_params]
    expansion = SimpleNamespace(
        request_id_suffix="-neg",
        prompt={"prompt": "negative prompt"},
        role="negative",
        apply_overrides=lambda entry_params, sampling_params_list: (companion_params, companion_spl),
    )

    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime._entry_uses_prebuilt_request = False
    runtime.prompt_expand_func = lambda prompt, entry_params: [expansion]
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
                "request_id": "req-parent",
                "prompt": {"prompt": "parent prompt"},
                "original_prompt": {"prompt": "parent prompt"},
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert [request_id for request_id, _, _ in observed_calls] == ["req-parent", "req-parent-neg"]
    parent_state = runtime.request_states["req-parent"]
    companion_state = runtime.request_states["req-parent-neg"]
    assert parent_state.data.stage0_request == {"request_id": "req-parent"}
    assert companion_state.data.stage0_request == {"request_id": "req-parent-neg"}
    assert companion_state.meta.final_stage_id == runtime.entry_stage_pos
    assert companion_state.meta.sampling_params_list == companion_spl
    assert companion_state.data.raw_prompt == {"prompt": "negative prompt"}
    assert runtime._companion_map == {"req-parent": {"negative": "req-parent-neg"}}
    assert runtime._companion_ids == {"req-parent-neg"}
    assert runtime._companion_to_parent == {"req-parent-neg": "req-parent"}
    assert runtime._companion_done == {"req-parent": set()}


def test_add_request_stops_expanding_cfg_companions_after_cleanup() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    abort_calls: list[list[str]] = []
    observed_request_ids: list[str] = []

    async def _accept_external_request(*, meta, data):
        observed_request_ids.append(meta.request_id)
        if meta.request_id == "req-parent-neg-2":
            raise ValueError("bad companion entry request")
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)

    def _make_expansion(suffix: str):
        companion_params = SamplingParams(max_tokens=8, temperature=0.0)
        companion_spl = [companion_params, downstream_params]
        return SimpleNamespace(
            request_id_suffix=suffix,
            prompt={"prompt": f"negative prompt {suffix}"},
            role=f"negative{suffix}",
            apply_overrides=lambda entry_params, sampling_params_list: (companion_params, companion_spl),
        )

    runtime.output_async_queue = output_queue
    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime._entry_uses_prebuilt_request = False
    runtime.prompt_expand_func = lambda prompt, entry_params: [
        _make_expansion("-neg-1"),
        _make_expansion("-neg-2"),
        _make_expansion("-neg-3"),
    ]
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
                "request_id": "req-parent",
                "prompt": {"prompt": "parent prompt"},
                "original_prompt": {"prompt": "parent prompt"},
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert observed_request_ids == ["req-parent", "req-parent-neg-1", "req-parent-neg-2"]
    assert abort_calls == [["req-parent", "req-parent-neg-1", "req-parent-neg-2"]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": "req-parent",
        "stage_id": 0,
        "error": "bad companion entry request",
    }


def test_add_request_ignores_cfg_expansion_hook_failure_and_keeps_parent_live() -> None:
    runtime = object.__new__(PipelineRuntime)
    observed_request_ids: list[str] = []

    async def _accept_external_request(*, meta, data):
        observed_request_ids.append(meta.request_id)
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)

    def _raise_on_expand(prompt, entry_params):
        raise RuntimeError("bad expansion hook")

    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime._entry_uses_prebuilt_request = False
    runtime.prompt_expand_func = _raise_on_expand
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
                "request_id": "req-parent",
                "prompt": {"prompt": "parent prompt"},
                "original_prompt": {"prompt": "parent prompt"},
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert observed_request_ids == ["req-parent"]
    parent_state = runtime.request_states["req-parent"]
    assert parent_state.data.stage0_request == {"request_id": "req-parent"}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert runtime._deferred_parents == {}


def test_add_request_cfg_expansion_hook_sees_pre_ingress_prompt_snapshot() -> None:
    runtime = object.__new__(PipelineRuntime)
    seen_prompt = {}

    async def _accept_external_request(*, meta, data):
        data.raw_prompt.setdefault("additional_information", {})["global_request_id"] = [meta.request_id]
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    def _expand(prompt, entry_params):
        seen_prompt["prompt"] = prompt
        return []

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)
    original_prompt = {"prompt": "parent prompt"}

    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime._entry_uses_prebuilt_request = False
    runtime.prompt_expand_func = _expand
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
                "request_id": "req-parent",
                "prompt": {"prompt": "parent prompt"},
                "original_prompt": original_prompt,
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert seen_prompt["prompt"] == {"prompt": "parent prompt"}
    assert "additional_information" not in seen_prompt["prompt"]
    assert runtime.request_states["req-parent"].data.raw_prompt["additional_information"]["global_request_id"] == [
        "req-parent"
    ]


def test_add_request_cleans_up_parent_when_cfg_expansion_object_fails() -> None:
    runtime = object.__new__(PipelineRuntime)
    output_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    abort_calls: list[list[str]] = []

    async def _accept_external_request(*, meta, data):
        data.stage0_request = {"request_id": meta.request_id}
        return data.stage0_request

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    stage0_params = SamplingParams(max_tokens=8, temperature=0.9)
    downstream_params = SamplingParams(max_tokens=16)

    def _bad_apply_overrides(entry_params, sampling_params_list):
        raise RuntimeError("bad companion overrides")

    expansion = SimpleNamespace(
        request_id_suffix="-neg",
        prompt={"prompt": "negative prompt"},
        role="negative",
        apply_overrides=_bad_apply_overrides,
    )

    runtime.output_async_queue = output_queue
    runtime.entry_runtime = SimpleNamespace(accept_external_request=_accept_external_request)
    runtime.entry_stage_pos = 0
    runtime.entry_stage_id = 5
    runtime._entry_uses_prebuilt_request = False
    runtime.prompt_expand_func = lambda prompt, entry_params: [expansion]
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
                "request_id": "req-parent",
                "prompt": {"prompt": "parent prompt"},
                "original_prompt": {"prompt": "parent prompt"},
                "sampling_params_list": [stage0_params, downstream_params],
                "final_stage_id": 1,
            },
        )
    )

    assert abort_calls == [["req-parent"]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert runtime._deferred_parents == {}
    assert output_queue.get_nowait() == {
        "type": "error",
        "request_id": "req-parent",
        "stage_id": 0,
        "error": "bad companion overrides",
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


def test_abort_parent_cleans_up_parent_and_companions() -> None:
    runtime = object.__new__(PipelineRuntime)
    abort_calls: list[list[str]] = []

    async def _abort(request_ids):
        abort_calls.append(list(request_ids))

    parent_id = "req-parent"
    companion_ids = ["req-parent-pos", "req-parent-neg"]
    runtime.stage_runtimes = [SimpleNamespace(abort=_abort)]
    runtime.request_states = {
        parent_id: PipelineRequestState(
            meta=RequestMeta(
                request_id=parent_id,
                final_stage_id=1,
                sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=6)],
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
    runtime._companion_done = {parent_id: {companion_ids[0]}}
    runtime._deferred_parents = {parent_id: {"stage_id": 0, "output": object()}}

    asyncio.run(PipelineRuntime._handle_abort(runtime, {"request_ids": [parent_id]}))

    assert abort_calls == [[parent_id, *companion_ids]]
    assert runtime.request_states == {}
    assert runtime._companion_map == {}
    assert runtime._companion_ids == set()
    assert runtime._companion_to_parent == {}
    assert runtime._companion_done == {}
    assert runtime._deferred_parents == {}
