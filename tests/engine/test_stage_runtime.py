from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.pipeline_state import PipelineData, RequestMeta
from vllm_omni.engine.stage_runtime import (
    DiffusionStageRuntime,
    LLMStageRuntime,
    StagePollResult,
    build_stage_runtimes,
)


class _FakeStageClient:
    def __init__(self, *, stage_id: int, stage_type: str, final_output: bool = False) -> None:
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = "image" if stage_type == "diffusion" else "text"
        self.engine_input_source = [0]
        self.custom_process_input_func = None
        self.add_request_calls: list[tuple] = []
        self.add_batch_request_calls: list[tuple] = []
        self.abort_calls: list[list[str]] = []
        self.shutdown_calls = 0
        self.raw_outputs = SimpleNamespace(outputs=[], timestamp=0.0, scheduler_stats=None)
        self.diffusion_output = None

    async def add_request_async(self, *args, **kwargs) -> None:
        self.add_request_calls.append((args, kwargs))

    async def add_batch_request_async(self, *args, **kwargs) -> None:
        self.add_batch_request_calls.append((args, kwargs))

    async def get_output_async(self):
        return self.raw_outputs

    def get_diffusion_output_nowait(self):
        return self.diffusion_output

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    async def collective_rpc_async(self, method: str, timeout=None, args=(), kwargs=None):
        return {"method": method, "timeout": timeout, "args": args, "kwargs": kwargs or {}}

    def set_engine_outputs(self, outputs) -> None:
        self.engine_outputs = outputs

    def process_engine_inputs(self, stage_list, prompt=None):
        return [{"prompt_token_ids": [7, 8, 9]}]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeOutputProcessor:
    def __init__(self) -> None:
        self.add_request_calls: list[dict] = []
        self.scheduler_stats = []
        self.result = SimpleNamespace(request_outputs=[], reqs_to_abort=[])

    def add_request(self, **kwargs) -> None:
        self.add_request_calls.append(kwargs)

    def process_outputs(self, outputs, timestamp, _):
        return self.result

    def update_scheduler_stats(self, scheduler_stats) -> None:
        self.scheduler_stats.append(scheduler_stats)


class _FakeInputProcessor:
    def __init__(self, request: EngineCoreRequest) -> None:
        self.request = request
        self.calls: list[dict[str, object]] = []

    def process_inputs(self, **kwargs):
        self.calls.append(kwargs)
        return self.request


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


@pytest.mark.asyncio
async def test_llm_stage_runtime_polls_outputs_and_forwards_abort() -> None:
    stage_client = _FakeStageClient(stage_id=0, stage_type="llm")
    processor = _FakeOutputProcessor()
    request_output = SimpleNamespace(request_id="req-1", finished=True)
    kv_ready_output = SimpleNamespace(request_id="req-1", kv_transfer_params={"kv_ready": True})
    processor.result = SimpleNamespace(
        request_outputs=[request_output],
        reqs_to_abort=["req-abort"],
    )
    stage_client.raw_outputs = SimpleNamespace(
        outputs=[kv_ready_output],
        timestamp=3.0,
        scheduler_stats={"scheduled": 1},
    )

    runtime = LLMStageRuntime(stage_client=stage_client, output_processor=processor, stage_vllm_config=None)
    polled = await runtime.poll_processed_outputs()

    assert isinstance(polled, StagePollResult)
    assert polled.request_outputs == [request_output]
    assert polled.kv_ready_outputs == [kv_ready_output]
    assert stage_client.abort_calls == [["req-abort"]]
    assert processor.scheduler_stats == [{"scheduled": 1}]


@pytest.mark.asyncio
async def test_llm_stage_runtime_updates_scheduler_stats_without_outputs() -> None:
    stage_client = _FakeStageClient(stage_id=0, stage_type="llm")
    processor = _FakeOutputProcessor()
    stage_client.raw_outputs = SimpleNamespace(
        outputs=[],
        timestamp=3.0,
        scheduler_stats={"scheduled": 2},
    )

    runtime = LLMStageRuntime(stage_client=stage_client, output_processor=processor, stage_vllm_config=None)
    polled = await runtime.poll_processed_outputs()

    assert isinstance(polled, StagePollResult)
    assert polled.request_outputs == []
    assert polled.kv_ready_outputs == []
    assert processor.scheduler_stats == [{"scheduled": 2}]


@pytest.mark.asyncio
async def test_llm_stage_runtime_registers_request_before_submit() -> None:
    stage_client = _FakeStageClient(stage_id=1, stage_type="llm")
    processor = _FakeOutputProcessor()
    request = SimpleNamespace(request_id="req-forward")

    runtime = LLMStageRuntime(stage_client=stage_client, output_processor=processor, stage_vllm_config=None)
    runtime.register_request(request=request, prompt=None)
    await runtime.submit(request=request)

    assert processor.add_request_calls[0]["request"] is request
    assert stage_client.add_request_calls[0][0] == (request,)


@pytest.mark.asyncio
async def test_llm_stage_runtime_accept_external_request_prepares_stage0_request() -> None:
    stage_client = _FakeStageClient(stage_id=0, stage_type="llm")
    processor = _FakeOutputProcessor()
    prepared_request = _make_engine_core_request(request_id="base-req")
    input_processor = _FakeInputProcessor(prepared_request)
    runtime = LLMStageRuntime(
        stage_client=stage_client,
        output_processor=processor,
        stage_vllm_config=None,
        input_processor=input_processor,
        supported_tasks=("generate", "speech"),
    )
    meta = RequestMeta(
        request_id="req-entry",
        final_stage_id=2,
        sampling_params_list=[SamplingParams(max_tokens=8), SamplingParams(max_tokens=4)],
        prompt_text="entry prompt",
        arrival_time=1.5,
        lora_request=None,
        tokenization_kwargs={"trim": True},
        trace_headers={"x-trace": "1"},
        priority=3,
        data_parallel_rank=7,
        reasoning_ended=True,
        resumable=False,
    )
    data = PipelineData(
        raw_prompt={
            "prompt": "raw prompt",
            "additional_information": {"speaker": ["vivian"]},
        },
        stage0_request=None,
        terminal_outputs={},
    )

    submitted = await runtime.accept_external_request(meta=meta, data=data)

    assert isinstance(submitted, OmniEngineCoreRequest)
    assert submitted is data.stage0_request
    assert submitted.external_req_id == "req-entry"
    assert submitted.reasoning_ended is True
    assert processor.add_request_calls[0]["request"] is submitted
    assert processor.add_request_calls[0]["prompt"] == "entry prompt"
    assert stage_client.add_request_calls[0][0] == (submitted,)
    assert input_processor.calls[0]["request_id"] == "req-entry"
    assert input_processor.calls[0]["supported_tasks"] == ("generate", "speech")
    assert input_processor.calls[0]["arrival_time"] == 1.5
    assert input_processor.calls[0]["resumable"] is False
    assert data.raw_prompt["additional_information"]["global_request_id"] == ["req-entry"]


@pytest.mark.asyncio
async def test_llm_stage_runtime_accept_streaming_update_uses_stage0_preprocessing() -> None:
    stage_client = _FakeStageClient(stage_id=0, stage_type="llm")
    processor = _FakeOutputProcessor()
    prepared_request = _make_engine_core_request(request_id="base-stream")
    input_processor = _FakeInputProcessor(prepared_request)
    runtime = LLMStageRuntime(
        stage_client=stage_client,
        output_processor=processor,
        stage_vllm_config=None,
        input_processor=input_processor,
        supported_tasks=("generate",),
    )
    meta = RequestMeta(
        request_id="req-stream",
        final_stage_id=1,
        sampling_params_list=[SamplingParams(max_tokens=6)],
        prompt_text=None,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        reasoning_ended=None,
        resumable=True,
    )
    data = PipelineData(
        raw_prompt={"prompt": "streaming prompt"},
        stage0_request=None,
        terminal_outputs={},
    )

    submitted = await runtime.accept_streaming_update(meta=meta, data=data)

    assert submitted is data.stage0_request
    assert isinstance(submitted, OmniEngineCoreRequest)
    assert processor.add_request_calls[0]["prompt"] == "streaming prompt"
    assert stage_client.add_request_calls[0][0] == (submitted,)
    assert input_processor.calls[0]["resumable"] is True
    assert input_processor.calls[0]["supported_tasks"] == ("generate",)


def test_llm_stage_runtime_requires_output_processor() -> None:
    stage_client = _FakeStageClient(stage_id=1, stage_type="llm")

    with pytest.raises(ValueError, match="requires an output_processor"):
        LLMStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)


@pytest.mark.asyncio
async def test_stage_runtime_abort_forwards_request_ids() -> None:
    stage_client = _FakeStageClient(stage_id=1, stage_type="llm")
    runtime = LLMStageRuntime(
        stage_client=stage_client,
        output_processor=_FakeOutputProcessor(),
        stage_vllm_config=None,
    )

    await runtime.abort(["req-abort"])

    assert stage_client.abort_calls == [["req-abort"]]


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_submit_forwards_single_request_contract() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)
    request = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    params = SimpleNamespace(scale=1.5)
    kv_sender_info = {7: {"host": "127.0.0.1", "zmq_port": 5001}}

    await runtime.submit(
        request=request,
        request_id="req-img",
        params=params,
        kv_sender_info=kv_sender_info,
    )

    assert stage_client.add_request_calls[0][0] == ("req-img", request, params)
    assert stage_client.add_request_calls[0][1] == {"kv_sender_info": kv_sender_info}
    assert stage_client.add_batch_request_calls == []


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_accept_external_request_passes_through_raw_prompt() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)
    params = SimpleNamespace(scale=1.5)
    raw_prompt = [SimpleNamespace(prompt_token_ids=[1, 2, 3])]
    meta = RequestMeta(
        request_id="req-diffusion",
        final_stage_id=2,
        sampling_params_list=[params],
        prompt_text=None,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        reasoning_ended=None,
        resumable=False,
    )
    data = PipelineData(raw_prompt=raw_prompt, stage0_request=None, terminal_outputs={})

    submitted = await runtime.accept_external_request(meta=meta, data=data)

    assert submitted is raw_prompt
    assert data.stage0_request is raw_prompt
    assert stage_client.add_batch_request_calls[0][0] == ("req-diffusion", raw_prompt, params)
    assert stage_client.add_request_calls == []


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_accept_streaming_update_passes_through_raw_prompt() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)
    params = SimpleNamespace(scale=2.0)
    raw_prompt = {"prompt": "streaming diffusion"}
    meta = RequestMeta(
        request_id="req-diffusion-stream",
        final_stage_id=2,
        sampling_params_list=[params],
        prompt_text=None,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        reasoning_ended=None,
        resumable=True,
    )
    data = PipelineData(raw_prompt=raw_prompt, stage0_request=None, terminal_outputs={})

    submitted = await runtime.accept_streaming_update(meta=meta, data=data)

    assert submitted is raw_prompt
    assert data.stage0_request is raw_prompt
    assert stage_client.add_request_calls[0][0] == ("req-diffusion-stream", raw_prompt, params)
    assert stage_client.add_batch_request_calls == []


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_submit_forwards_batch_request_contract() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)
    request = [SimpleNamespace(prompt_token_ids=[1, 2, 3]), SimpleNamespace(prompt_token_ids=[4, 5, 6])]
    params = SimpleNamespace(scale=2.0)
    kv_sender_info = {9: {"host": "127.0.0.1", "zmq_port": 5002}}

    await runtime.submit(
        request=request,
        request_id="req-batch",
        params=params,
        kv_sender_info=kv_sender_info,
    )

    assert stage_client.add_batch_request_calls[0][0] == ("req-batch", request, params)
    assert stage_client.add_batch_request_calls[0][1] == {"kv_sender_info": kv_sender_info}
    assert stage_client.add_request_calls == []


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_submit_requires_explicit_request_id_and_params() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)

    with pytest.raises(ValueError, match="request_id is required"):
        await runtime.submit(request=SimpleNamespace(), request_id=None, params=SimpleNamespace())

    with pytest.raises(ValueError, match="params are required"):
        await runtime.submit(request=SimpleNamespace(), request_id="req-img", params=None)


def test_build_stage_runtimes_rejects_unknown_stage_type() -> None:
    unknown_client = _FakeStageClient(stage_id=3, stage_type="mystery")

    with pytest.raises(ValueError, match="Unknown stage_type"):
        build_stage_runtimes(
            stage_clients=[unknown_client],
            output_processors=[_FakeOutputProcessor()],
            stage_vllm_configs=[SimpleNamespace()],
        )


@pytest.mark.asyncio
async def test_diffusion_stage_runtime_polls_nowait_outputs_without_processor() -> None:
    stage_client = _FakeStageClient(stage_id=2, stage_type="diffusion", final_output=True)
    stage_client.diffusion_output = SimpleNamespace(request_id="req-img", finished=True)
    runtime = DiffusionStageRuntime(stage_client=stage_client, output_processor=None, stage_vllm_config=None)

    polled = await runtime.poll_processed_outputs()

    assert polled.request_outputs[0].request_id == "req-img"
    assert polled.kv_ready_outputs == []


def test_build_stage_runtimes_selects_runtime_type() -> None:
    llm_client = _FakeStageClient(stage_id=0, stage_type="llm")
    diffusion_client = _FakeStageClient(stage_id=1, stage_type="diffusion", final_output=True)
    runtimes = build_stage_runtimes(
        stage_clients=[llm_client, diffusion_client],
        output_processors=[_FakeOutputProcessor(), None],
        stage_vllm_configs=[SimpleNamespace(), None],
    )

    assert isinstance(runtimes[0], LLMStageRuntime)
    assert isinstance(runtimes[1], DiffusionStageRuntime)


def test_build_stage_runtimes_wires_entry_input_processor_only_to_entry_runtime() -> None:
    llm_client_0 = _FakeStageClient(stage_id=1, stage_type="llm")
    llm_client_1 = _FakeStageClient(stage_id=2, stage_type="llm")
    input_processor = object()

    runtimes = build_stage_runtimes(
        stage_clients=[llm_client_0, llm_client_1],
        output_processors=[_FakeOutputProcessor(), _FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(), SimpleNamespace()],
        entry_stage_id=2,
        entry_input_processor=input_processor,
        supported_tasks=("generate", "speech"),
    )

    assert isinstance(runtimes[0], LLMStageRuntime)
    assert isinstance(runtimes[1], LLMStageRuntime)
    assert runtimes[0].input_processor is None
    assert runtimes[0].supported_tasks == ("generate",)
    assert runtimes[1].input_processor is input_processor
    assert runtimes[1].supported_tasks == ("generate", "speech")


@pytest.mark.asyncio
async def test_build_stage_runtimes_keeps_diffusion_entry_runtime_constructible() -> None:
    diffusion_client = _FakeStageClient(stage_id=3, stage_type="diffusion", final_output=True)
    input_processor = object()

    runtimes = build_stage_runtimes(
        stage_clients=[diffusion_client],
        output_processors=[None],
        stage_vllm_configs=[None],
        entry_stage_id=3,
        entry_input_processor=input_processor,
        supported_tasks=("generate", "speech"),
    )

    assert isinstance(runtimes[0], DiffusionStageRuntime)
    assert runtimes[0].stage_id == 3

    meta = RequestMeta(
        request_id="req-diffusion-entry",
        final_stage_id=3,
        sampling_params_list=[SimpleNamespace(scale=1.0)],
        prompt_text=None,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority=0,
        data_parallel_rank=None,
        reasoning_ended=None,
        resumable=False,
    )
    data = PipelineData(raw_prompt={"prompt": "diffusion entry"}, stage0_request=None, terminal_outputs={})

    submitted = await runtimes[0].accept_external_request(meta=meta, data=data)
    assert submitted == {"prompt": "diffusion entry"}
