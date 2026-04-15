from types import SimpleNamespace

import pytest

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
