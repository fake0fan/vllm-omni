import asyncio
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.pipeline_state import PipelineData, RequestMeta
from vllm_omni.engine.stage_runtime import build_stage_runtimes
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummySenderStage:
    stage_type = "llm"
    stage_id = 0
    final_output = False

    def __init__(self, sender_info):
        self._sender_info = sender_info
        self.engine_outputs = None
        self.calls = []

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs

    async def submit(self, *, request, request_id, params, kv_sender_info=None):
        self.calls.append(
            {
                "request": request,
                "request_id": request_id,
                "params": params,
                "kv_sender_info": kv_sender_info,
            }
        )

    def get_kv_sender_info(self):
        return self._sender_info


class _DummyDiffusionStage:
    stage_type = "diffusion"
    custom_process_input_func = None
    final_output = True

    def __init__(self, *, stage_id=1, engine_input_source=None, custom_process_input_func=None):
        self.stage_id = stage_id
        self.engine_input_source = engine_input_source or [0]
        self.custom_process_input_func = custom_process_input_func
        self.calls = []

    async def add_request_async(self, request_id, prompt, sampling_params, kv_sender_info=None):
        self.calls.append(
            {
                "method": "single",
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params": sampling_params,
                "kv_sender_info": kv_sender_info,
            }
        )

    async def add_batch_request_async(self, request_id, prompt, sampling_params, kv_sender_info=None):
        self.calls.append(
            {
                "method": "batch",
                "request_id": request_id,
                "prompt": prompt,
                "sampling_params": sampling_params,
                "kv_sender_info": kv_sender_info,
            }
        )


class _DummySubmitStage:
    def __init__(self):
        self.calls = []
        self.engine_outputs = None

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs

    async def submit(self, *, request, request_id, params, kv_sender_info=None):
        self.calls.append(
            {
                "request": request,
                "request_id": request_id,
                "params": params,
                "kv_sender_info": kv_sender_info,
            }
        )


def _build_request_state(*, request_id: str, prompt: object, sampling_params_list: list[object], final_stage_id: int):
    return OrchestratorRequestState(
        meta=RequestMeta(
            request_id=request_id,
            final_stage_id=final_stage_id,
            sampling_params_list=sampling_params_list,
            prompt_text=prompt.get("prompt") if isinstance(prompt, dict) else None,
            arrival_time=0.0,
            lora_request=None,
            tokenization_kwargs=None,
            trace_headers=None,
            priority=0,
            data_parallel_rank=None,
            reasoning_ended=None,
            resumable=False,
        ),
        data=PipelineData(
            raw_prompt=prompt,
            stage0_request=None,
            terminal_outputs={},
        ),
    )


def test_streaming_update_mutates_meta_sampling_params_directly():
    orchestrator = object.__new__(Orchestrator)
    stage = _DummySubmitStage()
    orchestrator.stage_runtimes = [stage]
    orchestrator.request_states = {}

    initial_params = SamplingParams(max_tokens=4)
    updated_params = SamplingParams(max_tokens=8)
    req_state = _build_request_state(
        request_id="req-stream",
        prompt={"prompt": "hello"},
        sampling_params_list=[initial_params],
        final_stage_id=0,
    )
    orchestrator.request_states["req-stream"] = req_state

    asyncio.run(
        Orchestrator._handle_streaming_update(
            orchestrator,
            {
                "request_id": "req-stream",
                "prompt": {"prompt": "updated"},
                "sampling_params_list": [updated_params],
            },
        )
    )

    assert req_state.meta.sampling_params_list == [updated_params]
    assert stage.calls == [
        {
            "request": {"prompt": "updated"},
            "request_id": "req-stream",
            "params": updated_params,
            "kv_sender_info": None,
        }
    ]


def test_streaming_update_refreshes_prompt_used_by_downstream_forwarding():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(stage_id=1, engine_input_source=[0])

    orchestrator.num_stages = 2
    orchestrator.stage_runtimes = build_stage_runtimes(
        stage_clients=[sender_stage, diffusion_stage],
        output_processors=[SimpleNamespace(add_request=lambda **_kwargs: None), None],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), None],
    )
    orchestrator.stage_clients = [runtime.stage_client for runtime in orchestrator.stage_runtimes]
    orchestrator._companion_map = {}
    orchestrator.stage_vllm_configs = [None, None]
    orchestrator.output_processors = [None, None]

    initial_prompt = {"prompt": "original"}
    updated_prompt = {"prompt": "updated"}
    req_state = _build_request_state(
        request_id="req-forward",
        prompt=initial_prompt,
        sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=8)],
        final_stage_id=1,
    )
    orchestrator.request_states["req-forward"] = req_state

    asyncio.run(
        Orchestrator._handle_streaming_update(
            orchestrator,
            {
                "request_id": "req-forward",
                "prompt": updated_prompt,
                "sampling_params_list": [SamplingParams(max_tokens=4), SamplingParams(max_tokens=8)],
            },
        )
    )

    output = SimpleNamespace(request_id="req-forward", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-forward", 0, output, req_state))

    assert req_state.data.raw_prompt == updated_prompt
    assert req_state.data.stage0_request == updated_prompt
    assert diffusion_stage.calls[0]["prompt"] == updated_prompt


def test_stage_engine_core_client_builds_kv_sender_info_from_tcp_address():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._omni_kv_config = None
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "10.20.30.40",
        "zmq_port": 50151,
    }


def test_stage_engine_core_client_falls_back_to_detected_ip_for_loopback(monkeypatch):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 1
    client.client_addresses = {"input_address": "tcp://127.0.0.1:1234"}
    client._omni_kv_config = None
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    monkeypatch.setattr(client, "_detect_local_ip", lambda: "192.168.0.12")
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "192.168.0.12",
        "zmq_port": 50152,
    }


def test_stage_engine_core_client_uses_connector_config_for_sender_port():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 3
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._omni_kv_config = {
        "omni_from_stage": "3",
        "connector_config": {
            "type": "MooncakeTransferEngineConnector",
            "role": "sender",
            "host": "10.20.30.99",
            "zmq_port": 51000,
        },
    }
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "10.20.30.99",
        "zmq_port": 51103,
    }


def test_stage_engine_core_client_preserves_explicit_loopback_sender_host():
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 2
    client.client_addresses = {"input_address": "tcp://10.20.30.40:1234"}
    client._kv_sender_info = None
    client._kv_sender_initialized = False
    client._omni_kv_config = {
        "omni_from_stage": "2",
        "connector_config": {
            "type": "MooncakeTransferEngineConnector",
            "role": "sender",
            "host": "127.0.0.1",
            "zmq_port": 51000,
        },
    }
    client._kv_sender_host = client._resolve_contact_host()
    client._initialize_kv_sender_endpoint()

    assert client.get_kv_sender_info() == {
        "host": "127.0.0.1",
        "zmq_port": 51102,
    }


def test_forward_to_diffusion_attaches_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(stage_id=1, engine_input_source=[0])

    orchestrator.num_stages = 2
    orchestrator.stage_runtimes = build_stage_runtimes(
        stage_clients=[sender_stage, diffusion_stage],
        output_processors=[SimpleNamespace(add_request=lambda **_kwargs: None), None],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), None],
    )
    orchestrator.stage_clients = [runtime.stage_client for runtime in orchestrator.stage_runtimes]
    orchestrator._companion_map = {}
    orchestrator.stage_vllm_configs = [None, None]
    orchestrator.output_processors = [None, None]

    params = OmniDiffusionSamplingParams()
    req_state = _build_request_state(
        request_id="req-1",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), params],
        final_stage_id=1,
    )

    output = SimpleNamespace(request_id="req-1", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-1", 0, output, req_state))

    assert sender_stage.engine_outputs == [output]
    assert diffusion_stage.calls[0]["request_id"] == "req-1"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.2", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0


def test_forward_to_diffusion_uses_engine_input_source_for_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    source_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    source_stage.stage_id = 0
    previous_stage = _DummySenderStage({"host": "10.0.0.9", "zmq_port": 59999})
    previous_stage.stage_id = 1
    diffusion_stage = _DummyDiffusionStage(stage_id=2, engine_input_source=[0])

    orchestrator.num_stages = 3
    orchestrator.stage_runtimes = build_stage_runtimes(
        stage_clients=[source_stage, previous_stage, diffusion_stage],
        output_processors=[
            SimpleNamespace(add_request=lambda **_kwargs: None),
            SimpleNamespace(add_request=lambda **_kwargs: None),
            None,
        ],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            None,
        ],
    )
    orchestrator.stage_clients = [runtime.stage_client for runtime in orchestrator.stage_runtimes]
    orchestrator._companion_map = {}
    orchestrator.stage_vllm_configs = [None, None, None]
    orchestrator.output_processors = [None, None, None]

    params = OmniDiffusionSamplingParams()
    req_state = _build_request_state(
        request_id="req-3",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), SamplingParams(max_tokens=4), params],
        final_stage_id=2,
    )

    output = SimpleNamespace(request_id="req-3", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-3", 1, output, req_state))

    assert previous_stage.engine_outputs == [output]
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.2", "zmq_port": 50151},
    }


def test_forward_to_diffusion_preserves_batch_prompt_from_custom_process_input_func():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.2", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(
        stage_id=1,
        engine_input_source=[0],
        custom_process_input_func=lambda *_args: [
            {"prompt": "frame-1"},
            {"prompt": "frame-2"},
        ],
    )

    orchestrator.num_stages = 2
    orchestrator.stage_runtimes = build_stage_runtimes(
        stage_clients=[sender_stage, diffusion_stage],
        output_processors=[SimpleNamespace(add_request=lambda **_kwargs: None), None],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), None],
    )
    assert orchestrator.stage_runtimes[1].custom_process_input_func is not None
    orchestrator.stage_clients = [runtime.stage_client for runtime in orchestrator.stage_runtimes]
    orchestrator._companion_map = {}
    orchestrator.stage_vllm_configs = [None, None]
    orchestrator.output_processors = [None, None]

    params = OmniDiffusionSamplingParams()
    req_state = _build_request_state(
        request_id="req-batch",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), params],
        final_stage_id=1,
    )

    output = SimpleNamespace(request_id="req-batch", finished=True)
    asyncio.run(Orchestrator._forward_to_next_stage(orchestrator, "req-batch", 0, output, req_state))

    assert diffusion_stage.calls[0]["method"] == "batch"
    assert diffusion_stage.calls[0]["prompt"] == [
        {"prompt": "frame-1"},
        {"prompt": "frame-2"},
    ]


def test_prewarm_diffusion_attaches_kv_sender_info():
    orchestrator = object.__new__(Orchestrator)
    sender_stage = _DummySenderStage({"host": "10.0.0.3", "zmq_port": 50151})
    diffusion_stage = _DummyDiffusionStage(stage_id=1, engine_input_source=[0])

    orchestrator.num_stages = 2
    orchestrator.stage_runtimes = build_stage_runtimes(
        stage_clients=[sender_stage, diffusion_stage],
        output_processors=[SimpleNamespace(add_request=lambda **_kwargs: None), None],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)), None],
    )
    orchestrator.stage_clients = [runtime.stage_client for runtime in orchestrator.stage_runtimes]
    orchestrator.output_processors = [runtime.output_processor for runtime in orchestrator.stage_runtimes]
    orchestrator.stage_vllm_configs = [runtime.stage_vllm_config for runtime in orchestrator.stage_runtimes]

    req_state = _build_request_state(
        request_id="req-2",
        prompt={"prompt": "hello"},
        sampling_params_list=[SamplingParams(max_tokens=4), OmniDiffusionSamplingParams()],
        final_stage_id=1,
    )

    stage0_request = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    asyncio.run(Orchestrator._prewarm_async_chunk_stages(orchestrator, "req-2", stage0_request, req_state))

    assert diffusion_stage.calls[0]["request_id"] == "req-2"
    assert diffusion_stage.calls[0]["kv_sender_info"] == {
        0: {"host": "10.0.0.3", "zmq_port": 50151},
    }
    assert req_state.stage_submit_ts[1] > 0
