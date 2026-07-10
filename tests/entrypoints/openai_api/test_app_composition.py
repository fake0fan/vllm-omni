# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for the composed upstream and Omni API surface."""

from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from pytest_mock import MockerFixture
from starlette.routing import Route, WebSocketRoute
from vllm.config import ProfilerConfig
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RouteKey = tuple[str, str]

OMNI_ROUTE_OWNERS = {
    ("GET", "/health"): "vllm_omni.entrypoints.openai.api_server.health",
    ("GET", "/v1/models"): "vllm_omni.entrypoints.openai.api_server.show_available_models",
    ("POST", "/v1/chat/completions"): "vllm_omni.entrypoints.openai.api_server.create_chat_completion",
}

ALWAYS_REMOVED_ROUTES = {
    ("POST", "/generative_scoring"),
    ("POST", "/is_scaling_elastic_ep"),
    ("POST", "/scale_elastic_ep"),
    ("POST", "/v1/chat/completions/batch"),
}

UPSTREAM_ROUTE_STATE = {
    ("GET", "/load"): "server_load_metrics",
    ("GET", "/ping"): "engine_client",
    ("GET", "/v1/responses/{response_id}"): "openai_serving_responses",
    ("POST", "/detokenize"): "serving_tokenization",
    ("POST", "/inference/v1/generate"): "serving_tokens",
    ("POST", "/invocations"): "serving_tokenization",
    ("POST", "/ping"): "engine_client",
    ("POST", "/tokenize"): "serving_tokenization",
    ("POST", "/v1/chat/completions/derender"): "openai_serving_render",
    ("POST", "/v1/chat/completions/render"): "openai_serving_render",
    ("POST", "/v1/completions"): "openai_serving_completion",
    ("POST", "/v1/completions/derender"): "openai_serving_render",
    ("POST", "/v1/completions/render"): "openai_serving_render",
    ("POST", "/v1/messages"): "anthropic_serving_messages",
    ("POST", "/v1/messages/count_tokens"): "anthropic_serving_messages",
    ("POST", "/v1/responses"): "openai_serving_responses",
    ("POST", "/v1/responses/{response_id}/cancel"): "openai_serving_responses",
}

STATELESS_UPSTREAM_ROUTES = {("GET", "/version")}


class FakeModelConfig:
    architecture = "FakeForCausalLM"

    def get_pooling_task(self, supported_tasks):
        del supported_tasks
        return None


@pytest.fixture
def server_args(monkeypatch: pytest.MonkeyPatch) -> Namespace:
    monkeypatch.delenv("VLLM_SERVER_DEV_MODE", raising=False)
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    parser = FlexibleArgumentParser()
    serve_parser = parser.add_subparsers().add_parser("serve")
    make_arg_parser(serve_parser)
    args = serve_parser.parse_args([])
    args.disable_fastapi_docs = True
    return args


def _route_owners(app: FastAPI) -> dict[RouteKey, list[str]]:
    owners: dict[RouteKey, list[str]] = defaultdict(list)
    for route in app.routes:
        if isinstance(route, Route):
            methods = route.methods or set()
        elif isinstance(route, WebSocketRoute):
            methods = {"WEBSOCKET"}
        else:
            continue

        owner = f"{route.endpoint.__module__}.{route.endpoint.__qualname__}"
        for method in methods:
            owners[(method, route.path)].append(owner)
    return owners


def _assert_app_contract(
    app: FastAPI,
    *,
    required_omni_state: dict[RouteKey, str],
    absent_routes: set[RouteKey],
) -> None:
    owners = _route_owners(app)
    assert {key: value for key, value in owners.items() if len(value) > 1} == {}
    assert ALWAYS_REMOVED_ROUTES.isdisjoint(owners)
    assert absent_routes.isdisjoint(owners)

    for route_key, state_name in required_omni_state.items():
        assert route_key in owners
        if expected_owner := OMNI_ROUTE_OWNERS.get(route_key):
            assert owners[route_key] == [expected_owner]
        assert getattr(app.state, state_name, None) is not None

    unclassified_upstream_routes: dict[RouteKey, list[str]] = {}
    for route_key, route_owners in owners.items():
        if not any(owner.startswith("vllm.") for owner in route_owners):
            continue
        if route_key in STATELESS_UPSTREAM_ROUTES:
            continue
        state_name = UPSTREAM_ROUTE_STATE.get(route_key)
        if state_name is None:
            unclassified_upstream_routes[route_key] = route_owners
        else:
            assert getattr(app.state, state_name, None) is not None
    assert unclassified_upstream_routes == {}

    handler = app.exception_handlers[EngineDeadError]
    assert handler is app.exception_handlers[EngineGenerateError]
    assert handler.__name__ == "omni_engine_error_handler"


def test_build_app_forwards_only_upstream_capabilities(
    server_args: Namespace,
    mocker: MockerFixture,
) -> None:
    upstream_build = mocker.patch.object(api_server, "build_openai_app", return_value=FastAPI())
    model_config = FakeModelConfig()

    api_server._build_omni_app(
        server_args,
        ("generate", "speech"),
        model_config,
        is_pure_diffusion=False,
        enable_profiler=False,
    )

    upstream_build.assert_called_once_with(server_args, ("generate",), model_config)
    assert api_server._get_upstream_supported_tasks(("generate",), is_pure_diffusion=True) == ()


@pytest.mark.parametrize("enable_omni_profiler", [False, True], ids=("upstream", "omni"))
def test_profiler_routes_keep_one_owner(
    server_args: Namespace,
    enable_omni_profiler: bool,
) -> None:
    server_args.profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/tmp")
    app = api_server._build_omni_app(
        server_args,
        ("generate",),
        FakeModelConfig(),
        is_pure_diffusion=False,
        enable_profiler=enable_omni_profiler,
    )
    owners = _route_owners(app)

    for route_key in (("POST", "/start_profile"), ("POST", "/stop_profile")):
        assert len(owners[route_key]) == 1
        expected_prefix = "vllm_omni." if enable_omni_profiler else "vllm."
        assert owners[route_key][0].startswith(expected_prefix)


@pytest.mark.asyncio
async def test_pure_diffusion_app_matches_initialized_state(
    server_args: Namespace,
    mocker: MockerFixture,
) -> None:
    engine = mocker.MagicMock()
    engine.stage_configs = [{"stage_type": "diffusion"}, {"stage_type": "diffusion"}]
    engine.model_config = None
    engine.get_vllm_config = mocker.AsyncMock(return_value=None)

    mocker.patch.object(api_server, "_DiffusionServingModels", return_value=mocker.MagicMock())
    for name in (
        "OmniOpenAIServingAudioGenerate",
        "OmniOpenAIServingChat",
        "OmniOpenAIServingSpeech",
        "OmniOpenAIServingVideo",
        "ServingRealtimeRobotOpenPI",
    ):
        constructor = mocker.patch.object(api_server, name)
        constructor.for_diffusion.return_value = mocker.MagicMock(name=name)
        constructor.create_policy_server.return_value = mocker.MagicMock(name=name)
    mocker.patch.object(api_server, "OmniStreamingVideoOutputHandler", return_value=mocker.MagicMock())

    app = api_server._build_omni_app(
        server_args,
        ("generate",),
        model_config=None,
        is_pure_diffusion=api_server._is_pure_diffusion(engine.stage_configs),
        enable_profiler=False,
    )
    await api_server.omni_init_app_state(engine, app.state, server_args)

    _assert_app_contract(
        app,
        required_omni_state={
            ("GET", "/health"): "engine_client",
            ("GET", "/v1/models"): "openai_serving_models",
            ("POST", "/v1/audio/generate"): "openai_serving_audio_generate",
            ("POST", "/v1/audio/speech"): "openai_serving_speech",
            ("POST", "/v1/chat/completions"): "openai_serving_chat",
            ("POST", "/v1/videos"): "openai_serving_video",
        },
        absent_routes={
            ("POST", "/detokenize"),
            ("POST", "/invocations"),
            ("POST", "/tokenize"),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supported_tasks", "required_omni_state", "absent_routes"),
    [
        (
            ("generate", "speech"),
            {("POST", "/v1/chat/completions"): "openai_serving_chat"},
            set(),
        ),
        (
            ("speech",),
            {("POST", "/v1/audio/speech"): "openai_serving_speech"},
            {("POST", "/invocations"), ("POST", "/v1/chat/completions")},
        ),
    ],
    ids=("llm", "tts-only"),
)
async def test_non_diffusion_app_matches_initialized_state(
    server_args: Namespace,
    mocker: MockerFixture,
    supported_tasks: tuple[str, ...],
    required_omni_state: dict[RouteKey, str],
    absent_routes: set[RouteKey],
) -> None:
    model_config = FakeModelConfig()
    engine = mocker.MagicMock()
    engine.stage_configs = [{"stage_type": "llm"}]
    engine.model_config = model_config
    engine.input_processor = mocker.MagicMock()
    engine.renderer = mocker.MagicMock()
    engine.get_vllm_config = mocker.AsyncMock(return_value=SimpleNamespace(model_config=model_config, lora_config=None))
    engine.get_supported_tasks = mocker.AsyncMock(return_value=supported_tasks)

    models = mocker.MagicMock()
    models.registry = mocker.MagicMock()
    models.init_static_loras = mocker.AsyncMock()
    mocker.patch.object(api_server, "OpenAIServingModels", return_value=models)
    mocker.patch.object(api_server, "load_chat_template", return_value="test-template")
    mocker.patch.object(api_server, "process_lora_modules", return_value=[])
    mocker.patch.object(api_server, "build_forced_aligner_config", return_value=None)

    speech = mocker.MagicMock()
    speech.warmup = mocker.AsyncMock()
    mocker.patch.object(api_server, "OmniOpenAIServingSpeech", return_value=speech)
    for name in (
        "AnthropicServingMessages",
        "OmniOpenAIServingAudioGenerate",
        "OmniOpenAIServingChat",
        "OmniOpenAIServingVideo",
        "OmniStreamingSpeechHandler",
        "OpenAIServingCompletion",
        "OpenAIServingRealtime",
        "OpenAIServingRender",
        "OpenAIServingResponses",
        "ServingTokenization",
        "ServingTokens",
    ):
        mocker.patch.object(api_server, name, return_value=mocker.MagicMock(name=name))
    mocker.patch.object(api_server, "create_streaming_video_handler", return_value=mocker.MagicMock())

    app = api_server._build_omni_app(
        server_args,
        supported_tasks,
        model_config,
        is_pure_diffusion=False,
        enable_profiler=False,
    )
    await api_server.omni_init_app_state(engine, app.state, server_args)

    _assert_app_contract(
        app,
        required_omni_state={
            ("GET", "/health"): "engine_client",
            ("GET", "/v1/models"): "openai_serving_models",
            **required_omni_state,
        },
        absent_routes=absent_routes,
    )
