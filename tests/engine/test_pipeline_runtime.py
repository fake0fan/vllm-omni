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


def test_orchestrator_module_re_exports_pipeline_runtime_symbols() -> None:
    assert Orchestrator is PipelineRuntime
    assert OrchestratorRequestState is PipelineRequestState
    assert orchestrator_build_engine_core_request_from_tokens is build_engine_core_request_from_tokens
