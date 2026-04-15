"""Compatibility wrapper for the extracted pipeline runtime."""

from vllm_omni.engine.pipeline_runtime import (
    PipelineRequestState,
    PipelineRuntime,
    build_engine_core_request_from_tokens,
)

Orchestrator = PipelineRuntime
OrchestratorRequestState = PipelineRequestState

__all__ = [
    "PipelineRuntime",
    "Orchestrator",
    "PipelineRequestState",
    "OrchestratorRequestState",
    "build_engine_core_request_from_tokens",
]
