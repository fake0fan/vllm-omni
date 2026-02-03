"""Stage engine components for ZMQ-based multi-stage pipeline execution.

This package provides:
- StageClient protocol for unified stage client interface
- LLMStageClientWrapper for wrapping vLLM's EngineCoreClient
- NonLLMStageClient for Diffusion/Audio stages
- create_stage_client factory for creating appropriate clients
- StageEngineCoreProc for running stage workers in background processes
"""

from vllm_omni.entrypoints.stage_engine.stage_client_factory import (
    StageClientType,
    create_stage_client,
)
from vllm_omni.entrypoints.stage_engine.stage_client_protocol import (
    BaseStageClient,
    LLMStageClientWrapper,
    StageClient,
)
from vllm_omni.entrypoints.stage_engine.stage_core_client import (
    NonLLMStageClient,
    StageEngineCoreClient,  # Backward compatibility alias
)
from vllm_omni.entrypoints.stage_engine.stage_core_proc import StageEngineCoreProc
from vllm_omni.entrypoints.stage_engine.stage_serialization import (
    StageMsgpackDecoder,
    StageMsgpackEncoder,
)

__all__ = [
    # Protocol and base classes
    "StageClient",
    "BaseStageClient",
    # Client implementations
    "LLMStageClientWrapper",
    "NonLLMStageClient",
    "StageEngineCoreClient",  # Backward compatibility alias
    # Factory
    "StageClientType",
    "create_stage_client",
    # Process
    "StageEngineCoreProc",
    # Serialization
    "StageMsgpackEncoder",
    "StageMsgpackDecoder",
]
