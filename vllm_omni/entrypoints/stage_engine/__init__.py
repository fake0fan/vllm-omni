"""Stage engine components for ZMQ-based multi-stage pipeline execution.

This package provides:
- StageClient protocol for unified stage client interface
- LLMStageClientWrapper for wrapping vLLM's EngineCoreClient
- NonLLMStageClient for Diffusion/Audio stages
- create_stage_client factory for creating appropriate clients
- launch_non_llm_stages for starting non-LLM stage worker processes
- StageEngineCoreProc for running stage workers in background processes

For LLM stages, use vLLM's EngineCoreClient.make_async_mp_client() directly.
This package reuses vLLM's EngineZmqAddresses and EngineHandshakeMetadata.
"""

from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
)

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
from vllm_omni.entrypoints.stage_engine.stage_launcher import (
    StageInfo,
    StageProcManager,
    launch_non_llm_stages,
)
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
    # Launcher (for non-LLM stages)
    "StageInfo",
    "StageProcManager",
    "launch_non_llm_stages",
    # vLLM types (re-exported for convenience)
    "EngineZmqAddresses",
    "EngineHandshakeMetadata",
    # Process
    "StageEngineCoreProc",
    # Serialization
    "StageMsgpackEncoder",
    "StageMsgpackDecoder",
]
