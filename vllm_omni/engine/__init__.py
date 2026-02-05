"""Engine components for vLLM-Omni.

This module provides:
- OmniEngineCoreRequest: Extended request with embeddings support
- OmniEngineCoreOutput: Extended output with multimodal support
- Input/Output processors for multi-stage pipelines
"""

# Import types first (no circular dependency)
from vllm_omni.engine.types import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
    OmniEngineCoreOutput,
    OmniEngineCoreOutputs,
    OmniEngineCoreRequest,
    PromptEmbedsPayload,
)

# Import processors (they import from types.py, not __init__.py)
from vllm_omni.engine.input_processor import (
    NonLLMEngineCoreRequest,
    NonLLMInputProcessor,
    OmniInputProcessor,
    create_input_processor,
)
from vllm_omni.engine.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
)

__all__ = [
    # Request/Response types
    "OmniEngineCoreRequest",
    "OmniEngineCoreOutput",
    "OmniEngineCoreOutputs",
    # Payload types for serialization
    "PromptEmbedsPayload",
    "AdditionalInformationEntry",
    "AdditionalInformationPayload",
    # Input processors
    "OmniInputProcessor",
    "NonLLMInputProcessor",
    "NonLLMEngineCoreRequest",
    "create_input_processor",
    # Output processors
    "MultimodalOutputProcessor",
    "OmniRequestState",
]
