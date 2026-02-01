"""Stage context for multi-stage pipeline orchestration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.base import OmniConnectorBase
    from vllm_omni.entrypoints.omni_stage import OmniStage
    from vllm_omni.entrypoints.stage_engine.stage_core_client import (
        StageEngineCoreClient,)


@dataclass
class StageContext:
    """Context for a single stage in a multi-stage pipeline.

    Encapsulates all information needed to interact with a stage:
    - Stage configuration (OmniStage)
    - ZMQ client for communication (StageEngineCoreClient)
    - Connectors for cross-stage data transfer
    - Output routing information

    Attributes:
        stage_id: Unique identifier for this stage (0-indexed)
        stage_config: Configuration for this stage (model, inputs, outputs)
        client: ZMQ client for submitting requests and receiving outputs
        connectors: Dictionary mapping (src_stage, dst_stage) to connector instances
        is_final_output: Whether this stage produces final user-visible output
        final_output_type: Type of final output ("text", "image", "audio", "latents")
    """

    stage_id: int
    stage_config: "OmniStage"
    client: "StageEngineCoreClient"
    connectors: dict[tuple[str, str], "OmniConnectorBase"]
    is_final_output: bool
    final_output_type: str | None
