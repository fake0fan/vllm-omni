# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage client factory for creating appropriate clients based on stage type.

This module provides a unified factory function that creates:
- vLLM's EngineCoreClient (wrapped) for LLM stages
- NonLLMStageClient for Diffusion/Audio stages

This allows AsyncOmni to work with different stage types through a common interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine.core_client import EngineCoreClient

from vllm_omni.entrypoints.stage_engine.stage_client_protocol import (
    LLMStageClientWrapper,
    StageClient,
)
from vllm_omni.entrypoints.stage_engine.stage_core_client import NonLLMStageClient

if TYPE_CHECKING:
    from vllm.v1.executor import Executor

logger = init_logger(__name__)

# Type alias for stage clients
StageClientType = Union[LLMStageClientWrapper, NonLLMStageClient]


def create_stage_client(
    stage_id: int,
    stage_type: str,
    vllm_config: VllmConfig | None = None,
    executor_class: type["Executor"] | None = None,
    input_address: str | None = None,
    output_address: str | None = None,
    log_stats: bool = False,
) -> StageClientType:
    """Create a stage client based on stage type.

    For LLM stages, this uses vLLM's EngineCoreClient.make_async_mp_client()
    wrapped in LLMStageClientWrapper to add stage metadata.

    For non-LLM stages (Diffusion, Audio), this uses NonLLMStageClient.

    Args:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("llm", "diffusion", "audio")
        vllm_config: VllmConfig for LLM stages (required for LLM, optional for others)
        executor_class: Executor class for the stage process
        input_address: ZMQ address for sending requests (auto-generated if None)
        output_address: ZMQ address for receiving outputs (auto-generated if None)
        log_stats: Whether to log statistics

    Returns:
        StageClient instance (LLMStageClientWrapper or NonLLMStageClient)

    Raises:
        ValueError: If stage_type is "llm" but vllm_config is None
        ValueError: If stage_type is unknown
    """
    if stage_type == "llm":
        return _create_llm_stage_client(
            stage_id=stage_id,
            vllm_config=vllm_config,
            executor_class=executor_class,
            input_address=input_address,
            output_address=output_address,
            log_stats=log_stats,
        )
    elif stage_type in ("diffusion", "audio"):
        return _create_non_llm_stage_client(
            stage_id=stage_id,
            stage_type=stage_type,
            vllm_config=vllm_config,
            executor_class=executor_class,
            input_address=input_address,
            output_address=output_address,
            log_stats=log_stats,
        )
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def _create_llm_stage_client(
    stage_id: int,
    vllm_config: VllmConfig | None,
    executor_class: type["Executor"] | None,
    input_address: str | None,
    output_address: str | None,
    log_stats: bool,
) -> LLMStageClientWrapper:
    """Create an LLM stage client using vLLM's EngineCoreClient.

    This directly uses vLLM's optimized EngineCoreClient with all its
    features (scheduler, KV cache, etc.) and wraps it to add stage metadata.

    Args:
        stage_id: Unique identifier for this stage
        vllm_config: VllmConfig for the LLM (required)
        executor_class: Executor class for the stage process
        input_address: ZMQ address for sending requests
        output_address: ZMQ address for receiving outputs
        log_stats: Whether to log statistics

    Returns:
        LLMStageClientWrapper wrapping vLLM's AsyncMPClient

    Raises:
        ValueError: If vllm_config is None
    """
    if vllm_config is None:
        raise ValueError("vllm_config is required for LLM stage")

    if executor_class is None:
        from vllm.v1.executor import Executor
        executor_class = Executor.get_class(vllm_config)

    # Build client_addresses if provided
    client_addresses = None
    if input_address is not None or output_address is not None:
        client_addresses = {}
        if input_address is not None:
            client_addresses["input_address"] = input_address
        if output_address is not None:
            client_addresses["output_address"] = output_address

    logger.info(
        f"Creating LLM stage {stage_id} using vLLM's EngineCoreClient.make_async_mp_client()"
    )

    # Use vLLM's factory method directly
    engine_core_client = EngineCoreClient.make_async_mp_client(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=log_stats,
        client_addresses=client_addresses,
    )

    # Wrap with stage metadata
    return LLMStageClientWrapper(
        engine_core_client=engine_core_client,
        stage_id=stage_id,
    )


def _create_non_llm_stage_client(
    stage_id: int,
    stage_type: str,
    vllm_config: VllmConfig | None,
    executor_class: type["Executor"] | None,
    input_address: str | None,
    output_address: str | None,
    log_stats: bool,
) -> NonLLMStageClient:
    """Create a non-LLM stage client (Diffusion or Audio).

    This uses our custom NonLLMStageClient which wraps AsyncOmniDiffusion
    or AsyncOmniAudio engines.

    Args:
        stage_id: Unique identifier for this stage
        stage_type: Type of stage ("diffusion" or "audio")
        vllm_config: Config for the stage (optional)
        executor_class: Executor class for the stage process
        input_address: ZMQ address for sending requests
        output_address: ZMQ address for receiving outputs
        log_stats: Whether to log statistics

    Returns:
        NonLLMStageClient instance
    """
    logger.info(
        f"Creating {stage_type} stage {stage_id} using NonLLMStageClient.make_client()"
    )

    return NonLLMStageClient.make_client(
        stage_id=stage_id,
        stage_type=stage_type,
        vllm_config=vllm_config,
        executor_class=executor_class,
        input_address=input_address,
        output_address=output_address,
        log_stats=log_stats,
    )
