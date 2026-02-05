"""Pipeline orchestrator for multi-stage vLLM-Omni pipelines.

This module provides PipelineOrchestrator, which manages request routing
across pipeline stages, handles stage-to-stage data transfer via connectors,
and collects final outputs.

Extracted from AsyncOmni.generate() and _run_output_handler().
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.stage_context import StageContext
from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates request routing across multi-stage pipelines.

    Manages:
    - Request submission to stage 0
    - Sequential and async_chunk execution modes
    - Stage-to-stage data transfer via OmniConnectors
    - Per-request state tracking (ClientRequestState)
    - Final output collection and aggregation

    Attributes:
        stages: List of StageContext instances for each stage
        execution_mode: "sequential" or "async_chunk"
        request_states: Dictionary mapping request_id to ClientRequestState
    """

    def __init__(
        self,
        stages: list[StageContext],
        execution_mode: str = "sequential",
    ):
        """Initialize the pipeline orchestrator.

        Args:
            stages: List of StageContext instances
            execution_mode: "sequential" or "async_chunk"
        """
        self.stages = stages
        self.execution_mode = execution_mode
        self.request_states: dict[str, ClientRequestState] = {}

        # Validate stages
        if not stages:
            raise ValueError("At least one stage is required")

        logger.info(
            f"PipelineOrchestrator initialized with {len(stages)} stages, "
            f"execution_mode={execution_mode}"
        )

    async def submit_request(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
    ) -> None:
        """Submit a request to stage 0.

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType for stage 0
            sampling_params_list: List of sampling params for each stage
        """
        # Create request state
        req_state = ClientRequestState(request_id=request_id)
        self.request_states[request_id] = req_state

        # Submit to stage 0
        stage_0 = self.stages[0]
        await stage_0.client.submit_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params_list[0],
        )

        logger.debug(f"Submitted request {request_id} to stage 0")

    async def process_pipeline(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorMetrics,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process a request through the pipeline.

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType for stage 0
            sampling_params_list: List of sampling params for each stage
            metrics: OrchestratorMetrics for tracking
            final_stage_id: ID of the final stage for E2E tracking

        Yields:
            OmniRequestOutput instances from final output stages
        """
        if self.execution_mode == "sequential":
            async for output in self._process_sequential_pipeline(
                request_id, prompt, sampling_params_list, metrics, final_stage_id
            ):
                yield output
        elif self.execution_mode == "async_chunk":
            async for output in self._process_async_chunk_pipeline(
                request_id, prompt, sampling_params_list, metrics, final_stage_id
            ):
                yield output
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")

    async def _process_sequential_pipeline(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorMetrics,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process pipeline in sequential mode (stage-by-stage).

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType for stage 0
            sampling_params_list: List of sampling params for each stage
            metrics: OrchestratorMetrics for tracking
            final_stage_id: ID of the final stage for E2E tracking

        Yields:
            OmniRequestOutput instances from final output stages
        """
        req_state = self.request_states[request_id]
        wall_start_ts = time.time()
        req_start_ts = {request_id: wall_start_ts}

        # Process each stage sequentially
        for stage_id, stage_ctx in enumerate(self.stages[: final_stage_id + 1]):
            finished = False
            engine_outputs = None

            # Collect outputs from this stage
            async for output in stage_ctx.client.get_outputs_async():
                if output.request_id != request_id:
                    continue

                # Process output
                finished = output.finished
                engine_outputs = output

                # Mark last output time
                metrics.stage_last_ts[stage_id] = max(
                    metrics.stage_last_ts[stage_id] or 0.0, time.time()
                )

                # Yield if this is a final output stage
                if stage_ctx.is_final_output:
                    yield self._create_final_output(
                        stage_id, stage_ctx, output, metrics, req_start_ts, wall_start_ts, final_stage_id
                    )

                if finished:
                    break

            # Forward to next stage if there is one
            if finished and stage_id < final_stage_id:
                await self._forward_to_next_stage(
                    stage_id,
                    stage_id + 1,
                    request_id,
                    prompt,
                    sampling_params_list,
                    engine_outputs,
                    metrics,
                )

    async def _process_async_chunk_pipeline(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorMetrics,
        final_stage_id: int,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process pipeline in async_chunk mode (streaming across stages).

        Args:
            request_id: Unique request identifier
            prompt: OmniPromptType for stage 0
            sampling_params_list: List of sampling params for each stage
            metrics: OrchestratorMetrics for tracking
            final_stage_id: ID of the final stage for E2E tracking

        Yields:
            OmniRequestOutput instances from final output stages
        """
        wall_start_ts = time.time()
        req_start_ts = {request_id: wall_start_ts}

        # Track which stages have finished
        all_stages_finished = {i: False for i in range(len(self.stages))}

        # Create tasks for collecting outputs from all stages
        stage_tasks = []
        for stage_id, stage_ctx in enumerate(self.stages[: final_stage_id + 1]):
            task = asyncio.create_task(
                self._collect_stage_outputs(
                    stage_id,
                    stage_ctx,
                    request_id,
                    prompt,
                    sampling_params_list,
                    metrics,
                    req_start_ts,
                    wall_start_ts,
                    final_stage_id,
                    all_stages_finished,
                )
            )
            stage_tasks.append(task)

        # Collect outputs from all stages concurrently
        async for output in self._merge_stage_outputs(stage_tasks):
            yield output

    async def _collect_stage_outputs(
        self,
        stage_id: int,
        stage_ctx: StageContext,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id: int,
        all_stages_finished: dict[int, bool],
    ) -> AsyncGenerator[OmniRequestOutput | None, None]:
        """Collect outputs from a single stage.

        Args:
            stage_id: Stage identifier
            stage_ctx: StageContext for this stage
            request_id: Request identifier
            prompt: Original prompt
            sampling_params_list: List of sampling params
            metrics: OrchestratorMetrics
            req_start_ts: Request start timestamps
            wall_start_ts: Wall clock start time
            final_stage_id: Final stage ID
            all_stages_finished: Dictionary tracking stage completion

        Yields:
            OmniRequestOutput instances or None
        """
        async for output in stage_ctx.client.get_outputs_async():
            if output.request_id != request_id:
                continue

            finished = output.finished
            engine_outputs = output

            # Mark last output time
            metrics.stage_last_ts[stage_id] = max(
                metrics.stage_last_ts[stage_id] or 0.0, time.time()
            )

            # Yield if this is a final output stage
            if stage_ctx.is_final_output:
                yield self._create_final_output(
                    stage_id, stage_ctx, output, metrics, req_start_ts, wall_start_ts, final_stage_id
                )
            else:
                yield None

            # Forward to next stage if not finished
            if not finished and stage_id < final_stage_id:
                await self._forward_to_next_stage(
                    stage_id,
                    stage_id + 1,
                    request_id,
                    prompt,
                    sampling_params_list,
                    engine_outputs,
                    metrics,
                )

            if finished:
                all_stages_finished[stage_id] = True
                break

    async def _merge_stage_outputs(
        self, stage_tasks: list[asyncio.Task]
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Merge outputs from multiple stage tasks.

        Args:
            stage_tasks: List of asyncio tasks collecting stage outputs

        Yields:
            OmniRequestOutput instances
        """
        # Use asyncio.as_completed to yield outputs as they arrive
        for task in asyncio.as_completed(stage_tasks):
            async for output in await task:
                if output is not None:
                    yield output

    async def _forward_to_next_stage(
        self,
        src_stage_id: int,
        dst_stage_id: int,
        request_id: str,
        original_prompt: OmniPromptType,
        sampling_params_list: list[OmniSamplingParams],
        engine_outputs: Any,
        metrics: OrchestratorMetrics,
    ) -> None:
        """Forward request to next stage via connector.

        Args:
            src_stage_id: Source stage ID
            dst_stage_id: Destination stage ID
            request_id: Request identifier
            original_prompt: Original prompt
            sampling_params_list: List of sampling params
            engine_outputs: Outputs from source stage
            metrics: OrchestratorMetrics
        """
        src_stage_ctx = self.stages[src_stage_id]
        dst_stage_ctx = self.stages[dst_stage_id]

        # Process inputs for next stage
        next_inputs = dst_stage_ctx.stage_config.process_engine_inputs(
            [s.stage_config for s in self.stages], original_prompt
        )

        # Get connector for this edge
        connector_key = (str(src_stage_id), str(dst_stage_id))
        connector = src_stage_ctx.connectors.get(connector_key)

        if not connector:
            raise RuntimeError(
                f"No connector found for edge {src_stage_id} -> {dst_stage_id}"
            )

        # Send via connector
        sent = try_send_via_connector(
            connector=connector,
            stage_id=src_stage_id,
            next_stage_id=dst_stage_id,
            req_id=request_id,
            next_inputs=next_inputs,
            sampling_params=sampling_params_list[dst_stage_id],
            original_prompt=original_prompt,
            next_stage_queue_submit_fn=lambda task: asyncio.create_task(
                dst_stage_ctx.client.submit_request(
                    request_id=task["request_id"],
                    prompt=task["prompt"],
                    sampling_params=task["sampling_params"],
                )
            ),
            metrics=metrics,
        )

        if not sent:
            raise RuntimeError(
                f"Failed to send request {request_id} to stage {dst_stage_id} via connector"
            )

        logger.debug(f"Forwarded request {request_id} from stage {src_stage_id} to {dst_stage_id}")

    def _create_final_output(
        self,
        stage_id: int,
        stage_ctx: StageContext,
        output: OmniRequestOutput,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
        final_stage_id: int,
    ) -> OmniRequestOutput:
        """Create final output for yielding.

        Args:
            stage_id: Stage identifier
            stage_ctx: StageContext
            output: OmniRequestOutput from stage
            metrics: OrchestratorMetrics
            req_start_ts: Request start timestamps
            wall_start_ts: Wall clock start time
            final_stage_id: Final stage ID

        Returns:
            OmniRequestOutput configured for final output
        """
        # Finalize metrics if this is the E2E final stage
        if stage_id == final_stage_id and output.finished:
            req_id = output.request_id
            rid_key = str(req_id)
            if rid_key not in metrics.e2e_done:
                metrics.on_finalize_request(
                    stage_id,
                    req_id,
                    req_start_ts.get(req_id, wall_start_ts),
                )

        # Extract images if needed
        images = []
        if stage_ctx.final_output_type == "image":
            if isinstance(output, OmniRequestOutput) and output.images:
                images = output.images
            elif hasattr(output, "images") and output.images:
                images = output.images

        # Create final output
        return OmniRequestOutput(
            request_id=output.request_id,
            stage_id=stage_id,
            final_output_type=stage_ctx.final_output_type or "text",
            request_output=output.request_output if hasattr(output, "request_output") else output,
            images=images,
            finished=output.finished,
        )

    async def abort_request(self, request_id: str) -> None:
        """Abort a request across all stages.

        Args:
            request_id: Request identifier to abort
        """
        # Remove from request states
        self.request_states.pop(request_id, None)

        # Abort in all stages
        for stage_ctx in self.stages:
            try:
                await stage_ctx.client.abort_request(request_id)
            except Exception as e:
                logger.exception(f"Error aborting request {request_id} in stage {stage_ctx.stage_id}")

        logger.info(f"Aborted request {request_id} across all stages")

    async def shutdown(self):
        """Shutdown all stage clients."""
        logger.info("Shutting down pipeline orchestrator")

        for stage_ctx in self.stages:
            try:
                await stage_ctx.client.shutdown()
            except Exception as e:
                logger.exception(f"Error shutting down stage {stage_ctx.stage_id}")

        logger.info("Pipeline orchestrator shutdown complete")
