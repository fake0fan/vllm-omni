from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from vllm_omni.engine.pipeline_state import PipelineData, RequestMeta
from vllm_omni.engine.stage0_processing import (
    prepare_stage0_llm_request,
    register_stage0_output,
)


@dataclass
class StagePollResult:
    request_outputs: list[Any] = field(default_factory=list)
    kv_ready_outputs: list[Any] = field(default_factory=list)


class StageRuntime(ABC):
    def __init__(self, *, stage_client: Any, output_processor: Any | None, stage_vllm_config: Any | None) -> None:
        self.stage_client = stage_client
        self.output_processor = output_processor
        self.stage_vllm_config = stage_vllm_config
        self.stage_id = stage_client.stage_id
        self.stage_type = stage_client.stage_type
        self.final_output = stage_client.final_output
        self.final_output_type = getattr(stage_client, "final_output_type", "text")
        self.engine_input_source = getattr(stage_client, "engine_input_source", None)
        self.custom_process_input_func = getattr(stage_client, "custom_process_input_func", None)

    def register_request(self, *, request: Any, prompt: Any | None) -> None:
        if self.output_processor is None:
            return
        self.output_processor.add_request(
            request=request,
            prompt=prompt,
            parent_req=None,
            request_index=0,
            queue=None,
        )

    async def accept_external_request(self, *, meta: RequestMeta, data: PipelineData) -> Any:
        request = data.raw_prompt
        data.stage0_request = request
        await self.submit(request=request, request_id=meta.request_id, params=meta.entry_params)
        return request

    async def accept_streaming_update(self, *, meta: RequestMeta, data: PipelineData) -> Any:
        return await self.accept_external_request(meta=meta, data=data)

    @abstractmethod
    async def submit(
        self,
        *,
        request: Any,
        request_id: str | None = None,
        params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def poll_processed_outputs(self) -> StagePollResult:
        raise NotImplementedError

    async def abort(self, request_ids: list[str]) -> None:
        await self.stage_client.abort_requests_async(request_ids)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
    ) -> Any:
        if hasattr(self.stage_client, "collective_rpc_async"):
            return await self.stage_client.collective_rpc_async(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs or {},
            )
        return {
            "supported": False,
            "todo": True,
            "reason": f"{self.stage_client.__class__.__name__}.collective_rpc_async is not implemented yet",
        }

    def set_engine_outputs(self, outputs: list[Any]) -> None:
        if hasattr(self.stage_client, "set_engine_outputs"):
            self.stage_client.set_engine_outputs(outputs)

    def process_engine_inputs(self, stage_list: list[Any], prompt: Any = None) -> list[Any]:
        return list(self.stage_client.process_engine_inputs(stage_list=stage_list, prompt=prompt))

    def shutdown(self) -> None:
        self.stage_client.shutdown()


class LLMStageRuntime(StageRuntime):
    def __init__(
        self,
        *,
        stage_client: Any,
        output_processor: Any | None,
        stage_vllm_config: Any | None,
        input_processor: Any | None = None,
        supported_tasks: Sequence[str] | None = None,
    ) -> None:
        if output_processor is None:
            raise ValueError("LLMStageRuntime requires an output_processor")
        super().__init__(
            stage_client=stage_client,
            output_processor=output_processor,
            stage_vllm_config=stage_vllm_config,
        )
        self.input_processor = input_processor
        self.supported_tasks = tuple(supported_tasks) if supported_tasks is not None else ("generate",)

    async def _accept_stage0_request(self, *, meta: RequestMeta, data: PipelineData) -> Any:
        if self.input_processor is None:
            raise ValueError("LLMStageRuntime requires an input_processor for entry requests")

        request = prepare_stage0_llm_request(
            meta=meta,
            data=data,
            input_processor=self.input_processor,
            supported_tasks=self.supported_tasks,
        )
        data.stage0_request = request
        register_stage0_output(
            self.output_processor,
            request=request,
            prompt_text=meta.prompt_text,
            original_prompt=data.raw_prompt,
        )
        await self.submit(request=request, request_id=meta.request_id, params=meta.entry_params)
        return request

    async def accept_external_request(self, *, meta: RequestMeta, data: PipelineData) -> Any:
        return await self._accept_stage0_request(meta=meta, data=data)

    async def accept_streaming_update(self, *, meta: RequestMeta, data: PipelineData) -> Any:
        return await self._accept_stage0_request(meta=meta, data=data)

    async def submit(
        self,
        *,
        request: Any,
        request_id: str | None = None,
        params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        await self.stage_client.add_request_async(request)

    async def poll_processed_outputs(self) -> StagePollResult:
        raw_outputs = await self.stage_client.get_output_async()
        if raw_outputs.scheduler_stats is not None:
            self.output_processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        if not raw_outputs.outputs:
            return StagePollResult()

        processed = self.output_processor.process_outputs(raw_outputs.outputs, raw_outputs.timestamp, None)
        if processed.reqs_to_abort:
            await self.stage_client.abort_requests_async(processed.reqs_to_abort)

        kv_ready_outputs = [
            raw_output
            for raw_output in raw_outputs.outputs
            if isinstance(getattr(raw_output, "kv_transfer_params", None), dict)
            and raw_output.kv_transfer_params.get("kv_ready")
        ]
        return StagePollResult(
            request_outputs=list(processed.request_outputs),
            kv_ready_outputs=kv_ready_outputs,
        )


class DiffusionStageRuntime(StageRuntime):
    async def submit(
        self,
        *,
        request: Any,
        request_id: str | None = None,
        params: Any | None = None,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if request_id is None:
            raise ValueError("request_id is required for diffusion stage submit")
        if params is None:
            raise ValueError("params are required for diffusion stage submit")
        if isinstance(request, list):
            await self.stage_client.add_batch_request_async(
                request_id,
                request,
                params,
                kv_sender_info=kv_sender_info,
            )
        else:
            await self.stage_client.add_request_async(
                request_id,
                request,
                params,
                kv_sender_info=kv_sender_info,
            )

    async def poll_processed_outputs(self) -> StagePollResult:
        output = self.stage_client.get_diffusion_output_nowait()
        if output is None:
            return StagePollResult()
        return StagePollResult(request_outputs=[output], kv_ready_outputs=[])


def build_stage_runtimes(
    *,
    stage_clients: list[Any],
    output_processors: list[Any | None],
    stage_vllm_configs: list[Any | None],
    entry_stage_id: int | None = None,
    entry_input_processor: Any | None = None,
    supported_tasks: Sequence[str] | None = None,
) -> list[StageRuntime]:
    runtimes: list[StageRuntime] = []
    for stage_client, output_processor, stage_vllm_config in zip(
        stage_clients,
        output_processors,
        stage_vllm_configs,
        strict=True,
    ):
        if stage_client.stage_type == "llm":
            runtime_cls = LLMStageRuntime
        elif stage_client.stage_type == "diffusion":
            runtime_cls = DiffusionStageRuntime
        else:
            raise ValueError(f"Unknown stage_type: {stage_client.stage_type!r}")
        runtime_kwargs: dict[str, Any] = {}
        if runtime_cls is LLMStageRuntime and stage_client.stage_id == entry_stage_id:
            runtime_kwargs["input_processor"] = entry_input_processor
            runtime_kwargs["supported_tasks"] = supported_tasks
        runtimes.append(
            runtime_cls(
                stage_client=stage_client,
                output_processor=output_processor,
                stage_vllm_config=stage_vllm_config,
                **runtime_kwargs,
            )
        )
    return runtimes
