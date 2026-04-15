from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestMeta:
    request_id: str
    final_stage_id: int
    sampling_params_list: list[Any] = field(default_factory=list)
    prompt_text: str | None = None
    arrival_time: float | None = None
    lora_request: Any = None
    tokenization_kwargs: dict[str, Any] | None = None
    trace_headers: dict[str, str] | None = None
    priority: int = 0
    data_parallel_rank: int | None = None
    reasoning_ended: bool | None = None
    resumable: bool = False

    @property
    def entry_params(self) -> Any:
        if not self.sampling_params_list:
            raise ValueError(f"RequestMeta {self.request_id} is missing stage-0 sampling params")
        return self.sampling_params_list[0]


@dataclass
class PipelineData:
    raw_prompt: Any = None
    stage0_request: Any = None
    terminal_outputs: list[Any] = field(default_factory=list)


@dataclass(init=False)
class PipelineRequestState:
    meta: RequestMeta
    data: PipelineData
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    active_stage_ids: set[int] = field(default_factory=set)
    cancelled: bool = False

    def __init__(
        self,
        meta: RequestMeta | None = None,
        data: PipelineData | None = None,
        stage_submit_ts: dict[int, float] | None = None,
        active_stage_ids: set[int] | None = None,
        cancelled: bool = False,
        **legacy_kwargs: Any,
    ) -> None:
        if meta is None and data is None and legacy_kwargs:
            legacy_sampling_params_list = legacy_kwargs.pop("sampling_params_list", [])
            legacy_terminal_outputs = legacy_kwargs.pop("terminal_outputs", [])
            meta = RequestMeta(
                request_id=legacy_kwargs.pop("global_request_id"),
                final_stage_id=legacy_kwargs.pop("final_stage_id", -1),
                sampling_params_list=[] if legacy_sampling_params_list is None else list(legacy_sampling_params_list),
                prompt_text=legacy_kwargs.pop("prompt_text", None),
                arrival_time=legacy_kwargs.pop("arrival_time", None),
                lora_request=legacy_kwargs.pop("lora_request", None),
                tokenization_kwargs=legacy_kwargs.pop("tokenization_kwargs", None),
                trace_headers=legacy_kwargs.pop("trace_headers", None),
                priority=legacy_kwargs.pop("priority", 0),
                data_parallel_rank=legacy_kwargs.pop("data_parallel_rank", None),
                reasoning_ended=legacy_kwargs.pop("reasoning_ended", None),
                resumable=legacy_kwargs.pop("resumable", False),
            )
            data = PipelineData(
                raw_prompt=legacy_kwargs.pop("original_prompt", None),
                stage0_request=legacy_kwargs.pop("stage0_request", None),
                terminal_outputs=[] if legacy_terminal_outputs is None else list(legacy_terminal_outputs),
            )
        elif meta is None or data is None:
            raise TypeError("PipelineRequestState requires either meta/data or legacy flat request fields")

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments for PipelineRequestState: {unexpected}")

        self.meta = meta
        self.data = data
        self.stage_submit_ts = dict(stage_submit_ts or {})
        self.active_stage_ids = set(active_stage_ids or set())
        self.cancelled = cancelled

    @property
    def request_id(self) -> str:
        return self.meta.request_id

    @property
    def global_request_id(self) -> str:
        return self.meta.request_id

    @global_request_id.setter
    def global_request_id(self, value: str) -> None:
        self.meta.request_id = value

    @property
    def prompt(self) -> Any:
        return self.data.raw_prompt

    @prompt.setter
    def prompt(self, value: Any) -> None:
        self.data.raw_prompt = value

    @property
    def original_prompt(self) -> Any:
        return self.data.raw_prompt

    @original_prompt.setter
    def original_prompt(self, value: Any) -> None:
        self.data.raw_prompt = value

    @property
    def sampling_params_list(self) -> list[Any]:
        return self.meta.sampling_params_list

    @sampling_params_list.setter
    def sampling_params_list(self, value: list[Any]) -> None:
        self.meta.sampling_params_list = list(value)

    @property
    def final_stage_id(self) -> int:
        return self.meta.final_stage_id

    @property
    def entry_params(self) -> Any:
        return self.meta.entry_params

    def mark_stage_submitted(self, stage_id: int, submitted_at: float) -> None:
        self.stage_submit_ts[stage_id] = submitted_at
        if not self.cancelled:
            self.active_stage_ids.add(stage_id)

    def mark_stage_finished(self, stage_id: int) -> None:
        self.active_stage_ids.discard(stage_id)

    def next_stage_already_submitted(self, stage_id: int) -> bool:
        return (stage_id + 1) in self.stage_submit_ts

    def cancel(self) -> None:
        self.cancelled = True
        self.active_stage_ids.clear()
