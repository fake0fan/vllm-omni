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
    terminal_outputs: dict[int, Any] = field(default_factory=dict)


@dataclass
class PipelineRequestState:
    meta: RequestMeta
    data: PipelineData
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    active_stage_ids: set[int] = field(default_factory=set)
    cancelled: bool = False

    @property
    def request_id(self) -> str:
        return self.meta.request_id

    @property
    def prompt(self) -> Any:
        return self.data.raw_prompt

    @property
    def original_prompt(self) -> Any:
        return self.data.raw_prompt

    @property
    def sampling_params_list(self) -> list[Any]:
        return self.meta.sampling_params_list

    @property
    def final_stage_id(self) -> int:
        return self.meta.final_stage_id

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
