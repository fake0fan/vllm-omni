from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineRequestState:
    global_request_id: str
    original_prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    active_stage_ids: set[int] = field(default_factory=set)
    cancelled: bool = False

    @property
    def request_id(self) -> str:
        return self.global_request_id

    @property
    def prompt(self) -> Any:
        return self.original_prompt

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
