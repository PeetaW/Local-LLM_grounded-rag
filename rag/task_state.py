# rag/task_state.py
# Plan-and-Execute 狀態表資料結構
# 供 Task 3-B/3-C 的 PlanExecutor 使用
# 預設由 PLAN_EXECUTE_ENABLED=False 控制，不影響現有 pipeline

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


class InferenceType(Enum):
    DIRECT           = "DIRECT"           # 直接來自論文引用
    INFERENCE_BRIDGE = "INFERENCE_BRIDGE" # 跨文獻推論橋接
    UNSUPPORTED      = "UNSUPPORTED"      # 無文獻支撐


@dataclass
class SubTask:
    id: str                              # 唯一 ID，如 "T1"
    question: str                        # 子問題文字
    depends_on: list[str] = field(default_factory=list)  # 前置任務 ID 列表

    # 執行結果（COMPLETED 後填入）
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None         # LLM 回答文字
    inference_type: Optional[InferenceType] = None
    grounding_score: Optional[float] = None
    error: Optional[str] = None          # FAILED 時的錯誤訊息


@dataclass
class ResearchPlan:
    question: str                        # 原始問題
    tasks: list[SubTask] = field(default_factory=list)

    def get_ready_tasks(self) -> list[SubTask]:
        """
        回傳「前置任務已全部 COMPLETED，且自身為 PENDING」的任務列表。
        這些任務可以立即開始執行。
        """
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def get_task_by_id(self, task_id: str) -> Optional[SubTask]:
        """根據 ID 取得任務，找不到回傳 None。"""
        return next((t for t in self.tasks if t.id == task_id), None)

    def all_completed(self) -> bool:
        """回傳 True 代表所有任務均已 COMPLETED。"""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def has_failed(self) -> bool:
        """回傳 True 代表有任何任務 FAILED。"""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    def to_summary(self) -> str:
        """
        轉成供最終整合用的結構化摘要。
        格式：每個已完成任務的問題 + 結論，依 ID 排序。
        """
        lines = []
        for t in sorted(self.tasks, key=lambda x: x.id):
            if t.status == TaskStatus.COMPLETED and t.result:
                inference_label = (
                    f"[{t.inference_type.value}]" if t.inference_type else ""
                )
                lines.append(f"## {t.id} {inference_label}\n問：{t.question}\n答：{t.result}\n")
        return "\n".join(lines)
