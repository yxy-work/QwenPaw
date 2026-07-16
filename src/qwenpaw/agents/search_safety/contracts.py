# -*- coding: utf-8 -*-
"""知识库检索安全层共享契约。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field


class SearchSafetyDecision(str, Enum):
    """一次知识库检索尝试的安全判定。"""

    ALLOWED = "allowed"
    DUPLICATE_REUSED = "duplicate_reused"
    SIMILAR_QUERY_BLOCKED = "similar_query_blocked"
    LIMIT_REACHED = "limit_reached"
    CONTEXT_BUDGET_REACHED = "context_budget_reached"


class ContextBudgetAssessment(BaseModel):
    """检索或模型请求的上下文预算判定快照。"""

    model_config = ConfigDict(frozen=True, extra="ignore")

    allowed: bool
    current_input_tokens: int = Field(ge=0)
    projected_input_tokens: int = Field(ge=0)
    safe_input_limit: int = Field(ge=0)
    estimator: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ContextWindowBudgetGuardProtocol(Protocol):
    """两层上下文预算检查器的稳定接口。"""

    async def assess_search_preflight(
        self,
        *,
        current_input_tokens: int,
        projected_search_result_tokens: int,
        inflight_reserved_tokens: int = 0,
    ) -> ContextBudgetAssessment:
        """在检索前预测新增结果能否安全进入上下文。"""

    async def assess_model_request(
        self,
        *,
        input_kwargs: Mapping[str, Any],
    ) -> ContextBudgetAssessment:
        """在最终模型请求发出前检查实际请求体。"""


class ContextWindowPreflightBlocked(RuntimeError):
    """最终模型请求被本地上下文预检阻止的内部控制流异常。"""

    def __init__(self, assessment: ContextBudgetAssessment) -> None:
        self.assessment = assessment
        super().__init__(
            "上下文窗口预检阻止模型请求："
            f"{assessment.reason} "
            f"({assessment.projected_input_tokens} > "
            f"{assessment.safe_input_limit})",
        )


__all__ = [
    "ContextBudgetAssessment",
    "ContextWindowBudgetGuardProtocol",
    "ContextWindowPreflightBlocked",
    "SearchSafetyDecision",
]
