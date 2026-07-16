# -*- coding: utf-8 -*-
"""按 reply 隔离的知识库检索安全状态。"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import SearchSafetyDecision


class SearchCallRecord(BaseModel):
    """单次检索判定所需的紧凑元数据。"""

    model_config = ConfigDict(extra="ignore")

    attempt_index: int = Field(default=0, ge=0)
    decision: SearchSafetyDecision = SearchSafetyDecision.ALLOWED
    exact_signature: str = ""
    normalized_kb_name: str = ""
    normalized_query: str = ""
    effective_filters_signature: str = ""
    matched_attempt_index: int | None = Field(default=None, ge=1)
    reserved_tokens: int = Field(default=0, ge=0)
    mcp_called: bool = False


class SearchSafetyState(BaseModel):
    """一个 ``reply_id`` 内的知识库检索安全状态。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    reply_id: str = Field(min_length=1)
    max_attempts_per_reply: int = Field(default=10, ge=1, le=50)
    observed_attempt_count: int = Field(default=0, ge=0)
    admitted_attempt_count: int = Field(default=0, ge=0)
    mcp_call_count: int = Field(default=0, ge=0)
    force_finalize: bool = False
    force_finalize_reason: str | None = None
    inflight_reserved_tokens: int = Field(default=0, ge=0)
    calls: list[SearchCallRecord] = Field(default_factory=list)
    exact_signature_to_call_index: dict[str, int] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_counter_invariants(self) -> "SearchSafetyState":
        """拒绝不可能的计数器组合。"""
        if self.admitted_attempt_count > self.observed_attempt_count:
            raise ValueError(
                "admitted_attempt_count 不得超过 observed_attempt_count",
            )
        if self.admitted_attempt_count > self.max_attempts_per_reply:
            raise ValueError(
                "admitted_attempt_count 不得超过 max_attempts_per_reply",
            )
        if self.mcp_call_count > self.admitted_attempt_count:
            raise ValueError(
                "mcp_call_count 不得超过 admitted_attempt_count",
            )
        return self


class SearchSafetyStateStore:
    """仅保存当前 reply，并通过单锁提供原子状态操作。"""

    __slots__ = ("_lock", "_state")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state: SearchSafetyState | None = None

    @property
    def active_reply_id(self) -> str | None:
        """返回当前唯一活跃的 reply 标识。"""
        return self._state.reply_id if self._state is not None else None

    @staticmethod
    def _validate_reply_id(reply_id: str) -> None:
        if not reply_id:
            raise ValueError("reply_id 不得为空")

    def _activate_locked(self, reply_id: str) -> SearchSafetyState:
        self._validate_reply_id(reply_id)
        if self._state is None or self._state.reply_id != reply_id:
            self._state = SearchSafetyState(reply_id=reply_id)
        return self._state

    def _state_for_operation_locked(
        self,
        reply_id: str,
    ) -> SearchSafetyState:
        """获取操作状态，并拒绝已淘汰 reply 的迟到操作。"""
        self._validate_reply_id(reply_id)
        if self._state is None:
            self._state = SearchSafetyState(reply_id=reply_id)
        elif self._state.reply_id != reply_id:
            raise RuntimeError(
                "reply_id 与当前活跃状态不一致；拒绝复用旧 reply 状态",
            )
        return self._state

    @staticmethod
    def _snapshot(state: SearchSafetyState) -> SearchSafetyState:
        return state.model_copy(deep=True)

    async def get_or_create(self, reply_id: str) -> SearchSafetyState:
        """获取当前 reply；reply 变化时立即替换旧状态。"""
        async with self._lock:
            return self._snapshot(self._activate_locked(reply_id))

    async def get(self, reply_id: str) -> SearchSafetyState | None:
        """仅在 reply 一致时返回状态快照。"""
        async with self._lock:
            if self._state is None or self._state.reply_id != reply_id:
                return None
            return self._snapshot(self._state)

    async def reserve_attempt_slot(
        self,
        reply_id: str,
        max_attempts_per_reply: int,
    ) -> tuple[int, int | None]:
        """原子记录尝试并分配安全槽位。"""
        if max_attempts_per_reply < 1:
            raise ValueError("max_attempts_per_reply 必须为正整数")
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            if state.admitted_attempt_count > max_attempts_per_reply:
                raise ValueError(
                    "新的 max_attempts_per_reply 低于已分配槽位数",
                )
            state.max_attempts_per_reply = max_attempts_per_reply
            state.observed_attempt_count += 1
            observed_index = state.observed_attempt_count
            if state.admitted_attempt_count >= max_attempts_per_reply:
                state.force_finalize = True
                if state.force_finalize_reason is None:
                    state.force_finalize_reason = (
                        "max_attempts_per_reply_reached"
                    )
                return observed_index, None

            state.admitted_attempt_count += 1
            admitted_index = state.admitted_attempt_count
            if admitted_index == max_attempts_per_reply:
                state.force_finalize = True
                if state.force_finalize_reason is None:
                    state.force_finalize_reason = (
                        "max_attempts_per_reply_reached"
                    )
            return observed_index, admitted_index

    async def mark_mcp_call(self, reply_id: str) -> int:
        """原子增加实际 MCP 调用计数并维持计数不变量。"""
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            if state.mcp_call_count >= state.admitted_attempt_count:
                raise RuntimeError(
                    "无法记录 MCP 调用：没有可用的 admitted attempt",
                )
            state.mcp_call_count += 1
            return state.mcp_call_count

    async def try_reserve_tokens(
        self,
        reply_id: str,
        token_count: int,
        max_inflight_tokens: int | None = None,
    ) -> bool:
        """在可选上限内原子预留并发结果 token。"""
        if token_count < 0:
            raise ValueError("token_count 不得为负数")
        if max_inflight_tokens is not None and max_inflight_tokens < 0:
            raise ValueError("max_inflight_tokens 不得为负数")
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            projected = state.inflight_reserved_tokens + token_count
            if (
                max_inflight_tokens is not None
                and projected > max_inflight_tokens
            ):
                return False
            state.inflight_reserved_tokens = projected
            return True

    async def release_tokens(self, reply_id: str, token_count: int) -> int:
        """原子释放先前预留的结果 token。"""
        if token_count < 0:
            raise ValueError("token_count 不得为负数")
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            if token_count > state.inflight_reserved_tokens:
                raise ValueError(
                    "释放 token 超过当前 inflight_reserved_tokens",
                )
            state.inflight_reserved_tokens -= token_count
            return state.inflight_reserved_tokens

    async def set_force_finalize(self, reply_id: str, reason: str) -> None:
        """原子设置强制收尾标志及原因。"""
        if not reason:
            raise ValueError("force_finalize reason 不得为空")
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            state.force_finalize = True
            state.force_finalize_reason = reason

    async def get_force_finalize(
        self,
        reply_id: str,
    ) -> tuple[bool, str | None]:
        """读取指定 reply 的强制收尾状态。"""
        async with self._lock:
            if self._state is not None and self._state.reply_id != reply_id:
                return False, None
            state = self._state_for_operation_locked(reply_id)
            return state.force_finalize, state.force_finalize_reason

    async def record_call(
        self,
        reply_id: str,
        record: SearchCallRecord,
    ) -> int:
        """原子追加紧凑调用记录，并登记首次 exact signature。"""
        async with self._lock:
            state = self._state_for_operation_locked(reply_id)
            call_index = len(state.calls)
            state.calls.append(record.model_copy(deep=True))
            if record.exact_signature:
                state.exact_signature_to_call_index.setdefault(
                    record.exact_signature,
                    call_index,
                )
            return call_index

    async def dump(self, reply_id: str) -> dict[str, Any] | None:
        """返回可直接 JSON 序列化的状态载荷。"""
        async with self._lock:
            if self._state is None or self._state.reply_id != reply_id:
                return None
            return self._state.model_dump(mode="json")

    def snapshot_sync(self, reply_id: str) -> SearchSafetyState:
        """在 AgentScope 同步 session 保存边界返回状态快照。

        调用方必须位于 Agent 实例尚未并发执行或同一事件循环的同步段；若
        Store 已处于异步临界区则明确拒绝，避免绕过锁读取中间状态。
        """
        self._validate_reply_id(reply_id)
        if self._lock.locked():
            raise RuntimeError("检索安全 Store 正在使用中，无法同步保存状态")
        if self._state is None or self._state.reply_id != reply_id:
            return SearchSafetyState(reply_id=reply_id)
        return self._snapshot(self._state)

    def restore_sync(
        self,
        reply_id: str,
        payload: Mapping[str, Any] | None,
    ) -> SearchSafetyState:
        """在 Agent 构造期的同步 session 恢复边界装载状态。"""
        self._validate_reply_id(reply_id)
        if self._lock.locked():
            raise RuntimeError("检索安全 Store 正在使用中，无法同步恢复状态")
        restored: SearchSafetyState | None = None
        if payload is not None:
            candidate = SearchSafetyState.model_validate(dict(payload))
            if candidate.reply_id == reply_id:
                restored = candidate
        self._state = restored or SearchSafetyState(reply_id=reply_id)
        return self._snapshot(self._state)

    async def restore(
        self,
        reply_id: str,
        payload: Mapping[str, Any] | None,
    ) -> SearchSafetyState:
        """恢复 reply 一致的旧状态；不一致时创建干净状态。"""
        self._validate_reply_id(reply_id)
        async with self._lock:
            restored: SearchSafetyState | None = None
            if payload is not None:
                candidate = SearchSafetyState.model_validate(dict(payload))
                if candidate.reply_id == reply_id:
                    restored = candidate
            self._state = restored or SearchSafetyState(reply_id=reply_id)
            return self._snapshot(self._state)


__all__ = [
    "SearchCallRecord",
    "SearchSafetyState",
    "SearchSafetyStateStore",
]
