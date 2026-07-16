# -*- coding: utf-8 -*-
"""KnowledgeBase ``search_knowledgebase`` 单轮安全中间件。"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentscope.message import TextBlock, ToolResultState
from agentscope.middleware import MiddlewareBase
from agentscope.tool import ToolResponse

from .contracts import (
    ContextWindowBudgetGuardProtocol,
    SearchSafetyDecision,
)
from .matching import (
    SearchMatchInput,
    build_exact_signature,
    build_search_match_input,
    queries_are_highly_similar,
)
from .state import SearchCallRecord, SearchSafetyStateStore

if TYPE_CHECKING:
    from qwenpaw.config.config import SearchKnowledgeBaseSafetyConfig

logger = logging.getLogger(__name__)

_SEARCH_CAPABILITY_NAME = "search_knowledgebase"
_IDENTITY_ATTRIBUTES = (
    "capability_id",
    "driver_name",
    "protocol",
    "original_capability_name",
)


@dataclass(frozen=True)
class _PreparedCall:
    short_response: ToolResponse | None
    reserved_tokens: int = 0


class KnowledgeBaseSearchSafetyMiddleware(MiddlewareBase):
    """对构造时注入的稳定 MCP capability 身份执行检索保护。"""

    def __init__(
        self,
        *,
        state_store: SearchSafetyStateStore,
        config: "SearchKnowledgeBaseSafetyConfig",
        target_capabilities: Iterable[Any],
        budget_guard: ContextWindowBudgetGuardProtocol | None = None,
        current_input_tokens_getter: (
            Callable[[Any], int | Awaitable[int]] | None
        ) = None,
        projected_search_result_tokens: int = 0,
    ) -> None:
        if projected_search_result_tokens < 0:
            raise ValueError("projected_search_result_tokens 不得为负数")
        self._state_store = state_store
        self._config = config
        self._budget_guard = budget_guard
        self._current_input_tokens_getter = (
            current_input_tokens_getter or (lambda _agent: 0)
        )
        self._projected_search_result_tokens = projected_search_result_tokens
        self._target_identities = frozenset(
            self._validated_target_identity(capability)
            for capability in target_capabilities
        )
        if not self._target_identities:
            raise ValueError("target_capabilities 不得为空")

        self._decision_lock = asyncio.Lock()

    @staticmethod
    def _identity_tuple(capability: Any) -> tuple[str, str, str, str] | None:
        values = tuple(
            getattr(capability, attribute, None)
            for attribute in _IDENTITY_ATTRIBUTES
        )
        if not all(isinstance(value, str) and value for value in values):
            return None
        return values  # type: ignore[return-value]

    @classmethod
    def _validated_target_identity(
        cls,
        capability: Any,
    ) -> tuple[str, str, str, str]:
        identity = cls._identity_tuple(capability)
        if identity is None:
            raise ValueError(
                "目标 capability 缺少稳定身份属性："
                + ", ".join(_IDENTITY_ATTRIBUTES),
            )
        if identity[2].casefold() != "mcp":
            raise ValueError("目标 capability 必须使用 MCP protocol")
        if identity[3] != _SEARCH_CAPABILITY_NAME:
            raise ValueError(
                "目标 capability 必须是原始 search_knowledgebase",
            )
        return identity

    async def _resolve_target_tool(
        self,
        agent: Any,
        tool_name: str,
    ) -> tuple[bool, Any | None]:
        try:
            tool = await agent.toolkit.get_tool(tool_name)
        except Exception:
            logger.warning(
                "无法读取工具稳定 capability 身份，安全中间件按普通工具旁路",
                exc_info=True,
            )
            return False, None
        identity = self._identity_tuple(tool)
        return identity in self._target_identities, tool

    @staticmethod
    def _short_response(payload: dict[str, Any]) -> ToolResponse:
        return ToolResponse(
            content=[
                TextBlock(
                    text=json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            ],
            state=ToolResultState.SUCCESS,
        )

    @classmethod
    def _duplicate_response(cls, reuse_attempt: int) -> ToolResponse:
        return cls._short_response(
            {
                "status": "duplicate_reused",
                "reuse_attempt": reuse_attempt,
                "instruction": (
                    f"不要重复检索；直接复用第{reuse_attempt}次检索证据。"
                ),
            },
        )

    @classmethod
    def _similar_response(cls, similar_attempt: int) -> ToolResponse:
        return cls._short_response(
            {
                "status": "similar_query_blocked",
                "similar_to_attempt": similar_attempt,
                "instruction": (
                    "仅改写措辞不会执行；请明确新增实体、时间、范围、"
                    "来源或矛盾，否则整理已有证据。"
                ),
            },
        )

    @classmethod
    def _limit_response(cls, max_attempts: int) -> ToolResponse:
        return cls._short_response(
            {
                "status": "limit_reached",
                "max_attempts": max_attempts,
                "force_finalize": True,
                "instruction": "停止调用工具，基于已有证据回答并说明不足。",
            },
        )

    @classmethod
    def _budget_response(cls) -> ToolResponse:
        return cls._short_response(
            {
                "status": "context_budget_reached",
                "force_finalize": True,
                "instruction": "剩余上下文不足；立即整理已有证据并回答。",
            },
        )

    async def _current_input_tokens(self, agent: Any) -> int:
        result = self._current_input_tokens_getter(agent)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, int) or isinstance(result, bool) or result < 0:
            raise ValueError("current_input_tokens_getter 必须返回非负整数")
        return result

    async def _record(
        self,
        reply_id: str,
        *,
        attempt_index: int,
        decision: SearchSafetyDecision,
        exact_signature: str = "",
        normalized_kb_name: str = "",
        normalized_query: str = "",
        effective_filters_signature: str = "",
        matched_attempt_index: int | None = None,
        reserved_tokens: int = 0,
        mcp_called: bool = False,
    ) -> None:
        await self._state_store.record_call(
            reply_id,
            SearchCallRecord(
                attempt_index=attempt_index,
                decision=decision,
                exact_signature=exact_signature,
                normalized_kb_name=normalized_kb_name,
                normalized_query=normalized_query,
                effective_filters_signature=effective_filters_signature,
                matched_attempt_index=matched_attempt_index,
                reserved_tokens=reserved_tokens,
                mcp_called=mcp_called,
            ),
        )

    @staticmethod
    def _find_executed_duplicate(
        state: Any,
        exact_signature: str,
    ) -> int | None:
        for record in state.calls:
            if record.exact_signature == exact_signature and record.mcp_called:
                return record.attempt_index
        return None

    def _find_similar(
        self,
        state: Any,
        match_input: SearchMatchInput,
    ) -> int | None:
        for record in state.calls:
            if not record.mcp_called or not record.normalized_query:
                continue
            if record.normalized_kb_name != match_input.normalized_kb_name:
                continue
            if (
                record.effective_filters_signature
                != match_input.effective_filters_signature
            ):
                continue
            if queries_are_highly_similar(
                record.normalized_query,
                match_input.normalized_query,
                containment_length_ratio=(
                    self._config.containment_length_ratio
                ),
                bigram_jaccard_threshold=(
                    self._config.bigram_jaccard_threshold
                ),
                sequence_matcher_threshold=(
                    self._config.sequence_matcher_threshold
                ),
            ):
                return record.attempt_index
        return None

    async def _assess_and_reserve_budget(
        self,
        agent: Any,
        reply_id: str,
    ) -> tuple[bool, int]:
        if (
            not self._config.context_preflight_enabled
            or self._budget_guard is None
        ):
            return True, 0
        state = await self._state_store.get(reply_id)
        if state is None:
            raise RuntimeError("预算预检时 reply 状态不存在")
        current_tokens = await self._current_input_tokens(agent)
        reserved_tokens = self._projected_search_result_tokens
        assessment = await self._budget_guard.assess_search_preflight(
            current_input_tokens=current_tokens,
            projected_search_result_tokens=reserved_tokens,
            inflight_reserved_tokens=state.inflight_reserved_tokens,
        )
        if not assessment.allowed:
            return False, 0

        remaining_after_current = max(
            0,
            assessment.safe_input_limit - assessment.projected_input_tokens,
        )
        max_inflight_tokens = (
            state.inflight_reserved_tokens
            + reserved_tokens
            + remaining_after_current
        )
        reserved = await self._state_store.try_reserve_tokens(
            reply_id,
            reserved_tokens,
            max_inflight_tokens=max_inflight_tokens,
        )
        return reserved, reserved_tokens if reserved else 0

    async def _prepare_target_call(
        self,
        agent: Any,
        reply_id: str,
        tool_call: Any,
        tool: Any,
    ) -> _PreparedCall:
        async with self._decision_lock:
            await self._state_store.get_or_create(reply_id)
            observed_index, admitted_index = (
                await self._state_store.reserve_attempt_slot(
                    reply_id,
                    self._config.max_attempts_per_reply,
                )
            )
            if admitted_index is None:
                await self._record(
                    reply_id,
                    attempt_index=observed_index,
                    decision=SearchSafetyDecision.LIMIT_REACHED,
                )
                return _PreparedCall(
                    self._limit_response(
                        self._config.max_attempts_per_reply,
                    ),
                )

            arguments = json.loads(tool_call.input)
            if not isinstance(arguments, dict):
                raise ValueError("search_knowledgebase 工具参数必须是 JSON object")
            match_input = build_search_match_input(
                arguments,
                getattr(tool, "input_schema", None),
            )
            exact_signature = build_exact_signature(match_input)
            state = await self._state_store.get(reply_id)
            if state is None:
                raise RuntimeError("精确去重时 reply 状态不存在")

            if self._config.exact_dedup_enabled:
                duplicate_attempt = self._find_executed_duplicate(
                    state,
                    exact_signature,
                )
                if duplicate_attempt is not None:
                    await self._record(
                        reply_id,
                        attempt_index=admitted_index,
                        decision=SearchSafetyDecision.DUPLICATE_REUSED,
                        exact_signature=exact_signature,
                        matched_attempt_index=duplicate_attempt,
                    )
                    return _PreparedCall(
                        self._duplicate_response(duplicate_attempt),
                    )

            similar_attempt = None
            if self._config.similar_query_guard_enabled:
                similar_attempt = self._find_similar(state, match_input)
                if (
                    similar_attempt is not None
                    and not self._config.similar_query_observe_only
                ):
                    await self._record(
                        reply_id,
                        attempt_index=admitted_index,
                        decision=SearchSafetyDecision.SIMILAR_QUERY_BLOCKED,
                        exact_signature=exact_signature,
                        normalized_kb_name=match_input.normalized_kb_name,
                        normalized_query=match_input.normalized_query,
                        effective_filters_signature=(
                            match_input.effective_filters_signature
                        ),
                        matched_attempt_index=similar_attempt,
                    )
                    return _PreparedCall(
                        self._similar_response(similar_attempt),
                    )

            budget_allowed, reserved_tokens = (
                await self._assess_and_reserve_budget(
                    agent,
                    reply_id,
                )
            )
            if not budget_allowed:
                await self._state_store.set_force_finalize(
                    reply_id,
                    "context_budget_reached",
                )
                await self._record(
                    reply_id,
                    attempt_index=admitted_index,
                    decision=SearchSafetyDecision.CONTEXT_BUDGET_REACHED,
                    exact_signature=exact_signature,
                )
                return _PreparedCall(self._budget_response())

            decision = (
                SearchSafetyDecision.SIMILAR_QUERY_BLOCKED
                if similar_attempt is not None
                else SearchSafetyDecision.ALLOWED
            )
            try:
                await self._state_store.mark_mcp_call(reply_id)
                await self._record(
                    reply_id,
                    attempt_index=admitted_index,
                    decision=decision,
                    exact_signature=exact_signature,
                    normalized_kb_name=match_input.normalized_kb_name,
                    normalized_query=match_input.normalized_query,
                    effective_filters_signature=(
                        match_input.effective_filters_signature
                    ),
                    matched_attempt_index=similar_attempt,
                    reserved_tokens=reserved_tokens,
                    mcp_called=True,
                )
            except Exception:
                await self._release_tokens_safely(
                    reply_id,
                    reserved_tokens,
                )
                raise
            return _PreparedCall(None, reserved_tokens)

    async def _release_tokens_safely(
        self,
        reply_id: str,
        reserved_tokens: int,
    ) -> None:
        if reserved_tokens <= 0:
            return
        try:
            await self._state_store.release_tokens(reply_id, reserved_tokens)
        except Exception:
            logger.warning(
                "知识库检索结束后释放在途 token 预留失败",
                exc_info=True,
            )

    async def _fail_safe(
        self,
        reply_id: str | None,
        reserved_tokens: int,
    ) -> ToolResponse:
        logger.warning(
            "知识库检索安全判定异常，已阻止高体积检索并强制收尾",
            exc_info=True,
        )
        if reply_id:
            await self._release_tokens_safely(reply_id, reserved_tokens)
            try:
                await self._state_store.set_force_finalize(
                    reply_id,
                    "search_safety_internal_error",
                )
            except Exception:
                logger.warning(
                    "知识库检索安全异常后设置 force_finalize 失败",
                    exc_info=True,
                )
        return self._budget_response()

    async def on_acting(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[Any, None]],
    ) -> AsyncGenerator[Any, None]:
        """识别目标 capability，执行计数、匹配和预算短路。"""
        tool_call = input_kwargs.get("tool_call")
        if not self._config.enabled or tool_call is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        is_target, tool = await self._resolve_target_tool(
            agent,
            tool_call.name,
        )
        if not is_target:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        reply_id = getattr(getattr(agent, "state", None), "reply_id", None)
        reserved_tokens = 0
        try:
            if not isinstance(reply_id, str) or not reply_id:
                raise ValueError("目标检索缺少有效 reply_id")
            prepared = await self._prepare_target_call(
                agent,
                reply_id,
                tool_call,
                tool,
            )
            reserved_tokens = prepared.reserved_tokens
        except Exception:
            yield await self._fail_safe(reply_id, reserved_tokens)
            return

        if prepared.short_response is not None:
            yield prepared.short_response
            return

        try:
            async for item in next_handler(**input_kwargs):
                yield item
        finally:
            await self._release_tokens_safely(reply_id, reserved_tokens)


__all__ = ["KnowledgeBaseSearchSafetyMiddleware"]
