# -*- coding: utf-8 -*-
"""阶段 2B 知识库检索安全中间件永久回归测试。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import TextBlock, ToolCallBlock, ToolResultState
from agentscope.tool import ToolResponse

from qwenpaw.agents.search_safety import (
    ContextBudgetAssessment,
    SearchSafetyDecision,
    SearchSafetyStateStore,
)
from qwenpaw.agents.search_safety.matching import (
    build_exact_signature,
    build_search_match_input,
    queries_are_highly_similar,
)
from qwenpaw.agents.search_safety.middleware import (
    KnowledgeBaseSearchSafetyMiddleware,
)
from qwenpaw.config.config import SearchKnowledgeBaseSafetyConfig


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "kb_name": {"type": "string"},
        "query": {"type": "string"},
        "mode": {"type": "string", "default": "hybrid"},
        "final_top_n": {"type": "integer", "default": 8},
        "response_detail": {"type": "string", "default": "compact"},
        "metadata": {"type": "object", "default": {}},
    },
    "required": ["kb_name", "query"],
}


class FakeCapabilityTool:
    """同时提供稳定身份与 AgentScope ``input_schema`` 的 fake 工具。"""

    def __init__(
        self,
        *,
        capability_id: str = "mcp:knowledgebase_remote:search_knowledgebase",
        driver_name: str = "knowledgebase_remote",
        protocol: str = "mcp",
        original_capability_name: str = "search_knowledgebase",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self.capability_id = capability_id
        self.driver_name = driver_name
        self.protocol = protocol
        self.original_capability_name = original_capability_name
        self.input_schema = input_schema or SEARCH_SCHEMA


class FakeToolkit:
    """仅实现中间件需要的异步工具查询。"""

    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools

    async def get_tool(self, name: str) -> Any | None:
        return self._tools.get(name)


class FakeBudgetGuard:
    """按预测 token 与固定上限返回预算判定。"""

    def __init__(self, safe_input_limit: int = 10_000) -> None:
        self.safe_input_limit = safe_input_limit
        self.assessments: list[tuple[int, int, int]] = []

    async def assess_search_preflight(
        self,
        *,
        current_input_tokens: int,
        projected_search_result_tokens: int,
        inflight_reserved_tokens: int = 0,
    ) -> ContextBudgetAssessment:
        projected = (
            current_input_tokens
            + projected_search_result_tokens
            + inflight_reserved_tokens
        )
        self.assessments.append(
            (
                current_input_tokens,
                projected_search_result_tokens,
                inflight_reserved_tokens,
            ),
        )
        return ContextBudgetAssessment(
            allowed=projected <= self.safe_input_limit,
            current_input_tokens=current_input_tokens,
            projected_input_tokens=projected,
            safe_input_limit=self.safe_input_limit,
            estimator="fake",
            reason="测试预算判定",
        )

    async def assess_model_request(
        self,
        *,
        input_kwargs: dict[str, Any],
    ) -> ContextBudgetAssessment:
        del input_kwargs
        raise AssertionError("阶段 2B 不应调用模型请求预检")


def _agent(reply_id: str, tools: dict[str, Any]) -> Any:
    return SimpleNamespace(
        state=SimpleNamespace(reply_id=reply_id),
        toolkit=FakeToolkit(tools),
    )


def _call(
    query: str,
    *,
    call_id: str = "call-1",
    name: str = "kb_tool",
    **arguments: Any,
) -> ToolCallBlock:
    payload = {"kb_name": "main", "query": query, **arguments}
    return ToolCallBlock(
        id=call_id,
        name=name,
        input=json.dumps(payload, ensure_ascii=False),
    )


async def _invoke(
    middleware: KnowledgeBaseSearchSafetyMiddleware,
    agent: Any,
    tool_call: ToolCallBlock,
    downstream_calls: list[ToolCallBlock],
    *,
    entered: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> list[Any]:
    async def downstream(**kwargs: Any) -> AsyncGenerator[ToolResponse, None]:
        downstream_calls.append(kwargs["tool_call"])
        if entered is not None:
            entered.set()
        if release is not None:
            await release.wait()
        yield ToolResponse(
            content=[TextBlock(text="真实 MCP 结果")],
            state=ToolResultState.SUCCESS,
        )

    return [
        item
        async for item in middleware.on_acting(
            agent,
            {"tool_call": tool_call},
            downstream,
        )
    ]


def _middleware(
    target: FakeCapabilityTool,
    *,
    config: SearchKnowledgeBaseSafetyConfig | None = None,
    budget_guard: FakeBudgetGuard | None = None,
    projected_tokens: int = 100,
) -> tuple[KnowledgeBaseSearchSafetyMiddleware, SearchSafetyStateStore]:
    store = SearchSafetyStateStore()
    middleware = KnowledgeBaseSearchSafetyMiddleware(
        state_store=store,
        config=config or SearchKnowledgeBaseSafetyConfig(),
        target_capabilities=[target],
        budget_guard=budget_guard or FakeBudgetGuard(),
        current_input_tokens_getter=lambda _agent: 0,
        projected_search_result_tokens=projected_tokens,
    )
    return middleware, store


def _payload(result: list[Any]) -> dict[str, Any]:
    assert len(result) == 1
    response = result[0]
    assert isinstance(response, ToolResponse)
    assert response.state == ToolResultState.SUCCESS
    assert len(response.content) == 1
    assert isinstance(response.content[0], TextBlock)
    assert len(response.content[0].text) < 420
    return json.loads(response.content[0].text)


def test_matching_normalizes_conservatively_and_resolves_defaults() -> None:
    """规范化保留意图信息，省略默认值与显式默认值签名一致。"""
    implicit = build_search_match_input(
        {"kb_name": " Main ", "query": "  ＡBC  不包括 2026！ "},
        SEARCH_SCHEMA,
    )
    explicit = build_search_match_input(
        {
            "kb_name": "main",
            "query": "abc 不包括 2026！",
            "mode": "hybrid",
            "final_top_n": 8,
            "response_detail": "compact",
            "metadata": {},
        },
        SEARCH_SCHEMA,
    )
    assert implicit.normalized_query == "abc 不包括 2026!"
    assert build_exact_signature(implicit) == build_exact_signature(explicit)


def test_metadata_recursive_order_is_stable_but_response_detail_matters() -> None:
    """metadata 递归 key 顺序不影响签名，输出粒度必须影响签名。"""
    left = build_search_match_input(
        {
            "kb_name": "main",
            "query": "安全策略",
            "metadata": {"b": 2, "a": {"y": 2, "x": 1}},
        },
        SEARCH_SCHEMA,
    )
    right = build_search_match_input(
        {
            "query": "安全策略",
            "kb_name": "main",
            "metadata": {"a": {"x": 1, "y": 2}, "b": 2},
        },
        SEARCH_SCHEMA,
    )
    full = build_search_match_input(
        {
            "kb_name": "main",
            "query": "安全策略",
            "metadata": {"a": {"x": 1, "y": 2}, "b": 2},
            "response_detail": "full",
        },
        SEARCH_SCHEMA,
    )
    assert build_exact_signature(left) == build_exact_signature(right)
    assert build_exact_signature(left) != build_exact_signature(full)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("查询 2025 年安全规则", "查询 2026 年安全规则"),
        ("查询 2026-07-15 安全规则", "查询 2026-07-16 安全规则"),
        ("查询系统 v2.0.1 规则", "查询系统 v2.0.2 规则"),
        ("查询包含附件的规则", "查询不包含附件的规则"),
        ("核对“安全预算”配置", "核对“并发预算”配置"),
    ],
)
def test_similar_guard_preserves_new_intent(first: str, second: str) -> None:
    """数字、日期、版本、否定与精确短语变化均视为新增意图。"""
    assert not queries_are_highly_similar(first, second)


@pytest.mark.asyncio
async def test_exact_duplicate_calls_mcp_once_and_returns_reuse_attempt() -> None:
    """完全重复占用槽位但不重复执行 MCP 或复制旧结果。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target)
    agent = _agent("reply-exact", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []

    first = await _invoke(middleware, agent, _call("安全策略"), downstream)
    second = await _invoke(middleware, agent, _call("安全策略"), downstream)

    assert first[0].content[0].text == "真实 MCP 结果"
    payload = _payload(second)
    assert payload["status"] == "duplicate_reused"
    assert payload["reuse_attempt"] == 1
    assert "真实 MCP 结果" not in json.dumps(payload, ensure_ascii=False)
    state = await store.get("reply-exact")
    assert state is not None
    assert state.observed_attempt_count == state.admitted_attempt_count == 2
    assert state.mcp_call_count == len(downstream) == 1


@pytest.mark.asyncio
async def test_defaults_and_metadata_order_deduplicate() -> None:
    """schema 默认值和 metadata 顺序不能绕过去重。"""
    target = FakeCapabilityTool()
    middleware, _ = _middleware(target)
    agent = _agent("reply-defaults", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    await _invoke(
        middleware,
        agent,
        _call("规则", metadata={"b": 2, "a": {"y": 2, "x": 1}}),
        downstream,
    )
    result = await _invoke(
        middleware,
        agent,
        _call(
            "规则",
            mode="hybrid",
            final_top_n=8,
            response_detail="compact",
            metadata={"a": {"x": 1, "y": 2}, "b": 2},
        ),
        downstream,
    )
    assert _payload(result)["status"] == "duplicate_reused"
    assert len(downstream) == 1


@pytest.mark.asyncio
async def test_response_detail_and_filter_changes_execute_separately() -> None:
    """KB、response_detail 与其他 filter 变化都必须进入 MCP。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target)
    agent = _agent("reply-filters", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    await _invoke(middleware, agent, _call("规则"), downstream)
    await _invoke(
        middleware,
        agent,
        _call("规则", response_detail="full"),
        downstream,
    )
    await _invoke(
        middleware,
        agent,
        _call("规则", mode="dense"),
        downstream,
    )
    await _invoke(
        middleware,
        agent,
        _call("规则", kb_name="other"),
        downstream,
    )
    state = await store.get("reply-filters")
    assert state is not None
    assert len(downstream) == state.mcp_call_count == 4
    assert json.loads(downstream[1].input)["response_detail"] == "full"


@pytest.mark.asyncio
async def test_similar_block_and_observe_only_modes() -> None:
    """block 返回短结果，observe-only 记录候选但仍执行。"""
    target = FakeCapabilityTool()
    blocking_config = SearchKnowledgeBaseSafetyConfig(
        similar_query_observe_only=False,
    )
    blocking, _ = _middleware(target, config=blocking_config)
    blocking_agent = _agent("reply-block", {"kb_tool": target})
    blocking_calls: list[ToolCallBlock] = []
    await _invoke(blocking, blocking_agent, _call("如何配置知识库远程检索服务"), blocking_calls)
    blocked = await _invoke(
        blocking,
        blocking_agent,
        _call("如何配置知识库远程检索服务？"),
        blocking_calls,
    )
    assert _payload(blocked)["status"] == "similar_query_blocked"
    assert len(blocking_calls) == 1

    observing, observing_store = _middleware(target)
    observing_agent = _agent("reply-observe", {"kb_tool": target})
    observing_calls: list[ToolCallBlock] = []
    await _invoke(observing, observing_agent, _call("如何配置知识库远程检索服务"), observing_calls)
    await _invoke(
        observing,
        observing_agent,
        _call("如何配置知识库远程检索服务？"),
        observing_calls,
    )
    state = await observing_store.get("reply-observe")
    assert state is not None
    assert len(observing_calls) == state.mcp_call_count == 2
    assert state.calls[-1].decision == SearchSafetyDecision.SIMILAR_QUERY_BLOCKED
    assert state.calls[-1].mcp_called


@pytest.mark.asyncio
async def test_new_numeric_date_version_negation_and_phrase_intents_execute() -> None:
    """高字符串相似度下的五类新增意图均不得拦截。"""
    target = FakeCapabilityTool()
    pairs = [
        ("查询 2025 年安全规则", "查询 2026 年安全规则"),
        ("查询 2026-07-15 安全规则", "查询 2026-07-16 安全规则"),
        ("查询系统 v2.0.1 规则", "查询系统 v2.0.2 规则"),
        ("查询包含附件的规则", "查询不包含附件的规则"),
        ("核对“安全预算”配置", "核对“并发预算”配置"),
    ]
    for index, (first_query, second_query) in enumerate(pairs):
        middleware, _ = _middleware(
            target,
            config=SearchKnowledgeBaseSafetyConfig(
                similar_query_observe_only=False,
            ),
        )
        agent = _agent(f"reply-new-{index}", {"kb_tool": target})
        downstream: list[ToolCallBlock] = []
        await _invoke(middleware, agent, _call(first_query), downstream)
        await _invoke(middleware, agent, _call(second_query), downstream)
        assert len(downstream) == 2


@pytest.mark.asyncio
async def test_similar_history_survives_store_round_trip() -> None:
    """进程恢复后仍应根据持久化紧凑字段拦截纯措辞改写。"""
    target = FakeCapabilityTool()
    config = SearchKnowledgeBaseSafetyConfig(
        similar_query_observe_only=False,
    )
    original, original_store = _middleware(target, config=config)
    original_agent = _agent("reply-resumed", {"kb_tool": target})
    original_calls: list[ToolCallBlock] = []
    await _invoke(
        original,
        original_agent,
        _call("如何配置知识库远程检索服务"),
        original_calls,
    )
    payload = await original_store.dump("reply-resumed")

    restored_store = SearchSafetyStateStore()
    await restored_store.restore("reply-resumed", payload)
    restored = KnowledgeBaseSearchSafetyMiddleware(
        state_store=restored_store,
        config=config,
        target_capabilities=[target],
        budget_guard=FakeBudgetGuard(),
        current_input_tokens_getter=lambda _agent: 0,
        projected_search_result_tokens=100,
    )
    restored_calls: list[ToolCallBlock] = []
    result = await _invoke(
        restored,
        original_agent,
        _call("如何配置知识库远程检索服务？", call_id="resumed"),
        restored_calls,
    )

    assert _payload(result)["status"] == "similar_query_blocked"
    assert restored_calls == []


@pytest.mark.asyncio
async def test_budget_rejection_is_success_and_forces_finalize() -> None:
    """预算拒绝不进入 MCP，且以成功短结果强制收尾。"""
    target = FakeCapabilityTool()
    guard = FakeBudgetGuard(safe_input_limit=50)
    middleware, store = _middleware(
        target,
        budget_guard=guard,
        projected_tokens=60,
    )
    agent = _agent("reply-budget", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    result = await _invoke(middleware, agent, _call("预算测试"), downstream)
    payload = _payload(result)
    assert payload["status"] == "context_budget_reached"
    assert payload["force_finalize"] is True
    state = await store.get("reply-budget")
    assert state is not None
    assert state.force_finalize
    assert state.observed_attempt_count == state.admitted_attempt_count == 1
    assert state.mcp_call_count == state.inflight_reserved_tokens == 0
    assert downstream == []


@pytest.mark.asyncio
async def test_concurrent_budget_reservation_blocks_second_large_result() -> None:
    """并发预算使用在途预留，不能让两个大结果同时获准。"""
    target = FakeCapabilityTool()
    guard = FakeBudgetGuard(safe_input_limit=100)
    middleware, store = _middleware(
        target,
        budget_guard=guard,
        projected_tokens=60,
    )
    agent = _agent("reply-reserve", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    entered = asyncio.Event()
    release = asyncio.Event()
    first_task = asyncio.create_task(
        _invoke(
            middleware,
            agent,
            _call("问题 1", call_id="reserve-1"),
            downstream,
            entered=entered,
            release=release,
        ),
    )
    await entered.wait()
    second = await _invoke(
        middleware,
        agent,
        _call("问题 2", call_id="reserve-2"),
        downstream,
    )
    assert _payload(second)["status"] == "context_budget_reached"
    release.set()
    await first_task
    state = await store.get("reply-reserve")
    assert state is not None
    assert state.inflight_reserved_tokens == 0
    assert state.mcp_call_count == len(downstream) == 1


@pytest.mark.asyncio
async def test_fifteen_serial_attempts_admit_ten_and_finalize_at_tenth() -> None:
    """串行 15 次只准入 10 次，第 10 次后立即 force finalize。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target)
    agent = _agent("reply-serial-15", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    results = []
    for index in range(15):
        results.append(
            await _invoke(
                middleware,
                agent,
                _call(f"独立问题 {index}", call_id=f"serial-{index}"),
                downstream,
            ),
        )
    state = await store.get("reply-serial-15")
    assert state is not None
    assert state.observed_attempt_count == 15
    assert state.admitted_attempt_count == 10
    assert state.mcp_call_count == len(downstream) == 10
    assert state.force_finalize
    assert [_payload(item)["status"] for item in results[10:]] == [
        "limit_reached",
    ] * 5


@pytest.mark.asyncio
async def test_duplicate_and_similar_shortcuts_still_consume_slots() -> None:
    """重复和相似短路均消耗 admitted 槽位并在第 10 次强制收尾。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(
        target,
        config=SearchKnowledgeBaseSafetyConfig(
            similar_query_observe_only=False,
        ),
    )
    agent = _agent("reply-short-slots", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    await _invoke(middleware, agent, _call("安全预算配置"), downstream)
    for index in range(1, 9):
        await _invoke(
            middleware,
            agent,
            _call("安全预算配置", call_id=f"duplicate-{index}"),
            downstream,
        )
    tenth = await _invoke(
        middleware,
        agent,
        _call("安全预算配置？", call_id="similar-tenth"),
        downstream,
    )
    eleventh = await _invoke(
        middleware,
        agent,
        _call("新增问题 11", call_id="limit-eleventh"),
        downstream,
    )
    state = await store.get("reply-short-slots")
    assert state is not None
    assert state.observed_attempt_count == 11
    assert state.admitted_attempt_count == 10
    assert state.mcp_call_count == len(downstream) == 1
    assert state.force_finalize
    assert _payload(tenth)["status"] == "similar_query_blocked"
    assert _payload(eleventh)["status"] == "limit_reached"


@pytest.mark.asyncio
async def test_fifteen_concurrent_attempts_never_start_eleventh_mcp() -> None:
    """并发 15 次仍由 Store 槽位确保最多 10 次 downstream。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target)
    agent = _agent("reply-concurrent-15", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    results = await asyncio.gather(
        *[
            _invoke(
                middleware,
                agent,
                _call(f"并发问题 {index}", call_id=f"concurrent-{index}"),
                downstream,
            )
            for index in range(15)
        ],
    )
    state = await store.get("reply-concurrent-15")
    assert state is not None
    assert state.observed_attempt_count == 15
    assert state.admitted_attempt_count == 10
    assert state.mcp_call_count == len(downstream) == 10
    assert sum(
        _payload(result).get("status") == "limit_reached"
        for result in results
        if result[0].content[0].text != "真实 MCP 结果"
    ) == 5


@pytest.mark.asyncio
async def test_non_target_tools_and_same_name_from_other_driver_bypass() -> None:
    """普通工具与另一 Driver 的同名 capability 完全旁路。"""
    target = FakeCapabilityTool()
    other_driver = FakeCapabilityTool(
        capability_id="mcp:other:search_knowledgebase",
        driver_name="other",
    )
    ordinary = SimpleNamespace(input_schema={})
    middleware, store = _middleware(target)
    agent = _agent(
        "reply-bypass",
        {
            "ordinary_search_knowledgebase": ordinary,
            "other_same_name": other_driver,
        },
    )
    downstream: list[ToolCallBlock] = []
    first = await _invoke(
        middleware,
        agent,
        ToolCallBlock(
            id="ordinary",
            name="ordinary_search_knowledgebase",
            input="{}",
        ),
        downstream,
    )
    second = await _invoke(
        middleware,
        agent,
        _call("同名工具", name="other_same_name"),
        downstream,
    )
    assert first[0].content[0].text == second[0].content[0].text == "真实 MCP 结果"
    assert len(downstream) == 2
    assert await store.get("reply-bypass") is None


@pytest.mark.asyncio
async def test_disabled_mode_bypasses_and_new_reply_resets_state() -> None:
    """关闭开关完整旁路；活跃 reply 变化时计数和私有匹配历史重置。"""
    target = FakeCapabilityTool()
    disabled, disabled_store = _middleware(
        target,
        config=SearchKnowledgeBaseSafetyConfig(enabled=False),
    )
    disabled_agent = _agent("reply-disabled", {"kb_tool": target})
    disabled_calls: list[ToolCallBlock] = []
    await _invoke(
        disabled,
        disabled_agent,
        _call("相同问题"),
        disabled_calls,
    )
    await _invoke(
        disabled,
        disabled_agent,
        _call("相同问题"),
        disabled_calls,
    )
    assert len(disabled_calls) == 2
    assert await disabled_store.get("reply-disabled") is None

    middleware, store = _middleware(target)
    first_agent = _agent("reply-first", {"kb_tool": target})
    second_agent = _agent("reply-second", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    await _invoke(middleware, first_agent, _call("相同问题"), downstream)
    await _invoke(middleware, second_agent, _call("相同问题"), downstream)
    state = await store.get("reply-second")
    assert state is not None
    assert state.observed_attempt_count == state.admitted_attempt_count == 1
    assert state.mcp_call_count == 1
    assert len(downstream) == 2


@pytest.mark.asyncio
async def test_downstream_error_still_releases_reserved_tokens() -> None:
    """实际 MCP 抛错保持原错误语义，但在途预算必须在 finally 释放。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target, projected_tokens=100)
    agent = _agent("reply-downstream-error", {"kb_tool": target})

    async def failing_downstream(
        **_kwargs: Any,
    ) -> AsyncGenerator[ToolResponse, None]:
        raise RuntimeError("真实 MCP 错误")
        yield ToolResponse()

    with pytest.raises(RuntimeError, match="真实 MCP 错误"):
        async for _item in middleware.on_acting(
            agent,
            {"tool_call": _call("下游错误")},
            failing_downstream,
        ):
            pass

    state = await store.get("reply-downstream-error")
    assert state is not None
    assert state.inflight_reserved_tokens == 0
    assert state.mcp_call_count == 1


@pytest.mark.asyncio
async def test_internal_error_after_token_reservation_releases_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """预算已预留后的内部异常也必须释放 token 并 fail-safe。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target, projected_tokens=100)
    agent = _agent("reply-reservation-error", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []

    async def fail_mark_mcp_call(
        _store: SearchSafetyStateStore,
        _reply_id: str,
    ) -> int:
        raise RuntimeError("测试注入的计数异常")

    monkeypatch.setattr(
        SearchSafetyStateStore,
        "mark_mcp_call",
        fail_mark_mcp_call,
    )
    result = await _invoke(
        middleware,
        agent,
        _call("预留后异常"),
        downstream,
    )
    state = await store.get("reply-reservation-error")
    assert _payload(result)["status"] == "context_budget_reached"
    assert state is not None
    assert state.inflight_reserved_tokens == 0
    assert state.force_finalize
    assert downstream == []


@pytest.mark.asyncio
async def test_internal_decision_error_fails_safe_without_sensitive_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已确认目标后的内部异常阻止高体积调用，用户侧无堆栈与敏感参数。"""
    target = FakeCapabilityTool()
    middleware, store = _middleware(target)
    agent = _agent("reply-failsafe", {"kb_tool": target})
    downstream: list[ToolCallBlock] = []
    malformed = ToolCallBlock(
        id="malformed",
        name="kb_tool",
        input='{"query":"机密检索词","metadata":{"secret":"证据"}',
    )
    warning_messages: list[str] = []

    def capture_warning(message: str, **_kwargs: Any) -> None:
        warning_messages.append(message)

    monkeypatch.setattr(
        "qwenpaw.agents.search_safety.middleware.logger.warning",
        capture_warning,
    )
    result = await _invoke(middleware, agent, malformed, downstream)
    response_text = result[0].content[0].text
    assert _payload(result)["status"] == "context_budget_reached"
    assert "机密检索词" not in response_text
    assert "secret" not in response_text
    assert "Traceback" not in response_text
    assert any("知识库检索安全判定异常" in item for item in warning_messages)
    state = await store.get("reply-failsafe")
    assert state is not None and state.force_finalize
    assert downstream == []
