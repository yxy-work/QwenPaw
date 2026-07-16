# -*- coding: utf-8 -*-
"""上下文窗口预算 guard 与模型预检 middleware 的永久测试。"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.agent import Agent
from agentscope.message import TextBlock, UserMsg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse

from qwenpaw.agents.search_safety import ContextWindowPreflightBlocked
from qwenpaw.agents.search_safety.preflight import (
    ContextWindowBudgetGuard,
    ContextWindowPreflightMiddleware,
)
from qwenpaw.config.config import SearchKnowledgeBaseSafetyConfig


class CountingModel:
    """返回可控 token 数并记录真实计数参数。"""

    def __init__(
        self,
        token_count: int = 0,
        *,
        context_size: int = 100,
        max_tokens: int | None = 10,
        error: Exception | None = None,
    ) -> None:
        self.token_count = token_count
        self.context_size = context_size
        self.parameters = SimpleNamespace(max_tokens=max_tokens)
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    async def count_tokens(self, messages: Any, tools: Any) -> int:
        self.calls.append((messages, tools))
        if self.error is not None:
            raise self.error
        return self.token_count


def safety_config(**overrides: Any) -> SearchKnowledgeBaseSafetyConfig:
    """构造便于边界测试的小窗口配置。"""
    values = {
        "default_reserved_completion_tokens": 10,
        "safety_margin_ratio": 0.10,
        "minimum_safety_margin_tokens": 10,
        "tool_message_overhead_tokens": 0,
    }
    values.update(overrides)
    return SearchKnowledgeBaseSafetyConfig(**values)


def fallback_agent_config(divisor: float = 4) -> Any:
    """构造只包含现有 QwenPaw bytes 估算配置的 agent config。"""
    return SimpleNamespace(
        id="fallback-agent",
        active_model=None,
        running=SimpleNamespace(
            light_context_config=SimpleNamespace(
                token_count_estimate_divisor=divisor,
                context_compact_config=SimpleNamespace(enabled=False),
                tool_result_pruning_config=SimpleNamespace(
                    pruning_old_msg_max_bytes=3000,
                    pruning_recent_msg_max_bytes=50000,
                ),
            ),
        ),
    )


def model_input(model: CountingModel, *, tools: Any = None) -> dict[str, Any]:
    """构造 AgentScope 2.0.2 的真实 ``on_model_call`` 输入形状。"""
    return {
        "current_model": model,
        "messages": [UserMsg(name="user", content="hello")],
        "tools": tools or [],
        "tool_choice": None,
    }


async def test_exact_counter_allows_limit_and_blocks_next_token() -> None:
    at_limit_model = CountingModel(token_count=80)
    over_limit_model = CountingModel(token_count=81)
    guard = ContextWindowBudgetGuard(safety_config())

    allowed = await guard.assess_model_request(
        input_kwargs=model_input(at_limit_model),
    )
    blocked = await guard.assess_model_request(
        input_kwargs=model_input(over_limit_model),
    )

    assert allowed.allowed is True
    assert allowed.projected_input_tokens == allowed.safe_input_limit == 80
    assert blocked.allowed is False
    assert blocked.projected_input_tokens == 81
    assert blocked.safe_input_limit == 80
    assert allowed.estimator == "model_count_tokens"


async def test_search_preflight_includes_inflight_result_and_overhead() -> None:
    model = CountingModel(context_size=200, max_tokens=20)
    guard = ContextWindowBudgetGuard(
        safety_config(
            safety_margin_ratio=0.05,
            minimum_safety_margin_tokens=10,
            tool_message_overhead_tokens=7,
        ),
        model=model,
    )

    assessment = await guard.assess_search_preflight(
        current_input_tokens=100,
        inflight_reserved_tokens=20,
        projected_search_result_tokens=43,
    )

    assert assessment.projected_input_tokens == 170
    assert assessment.safe_input_limit == 170
    assert assessment.allowed is True


async def test_actual_request_completion_overrides_model_and_fallback() -> None:
    model = CountingModel(token_count=65, context_size=100, max_tokens=20)
    guard = ContextWindowBudgetGuard(
        safety_config(default_reserved_completion_tokens=30),
    )
    input_kwargs = model_input(model)
    input_kwargs["max_output_tokens"] = 5

    assessment = await guard.assess_model_request(input_kwargs=input_kwargs)

    assert assessment.safe_input_limit == 85
    assert assessment.allowed is True


async def test_model_counter_receives_messages_and_tool_schema() -> None:
    tools = [{"type": "function", "function": {"name": "search"}}]
    model = CountingModel(token_count=10)
    guard = ContextWindowBudgetGuard(safety_config())
    kwargs = model_input(model, tools=tools)

    await guard.assess_model_request(input_kwargs=kwargs)

    assert model.calls == [(kwargs["messages"], tools)]


async def test_tool_schema_and_wrapper_overhead_affect_final_boundary() -> None:
    class SchemaAwareModel(CountingModel):
        async def count_tokens(self, messages: Any, tools: Any) -> int:
            self.calls.append((messages, tools))
            return 76 + len(tools or [])

    model = SchemaAwareModel(context_size=100, max_tokens=10)
    guard = ContextWindowBudgetGuard(
        safety_config(tool_message_overhead_tokens=4),
    )

    allowed = await guard.assess_model_request(
        input_kwargs=model_input(model),
    )
    blocked = await guard.assess_model_request(
        input_kwargs=model_input(
            model,
            tools=[{"type": "function", "function": {"name": "search"}}],
        ),
    )

    assert allowed.projected_input_tokens == allowed.safe_input_limit == 80
    assert allowed.allowed is True
    assert blocked.projected_input_tokens == 81
    assert blocked.allowed is False


async def test_concurrent_inflight_assessments_remain_input_isolated() -> None:
    guard = ContextWindowBudgetGuard(
        safety_config(
            safety_margin_ratio=0,
            minimum_safety_margin_tokens=0,
            tool_message_overhead_tokens=0,
        ),
        model=CountingModel(context_size=100, max_tokens=10),
    )

    low, high = await asyncio.gather(
        guard.assess_search_preflight(
            current_input_tokens=40,
            projected_search_result_tokens=20,
            inflight_reserved_tokens=30,
        ),
        guard.assess_search_preflight(
            current_input_tokens=40,
            projected_search_result_tokens=20,
            inflight_reserved_tokens=31,
        ),
    )

    assert low.projected_input_tokens == low.safe_input_limit == 90
    assert low.allowed is True
    assert high.projected_input_tokens == 91
    assert high.allowed is False


async def test_count_failure_uses_qwenpaw_byte_fallback_with_larger_margin() -> None:
    model = CountingModel(
        context_size=1000,
        max_tokens=10,
        error=RuntimeError("tokenizer unavailable"),
    )
    guard = ContextWindowBudgetGuard(
        safety_config(
            safety_margin_ratio=0,
            minimum_safety_margin_tokens=0,
        ),
        agent_config=fallback_agent_config(),
    )

    assessment = await guard.assess_model_request(
        input_kwargs=model_input(model),
    )

    assert assessment.allowed is True
    assert assessment.estimator == "qwenpaw_byte_estimate_x1.25"
    assert assessment.projected_input_tokens > 0


async def test_unsafe_fallback_blocks_without_calling_provider() -> None:
    model = CountingModel(
        context_size=100,
        max_tokens=10,
        error=RuntimeError("tokenizer unavailable"),
    )
    guard = ContextWindowBudgetGuard(
        safety_config(),
        agent_config=fallback_agent_config(divisor=2),
    )
    middleware = ContextWindowPreflightMiddleware(guard)
    provider_calls = 0

    async def provider(**kwargs: Any) -> ChatResponse:
        nonlocal provider_calls
        del kwargs
        provider_calls += 1
        raise AssertionError("不安全请求不得进入 provider")

    kwargs = model_input(model)
    kwargs["messages"] = [UserMsg(name="user", content="x" * 1000)]
    with pytest.raises(ContextWindowPreflightBlocked) as exc_info:
        await middleware.on_model_call(object(), kwargs, provider)

    assert exc_info.value.assessment.estimator.startswith(
        "qwenpaw_byte_estimate",
    )
    assert provider_calls == 0


async def test_all_token_estimators_fail_closed_without_provider_call() -> None:
    model = CountingModel(
        context_size=100,
        error=RuntimeError("tokenizer unavailable"),
    )
    guard = ContextWindowBudgetGuard(safety_config())
    middleware = ContextWindowPreflightMiddleware(guard)
    provider_calls = 0

    async def provider(**kwargs: Any) -> ChatResponse:
        nonlocal provider_calls
        del kwargs
        provider_calls += 1
        raise AssertionError("计数全失败时不得进入 provider")

    with pytest.raises(ContextWindowPreflightBlocked) as exc_info:
        await middleware.on_model_call(
            object(),
            model_input(model),
            provider,
        )

    assert exc_info.value.assessment.reason == "token_estimation_unavailable"
    assert exc_info.value.assessment.projected_input_tokens == 81
    assert exc_info.value.assessment.safe_input_limit == 80
    assert provider_calls == 0


async def test_default_completion_reservation_is_used_as_last_fallback() -> None:
    model = CountingModel(
        token_count=100,
        context_size=10000,
        max_tokens=None,
    )
    guard = ContextWindowBudgetGuard(
        safety_config(
            default_reserved_completion_tokens=8192,
            safety_margin_ratio=0,
            minimum_safety_margin_tokens=100,
        ),
    )

    assessment = await guard.assess_model_request(
        input_kwargs=model_input(model),
    )

    assert assessment.safe_input_limit == 1708
    assert assessment.allowed is True


async def test_middleware_uses_real_signature_and_calls_provider_when_safe() -> None:
    model = CountingModel(token_count=10)
    guard = ContextWindowBudgetGuard(safety_config())
    middleware = ContextWindowPreflightMiddleware(guard)
    provider_calls = 0

    async def provider(**kwargs: Any) -> str:
        nonlocal provider_calls
        provider_calls += 1
        assert kwargs["current_model"] is model
        return "safe"

    result = await middleware.on_model_call(
        object(),
        model_input(model),
        provider,
    )

    implementation = inspect.signature(type(middleware).on_model_call)
    contract = inspect.signature(MiddlewareBase.on_model_call)
    assert list(implementation.parameters) == list(contract.parameters)
    assert inspect.iscoroutinefunction(type(middleware).on_model_call)
    assert isinstance(middleware, MiddlewareBase)
    assert result == "safe"
    assert provider_calls == 1


async def test_real_agentscope_chain_blocks_before_fake_provider() -> None:
    class ProviderCountingModel(ChatModelBase):
        def __init__(self) -> None:
            super().__init__(
                credential=None,
                model="blocked-provider",
                parameters=ChatModelBase.Parameters(),
                stream=False,
                max_retries=0,
                context_size=100,
            )
            self.provider_calls = 0

        async def count_tokens(self, messages: Any, tools: Any) -> int:
            del messages, tools
            return 81

        async def _call_api(
            self,
            model_name: str,
            messages: Any,
            tools: Any = None,
            tool_choice: Any = None,
            **kwargs: Any,
        ) -> ChatResponse:
            del model_name, messages, tools, tool_choice, kwargs
            self.provider_calls += 1
            return ChatResponse(
                content=[TextBlock(text="不应到达")],
                is_last=True,
            )

    model = ProviderCountingModel()
    middleware = ContextWindowPreflightMiddleware(
        ContextWindowBudgetGuard(safety_config()),
    )
    agent = Agent(
        name="blocked-agent",
        system_prompt="system",
        model=model,
        middlewares=[middleware],
    )

    with pytest.raises(ContextWindowPreflightBlocked):
        await agent._call_model(
            [UserMsg(name="user", content="blocked")],
            [],
        )

    assert model.provider_calls == 0


async def test_current_routed_model_context_size_has_priority() -> None:
    injected = CountingModel(token_count=10, context_size=500)
    routed = CountingModel(token_count=10, context_size=120)
    guard = ContextWindowBudgetGuard(safety_config(), model=injected)

    assessment = await guard.assess_model_request(
        input_kwargs=model_input(routed),
    )

    assert assessment.safe_input_limit == 98


async def test_agent_model_config_helper_resolves_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CountingModel(token_count=10, context_size=0)
    agent_config = fallback_agent_config()
    monkeypatch.setattr(
        "qwenpaw.agents.search_safety.preflight.get_model_max_input_length",
        lambda config: 160 if config is agent_config else 0,
    )
    guard = ContextWindowBudgetGuard(
        safety_config(),
        agent_config=agent_config,
    )

    assessment = await guard.assess_model_request(
        input_kwargs=model_input(model),
    )

    assert assessment.safe_input_limit == 134


def test_byte_tool_limit_is_converted_conservatively_without_mutation() -> None:
    agent_config = fallback_agent_config(divisor=4)
    pruning = (
        agent_config.running.light_context_config.tool_result_pruning_config
    )
    before = vars(pruning).copy()
    guard = ContextWindowBudgetGuard(
        safety_config(),
        agent_config=agent_config,
    )

    projected = guard.estimate_projected_tool_result_tokens()

    assert projected >= 20000
    assert vars(pruning) == before


def test_guard_does_not_mutate_unrelated_runtime_controls() -> None:
    agent_config = fallback_agent_config()
    agent_config.running.max_iters = 100
    agent_config.running.retrieval = SimpleNamespace(
        dense_top_k=20,
        bm25_top_k=20,
        response_detail="full",
    )
    before = repr(agent_config)

    ContextWindowBudgetGuard(safety_config(), agent_config=agent_config)

    assert repr(agent_config) == before
    assert (
        agent_config.running.light_context_config.context_compact_config.enabled
        is False
    )
