'''阶段 3A fake KnowledgeBase MCP 与 fake model 端到端测试。'''
from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

from agentscope.agent import ReActConfig
from agentscope.credential import CredentialBase
from agentscope.message import Msg, TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import ToolChoice, Toolkit
from pydantic import BaseModel

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.agents.search_safety.contracts import SearchSafetyDecision
from qwenpaw.agents.search_safety.preflight import ContextWindowBudgetGuard
from qwenpaw.agents.search_safety.state import SearchSafetyStateStore
from qwenpaw.config.config import AgentProfileConfig
from qwenpaw.drivers.adapters.agentscope_tool import DriverCapabilityTool
from qwenpaw.drivers.capabilities import (
    CapabilityExposure,
    DriverCapability,
    DriverInvocation,
    DriverInvocationResult,
    format_capability_id,
)
from qwenpaw.runtime.builder import AgentBuilder


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "kb_name": {"type": "string"},
        "query": {"type": "string"},
        "mode": {"type": "string", "default": "hybrid"},
        "dense_top_k": {"type": "integer", "default": 20},
        "bm25_top_k": {"type": "integer", "default": 20},
        "rerank": {"type": "boolean", "default": True},
        "response_detail": {"type": "string", "default": "compact"},
    },
    "required": ["kb_name", "query"],
}


class FakeKnowledgeBaseMCP:
    '''记录实际 Driver/MCP 调用与并发峰值。'''

    def __init__(self, *, result_size: int = 32) -> None:
        self.result_size = result_size
        self.calls: list[DriverInvocation] = []
        self.active = 0
        self.max_active = 0

    async def invoke(self, invocation: DriverInvocation) -> DriverInvocationResult:
        self.calls.append(invocation)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return DriverInvocationResult(
            ok=True,
            value="E" * self.result_size,
        )


class ScriptedModel(ChatModelBase):
    '''返回预设 reasoning，并记录真实 provider 参数与超限调用。'''

    class Parameters(BaseModel):
        '''fake model 无额外参数。'''

    def __init__(
        self,
        outcomes: list[list[Any]],
        *,
        context_size: int = 1_000_000,
        token_count: int = 1000,
        block_after_tool_result: bool = False,
    ) -> None:
        super().__init__(
            credential=CredentialBase(name="fake"),
            model="fake-search-model",
            parameters=self.Parameters(),
            stream=False,
            max_retries=0,
            context_size=context_size,
        )
        self.outcomes = list(outcomes)
        self.token_count = token_count
        self.block_after_tool_result = block_after_tool_result
        self.provider_calls = 0
        self.provider_overlimit_count = 0
        self.received_tool_choices: list[ToolChoice | None] = []
        self.received_tools: list[list[dict] | None] = []
        self.received_messages: list[list[Msg]] = []

    async def count_tokens(self, messages: Any, tools: Any) -> int:
        del tools
        if self.block_after_tool_result and any(
            isinstance(message, Msg) and message.has_content_blocks("tool_result")
            for message in messages
        ):
            return 30_000
        return self.token_count

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del model_name, kwargs
        self.provider_calls += 1
        self.received_tool_choices.append(tool_choice)
        self.received_tools.append(tools)
        self.received_messages.append(
            [message.model_copy(deep=True) for message in messages],
        )
        safe_limit = self.context_size - 8192 - max(
            4096,
            int(self.context_size * 0.03),
        )
        if self.token_count + 512 > safe_limit:
            self.provider_overlimit_count += 1
        return ChatResponse(content=self.outcomes.pop(0), is_last=True)


def _config() -> AgentProfileConfig:
    '''返回真实默认安全配置、100 次 ReAct 与关闭 compact 的配置。'''
    return AgentProfileConfig(
        id="e2e",
        name="E2E",
        workspace_dir="/tmp/e2e",
    )


def _target_tool(fake_mcp: FakeKnowledgeBaseMCP) -> DriverCapabilityTool:
    '''创建真实 DriverCapabilityTool 身份边界。'''
    capability = DriverCapability(
        capability_id=format_capability_id(
            "mcp",
            "knowledgebase_remote",
            "tool",
            "invoke",
            "search_knowledgebase",
        ),
        driver_name="knowledgebase_remote",
        protocol="mcp",
        kind="tool",
        action="invoke",
        name="search_knowledgebase",
        input_schema=SEARCH_SCHEMA,
        exposure=CapabilityExposure(
            as_tool=True,
            tool_name="KB_REMOTE__ARBITRARY_DISPLAY",
            namespace="display_only",
        ),
    )
    tool = DriverCapabilityTool(capability, fake_mcp.invoke)
    tool.is_concurrency_safe = True
    return tool


def _call(index: int, query: str, **overrides: Any) -> ToolCallBlock:
    '''构造保持 ``response_detail=full`` 的目标调用。'''
    payload = {
        "kb_name": "main",
        "query": query,
        "mode": "hybrid",
        "dense_top_k": 20,
        "bm25_top_k": 20,
        "rerank": True,
        "response_detail": "full",
    }
    payload.update(overrides)
    return ToolCallBlock(
        id=f"search-{index}",
        name="KB_REMOTE__ARBITRARY_DISPLAY",
        input=json.dumps(payload, ensure_ascii=False),
    )


def _build_agent(
    model: ScriptedModel,
    fake_mcp: FakeKnowledgeBaseMCP,
) -> tuple[QwenPawAgent, SearchSafetyStateStore]:
    '''使用阶段 3A builder 装配真实 QwenPawAgent middleware 链。'''
    config = _config()
    toolkit = Toolkit(tools=[_target_tool(fake_mcp)])
    store = SearchSafetyStateStore()
    guard = ContextWindowBudgetGuard(
        config.running.search_knowledgebase_safety,
        agent_config=config,
        model=model,
    )
    middlewares = AgentBuilder._build_middlewares(
        type("Ctx", (), {"app_services": None, "workspace": None})(),
        config,
        toolkit=toolkit,
        model=model,
        search_safety_state_store=store,
        context_window_budget_guard=guard,
    )
    return (
        QwenPawAgent(
            name="E2E",
            model=model,
            system_prompt="system",
            toolkit=toolkit,
            react_config=ReActConfig(max_iters=config.running.max_iters),
            middlewares=middlewares,
            agent_config=config,
            search_safety_state_store=store,
        ),
        store,
    )


async def _run_reply(agent: QwenPawAgent, text: str) -> list[Any]:
    '''完整消费 AgentScope reply/acting/reasoning 工作流。'''
    return [
        event
        async for event in agent._reply(
            inputs=UserMsg(name="user", content=text),
        )
    ]


async def test_full_batch_workflow_force_finalizes_without_eleventh_mcp() -> None:
    queries = [
        "如何配置知识库远程检索服务",
        "如何配置知识库远程检索服务",
        "如何配置知识库远程检索服务？",
        "查询 2025 年知识库安全规则",
        "查询 2026 年知识库安全规则",
        "查询系统 v2.0.1 知识库规则",
        "查询系统 v2.0.2 知识库规则",
        "查询包含附件的知识库规则",
        "查询不包含附件的知识库规则",
        "核对新实体 Alpha 的知识库规则",
        "第 11 个独立问题",
        "第 12 个独立问题",
        "第 13 个独立问题",
        "第 14 个独立问题",
        "第 15 个独立问题",
    ]
    first_reasoning = [_call(index, query) for index, query in enumerate(queries)]
    model = ScriptedModel(
        [
            first_reasoning,
            [
                TextBlock(text="继续第 11-15 次检索"),
                _call(100, "provider 违规追加检索"),
            ],
            [TextBlock(text="基于已有证据的最终回答")],
        ],
    )
    fake_mcp = FakeKnowledgeBaseMCP()
    agent, store = _build_agent(model, fake_mcp)

    events = await _run_reply(agent, "请全面回答")
    reply_id = agent.state.reply_id
    state = await store.get(reply_id)

    assert state is not None
    decisions = Counter(record.decision for record in state.calls)
    assert state.observed_attempt_count == 15
    assert state.admitted_attempt_count == 10
    assert state.mcp_call_count == len(fake_mcp.calls) == 9
    assert decisions[SearchSafetyDecision.DUPLICATE_REUSED] == 1
    assert decisions[SearchSafetyDecision.SIMILAR_QUERY_BLOCKED] == 1
    assert decisions[SearchSafetyDecision.LIMIT_REACHED] == 5
    assert state.force_finalize is True
    assert model.received_tool_choices[-1] is not None
    assert model.received_tool_choices[-1].mode == "none"
    assert model.provider_calls == 3
    assert [
        choice.mode if choice is not None else None
        for choice in model.received_tool_choices
    ] == [None, "none", "none"]
    assert model.received_tools[0]
    assert model.received_tools[1:] == [[], []]
    assert [
        message.role for message in model.received_messages[-1]
    ] == ["system", "user"]
    assert all(
        not message.has_content_blocks(block_type)
        for message in model.received_messages[-1]
        for block_type in ("thinking", "tool_call", "tool_result")
    )
    assert model.provider_overlimit_count == 0
    assert fake_mcp.max_active > 1
    assert all(
        invocation.payload["response_detail"] == "full"
        for invocation in fake_mcp.calls
    )
    final_messages = [event for event in events if isinstance(event, Msg)]
    assert final_messages[-1].get_text_content() == "基于已有证据的最终回答"
    assert "继续第 11-15 次检索" not in "".join(
        message.get_text_content() for message in agent.state.context
    )
    print(
        "完整工作流统计="
        + json.dumps(
            {
                "observed": state.observed_attempt_count,
                "admitted": state.admitted_attempt_count,
                "mcp": state.mcp_call_count,
                "duplicate": decisions[
                    SearchSafetyDecision.DUPLICATE_REUSED
                ],
                "similar": decisions[
                    SearchSafetyDecision.SIMILAR_QUERY_BLOCKED
                ],
                "limit": decisions[SearchSafetyDecision.LIMIT_REACHED],
                "final_tool_choice": model.received_tool_choices[-1].mode,
                "provider_overlimit_count": model.provider_overlimit_count,
                "mcp_max_concurrency": fake_mcp.max_active,
                "final_text": final_messages[-1].get_text_content(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


async def test_final_preflight_blocks_known_overlimit_provider_request() -> None:
    model = ScriptedModel(
        [[TextBlock(text="不应到达 provider")]],
        context_size=100,
        token_count=81,
    )
    fake_mcp = FakeKnowledgeBaseMCP(result_size=100_000)
    agent, _ = _build_agent(model, fake_mcp)

    events = await _run_reply(agent, "预算不足场景")

    assert model.provider_calls == 0
    assert model.provider_overlimit_count == 0
    assert fake_mcp.calls == []
    final_messages = [event for event in events if isinstance(event, Msg)]
    final_text = final_messages[-1].get_text_content()
    assert "系统未发送该请求" in final_text
    assert "/clear" not in final_text


async def test_single_large_result_blocks_only_the_next_provider_call() -> None:
    model = ScriptedModel(
        [
            [_call(0, "单次大结果")],
            [TextBlock(text="已基于安全投影中的既有证据完成回答")],
        ],
        context_size=40_000,
        token_count=1000,
        block_after_tool_result=True,
    )
    fake_mcp = FakeKnowledgeBaseMCP(result_size=100_000)
    agent, store = _build_agent(model, fake_mcp)

    events = await _run_reply(agent, "单次大结果场景")
    state = await store.get(agent.state.reply_id)

    assert state is not None
    assert state.observed_attempt_count == 1
    assert state.admitted_attempt_count == 1
    assert state.mcp_call_count == len(fake_mcp.calls) == 1
    assert model.provider_calls == 2
    assert model.provider_overlimit_count == 0
    assert fake_mcp.calls[0].payload["response_detail"] == "full"
    assert model.received_tool_choices[-1] is not None
    assert model.received_tool_choices[-1].mode == "none"
    assert model.received_tools[-1] == []
    assert all(
        not message.has_content_blocks(block_type)
        for message in model.received_messages[-1]
        for block_type in ("thinking", "tool_call", "tool_result")
    )
    final_messages = [event for event in events if isinstance(event, Msg)]
    assert (
        final_messages[-1].get_text_content()
        == "已基于安全投影中的既有证据完成回答"
    )
    print(
        "大结果工作流统计="
        + json.dumps(
            {
                "observed": state.observed_attempt_count,
                "admitted": state.admitted_attempt_count,
                "mcp": state.mcp_call_count,
                "provider_calls": model.provider_calls,
                "provider_overlimit_count": model.provider_overlimit_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


async def test_session_restore_and_new_reply_are_isolated() -> None:
    model = ScriptedModel(
        [[_call(0, "恢复前查询")], [TextBlock(text="第一轮完成")]],
    )
    fake_mcp = FakeKnowledgeBaseMCP()
    agent, store = _build_agent(model, fake_mcp)
    await _run_reply(agent, "第一轮")
    old_reply_id = agent.state.reply_id
    snapshot = agent.state_dict()
    old_state = await store.get(old_reply_id)
    assert old_state is not None and old_state.mcp_call_count == 1

    restored_model = ScriptedModel([[TextBlock(text="新回复完成")]])
    restored_agent, restored_store = _build_agent(restored_model, fake_mcp)
    restored_agent.load_state_dict(snapshot)
    restored_state = await restored_store.get(old_reply_id)
    assert restored_state is not None and restored_state.mcp_call_count == 1

    await _run_reply(restored_agent, "新一轮问题")
    new_reply_id = restored_agent.state.reply_id
    new_state = await restored_store.get(new_reply_id)
    assert new_reply_id != old_reply_id
    assert new_state is not None
    assert new_state.observed_attempt_count == 0
    assert new_state.mcp_call_count == 0
    assert await restored_store.get(old_reply_id) is None
