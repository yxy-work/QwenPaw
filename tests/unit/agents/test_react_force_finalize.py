# -*- coding: utf-8 -*-
'''ReAct 强制收尾与检索安全状态持久化的永久回归测试。'''

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agentscope.agent import ReActConfig
from agentscope.credential import CredentialBase
from agentscope.event import TextBlockDeltaEvent
from agentscope.message import (
    Msg,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.state import AgentState
from agentscope.tool import Read, ToolChoice, Toolkit
from pydantic import BaseModel

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.agents.search_safety import (
    ContextBudgetAssessment,
    ContextWindowPreflightBlocked,
    SearchSafetyStateStore,
)


class FakeModel(ChatModelBase):
    '''记录模型调用并按顺序返回响应或抛出异常。'''

    class Parameters(BaseModel):
        '''fake model 无额外参数。'''

    def __init__(
        self,
        outcomes: list[Any],
        *,
        context_size: int = 128_000,
    ) -> None:
        super().__init__(
            credential=CredentialBase(name="fake"),
            model="fake-model",
            parameters=self.Parameters(),
            stream=False,
            max_retries=0,
            context_size=context_size,
        )
        self.outcomes = list(outcomes)
        self.received_tool_choices: list[Any] = []
        self.received_tools: list[list[dict] | None] = []
        self.received_messages: list[list[Msg]] = []

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        '''返回下一个预设结果。'''
        del model_name, kwargs
        self.received_tool_choices.append(tool_choice)
        self.received_tools.append(tools)
        self.received_messages.append(
            [message.model_copy(deep=True) for message in messages],
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ChatResponse(content=outcome, is_last=True)


class AgentScopeValidatingFakeModel(FakeModel):
    '''在返回 fake 响应前执行 AgentScope 正式类型校验。'''

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        '''复用真实 ``_validate_tool_choice`` 后返回预设结果。'''
        self._validate_tool_choice(tool_choice, tools)
        return await super()._call_api(
            model_name,
            messages,
            tools,
            tool_choice,
            **kwargs,
        )


class FakeScrollManager:
    '''固定保存并恢复 scroll 载荷。'''

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"persisted_ids": ["message-1"]}
        self.loaded: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def load_state(self, payload: dict[str, Any]) -> None:
        self.loaded = dict(payload)


class BlockingPreflightMiddleware(MiddlewareBase):
    '''在调用 fake provider 前模拟最终上下文预检阻断。'''

    def __init__(self, error: ContextWindowPreflightBlocked) -> None:
        self.error = error

    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Any,
    ) -> Any:
        '''直接抛出冻结控制流异常，不调用下游 provider。'''
        del agent, input_kwargs, next_handler
        raise self.error


class StateAgentShim:
    '''只提供生产状态持久化方法实际访问的属性。'''

    def __init__(
        self,
        *,
        state: AgentState | None = None,
        store: SearchSafetyStateStore | None = None,
        scroll: FakeScrollManager | None = None,
    ) -> None:
        self.state = state or AgentState()
        self._search_safety_state_store = (
            store or SearchSafetyStateStore()
        )
        self._context_manager = scroll


def make_agent(
    model: FakeModel,
    store: SearchSafetyStateStore,
    *,
    auto_continue: bool = False,
    middlewares: list[Any] | None = None,
    toolkit: Toolkit | None = None,
) -> QwenPawAgent:
    '''构造不访问外部服务的真实 QwenPawAgent。'''
    agent_config = SimpleNamespace(
        language="zh",
        running=SimpleNamespace(
            auto_continue_on_text_only=auto_continue,
            light_context_config=SimpleNamespace(
                context_compact_config=SimpleNamespace(enabled=False),
            ),
        ),
        tools=SimpleNamespace(builtin_tools={}),
    )
    return QwenPawAgent(
        name="QwenPaw",
        model=model,
        system_prompt="test",
        toolkit=toolkit or Toolkit(),
        react_config=ReActConfig(max_iters=4),
        middlewares=middlewares or [],
        agent_config=agent_config,
        search_safety_state_store=store,
    )


async def collect_reasoning(agent: QwenPawAgent) -> list[Any]:
    '''完整消费一次生产 ``_reasoning``。'''
    return [event async for event in agent._reasoning()]


def seed_current_reply_evidence(agent: QwenPawAgent) -> None:
    '''写入生产同型的当前用户问题、工具调用历史与已有证据。'''
    agent.state.context.extend(
        [
            UserMsg(name="user", content="请基于已有材料回答当前问题"),
            UserMsg(
                name="user",
                content="内部 auto-continue hint 不得替代用户问题",
                metadata={"qwenpaw_tag": "auto_continue"},
            ),
            Msg(
                name=agent.name,
                role="assistant",
                content=[
                    ThinkingBlock(thinking="继续规划更多检索"),
                    TextBlock(text="准备继续检索"),
                    ToolCallBlock(
                        id="completed-call",
                        name="KnowledgeBase_Remote__search_knowledgebase",
                        input='{"query":"敏感查询不应进入安全投影"}',
                    ),
                    ToolResultBlock(
                        id="completed-call",
                        name="KnowledgeBase_Remote__search_knowledgebase",
                        output="证据甲：质保比例为合同金额的 10%。",
                        state=ToolResultState.SUCCESS,
                    ),
                ],
            ),
        ],
    )


async def test_force_finalize_reaches_provider_as_none() -> None:
    store = SearchSafetyStateStore()
    model = FakeModel([[TextBlock(text="最终回答")]])
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert [choice.mode for choice in model.received_tool_choices] == ["none"]
    assert any(isinstance(event, Msg) for event in events)
    assert "最终回答" == "".join(
        event.delta
        for event in events
        if isinstance(event, TextBlockDeltaEvent)
    )


async def test_force_finalize_hides_tool_schemas_from_model_request() -> None:
    '''强制收尾请求保留 ``none``，同时不向 provider 发送工具 schema。'''
    store = SearchSafetyStateStore()
    model = FakeModel([[TextBlock(text="最终回答")]])
    toolkit = Toolkit()
    toolkit.tool_groups[0].tools.append(Read())
    agent = make_agent(model, store, toolkit=toolkit)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    await collect_reasoning(agent)

    assert model.received_tool_choices[0].mode == "none"
    assert model.received_tools == [[]]


async def test_force_finalize_uses_agentscope_tool_choice_object() -> None:
    '''强制收尾必须通过真实 AgentScope ``ToolChoice`` 校验。'''
    store = SearchSafetyStateStore()
    model = AgentScopeValidatingFakeModel(
        [[TextBlock(text="最终回答")]],
    )
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert len(model.received_tool_choices) == 1
    assert isinstance(model.received_tool_choices[0], ToolChoice)
    assert model.received_tool_choices[0].mode == "none"
    assert any(isinstance(event, Msg) for event in events)


async def test_media_retry_keeps_force_finalize_none() -> None:
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            ValueError("provider rejected image input"),
            [TextBlock(text="无媒体最终回答")],
        ],
    )
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert [choice.mode for choice in model.received_tool_choices] == [
        "none",
        "none",
    ]
    assert any(isinstance(event, Msg) for event in events)


async def test_force_finalize_text_does_not_auto_continue() -> None:
    store = SearchSafetyStateStore()
    model = FakeModel([[TextBlock(text="最终回答")]])
    agent = make_agent(model, store, auto_continue=True)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert any(isinstance(event, Msg) for event in events)
    assert not any(
        message.metadata.get("qwenpaw_message_tag") == "auto_continue"
        for message in agent.state.context
    )


async def test_normal_text_only_response_still_auto_continues() -> None:
    store = SearchSafetyStateStore()
    model = FakeModel([[TextBlock(text="继续计划")]])
    agent = make_agent(model, store, auto_continue=True)

    events = await collect_reasoning(agent)

    assert not any(isinstance(event, Msg) for event in events)
    assert agent.state.context[-1].role == "user"
    assert "system-hint" in agent.state.context[-1].get_text_content()


async def test_tool_call_returned_under_none_is_never_executable() -> None:
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            [
                ToolCallBlock(
                    id="malicious-call",
                    name="dangerous_tool",
                    input='{"apply":true}',
                ),
            ],
            [
                ToolCallBlock(
                    id="malicious-retry-call",
                    name="dangerous_tool",
                    input='{"apply":true}',
                ),
            ],
        ],
    )
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    final_messages = [event for event in events if isinstance(event, Msg)]
    assert len(final_messages) == 1
    assert "阻止" in final_messages[0].get_text_content()
    assert not any(
        type(event).__name__.startswith("ToolCall")
        for event in events
    )
    assert agent._get_executable_tool_calls() == []
    assert not any(
        message.has_content_blocks("tool_call")
        for message in agent.state.context
    )


async def test_forbidden_tool_call_gets_one_bounded_text_retry() -> None:
    '''provider 首次违约后只重试一次，并交付第二次的纯文本结果。'''
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            [
                TextBlock(text="继续检索"),
                ToolCallBlock(
                    id="forbidden-first",
                    name="dangerous_tool",
                    input='{"apply":true}',
                ),
            ],
            [TextBlock(text="基于已有证据形成的最终回答")],
        ],
    )
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert len(model.received_tool_choices) == 2
    assert all(
        choice.mode == "none" for choice in model.received_tool_choices
    )
    final_messages = [event for event in events if isinstance(event, Msg)]
    assert len(final_messages) == 1
    assert "基于已有证据" in final_messages[0].get_text_content()
    assert "继续检索" not in "".join(
        message.get_text_content() for message in agent.state.context
    )
    assert agent._get_executable_tool_calls() == []


async def test_bounded_retry_uses_plain_safe_projection() -> None:
    '''第二次请求只能包含当前问题与预算内纯文本证据。'''
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            [
                ToolCallBlock(
                    id="forbidden-first",
                    name="dangerous_tool",
                    input="{}",
                ),
            ],
            [TextBlock(text="质保比例为合同金额的 10%。")],
        ],
    )
    agent = make_agent(model, store)
    seed_current_reply_evidence(agent)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    await collect_reasoning(agent)

    assert len(model.received_messages) == 2
    repair_messages = model.received_messages[1]
    assert [message.role for message in repair_messages] == ["system", "user"]
    assert all(
        not message.has_content_blocks(block_type)
        for message in repair_messages
        for block_type in ("thinking", "tool_call", "tool_result")
    )
    repair_text = "\n".join(
        message.get_text_content() for message in repair_messages
    )
    assert "请基于已有材料回答当前问题" in repair_text
    assert "证据甲：质保比例为合同金额的 10%。" in repair_text
    assert "准备继续检索" not in repair_text
    assert "敏感查询不应进入安全投影" not in repair_text
    assert "内部 auto-continue hint" not in repair_text
    assert model.received_tools == [[], []]


async def test_safe_projection_fairly_crops_ten_large_results() -> None:
    '''十个大结果都保留片段，且投影严格落在保守字符预算内。'''
    store = SearchSafetyStateStore()
    model = FakeModel([], context_size=40_000)
    agent = make_agent(model, store)
    results = [
        ToolResultBlock(
            id=f"result-{index}",
            name="KnowledgeBase_Remote__search_knowledgebase",
            output=f"证据-{index}-" + "甲" * 20_000,
            state=ToolResultState.SUCCESS,
        )
        for index in range(1, 11)
    ]
    agent.state.context.extend(
        [
            UserMsg(name="user", content="请综合十次检索结果"),
            Msg(name=agent.name, role="assistant", content=results),
        ],
    )

    projection = agent._build_force_finalize_projection()

    assert [message.role for message in projection] == ["system", "user"]
    projected_text = projection[-1].get_text_content()
    for index in range(1, 11):
        assert f"[检索证据 {index}]" in projected_text
    assert len(projected_text) <= (
        agent._force_finalize_projection_char_budget() + 100
    )
    assert len(projected_text) < 200_000
    assert len(agent._truncate_text("超短预算", 1)) <= 1


async def test_second_forbidden_response_stops_after_single_retry() -> None:
    '''第二次仍违约时必须本地终止，不能形成第三次 provider 调用。'''
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            [
                ToolCallBlock(
                    id="forbidden-first",
                    name="dangerous_tool",
                    input="{}",
                ),
            ],
            [
                ToolCallBlock(
                    id="forbidden-second",
                    name="dangerous_tool",
                    input="{}",
                ),
            ],
        ],
    )
    agent = make_agent(model, store)
    seed_current_reply_evidence(agent)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    assert len(model.received_tool_choices) == 2
    final_messages = [event for event in events if isinstance(event, Msg)]
    assert len(final_messages) == 1
    final_text = final_messages[0].get_text_content()
    assert "证据甲：质保比例为合同金额的 10%。" in final_text
    assert "未执行额外检索" in final_text
    assert "请缩小问题范围后重试" not in final_text
    assert agent._get_executable_tool_calls() == []


async def test_local_terminal_message_emits_visible_text_delta() -> None:
    '''本地终态必须产生 channel 可消费的文本事件，不能只剩 ``Msg``。'''
    store = SearchSafetyStateStore()
    model = FakeModel(
        [
            [
                ToolCallBlock(
                    id="forbidden-first",
                    name="dangerous_tool",
                    input="{}",
                ),
            ],
            [
                ToolCallBlock(
                    id="forbidden-second",
                    name="dangerous_tool",
                    input="{}",
                ),
            ],
        ],
    )
    agent = make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = await collect_reasoning(agent)

    visible_text = "".join(
        event.delta
        for event in events
        if isinstance(event, TextBlockDeltaEvent)
    )
    assert "系统已阻止执行" in visible_text


async def test_preflight_block_returns_local_final_message() -> None:
    assessment = ContextBudgetAssessment(
        allowed=False,
        current_input_tokens=120,
        projected_input_tokens=120,
        safe_input_limit=100,
        estimator="test",
        reason="actual_request_exceeds_safe_limit",
    )
    store = SearchSafetyStateStore()
    model = FakeModel([[TextBlock(text="不应到达 provider")]])
    agent = make_agent(
        model,
        store,
        middlewares=[
            BlockingPreflightMiddleware(
                ContextWindowPreflightBlocked(assessment),
            ),
        ],
    )

    events = await collect_reasoning(agent)

    final_messages = [event for event in events if isinstance(event, Msg)]
    assert len(final_messages) == 1
    text = final_messages[0].get_text_content()
    assert "安全上限" in text
    assert "未发送" in text
    assert "/clear" not in text
    assert "没有可提取的已完成检索结果" in text
    assert model.received_tool_choices == []
    state = await store.get(agent.state.reply_id)
    assert state is not None
    assert state.force_finalize is True
    assert state.force_finalize_reason == "context_budget_reached"
    assert "安全上限" in "".join(
        event.delta
        for event in events
        if isinstance(event, TextBlockDeltaEvent)
    )


async def test_new_state_format_round_trip_is_json_safe() -> None:
    state = AgentState(reply_id="reply-persisted")
    store = SearchSafetyStateStore()
    await store.set_force_finalize(state.reply_id, "context_budget_reached")
    scroll = FakeScrollManager()
    agent = StateAgentShim(state=state, store=store, scroll=scroll)

    snapshot = QwenPawAgent.state_dict(agent)
    snapshot = json.loads(json.dumps(snapshot))
    restored_store = SearchSafetyStateStore()
    restored_scroll = FakeScrollManager()
    restored_agent = StateAgentShim(
        store=restored_store,
        scroll=restored_scroll,
    )
    QwenPawAgent.load_state_dict(restored_agent, snapshot, strict=True)

    restored = await restored_store.get("reply-persisted")
    assert restored is not None
    assert restored.force_finalize is True
    assert restored.force_finalize_reason == "context_budget_reached"
    assert snapshot["scroll"] == scroll.payload
    assert restored_scroll.loaded == scroll.payload


async def test_old_2_0_state_without_safety_field_loads_clean_store() -> None:
    state = AgentState(reply_id="reply-old-2")
    snapshot = {"state": state.model_dump(mode="json")}
    store = SearchSafetyStateStore()
    await store.set_force_finalize("stale-reply", "limit_reached")
    agent = StateAgentShim(store=store)

    QwenPawAgent.load_state_dict(agent, snapshot, strict=True)

    restored = await store.get("reply-old-2")
    assert restored is not None
    assert restored.force_finalize is False
    assert store.active_reply_id == "reply-old-2"


async def test_legacy_1_x_memory_loads_with_clean_safety_store() -> None:
    legacy_message = Msg(
        name="user",
        role="user",
        content=[TextBlock(text="旧消息")],
    )
    snapshot = {
        "memory": {
            "content": [[legacy_message.to_dict(), []]],
            "_compressed_summary": "旧摘要",
        },
    }
    store = SearchSafetyStateStore()
    await store.set_force_finalize("stale-reply", "limit_reached")
    agent = StateAgentShim(store=store)

    QwenPawAgent.load_state_dict(agent, snapshot, strict=True)

    restored = await store.get(agent.state.reply_id)
    assert agent.state.summary == "旧摘要"
    assert len(agent.state.context) == 1
    assert restored is not None
    assert restored.force_finalize is False


async def test_mismatched_restored_reply_discards_force_finalize() -> None:
    current_state = AgentState(reply_id="reply-current")
    stale_store = SearchSafetyStateStore()
    await stale_store.set_force_finalize("reply-stale", "limit_reached")
    stale_agent = StateAgentShim(
        state=AgentState(reply_id="reply-stale"),
        store=stale_store,
    )
    stale_payload = QwenPawAgent.state_dict(stale_agent)[
        "search_knowledgebase_safety"
    ]
    snapshot = {
        "state": current_state.model_dump(mode="json"),
        "search_knowledgebase_safety": stale_payload,
    }
    restored_store = SearchSafetyStateStore()
    restored_agent = StateAgentShim(store=restored_store)

    QwenPawAgent.load_state_dict(restored_agent, snapshot, strict=True)

    restored = await restored_store.get("reply-current")
    assert restored is not None
    assert restored.force_finalize is False
    assert restored_store.active_reply_id == "reply-current"


async def test_new_reply_does_not_inherit_force_finalize() -> None:
    store = SearchSafetyStateStore()
    await store.set_force_finalize("reply-old", "limit_reached")

    new_state = await store.get_or_create("reply-new")

    assert new_state.force_finalize is False
    assert new_state.force_finalize_reason is None
