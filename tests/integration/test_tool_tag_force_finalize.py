# -*- coding: utf-8 -*-
'''文本标签工具调用与强制收尾的真实 AgentScope 集成回归。'''

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from agentscope.agent import ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.event import ReplyStartEvent, TextBlockDeltaEvent
from agentscope.message import Msg, UserMsg
from agentscope.middleware import MiddlewareBase
from agentscope.tool import ToolChoice, Toolkit

from qwenpaw.agents.react_agent import QwenPawAgent
from qwenpaw.agents.search_safety import SearchSafetyStateStore
from qwenpaw.providers.openai_chat_model_compat import OpenAIChatModelCompat
from qwenpaw.runtime.envelope import Envelope
from qwenpaw.schemas import RunStatus


class FakeAsyncStream:
    '''模拟 OpenAI-compatible streaming response。'''

    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self._iterator: Any = None

    async def __aenter__(self) -> "FakeAsyncStream":
        self._iterator = iter(self._items)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False

    def __aiter__(self) -> "FakeAsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._iterator is None:
            raise StopAsyncIteration
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _text_chunk(content: str) -> Any:
    delta = SimpleNamespace(
        reasoning_content=None,
        content=content,
        tool_calls=None,
    )
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=delta)],
    )


class ScriptedTaggedToolModel(OpenAIChatModelCompat):
    '''返回生产同型分片 XML 工具调用的兼容模型。'''

    def __init__(self, *, repair_text: str | None = None) -> None:
        super().__init__(
            credential=OpenAICredential(
                api_key="sk-test",
                base_url="https://api.openai.com/v1",
            ),
            model="tagged-tool-model",
            stream=True,
            max_retries=0,
        )
        self.received_tool_choices: list[ToolChoice | None] = []
        self.received_tools: list[list[dict] | None] = []
        self.received_messages: list[list[Msg]] = []
        self.repair_text = repair_text

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **generate_kwargs: Any,
    ) -> AsyncGenerator:
        del model_name, generate_kwargs
        self.received_tool_choices.append(tool_choice)
        self.received_tools.append(tools)
        self.received_messages.append(
            [message.model_copy(deep=True) for message in messages],
        )
        if self.repair_text is not None and len(self.received_tools) == 2:
            return self._parse_stream_response(
                datetime.now(),
                FakeAsyncStream([_text_chunk(self.repair_text)]),
            )
        chunks: list[Any] = []
        for index in range(5):
            chunks.extend(
                [
                    _text_chunk("<tool_call>"),
                    _text_chunk(
                        "<function=KnowledgeBase_Remote__search_knowledgebase>"
                        "<parameter=kb_name>示例库</parameter>"
                        f"<parameter=query>检索-{index}</parameter>"
                        "<parameter=response_detail>full</parameter>",
                    ),
                    _text_chunk("</function></tool_call>"),
                ],
            )
        return self._parse_stream_response(
            datetime.now(),
            FakeAsyncStream(chunks),
        )


def _make_agent(
    model: ScriptedTaggedToolModel,
    store: SearchSafetyStateStore,
    *,
    middlewares: list[Any] | None = None,
) -> QwenPawAgent:
    agent_config = SimpleNamespace(
        language="zh",
        running=SimpleNamespace(
            auto_continue_on_text_only=False,
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
        toolkit=Toolkit(),
        react_config=ReActConfig(max_iters=4),
        middlewares=middlewares or [],
        agent_config=agent_config,
        search_safety_state_store=store,
    )


async def test_force_finalize_blocks_split_tag_calls_without_stream_leak() -> None:
    store = SearchSafetyStateStore()
    model = ScriptedTaggedToolModel()
    agent = _make_agent(model, store)
    await store.set_force_finalize(agent.state.reply_id, "limit_reached")

    events = [event async for event in agent._reasoning()]

    final_messages = [event for event in events if isinstance(event, Msg)]
    assert len(final_messages) == 1
    assert "阻止" in final_messages[0].get_text_content()
    assert len(model.received_tool_choices) == 2
    assert all(
        choice is not None and choice.mode == "none"
        for choice in model.received_tool_choices
    )
    assert model.received_tools == [[], []]
    assert agent._get_executable_tool_calls() == []
    assert not any(
        message.has_content_blocks("tool_call")
        for message in agent.state.context
    )
    rendered = "".join(
        event.get_text_content()
        for event in events
        if isinstance(event, Msg)
    )
    assert "<tool_call>" not in rendered
    assert "<function=" not in rendered
    assert "<parameter=" not in rendered


class ForceFinalizeOnReplyStartMiddleware(MiddlewareBase):
    '''在公开 reply 流产生新 reply_id 后立即设置强制收尾。'''

    def __init__(self, store: SearchSafetyStateStore) -> None:
        self._store = store

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Any,
    ) -> Any:
        '''把测试状态设置在生产同型的 ``ReplyStartEvent`` 边界。'''
        del agent
        async for event in next_handler(**input_kwargs):
            if isinstance(event, ReplyStartEvent):
                await self._store.set_force_finalize(
                    event.reply_id,
                    "limit_reached",
                )
            yield event


async def test_safe_projection_recovers_tagged_model_to_final_text() -> None:
    '''标签工具违约后以纯文本安全投影自动完成回答。'''
    store = SearchSafetyStateStore()
    model = ScriptedTaggedToolModel(
        repair_text="已在十次检索安全上限内基于现有证据完成回答。",
    )
    middleware = ForceFinalizeOnReplyStartMiddleware(store)
    agent = _make_agent(model, store, middlewares=[middleware])

    public_events = [
        event
        async for event in agent.reply_stream(
            inputs=UserMsg(name="user", content="请总结已有证据"),
        )
    ]

    visible_text = "".join(
        event.delta
        for event in public_events
        if isinstance(event, TextBlockDeltaEvent)
    )
    assert visible_text == "已在十次检索安全上限内基于现有证据完成回答。"
    assert model.received_tools == [[], []]
    assert [
        message.role for message in model.received_messages[-1]
    ] == ["system", "user"]
    assert all(
        not message.has_content_blocks(block_type)
        for message in model.received_messages[-1]
        for block_type in ("thinking", "tool_call", "tool_result")
    )
    assert "<tool_call>" not in visible_text


async def test_local_terminal_reaches_public_reply_stream_and_envelope() -> None:
    '''本地终态经公开流和 Envelope 形成可投递文本而非空 ``Done``。'''
    store = SearchSafetyStateStore()
    model = ScriptedTaggedToolModel()
    middleware = ForceFinalizeOnReplyStartMiddleware(store)
    agent = _make_agent(model, store, middlewares=[middleware])

    public_events = [
        event
        async for event in agent.reply_stream(
            inputs=UserMsg(name="user", content="请完成已有证据总结"),
        )
    ]

    visible_text = "".join(
        event.delta
        for event in public_events
        if isinstance(event, TextBlockDeltaEvent)
    )
    assert "系统已阻止执行" in visible_text
    assert "<tool_call>" not in visible_text
    assert "<function=" not in visible_text
    assert "Done" not in visible_text

    envelope = Envelope(session_id="test-session")
    envelope_events: list[Any] = []
    for event in public_events:
        envelope_events.extend(
            [item async for item in envelope.translate_event(event)],
        )
    envelope_events.extend([item async for item in envelope.finalize()])

    completed_messages = [
        event
        for event in envelope_events
        if getattr(event, "object", None) == "message"
        and getattr(event, "status", None) == RunStatus.Completed
    ]
    assert completed_messages
    rendered_text = "".join(
        getattr(content, "text", "")
        for content in completed_messages[-1].content
    )
    assert "系统已阻止执行" in rendered_text
    assert envelope.response.output
