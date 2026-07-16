# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from agentscope.credential import OpenAICredential
from agentscope.message import AssistantMsg, TextBlock, ToolCallBlock

from qwenpaw.providers.openai_chat_model_compat import (
    OpenAIChatModelCompat,
    _sanitize_tool_call,
)


class CompatHarnessOpenAIChatModel(OpenAIChatModelCompat):
    async def parse_stream_for_test(
        self,
        start_datetime: datetime,
        stream: Any,
    ) -> list[Any]:
        responses = []
        async for response in self._parse_stream_response(
            start_datetime,
            stream,
        ):
            responses.append(response)
        return responses


class FakeAsyncStream:
    def __init__(self, items: list[Any]):
        self._items = items
        self._iter = None

    async def __aenter__(self) -> "FakeAsyncStream":
        self._iter = iter(self._items)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def __aiter__(self) -> "FakeAsyncStream":
        return self

    async def __anext__(self) -> Any:
        assert self._iter is not None
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _make_chunk(tool_calls: list[Any]) -> Any:
    delta = SimpleNamespace(
        reasoning_content=None,
        content=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(usage=None, choices=[choice])


def _make_text_chunk(content: str) -> Any:
    delta = SimpleNamespace(
        reasoning_content=None,
        content=content,
        tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(usage=None, choices=[choice])


def _make_model() -> CompatHarnessOpenAIChatModel:
    return CompatHarnessOpenAIChatModel(
        credential=OpenAICredential(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        ),
        model="dummy",
        stream=True,
    )


async def test_stream_parser_skips_tool_call_without_function() -> None:
    model = _make_model()

    malformed_tool_call = SimpleNamespace(
        index=0,
        id="call_bad",
        function=None,
    )
    none_arguments_tool_call = SimpleNamespace(
        index=1,
        id="call_partial",
        function=SimpleNamespace(name="ping", arguments=None),
    )
    valid_tool_call = SimpleNamespace(
        index=0,
        id="call_ok",
        function=SimpleNamespace(name="ping", arguments='{"x":1}'),
    )

    stream = FakeAsyncStream(
        [
            _make_chunk([malformed_tool_call]),
            _make_chunk([none_arguments_tool_call]),
            _make_chunk([valid_tool_call]),
        ],
    )

    responses = await model.parse_stream_for_test(
        datetime.now(),
        stream,
    )

    assert responses
    tool_blocks = [
        block
        for response in responses
        for block in response.content
        if getattr(block, "type", None) in ("tool_use", "tool_call")
    ]
    assert tool_blocks
    last = tool_blocks[-1]
    assert getattr(last, "name", None) == "ping"
    block_input = getattr(last, "input", None)
    if isinstance(block_input, str):
        block_input = json.loads(block_input)
    assert block_input == {"x": 1}


async def test_text_tag_tool_calls_use_agentscope_block_contract() -> None:
    model = _make_model()
    tagged_call = (
        '<tool_call><function=search><parameter=kb_name>示例库</parameter>'
        '<parameter=query>履约保证金</parameter></function></tool_call>'
    )

    responses = await model.parse_stream_for_test(
        datetime.now(),
        FakeAsyncStream([_make_text_chunk(tagged_call)]),
    )

    final_response = responses[-1]
    assert final_response.is_last is True
    assert len(final_response.content) == 1
    tool_call = final_response.content[0]
    assert isinstance(tool_call, ToolCallBlock)
    assert tool_call.type == "tool_call"
    assert isinstance(tool_call.input, str)
    assert json.loads(tool_call.input) == {
        "kb_name": "示例库",
        "query": "履约保证金",
    }
    assistant_msg = AssistantMsg(name="assistant", content=[tool_call])
    assert isinstance(assistant_msg.content[0], ToolCallBlock)


async def test_split_text_tag_never_streams_raw_tool_markup() -> None:
    model = _make_model()
    stream = FakeAsyncStream(
        [
            _make_text_chunk("检索中。<tool_call>"),
            _make_text_chunk(
                "<function=search><parameter=query>废标条款</parameter>",
            ),
            _make_text_chunk("</function></tool_call>"),
        ],
    )

    responses = await model.parse_stream_for_test(datetime.now(), stream)

    streamed_text = "".join(
        block.text
        for response in responses[:-1]
        for block in response.content
        if isinstance(block, TextBlock)
    )
    assert streamed_text == "检索中。"
    assert "<tool_call>" not in streamed_text
    assert "<function=" not in streamed_text
    assert "<parameter=" not in streamed_text
    final_tool_calls = [
        block
        for block in responses[-1].content
        if isinstance(block, ToolCallBlock)
    ]
    assert len(final_tool_calls) == 1
    assert json.loads(final_tool_calls[0].input) == {"query": "废标条款"}


async def test_split_tag_markers_and_five_calls_remain_valid() -> None:
    model = _make_model()
    chunks = [_make_text_chunk("<tool_")]
    for index in range(5):
        prefix = "call>" if index == 0 else "<tool_call>"
        chunks.append(
            _make_text_chunk(
                f"{prefix}<function=search>"
                f"<parameter=query>检索-{index}</parameter></function>"
                "</tool_",
            ),
        )
        chunks.append(_make_text_chunk("call>"))

    responses = await model.parse_stream_for_test(
        datetime.now(),
        FakeAsyncStream(chunks),
    )

    streamed_text = "".join(
        block.text
        for response in responses[:-1]
        for block in response.content
        if isinstance(block, TextBlock)
    )
    assert streamed_text == ""
    final_tool_calls = [
        block
        for block in responses[-1].content
        if isinstance(block, ToolCallBlock)
    ]
    assert len(final_tool_calls) == 5
    assert len({block.id for block in final_tool_calls}) == 5
    assert [json.loads(block.input)["query"] for block in final_tool_calls] == [
        f"检索-{index}" for index in range(5)
    ]
    assistant_msg = AssistantMsg(
        name="assistant",
        content=final_tool_calls,
    )
    assert all(
        isinstance(block, ToolCallBlock)
        for block in assistant_msg.content
    )


def test_sanitize_tool_call_normalizes_non_string_arguments() -> None:
    none_arguments_tool_call = SimpleNamespace(
        index=0,
        id="call_partial",
        function=SimpleNamespace(name="ping", arguments=None),
    )
    non_string_arguments_tool_call = SimpleNamespace(
        index=1,
        id="call_dict",
        function=SimpleNamespace(name="ping", arguments={"x": 2}),
    )
    missing_arguments_tool_call = SimpleNamespace(
        index=2,
        id="call_missing_args",
        function=SimpleNamespace(name="ping"),
    )
    missing_name_tool_call = SimpleNamespace(
        index=3,
        id="call_missing_name",
        function=SimpleNamespace(arguments={"x": 3}),
    )
    missing_name_and_arguments_tool_call = SimpleNamespace(
        index=4,
        id="call_missing_both",
        function=SimpleNamespace(),
    )

    sanitized_none_arguments = _sanitize_tool_call(none_arguments_tool_call)
    assert sanitized_none_arguments is not None
    assert sanitized_none_arguments.function.name == "ping"
    assert sanitized_none_arguments.function.arguments == ""

    sanitized_non_string_arguments = _sanitize_tool_call(
        non_string_arguments_tool_call,
    )
    assert sanitized_non_string_arguments is not None
    assert sanitized_non_string_arguments.function.name == "ping"
    assert isinstance(sanitized_non_string_arguments.function.arguments, str)
    assert json.loads(sanitized_non_string_arguments.function.arguments) == {
        "x": 2,
    }

    sanitized_missing_arguments = _sanitize_tool_call(
        missing_arguments_tool_call,
    )
    assert sanitized_missing_arguments is not None
    assert sanitized_missing_arguments.function.name == "ping"
    assert sanitized_missing_arguments.function.arguments == ""

    sanitized_missing_name = _sanitize_tool_call(missing_name_tool_call)
    assert sanitized_missing_name is not None
    assert sanitized_missing_name.function.name == ""
    assert isinstance(sanitized_missing_name.function.arguments, str)
    assert json.loads(sanitized_missing_name.function.arguments) == {"x": 3}

    sanitized_missing_name_and_arguments = _sanitize_tool_call(
        missing_name_and_arguments_tool_call,
    )
    assert sanitized_missing_name_and_arguments is not None
    assert sanitized_missing_name_and_arguments.function.name == ""
    assert sanitized_missing_name_and_arguments.function.arguments == ""
