# -*- coding: utf-8 -*-
"""OpenAI chat model compatibility wrappers."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncGenerator

from agentscope.message import ToolCallBlock
from agentscope.model import OpenAIChatModel
from agentscope.model._model_response import ChatResponse

from qwenpaw.local_models.tag_parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    parse_tool_calls_from_text,
    text_contains_tool_call_tag,
)


def _battr(block: Any, key: str, default: Any = None) -> Any:
    """Read an attribute from a dict *or* Pydantic block."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _bset(block: Any, key: str, value: Any) -> None:
    """Set an attribute on a dict *or* Pydantic block."""
    if isinstance(block, dict):
        block[key] = value
    else:
        setattr(block, key, value)


class _TaggedToolStreamFilter:
    """跨 streaming delta 隐藏文本形式的工具调用标签。"""

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_tool_call = False

    @staticmethod
    def _partial_prefix_length(text: str, marker: str) -> int:
        """返回 text 尾部与 marker 前缀匹配的最长长度。"""
        maximum = min(len(text), len(marker) - 1)
        for length in range(maximum, 0, -1):
            if marker.startswith(text[-length:]):
                return length
        return 0

    def feed(self, delta: str) -> str:
        """接收一个 delta，仅返回工具标签之外可安全展示的文本。"""
        self._buffer += delta
        visible_parts: list[str] = []

        while self._buffer:
            if self._inside_tool_call:
                end_index = self._buffer.find(TOOL_CALL_END)
                if end_index < 0:
                    keep_length = min(
                        len(self._buffer),
                        len(TOOL_CALL_END) - 1,
                    )
                    self._buffer = self._buffer[-keep_length:]
                    break
                self._buffer = self._buffer[
                    end_index + len(TOOL_CALL_END) :
                ]
                self._inside_tool_call = False
                continue

            start_index = self._buffer.find(TOOL_CALL_START)
            if start_index >= 0:
                visible_parts.append(self._buffer[:start_index])
                self._buffer = self._buffer[
                    start_index + len(TOOL_CALL_START) :
                ]
                self._inside_tool_call = True
                continue

            keep_length = self._partial_prefix_length(
                self._buffer,
                TOOL_CALL_START,
            )
            if keep_length:
                visible_parts.append(self._buffer[:-keep_length])
                self._buffer = self._buffer[-keep_length:]
            else:
                visible_parts.append(self._buffer)
                self._buffer = ""
            break

        return "".join(visible_parts)


def _clean_tagged_text(text: str) -> tuple[str, list[ToolCallBlock]]:
    """将完整文本中的标签工具调用转换为 AgentScope 2.x block。"""
    parsed = parse_tool_calls_from_text(text)
    clean_text = "\n".join(
        part
        for part in (parsed.text_before.strip(), parsed.text_after.strip())
        if part
    )
    tool_calls = [
        ToolCallBlock(
            id=tool_call.id,
            name=tool_call.name,
            input=tool_call.raw_arguments,
        )
        for tool_call in parsed.tool_calls
    ]
    return clean_text, tool_calls


def _clone_with_overrides(obj: Any, **overrides: Any) -> Any:
    """Clone a stream object into a mutable namespace with overrides."""
    data = dict(getattr(obj, "__dict__", {}))
    data.update(overrides)
    return SimpleNamespace(**data)


def _sanitize_tool_call(tool_call: Any) -> Any | None:
    """Normalize a tool call for parser safety, or drop it if unusable."""
    if not hasattr(tool_call, "index"):
        return None

    function = getattr(tool_call, "function", None)
    if function is None:
        return None

    has_name = hasattr(function, "name")
    has_arguments = hasattr(function, "arguments")

    raw_name = getattr(function, "name", "")
    if isinstance(raw_name, str):
        safe_name = raw_name
    elif raw_name is None:
        safe_name = ""
    else:
        safe_name = str(raw_name)

    raw_arguments = getattr(function, "arguments", "")
    if isinstance(raw_arguments, str):
        safe_arguments = raw_arguments
    elif raw_arguments is None:
        safe_arguments = ""
    else:
        try:
            safe_arguments = json.dumps(raw_arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            safe_arguments = str(raw_arguments)

    if (
        has_name
        and has_arguments
        and isinstance(raw_name, str)
        and isinstance(
            raw_arguments,
            str,
        )
    ):
        return tool_call

    safe_function = SimpleNamespace(
        name=safe_name,
        arguments=safe_arguments,
    )
    return _clone_with_overrides(tool_call, function=safe_function)


def _sanitize_chunk(chunk: Any) -> Any:
    """Drop/normalize malformed tool-calls in a streaming chunk."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return chunk

    sanitized_choices: list[Any] = []
    changed = False

    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            sanitized_choices.append(choice)
            continue

        raw_tool_calls = getattr(delta, "tool_calls", None)
        if not raw_tool_calls:
            sanitized_choices.append(choice)
            continue

        choice_changed = False
        sanitized_tool_calls: list[Any] = []
        for tool_call in raw_tool_calls:
            sanitized = _sanitize_tool_call(tool_call)
            if sanitized is not tool_call:
                choice_changed = True
            if sanitized is not None:
                sanitized_tool_calls.append(sanitized)

        if choice_changed:
            changed = True
            sanitized_delta = _clone_with_overrides(
                delta,
                tool_calls=sanitized_tool_calls,
            )
            sanitized_choice = _clone_with_overrides(
                choice,
                delta=sanitized_delta,
            )
            sanitized_choices.append(sanitized_choice)
            continue

        sanitized_choices.append(choice)

    if not changed:
        return chunk
    return _clone_with_overrides(chunk, choices=sanitized_choices)


def _sanitize_stream_item(item: Any) -> Any:
    """Sanitize either plain stream chunks or structured stream items."""
    if hasattr(item, "chunk"):
        chunk = item.chunk
        sanitized_chunk = _sanitize_chunk(chunk)
        if sanitized_chunk is chunk:
            return item
        return _clone_with_overrides(item, chunk=sanitized_chunk)

    return _sanitize_chunk(item)


class _SanitizedStream:
    """Proxy OpenAI async stream that sanitizes each emitted item and
    captures ``extra_content`` from tool-call chunks (used by Gemini
    thinking models to carry ``thought_signature``)."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._ctx_stream: Any | None = None
        self.extra_contents: dict[str, Any] = {}

    async def __aenter__(self) -> "_SanitizedStream":
        self._ctx_stream = await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool | None:
        return await self._stream.__aexit__(exc_type, exc, tb)

    def __aiter__(self) -> "_SanitizedStream":
        return self

    async def __anext__(self) -> Any:
        if self._ctx_stream is None:
            raise StopAsyncIteration
        item = await self._ctx_stream.__anext__()
        self._capture_extra_content(item)
        return _sanitize_stream_item(item)

    def _capture_extra_content(self, item: Any) -> None:
        """Store ``extra_content`` keyed by tool-call id."""
        chunk = getattr(item, "chunk", item)
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            for tc in getattr(delta, "tool_calls", None) or []:
                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    continue
                extra = getattr(tc, "extra_content", None)
                if extra is None:
                    model_extra = getattr(tc, "model_extra", None)
                    if isinstance(model_extra, dict):
                        extra = model_extra.get("extra_content")
                if extra:
                    self.extra_contents[tc_id] = extra


# JSON Schema keywords whose value is itself a schema.
_SINGLE_SCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    },
)
# Keywords whose value is an array of schemas.
_ARRAY_SCHEMA_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "prefixItems"},
)
# Keywords whose value is an object whose values are schemas.
_MAP_SCHEMA_KEYWORDS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    },
)


# pylint: disable=too-many-branches
def _sanitize_boolean_schemas(schema: Any) -> Any:
    """Position-aware sanitizer for boolean JSON Schema values.

    JSON Schema uses booleans in two distinct ways:

    1. **Boolean schemas** — at a position where a schema is expected,
       ``true`` means "accept anything" and ``false`` means "reject
       everything".  Legal per spec but rejected by strict LLM providers
       (DeepSeek V4, OpenAI) that require an object schema.  We convert::

           true  → {}
           false → {"not": {}}

    2. **Boolean-valued keywords** — annotations like ``nullable``,
       ``deprecated``, ``readOnly``, ``writeOnly``, ``uniqueItems``,
       draft-04 ``exclusiveMinimum`` / ``exclusiveMaximum``.  These MUST
       remain booleans; providers validate them as ``type: boolean``.

    This walker recurses only into known schema-positions, so boolean
    annotations on ordinary keywords pass through unchanged.

    Special-cases retained:
    - ``additionalProperties: true``  → removed (JSON Schema default;
      explicit form rejected by some strict validators).
    - ``required: <bool>`` inside a property definition → removed
      (malformed; real JSON Schema uses ``required: ["field"]`` on the
      parent object).
    """
    if schema is True:
        return {}
    if schema is False:
        return {"not": {}}
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        # Strip special-cases intercepted before the keyword dispatch:
        # `additionalProperties: False` / `: <object>` still fall through
        # to the `_SINGLE_SCHEMA_KEYWORDS` branch below.
        if key == "additionalProperties" and value is True:
            continue
        if key == "required" and isinstance(value, bool):
            continue

        if key in _SINGLE_SCHEMA_KEYWORDS:
            if key == "items" and isinstance(value, list):
                # draft-07 tuple form
                result[key] = [_sanitize_boolean_schemas(v) for v in value]
            else:
                result[key] = _sanitize_boolean_schemas(value)
        elif key in _ARRAY_SCHEMA_KEYWORDS:
            if isinstance(value, list):
                result[key] = [_sanitize_boolean_schemas(v) for v in value]
            else:
                result[key] = value
        elif key in _MAP_SCHEMA_KEYWORDS:
            if isinstance(value, dict):
                result[key] = {
                    k: _sanitize_boolean_schemas(v) for k, v in value.items()
                }
            else:
                result[key] = value
        elif key == "dependencies" and isinstance(value, dict):
            # draft-07: value per key may be a schema or a string array.
            result[key] = {
                k: (
                    _sanitize_boolean_schemas(v)
                    if isinstance(v, (dict, bool))
                    else v
                )
                for k, v in value.items()
            }
        else:
            result[key] = value
    return result


def _collect_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Collect all named type definitions from a JSON Schema root.

    Supports both ``$defs`` (JSON Schema draft-2019+) and the legacy
    ``definitions`` keyword (draft-04/06/07).
    """
    defs: dict[str, Any] = {}
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema["$defs"])
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema["definitions"])
    return defs


def _resolve_local_ref(
    ref: str,
    defs: dict[str, Any],
) -> Any | None:
    """Resolve a local ``$ref`` of the form ``#/$defs/Name``.

    Returns the referenced schema dict, or ``None`` if *ref* is not a
    resolvable local reference.
    """
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
        return defs.get(parts[1])
    return None


def _inline_schema_refs(
    node: Any,
    defs: dict[str, Any],
    _resolving: frozenset,
) -> Any:
    """Recursively inline ``$ref`` nodes using the provided *defs* mapping.

    Inner recursive worker for :func:`_expand_schema_refs`.
    """
    if isinstance(node, list):
        return [_inline_schema_refs(item, defs, _resolving) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in _resolving:
            # Circular reference — break the cycle with an empty schema.
            return {}
        resolved = _resolve_local_ref(ref, defs)
        if resolved is not None:
            # Merge any sibling annotations (e.g. description) into the
            # resolved schema, then recurse to handle nested refs.
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            merged = {**resolved, **siblings}
            return _inline_schema_refs(merged, defs, _resolving | {ref})
        # External or unresolvable $ref — fall through and keep as-is.

    # Recurse into all values; drop $defs / definitions from the output
    # because all references have been resolved inline.
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "definitions"):
            continue
        result[key] = _inline_schema_refs(value, defs, _resolving)
    return result


def _expand_schema_refs(schema: Any) -> Any:
    """Inline all local ``$ref`` references in a JSON Schema.

    Some models (e.g. GLM-5.x via OpenCode Go) cannot process ``$ref`` /
    ``$defs`` patterns in tool parameter schemas.  When Pydantic generates
    schemas for complex nested types it emits a ``$defs`` block and refers to
    it with ``{"$ref": "#/$defs/TypeName"}``.  This function resolves every
    such reference by substituting the full definition inline, then drops the
    ``$defs`` / ``definitions`` sections so the output is a flat, self-
    contained schema that all providers can consume.

    Circular references are detected and replaced with an empty schema
    ``{}`` to avoid infinite recursion.

    Only local references of the form ``#/$defs/<name>`` or
    ``#/definitions/<name>`` are expanded; external ``$ref`` URLs are left
    unchanged.
    """
    if not isinstance(schema, dict):
        return schema
    defs = _collect_defs(schema)
    if not defs:
        # Fast-path: nothing to expand.
        return schema
    return _inline_schema_refs(schema, defs, frozenset())


def _sanitize_tool_schemas(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sanitize tool function schemas to be compatible with strict providers.

    Applies two passes over each tool's ``parameters`` schema:

    1. **$ref / $defs expansion** — inlines all local ``$ref`` references so
       that models which do not support ``$defs`` (e.g. GLM-5.x) receive a
       flat, self-contained schema.
    2. **Boolean schema sanitization** — replaces boolean JSON Schema values
       (``true`` / ``false``) that strict providers like DeepSeek V4 reject.
    """
    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            sanitized.append(tool)
            continue
        params = func.get("parameters")
        if not isinstance(params, dict):
            sanitized.append(tool)
            continue
        sanitized_params = _sanitize_boolean_schemas(
            _expand_schema_refs(params),
        )
        sanitized.append(
            {**tool, "function": {**func, "parameters": sanitized_params}},
        )
    return sanitized


class OpenAIChatModelCompat(OpenAIChatModel):
    """OpenAIChatModel with robust parsing for malformed tool-call chunks
    and transparent ``extra_content`` (Gemini thought_signature) relay.

    Accepts two extra constructor kwargs that ``OpenAIChatModel`` does not:

    * ``default_headers`` — injected as ``extra_headers`` on every API call
      (used for DashScope tracking headers, etc.).
    * ``extra_generate_kwargs`` — merged into every ``_call_api`` invocation
      (provider-level ``generate_kwargs`` that don't map to ``Parameters``).
    """

    def __init__(
        self,
        *,
        default_headers: dict[str, str] | None = None,
        extra_generate_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._default_headers = default_headers
        self._extra_generate_kwargs = extra_generate_kwargs or {}
        super().__init__(**kwargs)

    async def _call_api(
        self,
        model_name: str,
        messages: Any,
        tools: list[dict] | None = None,
        tool_choice: Any | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        merged = {**self._extra_generate_kwargs, **generate_kwargs}
        if self._default_headers:
            existing = merged.get("extra_headers") or {}
            merged["extra_headers"] = {**self._default_headers, **existing}
        return await super()._call_api(
            model_name,
            messages,
            tools,
            tool_choice,
            **merged,
        )

    def _format_tools(
        self,
        tools: list[dict] | None,
        tool_choice: Any | None,
    ) -> tuple[list[dict] | None, Any]:
        """Sanitize boolean sub-schemas before forwarding to base.

        Some MCP servers declare parameters using JSON Schema boolean values
        (e.g. ``additionalProperties: true``, ``items: true``) which are valid
        per spec but rejected by strict providers such as DeepSeek V4.
        """
        if tools:
            tools = _sanitize_tool_schemas(tools)
        return super()._format_tools(tools, tool_choice)

    # pylint: disable=too-many-branches, too-many-statements
    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        sanitized_response = _SanitizedStream(response)
        text_stream_filter = _TaggedToolStreamFilter()
        thinking_stream_filter = _TaggedToolStreamFilter()

        async for parsed in super()._parse_stream_response(
            start_datetime=start_datetime,
            response=sanitized_response,
        ):
            # Filter out malformed tool_use blocks (null id or empty name)
            # emitted by some OpenAI-compatible models, to prevent bad entries
            # from being persisted into session history (issue #4185).
            _tool_types = ("tool_use", "tool_call")

            parsed.content = [
                b
                for b in parsed.content
                if not (
                    (
                        b.get("type")
                        if isinstance(b, dict)
                        else getattr(b, "type", None)
                    )
                    in _tool_types
                    and (
                        not isinstance(
                            b.get("id")
                            if isinstance(b, dict)
                            else getattr(b, "id", None),
                            str,
                        )
                        or not (
                            b.get("name")
                            if isinstance(b, dict)
                            else getattr(b, "name", None)
                        )
                    )
                )
            ]

            if sanitized_response.extra_contents:
                for block in parsed.content:
                    btype = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if btype not in _tool_types:
                        continue
                    tool_id = (
                        block.get("id")
                        if isinstance(block, dict)
                        else getattr(block, "id", None)
                    )
                    if not isinstance(tool_id, str):
                        continue
                    ec = sanitized_response.extra_contents.get(tool_id)
                    if ec:
                        if isinstance(block, dict):
                            block["extra_content"] = ec
                        else:
                            block.extra_content = ec

            has_tool_use = any(
                (
                    b.get("type")
                    if isinstance(b, dict)
                    else getattr(b, "type", None)
                )
                in _tool_types
                for b in parsed.content
            )

            if not parsed.is_last:
                filtered_content: list[Any] = []
                for block in parsed.content:
                    block_type = _battr(block, "type")
                    if block_type == "text":
                        visible_text = text_stream_filter.feed(
                            _battr(block, "text") or "",
                        )
                        if visible_text:
                            _bset(block, "text", visible_text)
                            filtered_content.append(block)
                        continue
                    if block_type == "thinking":
                        visible_thinking = thinking_stream_filter.feed(
                            _battr(block, "thinking") or "",
                        )
                        if visible_thinking:
                            _bset(block, "thinking", visible_thinking)
                            filtered_content.append(block)
                        continue
                    filtered_content.append(block)
                parsed.content = filtered_content
                yield parsed
                continue

            if not has_tool_use:
                final_content: list[Any] = []
                tagged_tool_calls: list[ToolCallBlock] = []
                for block in parsed.content:
                    block_type = _battr(block, "type")
                    field_name = (
                        "thinking"
                        if block_type == "thinking"
                        else "text"
                        if block_type == "text"
                        else None
                    )
                    if field_name is None:
                        final_content.append(block)
                        continue
                    raw_text = _battr(block, field_name) or ""
                    if not text_contains_tool_call_tag(raw_text):
                        final_content.append(block)
                        continue
                    clean_text, converted_calls = _clean_tagged_text(raw_text)
                    tagged_tool_calls.extend(converted_calls)
                    if clean_text:
                        _bset(block, field_name, clean_text)
                        final_content.append(block)

                parsed.content = final_content + tagged_tool_calls

            yield parsed
