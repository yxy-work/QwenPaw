# -*- coding: utf-8 -*-
"""验证 DriverCapabilityTool 的稳定、只读 capability 身份。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.message import ToolResultState
from agentscope.permission import PermissionBehavior

from qwenpaw.drivers.adapters.agentscope_tool import DriverCapabilityTool
from qwenpaw.drivers.capabilities import (
    CapabilityExposure,
    DriverCapability,
    DriverInvocationResult,
    format_capability_id,
)
from qwenpaw.drivers.handlers.mcp import _mcp_tool_to_capability


def _capability(
    *,
    driver_name: str = "knowledgebase_remote",
    protocol: str = "mcp",
    original_name: str = "search_knowledgebase",
    display_tool_name: str = (
        "KnowledgeBase_Remote__search_knowledgebase"
    ),
) -> DriverCapability:
    return DriverCapability(
        capability_id=format_capability_id(
            protocol,
            driver_name,
            "tool",
            "invoke",
            original_name,
        ),
        driver_name=driver_name,
        protocol=protocol,
        kind="tool",
        action="invoke",
        name=original_name,
        description="Search KnowledgeBase",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        exposure=CapabilityExposure(
            as_tool=True,
            namespace=display_tool_name.rsplit("__", maxsplit=1)[0],
            tool_name=display_tool_name,
        ),
        metadata={"driver_key": driver_name},
    )


def _tool(capability: DriverCapability) -> DriverCapabilityTool:
    async def invoke(_invocation: Any) -> DriverInvocationResult:
        return DriverInvocationResult(ok=True, value="ok")

    return DriverCapabilityTool(capability, invoke)


def _identity(tool: DriverCapabilityTool) -> tuple[str, str, str]:
    return (
        tool.protocol,
        tool.driver_name,
        tool.original_capability_name,
    )


def test_identity_uses_stable_capability_fields() -> None:
    capability = _capability()
    tool = _tool(capability)

    assert tool.capability_id == capability.capability_id
    assert tool.driver_name == "knowledgebase_remote"
    assert tool.protocol == "mcp"
    assert tool.original_capability_name == "search_knowledgebase"


def test_actual_mcp_capability_keeps_driver_key_separate_from_display_name(
) -> None:
    capability = _mcp_tool_to_capability(
        "knowledgebase_remote",
        SimpleNamespace(
            name="search_knowledgebase",
            description="Search KnowledgeBase",
            inputSchema={"type": "object"},
        ),
        display_name="KnowledgeBase Remote",
    )
    tool = _tool(capability)

    assert tool.name == "KnowledgeBase_Remote__search_knowledgebase"
    assert tool.driver_name == "knowledgebase_remote"
    assert tool.original_capability_name == "search_knowledgebase"


@pytest.mark.parametrize(
    ("attribute_name", "replacement"),
    [
        ("capability_id", "driver://mcp/other/tools/other#invoke"),
        ("driver_name", "other"),
        ("protocol", "other"),
        ("original_capability_name", "other"),
    ],
)
def test_identity_attributes_are_read_only(
    attribute_name: str,
    replacement: str,
) -> None:
    tool = _tool(_capability())

    with pytest.raises(AttributeError):
        setattr(tool, attribute_name, replacement)


def test_display_namespace_and_name_do_not_change_identity() -> None:
    original = _tool(_capability())
    renamed = _tool(
        replace(
            _capability(),
            exposure=CapabilityExposure(
                as_tool=True,
                namespace="renamed_namespace",
                tool_name="renamed_namespace__renamed_tool",
            ),
        ),
    )

    assert original.name != renamed.name
    assert original.capability_id == renamed.capability_id
    assert _identity(original) == _identity(renamed)


def test_identity_distinguishes_same_tool_name_from_another_driver() -> None:
    knowledgebase_tool = _tool(_capability())
    other_driver_tool = _tool(_capability(driver_name="other_remote"))

    assert knowledgebase_tool.original_capability_name == (
        other_driver_tool.original_capability_name
    )
    assert _identity(knowledgebase_tool) != _identity(other_driver_tool)


def test_identity_distinguishes_similar_suffix_tool_name() -> None:
    target_tool = _tool(_capability())
    suffix_tool = _tool(
        _capability(original_name="archive_search_knowledgebase"),
    )

    assert target_tool.driver_name == suffix_tool.driver_name
    assert _identity(target_tool) != _identity(suffix_tool)


def test_identity_distinguishes_ordinary_mcp_tool() -> None:
    target_tool = _tool(_capability())
    ordinary_tool = _tool(_capability(original_name="list_documents"))

    assert ordinary_tool.protocol == "mcp"
    assert _identity(target_tool) != _identity(ordinary_tool)


@pytest.mark.asyncio
async def test_invocation_schema_permission_and_result_are_unchanged() -> None:
    capability = _capability()
    invocations: list[Any] = []

    async def invoke(invocation: Any) -> DriverInvocationResult:
        invocations.append(invocation)
        return DriverInvocationResult(
            ok=True,
            value={"answer": "unchanged"},
            metadata={"source": "driver"},
        )

    request_context = {"session_id": "session-2a"}
    tool = DriverCapabilityTool(capability, invoke, request_context)

    assert tool.name == capability.exposure.tool_name
    assert tool.description == capability.description
    assert tool.input_schema == capability.input_schema
    permission = await tool.check_permissions()
    assert permission.behavior is PermissionBehavior.ALLOW

    result = await tool(query="identity test")

    assert len(invocations) == 1
    assert invocations[0].capability_id == capability.capability_id
    assert invocations[0].payload == {"query": "identity test"}
    assert invocations[0].request_context == request_context
    assert result.state is ToolResultState.SUCCESS
    assert result.metadata == {"source": "driver"}
    assert len(result.content) == 1
    assert '"answer": "unchanged"' in result.content[0].text
