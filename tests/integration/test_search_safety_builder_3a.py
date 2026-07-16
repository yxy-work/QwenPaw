'''阶段 3A 检索安全 builder 永久集成测试。'''
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agentscope.tool import Toolkit

from qwenpaw.agents.search_safety import (
    ContextWindowBudgetGuard,
    ContextWindowPreflightMiddleware,
    KnowledgeBaseSearchSafetyMiddleware,
    SearchSafetyStateStore,
)
from qwenpaw.config.config import AgentProfileConfig
from qwenpaw.drivers.adapters.agentscope_tool import DriverCapabilityTool
from qwenpaw.drivers.capabilities import (
    CapabilityExposure,
    DriverCapability,
    DriverInvocationResult,
    format_capability_id,
)
from qwenpaw.runtime.builder import AgentBuilder
from qwenpaw.tool_calls import ToolCoordinatorMiddleware


async def _invoke(_request: Any) -> DriverInvocationResult:
    '''返回最小 Driver 成功结果。'''
    return DriverInvocationResult(ok=True, value="result")


def _driver_tool(
    *,
    driver_name: str = "knowledgebase_remote",
    protocol: str = "mcp",
    capability_name: str = "search_knowledgebase",
    display_name: str = "renamed_search",
) -> DriverCapabilityTool:
    '''创建带真实稳定 capability id 的 AgentScope adapter。'''
    capability = DriverCapability(
        capability_id=format_capability_id(
            protocol,
            driver_name,
            "tool",
            "invoke",
            capability_name,
        ),
        driver_name=driver_name,
        protocol=protocol,
        kind="tool",
        action="invoke",
        name=capability_name,
        input_schema={
            "type": "object",
            "properties": {
                "kb_name": {"type": "string"},
                "query": {"type": "string"},
                "response_detail": {
                    "type": "string",
                    "default": "compact",
                },
            },
            "required": ["kb_name", "query"],
        },
        exposure=CapabilityExposure(
            as_tool=True,
            tool_name=display_name,
            namespace="arbitrary",
        ),
    )
    return DriverCapabilityTool(capability, _invoke)


class CountingModel:
    '''为检索前 getter 提供可验证的真实 ``count_tokens``。'''

    context_size = 131072
    parameters = SimpleNamespace(max_tokens=8192)

    def __init__(self) -> None:
        self.inputs: list[tuple[Any, Any]] = []

    async def count_tokens(self, messages: Any, tools: Any) -> int:
        self.inputs.append((messages, tools))
        return 1234


class FakeAgent:
    '''提供与 AgentScope ``_prepare_model_input`` 相同的当前输入。'''

    def __init__(self, model: CountingModel) -> None:
        self.model = model
        self.messages = [SimpleNamespace(role="user", content="question")]
        self.tools = [{"type": "function", "function": {"name": "kb"}}]

    async def _prepare_model_input(self) -> dict[str, Any]:
        return {"messages": self.messages, "tools": self.tools}


def _agent_config() -> AgentProfileConfig:
    '''构造保持真实 running 默认值的 Agent 配置。'''
    return AgentProfileConfig(
        id="integration",
        name="Integration",
        workspace_dir="/tmp/integration",
    )


async def test_builder_selects_stable_knowledgebase_driver_family() -> None:
    remote = _driver_tool(display_name="NOT_A_SUFFIX")
    persistent = _driver_tool(driver_name="knowledgebase_persistent")
    future_backend = _driver_tool(driver_name="knowledgebase_archive")
    other_driver = _driver_tool(driver_name="other_driver")
    no_separator = _driver_tool(driver_name="knowledgebase")
    similar_prefix = _driver_tool(driver_name="knowledgebaseevil")
    non_mcp = _driver_tool(
        driver_name="knowledgebase_local",
        protocol="http",
    )
    similar_suffix = _driver_tool(
        capability_name="prefix_search_knowledgebase",
    )
    ordinary_mcp = _driver_tool(capability_name="ordinary_search")
    toolkit = Toolkit(
        tools=[
            remote,
            persistent,
            future_backend,
            other_driver,
            no_separator,
            similar_prefix,
            non_mcp,
            similar_suffix,
            ordinary_mcp,
        ],
    )

    selected = AgentBuilder._select_knowledgebase_search_capabilities(toolkit)

    assert selected == [remote, persistent, future_backend]
    assert selected[0].protocol == "mcp"
    assert selected[0].driver_name == "knowledgebase_remote"
    assert selected[0].original_capability_name == "search_knowledgebase"


async def test_builder_wires_same_store_guard_and_onion_order() -> None:
    config = _agent_config()
    target = _driver_tool()
    toolkit = Toolkit(tools=[target])
    model = CountingModel()
    store = SearchSafetyStateStore()
    guard = ContextWindowBudgetGuard(
        config.running.search_knowledgebase_safety,
        agent_config=config,
        model=model,
    )
    coordinator = object()
    ctx = SimpleNamespace(
        app_services=SimpleNamespace(tool_coordinator=coordinator),
        workspace=None,
    )

    middlewares = AgentBuilder._build_middlewares(
        ctx,
        config,
        toolkit=toolkit,
        model=model,
        search_safety_state_store=store,
        context_window_budget_guard=guard,
    )

    search_index = next(
        index
        for index, middleware in enumerate(middlewares)
        if isinstance(middleware, KnowledgeBaseSearchSafetyMiddleware)
    )
    coordinator_index = next(
        index
        for index, middleware in enumerate(middlewares)
        if isinstance(middleware, ToolCoordinatorMiddleware)
    )
    preflight_index = next(
        index
        for index, middleware in enumerate(middlewares)
        if isinstance(middleware, ContextWindowPreflightMiddleware)
    )
    search = middlewares[search_index]
    preflight = middlewares[preflight_index]

    assert search_index < coordinator_index
    assert preflight_index == len(middlewares) - 1
    assert search._state_store is store
    assert search._budget_guard is guard
    assert preflight._guard is guard
    assert search._projected_search_result_tokens == (
        guard.estimate_projected_tool_result_tokens()
    )

    fake_agent = FakeAgent(model)
    current_tokens = await search._current_input_tokens(fake_agent)
    assert current_tokens == 1234
    assert model.inputs == [(fake_agent.messages, fake_agent.tools)]


async def test_builder_omits_search_middleware_without_target_but_keeps_preflight() -> None:
    config = _agent_config()
    model = CountingModel()
    store = SearchSafetyStateStore()
    guard = ContextWindowBudgetGuard(
        config.running.search_knowledgebase_safety,
        agent_config=config,
        model=model,
    )

    middlewares = AgentBuilder._build_middlewares(
        SimpleNamespace(app_services=None, workspace=None),
        config,
        toolkit=Toolkit(tools=[_driver_tool(driver_name="other")]),
        model=model,
        search_safety_state_store=store,
        context_window_budget_guard=guard,
    )

    assert not any(
        isinstance(middleware, KnowledgeBaseSearchSafetyMiddleware)
        for middleware in middlewares
    )
    assert isinstance(middlewares[-1], ContextWindowPreflightMiddleware)


async def test_builder_master_switch_disables_both_safety_middlewares() -> None:
    '''总开关关闭时，检索短路和最终 provider 预检均不得装配。'''
    config = _agent_config()
    config.running.search_knowledgebase_safety.enabled = False
    model = CountingModel()
    store = SearchSafetyStateStore()
    guard = ContextWindowBudgetGuard(
        config.running.search_knowledgebase_safety,
        agent_config=config,
        model=model,
    )

    middlewares = AgentBuilder._build_middlewares(
        SimpleNamespace(app_services=None, workspace=None),
        config,
        toolkit=Toolkit(tools=[_driver_tool()]),
        model=model,
        search_safety_state_store=store,
        context_window_budget_guard=guard,
    )

    assert not any(
        isinstance(
            middleware,
            (
                KnowledgeBaseSearchSafetyMiddleware,
                ContextWindowPreflightMiddleware,
            ),
        )
        for middleware in middlewares
    )
