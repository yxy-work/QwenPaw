# -*- coding: utf-8 -*-
"""模型上下文窗口的检索前预算与最终请求预检。"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from agentscope.middleware import MiddlewareBase

from qwenpaw.agents.utils.token_counter import get_token_counter
from qwenpaw.config.config import (
    SearchKnowledgeBaseSafetyConfig,
    get_model_max_input_length,
)

from .contracts import (
    ContextBudgetAssessment,
    ContextWindowPreflightBlocked,
)

if TYPE_CHECKING:
    from agentscope.agent import Agent
    from agentscope.model import ChatResponse

logger = logging.getLogger(__name__)

_FALLBACK_SAFETY_FACTOR = 1.25
_BYTE_LIMIT_SAFETY_FACTOR = 1.20
_CONSERVATIVE_BYTE_DIVISOR = 3.0
_COMPLETION_TOKEN_KEYS = (
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
)


def _non_negative_int(value: Any) -> int | None:
    """仅接受非负、非布尔整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_int(value: Any) -> int | None:
    """仅接受正、非布尔整数。"""
    parsed = _non_negative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _json_safe(value: Any) -> Any:
    """将实际请求对象转换为稳定 JSON；不使用可能泄漏地址的 repr。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())

    raise TypeError(f"不支持的请求对象类型：{type(value).__name__}")


class ContextWindowBudgetGuard:
    """执行预测性检索预算和最终模型请求预算检查。"""

    def __init__(
        self,
        config: SearchKnowledgeBaseSafetyConfig,
        *,
        agent_config: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._config = config
        self._agent_config = agent_config
        self._model = model

    @property
    def enabled(self) -> bool:
        """返回最终模型预检开关。"""
        return bool(self._config.context_preflight_enabled)

    def _resolve_hard_limit(self, current_model: Any | None) -> int | None:
        """优先使用本次实际路由模型，其次使用 agent 模型配置。"""
        for model in (current_model, self._model):
            if model is None:
                continue
            context_size = _positive_int(getattr(model, "context_size", None))
            if context_size is not None:
                return context_size

        if self._agent_config is not None:
            try:
                return _positive_int(
                    get_model_max_input_length(self._agent_config),
                )
            except Exception:
                logger.warning(
                    "无法从 agent 模型配置解析上下文硬上限",
                    exc_info=True,
                )
        return None

    @staticmethod
    def _completion_from_mapping(values: Any) -> int | None:
        if not isinstance(values, Mapping):
            return None
        for key in _COMPLETION_TOKEN_KEYS:
            parsed = _non_negative_int(values.get(key))
            if parsed is not None:
                return parsed
        return None

    def _completion_from_model(self, model: Any | None) -> int | None:
        """读取实际模型 parameters，并兼容透明 retry wrapper。"""
        visited: set[int] = set()
        while model is not None and id(model) not in visited:
            visited.add(id(model))
            parameters = getattr(model, "parameters", None)
            for key in _COMPLETION_TOKEN_KEYS:
                parsed = _non_negative_int(getattr(parameters, key, None))
                if parsed is not None:
                    return parsed
            model = getattr(model, "_inner", None)
        return None

    def _completion_from_agent_config(self) -> int | None:
        """从活动 provider 的 ModelInfo 读取最大输出 token。"""
        agent_config = self._agent_config
        model_slot = getattr(agent_config, "active_model", None)
        provider_id = getattr(model_slot, "provider_id", "")
        model_name = getattr(model_slot, "model", "")
        if not provider_id or not model_name:
            return None
        try:
            from qwenpaw.providers import ProviderManager

            provider = ProviderManager.get_instance().get_provider(provider_id)
            model_info = provider.get_model_info(model_name) if provider else None
            return _non_negative_int(getattr(model_info, "max_tokens", None))
        except Exception:
            logger.warning(
                "无法从活动 provider 模型配置解析 completion token",
                exc_info=True,
            )
            return None

    def _resolve_reserved_completion(
        self,
        input_kwargs: Mapping[str, Any] | None,
        current_model: Any | None,
    ) -> int:
        """按实际请求、模型实例、ModelInfo、fallback 的顺序解析预留。"""
        if input_kwargs is not None:
            direct = self._completion_from_mapping(input_kwargs)
            if direct is not None:
                return direct
            for key in ("generate_kwargs", "generation_config"):
                nested = self._completion_from_mapping(input_kwargs.get(key))
                if nested is not None:
                    return nested

        model_value = self._completion_from_model(current_model or self._model)
        if model_value is not None:
            return model_value
        configured_value = self._completion_from_agent_config()
        if configured_value is not None:
            return configured_value
        return self._config.default_reserved_completion_tokens

    def _safe_input_limit(
        self,
        *,
        hard_limit: int,
        reserved_completion_tokens: int,
    ) -> int:
        margin = max(
            self._config.minimum_safety_margin_tokens,
            math.ceil(hard_limit * self._config.safety_margin_ratio),
        )
        return max(0, hard_limit - reserved_completion_tokens - margin)

    @staticmethod
    def _unavailable_assessment(
        reason: str,
        *,
        safe_input_limit: int = 0,
    ) -> ContextBudgetAssessment:
        """构造不可安全判定时的 fail-closed 快照。"""
        return ContextBudgetAssessment(
            allowed=False,
            current_input_tokens=0,
            projected_input_tokens=safe_input_limit + 1,
            safe_input_limit=safe_input_limit,
            estimator="unavailable",
            reason=reason,
        )

    async def assess_search_preflight(
        self,
        *,
        current_input_tokens: int,
        projected_search_result_tokens: int,
        inflight_reserved_tokens: int = 0,
    ) -> ContextBudgetAssessment:
        """预测检索结果、在途结果和工具消息包装后的输入体积。"""
        values = (
            current_input_tokens,
            projected_search_result_tokens,
            inflight_reserved_tokens,
        )
        if any(_non_negative_int(value) is None for value in values):
            raise ValueError("检索预算 token 参数必须为非负整数")

        hard_limit = self._resolve_hard_limit(None)
        if hard_limit is None:
            return self._unavailable_assessment("hard_limit_unavailable")
        reserved_completion = self._resolve_reserved_completion(None, self._model)
        safe_limit = self._safe_input_limit(
            hard_limit=hard_limit,
            reserved_completion_tokens=reserved_completion,
        )
        projected = (
            current_input_tokens
            + inflight_reserved_tokens
            + projected_search_result_tokens
            + self._config.tool_message_overhead_tokens
        )
        allowed = projected <= safe_limit
        return ContextBudgetAssessment(
            allowed=allowed,
            current_input_tokens=current_input_tokens,
            projected_input_tokens=projected,
            safe_input_limit=safe_limit,
            estimator="provided_formatted_input_tokens",
            reason=(
                "projected_input_within_safe_limit"
                if allowed
                else "projected_input_exceeds_safe_limit"
            ),
        )

    async def _fallback_count(
        self,
        *,
        messages: Any,
        tools: Any,
    ) -> int | None:
        """使用 QwenPaw 既有 bytes estimator，并施加更大安全系数。"""
        if self._agent_config is None:
            return None
        try:
            payload = json.dumps(
                {
                    "messages": _json_safe(messages),
                    "tools": _json_safe(tools),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            counter = get_token_counter(self._agent_config)
            raw_count = await counter.count(payload)
            parsed_count = _non_negative_int(raw_count)
            if parsed_count is None:
                return None
            return math.ceil(parsed_count * _FALLBACK_SAFETY_FACTOR)
        except Exception:
            logger.warning(
                "模型 token 计数失败后，QwenPaw bytes 降级估算也不可用",
                exc_info=True,
            )
            return None

    async def assess_model_request(
        self,
        *,
        input_kwargs: Mapping[str, Any],
    ) -> ContextBudgetAssessment:
        """统计 AgentScope 最终模型请求的 messages、tools 与包装余量。"""
        current_model = input_kwargs.get("current_model")
        hard_limit = self._resolve_hard_limit(current_model)
        if hard_limit is None:
            return self._unavailable_assessment("hard_limit_unavailable")

        reserved_completion = self._resolve_reserved_completion(
            input_kwargs,
            current_model,
        )
        safe_limit = self._safe_input_limit(
            hard_limit=hard_limit,
            reserved_completion_tokens=reserved_completion,
        )
        messages = input_kwargs.get("messages")
        tools = input_kwargs.get("tools")
        exact_count: int | None = None
        if current_model is not None and messages is not None:
            try:
                exact_count = _non_negative_int(
                    await current_model.count_tokens(messages, tools),
                )
            except Exception:
                logger.warning(
                    "实际路由模型 count_tokens 失败，进入 bytes 降级估算",
                    exc_info=True,
                )

        if exact_count is not None:
            current_tokens = exact_count
            estimator = "model_count_tokens"
        else:
            fallback_count = await self._fallback_count(
                messages=messages,
                tools=tools,
            )
            if fallback_count is None:
                return self._unavailable_assessment(
                    "token_estimation_unavailable",
                    safe_input_limit=safe_limit,
                )
            current_tokens = fallback_count
            estimator = "qwenpaw_byte_estimate_x1.25"

        projected = current_tokens + self._config.tool_message_overhead_tokens
        allowed = projected <= safe_limit
        return ContextBudgetAssessment(
            allowed=allowed,
            current_input_tokens=current_tokens,
            projected_input_tokens=projected,
            safe_input_limit=safe_limit,
            estimator=estimator,
            reason=(
                "actual_request_within_safe_limit"
                if allowed
                else "actual_request_exceeds_safe_limit"
            ),
        )

    def estimate_projected_tool_result_tokens(
        self,
        max_result_bytes: int | None = None,
    ) -> int:
        """将现有 bytes 工具结果上限保守换算为 token，不修改原配置。"""
        if max_result_bytes is None:
            try:
                pruning = (
                    self._agent_config.running.light_context_config
                    .tool_result_pruning_config
                )
                max_result_bytes = pruning.pruning_recent_msg_max_bytes
            except Exception as error:
                raise ValueError("无法读取工具结果 bytes 上限") from error
        parsed_bytes = _non_negative_int(max_result_bytes)
        if parsed_bytes is None:
            raise ValueError("工具结果 bytes 上限必须为非负整数")

        divisor = _CONSERVATIVE_BYTE_DIVISOR
        try:
            configured_divisor = float(
                self._agent_config.running.light_context_config
                .token_count_estimate_divisor,
            )
            if math.isfinite(configured_divisor) and configured_divisor > 0:
                divisor = min(configured_divisor, divisor)
        except Exception:
            pass
        return math.ceil(
            parsed_bytes / divisor * _BYTE_LIMIT_SAFETY_FACTOR,
        )


class ContextWindowPreflightMiddleware(MiddlewareBase):
    """在 provider 调用前执行 fail-closed 的最终上下文预检。"""

    def __init__(
        self,
        guard: ContextWindowBudgetGuard,
        *,
        enabled: bool | None = None,
    ) -> None:
        self._guard = guard
        self._enabled = guard.enabled if enabled is None else enabled

    async def on_model_call(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[
            ...,
            Awaitable["ChatResponse" | AsyncGenerator["ChatResponse", None]],
        ],
    ) -> "ChatResponse" | AsyncGenerator["ChatResponse", None]:
        """仅在预算安全时把同一实际请求继续传给内层 middleware。"""
        if not self._enabled:
            return await next_handler(**input_kwargs)

        assessment = await self._guard.assess_model_request(
            input_kwargs=input_kwargs,
        )
        if not assessment.allowed:
            state = getattr(agent, "state", None)
            model = input_kwargs.get("current_model")
            logger.warning(
                "上下文最终预检阻止模型请求：reply_id=%s model=%s "
                "estimator=%s current=%d projected=%d safe_limit=%d reason=%s",
                getattr(state, "reply_id", None),
                getattr(model, "model", type(model).__name__),
                assessment.estimator,
                assessment.current_input_tokens,
                assessment.projected_input_tokens,
                assessment.safe_input_limit,
                assessment.reason,
            )
            raise ContextWindowPreflightBlocked(assessment)
        return await next_handler(**input_kwargs)


__all__ = [
    "ContextWindowBudgetGuard",
    "ContextWindowPreflightMiddleware",
]
