# -*- coding: utf-8 -*-
"""QwenPaw Agent - Main agent implementation.

This module provides the main QwenPawAgent class built on ReActAgent,
with integrated tools, skills, and memory management.

Agent construction is fully delegated to :class:`AgentBuilder` — the
agent accepts all dependencies (model, prompt, toolkit, middlewares)
as constructor parameters and does not build them internally.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import uuid

from agentscope.agent import Agent, ReActConfig
from agentscope.event import (
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
)
from agentscope.message import (
    Msg,
    SystemMsg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    UserMsg,
)
from agentscope.state import AgentState
from agentscope.tool import ToolChoice, Toolkit

from .search_safety import (
    ContextWindowPreflightBlocked,
    SearchSafetyStateStore,
)
from .skill_system import get_workspace_skills_dir
from ..modes.coding import CodingModeMixin
from ..constant import (
    AUTO_CONTINUE_MESSAGE_TAG,
    MEDIA_UNSUPPORTED_PLACEHOLDER,
    QWENPAW_MESSAGE_TAG_KEY,
    WORKING_DIR,
)
from ..providers.model_capability_cache import get_capability_cache

if TYPE_CHECKING:
    from ..agents.memory import BaseMemoryManager
    from ..config.config import AgentProfileConfig

logger = logging.getLogger(__name__)

SEARCH_SAFETY_STATE_KEY = "search_knowledgebase_safety"
FORCE_FINALIZE_REPAIR_PROMPT = (
    "你现在只负责形成面向用户的最终答复。用户消息中已经包含本轮允许使用的"
    "检索证据；不得继续检索、输出函数标签或制定后续检索计划。请直接依据已有"
    "证据回答，区分有依据的结论与证据不足之处，并可简要说明本轮已达到检索"
    "安全上限。"
)
FORCE_FINALIZE_MAX_PROJECTED_CHARS = 48_000
FORCE_FINALIZE_MAX_QUESTION_CHARS = 12_000
FORCE_FINALIZE_LOCAL_EVIDENCE_CHARS = 8_000
FORCE_FINALIZE_FIXED_INPUT_TOKENS = 4_096
FORCE_FINALIZE_DEFAULT_RESERVED_COMPLETION_TOKENS = 8_192
FORCE_FINALIZE_DEFAULT_SAFETY_MARGIN_RATIO = 0.03
FORCE_FINALIZE_DEFAULT_MINIMUM_MARGIN_TOKENS = 4_096


class QwenPawAgent(CodingModeMixin, Agent):
    """QwenPaw Agent with integrated tools, skills, and memory management.

    This agent extends agentscope 2.0 ``Agent`` with:
    - Built-in tools (shell, file operations, browser, etc.)
    - Dynamic skill loading from working directory
    - Memory management with auto-compaction
    - Bootstrap guidance for first-time setup
    - Tool-guard security (via ``PolicyGuardedTool.check_permissions``)
    - Coding Mode features: Inline Diff (via CodingModeMixin)
    """

    def __init__(
        self,
        *,
        name: str,
        model: Any,
        system_prompt: str,
        toolkit: Toolkit,
        react_config: ReActConfig,
        middlewares: list,
        agent_config: "AgentProfileConfig",
        workspace_dir: Path | None = None,
        request_context: Optional[dict[str, str]] = None,
        memory_manager: "BaseMemoryManager | None" = None,
        offloader: Any = None,
        context_config: Any = None,
        context_manager: Any = None,
        effective_skills: Optional[list[str]] = None,
        governor: Any = None,
        search_safety_state_store: SearchSafetyStateStore | None = None,
    ):
        """Initialize QwenPawAgent.

        All construction dependencies (model, prompt, toolkit, middlewares)
        are provided externally by :class:`AgentBuilder`. The agent does
        not build any of these internally.
        """
        self._agent_config = agent_config
        self._request_context = dict(request_context or {})
        self._workspace_dir = workspace_dir
        self._language = agent_config.language
        # Optional context-management strategy. When None, the agent keeps its
        # native AgentScope compression (see compress_context /
        # _save_to_context).
        self._context_manager = context_manager
        self._search_safety_state_store = (
            search_safety_state_store or SearchSafetyStateStore()
        )
        self._force_finalize_repair_prompt_active = False

        # Register skills metadata on toolkit
        self._register_skills(toolkit, effective_skills=effective_skills or [])

        self._governor = governor

        self.memory_manager = memory_manager

        # Register memory tools into toolkit
        if self.memory_manager is not None:
            memory_tools = self.memory_manager.list_memory_tools()
            basic_group = toolkit.tool_groups[0]
            for tool_fn in memory_tools:
                from ..governance import PolicyGuardedTool

                basic_group.tools.append(
                    PolicyGuardedTool(
                        tool_fn,
                        governor=self._governor,
                        request_context=self._request_context,
                    ),
                )
            logger.debug(
                "Registered memory tools: %s",
                [fn.__name__ for fn in memory_tools],
            )

        init_kwargs: dict[str, Any] = {
            "name": name,
            "model": model,
            "system_prompt": system_prompt,
            "toolkit": toolkit,
            "react_config": react_config,
            "middlewares": middlewares,
            "offloader": offloader,
        }
        if context_config is not None:
            init_kwargs["context_config"] = context_config
        super().__init__(**init_kwargs)

        # Bypass agentscope's built-in permission engine — qwenpaw uses
        # its own PolicyGuardedTool.check_permissions for tool-guard.
        from agentscope.permission import PermissionMode

        self.state.permission_context.mode = PermissionMode.BYPASS

        # Tombstone for legacy ``getattr(agent, "memory", None)`` callers
        self.memory = None  # type: ignore[assignment]

        self._register_tool_call_hooks()

    async def compress_context(
        self,
        context_config: Any = None,
    ) -> None:
        """Delegate to the context manager, else native compression.

        With a ``context_manager`` injected (e.g. the scroll strategy), it owns
        compression. Otherwise fall back to AgentScope's native path, gated on
        ``context_compact_config.enabled``.
        """
        if self._context_manager is not None:
            await self._context_manager.compress(self, context_config)
            return
        try:
            lcc = self._agent_config.running.light_context_config
            if not lcc.context_compact_config.enabled:
                return
        except Exception:
            pass
        await super().compress_context(context_config)

    def _save_to_context(self, blocks: Any, usage: Any = None) -> None:
        """Append blocks, then let the context manager write them through."""
        super()._save_to_context(blocks, usage)
        if self._context_manager is not None:
            self._context_manager.on_save(self, blocks)

    async def _get_system_prompt(self) -> str:
        """在安全投影请求中使用独立、无工具导向的收尾指令。"""
        if self._force_finalize_repair_prompt_active:
            return FORCE_FINALIZE_REPAIR_PROMPT
        return await super()._get_system_prompt()

    async def _prepare_model_input(self) -> dict[str, Any]:
        """构造模型输入，并在强制收尾阶段彻底隐藏工具 schema。"""
        if self._force_finalize_repair_prompt_active:
            return {
                "messages": self._build_force_finalize_projection(),
                "tools": [],
            }
        input_kwargs = await super()._prepare_model_input()
        state = await self._search_safety_state_store.get(
            self.state.reply_id,
        )
        if state is not None and state.force_finalize:
            input_kwargs["tools"] = []
        return input_kwargs

    def _build_force_finalize_projection(self) -> list[Msg]:
        """生成只含当前问题和预算内纯文本证据的临时模型输入。"""
        question, evidence = self._current_reply_evidence()
        total_budget = self._force_finalize_projection_char_budget()
        question_budget = min(
            FORCE_FINALIZE_MAX_QUESTION_CHARS,
            max(0, total_budget // 4),
        )
        projected_question = self._truncate_text(question, question_budget)
        evidence_budget = max(0, total_budget - len(projected_question))
        evidence_text = self._render_bounded_evidence(
            evidence,
            evidence_budget,
        )
        if not projected_question:
            projected_question = "请依据本轮已有证据形成最终答复。"
        if not evidence_text:
            evidence_text = "本轮没有可提取的已完成检索结果。"
        projection = (
            "当前用户问题：\n"
            f"{projected_question}\n\n"
            "本轮已检索证据（已按上下文安全预算裁剪）：\n"
            f"{evidence_text}"
        )
        return [
            SystemMsg(name="system", content=FORCE_FINALIZE_REPAIR_PROMPT),
            UserMsg(name="user", content=projection),
        ]

    def _force_finalize_projection_char_budget(self) -> int:
        """按当前模型安全输入线计算保守的投影字符预算。"""
        hard_limit = self._model_context_size()
        if hard_limit is None:
            return FORCE_FINALIZE_MAX_PROJECTED_CHARS

        running = getattr(self._agent_config, "running", None)
        safety_config = getattr(
            running,
            "search_knowledgebase_safety",
            None,
        )
        reserved = int(
            getattr(
                safety_config,
                "default_reserved_completion_tokens",
                FORCE_FINALIZE_DEFAULT_RESERVED_COMPLETION_TOKENS,
            ),
        )
        margin_ratio = float(
            getattr(
                safety_config,
                "safety_margin_ratio",
                FORCE_FINALIZE_DEFAULT_SAFETY_MARGIN_RATIO,
            ),
        )
        minimum_margin = int(
            getattr(
                safety_config,
                "minimum_safety_margin_tokens",
                FORCE_FINALIZE_DEFAULT_MINIMUM_MARGIN_TOKENS,
            ),
        )
        margin = max(minimum_margin, math.ceil(hard_limit * margin_ratio))
        safe_input_tokens = max(0, hard_limit - reserved - margin)
        payload_tokens = max(
            0,
            safe_input_tokens - FORCE_FINALIZE_FIXED_INPUT_TOKENS,
        )
        return min(
            FORCE_FINALIZE_MAX_PROJECTED_CHARS,
            payload_tokens // 2,
        )

    def _model_context_size(self) -> int | None:
        """读取透明 model wrapper 链中的实际上下文大小。"""
        model = getattr(self, "model", None)
        visited: set[int] = set()
        while model is not None and id(model) not in visited:
            visited.add(id(model))
            context_size = getattr(model, "context_size", None)
            if (
                isinstance(context_size, int)
                and not isinstance(context_size, bool)
                and context_size > 0
            ):
                return context_size
            model = getattr(model, "_inner", None)
        return None

    def _current_reply_evidence(self) -> tuple[str, list[str]]:
        """提取最后一个用户问题之后已经完成的工具结果正文。"""
        context = getattr(self.state, "context", [])
        user_index = -1
        for index in range(len(context) - 1, -1, -1):
            message = context[index]
            metadata = getattr(message, "metadata", {})
            if (
                getattr(message, "role", None) == "user"
                and metadata.get(QWENPAW_MESSAGE_TAG_KEY)
                != AUTO_CONTINUE_MESSAGE_TAG
            ):
                user_index = index
                break

        question = (
            context[user_index].get_text_content()
            if user_index >= 0
            else ""
        )
        evidence: list[str] = []
        for message in context[user_index + 1 :]:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, ToolResultBlock):
                    continue
                text = self._tool_result_text(block)
                if text:
                    evidence.append(text)
        return question, evidence

    @staticmethod
    def _tool_result_text(block: ToolResultBlock) -> str:
        """将 ToolResultBlock 中的文本块转为无 metadata 的纯文本。"""
        if isinstance(block.output, str):
            return block.output.strip()
        text_parts = [
            item.text.strip()
            for item in block.output
            if isinstance(item, TextBlock) and item.text.strip()
        ]
        return "\n".join(text_parts)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """从开头保留文本，并用明确标记表示安全裁剪。"""
        if max_chars <= 0:
            return ""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        marker = "\n[已按安全预算裁剪]"
        if max_chars <= len(marker):
            return marker[:max_chars]
        keep = max(0, max_chars - len(marker))
        return f"{text[:keep].rstrip()}{marker}"

    def _render_bounded_evidence(
        self,
        evidence: list[str],
        max_chars: int,
    ) -> str:
        """公平分配字符预算，避免首个大结果挤占全部证据。"""
        if not evidence or max_chars <= 0:
            return ""
        header_allowance = sum(
            len(f"[检索证据 {index}]\n") + 2
            for index in range(1, len(evidence) + 1)
        )
        content_budget = max(0, max_chars - header_allowance)
        per_result = content_budget // len(evidence)
        rendered = [
            f"[检索证据 {index}]\n"
            f"{self._truncate_text(text, per_result)}"
            for index, text in enumerate(evidence, start=1)
        ]
        return "\n\n".join(rendered).strip()

    # Session persistence calls state_dict/load_state_dict on the agent;
    # these round-trip through self.state (AgentState pydantic model).
    def state_dict(self) -> dict:
        """Serialize the agent's 2.0 ``AgentState`` to a JSON-safe dict."""
        state = getattr(self, "state", None)
        if state is None:
            return {}
        out = {"state": state.model_dump(mode="json")}
        # Persist the scroll manager's dedup bookkeeping + eviction index so a
        # resumed session doesn't re-append its restored window to history.db.
        cm = getattr(self, "_context_manager", None)
        if cm is not None and hasattr(cm, "to_dict"):
            out["scroll"] = cm.to_dict()
        store = getattr(self, "_search_safety_state_store", None)
        if store is not None:
            safety_state = store.snapshot_sync(state.reply_id)
            out[SEARCH_SAFETY_STATE_KEY] = safety_state.model_dump(
                mode="json",
            )
        return out

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore ``self.state`` from a dict produced by :meth:`state_dict`.

        Handles two formats:
        - **2.0**: ``{"state": {AgentState dump}}``
        - **1.x legacy**: ``{"memory": {"content": [[msg, marks], ...],
          "_compressed_summary": "..."}}`` — converted on-the-fly so
          existing sessions survive the upgrade.
        """
        if not isinstance(state_dict, dict):
            if strict:
                raise KeyError("state_dict is not a dict")
            return

        search_safety_state_store = getattr(
            self,
            "_search_safety_state_store",
            None,
        )

        # --- 2.0 format (preferred) ---
        raw = state_dict.get("state")
        if raw is not None:
            try:
                self.state = AgentState.model_validate(raw)
            except Exception as exc:
                raise KeyError(
                    f"Could not load AgentState from snapshot: {exc}",
                ) from exc
            # Rehydrate the scroll manager's bookkeeping so the restored window
            # is recognized as already durable (no re-append on resume).
            cm = getattr(self, "_context_manager", None)
            scroll = state_dict.get("scroll")
            if (
                cm is not None
                and scroll is not None
                and hasattr(cm, "load_state")
            ):
                cm.load_state(scroll)
            if search_safety_state_store is not None:
                search_safety_state_store.restore_sync(
                    self.state.reply_id,
                    state_dict.get(SEARCH_SAFETY_STATE_KEY),
                )
            return

        # --- 1.x legacy format: migrate ``memory`` → ``state`` ---
        memory_raw = state_dict.get("memory")
        if isinstance(memory_raw, dict):
            from qwenpaw.app.chats.utils import parse_legacy_memory_state

            msgs, summary = parse_legacy_memory_state(memory_raw)
            self.state = AgentState()
            self.state.context.extend(msgs)
            self.state.summary = summary
            logger.info(
                "Migrated 1.x session: %d messages + summary(%d chars)",
                len(msgs),
                len(self.state.summary),
            )
            if search_safety_state_store is not None:
                search_safety_state_store.restore_sync(
                    self.state.reply_id,
                    state_dict.get(SEARCH_SAFETY_STATE_KEY),
                )
            return

        if strict:
            raise KeyError(
                "state_dict has neither 'state' nor 'memory' key",
            )

    async def close(self) -> None:
        """Shut down governor, release the history store, and clean up expired
        tool-result files."""
        gov = getattr(self, "_governor", None)
        if gov is not None:
            try:
                gov.stop()
            except Exception:
                logger.debug("governor stop failed", exc_info=True)

        # Scroll history: apply the retention window (if any) while the
        # connection is still open, then release it (db + -wal + -shm fds —
        # otherwise they accumulate across requests on a long-lived server).
        cm = getattr(self, "_context_manager", None)
        if cm is not None:
            if hasattr(cm, "purge_old"):
                try:
                    lcc = self._agent_config.running.light_context_config
                    cm.purge_old(lcc.scroll_config.history_retention_days)
                except Exception:
                    logger.debug(
                        "history retention purge failed",
                        exc_info=True,
                    )
            if hasattr(cm, "close"):
                try:
                    cm.close()
                except Exception:
                    logger.debug(
                        "context manager close failed",
                        exc_info=True,
                    )

        offloader = getattr(self, "offloader", None)
        if offloader is not None and hasattr(
            offloader,
            "cleanup_expired",
        ):
            try:
                lcc = self._agent_config.running.light_context_config
                trc = lcc.tool_result_pruning_config
                offloader.cleanup_expired(
                    retention_days=trc.offload_retention_days,
                )
            except Exception:
                logger.debug("offloader cleanup failed", exc_info=True)

    def _register_skills(
        self,
        toolkit: Toolkit,
        effective_skills: list[str],
    ) -> None:
        """Load and register skills from workspace directory.

        Skills are stored in ``toolkit._qp_skills`` (a dict) for downstream
        consumption (e.g. ``/skill_name`` slash commands in the runner).
        """
        if not hasattr(toolkit, "_qp_skills"):
            toolkit._qp_skills = {}  # pylint: disable=protected-access
        workspace_dir = self._workspace_dir or WORKING_DIR
        working_skills_dir = get_workspace_skills_dir(Path(workspace_dir))

        for skill_name in effective_skills:
            skill_dir = working_skills_dir / skill_name
            if skill_dir.exists():
                try:
                    # pylint: disable=protected-access
                    toolkit._qp_skills[skill_name] = {
                        "dir": str(skill_dir),
                    }
                    logger.debug("Registered skill: %s", skill_name)
                except Exception as e:
                    logger.error(
                        "Failed to register skill '%s': %s",
                        skill_name,
                        e,
                    )

    # ------------------------------------------------------------------
    # Media-block fallback: strip unsupported media blocks (image, audio,
    # video, file) from memory and retry when the model rejects them.
    # Unlike ``model_factory._fixup_media_list`` (which converts file
    # blocks to text placeholders so the user-facing message history
    # stays readable), this fallback strips them entirely — its purpose
    # is to make a previously-rejected request retryable, so leaving
    # residue would defeat the point.
    # ------------------------------------------------------------------

    _MEDIA_BLOCK_TYPES = {"image", "audio", "video", "file"}
    _MEDIA_MIME_PREFIXES = ("image/", "audio/", "video/")

    _AUTO_CONTINUE_MAX_EXTRA = 2
    _AUTO_CONTINUE_TAIL_CHARS = 600

    _AUTO_CONTINUE_HINT_EN = (
        "<system-hint>"
        "Your previous assistant turn had text only (no tool calls). "
        "Use the trailing excerpt in <previous-assistant-tail> (if present) "
        "plus the conversation to decide in this **reasoning** step: if the "
        "user's task still needs tools, emit tool_use now; if it is fully "
        "done, reply with a short text only (no tools). "
        "Do not stop with plans or code fences alone when tools are still "
        "needed."
        "</system-hint>"
    )
    _AUTO_CONTINUE_HINT_ZH = (
        "<system-hint>"
        "上轮助手仅文字、未调工具。请结合上下文与 <previous-assistant-tail> "
        "（若有）在本轮推理中判断：仍需执行则立刻 tool；已完结则简短收尾。"
        "需要操作时勿只输出计划或代码块。"
        "</system-hint>"
    )

    def _auto_continue_system_hint(self) -> str:
        """Pick hint by agent language (zh vs others)."""
        raw_lang = getattr(self._agent_config, "language", None)
        lang = (raw_lang or "").strip().lower()
        if lang == "zh":
            return self._AUTO_CONTINUE_HINT_ZH
        return self._AUTO_CONTINUE_HINT_EN

    @staticmethod
    def _auto_continue_tail_context(msg: Msg, max_chars: int) -> str:
        """Assistant text suffix for hint (fixed cut, not sentence NLP)."""
        raw = msg.get_text_content() if msg is not None else ""
        text = (raw or "").strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:].lstrip()

    # _auto_continue_if_text_only — replaced by inline logic in _reasoning()
    # which leverages the 2.0 outer react loop instead of a manual while-loop.

    def _get_model_key(self) -> str | None:
        """Return the capability-cache key for the active model."""
        model = getattr(self, "model", None)
        return getattr(model, "model_key", None)

    def _model_rejects_media(self) -> bool:
        """Check the capability cache for a learned ``rejects_media`` flag."""
        key = self._get_model_key()
        if key is None:
            return False
        return get_capability_cache().get(key, "rejects_media", False)

    def _proactive_strip_media_blocks(self) -> int:
        """Proactively strip media blocks from memory before model call.

        Only called when the active model does not support multimodal.
        Returns the number of blocks stripped.
        """
        return self._strip_media_blocks_from_memory()

    def _uses_request_time_media_normalization(self) -> bool:
        """Return True when request-time normalization can handle media."""
        return getattr(self, "formatter", None) is not None

    def _set_formatter_media_strip(self, enabled: bool) -> None:
        """Toggle request-time media stripping on the active formatter."""
        formatter = getattr(self, "formatter", None)
        if formatter is None:
            return
        setattr(formatter, "_qwenpaw_force_strip_media", enabled)

    # pylint: disable=too-many-branches,too-many-statements
    async def _reasoning(
        self,
        tool_choice: ToolChoice | None = None,
    ):
        """Forward 2.0 ``_reasoning`` events with proactive media
        stripping, passive bad-request retry, and auto-continue on
        text-only responses."""

        reply_id = self.state.reply_id
        search_safety_state = (
            await self._search_safety_state_store.get_or_create(reply_id)
        )
        effective_tool_choice = (
            ToolChoice(mode="none")
            if search_safety_state.force_finalize
            else tool_choice
        )
        suppress_provider_events = (
            effective_tool_choice is not None
            and effective_tool_choice.mode == "none"
        )
        buffered_provider_events: list[Any] = []
        context_before = self._get_last_msg()
        content_count_before = (
            len(context_before.content)
            if context_before is not None
            and isinstance(context_before.content, list)
            else 0
        )

        # ── Proactive media stripping ──
        from .model_factory import _supports_multimodal_for_current_model

        should_strip = (
            not _supports_multimodal_for_current_model()
            or self._model_rejects_media()
        )
        if should_strip:
            if self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(True)
            else:
                n = self._proactive_strip_media_blocks()
                if n > 0:
                    logger.warning(
                        "Proactively stripped %d media block(s) before "
                        "_reasoning (model lacks multimodal support).",
                        n,
                    )

        # ── Model call with passive retry on media error ──
        final_msg: Msg | None = None
        local_terminal = False
        try:
            async for evt in super()._reasoning(
                tool_choice=effective_tool_choice,
            ):
                if isinstance(evt, Msg):
                    final_msg = evt
                elif suppress_provider_events:
                    buffered_provider_events.append(evt)
                else:
                    yield evt
        except ContextWindowPreflightBlocked as exc:
            logger.warning(
                "Model request blocked by context preflight: "
                "reply_id=%s projected_input_tokens=%d "
                "safe_input_limit=%d reason=%s",
                reply_id,
                exc.assessment.projected_input_tokens,
                exc.assessment.safe_input_limit,
                exc.assessment.reason,
            )
            await self._search_safety_state_store.set_force_finalize(
                reply_id,
                "context_budget_reached",
            )
            effective_tool_choice = ToolChoice(mode="none")
            suppress_provider_events = True
            final_msg, buffered_provider_events, local_terminal = (
                await self._retry_force_finalize_once(
                    effective_tool_choice,
                )
            )
            if self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(False)
        except Exception as e:
            if not self._is_bad_request_or_media_error(e):
                raise

            model_key = self._get_model_key()
            if model_key:
                get_capability_cache().learn(
                    model_key,
                    "rejects_media",
                    True,
                )
            logger.warning(
                "_reasoning failed with media error (%s); "
                "stripping media and retrying.",
                e,
            )
            if self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(True)
            else:
                self._strip_media_blocks_from_memory()

            try:
                async for evt in super()._reasoning(
                    tool_choice=effective_tool_choice,
                ):
                    if isinstance(evt, Msg):
                        final_msg = evt
                    elif suppress_provider_events:
                        buffered_provider_events.append(evt)
                    else:
                        yield evt
            except ContextWindowPreflightBlocked as exc:
                logger.warning(
                    "Retried model request blocked by context preflight: "
                    "reply_id=%s projected_input_tokens=%d "
                    "safe_input_limit=%d reason=%s",
                    reply_id,
                    exc.assessment.projected_input_tokens,
                    exc.assessment.safe_input_limit,
                    exc.assessment.reason,
                )
                await self._search_safety_state_store.set_force_finalize(
                    reply_id,
                    "context_budget_reached",
                )
                effective_tool_choice = ToolChoice(mode="none")
                suppress_provider_events = True
                final_msg, buffered_provider_events, local_terminal = (
                    await self._retry_force_finalize_once(
                        effective_tool_choice,
                    )
                )
            finally:
                if self._uses_request_time_media_normalization():
                    self._set_formatter_media_strip(False)
        else:
            if should_strip and self._uses_request_time_media_normalization():
                self._set_formatter_media_strip(False)

        if (
            effective_tool_choice is not None
            and effective_tool_choice.mode == "none"
            and not local_terminal
        ):
            forbidden_tool_calls = self._rollback_new_reasoning_content(
                context_before,
                content_count_before,
            )
            if forbidden_tool_calls:
                logger.warning(
                    "Provider returned tool calls while tool_choice=none; "
                    "retrying once without tool schemas: "
                    "reply_id=%s tool_call_count=%d",
                    reply_id,
                    forbidden_tool_calls,
                )
                final_msg, buffered_provider_events, local_terminal = (
                    await self._retry_force_finalize_once(
                        effective_tool_choice,
                    )
                )

        if final_msg is None:
            return

        if suppress_provider_events and not local_terminal:
            for event in buffered_provider_events:
                yield event

        if local_terminal:
            for event in self._local_final_text_events(final_msg):
                yield event

        # ── Auto-continue: text-only → inject hint, let outer loop retry ──
        if not local_terminal and self._should_auto_continue(
            final_msg,
            effective_tool_choice,
        ):
            hint_body = self._auto_continue_system_hint()
            tail = self._auto_continue_tail_context(
                final_msg,
                self._AUTO_CONTINUE_TAIL_CHARS,
            )
            if tail:
                hint_body += (
                    "\n\n<previous-assistant-tail>\n"
                    f"{tail}\n"
                    "</previous-assistant-tail>"
                )
            logger.info(
                "Auto-continue: text-only response; injecting hint "
                "(tool_choice=%r)",
                effective_tool_choice,
            )
            self.state.context.append(
                Msg(
                    name="user",
                    role="user",
                    content=[TextBlock(type="text", text=hint_body)],
                    metadata={
                        QWENPAW_MESSAGE_TAG_KEY: AUTO_CONTINUE_MESSAGE_TAG,
                    },
                ),
            )
            return  # outer loop continues → _check_next_action → reasoning

        yield final_msg

    def _build_local_final_message(self, text: str) -> Msg:
        """保存并返回不依赖 provider 的本地 assistant 终态消息。"""
        block = TextBlock(type="text", text=text)
        self._save_to_context([block])
        return Msg(
            id=self.state.reply_id,
            name=self.name,
            role="assistant",
            content=[block.model_copy(deep=True)],
        )

    def _build_safe_evidence_fallback_message(self, reason: str) -> Msg:
        """provider 无法收尾时直接交付已有证据，而不是拒绝回答。"""
        _, evidence = self._current_reply_evidence()
        bounded_evidence = self._render_bounded_evidence(
            evidence,
            FORCE_FINALIZE_LOCAL_EVIDENCE_CHARS,
        )
        if bounded_evidence:
            text = (
                "本轮已在安全上限内停止继续检索，未执行额外检索。"
                f"{reason}以下直接返回安全范围内的已有证据摘录；摘录未做额外"
                "推断：\n\n"
                f"{bounded_evidence}"
            )
        else:
            text = (
                "本轮已在安全上限内停止继续处理，未执行额外检索。"
                f"{reason}当前回复中没有可提取的已完成检索结果，因此无法生成"
                "有证据支撑的结论。"
            )
        return self._build_local_final_message(text)

    def _rollback_new_reasoning_content(
        self,
        context_before: Msg | None,
        content_count_before: int,
    ) -> int:
        """检测违规工具调用，并回滚本次 reasoning 新增的全部内容。"""
        current = self._get_last_msg()
        if current is None or not isinstance(current.content, list):
            return 0

        start_index = content_count_before if current is context_before else 0
        prefix = current.content[:start_index]
        appended = current.content[start_index:]
        forbidden_count = sum(
            isinstance(block, ToolCallBlock) for block in appended
        )
        if forbidden_count:
            current.content = prefix
        return forbidden_count

    async def _retry_force_finalize_once(
        self,
        tool_choice: ToolChoice,
    ) -> tuple[Msg | None, list[Any], bool]:
        """首次违规后执行恰好一次无工具 schema 的纯文本收尾请求。"""
        context_before = self._get_last_msg()
        content_count_before = (
            len(context_before.content)
            if context_before is not None
            and isinstance(context_before.content, list)
            else 0
        )
        final_msg: Msg | None = None
        buffered_events: list[Any] = []
        self._force_finalize_repair_prompt_active = True
        try:
            async for event in super()._reasoning(tool_choice=tool_choice):
                if isinstance(event, Msg):
                    final_msg = event
                else:
                    buffered_events.append(event)
        except ContextWindowPreflightBlocked as exc:
            logger.warning(
                "Force-finalize retry blocked by context preflight: "
                "reply_id=%s projected_input_tokens=%d "
                "safe_input_limit=%d reason=%s",
                self.state.reply_id,
                exc.assessment.projected_input_tokens,
                exc.assessment.safe_input_limit,
                exc.assessment.reason,
            )
            return (
                self._build_safe_evidence_fallback_message(
                    "安全投影仍超过模型输入上限，系统未发送该请求。",
                ),
                [],
                True,
            )
        finally:
            self._force_finalize_repair_prompt_active = False

        forbidden_tool_calls = self._rollback_new_reasoning_content(
            context_before,
            content_count_before,
        )
        if forbidden_tool_calls:
            logger.error(
                "Provider returned tool calls on bounded force-finalize "
                "retry; blocked execution: reply_id=%s tool_call_count=%d",
                self.state.reply_id,
                forbidden_tool_calls,
            )
            return (
                self._build_safe_evidence_fallback_message(
                    "模型仍返回工具请求，系统已阻止执行。",
                ),
                [],
                True,
            )
        return final_msg, buffered_events, False

    def _local_final_text_events(self, message: Msg) -> list[Any]:
        """把本地终态消息转换为 channel 可见的 AgentScope 文本事件。"""
        text = message.get_text_content()
        if not text:
            return []
        block_id = uuid.uuid4().hex
        return [
            TextBlockStartEvent(
                reply_id=self.state.reply_id,
                block_id=block_id,
            ),
            TextBlockDeltaEvent(
                reply_id=self.state.reply_id,
                block_id=block_id,
                delta=text,
            ),
            TextBlockEndEvent(
                reply_id=self.state.reply_id,
                block_id=block_id,
            ),
        ]

    def _should_auto_continue(
        self,
        msg: Msg,
        tool_choice: ToolChoice | None,
    ) -> bool:
        """Check if auto-continue should be triggered."""
        running = getattr(self, "_agent_config", None)
        running = getattr(running, "running", None)
        if running is None or not getattr(
            running,
            "auto_continue_on_text_only",
            False,
        ):
            return False

        if msg is None or msg.has_content_blocks("tool_call"):
            return False

        if tool_choice is not None and tool_choice.mode == "none":
            return False

        if self.state.cur_iter >= self.react_config.max_iters - 1:
            return False

        return True

    @staticmethod
    def _is_bad_request_or_media_error(exc: Exception) -> bool:
        """Return True only for errors that genuinely look media-related.

        A bare 400 is no longer sufficient — provider gateways return
        400 for many unrelated reasons (request too large, malformed
        block fields, exceeded context length) and treating them all as
        "media rejected" poisons the capability cache, causing
        subsequent requests to silently drop user-uploaded images.
        """
        error_str = str(exc).lower()

        # Veto: errors clearly about request size / context length are
        # never about media support — stripping media may incidentally
        # make the next request fit, but it's a coincidence, not a
        # learned capability.
        size_signals = (
            "too large",
            "toolarge",
            "max bytes",
            "request body",
            "context length",
            "context_length",
            "maximum context",
            "max_tokens",
        )
        if any(sig in error_str for sig in size_signals):
            return False

        # Match only when the error message itself names a media modality.
        media_keywords = (
            "image",
            "audio",
            "video",
            "vision",
            "multimodal",
            "image_url",
        )
        return any(kw in error_str for kw in media_keywords)

    def _is_media_block(self, block: Any) -> bool:
        """Return True if *block* carries image/audio/video data."""
        if isinstance(block, dict):
            return block.get("type") in self._MEDIA_BLOCK_TYPES
        btype = getattr(block, "type", None)
        if btype in self._MEDIA_BLOCK_TYPES:
            return True
        if btype == "data":
            source = getattr(block, "source", None)
            mt = getattr(source, "media_type", "") or ""
            return mt.startswith(self._MEDIA_MIME_PREFIXES)
        return False

    # ------------------------------------------------------------------
    # Tool call enhancement: hint injection + hook registration
    # ------------------------------------------------------------------

    def _get_tool_coordinator(self) -> Any:
        """Return the ToolCoordinator from request_context, or None."""
        return (self._request_context or {}).get("tool_coordinator")

    async def _inject_pending_hints(self) -> None:
        """Pop background-tool hints and append them to agent context."""
        mgr = self._get_tool_coordinator()
        if mgr is None:
            return
        session_id = (self._request_context or {}).get("session_id", "")
        if not session_id:
            return
        hints = await mgr.pop_pending_hints(session_id)
        for hint in hints:
            self.state.context.append(hint)

    async def _reply(self, **kwargs: Any) -> Any:
        """Override to inject pending background-tool hints before reply."""
        await self._inject_pending_hints()
        async for evt in super()._reply(**kwargs):
            yield evt

    def _register_tool_call_hooks(self) -> None:
        """Register per-tool default timeouts on the ToolCoordinator."""
        mgr = self._get_tool_coordinator()
        if mgr is None:
            return

        mgr.hooks.register(
            "execute_shell_command",
            default_timeout_secs=60.0,
        )
        mgr.hooks.register("chat_with_agent", default_timeout_secs=300.0)
        mgr.hooks.register("check_agent_task", default_timeout_secs=30.0)
        mgr.hooks.register("grep_search", default_timeout_secs=30.0)
        mgr.hooks.register("glob_search", default_timeout_secs=15.0)
        mgr.hooks.register("ast_search", default_timeout_secs=35.0)
        mgr.hooks.register(
            "desktop_screenshot",
            default_timeout_secs=30.0,
        )
        for name in (
            "lsp_definition",
            "lsp_references",
            "lsp_rename",
            "lsp_hover",
            "lsp_diagnostics",
        ):
            mgr.hooks.register(name, default_timeout_secs=20.0)
        mgr.hooks.register(
            "browser_use",
            max_internal_timeout_secs=3600.0,
        )

        agent_id = (self._request_context or {}).get(
            "agent_id",
            self.name,
        )
        mgr.clear_agent_tool_timeouts(agent_id)
        builtin_tools = (
            getattr(
                getattr(self._agent_config, "tools", None),
                "builtin_tools",
                None,
            )
            or {}
        )
        for tool_name, cfg in builtin_tools.items():
            t = getattr(cfg, "timeout_seconds", None)
            if t is not None and t > 0:
                mgr.set_agent_tool_timeout(
                    agent_id,
                    tool_name,
                    float(t),
                )

    # pylint: disable=too-many-nested-blocks
    def _strip_media_blocks_from_memory(self) -> int:
        """Remove media blocks (image/audio/video/DataBlock) from all messages.

        Also strips media blocks nested inside ToolResultBlock outputs.
        Inserts placeholder text when stripping leaves content empty to
        avoid malformed API requests.

        Returns:
            Total number of media blocks removed.
        """
        total_stripped = 0

        for msg in self.state.context:
            if not isinstance(msg.content, list):
                continue

            new_content = []
            stripped_this_message = 0
            for block in msg.content:
                if self._is_media_block(block):
                    total_stripped += 1
                    stripped_this_message += 1
                    continue

                btype = (
                    block.get("type")
                    if isinstance(block, dict)
                    else getattr(block, "type", None)
                )
                if btype == "tool_result":
                    output = (
                        block.get("output")
                        if isinstance(block, dict)
                        else getattr(block, "output", None)
                    )
                    if isinstance(output, list):
                        filtered = [
                            item
                            for item in output
                            if not self._is_media_block(item)
                        ]
                        stripped_count = len(output) - len(filtered)
                        total_stripped += stripped_count
                        stripped_this_message += stripped_count
                        if stripped_count > 0:
                            if isinstance(block, dict):
                                block["output"] = (
                                    filtered or MEDIA_UNSUPPORTED_PLACEHOLDER
                                )
                            else:
                                block.output = (
                                    filtered or MEDIA_UNSUPPORTED_PLACEHOLDER
                                )

                new_content.append(block)

            if not new_content and stripped_this_message > 0:
                new_content.append(
                    TextBlock(type="text", text=MEDIA_UNSUPPORTED_PLACEHOLDER),
                )

            msg.content = new_content

        return total_stripped
