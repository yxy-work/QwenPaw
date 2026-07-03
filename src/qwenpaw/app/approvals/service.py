# -*- coding: utf-8 -*-
"""Approval service for sensitive tool execution.

The ``ApprovalService`` is the single central store for pending /
completed approval records.  Approval is granted exclusively via
the ``/daemon approve`` command in the chat interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...constant import TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS
from ...security.tool_guard.approval import ApprovalDecision
from ...schemas import (
    ContentType,
    Event,
    MessageType,
    Role,
    RunStatus,
    TextContent,
)
from .display import approval_display_fields
from .models import ApprovalRequestSummary

if TYPE_CHECKING:
    from ...security.tool_guard.models import ToolGuardResult

logger = logging.getLogger(__name__)

_GC_MAX_AGE_SECONDS = 3600.0
_GC_MAX_COMPLETED = 500
_GC_PENDING_MAX_AGE_SECONDS = 1800.0
_GC_MAX_PENDING = 200


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass
class PendingApproval:
    """In-memory record for one pending approval."""

    request_id: str
    session_id: str
    root_session_id: str  # Root session for cross-session approval routing
    owner_agent_id: str  # Conversation owner/root agent
    user_id: str
    channel: str
    agent_id: str  # Which agent is requesting approval
    tool_name: str
    created_at: float
    future: asyncio.Future[ApprovalDecision]
    timeout_seconds: float = 300.0  # Approval timeout in seconds
    status: str = "pending"
    resolved_at: float | None = None
    result_summary: str = ""
    findings_count: int = 0
    severity: str = "medium"  # For frontend display
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------


class ApprovalService:
    """Global singleton approval service.

    Manages all tool approval requests across sessions and agents.
    Approval is resolved via ``/approval`` control command.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending: dict[str, PendingApproval] = {}
        self._channel_manager: Any | None = None
        self._channel_managers: list[Any] = []

    def set_channel_manager(self, channel_manager: Any) -> None:
        """Store a reference to the channel manager for push notifications."""
        self._channel_manager = channel_manager
        if channel_manager not in self._channel_managers:
            self._channel_managers.append(channel_manager)

    # ------------------------------------------------------------------
    # Core approval lifecycle
    # ------------------------------------------------------------------

    async def create_pending(
        self,
        *,
        session_id: str,
        root_session_id: str,
        owner_agent_id: str,
        user_id: str,
        channel: str,
        agent_id: str,
        tool_name: str,
        result: "ToolGuardResult",
        timeout_seconds: float = TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS,
        extra: dict[str, Any] | None = None,
    ) -> PendingApproval:
        """Create a pending approval record and return it."""
        from ...security.tool_guard.approval import format_findings_summary

        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()

        pending = PendingApproval(
            request_id=request_id,
            session_id=session_id,
            root_session_id=root_session_id,
            owner_agent_id=owner_agent_id,
            user_id=user_id,
            channel=channel,
            agent_id=agent_id,
            tool_name=tool_name,
            created_at=time.time(),
            future=loop.create_future(),
            timeout_seconds=timeout_seconds,
            result_summary=format_findings_summary(result),
            findings_count=result.findings_count,
            severity=result.max_severity.value,
            extra=dict(extra or {}),
        )

        async with self._lock:
            self._pending[request_id] = pending
            self._gc_pending_locked()

        logger.info(
            "Approval pending created: request_id=%s agent_id=%s tool=%s "
            "severity=%s session=%s root=%s",
            request_id[:8],
            agent_id,
            tool_name,
            pending.severity,
            session_id[:8],
            root_session_id[:8],
        )
        self._schedule_channel_notification(pending)

        return pending

    async def create_pending_summary(
        self,
        *,
        session_id: str,
        root_session_id: str,
        owner_agent_id: str,
        user_id: str,
        channel: str,
        agent_id: str,
        summary: ApprovalRequestSummary,
        timeout_seconds: float = TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS,
        extra: dict[str, Any] | None = None,
    ) -> PendingApproval:
        """Create a pending approval from a generic summary."""
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        merged_extra = {
            "source_type": summary.source_type,
            **summary.payload,
            **dict(extra or {}),
        }
        pending = PendingApproval(
            request_id=request_id,
            session_id=session_id,
            root_session_id=root_session_id,
            owner_agent_id=owner_agent_id,
            user_id=user_id,
            channel=channel,
            agent_id=agent_id,
            tool_name=summary.name,
            created_at=time.time(),
            future=loop.create_future(),
            timeout_seconds=timeout_seconds,
            result_summary=summary.result_summary,
            findings_count=summary.findings_count,
            severity=summary.severity,
            extra=merged_extra,
        )
        async with self._lock:
            self._pending[request_id] = pending
            self._gc_pending_locked()
        logger.info(
            "Generic approval pending created: request_id=%s agent_id=%s "
            "name=%s source=%s session=%s root=%s",
            request_id[:8],
            agent_id,
            summary.name,
            summary.source_type,
            session_id[:8],
            root_session_id[:8],
        )
        self._schedule_channel_notification(pending)
        return pending

    # ------------------------------------------------------------------
    # Channel notification
    # ------------------------------------------------------------------

    def _schedule_channel_notification(
        self,
        pending: PendingApproval,
    ) -> None:
        """异步通知外部 Channel 展示审批卡片。"""
        if self._channel_manager is None:
            return
        if not pending.channel or not pending.session_id:
            return
        task = asyncio.create_task(self._notify_channel_pending(pending))
        task.add_done_callback(self._log_channel_notification_result)

    @staticmethod
    def _log_channel_notification_result(
        task: "asyncio.Task[None]",
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("Approval channel notification cancelled")
        except Exception:
            logger.exception("Approval channel notification failed")

    async def _notify_channel_pending(
        self,
        pending: PendingApproval,
    ) -> None:
        """向 ChannelManager 推送一个可由交互卡片处理器识别的事件。"""
        channel_manager = await self._resolve_channel_manager(
            pending.channel,
        )
        if channel_manager is None:
            logger.warning(
                "Approval channel notification skipped: channel not found: %s",
                pending.channel,
            )
            return
        event = self._build_approval_event(pending)
        await channel_manager.send_event(
            channel=pending.channel,
            user_id=pending.user_id,
            session_id=pending.session_id,
            event=event,
            meta={},
        )

    async def _resolve_channel_manager(self, channel: str) -> Any | None:
        """选择包含目标 channel 的 ChannelManager。"""
        managers = list(self._channel_managers)
        if self._channel_manager is not None:
            managers.append(self._channel_manager)

        seen: set[int] = set()
        legacy_fallback: Any | None = None
        for manager in reversed(managers):
            manager_id = id(manager)
            if manager_id in seen:
                continue
            seen.add(manager_id)
            get_channel = getattr(manager, "get_channel", None)
            if get_channel is None:
                if legacy_fallback is None:
                    legacy_fallback = manager
                continue
            try:
                result = get_channel(channel)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception:
                logger.debug(
                    "Approval channel manager probe failed for channel=%s",
                    channel,
                    exc_info=True,
                )
                continue
            if result is not None:
                return manager
        return legacy_fallback

    @staticmethod
    def _build_approval_event(pending: PendingApproval) -> Event:
        """构造外部 Channel 可渲染的审批提示事件。"""
        display = approval_display_fields(pending)
        tool_name = display["tool_display_name"]
        tool_source = display["tool_source"]
        body_text = "\n".join(
            [
                "Tool approval required.",
                f"Tool: {tool_name}",
                f"Source: {tool_source}",
                f"Severity: {pending.severity}",
                f"Request ID: {pending.request_id}",
                pending.result_summary,
                "",
                f"Approve: /approval approve {pending.request_id}",
                f"Deny: /approval deny {pending.request_id}",
            ],
        ).strip()
        return Event(
            object="message",
            status=RunStatus.Completed,
            type=MessageType.MESSAGE,
            role=Role.ASSISTANT,
            content=[
                TextContent(
                    type=ContentType.TEXT,
                    text=body_text,
                    status=RunStatus.Completed,
                ),
            ],
            metadata={
                "metadata": {
                    "message_type": "tool_guard_approval",
                    "approval_request_id": pending.request_id,
                    "tool_name": tool_name,
                    "tool_source": tool_source,
                    "severity": pending.severity,
                    "findings_count": pending.findings_count,
                    "result_summary": pending.result_summary,
                    "agent_id": pending.agent_id,
                    "session_id": pending.session_id,
                    "root_session_id": pending.root_session_id,
                },
            },
        )

    async def resolve_request(
        self,
        request_id: str,
        decision: ApprovalDecision,
    ) -> PendingApproval | None:
        """Resolve pending approval by setting Future result."""
        async with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                logger.warning(
                    "Approval request %s not found (already resolved?)",
                    request_id[:8],
                )
                return None

            pending.status = decision.value
            pending.resolved_at = time.time()

        # Set Future result outside lock
        if not pending.future.done():
            pending.future.set_result(decision)

        logger.info(
            "Approval request %s resolved: decision=%s tool=%s",
            request_id[:8],
            decision.value,
            pending.tool_name,
        )

        return pending

    async def get_request(self, request_id: str) -> PendingApproval | None:
        """Get a pending request by id."""
        async with self._lock:
            return self._pending.get(request_id)

    async def get_pending_by_session(
        self,
        session_id: str,
    ) -> PendingApproval | None:
        """Return the next pending approval for *session_id* (FIFO).

        Pending approvals are consumed in creation order, so repeated
        ``/approve`` inputs walk the queue from oldest to newest.
        """
        async with self._lock:
            for pending in self._pending.values():
                if (
                    pending.session_id == session_id
                    and pending.status == "pending"
                ):
                    return pending
        return None

    async def get_all_pending_by_session(
        self,
        session_id: str,
    ) -> list[PendingApproval]:
        """Return all pending approvals for *session_id* (FIFO order)."""
        async with self._lock:
            return [
                p
                for p in self._pending.values()
                if p.session_id == session_id and p.status == "pending"
            ]

    async def list_pending_by_session(
        self,
        session_id: str,
        include_subagents: bool = True,  # pylint: disable=unused-argument
    ) -> list[PendingApproval]:
        """List all pending approvals for a session (FIFO order).

        Args:
            session_id: Session ID to filter
            include_subagents: If False, exclude sub-agent approvals (future)

        Returns:
            List of pending approvals sorted by creation time
        """
        async with self._lock:
            result = [
                p
                for p in self._pending.values()
                if p.session_id == session_id and p.status == "pending"
            ]
            return sorted(result, key=lambda p: p.created_at)

    async def get_pending_by_root_session(
        self,
        root_session_id: str,
    ) -> list[PendingApproval]:
        """Get all pending approvals for root session and its children.

        Args:
            root_session_id: Root session ID

        Returns:
            List of pending approvals sorted by creation time (FIFO)
        """
        async with self._lock:
            result = [
                p
                for p in self._pending.values()
                if p.root_session_id == root_session_id
                and p.status == "pending"
            ]
            return sorted(result, key=lambda p: p.created_at)

    async def get_all_pending_by_agent(
        self,
        agent_id: str,
    ) -> list[PendingApproval]:
        """Get all pending approvals for an agent (across all sessions).

        Used by /approval list --all command.

        Args:
            agent_id: Agent ID

        Returns:
            List of pending approvals sorted by creation time (FIFO)
        """
        async with self._lock:
            result = [
                p
                for p in self._pending.values()
                if p.agent_id == agent_id and p.status == "pending"
            ]
            return sorted(result, key=lambda p: p.created_at)

    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: float,
    ) -> ApprovalDecision:
        """Block and wait for approval decision with timeout.

        Args:
            request_id: Approval request ID
            timeout_seconds: Maximum wait time in seconds

        Returns:
            ApprovalDecision (APPROVED/DENIED/TIMEOUT)

        Raises:
            ValueError: If request_id not found
        """
        async with self._lock:
            pending = self._pending.get(request_id)

        if pending is None:
            raise ValueError(f"Approval request {request_id} not found")

        try:
            decision = await asyncio.wait_for(
                pending.future,
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            decision = ApprovalDecision.TIMEOUT
            await self.resolve_request(request_id, decision)

        return decision

    async def cancel_stale_pending_for_tool_call(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> int:
        """Cancel pending approvals whose stored tool_call id matches.

        When a tool call is replayed (e.g. after approval triggers
        sibling replay), the guard may create a *new* pending for the
        same logical tool call.  This method cancels the old pending
        first so orphaned records don't accumulate.

        Returns the number of records cancelled.
        """
        now = time.time()
        cancelled = 0
        async with self._lock:
            to_cancel = [
                k
                for k, p in self._pending.items()
                if p.session_id == session_id
                and p.status == "pending"
                and isinstance(p.extra.get("tool_call"), dict)
                and p.extra["tool_call"].get("id") == tool_call_id
            ]
            for k in to_cancel:
                pending = self._pending.pop(k)
                if not pending.future.done():
                    pending.future.set_result(ApprovalDecision.TIMEOUT)
                pending.status = "superseded"
                pending.resolved_at = now
                cancelled += 1
        if cancelled:
            logger.info(
                "Tool guard: cancelled %d stale pending approval(s) "
                "for tool_call %s (session %s)",
                cancelled,
                tool_call_id,
                session_id[:8],
            )
        return cancelled

    async def cancel_all_pending_by_root_session(
        self,
        root_session_id: str,
    ) -> int:
        """Cancel all pending approvals for root session and its children.

        Called when user stops/cancels a task (e.g., /stop command or
        SSE disconnect). Auto-denies all pending approvals to unblock
        waiting tasks.

        Args:
            root_session_id: Root session ID

        Returns:
            Number of approvals cancelled
        """
        now = time.time()
        cancelled = 0
        async with self._lock:
            to_cancel = [
                k
                for k, p in self._pending.items()
                if p.root_session_id == root_session_id
                and p.status == "pending"
            ]
            for k in to_cancel:
                pending = self._pending.pop(k)
                if not pending.future.done():
                    pending.future.set_result(ApprovalDecision.DENIED)
                pending.status = "cancelled"
                pending.resolved_at = now
                cancelled += 1
        if cancelled:
            logger.info(
                "Cancelled %d pending approval(s) for root session %s",
                cancelled,
                root_session_id[:8]
                if len(root_session_id) >= 8
                else root_session_id,
            )
        return cancelled

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def _gc_pending_locked(self) -> None:
        """Evict stale pending records whose futures were never resolved.

        Caller must hold ``_lock``.
        """
        now = time.time()
        expired = [
            k
            for k, v in self._pending.items()
            if now - v.created_at > _GC_PENDING_MAX_AGE_SECONDS
        ]
        for k in expired:
            pending = self._pending.pop(k)
            if not pending.future.done():
                pending.future.set_result(ApprovalDecision.TIMEOUT)
            pending.status = "timeout"
            pending.resolved_at = now

        overflow = len(self._pending) - _GC_MAX_PENDING
        if overflow <= 0:
            return
        ordered = sorted(
            self._pending.items(),
            key=lambda item: item[1].created_at,
        )
        for key, pending in ordered[:overflow]:
            del self._pending[key]
            if not pending.future.done():
                pending.future.set_result(ApprovalDecision.TIMEOUT)
            pending.status = "timeout"
            pending.resolved_at = now


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_approval_service: ApprovalService | None = None


def get_approval_service() -> ApprovalService:
    """Return the process-wide approval service singleton."""
    global _approval_service
    if _approval_service is None:
        _approval_service = ApprovalService()
    return _approval_service
