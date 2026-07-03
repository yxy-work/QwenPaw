# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""WeCom tool-guard approval card (self-contained).

Builders + callback parser + outbound ``render`` + inbound ``handle``
all live here; the dispatcher reads the module-level metadata
(``NAME`` / ``MESSAGE_TYPE`` / ``TASK_ID_PREFIX``) plus ``render`` /
``handle`` to wire it in.

Refs: https://developer.work.weixin.qq.com/document/path/101032
      https://developer.work.weixin.qq.com/document/path/101027
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from . import context

if TYPE_CHECKING:
    from ..channel import WecomChannel

logger = logging.getLogger(__name__)


# =====================================================================
# Module-level metadata (read by the dispatcher when registering)
# =====================================================================

NAME = "tool_guard_approval"

# Outbound metadata.message_type that triggers this card kind.
MESSAGE_TYPE = "tool_guard_approval"

# Unique prefix embedded in ``task_id`` so the dispatcher can route the
# inbound callback to this card kind.  ``request_id`` is a UUID4
# (hex + dashes), all of which are valid task_id chars per WeCom spec
# (``[0-9a-zA-Z_\-@]``, ≤128 bytes), so no sanitisation is needed.
TASK_ID_PREFIX = "tg_approval_"


# =====================================================================
# Constants (internal)
# =====================================================================

# Button key prefixes encoded into the JSON payload of each button.
APPROVE_KEY = "approve"
DENY_KEY = "deny"

# Placeholder url for the resolved card's required card_action.  WeCom
# rejects ``text_notice`` cards without a card_action of type 1 or 2.
_RESOLVED_CARD_URL = "https://qwenpaw.agentscope.io"
_APPROVAL_CARD_ICON_URL = (
    "https://img.icons8.com/ios-filled/100/2563eb/shield.png"
)
_APPROVAL_FALLBACK_DESC = (
    "Tool '{tool_name}' requires user approval per governance policy."
)


# =====================================================================
# Builders
# =====================================================================


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _approval_desc(tool_name: str, approval_summary: str = "") -> str:
    """提取卡片可展示的审批说明，过滤操作元数据。"""
    for raw_line in approval_summary.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("Request ID:", "Approve:", "Deny:")):
            continue
        line = re.sub(r"^[-*·]\s*(?:\[[A-Z]+\]\s*)?", "", line).strip()
        if line:
            return line
    return _APPROVAL_FALLBACK_DESC.format(tool_name=tool_name)


def _build_button_key(
    action: str,
    request_id: str,
    tool_name: str,
    severity: str,
    session_ctx: Dict[str, Any],
) -> str:
    """Encode action + ctx into a button ``key`` (≤1024 bytes per WeCom);
    raises :class:`ValueError` when the payload would overflow.
    """
    payload = json.dumps(
        {
            "a": action,
            "rid": request_id,
            "tool": tool_name,
            "sev": severity,
            **session_ctx,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded_len = len(payload.encode("utf-8"))
    if encoded_len > 1024:
        raise ValueError(
            f"button key payload too large: {encoded_len} bytes (limit 1024)",
        )
    return payload


def build_approval_card(
    *,
    request_id: str,
    tool_name: str,
    severity: str,
    approval_summary: str = "",
    session_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``button_interaction`` approval card."""
    severity_lower = (severity or "medium").lower()
    ctx = session_ctx or {}

    return {
        "card_type": "button_interaction",
        "task_id": f"{TASK_ID_PREFIX}{request_id}",
        "source": {
            "icon_url": _APPROVAL_CARD_ICON_URL,
            "desc": "QwenPaw",
        },
        "main_title": {
            "title": "工具调用许可",
            "desc": _approval_desc(tool_name, approval_summary),
        },
        # button_list MUST live at the root.  Do NOT wrap it in
        # card_action (that field is for whole-card click-to-jump and
        # requires a url when type=1).
        "button_list": [
            {
                "text": "同意",
                "style": 1,
                "key": _build_button_key(
                    APPROVE_KEY,
                    request_id,
                    tool_name,
                    severity_lower,
                    ctx,
                ),
            },
            {
                "text": "拒绝",
                "style": 2,
                "key": _build_button_key(
                    DENY_KEY,
                    request_id,
                    tool_name,
                    severity_lower,
                    ctx,
                ),
            },
        ],
    }


def build_resolved_card(
    *,
    task_id: str,
    action: str,
) -> Dict[str, Any]:
    """构造点击后的极简占位卡，移除按钮和敏感元数据。

    WeCom requires ``card_action`` on ``text_notice`` cards, with
    ``type`` in {1, 2} (0 is rejected by the bot endpoint).  We provide
    a project URL so it stays meaningful when clicked.
    """
    if action == APPROVE_KEY:
        title = "审批已通过"
    elif action == DENY_KEY:
        title = "审批已拒绝"
    else:
        title = "审批已处理"

    return {
        "card_type": "text_notice",
        "task_id": task_id,
        "main_title": {
            "title": _truncate(title, 36),
            "desc": "",
        },
        "card_action": {"type": 1, "url": _RESOLVED_CARD_URL},
    }


# =====================================================================
# Parser
# =====================================================================


def parse_card_event(
    event_body: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract tool-guard fields from a ``template_card_event`` callback.

    Returns ``None`` when ``event_key`` is malformed or the action is
    unknown.  Prefix routing is the dispatcher's job.
    """
    event = event_body.get("event") or {}
    tce = event.get("template_card_event") or event

    event_key = str(tce.get("event_key") or "")
    try:
        ctx = json.loads(event_key)
    except (json.JSONDecodeError, TypeError):
        return None

    action = str(ctx.get("a") or "")
    if action not in (APPROVE_KEY, DENY_KEY):
        return None

    from_info = event_body.get("from") or {}
    return {
        "action": action,
        "request_id": str(ctx.get("rid") or ""),
        "task_id": str(tce.get("task_id") or ""),
        "tool_name": str(ctx.get("tool") or ""),
        "severity": str(ctx.get("sev") or "medium"),
        "session_ctx": {
            k: v
            for k, v in ctx.items()
            if k not in ("a", "rid", "tool", "sev")
        },
        "user_id": str(from_info.get("userid") or ""),
    }


# =====================================================================
# Outbound: render
# =====================================================================


async def render(
    channel: "WecomChannel",
    to_handle: str,
    event: Any,
    send_meta: Dict[str, Any],
    meta: Dict[str, Any],
) -> bool:
    """Render a tool-guard event as a button_interaction card.

    Streams the full guard details first (reusing the active processing
    stream when present, to avoid leaving an empty bubble), then posts
    the approval card.
    """
    request_id = str(meta.get("approval_request_id") or "")
    if not request_id:
        return False

    if not channel.enabled or not channel._client:
        return False

    frame = send_meta.get("wecom_frame")
    chatid = str(send_meta.get("wecom_chatid") or "")
    if not frame and not chatid:
        logger.warning(
            "wecom approval card: no frame/chatid for to_handle=%s",
            (to_handle or "")[:40],
        )
        return False

    body_text = context.extract_body_text(getattr(event, "content", None))
    session_ctx = context.build_session_ctx(to_handle, send_meta)

    try:
        template_card = build_approval_card(
            request_id=request_id,
            tool_name=str(meta.get("tool_name") or "tool"),
            severity=str(meta.get("severity") or "medium"),
            approval_summary=str(meta.get("result_summary") or body_text),
            session_ctx=session_ctx,
        )
    except ValueError as exc:
        # Skip the card and let default text rendering take over.
        logger.warning(
            "wecom approval card: %s; skipping card for request_id=%s",
            exc,
            request_id[:8],
        )
        return False

    # Stream the guard details first only when replying to an existing frame.
    # Proactive approval pushes send a single interactive card.  In that path,
    # close the earlier "thinking" placeholder so the post-approval answer
    # starts below the approval status instead of overwriting an old bubble.
    if frame and not send_meta.get("_skip_stream_detail"):
        await context.send_stream_detail(channel, frame, send_meta, body_text)
    elif not frame:
        discard_processing = getattr(
            channel,
            "discard_processing_stream_for_session",
            None,
        )
        session_id = str(session_ctx.get("session_id") or "")
        if discard_processing and session_id:
            await discard_processing(session_id)
    try:
        if frame:
            await channel._client.reply_template_card(
                frame,
                template_card,
            )
        else:
            await channel._client.send_message(
                chatid,
                {
                    "msgtype": "template_card",
                    "template_card": template_card,
                },
            )
        logger.info(
            "wecom approval card sent: request_id=%s tool=%s",
            request_id[:8],
            meta.get("tool_name", ""),
        )
        return True
    except Exception:
        logger.exception(
            "wecom approval card send failed: request_id=%s",
            request_id[:8],
        )
        return False


# =====================================================================
# Inbound: handle
# =====================================================================


async def handle(
    channel: "WecomChannel",
    frame: Any,
) -> None:
    """Process a tool-guard ``template_card_event`` callback."""
    body = frame.get("body") or {} if isinstance(frame, dict) else {}
    parsed = parse_card_event(body)
    if not parsed:
        return

    action = parsed["action"]
    request_id = parsed["request_id"]
    task_id = parsed["task_id"]
    user_id = parsed.get("user_id") or ""

    logger.info(
        "wecom card event: action=%s request_id=%s user=%s",
        action,
        request_id[:8],
        user_id[:20],
    )

    # 1. Dismiss the interactive card surface first (must be <5s).
    await _dismiss_approval_card(
        channel,
        frame,
        task_id,
        action,
    )

    # 2. Send the visible status before resolving the pending future.  This
    # keeps the programmatic approval result above the agent's follow-up reply.
    await _resolve_card_approval(
        channel,
        action=action,
        request_id=request_id,
        session_ctx=parsed.get("session_ctx") or {},
        user_id=user_id,
    )


async def _dismiss_approval_card(
    channel: "WecomChannel",
    frame: Any,
    task_id: str,
    action: str,
) -> None:
    """将审批交互卡替换为不可点击的极简已处理占位。"""
    if not channel._client:
        return

    resolved_card = build_resolved_card(
        task_id=task_id,
        action=action,
    )

    try:
        await channel._client.update_template_card(
            frame,
            resolved_card,
        )
        logger.info(
            "wecom approval card dismissed: task_id=%s action=%s",
            task_id[:20],
            action,
        )
    except Exception:
        logger.exception(
            "wecom approval card dismiss failed: task_id=%s",
            task_id[:20],
        )


async def _resolve_card_approval(
    channel: "WecomChannel",
    *,
    action: str,
    request_id: str,
    session_ctx: Dict[str, Any],
    user_id: str,
) -> None:
    """先发送 WeCom 审批状态，再释放等待中的审批 future。"""
    from qwenpaw.app.approvals import get_approval_service
    from qwenpaw.security.tool_guard.approval import ApprovalDecision

    svc = get_approval_service()
    pending = await svc.get_request(request_id)
    if pending is None:
        logger.info(
            "wecom card action ignored: request already resolved request=%s",
            request_id[:8],
        )
        return

    decision = (
        ApprovalDecision.APPROVED
        if action == APPROVE_KEY
        else ApprovalDecision.DENIED
    )
    status_text = _build_resolution_status(
        action=action,
        tool_name=pending.tool_name,
        request_id=request_id,
    )
    session_id = str(session_ctx.get("session_id") or "")
    chatid = str(session_ctx.get("chatid") or "")
    chat_type = str(session_ctx.get("chat_type") or "single")
    to_handle = session_id or (
        f"wecom:group:{chatid}" if chat_type == "group" else f"wecom:{chatid}"
    )

    try:
        await channel.send(
            to_handle,
            status_text,
            {
                "wecom_sender_id": str(
                    session_ctx.get("sender_id") or user_id or "",
                ),
                "wecom_chatid": chatid,
                "wecom_chat_type": chat_type,
                "from_card_action": True,
            },
        )
    except Exception:
        logger.exception(
            "wecom card action: status send failed request=%s",
            request_id[:8],
        )

    resolved = await svc.resolve_request(request_id, decision)
    if resolved is None:
        logger.info(
            "wecom card action: request resolved concurrently request=%s",
            request_id[:8],
        )
    else:
        logger.info(
            "wecom card action resolved: action=%s request=%s session=%s",
            action,
            request_id[:8],
            session_id[:12],
        )


def _build_resolution_status(
    *,
    action: str,
    tool_name: str,
    request_id: str,
) -> str:
    """构造 WeCom 卡片点击后的程序化审批状态文本。"""
    if action == APPROVE_KEY:
        return (
            "**工具已批准**\n\n"
            f"- 工具: `{tool_name}`\n"
            f"- 请求 ID: `{request_id[:16]}`\n"
            "- 状态: 正在执行..."
        )
    return (
        "**工具已拒绝**\n\n"
        f"- 工具: `{tool_name}`\n"
        f"- 请求 ID: `{request_id[:16]}`\n"
        "- 原因: 用户拒绝"
    )
