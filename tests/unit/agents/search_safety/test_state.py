# -*- coding: utf-8 -*-
"""知识库检索安全契约与 reply 级状态的永久单元测试。"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from qwenpaw.agents.search_safety import (
    ContextBudgetAssessment,
    ContextWindowBudgetGuardProtocol,
    ContextWindowPreflightBlocked,
    SearchCallRecord,
    SearchSafetyDecision,
    SearchSafetyState,
    SearchSafetyStateStore,
)


def test_contract_values_and_context_assessment_are_frozen() -> None:
    assert [decision.value for decision in SearchSafetyDecision] == [
        "allowed",
        "duplicate_reused",
        "similar_query_blocked",
        "limit_reached",
        "context_budget_reached",
    ]
    assessment = ContextBudgetAssessment(
        allowed=False,
        current_input_tokens=100,
        projected_input_tokens=120,
        safe_input_limit=110,
        estimator="model_tokenizer",
        reason="projected_input_exceeds_safe_limit",
    )

    assert assessment.projected_input_tokens == 120
    with pytest.raises((AttributeError, ValidationError)):
        assessment.allowed = True


def test_preflight_exception_carries_assessment() -> None:
    assessment = ContextBudgetAssessment(
        allowed=False,
        current_input_tokens=120,
        projected_input_tokens=120,
        safe_input_limit=110,
        estimator="model_tokenizer",
        reason="actual_request_exceeds_safe_limit",
    )

    error = ContextWindowPreflightBlocked(assessment)

    assert error.assessment is assessment
    assert "actual_request_exceeds_safe_limit" in str(error)


def test_budget_guard_protocol_defines_both_preflight_layers() -> None:
    assert hasattr(ContextWindowBudgetGuardProtocol, "assess_search_preflight")
    assert hasattr(ContextWindowBudgetGuardProtocol, "assess_model_request")


def test_call_record_persists_similarity_matching_fields() -> None:
    """恢复后必须具备重建相似 query 历史所需的紧凑字段。"""
    record = SearchCallRecord(
        attempt_index=1,
        exact_signature="signature",
        normalized_kb_name="main",
        normalized_query="如何配置知识库检索",
        effective_filters_signature='{"mode":"hybrid"}',
        mcp_called=True,
    )

    restored = SearchCallRecord.model_validate_json(record.model_dump_json())

    assert restored.normalized_kb_name == "main"
    assert restored.normalized_query == "如何配置知识库检索"
    assert restored.effective_filters_signature == '{"mode":"hybrid"}'


def test_store_exposes_validated_sync_session_boundary() -> None:
    """AgentScope 同步 state API 不应访问 Store 私有字段。"""
    store = SearchSafetyStateStore()
    store.restore_sync(
        "reply-sync",
        {
            "reply_id": "reply-sync",
            "observed_attempt_count": 1,
            "admitted_attempt_count": 1,
        },
    )

    snapshot = store.snapshot_sync("reply-sync")

    assert snapshot.reply_id == "reply-sync"
    assert snapshot.admitted_attempt_count == 1


def test_legacy_state_loads_and_json_round_trips() -> None:
    legacy_payload = {"reply_id": "reply-1", "observed_attempt_count": 2}

    state = SearchSafetyState.model_validate(legacy_payload)
    restored = SearchSafetyState.model_validate_json(state.model_dump_json())
    json.dumps(state.model_dump(mode="json"))

    assert restored == state
    assert restored.admitted_attempt_count == 0
    assert restored.mcp_call_count == 0
    assert restored.force_finalize is False
    assert restored.force_finalize_reason is None
    assert restored.inflight_reserved_tokens == 0
    assert restored.calls == []
    assert restored.exact_signature_to_call_index == {}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "reply_id": "r",
            "observed_attempt_count": 11,
            "admitted_attempt_count": 11,
        },
        {
            "reply_id": "r",
            "observed_attempt_count": 0,
            "admitted_attempt_count": 1,
        },
        {
            "reply_id": "r",
            "observed_attempt_count": 1,
            "admitted_attempt_count": 1,
            "mcp_call_count": 2,
        },
    ],
)
def test_state_rejects_counter_invariant_violations(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        SearchSafetyState.model_validate(payload)


async def test_reply_change_replaces_the_only_active_state() -> None:
    store = SearchSafetyStateStore()
    await store.reserve_attempt_slot("reply-old", max_attempts_per_reply=10)

    new_state = await store.get_or_create("reply-new")
    old_state = await store.get("reply-old")

    assert new_state.reply_id == "reply-new"
    assert new_state.observed_attempt_count == 0
    assert old_state is None
    assert store.active_reply_id == "reply-new"


async def test_concurrent_attempt_allocation_never_assigns_slot_eleven() -> None:
    store = SearchSafetyStateStore()
    allocations = await asyncio.gather(
        *(
            store.reserve_attempt_slot(
                "reply-concurrent",
                max_attempts_per_reply=10,
            )
            for _ in range(15)
        ),
    )
    state = await store.get("reply-concurrent")

    assert state is not None
    assert state.observed_attempt_count == 15
    assert state.admitted_attempt_count == 10
    assert state.force_finalize is True
    assert state.force_finalize_reason == "max_attempts_per_reply_reached"
    assert sorted(
        admitted_index
        for _, admitted_index in allocations
        if admitted_index is not None
    ) == list(range(1, 11))
    assert sum(item[1] is None for item in allocations) == 5


async def test_mcp_count_cannot_exceed_admitted_attempts() -> None:
    store = SearchSafetyStateStore()
    await store.reserve_attempt_slot("reply-mcp", max_attempts_per_reply=10)

    assert await store.mark_mcp_call("reply-mcp") == 1
    with pytest.raises(RuntimeError, match="admitted"):
        await store.mark_mcp_call("reply-mcp")


async def test_token_reservation_and_release_are_atomic() -> None:
    store = SearchSafetyStateStore()
    results = await asyncio.gather(
        *(
            store.try_reserve_tokens(
                "reply-budget",
                token_count=60,
                max_inflight_tokens=100,
            )
            for _ in range(2)
        ),
    )

    assert sorted(results) == [False, True]
    state = await store.get("reply-budget")
    assert state is not None
    assert state.inflight_reserved_tokens == 60
    assert await store.release_tokens("reply-budget", 60) == 0


async def test_force_finalize_read_write_is_reply_isolated() -> None:
    store = SearchSafetyStateStore()
    await store.set_force_finalize("reply-a", "context_budget_reached")

    assert await store.get_force_finalize("reply-a") == (
        True,
        "context_budget_reached",
    )
    await store.get_or_create("reply-b")
    assert await store.get_force_finalize("reply-b") == (False, None)
    assert await store.get("reply-a") is None


async def test_stale_reply_completion_cannot_resurrect_old_state() -> None:
    store = SearchSafetyStateStore()
    await store.try_reserve_tokens("reply-old", 50)
    await store.get_or_create("reply-new")

    with pytest.raises(RuntimeError, match="reply"):
        await store.release_tokens("reply-old", 50)

    assert store.active_reply_id == "reply-new"
    assert await store.get("reply-old") is None


async def test_call_recording_builds_exact_signature_index() -> None:
    store = SearchSafetyStateStore()
    record = SearchCallRecord(
        attempt_index=1,
        decision=SearchSafetyDecision.ALLOWED,
        exact_signature="signature-1",
        reserved_tokens=500,
        mcp_called=True,
    )

    call_index = await store.record_call("reply-call", record)
    state = await store.get("reply-call")

    assert call_index == 0
    assert state is not None
    assert state.calls == [record]
    assert state.exact_signature_to_call_index == {"signature-1": 0}


async def test_store_dump_restore_and_reply_mismatch_handling() -> None:
    store = SearchSafetyStateStore()
    await store.reserve_attempt_slot("reply-persisted", 10)
    await store.try_reserve_tokens("reply-persisted", 123)
    payload = await store.dump("reply-persisted")
    json.dumps(payload)

    restored_store = SearchSafetyStateStore()
    restored = await restored_store.restore("reply-persisted", payload)
    mismatched = await restored_store.restore("reply-current", payload)

    assert restored.reply_id == "reply-persisted"
    assert restored.admitted_attempt_count == 1
    assert restored.inflight_reserved_tokens == 123
    assert mismatched.reply_id == "reply-current"
    assert mismatched.observed_attempt_count == 0
    assert await restored_store.get("reply-persisted") is None
