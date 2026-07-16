# -*- coding: utf-8 -*-
"""知识库检索安全配置的永久单元测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from qwenpaw.config.config import (
    AgentsRunningConfig,
    SearchKnowledgeBaseSafetyConfig,
)


def test_search_safety_defaults_are_attached_to_running_config() -> None:
    config = AgentsRunningConfig()
    safety = config.search_knowledgebase_safety

    assert safety == SearchKnowledgeBaseSafetyConfig()
    assert safety.enabled is True
    assert safety.max_attempts_per_reply == 10
    assert safety.exact_dedup_enabled is True
    assert safety.similar_query_guard_enabled is True
    assert safety.similar_query_observe_only is True
    assert safety.containment_length_ratio == 0.80
    assert safety.bigram_jaccard_threshold == 0.85
    assert safety.sequence_matcher_threshold == 0.92
    assert safety.context_preflight_enabled is True
    assert safety.default_reserved_completion_tokens == 8192
    assert safety.safety_margin_ratio == 0.03
    assert safety.minimum_safety_margin_tokens == 4096
    assert safety.tool_message_overhead_tokens == 512


def test_legacy_running_config_without_safety_block_loads_defaults() -> None:
    config = AgentsRunningConfig.model_validate({"max_iters": 7})

    assert config.max_iters == 7
    assert config.search_knowledgebase_safety.max_attempts_per_reply == 10


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_attempts_per_reply", 0),
        ("max_attempts_per_reply", 51),
        ("containment_length_ratio", -0.01),
        ("containment_length_ratio", 1.01),
        ("bigram_jaccard_threshold", -0.01),
        ("bigram_jaccard_threshold", 1.01),
        ("sequence_matcher_threshold", -0.01),
        ("sequence_matcher_threshold", 1.01),
        ("default_reserved_completion_tokens", -1),
        ("safety_margin_ratio", -0.01),
        ("safety_margin_ratio", 1.01),
        ("minimum_safety_margin_tokens", -1),
        ("tool_message_overhead_tokens", -1),
    ],
)
def test_search_safety_rejects_invalid_boundaries(
    field_name: str,
    invalid_value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseSafetyConfig(
            **{field_name: invalid_value},
        )


def test_search_safety_does_not_define_retrieval_or_detail_controls() -> None:
    field_names = set(SearchKnowledgeBaseSafetyConfig.model_fields)

    assert "response_detail" not in field_names
    assert not field_names.intersection(
        {
            "dense_top_k",
            "bm25_top_k",
            "rerank_top_k",
            "final_top_n",
        },
    )
