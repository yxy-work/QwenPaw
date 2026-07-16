# -*- coding: utf-8 -*-
"""知识库检索参数规范化、精确签名与保守相似判定。"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

_WHITESPACE_PATTERN = re.compile(r"\s+")
_NUMBER_PATTERN = re.compile(
    r"(?:v(?:ersion)?\s*)?\d+(?:[._/\-]\d+)*",
    flags=re.IGNORECASE,
)
_QUOTED_PHRASE_PATTERN = re.compile(
    r'"([^"]+)"|\'([^\']+)\'|“([^”]+)”|‘([^’]+)’|「([^」]+)」|『([^』]+)』',
)
_CHINESE_NEGATION_MARKERS = (
    "不包括",
    "不包含",
    "排除",
    "不是",
    "不得",
    "不要",
    "无需",
    "没有",
    "除外",
)
_LATIN_NEGATION_PATTERN = re.compile(
    r"\b(?:not|exclude|excluding|excluded|without|except)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchMatchInput:
    """一次检索用于匹配的规范化有效参数。"""

    effective_arguments: Mapping[str, Any]
    normalized_kb_name: str
    normalized_query: str
    effective_filters_signature: str


def normalize_search_text(value: str) -> str:
    """执行 NFKC、首尾清理、空白折叠与大小写归一化。"""
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def _resolve_local_ref(
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    current: Any = root_schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return schema
        current = current[part]
    return current if isinstance(current, Mapping) else schema


def _apply_defaults(
    value: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
) -> Any:
    resolved_schema = _resolve_local_ref(schema, root_schema)
    if isinstance(value, dict):
        result = deepcopy(value)
        properties = resolved_schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, property_schema_value in properties.items():
                if not isinstance(property_schema_value, Mapping):
                    continue
                property_schema = _resolve_local_ref(
                    property_schema_value,
                    root_schema,
                )
                if key not in result and "default" in property_schema:
                    result[key] = deepcopy(property_schema["default"])
                if key in result:
                    result[key] = _apply_defaults(
                        result[key],
                        property_schema,
                        root_schema,
                    )
        return result
    if isinstance(value, list):
        item_schema = resolved_schema.get("items", {})
        if isinstance(item_schema, Mapping):
            return [
                _apply_defaults(item, item_schema, root_schema)
                for item in value
            ]
    return deepcopy(value)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_search_match_input(
    arguments: Mapping[str, Any],
    input_schema: Mapping[str, Any] | None,
) -> SearchMatchInput:
    """解析 schema 默认值并构造精确/相似匹配所需参数。"""
    schema = input_schema or {}
    effective = _apply_defaults(dict(arguments), schema, schema)
    kb_name = effective.get("kb_name")
    query = effective.get("query")
    if not isinstance(kb_name, str) or not isinstance(query, str):
        raise ValueError("kb_name 与 query 必须是字符串")

    normalized_kb_name = normalize_search_text(kb_name)
    normalized_query = normalize_search_text(query)
    effective["kb_name"] = normalized_kb_name
    effective["query"] = normalized_query
    filters = {
        key: value
        for key, value in effective.items()
        if key not in {"kb_name", "query"}
    }
    return SearchMatchInput(
        effective_arguments=_canonicalize(effective),
        normalized_kb_name=normalized_kb_name,
        normalized_query=normalized_query,
        effective_filters_signature=_stable_json(filters),
    )


def build_exact_signature(match_input: SearchMatchInput) -> str:
    """返回覆盖全部有效参数的紧凑稳定签名。"""
    canonical = _stable_json(match_input.effective_arguments)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _character_bigrams(value: str) -> set[str]:
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _bigram_jaccard(left: str, right: str) -> float:
    left_bigrams = _character_bigrams(left)
    right_bigrams = _character_bigrams(right)
    union = left_bigrams | right_bigrams
    if not union:
        return 1.0 if left == right else 0.0
    return len(left_bigrams & right_bigrams) / len(union)


def _number_markers(value: str) -> tuple[str, ...]:
    return tuple(_NUMBER_PATTERN.findall(value))


def _quoted_phrases(value: str) -> tuple[str, ...]:
    phrases: list[str] = []
    for match in _QUOTED_PHRASE_PATTERN.finditer(value):
        phrase = next(group for group in match.groups() if group is not None)
        phrases.append(normalize_search_text(phrase))
    return tuple(phrases)


def _negation_markers(value: str) -> tuple[str, ...]:
    markers = [
        marker for marker in _CHINESE_NEGATION_MARKERS if marker in value
    ]
    markers.extend(match.group(0).casefold() for match in _LATIN_NEGATION_PATTERN.finditer(value))
    return tuple(markers)


def _has_new_intent_marker(left: str, right: str) -> bool:
    return any(
        extractor(left) != extractor(right)
        for extractor in (
            _number_markers,
            _quoted_phrases,
            _negation_markers,
        )
    )


def queries_are_highly_similar(
    left: str,
    right: str,
    *,
    containment_length_ratio: float = 0.80,
    bigram_jaccard_threshold: float = 0.85,
    sequence_matcher_threshold: float = 0.92,
) -> bool:
    """按保守阈值判断纯措辞改写，并放行明确新增意图。"""
    normalized_left = normalize_search_text(left)
    normalized_right = normalize_search_text(right)
    if normalized_left == normalized_right:
        return True
    if _has_new_intent_marker(normalized_left, normalized_right):
        return False

    shorter, longer = sorted(
        (normalized_left, normalized_right),
        key=len,
    )
    containment = bool(longer) and shorter in longer
    containment_ratio = len(shorter) / len(longer) if longer else 1.0
    if containment and containment_ratio >= containment_length_ratio:
        return True
    if _bigram_jaccard(normalized_left, normalized_right) >= bigram_jaccard_threshold:
        return True
    return (
        SequenceMatcher(None, normalized_left, normalized_right).ratio()
        >= sequence_matcher_threshold
    )


__all__ = [
    "SearchMatchInput",
    "build_exact_signature",
    "build_search_match_input",
    "normalize_search_text",
    "queries_are_highly_similar",
]
