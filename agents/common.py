"""Helpers shared by all agents: safe prompt data and retrieval."""

from __future__ import annotations

import json
from typing import Any

from rag.knowledge_base import Hit, format_context, get_knowledge_base, retrieval_summary
from security.injection_guard import GuardResult, guard_structure, wrap_untrusted

INTERNAL_KEYS = frozenset({"security", "grounding", "retrieved", "unverified_skills"})


def untrusted_json(label: str, data: dict[str, Any]) -> tuple[str, GuardResult]:
    """Render model-produced data for a later prompt, guarded and isolated.

    Output derived from a resume can carry injected text forward, so it is
    scanned and wrapped exactly like the resume itself.
    """
    payload = {key: value for key, value in data.items() if key not in INTERNAL_KEYS}
    cleaned, report = guard_structure(payload)
    return wrap_untrusted(label, json.dumps(cleaned, indent=2, ensure_ascii=False)), report


def retrieve_context(role: str, topic: str, kinds: tuple[str, ...], k: int = 2) -> tuple[str, list[Hit]]:
    """Retrieve reference chunks for `role`; `topic` words steer BM25 toward a chunk kind."""
    hits = get_knowledge_base().retrieve_for_role(role, topic, k=k, kinds=kinds)
    return format_context(hits), hits


def with_retrieval(result: dict[str, Any], hits: list[Hit]) -> dict[str, Any]:
    result["retrieved"] = retrieval_summary(hits)
    return result


def merge_security(*reports: GuardResult) -> dict[str, Any] | None:
    """Combine guard reports into one summary, or None when nothing was found."""
    findings = [f for report in reports for f in report.findings]
    if not findings:
        return None
    merged = GuardResult(text="", findings=findings)
    merged.neutralized = sum(r.neutralized for r in reports)
    merged.invisible_removed = sum(r.invisible_removed for r in reports)
    merged.risk_score = min(100, sum(f.weight for f in findings))
    merged.risk_level = max((r.risk_level for r in reports), key={"none": 0, "low": 1, "medium": 2, "high": 3}.get)
    return merged.summary()
