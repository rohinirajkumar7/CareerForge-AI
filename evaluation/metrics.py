"""Offline evaluation: retrieval quality, injection defence and PII redaction.

Everything here runs without an API key. The datasets in evaluation/datasets were
written by the project author alongside the rules, so the numbers are a regression
guard and a sanity check, not an independent benchmark. Add real examples to make
them harder.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from privacy.redactor import redact
from rag.bm25 import tokenize
from rag.knowledge_base import DEFAULT_MIN_SCORE, KnowledgeBase
from security.injection_guard import guard_untrusted

DATASETS = Path(__file__).resolve().parent / "datasets"
K_VALUES = (1, 3, 5)
THRESHOLD_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def load_dataset(name: str) -> dict[str, Any]:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- retrieval


def _overlap_scores(kb: KnowledgeBase, query: str) -> list[float]:
    """Baseline: how many distinct query terms appear in the chunk (no idf, no length norm)."""
    terms = set(tokenize(query))
    return [float(len(terms & set(kb._index_tokens(chunk)))) for chunk in kb.chunks]


def _rank(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _score_ranking(ranked_ids: list[str], relevant: set[str]) -> dict[str, Any]:
    first = next((pos for pos, cid in enumerate(ranked_ids, 1) if cid in relevant), None)
    row: dict[str, Any] = {"first_relevant_rank": first, "top3": ranked_ids[:3]}
    for k in K_VALUES:
        found = relevant & set(ranked_ids[:k])
        row[f"hit@{k}"] = 1.0 if found else 0.0
        row[f"recall@{k}"] = len(found) / len(relevant)
    row["rr"] = 1.0 / first if first else 0.0
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [f"hit@{k}" for k in K_VALUES] + [f"recall@{k}" for k in K_VALUES] + ["rr"]
    summary = {key: round(mean(row[key] for row in rows), 3) for key in keys}
    summary["mrr"] = summary.pop("rr")
    return summary


def evaluate_retrieval(kb: KnowledgeBase, dataset: dict[str, Any]) -> dict[str, Any]:
    ids = [c.id for c in kb.chunks]
    per_query: list[dict[str, Any]] = []
    bm25_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for item in dataset["queries"]:
        relevant = set(item["relevant"])
        bm_scores = kb._bm25.scores(tokenize(item["query"]))
        bm = _score_ranking([ids[i] for i in _rank(bm_scores)], relevant)
        ov = _score_ranking([ids[i] for i in _rank(_overlap_scores(kb, item["query"]))], relevant)
        bm25_rows.append(bm)
        base_rows.append(ov)
        per_query.append(
            {
                "query": item["query"],
                "relevant": item["relevant"],
                "bm25_rank": bm["first_relevant_rank"],
                "baseline_rank": ov["first_relevant_rank"],
                "bm25_top3": bm["top3"],
                "top_score": round(max(bm_scores), 2),
            }
        )

    on_topic = [max(kb._bm25.scores(tokenize(q["query"]))) for q in dataset["queries"]]
    off_topic = [max(kb._bm25.scores(tokenize(q)) or [0.0]) for q in dataset["off_topic"]]
    sweep = [
        {
            "min_score": threshold,
            "answered_on_topic": round(mean(s >= threshold for s in on_topic), 3),
            "abstained_off_topic": round(mean(s < threshold for s in off_topic), 3),
        }
        for threshold in THRESHOLD_SWEEP
    ]
    return {
        "queries": len(per_query),
        "chunks": len(kb.chunks),
        "bm25": _aggregate(bm25_rows),
        "baseline_overlap": _aggregate(base_rows),
        "per_query": per_query,
        "threshold_sweep": sweep,
        "configured_min_score": DEFAULT_MIN_SCORE,
        "off_topic_top_scores": [round(s, 2) for s in off_topic],
    }


def evaluate_role_resolution(kb: KnowledgeBase, dataset: dict[str, Any]) -> dict[str, Any]:
    """Does the role gate pick the right reference chunks, and abstain for roles we do not cover?"""
    known_rows: list[dict[str, Any]] = []
    for role, expected in dataset["known"]:
        hits = kb.retrieve_for_role(role, "required skills tools", k=1, kinds=("skills",))
        got = hits[0].chunk.id.split(":")[0] if hits else None
        known_rows.append({"role": role, "expected": expected, "got": got, "correct": got == expected})
    unknown_rows: list[dict[str, Any]] = []
    for role in dataset["unknown"]:
        hits = kb.retrieve_for_role(role, "required skills tools", k=1, kinds=("skills",))
        unknown_rows.append({"role": role, "got": hits[0].chunk.id if hits else None, "abstained": not hits})
    return {
        "known_accuracy": round(mean(r["correct"] for r in known_rows), 3),
        "unknown_abstain_rate": round(mean(r["abstained"] for r in unknown_rows), 3),
        "known": known_rows,
        "unknown": unknown_rows,
    }


# ---------------------------------------------------------------- injection


def evaluate_injection(dataset: dict[str, Any]) -> dict[str, Any]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    missed: list[dict[str, str]] = []
    neutralized = 0
    for case in dataset["attacks"]:
        result = guard_untrusted(case["text"])
        by_category[case["category"]].append(result.flagged)
        neutralized += result.neutralized > 0
        if not result.flagged:
            missed.append({"category": case["category"], "text": case["text"], "risk": result.risk_level})

    false_positives: list[dict[str, Any]] = []
    for case in dataset["benign"]:
        result = guard_untrusted(case["text"])
        if result.flagged:
            false_positives.append(
                {
                    "text": case["text"],
                    "risk": result.risk_level,
                    "rules": [f.rule for f in result.findings],
                    "hard": case.get("hard", False),
                }
            )

    attacks = len(dataset["attacks"])
    benign = len(dataset["benign"])
    flagged = attacks - len(missed)
    return {
        "attacks": attacks,
        "benign": benign,
        "detection_rate": round(flagged / attacks, 3),
        "neutralization_rate": round(neutralized / attacks, 3),
        "false_positive_rate": round(len(false_positives) / benign, 3),
        "by_category": [
            {"category": cat, "cases": len(flags), "detected": sum(flags), "rate": round(sum(flags) / len(flags), 3)}
            for cat, flags in sorted(by_category.items())
        ],
        "missed": missed,
        "false_positives": false_positives,
    }


# ---------------------------------------------------------------- privacy


def _leaked(kind: str, value: str, text: str) -> bool:
    if kind == "NAME":
        return re.search(rf"\b{re.escape(value)}\b", text, re.IGNORECASE) is not None
    if kind == "PHONE":
        digits = re.sub(r"\D", "", value)
        return digits in re.sub(r"\D", "", text)
    return value.lower() in text.lower()


def evaluate_privacy(dataset: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, int] = defaultdict(int)
    caught: dict[str, int] = defaultdict(int)
    leaks: list[dict[str, str]] = []
    for resume in dataset["resumes"]:
        redacted = redact(resume["text"]).text
        for kind, value in resume["pii"]:
            totals[kind] += 1
            if _leaked(kind, value, redacted):
                leaks.append({"type": kind, "value": value})
            else:
                caught[kind] += 1

    changed = [line for line in dataset["benign_lines"] if redact(line, detect_name=False).text != line]
    total_items = sum(totals.values())
    return {
        "resumes": len(dataset["resumes"]),
        "pii_items": total_items,
        "recall": round(sum(caught.values()) / total_items, 3),
        "by_type": [
            {
                "type": kind,
                "items": totals[kind],
                "redacted": caught[kind],
                "recall": round(caught[kind] / totals[kind], 3),
            }
            for kind in sorted(totals)
        ],
        "leaks": leaks,
        "benign_lines": len(dataset["benign_lines"]),
        "over_redacted_lines": changed,
        "over_redaction_rate": round(len(changed) / len(dataset["benign_lines"]), 3),
    }
