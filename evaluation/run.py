"""Run the evaluation suite and store results for the dashboard.

python -m evaluation.run            # offline metrics, no API key needed
python -m evaluation.run --live     # also call the real model (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    evaluate_injection,
    evaluate_privacy,
    evaluate_retrieval,
    evaluate_role_resolution,
    load_dataset,
)
from rag.knowledge_base import get_knowledge_base

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "latest.json"


def run_offline() -> dict[str, Any]:
    kb = get_knowledge_base()
    return {
        "retrieval": evaluate_retrieval(kb, load_dataset("retrieval.json")),
        "role_resolution": evaluate_role_resolution(kb, load_dataset("roles.json")),
        "injection": evaluate_injection(load_dataset("injection.json")),
        "privacy": evaluate_privacy(load_dataset("pii.json")),
    }


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(
    offline: dict[str, Any] | None = None, live: dict[str, Any] | None = None, path: Path = RESULTS_PATH
) -> dict[str, Any]:
    """Write results, keeping whichever section is not being replaced."""
    previous = load_results(path) or {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline": offline if offline is not None else previous.get("offline"),
        "live": live if live is not None else previous.get("live"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also run the live model evaluation")
    parser.add_argument("--runs", type=int, default=3, help="runs per resume for --live")
    args = parser.parse_args()

    offline = run_offline()
    live = None
    if args.live:
        from core.llm import model_name
        from evaluation.live import run_live

        live = run_live(args.runs)
        live["model"] = model_name()
    save_results(offline, live)
    print(f"saved {RESULTS_PATH}")
    retrieval, injection, privacy = offline["retrieval"], offline["injection"], offline["privacy"]
    print(f"retrieval hit@3={retrieval['bm25']['hit@3']} mrr={retrieval['bm25']['mrr']}")
    print(f"injection detection={injection['detection_rate']} false_positives={injection['false_positive_rate']}")
    print(f"privacy recall={privacy['recall']} over_redaction={privacy['over_redaction_rate']}")


if __name__ == "__main__":
    main()
