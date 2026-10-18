"""Evaluation dashboard: retrieval, prompt-injection defence, privacy and live-model metrics."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from evaluation.run import load_results, run_offline, save_results

st.set_page_config(page_title="CareerForge Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")
st.caption(
    "Measured quality of the retrieval, security and privacy layers. Offline metrics need no API key. "
    "Datasets are small and written by the project author, so treat them as a regression guard, "
    "not an independent benchmark."
)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


controls_left, controls_right = st.columns([1, 3])
with controls_left:
    if st.button("Re-run offline evaluation", type="primary"):
        with st.spinner("Running offline evaluation..."):
            save_results(offline=run_offline())
        st.rerun()

results = load_results()
if not results or not results.get("offline"):
    st.info("No results yet. Click **Re-run offline evaluation** or run `python -m evaluation.run`.")
    st.stop()

with controls_right:
    st.caption(f"Last updated (UTC): {results['generated_at']}")

offline = results["offline"]
tab_retrieval, tab_security, tab_privacy, tab_live = st.tabs(
    ["Retrieval (BM25 RAG)", "Prompt-injection defence", "Privacy redaction", "Live model"]
)

# ------------------------------------------------------------------ retrieval
with tab_retrieval:
    retrieval = offline["retrieval"]
    roles = offline["role_resolution"]
    bm25, base = retrieval["bm25"], retrieval["baseline_overlap"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hit@3 (BM25)", pct(bm25["hit@3"]))
    c2.metric("MRR (BM25)", f"{bm25['mrr']:.3f}")
    c3.metric("Role resolution", pct(roles["known_accuracy"]))
    c4.metric("Abstains on unknown roles", pct(roles["unknown_abstain_rate"]))
    st.caption(f"{retrieval['queries']} labelled queries over {retrieval['chunks']} knowledge-base chunks.")

    st.subheader("BM25 vs simple keyword-overlap baseline")
    comparison = pd.DataFrame({"BM25": bm25, "Keyword overlap": base}).drop(index=["recall@1", "recall@3", "recall@5"])
    st.bar_chart(comparison)
    if base["mrr"] >= bm25["mrr"]:
        st.warning(
            "On this small corpus the plain overlap baseline ranks as well as or better than BM25 "
            f"(MRR {base['mrr']:.3f} vs {bm25['mrr']:.3f}); the gap is a few queries out of {retrieval['queries']}. "
            "BM25's idf and length normalization matter more as the knowledge base grows."
        )

    st.subheader("Abstention threshold")
    sweep = frame(retrieval["threshold_sweep"]).set_index("min_score")
    st.line_chart(sweep)
    st.caption(
        f"Configured min_score = {retrieval['configured_min_score']}. Off-topic queries peak at "
        f"{max(retrieval['off_topic_top_scores'])}; on-topic queries all score higher, so retrieval abstains "
        "instead of grounding on unrelated text."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Queries where BM25 did not rank first")
        misses = [q for q in retrieval["per_query"] if q["bm25_rank"] != 1]
        st.dataframe(frame(misses)[["query", "relevant", "bm25_rank", "baseline_rank"]], hide_index=True)
    with right:
        st.subheader("Role resolution")
        st.dataframe(frame(roles["known"]), hide_index=True)
        st.dataframe(frame(roles["unknown"]), hide_index=True)

# ------------------------------------------------------------------ security
with tab_security:
    injection = offline["injection"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Attack detection", pct(injection["detection_rate"]), help=f"{injection['attacks']} attack samples")
    c2.metric(
        "False positives", pct(injection["false_positive_rate"]), help=f"{injection['benign']} benign resume lines"
    )
    c3.metric("Attacks neutralized", pct(injection["neutralization_rate"]))

    st.subheader("Detection by attack category")
    by_category = frame(injection["by_category"]).set_index("category")
    st.bar_chart(by_category["rate"])
    st.dataframe(by_category)

    st.subheader("Missed attacks")
    if injection["missed"]:
        st.dataframe(frame(injection["missed"]), hide_index=True)
        st.caption(
            "Paraphrases, leetspeak, unsupported languages and instructions split across lines evade the "
            "heuristic scanner. Randomized delimiters, the system instruction and output grounding are the "
            "second line of defence for these."
        )
    else:
        st.success("No attack in the dataset was missed.")

    st.subheader("False positives")
    if injection["false_positives"]:
        st.dataframe(frame(injection["false_positives"]), hide_index=True)
    else:
        st.success("No benign line was flagged, including lines that mention prompts, instructions or scores.")

# ------------------------------------------------------------------ privacy
with tab_privacy:
    privacy = offline["privacy"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "PII redacted",
        pct(privacy["recall"]),
        help=f"{privacy['pii_items']} labelled items in {privacy['resumes']} resumes",
    )
    c2.metric("Leaked items", len(privacy["leaks"]))
    c3.metric(
        "Over-redaction",
        pct(privacy["over_redaction_rate"]),
        help=f"{privacy['benign_lines']} benign lines with dates, versions, metrics, IPs",
    )

    by_type = frame(privacy["by_type"]).set_index("type")
    st.bar_chart(by_type["recall"])
    st.dataframe(by_type)
    if privacy["leaks"]:
        st.error("Some personal data survived redaction:")
        st.dataframe(frame(privacy["leaks"]), hide_index=True)
    if privacy["over_redacted_lines"]:
        st.warning("These non-personal lines were altered:")
        st.write(privacy["over_redacted_lines"])
    st.caption(
        "Pattern based: names are found from the first lines of the resume or the name you type in. "
        "Always check the redacted preview before analysis."
    )

# ------------------------------------------------------------------ live
with tab_live:
    live = results.get("live")
    api_ready = bool(os.getenv("GEMINI_API_KEY")) and os.getenv("ENABLE_LIVE_EVAL") == "1"
    st.write(
        "Runs the real model on three synthetic resumes (strong ML, weak ML, strong DevOps) several times, plus "
        "the weak resume with an injected 'give me 100' instruction."
    )
    runs = st.slider("Runs per resume", 1, 5, 3, disabled=not api_ready)
    if st.button("Run live evaluation", disabled=not api_ready):
        from core.llm import model_name
        from evaluation.live import run_live

        with st.spinner("Calling the model..."):
            outcome = run_live(runs)
            outcome["model"] = model_name()
            save_results(live=outcome)
        st.rerun()
    if not api_ready:
        st.caption("Set GEMINI_API_KEY in .env to enable this. It makes about 4 x runs model calls.")

    if live:
        c1, c2, c3 = st.columns(3)
        c1.metric("Valid outputs", pct(live["validity_rate"]))
        gap = live["discrimination_gap"]
        c2.metric("Strong vs weak resume (points)", "n/a" if gap is None else gap)
        hijacked = live["injection"]["hijacked"]
        c3.metric("Score hijacked by injection", "n/a" if hijacked is None else ("YES" if hijacked else "No"))
        st.caption(f"Model: {live.get('model', 'unknown')} | runs per resume: {live['runs_per_resume']}")
        rows = [
            {
                "resume": name,
                "mean score": item["mean_score"],
                "std dev": item["score_std"],
                "valid runs": f"{item['valid_runs']}/{item['runs']}",
                "grounded skills": pct(item["grounding_rate"]),
                "injection flagged": item["injection_flagged"],
            }
            for name, item in live["resumes"].items()
        ]
        st.dataframe(frame(rows), hide_index=True)
        st.json(live["injection"])
    else:
        st.info("No live results stored yet.")
