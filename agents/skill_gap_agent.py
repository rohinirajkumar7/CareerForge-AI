"""Skill-Gap Agent: compare the candidate profile with curated role requirements (BM25 RAG)."""

from __future__ import annotations

from typing import Any

from agents.common import merge_security, retrieve_context, untrusted_json, with_retrieval
from core.llm import generate_json
from core.schemas import SkillGap
from rag.knowledge_base import cited_sources

TASK = """You are the Skill-Gap Intelligence Agent inside CareerForge AI.

Compare the candidate's current profile with the skills normally required for the target career.

TARGET CAREER:
{target_role}

{reference}

{profile}

Determine:
1. How well the candidate currently matches the target role.
2. Relevant skills the candidate already has.
3. Important skills that are missing.
4. Skills to learn first.
5. The candidate's readiness level.
6. A short explanation.

Return ONLY valid JSON using exactly this structure:
{{
    "career_match_score": 0,
    "existing_relevant_skills": [],
    "missing_skills": [],
    "priority_skills": [],
    "readiness_level": "",
    "assessment": "",
    "sources": []
}}

career_match_score must be between 0 and 100.
Do not claim the candidate has a skill unless it appears in the profile.
In sources, list the ids of the reference chunks you relied on (empty if none).
"""


def analyze_skill_gap(resume_analysis: dict[str, Any], target_role: str) -> dict[str, Any]:
    reference, hits = retrieve_context(target_role, "required skills tools", ("skills", "overview"))
    profile, report = untrusted_json("candidate profile", resume_analysis)
    result = generate_json(TASK.format(target_role=target_role, reference=reference, profile=profile), SkillGap)
    result.sources = cited_sources(result.sources, hits)
    output = with_retrieval(result.model_dump(), hits)
    output["security"] = merge_security(report)
    return output
