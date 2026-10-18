"""Career Planner Agent: a 4-phase roadmap grounded in curated learning paths."""

from __future__ import annotations

from typing import Any

from agents.common import merge_security, retrieve_context, untrusted_json, with_retrieval
from core.llm import generate_json
from core.schemas import Roadmap
from rag.knowledge_base import cited_sources

TASK = """You are the Career Planner Agent inside CareerForge AI.

Create a personalized career roadmap from the candidate's CURRENT skills and IDENTIFIED skill gaps.

TARGET ROLE:
{target_role}

{reference}

{profile}

{gap}

IMPORTANT:
- Do not recommend learning skills the candidate already has unless advanced knowledge is required.
- Prioritize the missing skills from the skill-gap analysis.
- Make the roadmap practical and achievable, with hands-on activities, not just theory.
- Create exactly 4 phases.

Return ONLY valid JSON using exactly this structure:
{{
    "roadmap_title": "",
    "estimated_duration": "",
    "phases": [
        {{"phase": 1, "title": "", "duration": "", "skills": [], "activities": [], "milestone": ""}}
    ],
    "final_outcome": "",
    "sources": []
}}

In sources, list the ids of the reference chunks you relied on (empty if none).
"""


def generate_career_roadmap(
    resume_analysis: dict[str, Any], skill_gap: dict[str, Any], target_role: str
) -> dict[str, Any]:
    reference, hits = retrieve_context(target_role, "learning path roadmap", ("roadmap", "skills"))
    profile, profile_report = untrusted_json("candidate profile", resume_analysis)
    gap, gap_report = untrusted_json("skill gap analysis", skill_gap)
    prompt = TASK.format(target_role=target_role, reference=reference, profile=profile, gap=gap)
    result = generate_json(prompt, Roadmap)
    result.sources = cited_sources(result.sources, hits)
    output = with_retrieval(result.model_dump(), hits)
    output["security"] = merge_security(profile_report, gap_report)
    return output
