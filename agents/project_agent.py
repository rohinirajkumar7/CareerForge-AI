"""Portfolio Project Agent: three projects that close the candidate's gaps."""

from __future__ import annotations

from typing import Any

from agents.common import merge_security, retrieve_context, untrusted_json, with_retrieval
from core.llm import generate_json
from core.schemas import ProjectPlan
from rag.knowledge_base import cited_sources

TASK = """You are the Portfolio Project Agent inside CareerForge AI.

Recommend portfolio projects that make this candidate more competitive for the TARGET CAREER.

TARGET CAREER:
{target_role}

{reference}

{profile}

{gap}

{roadmap}

Generate exactly 3 personalized portfolio projects.

RULES:
- Projects must address skills the candidate is currently missing.
- Do not repeat projects already in the candidate's resume.
- Keep them realistic for a college student and progressively more advanced.
- At least one should be strong enough to become a major resume/GitHub project.
- Recommend technologies relevant to the target career and say which gaps each project addresses.

Return ONLY valid JSON using exactly this structure:
{{
    "projects": [
        {{
            "project_number": 1, "title": "", "difficulty": "", "description": "",
            "skills_developed": [], "technologies": [], "career_gaps_addressed": [], "resume_value": ""
        }}
    ],
    "sources": []
}}

In sources, list the ids of the reference chunks you relied on (empty if none).
"""


def recommend_projects(
    resume_analysis: dict[str, Any],
    skill_gap: dict[str, Any],
    career_roadmap: dict[str, Any],
    target_role: str,
) -> dict[str, Any]:
    reference, hits = retrieve_context(target_role, "portfolio project ideas", ("projects",))
    profile, r1 = untrusted_json("candidate profile", resume_analysis)
    gap, r2 = untrusted_json("skill gap analysis", skill_gap)
    roadmap, r3 = untrusted_json("career roadmap", career_roadmap)
    prompt = TASK.format(target_role=target_role, reference=reference, profile=profile, gap=gap, roadmap=roadmap)
    result = generate_json(prompt, ProjectPlan)
    result.sources = cited_sources(result.sources, hits)
    output = with_retrieval(result.model_dump(), hits)
    output["security"] = merge_security(r1, r2, r3)
    return output
