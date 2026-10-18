"""Resume Intelligence Agent: extract a profile and score it, defended against injection."""

from __future__ import annotations

import re
from typing import Any

from core.llm import generate_json
from core.schemas import ResumeAnalysis
from security.injection_guard import MAX_RESUME_CHARS, guard_untrusted, wrap_untrusted

SKILL_FIELDS = ("technical_skills", "programming_languages", "frameworks_tools")

TASK = """You are the Resume Intelligence Agent inside CareerForge AI.

Analyze the candidate's resume objectively and extract:
1. Technical skills  2. Programming languages  3. Frameworks and tools  4. Projects
5. Work/internship experience  6. Certifications  7. Education
8. Major strengths  9. Areas that could be improved
Then give a resume_score from 0 to 100 using only evidence in the resume.

For every skill you list, add an entry to skill_evidence with the skill and a short quote copied
word for word from the resume that proves it. Do not list skills you cannot quote.
Do not invent information that does not appear in the resume.

Return ONLY valid JSON using exactly this structure:
{
    "technical_skills": [], "programming_languages": [], "frameworks_tools": [],
    "projects": [], "experience": [], "certifications": [], "education": [],
    "strengths": [], "improvements": [], "resume_score": 0,
    "skill_evidence": [{"skill": "", "quote": ""}]
}
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _quote_present(quote: str, haystack: str) -> bool:
    """True when the quote (ignoring '...' gaps) appears in the resume text."""
    parts = [p.strip() for p in quote.split("...") if len(p.strip()) >= 6]
    return bool(parts) and all(_normalize(p) in haystack for p in parts)


def ground_skills(analysis: ResumeAnalysis, resume_text: str) -> dict[str, Any]:
    """Keep only skills the resume supports; report the rest as unverified.

    A skill counts as supported when its own name appears in the resume or when the
    model supplied a verbatim quote that does. Fabricated skills therefore never
    reach later stages, whether they come from a hallucination or from an injection.
    """
    haystack = _normalize(resume_text)
    quoted = {_normalize(e.skill) for e in analysis.skill_evidence if _quote_present(e.quote, haystack)}
    verified = unverified = 0
    result = analysis.model_dump()
    dropped: list[str] = []
    for field_name in SKILL_FIELDS:
        kept: list[str] = []
        for skill in result[field_name]:
            if _normalize(skill) in haystack or _normalize(skill) in quoted:
                kept.append(skill)
                verified += 1
            else:
                dropped.append(skill)
                unverified += 1
        result[field_name] = kept
    total = verified + unverified
    result["unverified_skills"] = dropped
    result["grounding"] = {"verified": verified, "total": total, "rate": round(verified / total, 3) if total else None}
    return result


def analyze_resume(resume_text: str) -> dict[str, Any]:
    guard = guard_untrusted(resume_text, max_chars=MAX_RESUME_CHARS)
    prompt = f"{TASK}\n{wrap_untrusted('resume', guard.text)}"
    analysis = generate_json(prompt, ResumeAnalysis)
    result = ground_skills(analysis, guard.text)
    result["security"] = guard.summary() if guard.findings or guard.truncated else None
    return result
