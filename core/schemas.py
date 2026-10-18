"""Validated output models for every CareerForge stage.

Language models drift: scores arrive as strings, lists contain objects, keys go
missing. These models coerce what is safely coercible and reject the rest, so the
UI never has to guess what shape it received.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if isinstance(value, dict):
        value = [value]
    items: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = "; ".join(f"{k}: {v}" for k, v in item.items())
        text = _to_text(item)
        if text:
            items.append(text)
    return items


def _bounded_int(low: int, high: int):
    def convert(value: Any) -> int:
        try:
            number = int(round(float(str(value).strip().rstrip("%").split("/")[0])))
        except (TypeError, ValueError):
            return low
        return max(low, min(high, number))

    return convert


Text = Annotated[str, BeforeValidator(_to_text)]
TextList = Annotated[list[str], BeforeValidator(_to_text_list)]
Score100 = Annotated[int, BeforeValidator(_bounded_int(0, 100))]
Score10 = Annotated[int, BeforeValidator(_bounded_int(0, 10))]
Count = Annotated[int, BeforeValidator(_bounded_int(0, 99))]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SkillEvidence(_Model):
    skill: Text = ""
    quote: Text = ""


class ResumeAnalysis(_Model):
    technical_skills: TextList = []
    programming_languages: TextList = []
    frameworks_tools: TextList = []
    projects: TextList = []
    experience: TextList = []
    certifications: TextList = []
    education: TextList = []
    strengths: TextList = []
    improvements: TextList = []
    resume_score: Score100 = 0
    skill_evidence: list[SkillEvidence] = Field(default_factory=list)


class SkillGap(_Model):
    career_match_score: Score100 = 0
    readiness_level: Text = ""
    existing_relevant_skills: TextList = []
    missing_skills: TextList = []
    priority_skills: TextList = []
    assessment: Text = ""
    sources: TextList = []


class RoadmapPhase(_Model):
    phase: Count = 0
    title: Text = ""
    duration: Text = ""
    skills: TextList = []
    activities: TextList = []
    milestone: Text = ""


class Roadmap(_Model):
    roadmap_title: Text = ""
    estimated_duration: Text = ""
    phases: list[RoadmapPhase] = Field(default_factory=list)
    final_outcome: Text = ""
    sources: TextList = []


class Project(_Model):
    project_number: Count = 0
    title: Text = ""
    difficulty: Text = ""
    description: Text = ""
    skills_developed: TextList = []
    technologies: TextList = []
    career_gaps_addressed: TextList = []
    resume_value: Text = ""


class ProjectPlan(_Model):
    projects: list[Project] = Field(default_factory=list)
    sources: TextList = []


class InterviewQuestion(_Model):
    question_number: Count = 0
    question: Text = ""
    category: Text = "Technical"
    difficulty: Text = ""
    expected_points: TextList = []


class Interview(_Model):
    questions: list[InterviewQuestion] = Field(default_factory=list)
    sources: TextList = []


class AnswerEvaluation(_Model):
    score: Score10 = 0
    feedback: Text = ""
    strengths: TextList = []
    missing_points: TextList = []
    improved_answer: Text = ""
