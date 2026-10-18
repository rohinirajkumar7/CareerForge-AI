import pytest

from core import llm
from core.llm import LLMError, TransientLLMError, generate_json
from core.schemas import AnswerEvaluation, ResumeAnalysis, SkillGap


def no_sleep(_seconds):
    return None


def test_fenced_json_is_parsed(fake_llm):
    fake_llm('```json\n{"career_match_score": 72}\n```')
    assert generate_json("p", SkillGap).career_match_score == 72


def test_invalid_json_is_retried_then_succeeds(fake_llm):
    fake = fake_llm("not json", '{"career_match_score": 10}')
    assert generate_json("p", SkillGap, sleep=no_sleep).career_match_score == 10
    assert len(fake.calls) == 2


def test_transient_errors_back_off_and_retry(fake_llm):
    waits = []
    fake_llm(TransientLLMError("429"), TransientLLMError("503"), '{"career_match_score": 5}')
    assert generate_json("p", SkillGap, sleep=waits.append).career_match_score == 5
    assert waits == [1.0, 2.0]


def test_exhausted_retries_raise_a_user_safe_error(fake_llm):
    fake = fake_llm("garbage SECRET-PAYLOAD")
    with pytest.raises(LLMError) as caught:
        generate_json("p", SkillGap, retries=1, sleep=no_sleep)
    assert "SECRET-PAYLOAD" not in str(caught.value) and len(fake.calls) == 2


def test_llm_errors_are_not_retried(fake_llm):
    fake = fake_llm(LLMError("bad key"))
    with pytest.raises(LLMError):
        generate_json("p", SkillGap, sleep=no_sleep)
    assert len(fake.calls) == 1


def test_system_rules_and_temperature_are_sent(fake_llm):
    fake = fake_llm("{}")
    generate_json("task", SkillGap, temperature=0.1)
    assert "untrusted DATA" in fake.calls[0]["system"] and fake.calls[0]["temperature"] == 0.1


def test_default_transport_can_be_restored():
    llm.set_transport(lambda *_: "{}")
    llm.set_transport(None)
    assert llm._transport is llm.gemini_transport


def test_scores_are_coerced_and_clamped():
    assert ResumeAnalysis(resume_score="85/100").resume_score == 85
    assert ResumeAnalysis(resume_score=150).resume_score == 100
    assert ResumeAnalysis(resume_score="abc").resume_score == 0
    assert AnswerEvaluation(score="9.6").score == 10
    assert AnswerEvaluation(score=-3).score == 0


def test_lists_and_text_are_normalized():
    analysis = ResumeAnalysis(technical_skills=None, projects=[{"name": "Bot", "tech": "Python"}, 3], strengths="Fast")
    assert analysis.technical_skills == []
    assert analysis.projects == ["name: Bot; tech: Python", "3"]
    assert analysis.strengths == ["Fast"]


def test_unknown_keys_are_ignored():
    assert "hacked" not in SkillGap.model_validate({"career_match_score": 1, "hacked": True}).model_dump()
