import base64
import logging
from pathlib import Path

import streamlit as st

from utils.resume_parser import extract_resume_text
from agents.resume_agent import analyze_resume
from agents.skill_gap_agent import analyze_skill_gap
from agents.career_planner_agent import generate_career_roadmap
from agents.project_agent import recommend_projects
from agents.interview_agent import generate_interview, evaluate_answer
from core.llm import LLMError
from privacy.redactor import redact
from security.injection_guard import guard_untrusted

logger = logging.getLogger("careerforge.app")


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="CareerForge AI",
    page_icon="🚀",
    layout="wide"
)


# =========================================================
# BACKGROUND + UI
# =========================================================

def set_background():
    image_path = Path("assets/careerforge_bg.png")

    if image_path.exists():
        encoded = base64.b64encode(
            image_path.read_bytes()
        ).decode()

        background = f"""
        background-image:
        linear-gradient(
            rgba(4, 9, 20, 0.84),
            rgba(4, 9, 20, 0.90)
        ),
        url("data:image/png;base64,{encoded}");
        """
    else:
        background = "background-color:#050914;"

    st.markdown(
        f"""
        <style>

        .stApp {{
            {background}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }}

        .hero {{
            padding: 35px;
            border-radius: 22px;
            background: rgba(5, 12, 28, 0.78);
            border: 1px solid rgba(90, 170, 255, 0.25);
            backdrop-filter: blur(12px);
            text-align: center;
            margin-bottom: 25px;
        }}

        .hero h1 {{
            font-size: 46px;
            margin-bottom: 5px;
        }}

        .hero p {{
            font-size: 18px;
            opacity: .85;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(5, 12, 28, 0.78);
            border: 1px solid rgba(100, 180, 255, .22);
            padding: 18px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}

        div[data-testid="stExpander"] {{
            background: rgba(5, 12, 28, 0.65);
            border-radius: 12px;
        }}

        div.stButton > button {{
            border-radius: 10px;
            font-weight: 600;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# =========================================================
# HELPERS
# =========================================================

def show_error(prefix, error):
    """Show a safe message; keep technical details in the log, not on screen."""
    if isinstance(error, LLMError):
        st.error(f"{prefix}: {error}")
        return
    logger.exception("%s", prefix, exc_info=error)
    st.error(f"{prefix}. Please try again; details were written to the log.")


def privacy_controls():
    """Privacy options shown before any resume text is analysed."""
    st.markdown("##### 🔒 Privacy")
    enabled = st.checkbox(
        "Redact personal information before sending to the AI (recommended)",
        value=True,
        key="redact_enabled",
        help="Emails, phone numbers, profile links, ID numbers, date of birth, "
             "address lines and your name are replaced with placeholders.",
    )
    names = st.text_input(
        "Name to redact (optional - auto-detected from the first lines)",
        key="redact_name",
        disabled=not enabled,
    )
    return enabled, [n.strip() for n in names.split(",") if n.strip()]


def prepare_resume_text(raw_text, enabled, names):
    """Return (text the AI will see, redaction counts). The original is not kept."""
    if not enabled:
        return raw_text, {}
    result = redact(raw_text, names=names)
    return result.text, result.counts


def show_redaction_summary(enabled, counts):
    if not enabled:
        st.warning("Redaction is off: your full resume text will be sent to the AI service.")
    elif counts:
        found = ", ".join(f"{n} {kind.lower().replace('_', ' ')}" for kind, n in sorted(counts.items()))
        st.info(f"Redacted before analysis: {found}.")
    else:
        st.info("No personal details were detected. Check the preview below before analysing.")


def show_input_scan(text):
    """Warn the user if the resume contains text that looks like instructions to the AI."""
    report = guard_untrusted(text)
    if report.flagged:
        st.warning(
            f"This document contains text that looks like instructions to an AI "
            f"(risk: {report.risk_level}). It will be removed before analysis and will not affect your results."
        )
        with st.expander("What was flagged"):
            for finding in report.findings:
                st.write(f"• **{finding.rule}** - {finding.snippet}")


def show_analysis_notes(analysis):
    """Surface injection findings and how many extracted skills are backed by the resume text."""
    security = analysis.get("security")
    if security and security.get("risk_level") in ("medium", "high"):
        st.warning(
            f"Possible prompt-injection text was found in the resume "
            f"(risk: {security['risk_level']}) and was removed before analysis."
        )
    grounding = analysis.get("grounding") or {}
    if grounding.get("total"):
        st.caption(
            f"Skill grounding: {grounding['verified']} of {grounding['total']} extracted skills "
            f"were verified in your resume text."
        )
    unverified = analysis.get("unverified_skills") or []
    if unverified:
        with st.expander(f"{len(unverified)} skill(s) not found in the resume (excluded)"):
            for skill in unverified:
                st.write(f"• {skill}")


def show_sources(result):
    """List the curated reference material that was retrieved for this result."""
    retrieved = result.get("retrieved") or []
    cited = set(result.get("sources") or [])
    if not retrieved:
        st.caption("No curated reference material matched this role; advice uses general knowledge.")
        return
    with st.expander("Reference material used (BM25 retrieval)"):
        for item in retrieved:
            mark = "cited" if item["id"] in cited else "retrieved"
            st.write(f"• `{item['id']}` - {item['title']} (score {item['score']}, {mark})")


def clear_user_data():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


with st.sidebar:
    st.markdown("### 🔒 Your data")
    st.caption(
        "Your resume is processed in memory for this session only and is never saved by this app. "
        "Only the redacted text is sent to the AI service."
    )
    if st.button("Delete my data"):
        clear_user_data()
        st.rerun()


def safe_score(value):
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def show_list(title, items):
    st.subheader(title)

    if items:
        for item in items:
            st.write(f"• {item}")
    else:
        st.caption("Nothing identified.")


def clear_after(stage):
    order = [
        "skill_gap",
        "career_roadmap",
        "project_recommendations",
        "interview"
    ]

    if stage in order:
        start = order.index(stage)

        for key in order[start:]:
            st.session_state.pop(key, None)

    # Clear old interview evaluations
    for key in list(st.session_state.keys()):
        if key.startswith("evaluation_"):
            del st.session_state[key]


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>CareerForge AI</h1>
        <p>
        Multi-Agent Career Intelligence Platform
        </p>
        <p>
        Understand your profile • Discover skill gaps •
        Build your roadmap • Strengthen your portfolio •
        Prepare for interviews
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# JOURNEY STATUS
# =========================================================

analysis = st.session_state.get("resume_analysis", {})
gap = st.session_state.get("skill_gap", {})

resume_score = safe_score(
    analysis.get("resume_score", 0)
)

career_score = safe_score(
    gap.get("career_match_score", 0)
)

readiness = gap.get(
    "readiness_level",
    "Not assessed"
)

target = st.session_state.get(
    "target_role",
    "Not selected"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Resume Score", f"{resume_score}/100")
c2.metric("Career Match", f"{career_score}%")
c3.metric("Readiness", readiness)
c4.metric("Target Role", target)

st.caption(
    "Resume Intelligence → Skill Gap → Career Planner → "
    "Portfolio Strategist → Interview Coach"
)

st.divider()


# =========================================================
# 01 RESUME INTELLIGENCE
# =========================================================

st.header("01 · Resume Intelligence")

st.write(
    "Upload your resume to create the candidate profile "
    "used throughout CareerForge."
)

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

if uploaded_resume:

    redact_enabled, extra_names = privacy_controls()

    try:
        raw_text = extract_resume_text(uploaded_resume)

        if raw_text:
            resume_text, redaction_counts = prepare_resume_text(
                raw_text, redact_enabled, extra_names
            )

            # Only the text the AI will see is kept in the session.
            st.session_state["resume_text"] = resume_text

            st.success(
                f"Resume processed successfully — "
                f"{len(resume_text.split())} words extracted."
            )
            show_redaction_summary(redact_enabled, redaction_counts)

            with st.expander("What the AI will see"):
                st.text_area(
                    "Resume Content",
                    resume_text,
                    height=300,
                    disabled=True
                )

            show_input_scan(resume_text)

            if st.button(
                "Analyze My Resume",
                type="primary"
            ):
                with st.spinner(
                    "Resume Intelligence Agent is analyzing..."
                ):
                    try:
                        result = analyze_resume(resume_text)

                        st.session_state[
                            "resume_analysis"
                        ] = result

                        clear_after("skill_gap")

                        st.success(
                            "Resume analysis completed."
                        )
                        st.rerun()

                    except Exception as error:
                        show_error("Resume analysis failed", error)

        else:
            st.warning(
                "No readable text was found in the resume."
            )

    except Exception as error:
        show_error("Resume processing failed", error)


# ---------------- RESUME REPORT ----------------

if "resume_analysis" in st.session_state:

    analysis = st.session_state["resume_analysis"]

    st.subheader("Resume Intelligence Report")

    score = safe_score(
        analysis.get("resume_score", 0)
    )

    st.metric(
        "CareerForge Resume Score",
        f"{score}/100"
    )

    st.progress(score / 100)

    show_analysis_notes(analysis)

    col1, col2 = st.columns(2)

    with col1:
        show_list(
            "Technical Skills",
            analysis.get("technical_skills", [])
        )

        show_list(
            "Strengths",
            analysis.get("strengths", [])
        )

    with col2:
        show_list(
            "Tools & Frameworks",
            analysis.get("frameworks_tools", [])
        )

        show_list(
            "Areas for Improvement",
            analysis.get("improvements", [])
        )

    with st.expander("Projects Identified"):
        for item in analysis.get("projects", []):
            st.write(f"• {item}")

    with st.expander("Experience"):
        for item in analysis.get("experience", []):
            st.write(f"• {item}")

    with st.expander("Certifications"):
        for item in analysis.get("certifications", []):
            st.write(f"• {item}")


st.divider()


# =========================================================
# 02 CAREER ANALYSIS
# =========================================================

st.header("02 · Career Intelligence")

if "resume_analysis" not in st.session_state:

    st.info(
        "Complete Resume Intelligence to unlock Career Analysis."
    )

else:

    roles = [
        "AI / Machine Learning Engineer",
        "Data Scientist",
        "Data Analyst",
        "Software Engineer",
        "Full Stack Developer",
        "Cybersecurity Engineer",
        "Cloud Engineer",
        "DevOps Engineer"
    ]

    saved_role = st.session_state.get(
        "target_role",
        roles[0]
    )

    if saved_role not in roles:
        saved_role = roles[0]

    target_role = st.selectbox(
        "Select your target career",
        roles,
        index=roles.index(saved_role)
    )

    if st.button(
        "Analyze My Career Gap",
        type="primary"
    ):

        with st.spinner(
            f"Analyzing your readiness for {target_role}..."
        ):
            try:
                result = analyze_skill_gap(
                    st.session_state["resume_analysis"],
                    target_role
                )

                st.session_state["target_role"] = target_role
                st.session_state["skill_gap"] = result

                clear_after("career_roadmap")

                st.success(
                    "Career-gap analysis completed."
                )
                st.rerun()

            except Exception as error:
                show_error("Career analysis failed", error)


# ---------------- CAREER REPORT ----------------

if "skill_gap" in st.session_state:

    gap = st.session_state["skill_gap"]

    score = safe_score(
        gap.get("career_match_score", 0)
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Career Match Score",
        f"{score}/100"
    )

    c2.metric(
        "Career Readiness",
        gap.get("readiness_level", "N/A")
    )

    st.progress(score / 100)

    show_sources(gap)

    st.subheader("Career Assessment")

    st.write(
        gap.get("assessment", "")
    )

    col1, col2 = st.columns(2)

    with col1:
        show_list(
            "Skills You Already Have",
            gap.get(
                "existing_relevant_skills",
                []
            )
        )

    with col2:
        show_list(
            "Missing Skills",
            gap.get("missing_skills", [])
        )

    show_list(
        "Priority Skills to Learn",
        gap.get("priority_skills", [])
    )


st.divider()


# =========================================================
# 03 CAREER ROADMAP
# =========================================================

st.header("03 · Personalized Career Roadmap")

if "skill_gap" not in st.session_state:

    st.info(
        "Complete Career Analysis to unlock your roadmap."
    )

else:

    if "career_roadmap" not in st.session_state:

        if st.button(
            "Build My Personalized Roadmap",
            type="primary"
        ):

            with st.spinner(
                "Career Planner Agent is building your roadmap..."
            ):
                try:
                    roadmap = generate_career_roadmap(
                        st.session_state["resume_analysis"],
                        st.session_state["skill_gap"],
                        st.session_state["target_role"]
                    )

                    st.session_state[
                        "career_roadmap"
                    ] = roadmap

                    clear_after(
                        "project_recommendations"
                    )

                    st.success(
                        "Personalized roadmap created."
                    )
                    st.rerun()

                except Exception as error:
                    show_error("Roadmap generation failed", error)


if "career_roadmap" in st.session_state:

    roadmap = st.session_state[
        "career_roadmap"
    ]

    show_sources(roadmap)

    st.subheader(
        roadmap.get(
            "roadmap_title",
            "Your Career Roadmap"
        )
    )

    st.info(
        "Estimated Duration: "
        + roadmap.get(
            "estimated_duration",
            "N/A"
        )
    )

    for phase in roadmap.get("phases", []):

        title = (
            f"Phase {phase.get('phase', '')} — "
            f"{phase.get('title', '')}"
        )

        with st.expander(
            title,
            expanded=True
        ):

            st.write(
                f"**Duration:** "
                f"{phase.get('duration', 'N/A')}"
            )

            show_list(
                "Skills to Develop",
                phase.get("skills", [])
            )

            show_list(
                "Practical Activities",
                phase.get("activities", [])
            )

            st.markdown("#### Milestone")

            st.success(
                phase.get("milestone", "")
            )

    st.subheader("Expected Outcome")

    st.write(
        roadmap.get("final_outcome", "")
    )


st.divider()


# =========================================================
# 04 PORTFOLIO PROJECTS
# =========================================================

st.header("04 · Portfolio Project Strategy")

if "career_roadmap" not in st.session_state:

    st.info(
        "Generate your Career Roadmap to unlock "
        "personalized project recommendations."
    )

else:

    if "project_recommendations" not in st.session_state:

        if st.button(
            "Recommend Portfolio Projects",
            type="primary"
        ):

            with st.spinner(
                "Portfolio Agent is designing your projects..."
            ):
                try:
                    projects = recommend_projects(
                        st.session_state["resume_analysis"],
                        st.session_state["skill_gap"],
                        st.session_state["career_roadmap"],
                        st.session_state["target_role"]
                    )

                    st.session_state[
                        "project_recommendations"
                    ] = projects

                    st.success(
                        "Portfolio strategy generated."
                    )
                    st.rerun()

                except Exception as error:
                    show_error("Project generation failed", error)


if "project_recommendations" in st.session_state:

    projects = st.session_state[
        "project_recommendations"
    ].get("projects", [])

    show_sources(st.session_state["project_recommendations"])

    for project in projects:

        st.subheader(
            f"{project.get('project_number', '')}. "
            f"{project.get('title', 'Project')}"
        )

        st.caption(
            "Difficulty: "
            + project.get("difficulty", "N/A")
        )

        st.write(
            project.get("description", "")
        )

        col1, col2 = st.columns(2)

        with col1:
            show_list(
                "Skills Developed",
                project.get(
                    "skills_developed",
                    []
                )
            )

            show_list(
                "Technologies",
                project.get(
                    "technologies",
                    []
                )
            )

        with col2:
            show_list(
                "Career Gaps Addressed",
                project.get(
                    "career_gaps_addressed",
                    []
                )
            )

            st.markdown(
                "### Resume Value"
            )

            st.write(
                project.get(
                    "resume_value",
                    ""
                )
            )

        st.divider()


# =========================================================
# 05 AI INTERVIEW
# =========================================================

st.header("05 · AI Interview Coach")

if "skill_gap" not in st.session_state:

    st.info(
        "Complete Career Analysis to unlock "
        "your personalized mock interview."
    )

else:

    if st.button(
        "Generate Personalized Mock Interview",
        type="primary"
    ):

        with st.spinner(
            "Interview Agent is preparing your questions..."
        ):
            try:
                interview = generate_interview(
                    st.session_state["resume_analysis"],
                    st.session_state["skill_gap"],
                    st.session_state["target_role"]
                )

                st.session_state[
                    "interview"
                ] = interview

                for key in list(
                    st.session_state.keys()
                ):
                    if key.startswith(
                        "evaluation_"
                    ):
                        del st.session_state[key]

                st.success(
                    "Mock interview generated."
                )
                st.rerun()

            except Exception as error:
                show_error("Interview generation failed", error)


# ---------------- INTERVIEW QUESTIONS ----------------

if "interview" in st.session_state:

    interview_data = st.session_state["interview"]

    if isinstance(interview_data, dict):
        show_sources(interview_data)

    # Gemini may return either a list directly
    # or a dictionary containing a "questions" list.
    if isinstance(interview_data, list):
        questions = interview_data
    elif isinstance(interview_data, dict):
        questions = interview_data.get("questions", [])
    else:
        questions = []

    for index, data in enumerate(questions):

        number = data.get(
            "question_number",
            index + 1
        )

        question = data.get(
            "question",
            ""
        )

        st.subheader(
            f"Question {number}"
        )

        st.caption(
            f"{data.get('category', 'Technical')} • "
            f"{data.get('difficulty', 'N/A')}"
        )

        st.write(
            f"**{question}**"
        )

        answer = st.text_area(
            "Your Answer",
            key=f"answer_{index}",
            height=130
        )

        if st.button(
            f"Evaluate Answer {number}",
            key=f"evaluate_{index}"
        ):

            if not answer.strip():

                st.warning(
                    "Enter your answer first."
                )

            else:

                with st.spinner(
                    "Interview Agent is evaluating your answer..."
                ):

                    try:
                        evaluation = evaluate_answer(
                            question,
                            data.get(
                                "expected_points",
                                []
                            ),
                            answer,
                            st.session_state[
                                "target_role"
                            ]
                        )

                        st.session_state[
                            f"evaluation_{index}"
                        ] = evaluation

                    except Exception as error:
                        show_error("Evaluation failed", error)

        key = f"evaluation_{index}"

        if key in st.session_state:

            evaluation = st.session_state[key]

            st.metric(
                "Interview Score",
                f"{evaluation.get('score', 0)}/10"
            )

            st.write(
                "**AI Feedback**"
            )

            st.write(
                evaluation.get(
                    "feedback",
                    ""
                )
            )

            col1, col2 = st.columns(2)

            with col1:
                show_list(
                    "What You Did Well",
                    evaluation.get(
                        "strengths",
                        []
                    )
                )

            with col2:
                show_list(
                    "Missing Points",
                    evaluation.get(
                        "missing_points",
                        []
                    )
                )

            with st.expander(
                "View Improved Answer"
            ):
                st.write(
                    evaluation.get(
                        "improved_answer",
                        ""
                    )
                )

        st.divider()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "CareerForge AI • Multi-Agent Career Intelligence Platform "
    "• Supporting UN SDG 8 — Decent Work & Economic Growth"
)