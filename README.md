# CareerForge AI

CareerForge AI is a Streamlit application that uses Google's Gemini models (via the `google-genai` SDK) to act as an end-to-end AI career coach. Upload a resume and it will analyze it, identify skill gaps against a target role, generate a personalized roadmap, suggest portfolio projects, and run a mock interview with answer evaluation.

## Features

The app is organized into five sequential stages:

1. **Resume Intelligence** — Upload a PDF or DOCX resume and get an AI-generated analysis report (strengths, weaknesses, scoring, etc.).
2. **Career Intelligence** — Compare the parsed resume against a target role to surface skill gaps.
3. **Personalized Career Roadmap** — Generate a step-by-step roadmap to close those gaps and reach the target role.
4. **Portfolio Project Strategy** — Get tailored project recommendations to strengthen the portfolio for that role.
5. **AI Interview Coach** — Practice with AI-generated interview questions and receive evaluation/feedback on answers.

## Project Structure

```
CareerForge-AI-main/
├── app.py                     # Streamlit entrypoint / UI
├── agents/
│   ├── resume_agent.py        # Resume analysis
│   ├── skill_gap_agent.py     # Skill gap analysis
│   ├── career_planner_agent.py# Career roadmap generation
│   ├── project_agent.py       # Project recommendations
│   └── interview_agent.py     # Interview Q&A generation + evaluation
├── utils/
│   └── resume_parser.py       # PDF/DOCX text extraction
├── assets/
│   └── careerforge_bg.png     # UI background image
├── Screenshots/                # App screenshots used in this README
└── requirements.txt
```

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[google-genai](https://pypi.org/project/google-genai/)** — Gemini API client
- **pypdf** / **python-docx** — Resume parsing (PDF/DOCX)
- **pandas** — Data handling
- **python-dotenv** — Environment variable management

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CareerForge-AI.git
cd CareerForge-AI
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

You can generate a key from [Google AI Studio](https://aistudio.google.com/).

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Screenshots

See the `Screenshots/` folder for examples of the Resume Analysis, Career Gap Analysis, Career Roadmap, Project Suggestions, and Interview Coach flows.

---

## RAG, security, privacy and evaluation

### BM25 retrieval (RAG)
- `rag/bm25.py` is a from-scratch BM25 (Okapi) ranker with a resume-aware tokenizer (`c++`, `node.js`, `ci/cd`, `full-stack`).
- `data/knowledge_base.json` holds 40 curated chunks (8 roles x overview, skills, interview, projects, roadmap).
- Agents retrieve chunks for the target role, put them in the prompt as trusted reference material and cite chunk ids in `sources`. Citations that do not point at a retrieved chunk are dropped.
- A role gate (`retrieve_for_role`) stops generic words like "engineer" from pulling text about an unrelated role. Roles not in the knowledge base get no reference material and the UI says so.

### Prompt-injection defence (`security/injection_guard.py`)
1. **Sanitize**: strip invisible characters, defuse delimiter look-alikes, cap length.
2. **Scan**: 19 weighted heuristic rules (override, score demands, role tags, JSON-key injection, exfiltration, multilingual, ...).
3. **Neutralize**: sentences matching a strong rule are removed before the model sees them.
4. **Isolate**: untrusted text is wrapped in randomized markers; the system instruction says it is data only.
5. **Validate**: outputs are schema-checked (`core/schemas.py`); extracted skills must appear in the resume or be backed by a verbatim quote, otherwise they are excluded. Model output is re-guarded before it enters later prompts.

### Privacy (`privacy/redactor.py`)
Emails, phone numbers, LinkedIn/GitHub/other links, PAN/Aadhaar/SSN-style IDs, dates of birth, labelled addresses and the candidate's name are replaced with placeholders **before** anything is sent to the AI. The UI shows exactly what the AI will see, keeps only the redacted text in the session, and has a "Delete my data" button.

### Evaluation dashboard
```bash
python -m evaluation.run          # offline metrics, no API key
python -m evaluation.run --live   # also measure the real model (needs GEMINI_API_KEY)
streamlit run app.py              # then open "Evaluation Dashboard" in the sidebar
```
Measured on the bundled datasets (offline):

| Area | Result |
|---|---|
| Retrieval, BM25 | Hit@3 100%, MRR 0.944 (30 queries, 40 chunks) |
| Retrieval, keyword-overlap baseline | Hit@3 100%, MRR 0.983 |
| Role resolution | 16/16 known roles and aliases; 8/8 unsupported roles abstain |
| Injection detection | 31/36 attacks (86.1%); 0/32 benign lines flagged |
| PII redaction | 28/28 items; 0/10 non-personal lines altered |

### Honest limitations
- The datasets were written by the project author next to the rules, so these numbers are a regression guard, not an independent benchmark. Add real examples to make them harder.
- On this small corpus a plain keyword-overlap baseline ranks as well as BM25 (slightly better on MRR); BM25's advantage should grow with the corpus.
- The injection scanner misses paraphrased, obfuscated (leetspeak), unsupported-language and line-split attacks (the 5 "hard" cases in the dashboard). Isolation and output grounding limit the damage when it misses.
- Redaction is pattern based: names rely on the first lines of the resume or the name you type in; unusual address formats are not detected.
- The live evaluation was written and unit-tested with a fake model but has not been run against the real Gemini API.

### Tests
```bash
pip install -r requirements-dev.txt
pytest          # 176 tests, no network or API key needed
ruff check .
```
