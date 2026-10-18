from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_main_app_loads_with_privacy_sidebar():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Delete my data" == b.label for b in at.sidebar.button)


def test_evaluation_dashboard_renders_all_tabs():
    at = AppTest.from_file(str(ROOT / "pages" / "1_Evaluation_Dashboard.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
    labels = {m.label for m in at.metric}
    assert {"Hit@3 (BM25)", "Attack detection", "PII redacted"} <= labels
