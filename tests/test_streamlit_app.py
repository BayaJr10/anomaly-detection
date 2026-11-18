"""Standalone Streamlit app renders and explains flags without errors."""
import pytest

from airq.config import resolve

pytestmark = pytest.mark.skipif(not resolve("models/meta_model.joblib").exists(), reason="train the model first")


def test_streamlit_app_runs_on_demo_sample():
    st_test = pytest.importorskip("streamlit.testing.v1")
    at = st_test.AppTest.from_file(str(resolve("streamlit_app/app.py")), default_timeout=300).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert int(metrics["Flagged"].replace(",", "")) > 0
    assert any("Why flagged" in m.value for m in at.markdown)
    pick = next(s for s in at.selectbox if s.label == "Flagged reading")
    pick.select_index(min(2, len(pick.options) - 1)).run()
    assert not at.exception
