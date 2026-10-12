from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_a_trained_model():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert any("AI-Powered Medical Image Analysis" in title.value for title in app.title)
    assert any("Educational prototype only" in warning.value for warning in app.warning)
    assert any("No evaluation metrics are available yet" in element.value for element in app.info)