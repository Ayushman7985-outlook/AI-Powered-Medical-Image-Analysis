"""Streamlit interface for educational inference with a saved model."""

import json

import streamlit as st # type: ignore

from src.config import DEFAULT_SETTINGS
from src.predict import load_trained_model, predict_image
from src.preprocessing import ImagePreprocessingError

st.set_page_config(page_title="AI Chest X-ray Learning Demo", page_icon="🩻", layout="wide")


@st.cache_resource(show_spinner="Loading the saved model…")
def cached_model(model_path: str, modified_time: float):
    """Cache only for this exact file version; mtime changes invalidate retrained models."""
    del modified_time
    return load_trained_model(model_path)


def main() -> None:
    st.title("AI-Powered Medical Image Analysis")
    st.subheader("Educational chest X-ray classification: NORMAL / PNEUMONIA")
    st.warning(
        "Educational prototype only — not a medical diagnostic device or medical advice. "
        "Model confidence is not clinical certainty. A qualified healthcare professional must make medical decisions."
    )
    st.write("Upload an image to run the locally trained CNN. Public-dataset performance does not establish clinical reliability.")
    settings = DEFAULT_SETTINGS
    model_path = settings.model_path
    with st.sidebar:
        st.header("Project information")
        st.write("Model: compact CNN trained locally from the documented public dataset.")
        st.write(f"Input size: {settings.image_width} × {settings.image_height} pixels")
        st.write("Positive class: PNEUMONIA")
        st.caption("No training or dataset download is started by this app.")

    uploaded = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        left, right = st.columns([1, 1])
        with left:
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)
        with right:
            if st.button("Analyze image", type="primary"):
                if not model_path.is_file():
                    st.error(f"No trained model found at {model_path}. Train it using the README instructions first.")
                else:
                    try:
                        model = cached_model(str(model_path), model_path.stat().st_mtime)
                        result = predict_image(image_bytes, model=model, settings=settings)
                        st.markdown(f"### Prediction: {result['predicted_class']}")
                        st.metric("Model confidence (uncalibrated)", f"{result['model_confidence']:.1%}")
                        st.caption(result["confidence_note"])
                    except (ImagePreprocessingError, OSError, RuntimeError, ValueError) as exc:
                        st.error(f"Unable to analyze this image: {exc}")

    st.divider()
    st.header("Available model artifacts")
    metric_path = settings.outputs_dir / "metrics.json"
    if metric_path.is_file():
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            st.caption("Saved test-split results from the actual evaluation run, not clinical validation.")
            metric_columns = st.columns(4)
            for column, key, label in zip(metric_columns, ("accuracy", "precision", "recall", "f1_score"),
                                          ("Accuracy", "Precision", "Recall", "F1 score")):
                value = metrics.get(key)
                column.metric(label, f"{value:.3f}" if isinstance(value, (int, float)) else "—")
            st.caption(f"Test examples: {metrics.get('sample_count', 'unknown')} · Positive class: {metrics.get('positive_class', 'unknown')}")
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read saved metrics: {exc}")
    else:
        st.info("No evaluation metrics are available yet. Train a model, then run the separate evaluation command.")

    chart_columns = st.columns(2)
    for column, filename, caption in zip(chart_columns, ("training_accuracy.png", "training_loss.png"),
                                         ("Training and validation accuracy", "Training and validation loss")):
        chart = settings.outputs_dir / filename
        with column:
            if chart.is_file():
                st.image(str(chart), caption=caption, use_container_width=True)
    for filename, caption in (("confusion_matrix.png", "Test-set confusion matrix"),
                              ("prediction_examples.png", "Actual versus predicted test examples")):
        chart = settings.outputs_dir / filename
        if chart.is_file():
            st.image(str(chart), caption=caption, use_container_width=True)

    with st.expander("Limitations and responsible use"):
        st.write("This compact model may learn dataset-specific shortcuts and can be wrong. The source data may be imbalanced or biased, and new hospitals, equipment, populations, or image protocols create distribution shift. Scores are uncalibrated model outputs, not patient-level disease probabilities. Do not use this interface for screening, triage, diagnosis, or treatment decisions.")
    with st.expander("How to train the model"):
        st.code("python main.py train\npython main.py evaluate", language="powershell")
        st.write("Place the public dataset in data/chest_xray first. Training does not run automatically in this app.")


if __name__ == "__main__":
    main()
