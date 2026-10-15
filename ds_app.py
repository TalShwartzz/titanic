"""Streamlit Titanic inference + held-out validation dashboard.

Launch: streamlit run ds_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import REQUIRED_FEATURES
from src.evaluation import evaluate_binary, load_bundle, predict_probabilities

st.set_page_config(page_title="Titanic | Survival Intelligence", page_icon="🚢", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px}
h1 {font-weight: 800 !important; letter-spacing: -0.035em}
[data-testid="stMetric"] {background: rgba(120,130,150,0.07);padding: 14px;border-radius: 12px}
[data-testid="stSidebar"] {border-right: 1px solid rgba(120,130,150,0.2)}
.small-note {color:#80909e;font-size:0.9rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained PyTorch model...")
def cached_model(folder: str, stamp: tuple):
    return load_bundle(folder)


def probability_figure(y_true, probs) -> go.Figure:
    fig = go.Figure()
    labels = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    for class_id, name, color in [(0, "Did not survive", "#ef6c71"), (1, "Survived", "#3db9a3")]:
        fig.add_histogram(x=probs[labels == class_id], name=name, marker_color=color, opacity=0.7, nbinsx=24)
    fig.update_layout(barmode="overlay", xaxis_title="Predicted survival probability", yaxis_title="Passengers", legend_orientation="h")
    return fig


def matrix_figure(matrix) -> go.Figure:
    matrix = np.asarray(matrix)
    fig = px.imshow(
        matrix, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
        text_auto=True, color_continuous_scale="Blues", aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Prediction", yaxis_title="True label")
    return fig


def performance_figures(y, p):
    roc_fpr, roc_tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision–recall curve"))
    fig.add_trace(go.Scatter(x=roc_fpr, y=roc_tpr, name="MLP ROC", line=dict(color="#3c94de", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", line=dict(color="gray", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=recall, y=precision, name="MLP PR", line=dict(color="#3db9a3", width=3)), row=1, col=2)
    fig.update_xaxes(title_text="False positive rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)
    fig.update_layout(height=370, showlegend=False)
    return fig


def display_metrics(metrics):
    cols = st.columns(5)
    entries = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC AUC", "roc_auc")]
    for col, (label, key) in zip(cols, entries):
        value = metrics[key]
        col.metric(label, f"{value:.3f}" if value is not None else "N/A")


st.title("🚢 Titanic Survival Intelligence")
st.caption("PyTorch classifier · Kaggle Titanic train.csv · leakage-safe validation & inference")

st.sidebar.header("Model & data")
artifact_path = st.sidebar.text_input("Trained artifacts directory", value="artifacts")
model_folder = Path(artifact_path).expanduser()
required_artifacts = ["model.pt", "preprocessor.joblib", "model_config.json"]
missing = [file for file in required_artifacts if not (model_folder / file).is_file()]
if missing:
    st.error("Missing trained model artifacts: " + ", ".join(missing))
    st.info("Run `python train.py` first. If Kaggle credentials are missing, follow README.md to enable downloading train.csv.")
    st.stop()

try:
    timestamps = tuple((model_folder / name).stat().st_mtime_ns for name in required_artifacts)
    preprocessor, model, config = cached_model(str(model_folder.resolve()), timestamps)
except Exception as exc:
    st.error(f"Could not load trained artifacts: {exc}")
    st.stop()

st.sidebar.success("Model loaded")
st.sidebar.caption(f"{config['n_train']} training rows · {config['n_validation']} validation rows · seed {config['seed']}")
source = st.sidebar.radio("Inference input", ["Held-out validation", "Upload CSV", "Local CSV path"])
df = None
source_name = ""

try:
    if source == "Held-out validation":
        csv_path = model_folder / "validation.csv"
        if not csv_path.is_file():
            st.error("No saved validation split found. Rerun `python train.py`.")
            st.stop()
        df = pd.read_csv(csv_path)
        source_name = str(csv_path)
    elif source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload a Titanic-style CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            source_name = uploaded.name
    else:
        supplied_path = st.sidebar.text_input("CSV path", value="data/example_inference.csv")
        if supplied_path.strip():
            csv_path = Path(supplied_path).expanduser()
            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                source_name = str(csv_path)
            else:
                st.warning(f"CSV not found: {csv_path}")
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Cannot read CSV: {exc}")
    st.stop()

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption("Threshold changes the class assignment, not predicted probabilities.")
with st.sidebar.expander("Expected input schema"):
    st.code(", ".join(REQUIRED_FEATURES))
    st.caption("Name is optional (title extraction). PassengerId and Survived are not features. Survived is only used for evaluation if provided.")

if df is None:
    st.info("Upload a CSV or enter a local path to begin inference.")
    st.stop()
if df.empty:
    st.warning("The chosen CSV has no rows.")
    st.stop()

try:
    probabilities = predict_probabilities(df, preprocessor, model)
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Invalid input schema: {exc}")
    st.stop()

predictions = (probabilities >= threshold).astype(int)
results = pd.DataFrame(index=df.index)
if "PassengerId" in df:
    results["PassengerId"] = df["PassengerId"]
results["survival_probability"] = np.round(probabilities, 6)
results["predicted_survived"] = predictions
if "Survived" in df:
    results["actual_survived"] = df["Survived"]

st.markdown(f"**Dataset:** `{source_name}` · **{len(df):,} rows** · **{len(df.columns)} raw columns**")
section_data, section_eval, section_infer = st.tabs(["Data overview", "Evaluation", "Predictions & export"])

with section_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers", f"{len(df):,}")
    c2.metric("Predicted survivors", f"{predictions.sum():,}")
    c3.metric("Average survival probability", f"{probabilities.mean():.1%}")
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.subheader("Missing values")
    missing_values = df.isna().sum().sort_values(ascending=False)
    if (missing_values > 0).any():
        missing_fig = px.bar(x=missing_values.index, y=missing_values.values, labels={"x": "Column", "y": "Missing rows"})
        st.plotly_chart(missing_fig, use_container_width=True)
    else:
        st.success("No missing values detected.")
    st.caption("Numerical values are imputed using training medians. Encoders and scalers were fit only on training rows.")

with section_eval:
    truth = pd.to_numeric(df["Survived"], errors="coerce") if "Survived" in df else None
    if truth is None:
        st.info("This CSV has no `Survived` label, so supervised evaluation is unavailable. Predictions are available in the next tab.")
    elif truth.isna().any() or not truth.isin([0, 1]).all():
        st.warning("`Survived` contains missing or non-binary values. Evaluation requires all rows to have valid 0/1 labels.")
    else:
        truth = truth.to_numpy(dtype=int)
        report = evaluate_binary(truth, probabilities, threshold=threshold)
        st.subheader("Classification metrics")
        display_metrics(report)
        st.caption(f"Metrics computed from this CSV with the current threshold ({threshold:.2f}). At one-class inputs, AUC is undefined.")
        left, right = st.columns([0.85, 1.15])
        with left:
            st.plotly_chart(matrix_figure(report["confusion_matrix"]), use_container_width=True)
        with right:
            st.plotly_chart(probability_figure(truth, probabilities), use_container_width=True)
        if len(np.unique(truth)) == 2:
            st.plotly_chart(performance_figures(truth, probabilities), use_container_width=True)
            frac_pos, mean_pred = calibration_curve(truth, probabilities, n_bins=8, strategy="quantile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Observed", line_color="#3db9a3"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash", color="gray")))
            fig.update_layout(title="Calibration (diagnostic)", xaxis_title="Mean predicted probability", yaxis_title="Observed survival fraction", height=335)
            st.plotly_chart(fig, use_container_width=True)
        if source == "Held-out validation" and abs(threshold - 0.5) < 1e-9:
            metrics_path = model_folder / "metrics.json"
            if metrics_path.is_file():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.subheader("Same-split baseline comparison")
                benchmark = pd.DataFrame({
                    "Model": ["PyTorch MLP", "Logistic regression"],
                    "Accuracy": [saved["pytorch_mlp"]["accuracy"], saved["logistic_regression"]["accuracy"]],
                    "F1": [saved["pytorch_mlp"]["f1"], saved["logistic_regression"]["f1"]],
                    "ROC AUC": [saved["pytorch_mlp"]["roc_auc"], saved["logistic_regression"]["roc_auc"]],
                })
                st.dataframe(benchmark.style.format({"Accuracy": "{:.3f}", "F1": "{:.3f}", "ROC AUC": "{:.3f}"}), hide_index=True, use_container_width=True)
                st.caption("Logistic regression uses the identical split and training-only preprocessing; it is an evaluation baseline, not the deployed inference model.")
            history_path = model_folder / "history.csv"
            if history_path.is_file():
                history = pd.read_csv(history_path)
                st.subheader("Training dynamics")
                fig = px.line(
                    history, x="epoch", y=["train_loss", "val_loss"],
                    labels={"value": "Binary cross-entropy", "epoch": "Epoch", "variable": "Curve"},
                    title="Fixed-epoch training loss (diagnostic only)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The final epoch is saved by design; validation loss was not used for checkpoint selection.")

with section_infer:
    st.subheader("Predictions")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions.csv", results.to_csv(index=False).encode("utf-8"),
        file_name="titanic_predictions.csv", mime="text/csv", type="primary",
    )
    st.caption("1 = predicted survivor, 0 = predicted non-survivor. Predictions are based only on passenger features.")
    if "actual_survived" in results and results["actual_survived"].isin([0, 1]).all():
        with st.expander("Inspect misclassified passengers"):
            errors = results[results["actual_survived"].astype(int) != results["predicted_survived"]]
            st.dataframe(errors, use_container_width=True, hide_index=True)
