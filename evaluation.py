"""evaluation.py
Responsible only for MEASURING model performance and building charts.
"""

import pandas as pd
import plotly.express as px
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def evaluate_model(pipeline, X_test, y_test):
    """Predict on the test set and calculate all metrics (real values, nothing fake)."""
    y_pred = pipeline.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }


def confusion_matrix_figure(cm):
    """Heatmap of the confusion matrix."""
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted Normal", "Predicted Attack"],
        y=["Actual Normal", "Actual Attack"],
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      height=380, coloraxis_showscale=False)
    return fig


def comparison_table(results):
    """results = {model_name: metrics_dict}  ->  DataFrame with one row per model."""
    rows = []
    for name, m in results.items():
        rows.append({"Model": name, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
                     "Recall": m["Recall"], "F1-Score": m["F1-Score"],
                     "Training Time (s)": m.get("train_time")})
    return pd.DataFrame(rows)


def comparison_figure(results):
    """Grouped bar chart comparing all trained models."""
    table = comparison_table(results)
    long = table.melt(id_vars="Model",
                      value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                      var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_sequence=["#1f77ff", "#4cc9f0", "#7b8cff", "#a0c4ff"])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.05], height=420)
    return fig
