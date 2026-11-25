"""app.py
Streamlit frontend. Run with:  streamlit run app.py
Each page calls functions from the src/ folder (backend).
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import get_missing_values, load_csv, validate_dataset
from src.evaluation import (comparison_figure, comparison_table,
                            confusion_matrix_figure, evaluate_model)
from src.predict import list_saved_models, load_saved_model, predict_traffic
from src.preprocessing import (clean_data, get_column_types,
                               split_features_target, split_train_test)
from src.train_model import MODEL_NAMES, save_model, train_model

st.set_page_config(page_title="Network Intrusion Detection System", page_icon="🛡️", layout="wide")

# ---------- Dark navy + blue theme (simple CSS) ----------
st.markdown("""
<style>
.stApp { background-color: #0a1128; color: #e6edf7; }
[data-testid="stSidebar"] { background-color: #0d1b3e; }
h1, h2, h3 { color: #4da3ff !important; }
[data-testid="stMetric"] { background-color: #12224d; border: 1px solid #1f3a8a;
                           border-radius: 10px; padding: 14px; }
[data-testid="stMetricValue"] { color: #4da3ff; }
.stButton > button, .stDownloadButton > button { background-color: #1f6feb; color: white;
                           border: none; border-radius: 8px; }
.stButton > button:hover, .stDownloadButton > button:hover { background-color: #3b8bff; color: white; }
.card { background-color: #12224d; border: 1px solid #1f3a8a; border-radius: 10px;
        padding: 16px; height: 100%; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state: keeps data between page changes ----------
DEFAULTS = {"raw_df": None, "file_name": None, "split": None,
            "results": {}, "pipelines": {}, "prediction": None}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def need_dataset():
    """Stop the page with a clear message if no valid dataset is loaded."""
    if st.session_state.raw_df is None:
        st.warning("No dataset uploaded yet. Go to **Upload Dataset** and upload a CSV file.")
        st.stop()


def need_split():
    """Stop the page if preprocessing has not been done."""
    need_dataset()
    if st.session_state.split is None:
        st.warning("Data is not preprocessed yet. Go to **Data Preprocessing** first.")
        st.stop()


# ---------- Pages ----------
def page_home():
    st.title("🛡️ Network Intrusion Detection System")
    st.write("A machine learning web application that classifies network traffic as "
             "**Normal** or **Attack** using the UNSW-NB15 dataset.")

    st.subheader("What is Network Intrusion Detection?")
    st.write("A Network Intrusion Detection System (NIDS) monitors network traffic and raises an "
             "alert when it finds suspicious activity such as DoS, exploits or port scans. "
             "Here, supervised machine learning models learn from labelled traffic records "
             "(0 = Normal, 1 = Attack) and then predict the label of new traffic.")

    st.subheader("Project Features")
    features = [
        ("📂 Upload Dataset", "Upload the UNSW-NB15 CSV, preview it and check missing values."),
        ("🧹 Preprocessing", "Imputation, one-hot encoding, scaling and a stratified train-test split."),
        ("🤖 Model Training", "Random Forest, KNN, SVM and Logistic Regression."),
        ("📊 Evaluation", "Accuracy, Precision, Recall, F1-score, confusion matrix and comparison chart."),
        ("🔍 Prediction", "Predict Normal or Attack for network traffic records."),
        ("⬇️ Download", "Download prediction results as a CSV file."),
    ]
    cols = st.columns(3)
    for i, (title, text) in enumerate(features):
        cols[i % 3].markdown(f"<div class='card'><b>{title}</b><br>{text}</div>", unsafe_allow_html=True)
        cols[i % 3].write("")

    st.subheader("How to use")
    st.markdown("1. Upload Dataset → 2. Data Preprocessing → 3. Model Training → "
                "4. Model Evaluation → 5. Network Traffic Prediction → 6. Download Results")


def page_upload():
    st.title("📂 Upload Dataset")
    st.write("Upload the UNSW-NB15 training or testing CSV "
             "(e.g. `UNSW_NB15_training-set.csv`). It must contain the `label` column.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    # Load only when a NEW file is uploaded (avoids re-reading the data on every click)
    if uploaded is not None and uploaded.name != st.session_state.file_name:
        try:
            df = load_csv(uploaded)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        problems = validate_dataset(df)
        if problems:
            for p in problems:
                st.error(p)
            st.stop()
        st.session_state.raw_df = df
        st.session_state.file_name = uploaded.name
        # A new dataset means old results are no longer valid
        st.session_state.split = None
        st.session_state.results = {}
        st.session_state.pipelines = {}

    if st.session_state.raw_df is None:
        st.info("Please upload a CSV file to continue.")
        return

    df = st.session_state.raw_df
    st.success(f"Dataset loaded: {st.session_state.file_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    left, right = st.columns(2)
    with left:
        st.subheader("Missing Values")
        missing = get_missing_values(df)
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing)
    with right:
        st.subheader("Normal vs Attack")
        counts = df["label"].map({0: "Normal", 1: "Attack"}).value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        fig = px.pie(counts, names="Class", values="Count", hole=0.45,
                     color="Class", color_discrete_map={"Normal": "#4da3ff", "Attack": "#ff5c7a"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig)


def page_preprocessing():
    st.title("🧹 Data Preprocessing")
    need_dataset()

    st.write("Steps performed:")
    st.markdown(
        "- Remove `id` and `attack_cat` (`attack_cat` would leak the answer)\n"
        "- Convert numeric columns to numbers\n"
        "- Separate features (X) and target (`label`)\n"
        "- Stratified train-test split\n"
        "- Inside the model pipeline (fitted on training data only, so no data leakage): "
        "median imputation + scaling for numeric columns, most-frequent imputation + "
        "one-hot encoding for categorical columns (`proto`, `service`, `state`)")

    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100

    if st.button("Run Preprocessing"):
        try:
            with st.spinner("Preprocessing..."):
                clean_df = clean_data(st.session_state.raw_df)
                X, y = split_features_target(clean_df)
                X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size)
            st.session_state.split = {"X_train": X_train, "X_test": X_test,
                                      "y_train": y_train, "y_test": y_test}
            # Old models were trained on an older split, so reset them
            st.session_state.results = {}
            st.session_state.pipelines = {}
        except Exception as error:
            st.error(f"Preprocessing failed: {error}")
            st.stop()

    split = st.session_state.split
    if split is None:
        st.info("Click **Run Preprocessing** to continue.")
        return

    st.success("Preprocessing completed.")
    numeric_cols, categorical_cols = get_column_types(split["X_train"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", f"{len(split['X_train']):,}")
    c2.metric("Testing rows", f"{len(split['X_test']):,}")
    c3.metric("Numeric features", len(numeric_cols))
    c4.metric("Categorical features", len(categorical_cols))
    st.write("**Categorical columns:**", ", ".join(categorical_cols))
    st.write("**Attack share in train / test:** "
             f"{split['y_train'].mean():.1%} / {split['y_test'].mean():.1%}")
    st.subheader("Features preview (X)")
    st.dataframe(split["X_train"].head(10))


def page_training():
    st.title("🤖 Model Training")
    need_split()
    split = st.session_state.split

    svm_rows = st.sidebar.number_input("Max training rows for SVM", 2000, 100000, 20000, step=2000,
                                       help="SVM is slow on large data, so it trains on a sample.")

    def train_and_store(name):
        """Train one model, evaluate it on the test set, save it with Joblib."""
        pipeline, seconds = train_model(name, split["X_train"], split["y_train"], svm_rows)
        metrics = evaluate_model(pipeline, split["X_test"], split["y_test"])
        metrics["train_time"] = seconds
        save_model(pipeline, name, split["X_train"].columns)
        st.session_state.pipelines[name] = pipeline
        st.session_state.results[name] = metrics

    choice = st.selectbox("Select a model", MODEL_NAMES)
    col1, col2 = st.columns(2)

    try:
        if col1.button("Train selected model"):
            with st.spinner(f"Training {choice}..."):
                train_and_store(choice)
            st.success(f"{choice} trained and saved.")

        if col2.button("Train all models"):
            bar = st.progress(0.0)
            for i, name in enumerate(MODEL_NAMES):
                bar.progress(i / len(MODEL_NAMES), text=f"Training {name}...")
                train_and_store(name)
            bar.progress(1.0, text="All models trained.")
            st.success("All models trained and saved in the `models/` folder.")
    except Exception as error:
        st.error(f"Training failed: {error}")

    if st.session_state.results:
        st.subheader("Training Results")
        table = comparison_table(st.session_state.results)
        st.dataframe(table.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                                         "Recall": "{:.4f}", "F1-Score": "{:.4f}"}))
    else:
        st.info("No model trained yet.")


def page_evaluation():
    st.title("📊 Model Evaluation")
    need_split()
    results = st.session_state.results
    if not results:
        st.warning("No trained models yet. Go to **Model Training** first.")
        st.stop()

    name = st.selectbox("Select a trained model", list(results.keys()))
    m = results[name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
    c2.metric("Precision", f"{m['Precision']:.4f}")
    c3.metric("Recall", f"{m['Recall']:.4f}")
    c4.metric("F1-Score", f"{m['F1-Score']:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        st.plotly_chart(confusion_matrix_figure(m["confusion_matrix"]))
    with right:
        st.subheader("Model Comparison")
        st.plotly_chart(comparison_figure(results))

    st.subheader("Comparison Table")
    st.dataframe(comparison_table(results))


def page_prediction():
    st.title("🔍 Network Traffic Prediction")
    saved = list_saved_models()
    if not saved:
        st.warning("No trained model found. Train a model first in **Model Training**.")
        st.stop()

    model_file = st.selectbox("Select a saved model", saved)
    source = st.radio("Traffic data source",
                      ["Sample from test set", "Upload a new CSV"], horizontal=True)

    data = None
    if source == "Sample from test set":
        if st.session_state.split is None:
            st.warning("Run preprocessing first, or choose **Upload a new CSV**.")
            st.stop()
        split = st.session_state.split
        n = st.slider("Number of records", 10, min(5000, len(split["X_test"])), 100)
        data = split["X_test"].sample(n, random_state=1).copy()
        data["label"] = split["y_test"].loc[data.index]   # real label, only for comparison
    else:
        up = st.file_uploader("Upload traffic CSV (same columns as the training data)",
                              type=["csv"], key="predict_upload")
        if up is None:
            st.info("Upload a CSV file to predict.")
            st.stop()
        try:
            data = load_csv(up)
        except ValueError as error:
            st.error(str(error))
            st.stop()

    if st.button("Predict"):
        try:
            bundle = load_saved_model(model_file)
            st.session_state.prediction = predict_traffic(bundle, data)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.stop()

    result = st.session_state.prediction
    if result is None:
        return

    total = len(result)
    attacks = int((result["Prediction"] == 1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{total:,}")
    c2.metric("🟢 Normal", f"{total - attacks:,}")
    c3.metric("🔴 Attack", f"{attacks:,}")
    if attacks > 0:
        st.error(f"⚠️ {attacks} of {total} records were classified as ATTACK traffic.")
    else:
        st.success("All records were classified as NORMAL traffic.")

    show = ["Result", "Prediction"] + [c for c in result.columns if c not in ("Result", "Prediction")]
    st.dataframe(result[show].head(200))

    st.info("Go to **Download Results** to save these predictions as a CSV file.")


def page_download():
    st.title("⬇️ Download Results")
    result = st.session_state.prediction
    if result is None:
        st.warning("No prediction results yet. Run a prediction in **Network Traffic Prediction** first.")
        st.stop()
    st.write(f"{len(result):,} prediction records are ready.")
    st.dataframe(result.head(20))
    st.download_button("⬇️ Download predictions as CSV",
                       result.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_results.csv", mime="text/csv")


# ---------- Sidebar navigation ----------
PAGES = {
    "🏠 Home Dashboard": page_home,
    "📂 Upload Dataset": page_upload,
    "🧹 Data Preprocessing": page_preprocessing,
    "🤖 Model Training": page_training,
    "📊 Model Evaluation": page_evaluation,
    "🔍 Network Traffic Prediction": page_prediction,
    "⬇️ Download Results": page_download,
}
st.sidebar.title("🛡️ NIDS")
selected = st.sidebar.radio("Navigation", list(PAGES.keys()))
PAGES[selected]()
