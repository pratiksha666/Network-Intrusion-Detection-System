"""data_loader.py
Responsible only for LOADING and VALIDATING the dataset.
"""

import pandas as pd

# ---- Dataset constants (UNSW-NB15) ----
TARGET_COLUMN = "label"                          # 0 = Normal, 1 = Attack
CATEGORICAL_COLUMNS = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "attack_cat"]              # id = row number, attack_cat leaks the answer
MIN_ROWS = 100


def load_csv(file):
    """Read a CSV (file path or Streamlit uploaded file) into a DataFrame."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")
    except Exception as error:
        raise ValueError(f"Could not read the CSV file: {error}")

    # Make column names consistent: lowercase and no extra spaces
    df.columns = df.columns.str.strip().str.lower()
    return df


def validate_dataset(df):
    """Check that the dataset looks like UNSW-NB15.
    Returns a list of problems (empty list = dataset is valid)."""
    problems = []

    if len(df) < MIN_ROWS:
        problems.append(f"Dataset has only {len(df)} rows. At least {MIN_ROWS} are needed.")

    if TARGET_COLUMN not in df.columns:
        problems.append(f"Target column '{TARGET_COLUMN}' is missing.")
    else:
        labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            problems.append(f"'{TARGET_COLUMN}' must contain only 0 (Normal) and 1 (Attack).")
        elif labels.nunique() < 2:
            problems.append(f"'{TARGET_COLUMN}' has only one class. Both Normal and Attack are needed.")

    missing_cat = [c for c in CATEGORICAL_COLUMNS if c not in df.columns]
    if missing_cat:
        problems.append(f"Missing categorical columns: {', '.join(missing_cat)}.")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN and c not in DROP_COLUMNS]
    if len(feature_cols) < 10:
        problems.append("Too few feature columns. Use the UNSW-NB15 training/testing set CSV.")

    return problems


def get_missing_values(df):
    """Return a table of columns that have missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Percentage (%)": (missing.values / len(df) * 100).round(2),
    })
