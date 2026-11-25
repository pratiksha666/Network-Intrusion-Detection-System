"""preprocessing.py
Responsible only for CLEANING data, splitting it, and building the
preprocessing step (imputing, encoding, scaling).

The imputer / encoder / scaler are placed inside a Scikit-learn Pipeline so they
are fitted on the TRAINING data only. This prevents data leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import CATEGORICAL_COLUMNS, DROP_COLUMNS, TARGET_COLUMN


def clean_data(df):
    """Basic cleaning: drop useless columns, fix data types, remove bad rows."""
    df = df.copy()

    # 'id' is just a row number; 'attack_cat' tells the attack type (would leak the label)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Categorical columns -> clean text; numeric columns -> numbers
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype("string").str.strip().astype(object)
            df[col] = df[col].where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def split_features_target(df):
    """Separate the input features (X) from the target (y). Target is NOT in X."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """Stratified split keeps the Normal/Attack ratio the same in train and test."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def get_column_types(X):
    """Return (numeric_columns, categorical_columns)."""
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_preprocessor(X):
    """ColumnTransformer:
       - numeric columns     : fill missing with median, then StandardScaler
       - categorical columns : fill missing with most frequent, then One-Hot Encoding
    """
    numeric_cols, categorical_cols = get_column_types(X)

    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("numeric", numeric_steps, numeric_cols),
        ("categorical", categorical_steps, categorical_cols),
    ])
