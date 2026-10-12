"""Dataset handling: loading, ordinal labels, standardization, and splits.

CSV schema (one row per labelled interval-level observation):

    user_id, timestamp, gsr, heart_rate, skin_temperature, stress_label

user_id and timestamp are used only for grouping/leakage control and the
chronological split; they are never predictive features.

Labels are ORDINAL and mapped explicitly (low=0, medium=1, high=2) via
labels.py. This avoids sklearn LabelEncoder's alphabetical order, which would
silently break every ordinal metric.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["low", "medium", "high"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

FEATURES = ["gsr", "heart_rate", "skin_temperature"]


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = FEATURES + ["stress_label", "user_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    raw = df["stress_label"].astype(str).str.strip().str.lower()
    if not set(raw).issubset(set(CLASS_NAMES)):
        raise ValueError(f"stress_label must be a subset of {CLASS_NAMES}; got {sorted(set(raw))}")
    df["y"] = raw.map(LABEL_TO_ID).astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(np.float32), df["y"].to_numpy(np.int64)


def standardize(train_X, *other_X):
    """Fit StandardScaler on TRAIN ONLY; apply to all partitions."""
    scaler = StandardScaler().fit(train_X)
    return (scaler, scaler.transform(train_X), *[scaler.transform(X) for X in other_X])


def loso_folds(df: pd.DataFrame):
    """Yield (held_out_user, train_df, test_df) per participant (LOSO)."""
    for user in sorted(df["user_id"].unique()):
        test = df[df["user_id"] == user].reset_index(drop=True)
        train = df[df["user_id"] != user].reset_index(drop=True)
        yield user, train, test


def chronological_split(df: pd.DataFrame, frac=(0.6, 0.2, 0.2)):
    """Per-user chronological split pooled across users -> (train, val, test)."""
    if "timestamp" not in df.columns:
        raise ValueError("chronological split requires a 'timestamp' column")
    tr, va, te = [], [], []
    for _, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        dates = sorted(g["timestamp"].dt.normalize().unique())
        n = len(dates)
        i1, i2 = int(round(n * frac[0])), int(round(n * (frac[0] + frac[1])))
        day = g["timestamp"].dt.normalize()
        tr.append(g[day.isin(set(dates[:i1]))])
        va.append(g[day.isin(set(dates[i1:i2]))])
        te.append(g[day.isin(set(dates[i2:]))])
    cat = lambda parts: pd.concat(parts).reset_index(drop=True)
    return cat(tr), cat(va), cat(te)
