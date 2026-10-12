"""Evaluation metrics, including ordinal-aware measures.

All functions take integer label arrays with the ordinal coding
low=0, medium=1, high=2 (see labels.py / data.py).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

LABELS = [0, 1, 2]


def basic(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
    }


def per_class_recall(y_true, y_pred) -> dict:
    r = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    return {"recall_low": r[0], "recall_medium": r[1], "recall_high": r[2]}


def ordinal(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return {
        "ordinal_mae": float(np.mean(np.abs(yt - yp))),
        "qwk": cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS),
        "within_one_acc": float(np.mean(np.abs(yt - yp) <= 1)),
    }


def all_metrics(y_true, y_pred) -> dict:
    out = {}
    out.update(basic(y_true, y_pred))
    out.update(per_class_recall(y_true, y_pred))
    out.update(ordinal(y_true, y_pred))
    return out


def cm(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABELS)
