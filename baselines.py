"""Classical baselines for the three-feature representation.

Settings are fixed and reproducible, not exhaustively tuned.
"""

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def sklearn_baselines() -> dict:
    return {
        "Logistic regression": lambda: LogisticRegression(max_iter=5000),
        "LDA": lambda: LinearDiscriminantAnalysis(),
        "QDA": lambda: QuadraticDiscriminantAnalysis(),
        "SVM (linear)": lambda: SVC(kernel="linear", C=1.0),
        "SVM (RBF)": lambda: SVC(kernel="rbf", C=10.0, gamma="scale"),
    }


class OrderedLogit:
    """statsmodels OrderedModel (logit link) -> integer predictions in {0,1,2}."""

    def __init__(self):
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        self._OM = OrderedModel
        self._res = None

    def fit(self, X, y):
        self._res = self._OM(np.asarray(y), np.asarray(X), distr="logit").fit(
            method="bfgs", disp=False
        )
        return self

    def predict(self, X):
        proba = self._res.model.predict(self._res.params, exog=np.asarray(X))
        return proba.argmax(axis=1).astype(int)


def ordered_logit_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False
