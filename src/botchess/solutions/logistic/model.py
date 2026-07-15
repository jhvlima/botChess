"""Solução linear de classificação de pares posição-lance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NAME = "logistic"


def build_estimator(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=0.0003,
                    l1_ratio=0.05,
                    max_iter=1500,
                    tol=1e-4,
                    class_weight="balanced",
                    average=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_estimator(
    estimator: Pipeline, matrix: np.ndarray, labels: np.ndarray
) -> Any:
    return estimator.fit(matrix, labels)
