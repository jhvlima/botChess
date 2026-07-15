"""Solução de árvores que venceu a comparação experimental."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


NAME = "hist_gradient_boosting"


def build_estimator(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )


def fit_estimator(
    estimator: HistGradientBoostingClassifier,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> Any:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    weights = np.ones(len(labels), dtype=np.float32)
    weights[labels == 1] = negatives / positives
    return estimator.fit(matrix, labels, sample_weight=weights)
