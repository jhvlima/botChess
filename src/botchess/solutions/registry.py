"""Registro único das soluções comparáveis pelo pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .frequency.model import train as train_frequency
from .hist_gradient_boosting.model import (
    build_estimator as build_hist_gradient_boosting,
)
from .hist_gradient_boosting.model import fit_estimator as fit_hist_gradient_boosting
from .logistic.model import build_estimator as build_logistic
from .logistic.model import fit_estimator as fit_logistic


@dataclass(frozen=True)
class SolutionSpec:
    name: str
    candidate_model: bool
    train_direct: Callable[[Iterable[dict[str, Any]]], Any] | None = None
    build_estimator: Callable[[int], Any] | None = None
    fit_estimator: Callable[[Any, np.ndarray, np.ndarray], Any] | None = None


SOLUTIONS = {
    "frequency": SolutionSpec(
        name="frequency",
        candidate_model=False,
        train_direct=train_frequency,
    ),
    "logistic": SolutionSpec(
        name="logistic",
        candidate_model=True,
        build_estimator=build_logistic,
        fit_estimator=fit_logistic,
    ),
    "hist_gradient_boosting": SolutionSpec(
        name="hist_gradient_boosting",
        candidate_model=True,
        build_estimator=build_hist_gradient_boosting,
        fit_estimator=fit_hist_gradient_boosting,
    ),
}

SOLUTION_NAMES = tuple(SOLUTIONS)


def get_solution(name: str) -> SolutionSpec:
    try:
        return SOLUTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Solução desconhecida: {name}") from exc
