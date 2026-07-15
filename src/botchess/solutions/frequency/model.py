"""Solução simples: frequência histórica de cada movimento UCI."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from ...features import FEATURE_COUNT
from ..base import ModelBundle


NAME = "frequency"


def train(examples: Iterable[dict[str, Any]]) -> ModelBundle:
    examples = list(examples)
    counts = Counter(str(example["move_uci"]) for example in examples)
    ratings = [
        int(example["player_rating"])
        for example in examples
        if example.get("player_rating") is not None
    ]
    default_rating = int(np.median(ratings)) if ratings else 1200
    return ModelBundle(
        name=NAME,
        kind="frequency",
        move_counts=dict(counts),
        metadata={"default_rating": default_rating, "feature_count": FEATURE_COUNT},
    )
