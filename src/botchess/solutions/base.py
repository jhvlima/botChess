"""Contrato compartilhado pelos modelos de todas as soluções."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import joblib
import numpy as np

from ..features import candidate_features


@dataclass
class ModelBundle:
    name: str
    kind: str
    estimator: Any = None
    move_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_legal_moves(
        self, board: chess.Board, context: dict[str, Any] | None = None
    ) -> tuple[list[chess.Move], np.ndarray]:
        moves = list(board.legal_moves)
        if not moves:
            return [], np.asarray([], dtype=np.float64)

        if self.kind == "frequency":
            raw = np.asarray(
                [self.move_counts.get(move.uci(), 0) + 1.0 for move in moves],
                dtype=np.float64,
            )
        else:
            matrix = np.vstack([candidate_features(board, move, context) for move in moves])
            probabilities = self.estimator.predict_proba(matrix)
            classes = list(self.estimator.classes_)
            positive_index = classes.index(1)
            raw = np.asarray(probabilities[:, positive_index], dtype=np.float64)
            raw = np.maximum(raw, 1e-12)

        total = float(raw.sum())
        scores = raw / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
        return moves, scores

    def rank(
        self, board: chess.Board, context: dict[str, Any] | None = None
    ) -> list[tuple[chess.Move, float]]:
        moves, scores = self.score_legal_moves(board, context)
        order = np.argsort(-scores, kind="stable")
        return [(moves[index], float(scores[index])) for index in order]

    def choose_move(
        self, board: chess.Board, context: dict[str, Any] | None = None
    ) -> chess.Move | None:
        ranking = self.rank(board, context)
        return ranking[0][0] if ranking else None


def save_model(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_model(path: Path) -> ModelBundle:
    value = joblib.load(path)
    if not isinstance(value, ModelBundle):
        raise TypeError(f"Artefato não é um modelo botChess: {path}")
    return value
