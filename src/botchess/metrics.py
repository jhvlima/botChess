"""Métricas de ranqueamento e recortes comportamentais."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Iterable

import chess

from .solutions.base import ModelBundle


def rating_bucket(value: Any) -> str:
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if rating < 1000:
        return "<1000"
    if rating < 1200:
        return "1000-1199"
    if rating < 1400:
        return "1200-1399"
    if rating < 1600:
        return "1400-1599"
    return "1600+"


def _empty_accumulator() -> dict[str, float]:
    return {
        "positions": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "top5": 0.0,
        "reciprocal_rank": 0.0,
        "true_probability": 0.0,
        "legal_prediction": 0.0,
    }


def _finalize(values: dict[str, float]) -> dict[str, float | int]:
    count = int(values["positions"])
    if count == 0:
        return {"positions": 0}
    return {
        "positions": count,
        "top1_accuracy": values["top1"] / count,
        "top3_accuracy": values["top3"] / count,
        "top5_accuracy": values["top5"] / count,
        "mrr": values["reciprocal_rank"] / count,
        "mean_true_move_probability": values["true_probability"] / count,
        "legal_prediction_rate": values["legal_prediction"] / count,
    }


def evaluate_model(
    model: ModelBundle, examples: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    overall = _empty_accumulator()
    breakdowns: dict[str, dict[str, dict[str, float]]] = {
        field: defaultdict(_empty_accumulator)
        for field in ("color", "phase", "rating_bucket", "eco")
    }
    eco_counts: Counter[str] = Counter()
    start = time.perf_counter()
    invalid_targets = 0

    for example in examples:
        board = chess.Board(example["fen"])
        actual = chess.Move.from_uci(example["move_uci"])
        if actual not in board.legal_moves:
            invalid_targets += 1
            continue
        ranking = model.rank(board, example)
        ranked_moves = [move for move, _ in ranking]
        rank = ranked_moves.index(actual) + 1
        true_probability = ranking[rank - 1][1]
        predicted = ranked_moves[0] if ranked_moves else None

        dimensions = {
            "color": str(example.get("color", "unknown")),
            "phase": str(example.get("phase", "unknown")),
            "rating_bucket": rating_bucket(example.get("player_rating")),
            "eco": str(example.get("eco", "unknown")),
        }
        if dimensions["eco"]:
            eco_counts[dimensions["eco"]] += 1

        accumulators = [overall] + [
            breakdowns[field][value] for field, value in dimensions.items()
        ]
        for accumulator in accumulators:
            accumulator["positions"] += 1
            accumulator["top1"] += float(rank <= 1)
            accumulator["top3"] += float(rank <= 3)
            accumulator["top5"] += float(rank <= 5)
            accumulator["reciprocal_rank"] += 1.0 / rank
            accumulator["true_probability"] += true_probability
            accumulator["legal_prediction"] += float(
                predicted is not None and predicted in board.legal_moves
            )

    elapsed = time.perf_counter() - start
    finalized_breakdowns: dict[str, dict[str, Any]] = {}
    for field, groups in breakdowns.items():
        if field == "eco":
            selected = {name for name, _ in eco_counts.most_common(12)}
            groups = {name: values for name, values in groups.items() if name in selected}
        finalized_breakdowns[field] = {
            name: _finalize(values) for name, values in sorted(groups.items())
        }

    result = _finalize(overall)
    result.update(
        {
            "invalid_targets": invalid_targets,
            "evaluation_seconds": elapsed,
            "milliseconds_per_position": (
                elapsed * 1000 / int(overall["positions"])
                if overall["positions"]
                else 0.0
            ),
            "breakdowns": finalized_breakdowns,
        }
    )
    return result
