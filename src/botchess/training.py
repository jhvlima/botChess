"""Treino comparativo de modelos de ranqueamento de lances legais."""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from pathlib import Path
from typing import Any, Iterable

import chess
import numpy as np

from .data import read_jsonl
from .features import FEATURE_COUNT, candidate_features
from .metrics import evaluate_model
from .solutions.base import ModelBundle, save_model
from .solutions.registry import SOLUTION_NAMES, get_solution


MODEL_NAMES = SOLUTION_NAMES


def _sample_candidates(
    board: chess.Board,
    actual: chess.Move,
    max_negatives: int,
    rng: random.Random,
) -> list[chess.Move]:
    negatives = [move for move in board.legal_moves if move != actual]
    if max_negatives >= 0 and len(negatives) > max_negatives:
        negatives = rng.sample(negatives, max_negatives)
    candidates = [actual, *negatives]
    rng.shuffle(candidates)
    return candidates


def build_candidate_matrix(
    examples: list[dict[str, Any]], max_negatives: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    max_candidates = len(examples) * (max_negatives + 1 if max_negatives >= 0 else 64)
    matrix = np.empty((max_candidates, FEATURE_COUNT), dtype=np.float32)
    labels = np.empty(max_candidates, dtype=np.int8)
    cursor = 0

    for example in examples:
        board = chess.Board(example["fen"])
        actual = chess.Move.from_uci(example["move_uci"])
        if actual not in board.legal_moves:
            raise ValueError(f"Alvo ilegal no exemplo {example.get('example_id')}")
        rng = random.Random(f"{seed}:{example.get('example_id')}")
        candidates = _sample_candidates(board, actual, max_negatives, rng)
        required = cursor + len(candidates)
        if required > len(matrix):
            growth = max(required - len(matrix), len(matrix) // 2, 1024)
            matrix = np.resize(matrix, (len(matrix) + growth, FEATURE_COUNT))
            labels = np.resize(labels, len(labels) + growth)
        for move in candidates:
            matrix[cursor] = candidate_features(board, move, example)
            labels[cursor] = int(move == actual)
            cursor += 1

    return matrix[:cursor], labels[:cursor]


def _peak_rss_mb() -> float:
    # Linux informa ru_maxrss em KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def train_models(
    train_examples: list[dict[str, Any]],
    model_names: tuple[str, ...],
    max_negatives: int,
    seed: int,
) -> tuple[dict[str, ModelBundle], dict[str, dict[str, Any]]]:
    models: dict[str, ModelBundle] = {}
    stats: dict[str, dict[str, Any]] = {}
    ratings = [
        int(example["player_rating"])
        for example in train_examples
        if example.get("player_rating") is not None
    ]
    default_rating = int(np.median(ratings)) if ratings else 1200

    direct_names = tuple(
        name for name in model_names if not get_solution(name).candidate_model
    )
    for name in direct_names:
        solution = get_solution(name)
        if solution.train_direct is None:
            raise RuntimeError(f"Solução direta sem função de treino: {name}")
        start = time.perf_counter()
        models[name] = solution.train_direct(train_examples)
        stats[name] = {
            "training_seconds": time.perf_counter() - start,
            "candidate_rows": len(train_examples),
            "peak_rss_mb": _peak_rss_mb(),
        }

    learned_names = tuple(
        name for name in model_names if get_solution(name).candidate_model
    )
    if learned_names:
        matrix_start = time.perf_counter()
        matrix, labels = build_candidate_matrix(train_examples, max_negatives, seed)
        matrix_seconds = time.perf_counter() - matrix_start
        positives = int(labels.sum())
        negatives = len(labels) - positives
        if not positives or not negatives:
            raise RuntimeError("O treino precisa de exemplos positivos e negativos")

        for name in learned_names:
            solution = get_solution(name)
            if solution.build_estimator is None or solution.fit_estimator is None:
                raise RuntimeError(f"Solução candidata incompleta: {name}")
            estimator = solution.build_estimator(seed)
            start = time.perf_counter()
            solution.fit_estimator(estimator, matrix, labels)
            elapsed = time.perf_counter() - start
            models[name] = ModelBundle(
                name=name,
                kind="candidate_classifier",
                estimator=estimator,
                metadata={
                    "default_rating": default_rating,
                    "feature_count": FEATURE_COUNT,
                    "max_negative_candidates": max_negatives,
                    "seed": seed,
                },
            )
            stats[name] = {
                "matrix_build_seconds": matrix_seconds,
                "training_seconds": elapsed,
                "candidate_rows": len(labels),
                "positive_rows": positives,
                "negative_rows": negatives,
                "peak_rss_mb": _peak_rss_mb(),
            }
    return models, stats


def _metric(value: Any) -> str:
    return f"{float(value):.4f}"


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Comparação dos modelos",
        "",
        "Resultados produzidos pelo pipeline local em divisões cronológicas. A seleção",
        "usa validação; o conjunto de teste não participa da escolha.",
        "",
        "| Modelo | Val. Top-1 | Val. Top-3 | Val. MRR | Teste Top-1 | Teste Top-3 | Teste MRR | ms/posição |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        validation = result["validation"]
        test = result["test"]
        lines.append(
            "| {name} | {v1} | {v3} | {vmrr} | {t1} | {t3} | {tmrr} | {latency} |".format(
                name=name,
                v1=_metric(validation["top1_accuracy"]),
                v3=_metric(validation["top3_accuracy"]),
                vmrr=_metric(validation["mrr"]),
                t1=_metric(test["top1_accuracy"]),
                t3=_metric(test["top3_accuracy"]),
                tmrr=_metric(test["mrr"]),
                latency=f"{float(test['milliseconds_per_position']):.2f}",
            )
        )
    lines.extend(
        [
            "",
            f"Modelo selecionado: **{report['selected_model']}**.",
            "",
            "Critério: maior Top-1 na validação, com MRR como desempate. Todos os",
            "modelos ranqueiam somente lances legais; por isso a taxa de legalidade",
            "esperada é 100%.",
            "",
            "Depois da seleção, o vencedor foi retreinado com treino + validação.",
            "No teste isolado, o artefato final obteve Top-1 de "
            f"**{_metric(report['final_model']['test']['top1_accuracy'])}**, Top-3 de "
            f"**{_metric(report['final_model']['test']['top3_accuracy'])}** e MRR de "
            f"**{_metric(report['final_model']['test']['mrr'])}**.",
            "",
            "Os recortes completos por cor, fase, rating e ECO estão em",
            "[`latest_metrics.json`](latest_metrics.json).",
            "",
        ]
    )
    return "\n".join(lines)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    train_examples = list(read_jsonl(args.data_dir / "train.jsonl"))
    validation_examples = list(read_jsonl(args.data_dir / "validation.jsonl"))
    test_examples = list(read_jsonl(args.data_dir / "test.jsonl"))
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    if not train_examples or not validation_examples or not test_examples:
        raise RuntimeError("Os três conjuntos precisam conter exemplos")

    model_names = tuple(args.models)
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {', '.join(sorted(unknown))}")
    models, training_stats = train_models(
        train_examples, model_names, args.max_negatives, args.seed
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_results: dict[str, Any] = {}
    for name, model in models.items():
        artifact_path = args.artifacts_dir / "solutions" / name / "model.joblib"
        save_model(model, artifact_path)
        validation_metrics = evaluate_model(model, validation_examples)
        test_metrics = evaluate_model(model, test_examples)
        training_stats[name]["artifact_bytes"] = artifact_path.stat().st_size
        model_results[name] = {
            "training": training_stats[name],
            "validation": validation_metrics,
            "test": test_metrics,
            "artifact": str(artifact_path),
        }

    selected = max(
        model_results,
        key=lambda name: (
            model_results[name]["validation"]["top1_accuracy"],
            model_results[name]["validation"]["mrr"],
            -model_results[name]["validation"]["milliseconds_per_position"],
        ),
    )

    # A validação decide a abordagem. Só depois disso o vencedor aproveita os
    # dois conjuntos não finais; o teste continua sem participar do ajuste.
    final_models, final_training_stats = train_models(
        train_examples + validation_examples,
        (selected,),
        args.max_negatives,
        args.seed,
    )
    final_model = final_models[selected]
    final_path = args.artifacts_dir / "final_model.joblib"
    save_model(final_model, final_path)
    final_training_stats[selected]["artifact_bytes"] = final_path.stat().st_size
    final_test_metrics = evaluate_model(final_model, test_examples)

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "selection_rule": "validation_top1_then_mrr_then_latency",
        "selected_model": selected,
        "final_artifact": str(final_path),
        "dataset_examples": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples),
        },
        "models": model_results,
        "final_model": {
            "name": selected,
            "trained_examples": len(train_examples) + len(validation_examples),
            "training": final_training_stats[selected],
            "test": final_test_metrics,
        },
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "latest_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.reports_dir / "model-comparison.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina e compara políticas de jogadas")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES
    )
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Limite opcional para testes rápidos; 0 usa todo o treino",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_training(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
