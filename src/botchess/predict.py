"""Interface de linha de comando para consultar o modelo final."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import chess

from .data import classify_phase
from .solutions.base import load_model


def build_context(board: chess.Board, rating: int | None, opponent_rating: int | None) -> dict:
    return {
        "player_rating": rating,
        "opponent_rating": opponent_rating,
        "phase": classify_phase(board),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ranqueia lances no estilo do modelo")
    parser.add_argument("--model", type=Path, default=Path("artifacts/demo_model.joblib"))
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--rating", type=int)
    parser.add_argument("--opponent-rating", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        raise SystemExit(f"FEN inválida: {exc}") from exc
    model = load_model(args.model)
    rating = args.rating or model.metadata.get("default_rating")
    opponent_rating = args.opponent_rating or rating
    context = build_context(board, rating, opponent_rating)
    ranking = model.rank(board, context)[: max(args.top_k, 1)]
    rows = [
        {
            "rank": index,
            "uci": move.uci(),
            "san": board.san(move),
            "probability": probability,
        }
        for index, (move, probability) in enumerate(ranking, start=1)
    ]
    if args.json:
        print(json.dumps({"fen": board.fen(), "model": model.name, "moves": rows}, indent=2))
    else:
        print(f"Modelo: {model.name}")
        print(f"Posição: {board.fen()}")
        for row in rows:
            print(
                f"{row['rank']}. {row['san']} ({row['uci']}) "
                f"p={row['probability']:.4f}"
            )


if __name__ == "__main__":
    main()
