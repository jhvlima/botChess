"""Adaptador UCI mínimo para jogar com o modelo personalizado."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Iterable

import chess

from .data import classify_phase
from .solutions.base import load_model


def parse_position(command: str) -> chess.Board:
    tokens = shlex.split(command)
    if len(tokens) < 2 or tokens[0] != "position":
        raise ValueError("comando position inválido")
    index = 1
    if tokens[index] == "startpos":
        board = chess.Board()
        index += 1
    elif tokens[index] == "fen":
        if len(tokens) < index + 7:
            raise ValueError("FEN incompleta")
        board = chess.Board(" ".join(tokens[index + 1 : index + 7]))
        index += 7
    else:
        raise ValueError("position deve usar startpos ou fen")
    if index < len(tokens):
        if tokens[index] != "moves":
            raise ValueError("esperado token moves")
        for token in tokens[index + 1 :]:
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                raise ValueError(f"lance ilegal no histórico: {token}")
            board.push(move)
    return board


def run_uci(model_path: Path, rating: int | None) -> None:
    model = load_model(model_path)
    board = chess.Board()
    effective_rating = rating or model.metadata.get("default_rating", 1200)
    for raw_line in sys.stdin:
        command = raw_line.strip()
        if not command:
            continue
        if command == "uci":
            print("id name botChess")
            print("id author jhvlima")
            print("uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command == "ucinewgame":
            board = chess.Board()
        elif command.startswith("position "):
            try:
                board = parse_position(command)
            except ValueError as exc:
                print(f"info string erro: {exc}", flush=True)
        elif command.startswith("go"):
            context = {
                "player_rating": effective_rating,
                "opponent_rating": effective_rating,
                "phase": classify_phase(board),
            }
            move = model.choose_move(board, context)
            print(f"bestmove {move.uci() if move else '(none)'}", flush=True)
        elif command == "stop":
            continue
        elif command == "quit":
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa o modelo como engine UCI")
    parser.add_argument("--model", type=Path, default=Path("artifacts/demo_model.joblib"))
    parser.add_argument("--rating", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_uci(args.model, args.rating)


if __name__ == "__main__":
    main()
