"""Auditoria independente dos invariantes do dataset preparado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import chess

from .data import SPLITS, read_jsonl


def _raw_game_colors(raw_dir: Path, username: str) -> dict[str, str]:
    target = username.casefold()
    colors: dict[str, str] = {}
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("games", []):
            pgn = str(raw.get("pgn") or "")
            fallback = hashlib.sha256(pgn.encode("utf-8")).hexdigest()[:20]
            game_id = str(raw.get("uuid") or raw.get("url") or fallback)
            white = str((raw.get("white") or {}).get("username", "")).casefold()
            black = str((raw.get("black") or {}).get("username", "")).casefold()
            if white == target:
                colors[game_id] = "white"
            elif black == target:
                colors[game_id] = "black"
    return colors


def validate_dataset(data_dir: Path, raw_dir: Path, username: str) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_colors = _raw_game_colors(raw_dir, username)
    seen_example_ids: set[str] = set()
    prior_positions: set[str] = set()
    prior_games: set[str] = set()
    previous_last_date: str | None = None
    result: dict[str, Any] = {"valid": False, "splits": {}, "errors": errors}

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        expected_hash = manifest["splits"][split]["sha256"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"hash divergente em {split}")

        positions: set[str] = set()
        games: set[str] = set()
        dates: list[str] = []
        rows = 0
        for row in read_jsonl(path):
            rows += 1
            example_id = str(row.get("example_id"))
            if example_id in seen_example_ids:
                errors.append(f"example_id duplicado: {example_id}")
            seen_example_ids.add(example_id)

            game_id = str(row.get("game_id"))
            games.add(game_id)
            if game_id not in raw_colors:
                errors.append(f"partida não encontrada na fonte: {game_id}")
            elif raw_colors[game_id] != row.get("color"):
                errors.append(f"cor do jogador-alvo incorreta: {example_id}")

            board = chess.Board(row["fen"])
            expected_turn = chess.WHITE if row.get("color") == "white" else chess.BLACK
            if board.turn != expected_turn:
                errors.append(f"turno não pertence ao jogador-alvo: {example_id}")
            try:
                move = chess.Move.from_uci(row["move_uci"])
            except ValueError:
                errors.append(f"UCI inválido: {example_id}")
            else:
                if move not in board.legal_moves:
                    errors.append(f"rótulo ilegal: {example_id}")

            key = str(row.get("position_key"))
            positions.add(key)
            if key in prior_positions:
                errors.append(f"posição atravessa splits: {example_id}")
            dates.append(str(row.get("date")))

        if games & prior_games:
            errors.append(f"partida atravessa splits: {split}")
        first_date = min(dates) if dates else None
        last_date = max(dates) if dates else None
        if previous_last_date and first_date and first_date < previous_last_date:
            errors.append(f"ordem cronológica violada em {split}")
        previous_last_date = last_date or previous_last_date
        prior_positions.update(positions)
        prior_games.update(games)
        result["splits"][split] = {
            "examples": rows,
            "games_with_examples": len(games),
            "unique_positions": len(positions),
            "first_date": first_date,
            "last_date": last_date,
        }

    result["valid"] = not errors
    result["checked_examples"] = len(seen_example_ids)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida o dataset do botChess")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("partidas"))
    parser.add_argument("--username", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = validate_dataset(args.data_dir, args.raw_dir, args.username)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
