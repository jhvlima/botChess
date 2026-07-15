"""Preparação cronológica de partidas no esquema JSON esperado.

O contrato central deste módulo é simples: cada linha JSONL representa a
posição *antes* de uma jogada do jogador-alvo e tem como rótulo apenas a jogada
feita por ele. Jogadas dos adversários são usadas para avançar o tabuleiro,
nunca como rótulos.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import chess
import chess.pgn


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    date: str
    sort_timestamp: int
    pgn: str
    player_color: str
    player_rating: int | None
    opponent_rating: int | None
    time_class: str
    eco: str


def normalized_position_key(board: chess.Board) -> str:
    """FEN sem contadores, mantendo tudo que altera os lances legais."""
    return " ".join(board.fen(en_passant="legal").split()[:4])


def classify_phase(board: chess.Board) -> str:
    """Classificação pequena e interpretável para relatórios."""
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    non_pawn_non_king = sum(
        len(board.pieces(piece_type, color))
        for color in chess.COLORS
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if board.fullmove_number <= 10 and queens == 2:
        return "opening"
    if queens == 0 and non_pawn_non_king <= 6:
        return "endgame"
    return "middlegame"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _game_date(raw: dict[str, Any], game: chess.pgn.Game) -> tuple[str, int]:
    end_time = _safe_int(raw.get("end_time"))
    if end_time is not None:
        dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
        return dt.date().isoformat(), end_time

    value = game.headers.get("UTCDate") or game.headers.get("Date") or "0000.00.00"
    cleaned = value.replace(".", "-").replace("?", "0")
    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.date().isoformat(), int(dt.timestamp())
    except ValueError:
        return cleaned, 0


def _player_side(raw: dict[str, Any], username: str) -> tuple[chess.Color, dict, dict] | None:
    target = username.casefold()
    white = raw.get("white") or {}
    black = raw.get("black") or {}
    if str(white.get("username", "")).casefold() == target:
        return chess.WHITE, white, black
    if str(black.get("username", "")).casefold() == target:
        return chess.BLACK, black, white
    return None


def load_games(
    raw_dir: Path,
    username: str,
    time_class: str = "rapid",
    min_fullmoves: int = 15,
) -> tuple[list[GameRecord], Counter[str]]:
    """Lê os JSONs mensais e devolve partidas válidas, sem duplicatas."""
    stats: Counter[str] = Counter()
    records: dict[str, GameRecord] = {}
    seen_pgn_hashes: set[str] = set()

    for path in sorted(raw_dir.glob("*.json")):
        stats["source_files"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["invalid_json_files"] += 1
            continue

        games = payload.get("games") if isinstance(payload, dict) else None
        if not isinstance(games, list):
            stats["invalid_schema_files"] += 1
            continue

        for raw in games:
            stats["games_seen"] += 1
            if raw.get("rules") != "chess":
                stats["filtered_rules"] += 1
                continue
            if time_class != "all" and raw.get("time_class") != time_class:
                stats["filtered_time_class"] += 1
                continue

            side = _player_side(raw, username)
            if side is None:
                stats["filtered_player_missing"] += 1
                continue
            color, player, opponent = side

            pgn_text = raw.get("pgn")
            if not isinstance(pgn_text, str) or not pgn_text.strip():
                stats["filtered_missing_pgn"] += 1
                continue
            try:
                parsed = chess.pgn.read_game(io.StringIO(pgn_text))
            except (ValueError, UnicodeError):
                parsed = None
            if parsed is None or parsed.errors:
                stats["filtered_invalid_pgn"] += 1
                continue
            if parsed.end().ply() < min_fullmoves * 2:
                stats["filtered_short_game"] += 1
                continue

            date, timestamp = _game_date(raw, parsed)
            digest = hashlib.sha256(pgn_text.encode("utf-8")).hexdigest()[:20]
            if digest in seen_pgn_hashes:
                stats["duplicate_game_content"] += 1
                continue
            game_id = str(raw.get("uuid") or raw.get("url") or digest)
            record = GameRecord(
                game_id=game_id,
                date=date,
                sort_timestamp=timestamp,
                pgn=pgn_text,
                player_color="white" if color == chess.WHITE else "black",
                player_rating=_safe_int(player.get("rating")),
                opponent_rating=_safe_int(opponent.get("rating")),
                time_class=str(raw.get("time_class", "unknown")),
                eco=str(parsed.headers.get("ECO") or "unknown"),
            )
            if game_id in records:
                stats["duplicate_games"] += 1
            else:
                records[game_id] = record
                seen_pgn_hashes.add(digest)
                stats["games_accepted"] += 1

    ordered = sorted(records.values(), key=lambda item: (item.sort_timestamp, item.game_id))
    return ordered, stats


def split_games(
    games: list[GameRecord], ratios: tuple[float, float, float]
) -> dict[str, list[GameRecord]]:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("As proporções de treino, validação e teste devem ser positivas")
    total_ratio = sum(ratios)
    normalized = tuple(value / total_ratio for value in ratios)
    n_games = len(games)
    train_end = int(n_games * normalized[0])
    validation_end = train_end + int(n_games * normalized[1])
    if n_games >= 3:
        train_end = min(max(train_end, 1), n_games - 2)
        validation_end = min(max(validation_end, train_end + 1), n_games - 1)
    return {
        "train": games[:train_end],
        "validation": games[train_end:validation_end],
        "test": games[validation_end:],
    }


def examples_for_game(record: GameRecord) -> Iterator[dict[str, Any]]:
    game = chess.pgn.read_game(io.StringIO(record.pgn))
    if game is None:
        return
    player_color = chess.WHITE if record.player_color == "white" else chess.BLACK
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if board.turn == player_color:
            yield {
                "example_id": f"{record.game_id}:{ply}",
                "game_id": record.game_id,
                "date": record.date,
                "fen": board.fen(en_passant="legal"),
                "position_key": normalized_position_key(board),
                "move_uci": move.uci(),
                "move_san": board.san(move),
                "color": record.player_color,
                "ply": ply,
                "fullmove_number": board.fullmove_number,
                "phase": classify_phase(board),
                "eco": record.eco,
                "player_rating": record.player_rating,
                "opponent_rating": record.opponent_rating,
                "time_class": record.time_class,
            }
        board.push(move)


def write_dataset(
    split_records: dict[str, list[GameRecord]], output_dir: Path
) -> dict[str, Any]:
    """Escreve JSONL e remove posições que atravessariam os cortes temporais."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_from_earlier_splits: set[str] = set()
    summary: dict[str, Any] = {"splits": {}}

    for split in SPLITS:
        rows: list[dict[str, Any]] = []
        all_keys_in_split: set[str] = set()
        dropped = 0
        records = split_records[split]
        for record in records:
            for row in examples_for_game(record):
                key = row["position_key"]
                all_keys_in_split.add(key)
                if key in seen_from_earlier_splits:
                    dropped += 1
                    continue
                row["split"] = split
                rows.append(row)

        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        dates = [record.date for record in records]
        summary["splits"][split] = {
            "games": len(records),
            "examples": len(rows),
            "positions_dropped_cross_split": dropped,
            "first_date": min(dates) if dates else None,
            "last_date": max(dates) if dates else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        seen_from_earlier_splits.update(all_keys_in_split)

    return summary


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}:{line_number}") from exc


def parse_ratios(value: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use três números separados por vírgulas") from exc
    if len(values) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Use três proporções positivas")
    return values  # type: ignore[return-value]


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    games, counters = load_games(
        args.raw_dir,
        username=args.username,
        time_class=args.time_class,
        min_fullmoves=args.min_fullmoves,
    )
    if len(games) < 3:
        raise RuntimeError("São necessárias pelo menos três partidas válidas")
    splits = split_games(games, args.split_ratios)
    summary = write_dataset(splits, args.output_dir)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "player": args.username,
        "filters": {
            "rules": "chess",
            "time_class": args.time_class,
            "min_fullmoves": args.min_fullmoves,
        },
        "split_strategy": {
            "type": "chronological_by_game",
            "ratios": list(args.split_ratios),
            "cross_split_position_policy": "drop_from_later_split",
        },
        "counters": dict(sorted(counters.items())),
        **summary,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converte JSONs mensais em exemplos do jogador-alvo"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("partidas"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--username", required=True)
    parser.add_argument("--time-class", default="rapid", help="Use 'all' para não filtrar")
    parser.add_argument("--min-fullmoves", type=int, default=15)
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.70, 0.15, 0.15))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = prepare_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
