from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
import urllib.error

import chess

from botchess.data import examples_for_game, load_games, split_games, write_dataset
from botchess.download import fetch_json, main as download_main
from botchess.features import FEATURE_COUNT, candidate_features
from botchess.solutions.frequency import train as train_frequency
from botchess.solutions.registry import SOLUTION_NAMES, get_solution
from botchess.training import render_report
from botchess.uci import parse_position
from botchess.validation import validate_dataset


def raw_game(
    game_id: str, date: str, player_color: str, moves: str, eco: str = "C20"
) -> dict:
    white = "ChefeJH" if player_color == "white" else "Opponent"
    black = "ChefeJH" if player_color == "black" else "Opponent"
    pgn = (
        f'[Event "Test"]\n[Date "{date}"]\n[White "{white}"]\n'
        f'[Black "{black}"]\n[Result "*"]\n[ECO "{eco}"]\n\n{moves} *\n'
    )
    parsed_date = datetime.strptime(date, "%Y.%m.%d").replace(tzinfo=timezone.utc)
    return {
        "uuid": game_id,
        "pgn": pgn,
        "rules": "chess",
        "time_class": "rapid",
        "end_time": int(parsed_date.timestamp()),
        "white": {"username": white, "rating": 1200},
        "black": {"username": black, "rating": 1250},
    }


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw_dir = self.root / "raw"
        self.data_dir = self.root / "processed"
        self.raw_dir.mkdir()
        games = [
            raw_game("g1", "2024.01.01", "white", "1. e4 e5 2. Nf3 Nc6"),
            raw_game("g2", "2024.01.02", "black", "1. d4 d5 2. c4 e6"),
            raw_game("g3", "2024.01.03", "white", "1. e4 c5 2. Nf3 d6"),
            raw_game("g4", "2024.01.04", "black", "1. c4 e5 2. Nc3 Nf6"),
            raw_game("g5", "2024.01.05", "white", "1. d4 Nf6 2. c4 e6"),
            raw_game("g6", "2024.01.06", "black", "1. e4 c5 2. Nf3 d6"),
        ]
        (self.raw_dir / "2024-01.json").write_text(
            json.dumps({"games": games}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_only_player_moves_become_labels(self) -> None:
        games, stats = load_games(self.raw_dir, "ChefeJH", min_fullmoves=1)
        self.assertEqual(stats["games_accepted"], 6)
        white_moves = [row["move_uci"] for row in examples_for_game(games[0])]
        black_moves = [row["move_uci"] for row in examples_for_game(games[1])]
        self.assertEqual(white_moves, ["e2e4", "g1f3"])
        self.assertEqual(black_moves, ["d7d5", "e7e6"])

    def test_dataset_is_chronological_disjoint_and_valid(self) -> None:
        games, _ = load_games(self.raw_dir, "ChefeJH", min_fullmoves=1)
        splits = split_games(games, (0.5, 0.25, 0.25))
        summary = write_dataset(splits, self.data_dir)
        (self.data_dir / "manifest.json").write_text(
            json.dumps({"splits": summary["splits"]}), encoding="utf-8"
        )
        result = validate_dataset(self.data_dir, self.raw_dir, "ChefeJH")
        self.assertTrue(result["valid"], result["errors"])

    def test_candidate_features_and_frequency_prediction_are_legal(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        vector = candidate_features(board, move, {"phase": "opening"})
        self.assertEqual(vector.shape, (FEATURE_COUNT,))
        with self.assertRaises(ValueError):
            candidate_features(board, chess.Move.from_uci("e2e5"))
        model = train_frequency([{"move_uci": "e2e4", "player_rating": 1200}])
        prediction = model.choose_move(board)
        self.assertIn(prediction, board.legal_moves)

    def test_solution_registry_describes_all_trainable_approaches(self) -> None:
        self.assertEqual(
            SOLUTION_NAMES,
            ("frequency", "logistic", "hist_gradient_boosting"),
        )
        self.assertFalse(get_solution("frequency").candidate_model)
        self.assertTrue(get_solution("logistic").candidate_model)
        self.assertTrue(get_solution("hist_gradient_boosting").candidate_model)
        with self.assertRaisesRegex(ValueError, "Solução desconhecida"):
            get_solution("maia")

    def test_generated_report_links_to_metrics_in_its_own_directory(self) -> None:
        report = {
            "models": {
                "dummy": {
                    "validation": {
                        "top1_accuracy": 1.0,
                        "top3_accuracy": 1.0,
                        "mrr": 1.0,
                    },
                    "test": {
                        "top1_accuracy": 1.0,
                        "top3_accuracy": 1.0,
                        "mrr": 1.0,
                        "milliseconds_per_position": 1.0,
                    },
                }
            },
            "selected_model": "dummy",
            "final_model": {
                "test": {
                    "top1_accuracy": 1.0,
                    "top3_accuracy": 1.0,
                    "mrr": 1.0,
                }
            },
        }
        rendered = render_report(report)
        self.assertIn("[`latest_metrics.json`](latest_metrics.json)", rendered)
        self.assertNotIn("reports/latest_metrics.json", rendered)

    def test_uci_position_parser(self) -> None:
        board = parse_position("position startpos moves e2e4 e7e5 g1f3")
        self.assertEqual(board.peek(), chess.Move.from_uci("g1f3"))
        fen_board = parse_position(f"position fen {chess.STARTING_FEN} moves d2d4")
        self.assertEqual(fen_board.peek(), chess.Move.from_uci("d2d4"))

    def test_download_retries_temporary_network_failure(self) -> None:
        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with (
            mock.patch(
                "botchess.download.urllib.request.urlopen",
                side_effect=[
                    urllib.error.URLError("temporário"),
                    Response(b'{"games": []}'),
                ],
            ) as urlopen,
            mock.patch("botchess.download.time.sleep") as sleep,
        ):
            payload = fetch_json("https://example.test/games", timeout=1, retries=1)
        self.assertEqual(payload, {"games": []})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_download_cli_requires_authorization_confirmation(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            download_main(["--username", "ExamplePlayer"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
