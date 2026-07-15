from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess

from botchess.demo import create_demo_model, main
from botchess.solutions.base import load_model


class DemoModelTests(unittest.TestCase):
    def test_demo_model_is_synthetic_and_returns_a_legal_move(self) -> None:
        model = create_demo_model()

        self.assertEqual(model.name, "synthetic_demo")
        self.assertEqual(model.metadata["data_source"], "synthetic_curated")
        self.assertEqual(model.choose_move(chess.Board()), chess.Move.from_uci("e2e4"))

    def test_cli_writes_a_loadable_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo.joblib"
            main(["--output", str(output)])

            loaded = load_model(output)
            self.assertEqual(loaded.name, "synthetic_demo")
            self.assertEqual(loaded.metadata["intended_use"], "local_interface_demo")


if __name__ == "__main__":
    unittest.main()
