from __future__ import annotations

import unittest
from unittest import mock

import chess

from botchess.play_modes import select_move
from botchess.solutions.base import ModelBundle


class PlayModeTests(unittest.TestCase):
    @staticmethod
    def _select(
        fen: str,
        counts: dict[str, int],
        mode: str,
        phase: str,
    ):
        board = chess.Board(fen)
        model = ModelBundle(
            name="fixture",
            kind="frequency",
            move_counts=counts,
        )
        original_fen = board.fen()
        selection = select_move(model, board, {"phase": phase}, mode)
        if selection is None:
            raise AssertionError("A fixture deveria possuir lances legais.")
        if board.fen() != original_fen:
            raise AssertionError("A seleção alterou o tabuleiro recebido.")
        return selection

    def test_imitator_keeps_the_first_move_from_the_personal_ranking(self) -> None:
        selection = self._select(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            {"a7a6": 100, "d8h4": 90},
            "imitator",
            "opening",
        )
        self.assertEqual(selection.move.uci(), "a7a6")
        self.assertEqual(selection.style_rank, 1)
        self.assertEqual(selection.reason, "imitation")

    def test_coherent_mode_finds_an_immediate_mate_outside_top_one(self) -> None:
        selection = self._select(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            {"a7a6": 100, "d8h4": 90},
            "coherent",
            "opening",
        )
        self.assertEqual(selection.move.uci(), "d8h4")
        self.assertEqual(selection.style_rank, 2)
        self.assertEqual(selection.reason, "mate_in_one")

    def test_coherent_mode_avoids_allowing_mate_in_one(self) -> None:
        selection = self._select(
            "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2",
            {"g2g4": 100, "g2g3": 90},
            "coherent",
            "opening",
        )
        self.assertEqual(selection.move.uci(), "g2g3")
        self.assertEqual(selection.style_rank, 2)
        self.assertEqual(selection.reason, "avoided_mate")

    def test_coherent_mode_avoids_stalemate_while_ahead(self) -> None:
        selection = self._select(
            "8/8/8/8/8/8/8/k1KR4 w - - 0 1",
            {"d1d2": 100, "d1d8": 90},
            "coherent",
            "endgame",
        )
        self.assertEqual(selection.move.uci(), "d1d8")
        self.assertEqual(selection.style_rank, 2)
        self.assertEqual(selection.reason, "avoided_stalemate")

    def test_coherent_endgame_rejects_an_avoidable_rook_loss(self) -> None:
        selection = self._select(
            "4k3/6b1/8/8/3R4/8/8/4K3 w - - 0 20",
            {"e1f1": 100, "d4d1": 90},
            "coherent",
            "endgame",
        )
        self.assertEqual(selection.move.uci(), "d4d1")
        self.assertEqual(selection.style_rank, 2)
        self.assertEqual(selection.reason, "material_guard")
        self.assertEqual(selection.reply_floor_cp, 170)

    def test_coherent_mode_does_not_force_a_gain_when_style_move_is_safe(self) -> None:
        selection = self._select(
            "4k3/p7/8/8/8/8/8/R3K3 w - - 0 20",
            {"e1f1": 100, "a1a7": 90},
            "coherent",
            "endgame",
        )
        self.assertEqual(selection.move.uci(), "e1f1")
        self.assertEqual(selection.style_rank, 1)
        self.assertEqual(selection.reason, "imitation")

    def test_coherent_mode_sees_an_immediate_recapture(self) -> None:
        selection = self._select(
            "3rk3/8/8/8/8/8/3R4/4K3 w - - 0 20",
            {"d2d1": 100, "d2a2": 90},
            "coherent",
            "endgame",
        )
        self.assertEqual(selection.move.uci(), "d2d1")
        self.assertEqual(selection.style_rank, 1)
        self.assertEqual(selection.reply_floor_cp, 0)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Modo de jogo desconhecido"):
            self._select(
                chess.STARTING_FEN,
                {"e2e4": 1},
                "unknown",
                "opening",
            )

    def test_coherent_selection_preserves_existing_move_history(self) -> None:
        board = chess.Board()
        for move_uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
            board.push_uci(move_uci)
        model = ModelBundle(name="fixture", kind="frequency")
        original_fen = board.fen()
        original_stack = tuple(board.move_stack)
        original_claim = board.can_claim_threefold_repetition()

        selection = select_move(
            model,
            board,
            {"phase": "opening"},
            "coherent",
        )

        self.assertIsNotNone(selection)
        self.assertEqual(board.fen(), original_fen)
        self.assertEqual(tuple(board.move_stack), original_stack)
        self.assertEqual(board.can_claim_threefold_repetition(), original_claim)

    def test_incomplete_model_ranking_is_rejected(self) -> None:
        board = chess.Board()
        model = ModelBundle(name="fixture", kind="frequency")
        incomplete = [(chess.Move.from_uci("e2e4"), 1.0)]
        with (
            mock.patch.object(model, "rank", return_value=incomplete),
            self.assertRaisesRegex(ValueError, "todos os lances legais"),
        ):
            select_move(model, board, {"phase": "opening"}, "coherent")


if __name__ == "__main__":
    unittest.main()
