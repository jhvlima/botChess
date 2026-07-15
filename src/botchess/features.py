"""Representação compacta de pares (posição, lance candidato)."""

from __future__ import annotations

from typing import Any

import chess
import numpy as np


PIECE_TYPES = tuple(range(chess.PAWN, chess.KING + 1))


def _build_feature_names() -> tuple[str, ...]:
    names = [f"board_square_{square}" for square in chess.SQUARES]
    names.extend(
        [
            "own_pawns",
            "own_knights",
            "own_bishops",
            "own_rooks",
            "own_queens",
            "own_kings",
            "opponent_pawns",
            "opponent_knights",
            "opponent_bishops",
            "opponent_rooks",
            "opponent_queens",
            "opponent_kings",
            "material_balance",
            "own_kingside_castling",
            "own_queenside_castling",
            "opponent_kingside_castling",
            "opponent_queenside_castling",
            "fullmove_number",
            "halfmove_clock",
            "player_rating",
            "opponent_rating",
            "rating_difference",
            "phase_opening",
            "phase_middlegame",
            "phase_endgame",
            "legal_move_count",
            "from_file",
            "from_rank",
            "to_file",
            "to_rank",
            "delta_file",
            "delta_rank",
        ]
    )
    names.extend(f"moving_piece_{chess.piece_name(piece)}" for piece in PIECE_TYPES)
    names.extend(f"captured_piece_{chess.piece_name(piece)}" for piece in PIECE_TYPES)
    names.extend(
        [
            "is_capture",
            "is_castling",
            "is_en_passant",
            "gives_check",
            "is_promotion",
            "promotion_knight",
            "promotion_bishop",
            "promotion_rook",
            "promotion_queen",
            "destination_centrality",
            "forward_progress",
        ]
    )
    return tuple(names)


FEATURE_NAMES = _build_feature_names()
FEATURE_COUNT = len(FEATURE_NAMES)


def _oriented_square(square: chess.Square, perspective: chess.Color) -> chess.Square:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def _rating(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 3000.0)) / 3000.0
    except (TypeError, ValueError):
        return 0.4


def candidate_features(
    board: chess.Board, move: chess.Move, context: dict[str, Any] | None = None
) -> np.ndarray:
    """Codifica apenas informações disponíveis antes da jogada."""
    if move not in board.legal_moves:
        raise ValueError(f"Lance ilegal para a posição: {move.uci()}")
    context = context or {}
    perspective = board.turn
    values: list[float] = []

    # Ocupação vista da perspectiva do jogador da vez. Peças próprias são
    # positivas e peças adversárias negativas.
    for oriented in chess.SQUARES:
        original = oriented if perspective == chess.WHITE else chess.square_mirror(oriented)
        piece = board.piece_at(original)
        if piece is None:
            values.append(0.0)
        else:
            sign = 1.0 if piece.color == perspective else -1.0
            values.append(sign * piece.piece_type / 6.0)

    for color in (perspective, not perspective):
        for piece_type in PIECE_TYPES:
            values.append(len(board.pieces(piece_type, color)) / 8.0)

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    own_material = sum(
        len(board.pieces(piece_type, perspective)) * value
        for piece_type, value in piece_values.items()
    )
    opponent_material = sum(
        len(board.pieces(piece_type, not perspective)) * value
        for piece_type, value in piece_values.items()
    )
    values.append((own_material - opponent_material) / 39.0)
    values.extend(
        [
            float(board.has_kingside_castling_rights(perspective)),
            float(board.has_queenside_castling_rights(perspective)),
            float(board.has_kingside_castling_rights(not perspective)),
            float(board.has_queenside_castling_rights(not perspective)),
            min(board.fullmove_number, 100) / 100.0,
            min(board.halfmove_clock, 100) / 100.0,
        ]
    )
    player_rating = _rating(context.get("player_rating"))
    opponent_rating = _rating(context.get("opponent_rating"))
    values.extend([player_rating, opponent_rating, player_rating - opponent_rating])

    phase = str(context.get("phase", "middlegame"))
    values.extend(float(phase == name) for name in ("opening", "middlegame", "endgame"))
    legal_count = board.legal_moves.count()
    values.append(min(legal_count, 60) / 60.0)

    from_square = _oriented_square(move.from_square, perspective)
    to_square = _oriented_square(move.to_square, perspective)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)
    values.extend(
        [
            from_file / 7.0,
            from_rank / 7.0,
            to_file / 7.0,
            to_rank / 7.0,
            (to_file - from_file) / 7.0,
            (to_rank - from_rank) / 7.0,
        ]
    )

    moving_piece = board.piece_type_at(move.from_square)
    captured_piece = board.piece_type_at(move.to_square)
    if board.is_en_passant(move):
        captured_piece = chess.PAWN
    values.extend(float(moving_piece == piece_type) for piece_type in PIECE_TYPES)
    values.extend(float(captured_piece == piece_type) for piece_type in PIECE_TYPES)
    values.extend(
        [
            float(board.is_capture(move)),
            float(board.is_castling(move)),
            float(board.is_en_passant(move)),
            float(board.gives_check(move)),
            float(move.promotion is not None),
            float(move.promotion == chess.KNIGHT),
            float(move.promotion == chess.BISHOP),
            float(move.promotion == chess.ROOK),
            float(move.promotion == chess.QUEEN),
            (3.5 - abs(to_file - 3.5) + 3.5 - abs(to_rank - 3.5)) / 7.0,
            (to_rank - from_rank) / 7.0,
        ]
    )

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,):
        raise AssertionError(f"Esperadas {FEATURE_COUNT} features, obtidas {result.shape}")
    return result
