"""Políticas de escolha usadas ao jogar contra o modelo.

O modo imitador usa diretamente o ranking aprendido. O modo coerente mantém
esse ranking como preferência, mas descarta erros táticos rasos quando existe
uma alternativa segura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from .data import classify_phase
from .solutions.base import ModelBundle


PLAY_MODES = ("imitator", "coherent")

PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE_CP = 100_000

# O filtro fica mais exigente conforme a partida avança. Na abertura, apenas
# as proteções contra mate imediato são aplicadas para não apagar o repertório
# pessoal. No meio-jogo, a margem barra a perda limpa de uma peça menor; no
# final, admite a perda de até um peão, mas não de dois.
MATERIAL_LOSS_TOLERANCE_CP: dict[str, int | None] = {
    "opening": None,
    "middlegame": 300,
    "endgame": 150,
}


@dataclass(frozen=True)
class MoveSelection:
    """Resultado da política, incluindo a posição no ranking pessoal."""

    move: chess.Move
    style_score: float
    style_rank: int
    reason: str
    reply_floor_cp: int | None = None


@dataclass(frozen=True)
class _Candidate:
    move: chess.Move
    style_score: float
    style_rank: int
    immediate_mate: bool
    immediate_stalemate: bool
    allows_opponent_mate: bool
    reply_floor_cp: int


def material_score_cp(board: chess.Board, perspective: chess.Color) -> int:
    """Balanço material em centipawns visto por ``perspective``."""

    own = sum(
        len(board.pieces(piece_type, perspective)) * value
        for piece_type, value in PIECE_VALUES_CP.items()
    )
    opponent = sum(
        len(board.pieces(piece_type, not perspective)) * value
        for piece_type, value in PIECE_VALUES_CP.items()
    )
    return own - opponent


def select_move(
    model: ModelBundle,
    board: chess.Board,
    context: dict[str, Any] | None = None,
    mode: str = "imitator",
) -> MoveSelection | None:
    """Seleciona um lance sem alterar ``board``.

    Em ``coherent``, a ordem do modelo continua sendo o desempate principal.
    A análise olha somente o lance candidato, uma resposta do adversário e uma
    recaptura direta quando houver; ela é uma proteção heurística, não uma
    busca completa de engine.
    """

    if mode not in PLAY_MODES:
        raise ValueError(f"Modo de jogo desconhecido: {mode}")

    ranking = model.rank(board, context)
    if not ranking:
        return None
    legal_moves = set(board.legal_moves)
    ranked_moves = [move for move, _score in ranking]
    if len(ranked_moves) != len(set(ranked_moves)):
        raise ValueError("O ranking do modelo contém lances duplicados.")
    if set(ranked_moves) != legal_moves:
        raise ValueError("O ranking do modelo deve conter todos os lances legais.")

    if mode == "imitator":
        move, score = ranking[0]
        return MoveSelection(
            move=move,
            style_score=score,
            style_rank=1,
            reason="imitation",
        )

    perspective = board.turn
    phase = str((context or {}).get("phase") or classify_phase(board))
    candidates = _analyze_candidates(board, ranking, perspective)

    mating = [candidate for candidate in candidates if candidate.immediate_mate]
    if mating:
        chosen = mating[0]
        return _selection(chosen, "mate_in_one")

    pool = candidates
    reason = "imitation"

    safe_from_mate = [
        candidate for candidate in pool if not candidate.allows_opponent_mate
    ]
    if safe_from_mate:
        if pool[0].allows_opponent_mate:
            reason = "avoided_mate"
        pool = safe_from_mate

    current_material = material_score_cp(board, perspective)
    if current_material > 0:
        continuing = [
            candidate for candidate in pool if not candidate.immediate_stalemate
        ]
        if continuing:
            if pool[0].immediate_stalemate and reason == "imitation":
                reason = "avoided_stalemate"
            pool = continuing

    tolerance = MATERIAL_LOSS_TOLERANCE_CP.get(phase, 300)
    if tolerance is not None:
        material_floor = current_material - tolerance
        material_safe = [
            candidate
            for candidate in pool
            if candidate.reply_floor_cp >= material_floor
        ]
        if material_safe:
            if (
                pool[0].reply_floor_cp < material_floor
                and reason == "imitation"
            ):
                reason = "material_guard"
            pool = material_safe
        else:
            # Se toda alternativa perde mais material que a margem, evita a
            # pior perda possível sem abandonar o ranking como desempate.
            best_floor = max(candidate.reply_floor_cp for candidate in pool)
            best_defenses = [
                candidate
                for candidate in pool
                if candidate.reply_floor_cp == best_floor
            ]
            if pool[0].reply_floor_cp != best_floor and reason == "imitation":
                reason = "material_guard"
            pool = best_defenses

    return _selection(pool[0], reason)


def _selection(candidate: _Candidate, reason: str) -> MoveSelection:
    return MoveSelection(
        move=candidate.move,
        style_score=candidate.style_score,
        style_rank=candidate.style_rank,
        reason=reason,
        reply_floor_cp=candidate.reply_floor_cp,
    )


def _analyze_candidates(
    board: chess.Board,
    ranking: list[tuple[chess.Move, float]],
    perspective: chess.Color,
) -> list[_Candidate]:
    candidates = []
    for style_rank, (move, style_score) in enumerate(ranking, start=1):
        board.push(move)
        try:
            immediate_mate = board.is_checkmate()
            immediate_stalemate = board.is_stalemate()
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                reply_floor = _outcome_score(outcome, perspective)
                allows_opponent_mate = False
            else:
                reply_floor, allows_opponent_mate = _worst_reply(
                    board, perspective
                )
        finally:
            board.pop()
        candidates.append(
            _Candidate(
                move=move,
                style_score=style_score,
                style_rank=style_rank,
                immediate_mate=immediate_mate,
                immediate_stalemate=immediate_stalemate,
                allows_opponent_mate=allows_opponent_mate,
                reply_floor_cp=reply_floor,
            )
        )
    return candidates


def _worst_reply(
    board: chess.Board, perspective: chess.Color
) -> tuple[int, bool]:
    worst_score = MATE_SCORE_CP
    allows_opponent_mate = False
    replies = tuple(board.legal_moves)
    for reply in replies:
        reply_is_capture = board.is_capture(reply)
        reply_square = reply.to_square
        board.push(reply)
        try:
            outcome = board.outcome(claim_draw=False)
            if outcome is not None:
                score = _outcome_score(outcome, perspective)
                if outcome.winner == (not perspective) and board.is_checkmate():
                    allows_opponent_mate = True
            else:
                score = material_score_cp(board, perspective)
                if reply_is_capture:
                    recapture_score = _best_recapture_score(
                        board,
                        perspective,
                        reply_square,
                    )
                    if recapture_score is not None:
                        score = max(score, recapture_score)
        finally:
            board.pop()
        worst_score = min(worst_score, score)
    return worst_score, allows_opponent_mate


def _best_recapture_score(
    board: chess.Board,
    perspective: chess.Color,
    target_square: chess.Square,
) -> int | None:
    """Avalia somente recapturas legais na casa da última captura."""

    best_score: int | None = None
    recaptures = tuple(
        move
        for move in board.legal_moves
        if move.to_square == target_square and board.is_capture(move)
    )
    for recapture in recaptures:
        board.push(recapture)
        try:
            outcome = board.outcome(claim_draw=False)
            score = (
                _outcome_score(outcome, perspective)
                if outcome is not None
                else material_score_cp(board, perspective)
            )
        finally:
            board.pop()
        best_score = score if best_score is None else max(best_score, score)
    return best_score


def _outcome_score(outcome: chess.Outcome, perspective: chess.Color) -> int:
    if outcome.winner is None:
        return 0
    return MATE_SCORE_CP if outcome.winner == perspective else -MATE_SCORE_CP
