"""Estado e regras das partidas usadas pela interface web local."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any

import chess

from .data import classify_phase
from .play_modes import PLAY_MODES, select_move
from .solutions.base import ModelBundle


COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}
COLOR_VALUES = {name: color for color, name in COLOR_NAMES.items()}
TERMINATION_NAMES = {
    "CHECKMATE": "xeque-mate",
    "STALEMATE": "afogamento",
    "INSUFFICIENT_MATERIAL": "material insuficiente",
    "SEVENTYFIVE_MOVES": "regra dos 75 lances",
    "FIVEFOLD_REPETITION": "repetição quíntupla",
    "FIFTY_MOVES": "regra dos 50 lances",
    "THREEFOLD_REPETITION": "repetição tripla",
}


class GameError(Exception):
    """Erro esperado que pode ser apresentado como resposta JSON."""

    status_code = 400
    code = "invalid_request"


class GameNotFound(GameError):
    status_code = 404
    code = "game_not_found"


class InvalidMove(GameError):
    status_code = 409
    code = "invalid_move"


class BotMoveError(GameError):
    status_code = 500
    code = "bot_failure"


def parse_human_color(value: Any) -> chess.Color:
    if not isinstance(value, str) or value not in COLOR_VALUES:
        raise GameError("A cor deve ser 'white' ou 'black'.")
    return COLOR_VALUES[value]


def parse_rating(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise GameError("O rating deve ser um número inteiro.")
    if not 100 <= value <= 4000:
        raise GameError("O rating deve estar entre 100 e 4000.")
    return value


def parse_play_mode(value: Any) -> str:
    if value is None:
        return "imitator"
    if not isinstance(value, str) or value not in PLAY_MODES:
        raise GameError("O modo deve ser 'imitator' ou 'coherent'.")
    return value


@dataclass
class GameSession:
    game_id: str
    board: chess.Board
    human_color: chess.Color
    human_rating: int
    bot_rating: int
    mode: str
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    last_bot_move: dict[str, Any] | None = None


class GameStore:
    """Armazena partidas locais em memória e aplica as respostas do modelo."""

    def __init__(
        self,
        model: ModelBundle,
        bot_rating: int | None = None,
        max_sessions: int = 100,
        ttl_seconds: int = 12 * 60 * 60,
    ) -> None:
        self.model = model
        default_rating = int(model.metadata.get("default_rating", 1200))
        self.bot_rating = parse_rating(
            bot_rating,
            default_rating,
        )
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, GameSession] = OrderedDict()
        self._lock = threading.RLock()

    def create(
        self,
        human_color: Any,
        human_rating: Any = None,
        mode: Any = None,
    ) -> dict[str, Any]:
        color = parse_human_color(human_color)
        rating = parse_rating(human_rating, self.bot_rating)
        play_mode = parse_play_mode(mode)
        with self._lock:
            self._remove_expired()
            while len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
            session = GameSession(
                game_id=uuid.uuid4().hex,
                board=chess.Board(),
                human_color=color,
                human_rating=rating,
                bot_rating=self.bot_rating,
                mode=play_mode,
            )
            if session.board.turn != session.human_color:
                session.last_bot_move = self._play_bot(session)
            self._sessions[session.game_id] = session
            return self._serialize(session)

    def get(self, game_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require(game_id)
            return self._serialize(session)

    def play(self, game_id: str, move_uci: Any) -> dict[str, Any]:
        if not isinstance(move_uci, str):
            raise GameError("O lance deve ser informado em formato UCI.")
        with self._lock:
            session = self._require(game_id)
            board = session.board
            if board.is_game_over():
                raise InvalidMove("A partida já terminou.")
            if board.turn != session.human_color:
                raise InvalidMove("Ainda não é a sua vez.")
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError as exc:
                raise InvalidMove("Formato de lance inválido.") from exc
            if move not in board.legal_moves:
                raise InvalidMove("Esse lance não é legal na posição atual.")

            human_move = self._move_details(session, move, "human")
            original_ply = len(board.move_stack)
            try:
                board.push(move)
                bot_move = None
                if not board.is_game_over():
                    bot_move = self._play_bot(session)
            except Exception:
                while len(board.move_stack) > original_ply:
                    board.pop()
                raise
            if bot_move is not None:
                session.last_bot_move = bot_move
            state = self._serialize(session)
            state["human_move"] = human_move
            state["bot_move"] = bot_move
            return state

    def delete(self, game_id: str) -> None:
        with self._lock:
            if self._sessions.pop(game_id, None) is None:
                raise GameNotFound("Partida não encontrada.")

    def _require(self, game_id: str) -> GameSession:
        self._remove_expired()
        session = self._sessions.get(game_id)
        if session is None:
            raise GameNotFound("Partida não encontrada ou expirada.")
        session.last_access = time.monotonic()
        self._sessions.move_to_end(game_id)
        return session

    def _remove_expired(self) -> None:
        cutoff = time.monotonic() - self.ttl_seconds
        expired = [
            game_id
            for game_id, session in self._sessions.items()
            if session.last_access < cutoff
        ]
        for game_id in expired:
            self._sessions.pop(game_id, None)

    def _play_bot(self, session: GameSession) -> dict[str, Any]:
        board = session.board
        context = {
            "player_rating": session.bot_rating,
            "opponent_rating": session.human_rating,
            "phase": classify_phase(board),
        }
        try:
            selection = select_move(
                self.model,
                board,
                context=context,
                mode=session.mode,
            )
            if selection is None:
                raise BotMoveError("O bot não encontrou um lance legal.")
            move = selection.move
            if move not in board.legal_moves:
                raise BotMoveError("O bot retornou um lance ilegal.")
            details = self._move_details(session, move, "bot")
            details.update(
                score=selection.style_score,
                style_rank=selection.style_rank,
                selection_reason=selection.reason,
                coherence_applied=(
                    session.mode == "coherent" and selection.style_rank > 1
                ),
            )
            board.push(move)
            return details
        except BotMoveError:
            raise
        except Exception as exc:
            raise BotMoveError("O bot não conseguiu calcular uma resposta.") from exc

    @staticmethod
    def _move_details(
        session: GameSession, move: chess.Move, actor: str
    ) -> dict[str, Any]:
        board = session.board
        return {
            "ply": len(board.move_stack) + 1,
            "move_number": board.fullmove_number,
            "color": COLOR_NAMES[board.turn],
            "actor": actor,
            "uci": move.uci(),
            "san": board.san(move),
        }

    def _serialize(self, session: GameSession) -> dict[str, Any]:
        board = session.board
        outcome = board.outcome()
        game_over = outcome is not None
        termination = outcome.termination.name if outcome else None
        legal_moves = []
        if not game_over and board.turn == session.human_color:
            legal_moves = sorted(move.uci() for move in board.legal_moves)
        return {
            "id": session.game_id,
            "fen": board.fen(),
            "human_color": COLOR_NAMES[session.human_color],
            "bot_color": COLOR_NAMES[not session.human_color],
            "turn": COLOR_NAMES[board.turn],
            "legal_moves": legal_moves,
            "last_move": board.peek().uci() if board.move_stack else None,
            "last_bot_move": session.last_bot_move,
            "history": self._history(board, session.human_color),
            "check": board.is_check(),
            "game_over": game_over,
            "result": outcome.result() if outcome else None,
            "termination": termination,
            "status": (
                "finished"
                if game_over
                else ("check" if board.is_check() else "playing")
            ),
            "message": self._status_message(session, outcome),
            "model": self.model.name,
            "mode": session.mode,
            "human_rating": session.human_rating,
            "bot_rating": session.bot_rating,
        }

    @staticmethod
    def _history(board: chess.Board, human_color: chess.Color) -> list[dict[str, Any]]:
        replay = chess.Board()
        rows = []
        for ply, move in enumerate(board.move_stack, start=1):
            color = replay.turn
            rows.append(
                {
                    "ply": ply,
                    "move_number": replay.fullmove_number,
                    "color": COLOR_NAMES[color],
                    "actor": "human" if color == human_color else "bot",
                    "uci": move.uci(),
                    "san": replay.san(move),
                }
            )
            replay.push(move)
        return rows

    @staticmethod
    def _status_message(
        session: GameSession, outcome: chess.Outcome | None
    ) -> str:
        if outcome is not None:
            reason = TERMINATION_NAMES.get(
                outcome.termination.name, outcome.termination.name.lower()
            )
            if outcome.winner is None:
                return f"Empate por {reason}."
            if outcome.winner == session.human_color:
                return f"Você venceu por {reason}."
            return f"O bot venceu por {reason}."
        if session.board.turn == session.human_color:
            if session.board.is_check():
                return "Sua vez — você está em xeque."
            return "Sua vez."
        return "Vez do bot."
