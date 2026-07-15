"""Servidor local da interface gráfica do botChess."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
from typing import Any, Iterable
import webbrowser

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.serving import make_server

from .solutions.base import ModelBundle, load_model
from .web_game import GameError, GameStore


def create_app(
    model: ModelBundle,
    bot_rating: int | None = None,
    config: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=16 * 1024)
    if config:
        app.config.update(config)
    store = GameStore(model, bot_rating=bot_rating)
    app.extensions["botchess_games"] = store

    @app.after_request
    def secure_local_response(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(GameError)
    def handle_game_error(error: GameError):
        if error.status_code >= 500:
            app.logger.exception("Falha ao calcular a jogada do bot")
        return (
            jsonify(error={"code": error.code, "message": str(error)}),
            error.status_code,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_request(_error: RequestEntityTooLarge):
        return (
            jsonify(
                error={
                    "code": "request_too_large",
                    "message": "A requisição excede o limite permitido.",
                }
            ),
            413,
        )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=model.name)

    @app.post("/api/games")
    def create_game():
        payload = _json_object()
        return jsonify(
            store.create(
                human_color=payload.get("human_color"),
                human_rating=payload.get("human_rating"),
                mode=payload.get("mode"),
            )
        ), 201

    @app.get("/api/games/<game_id>")
    def get_game(game_id: str):
        return jsonify(store.get(game_id))

    @app.post("/api/games/<game_id>/moves")
    def play_move(game_id: str):
        payload = _json_object()
        return jsonify(store.play(game_id, payload.get("uci")))

    @app.delete("/api/games/<game_id>")
    def delete_game(game_id: str):
        store.delete(game_id)
        return "", 204

    return app


def _json_object() -> dict[str, Any]:
    if not request.is_json:
        raise GameError("Envie a requisição como JSON.")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise GameError("O corpo JSON deve ser um objeto.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Abre uma página local para jogar contra o botChess"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/demo_model.joblib")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rating", type=int, help="Rating usado no contexto do bot")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Não abre o navegador automaticamente",
    )
    return parser


def browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}"


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("A porta deve estar entre 1 e 65535")
    try:
        model = load_model(args.model)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"Não foi possível carregar o modelo: {exc}") from exc

    try:
        app = create_app(model, bot_rating=args.rating)
    except GameError as exc:
        raise SystemExit(f"Configuração inválida: {exc}") from exc
    try:
        server = make_server(args.host, args.port, app, threaded=True)
    except OSError as exc:
        raise SystemExit(f"Não foi possível iniciar o servidor: {exc}") from exc

    url = browser_url(args.host, args.port)
    print(f"botChess disponível em {url}")
    print("Pressione Ctrl+C para encerrar.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Aviso: o servidor está acessível fora deste computador.")
    if not args.no_browser:
        opener = threading.Timer(0.4, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
