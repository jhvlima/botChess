"""Coletor opcional para usos expressamente autorizados da PubAPI."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


USER_AGENT = "botChess/0.1 (authorized use; Chess.com published data API)"


def fetch_json(url: str, timeout: float, retries: int = 4) -> dict:
    if retries < 0:
        raise ValueError("retries não pode ser negativo")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} ao acessar {url}")
                return json.load(response)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 8)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
        ):
            if attempt == retries:
                raise
            delay = min(2**attempt, 8)
        print(
            f"Falha temporária em {url}; nova tentativa em {delay:.0f}s "
            f"({attempt + 1}/{retries})",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("laço de retentativas terminou inesperadamente")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def download_archives(
    username: str, output_dir: Path, timeout: float, retries: int, force: bool
) -> dict:
    base = f"https://api.chess.com/pub/player/{username.casefold()}/games/archives"
    archive_payload = fetch_json(base, timeout, retries)
    urls = archive_payload.get("archives")
    if not isinstance(urls, list):
        raise RuntimeError("Resposta da API não contém uma lista 'archives'")

    summary = {"available": len(urls), "downloaded": 0, "cached": 0, "games": 0}
    for url in urls:
        parts = str(url).rstrip("/").split("/")
        if len(parts) < 2:
            raise RuntimeError(f"URL mensal inesperada: {url}")
        target = output_dir / f"{parts[-2]}-{parts[-1]}.json"
        if target.exists() and not force:
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
                if not isinstance(payload.get("games"), list):
                    raise ValueError
            except (OSError, json.JSONDecodeError, ValueError):
                payload = fetch_json(str(url), timeout, retries)
                atomic_write_json(target, payload)
                summary["downloaded"] += 1
            else:
                summary["cached"] += 1
        else:
            payload = fetch_json(str(url), timeout, retries)
            if not isinstance(payload.get("games"), list):
                raise RuntimeError(f"Resposta mensal inválida: {url}")
            atomic_write_json(target, payload)
            summary["downloaded"] += 1
        summary["games"] += len(payload["games"])
        print(f"{target}: {len(payload['games'])} partidas", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baixa JSONs mensais quando existe autorização aplicável"
    )
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("partidas"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--confirm-authorization",
        action="store_true",
        help="Confirma que o uso pretendido possui a autorização necessária",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.confirm_authorization:
        parser.error(
            "confirme a autorização aplicável com --confirm-authorization; "
            "consulte DATA_POLICY.md"
        )
    try:
        summary = download_archives(
            args.username, args.output_dir, args.timeout, args.retries, args.force
        )
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Falha na API do Chess.com: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Falha de rede: {exc.reason}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
