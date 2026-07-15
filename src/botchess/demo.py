"""Gera o modelo sintético usado para demonstrar a interface local."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .features import FEATURE_COUNT
from .solutions.base import ModelBundle, save_model


# Preferências artificiais por movimentos comuns. Elas servem apenas para que
# uma clonagem nova consiga exercitar a interface sem usar partidas reais.
DEMO_MOVE_COUNTS = {
    "e2e4": 100,
    "e7e5": 100,
    "d2d4": 90,
    "d7d5": 90,
    "g1f3": 80,
    "g8f6": 80,
    "b1c3": 70,
    "b8c6": 70,
    "f1c4": 60,
    "f8c5": 60,
    "c2c4": 55,
    "c7c5": 55,
    "e1g1": 50,
    "e8g8": 50,
    "c1f4": 40,
    "c8f5": 40,
    "d1d2": 30,
    "d8d7": 30,
}


def create_demo_model() -> ModelBundle:
    """Cria um bundle determinístico que não depende de dados externos."""

    return ModelBundle(
        name="synthetic_demo",
        kind="frequency",
        move_counts=dict(DEMO_MOVE_COUNTS),
        metadata={
            "default_rating": 1000,
            "feature_count": FEATURE_COUNT,
            "data_source": "synthetic_curated",
            "intended_use": "local_interface_demo",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera o modelo sintético da demonstração do botChess"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/demo_model.joblib"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    save_model(create_demo_model(), args.output)
    print(f"Modelo sintético criado em {args.output}")


if __name__ == "__main__":
    main()
