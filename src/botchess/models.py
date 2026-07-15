"""Compatibilidade para artefatos criados antes da organização por soluções."""

from .solutions.base import ModelBundle, load_model, save_model
from .solutions.frequency.model import train as train_frequency

__all__ = ["ModelBundle", "load_model", "save_model", "train_frequency"]
