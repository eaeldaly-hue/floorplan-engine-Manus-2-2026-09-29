"""Backward-compatible module entry point for the room-centric analyzer."""

from .analyzer import FloorPlanAnalyzer
from .cli import main

__all__ = ["FloorPlanAnalyzer", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
