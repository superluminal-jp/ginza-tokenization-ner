"""Data models for GiNZA JSON processor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InputItem:
    """Represents a single input text item.

    Attributes
    ----------
    id:
        Stable identifier for the text. Auto-generated if not provided.
    text:
        Original text to process.
    """

    id: str
    text: str


