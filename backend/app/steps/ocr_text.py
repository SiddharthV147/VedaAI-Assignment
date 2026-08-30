"""Preprocessing helpers for noisy handwriting-OCR text.

TrOCR output is dirty in predictable ways: stray leading punctuation, printed
page furniture picked up as text, and glyph confusion between letters and
digits.  Everything in this module works on a single line and knows nothing
about questions, so the question detector can stay focused on structure.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PAGE_HEADER = re.compile(r"^=+\s*page\s+(\d+)\s*=+$", re.IGNORECASE)

_LEADING_NOISE = re.compile(r"""^[\s#*~^`'"“”‘’·•,;:!?_=+/\\|]+""")
_TRAILING_NOISE = re.compile(r"""[\s#*~^`]+$""")
_WHITESPACE = re.compile(r"\s+")
_WORDS = re.compile(r"[A-Za-z]+")

# Pre-printed furniture on answer sheets ("Question Number", "Space for
# writing", "Rough work").  OCR mangles the first characters far more often
# than the last word, so the tail word is the reliable signal.
_HEADER_TAIL_WORDS = frozenset(
    {"number", "no", "nos", "writing", "written", "work"}
)

# Characters the recogniser emits in place of a digit inside a marker.
DIGIT_LOOKALIKES = {
    "O": "0",
    "o": "0",
    "D": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "|": "1",
    "J": "1",
    "Y": "1",
    "y": "1",
    "Z": "2",
    "z": "2",
    "A": "4",
    "S": "5",
    "s": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
}


@dataclass(frozen=True)
class Line:
    """One cleaned OCR line together with its position in the document."""

    page: int
    index: int
    text: str

    @property
    def position(self) -> tuple[int, int]:
        return self.page, self.index


@dataclass(frozen=True)
class Page:
    number: int
    lines: tuple[Line, ...]


def clean_line(raw: str) -> str:
    """Normalise unicode/whitespace and drop decorative edge characters."""
    text = unicodedata.normalize("NFKC", raw)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _LEADING_NOISE.sub("", text)
    text = _TRAILING_NOISE.sub("", text)
    return text.strip()


def parse_pages(text: str) -> list[Page]:
    """Split ``extracted_text.txt`` into pages of cleaned lines."""
    pages: list[Page] = []
    number: int | None = None
    lines: list[Line] = []

    def flush() -> None:
        if number is not None:
            pages.append(Page(number=number, lines=tuple(lines)))

    for raw in text.splitlines():
        header = PAGE_HEADER.match(raw.strip())
        if header:
            flush()
            number = int(header.group(1))
            lines = []
            continue
        if number is None:
            continue
        cleaned = clean_line(raw)
        if cleaned:
            lines.append(Line(page=number, index=len(lines), text=cleaned))

    flush()
    return pages


def words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def letter_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def letter_ratio(text: str) -> float:
    stripped = [ch for ch in text if not ch.isspace()]
    if not stripped:
        return 0.0
    return letter_count(text) / len(stripped)


def is_printed_header(text: str) -> bool:
    """True for pre-printed page furniture rather than handwriting."""
    if any(ch.isdigit() for ch in text):
        return False
    tokens = words(text)
    return bool(tokens) and tokens[-1] in _HEADER_TAIL_WORDS


def is_punctuation_only(text: str) -> bool:
    return not any(ch.isalnum() for ch in text)


def map_lookalikes(token: str) -> str:
    """Rewrite letters that stand in for digits: ``2Y`` -> ``21``."""
    return "".join(DIGIT_LOOKALIKES.get(ch, ch) for ch in token)


def digits_only(token: str) -> str:
    return re.sub(r"\D", "", token)
