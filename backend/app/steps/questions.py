"""Locate question markers in OCR text and map every question to a page.

Handwritten markers survive OCR badly: ``Q.14`` comes back as ``8.4``,
``Q.21`` as ``Q.2Y``, ``Q.20`` as ``g20"`` and ``Q.23`` as ``gill``.  Instead of
enumerating those spellings, the scanner works in three layers:

1. *Shape* - a marker is a head (``Q``/``Ques``/``Ans``/``Sol``, or a glyph the
   recogniser confuses with ``Q``) followed by a small number, or a bare
   ``12)``.
2. *Expectation* - answer sheets are written in order, so the next marker is
   almost always ``last + 1``.  Candidate readings of a garbled number are
   scored against that expectation, which repairs dropped or swapped digits
   without hardcoding any particular misreading.
3. *Sequence inference* - questions OCR destroyed entirely are recovered from
   the gap between two detected neighbours.

Nothing here is tied to a particular paper: numbering may run continuously or
restart inside every section/part, and any of the three layers may fail on a
given line without breaking the ones around it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from app.steps.ocr_text import (
    Line,
    Page,
    digits_only,
    is_printed_header,
    is_punctuation_only,
    letter_count,
    letter_ratio,
    map_lookalikes,
    parse_pages,
)

MAX_QUESTION_NUMBER = 120
# How far ahead of the expected number a marker may jump before we distrust it.
MAX_SKIP_EXPLICIT = 3
MAX_SKIP_BARE = 2
# A section that restarts numbering must open near 1.
MAX_SKIP_RESTART = 1
# Only short runs of consecutive misses are safe to infer.
MAX_INFERRED_RUN = 2
# Pages before the first readable marker that may hold the opening questions.
LEADING_SEARCH_PAGES = 1
MIN_ANSWER_LETTERS = 2
MIN_WEAK_TAIL_LETTERS = 8

# Head characters the recogniser produces for a handwritten "Q": the loop and
# tail are read as a digit or as a round lowercase letter.
Q_GLYPHS = "QqOo0Gg98a"
_NUM_CHARS = "0-9OoIilYyBSZGA|"

_RE_SECTION = re.compile(
    r"""\b(?P<kind>s[e3]ct[i1l]?[o0]ns?|parts?)\b
        \s*[-–—:.,]?\s*
        (?P<letter>[A-Ha-h])(?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)
_RE_WORD_HEAD = re.compile(
    rf"""^\(?\s*
        (?:q(?:ue?s?(?:t(?:ion)?)?)?|ans(?:wer)?|sol(?:ution)?)
        \s*(?:n[o0]s?)?\s*[.\-–—_:,)'"]*\s*
        (?P<num>[0-9][{_NUM_CHARS}]{{0,2}})(?![0-9])
    """,
    re.IGNORECASE | re.VERBOSE,
)
# No leading bracket and no ")" separator here, otherwise MCQ options such as
# "(a) 5" and "a) 5" would read as question five.
_RE_GLYPH_HEAD = re.compile(
    rf"""^[{Q_GLYPHS}]
        \s*[.\-–—_:,'"]{{0,2}}\s*
        (?P<num>[0-9][{_NUM_CHARS}]{{0,2}})(?![0-9])
    """,
    re.VERBOSE,
)
_RE_BARE_HEAD = re.compile(
    r"^\(?\s*(?P<num>\d{1,2})(?!\d)\s*(?P<closer>[).:,\-–—]?)\s*(?P<rest>.*)$"
)
# "1917 GBI Both ..." is "Q.17" with the head glyph absorbed into the number.
_RE_LONG_HEAD = re.compile(r"^\(?\s*(?P<num>\d{3,4})(?!\d)\s+(?P<rest>.+)$")

# A head glyph followed by punctuation instead of a legible number.
_RE_WEAK_PUNCT = re.compile(rf"^[{Q_GLYPHS}]\s*[.\-–—_:)]\s*(?![0-9])")
# A head glyph swallowed into a short nonsense word, e.g. "gill" for "Q.23".
_RE_WEAK_WORD = re.compile(rf"^(?P<word>[{Q_GLYPHS}][A-Za-z]{{1,3}})\b")

# Common short English words opening with a head glyph.  They are ordinary
# prose, not a mangled marker, and would otherwise flood the weak-marker pass.
_WEAK_WORD_STOPLIST = frozenset(
    """
    a an and any are as at all also able add age ago aim area away
    o of on or our out off old oil one once only own over
    go got get gas gap gum guy give gave good gone goes gold grow
    q qty
    """.split()
)


@dataclass(frozen=True)
class QuestionMark:
    number: int
    page: int
    section: str
    inferred: bool = False
    line_index: int | None = None


@dataclass(frozen=True)
class _Candidate:
    """A marker that was read from the page, with its position in the text."""

    number: int
    section: str
    line: Line


@dataclass
class QuestionIndex:
    marks: list[QuestionMark] = field(default_factory=list)

    @property
    def inferred(self) -> list[QuestionMark]:
        return [mark for mark in self.marks if mark.inferred]

    @property
    def detected(self) -> list[QuestionMark]:
        return [mark for mark in self.marks if not mark.inferred]

    def as_mapping(self) -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[int, int]] = {}
        for mark in self.marks:
            grouped.setdefault(mark.section or "Unknown", {})[mark.number] = mark.page
        return {
            section: {
                f"question{number}": str(page) for number, page in sorted(pages.items())
            }
            for section, pages in grouped.items()
        }


def _section_label(match: re.Match[str]) -> str:
    kind = "Part" if match.group("kind").lower().startswith("p") else "Section"
    return f"{kind} {match.group('letter').upper()}"


def _number_candidates(token: str, *, allow_repair: bool) -> list[int]:
    """Plausible readings of a garbled number, best guess first."""
    plain = digits_only(token)
    mapped = digits_only(map_lookalikes(token))

    ordered: list[int] = []

    def add(value: int) -> None:
        if 1 <= value <= MAX_QUESTION_NUMBER and value not in ordered:
            ordered.append(value)

    if plain:
        add(int(plain))
    if mapped and mapped != plain:
        add(int(mapped))

    base = plain or mapped
    if not base:
        return ordered

    if len(base) >= 3:
        add(int(base[-2:]))
        add(int(base[:2]))
    elif allow_repair and base != "0":
        digit = int(base)
        # A dropped leading digit ("8.4" for "Q.14")...
        for tens in range(1, 10):
            add(tens * 10 + digit)
        # ...or a dropped trailing digit ("0.3" for "Q.31").
        for ones in range(10):
            add(digit * 10 + ones)

    return ordered


def _pick_number(candidates: list[int], expected: int | None, max_skip: int) -> int | None:
    """Prefer the expected number, else the best reading just ahead of it.

    ``expected is None`` means nothing has been read yet, so there is no
    sequence to lean on and the plainest reading has to be trusted.
    """
    if expected is None:
        return candidates[0] if candidates else None
    if expected in candidates:
        return expected
    for value in candidates:
        if expected <= value <= expected + max_skip:
            return value
    return None


def _rest_is_plausible(rest: str) -> bool:
    """A marker is followed by an answer, by punctuation, or by nothing."""
    rest = rest.strip()
    return (
        not rest
        or is_punctuation_only(rest)
        or letter_count(rest) >= MIN_ANSWER_LETTERS
    )


def _read_explicit(text: str, expected: int | None) -> int | None:
    for pattern in (_RE_WORD_HEAD, _RE_GLYPH_HEAD):
        match = pattern.match(text)
        if not match:
            continue
        rest = text[match.end() :]
        if letter_count(rest) == 0 and any(ch.isdigit() for ch in rest):
            continue  # numeric working such as "0 0" or "8.150 - 3.2"
        candidates = _number_candidates(
            match.group("num"), allow_repair=expected is not None
        )
        number = _pick_number(candidates, expected, MAX_SKIP_EXPLICIT)
        if number is not None:
            return number
    return None


def _read_bare(text: str, page: int, expected: int) -> int | None:
    match = _RE_BARE_HEAD.match(text)
    if not match:
        return None
    rest = match.group("rest")
    number = int(match.group("num"))
    if not _rest_is_plausible(rest):
        return None
    if number == page and (not rest or is_punctuation_only(rest)):
        return None  # the printed page number
    if number == expected:
        return number
    if match.group("closer") == ")" and expected <= number <= expected + MAX_SKIP_BARE:
        return number
    return None


def _read_long(text: str, expected: int) -> int | None:
    match = _RE_LONG_HEAD.match(text)
    if not match:
        return None
    rest = match.group("rest")
    if letter_count(rest) < 3 or letter_ratio(rest) < 0.5:
        return None
    candidates = _number_candidates(match.group("num"), allow_repair=False)
    return _pick_number(candidates, expected, MAX_SKIP_BARE)


def _scan_marker(line: Line, last_number: int) -> int | None:
    """Return the question number this line starts, if any."""
    expected = last_number + 1
    return (
        _read_explicit(line.text, expected if last_number else None)
        or _read_bare(line.text, line.page, expected)
        or _read_long(line.text, expected)
    )


def _is_weak_marker(line: Line) -> bool:
    """True when a line looks like a marker whose number is unreadable."""
    text = line.text
    if text.startswith("("):
        return False  # "( b )" and "( iii )" are sub-parts, not questions
    if _RE_WEAK_PUNCT.match(text):
        return True
    match = _RE_WEAK_WORD.match(text)
    if not match:
        return False
    if match.group("word").lower() in _WEAK_WORD_STOPLIST:
        return False
    return letter_count(text[match.end() :]) >= MIN_WEAK_TAIL_LETTERS


def _scan(pages: list[Page]) -> tuple[list[_Candidate], list[Line]]:
    found: list[_Candidate] = []
    weak: list[Line] = []
    section = ""
    last_number = 0
    restart_until_page = 0

    for page in pages:
        for line in page.lines:
            if is_printed_header(line.text):
                continue

            heading = _RE_SECTION.search(line.text)
            if heading:
                section = _section_label(heading)
                restart_until_page = line.page + 1
                remainder = line.text[heading.end() :].strip()
                if not remainder:
                    continue
                line = replace(line, text=remainder)

            number = _scan_marker(line, last_number)
            if number is not None and number <= last_number:
                number = None
            if number is None and last_number and line.page <= restart_until_page:
                # Papers that number each section from one start over here.
                # Only an unambiguous "Q<n>" head may reset the sequence.
                restart = _read_explicit(line.text, 1)
                if restart is not None and restart <= 1 + MAX_SKIP_RESTART:
                    number = restart

            if number is not None:
                found.append(_Candidate(number=number, section=section, line=line))
                last_number = number
                restart_until_page = 0
                continue

            if _is_weak_marker(line):
                weak.append(line)

    return found, weak


def _interpolate(prev_page: int, next_page: int, index: int, gap: int) -> int:
    span = next_page - prev_page
    offset = int(span * index / (gap + 1) + 0.5)
    return max(prev_page, min(next_page, prev_page + offset))


def _weak_between(
    weak: list[Line],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[Line]:
    return [line for line in weak if start < line.position < end]


def _fill_leading(first: _Candidate, weak: list[Line]) -> list[QuestionMark]:
    """Recover the opening questions when the sheet starts unreadably."""
    missing = list(range(1, first.number))
    if not missing or len(missing) > MAX_INFERRED_RUN:
        return []
    hits = [
        line
        for line in _weak_between(weak, (0, 0), first.line.position)
        if line.page >= first.line.page - LEADING_SEARCH_PAGES
    ]
    if len(hits) != len(missing):
        return []
    return [
        QuestionMark(
            number=number,
            page=line.page,
            section=first.section,
            inferred=True,
            line_index=line.index,
        )
        for number, line in zip(missing, hits)
    ]


def _fill_between(
    previous: _Candidate,
    current: _Candidate,
    weak: list[Line],
) -> list[QuestionMark]:
    missing = list(range(previous.number + 1, current.number))
    if not missing or len(missing) > MAX_INFERRED_RUN:
        return []

    prev_page = previous.line.page
    next_page = current.line.page
    section = previous.section or current.section

    if prev_page == next_page:
        # Both neighbours sit on one page, so anything between them does too.
        return [
            QuestionMark(number=number, page=prev_page, section=section, inferred=True)
            for number in missing
        ]

    hits = _weak_between(weak, previous.line.position, current.line.position)
    if len(hits) == len(missing):
        return [
            QuestionMark(
                number=number,
                page=line.page,
                section=section,
                inferred=True,
                line_index=line.index,
            )
            for number, line in zip(missing, hits)
        ]

    return [
        QuestionMark(
            number=number,
            page=_interpolate(prev_page, next_page, index, len(missing)),
            section=section,
            inferred=True,
        )
        for index, number in enumerate(missing, start=1)
    ]


def _fill_gaps(found: list[_Candidate], weak: list[Line]) -> list[QuestionMark]:
    marks: list[QuestionMark] = []

    for position, candidate in enumerate(found):
        previous = found[position - 1] if position else None
        starts_run = previous is None or candidate.number <= previous.number
        if starts_run:
            marks.extend(_fill_leading(candidate, weak))
        else:
            marks.extend(_fill_between(previous, candidate, weak))
        marks.append(
            QuestionMark(
                number=candidate.number,
                page=candidate.line.page,
                section=candidate.section,
                line_index=candidate.line.index,
            )
        )

    return marks


def analyze_pages(pages: list[Page]) -> QuestionIndex:
    """Detect every question marker from already-split OCR pages."""
    found, weak = _scan(pages)
    return QuestionIndex(marks=_fill_gaps(found, weak))


def analyze_questions(text: str) -> QuestionIndex:
    """Detect every question marker in ``extracted_text.txt`` content."""
    return analyze_pages(parse_pages(text))


def map_questions_to_pages(text: str) -> dict[str, dict[str, str]]:
    return analyze_questions(text).as_mapping()
