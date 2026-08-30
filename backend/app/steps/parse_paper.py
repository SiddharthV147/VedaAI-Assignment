"""Turn OCR lines from a printed question paper into the nested question tree.

This is the same parser the previous API used.  Only the source of the lines
changed: CRAFT + TrOCR instead of PaddleOCR.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

log = logging.getLogger(__name__)

_RE_SECTION = re.compile(r"^(?:SECTION|SEC)\s*[-:]*\s*([A-E])\b", re.IGNORECASE)
_RE_MAIN_Q = re.compile(
    r"^(?:Q|Q\.|Q\s|Question\s*|Qno\.|Q\.No\.)\s*0*([1-9][0-9]*)", re.IGNORECASE
)
_RE_MAIN_Q_ALT = re.compile(r"^0*([1-9][0-9]*)\s*\.")

_RE_MCQ_SECTION_HINT = re.compile(
    r"multiple\s*choice|MCQ|"
    r"select\s+(?:the\s+)?(?:most\s+)?(?:appropriate|correct)\s+option|"
    r"choose\s+(?:the\s+)?(?:correct|best|right)\s+(?:option|answer)",
    re.IGNORECASE,
)

_RE_MCQ_OPTION_LINE = re.compile(
    r"""
    ^\s*
    (?:
        \(\s*[A-D]\s*\)
      | \(\s*[A-D]\s*(?=[^\)])
      | [A-D]\s*\)
      | [A-D]\.(?=\s+\S)
      | [A-D]\s+(?=\d)
      | [A-D]\s+(?=[A-Z][a-z\d])
    )
    """,
    re.VERBOSE,
)

_RE_SUB = re.compile(
    r"^\(\s*([a-z]{1,4}|[ivxIVX]{1,5})\s*\)(?!\s*[A-D]\b)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\b\d+/\d+/\d+\b", " ", text)
    text = re.sub(r"\b\d+\s*\|\s*Page\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bP\.T\.O\.?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\$#~_{}\[\]\\|]+", " ", text)
    text = re.sub(r"^[\bA-Z0-9\W]{1,25}\s+(?=[A-Z][a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_section_map(lines: Iterable[tuple[int, str, list[float]]]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "General"
    for _, text, _ in lines:
        if _RE_SECTION.match(text):
            current = f"Section {_RE_SECTION.match(text).group(1).upper()}"
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(text)
    return sections


def _is_mcq_section(section_lines: list[str]) -> bool:
    mcq_votes = 0
    sub_votes = 0
    for text in section_lines:
        if _RE_MCQ_SECTION_HINT.search(text):
            return True
        if _RE_MCQ_OPTION_LINE.match(text):
            mcq_votes += 1
        elif _RE_SUB.match(text):
            sub_votes += 1
    return mcq_votes > 0 and mcq_votes >= sub_votes


def parse_questions(lines: list[tuple[int, str, list[float]]]) -> dict:
    section_map = _build_section_map(lines)
    mcq_flags = {
        sec: _is_mcq_section(sec_lines)
        for sec, sec_lines in section_map.items()
        if sec != "General"
    }
    log.info("MCQ section flags: %s", mcq_flags)

    nested: dict = {}
    current_section = "General"
    is_mcq = False
    current_q: str | None = None
    current_sub: str | None = None

    for _, text, _ in lines:
        sec_m = _RE_SECTION.match(text)
        if sec_m:
            current_section = f"Section {sec_m.group(1).upper()}"
            is_mcq = mcq_flags.get(current_section, False)
            current_q = None
            current_sub = None
            nested.setdefault(current_section, {})
            continue

        q_m = _RE_MAIN_Q.match(text) or _RE_MAIN_Q_ALT.match(text)
        if q_m:
            if current_section == "General":
                current_section = "Section A"
                nested.setdefault(current_section, {})
            current_q = f"Question {q_m.group(1)}"
            current_sub = None
            nested[current_section].setdefault(current_q, {"_text": ""})
            content = text[q_m.end() :].strip()
            if content and not (is_mcq and _RE_MCQ_OPTION_LINE.match(content)):
                nested[current_section][current_q]["_text"] += content + " "
            continue

        if is_mcq and _RE_MCQ_OPTION_LINE.match(text):
            continue

        if not is_mcq and current_q and current_section != "General":
            sub_m = _RE_SUB.match(text)
            if sub_m:
                current_sub = f"({sub_m.group(1).lower()})"
                nested[current_section][current_q].setdefault(current_sub, {"_text": ""})
                content = text[sub_m.end() :].strip()
                if content:
                    nested[current_section][current_q][current_sub]["_text"] += content + " "
                continue

        if current_q and current_section in nested and current_section != "General":
            if not is_mcq and current_sub:
                nested[current_section][current_q][current_sub]["_text"] += text + " "
            else:
                nested[current_section][current_q]["_text"] += text + " "

    nested.pop("General", None)

    def recursive_clean(node: dict) -> None:
        to_delete = []
        for key, value in node.items():
            if isinstance(value, dict):
                recursive_clean(value)
                if not value:
                    to_delete.append(key)
            elif isinstance(value, str):
                node[key] = clean_text(value)
        for key in to_delete:
            del node[key]

    recursive_clean(nested)
    return {"questions": nested}
