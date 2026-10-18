"""Redact personal information before text leaves the user's machine.

Pattern based and deliberately conservative: it removes contact details, profile
links, government-style ID numbers, dates of birth, labelled addresses and the
candidate's name. It does not understand free-form text, so the UI shows the
redacted text to the user before anything is sent to the AI service.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+|\b(?:linkedin|github)\.com/[^\s<>()\[\]]+", re.IGNORECASE)
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
DOB = re.compile(
    r"(?i)\b(d\.?o\.?b\.?|date\s+of\s+birth|born(?:\s+on)?)(\s*[:\-]?\s*)"
    r"(\d{1,2}[/.\-\s]\w{1,9}[/.\-\s]\d{2,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
)
ADDRESS = re.compile(
    r"(?im)^(\s*(?:permanent\s+|current\s+|residential\s+|home\s+|mailing\s+)?address\s*[:\-]\s*)(.+)$"
)
PHONE_CANDIDATE = re.compile(r"(?<![\w.@/])(\+?\d[\d\s().\-]{8,17}\d)(?![\w@])")
IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

NAME_STOPWORDS = {
    "resume",
    "curriculum",
    "vitae",
    "cv",
    "profile",
    "summary",
    "objective",
    "contact",
    "education",
    "experience",
    "skills",
    "projects",
    "engineer",
    "developer",
    "student",
    "intern",
    "analyst",
    "scientist",
    "software",
    "data",
    "full",
    "stack",
    "machine",
    "learning",
    "career",
    "about",
    "me",
    "page",
    "of",
}
NAME_WORD = re.compile(r"^[A-Z][A-Za-z'’.\-]+$")


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def guess_name(text: str) -> str | None:
    """Return the candidate's name if one of the first lines looks like a name."""
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:3]:
        words = line.split()
        if not 2 <= len(words) <= 4 or len(line) > 40 or re.search(r"[\d@:/|]", line):
            continue
        if any(w.lower().strip(".,") in NAME_STOPWORDS for w in words):
            continue
        if all(NAME_WORD.match(w) or w.isupper() for w in words):
            return line
    return None


def _phone_ok(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15 or IPV4.match(candidate.strip()):
        return False
    has_structure = candidate.startswith("+") or bool(re.search(r"[\s().\-]", candidate))
    return has_structure or len(digits) == 10


def _name_patterns(names: Iterable[str]) -> list[re.Pattern[str]]:
    parts: set[str] = set()
    for name in names:
        name = name.strip()
        if not name:
            continue
        parts.add(name)
        parts.update(p for p in re.split(r"\s+", name) if len(p) >= 3)
    return [re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in sorted(parts, key=len, reverse=True)]


def redact(
    text: str,
    *,
    names: Iterable[str] = (),
    detect_name: bool = True,
    redact_links: bool = True,
) -> RedactionResult:
    """Replace personal data with typed placeholders such as [EMAIL] and [PHONE]."""
    counts: Counter[str] = Counter()

    def simple(placeholder: str):
        def replace(_match: re.Match[str]) -> str:
            counts[placeholder.strip("[]")] += 1
            return placeholder

        return replace

    def keep_prefix(placeholder: str, prefix_groups: tuple[int, ...]):
        def replace(match: re.Match[str]) -> str:
            counts[placeholder.strip("[]")] += 1
            return "".join(match.group(i) for i in prefix_groups) + placeholder

        return replace

    def link(match: re.Match[str]) -> str:
        value = match.group(0).lower()
        kind = "LINKEDIN" if "linkedin" in value else "GITHUB" if "github" in value else "URL"
        counts[kind] += 1
        return f"[{kind}]"

    def phone(match: re.Match[str]) -> str:
        if not _phone_ok(match.group(1)):
            return match.group(0)
        counts["PHONE"] += 1
        return "[PHONE]"

    out = EMAIL.sub(simple("[EMAIL]"), text)
    if redact_links:
        out = URL.sub(link, out)
    for pattern in (PAN, AADHAAR, SSN):
        out = pattern.sub(simple("[ID_NUMBER]"), out)
    out = DOB.sub(keep_prefix("[DOB]", (1, 2)), out)
    out = ADDRESS.sub(keep_prefix("[ADDRESS]", (1,)), out)
    out = PHONE_CANDIDATE.sub(phone, out)

    to_hide = [name for name in names if name and name.strip()]
    if detect_name and (guessed := guess_name(text)):
        to_hide.append(guessed)
    for pattern in _name_patterns(to_hide):
        out, replaced = pattern.subn("[NAME]", out)
        counts["NAME"] += replaced

    return RedactionResult(text=out, counts=dict(counts))
