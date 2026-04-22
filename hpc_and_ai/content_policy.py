from __future__ import annotations

import re
from typing import Optional, Tuple

# Lihtne algversioon. Hiljem saad seda laiendada, kui tahad rohkem kontrolli.
HEALTH_PATTERNS = [
    r"\btervis\b",
    r"\bmeditsiin\b",
    r"\bdiagnoos\b",
    r"\bravi\b",
    r"\bmedikament",
    r"\bpill\b",
    r"\bvererõhk\b",
    r"\bsüda\b",
    r"\bvalu\b",
    r"\bpeavalu\b",
    r"\bkõhuvalu\b",
    r"\bpalavik\b",
    r"\bdepress",
    r"\bärev",
    r"\bhaigus\b",
    r"\bhaige\b",
]

POLITICS_PATTERNS = [
    r"\bpoliitik\b",
    r"\bvalim",
    r"\bpartei\b",
    r"\bpeaminister\b",
    r"\bpresident\b",
    r"\breformierakond\b",
    r"\bkeskerakond\b",
    r"\bisme\b",
    r"\bisamaa\b",
    r"\bsotsiaaldemokraat",
    r"\bekre\b",
    r"\briigikogu\b",
    r"\bvalitsus\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_blocked_topic(text: str) -> Tuple[bool, Optional[str]]:
    """
    Tagastab:
      (True, "health")  -> terviseteema
      (True, "politics") -> poliitikateema
      (False, None) -> lubatud
    """
    if _matches_any(text, HEALTH_PATTERNS):
        return True, "health"
    if _matches_any(text, POLITICS_PATTERNS):
        return True, "politics"
    return False, None


def refusal_text(blocked_type: Optional[str]) -> str:
    if blocked_type == "health":
        return (
            "Ma ei saa tervise teemal nõu anda. "
            "Kui see on kiire või tõsine mure, võta ühendust arsti või erakorralise abiga."
        )
    if blocked_type == "politics":
        return (
            "Ma ei vasta poliitika küsimustele. "
            "Võin rääkida hoopis käsitööst, luuletustest, faktidest, muistenditest või mingil muul teemal."
        )
    return "Ma ei saa sellele küsimusele vastata."


def should_block_search(text: str) -> Tuple[bool, Optional[str]]:
    """
    Sama kontroll, aga mõeldud eraldi juhul, kui tahame enne web-search'i otsustada.
    """
    return is_blocked_topic(text)
