from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Word:
    word: str
    start: float
    end: float
    score: float | None = None

def validate_words(words: list[Word]) -> None:
    if not words: raise ValueError("alignment returned no words")
    for a, b in zip(words, words[1:]):
        if a.end < a.start or b.start < a.end:
            raise ValueError("non-monotonic/overlapping word alignment")

def gaps(words: list[Word], threshold: float = .18) -> list[dict]:
    validate_words(words)
    out = []
    for i, (a, b) in enumerate(zip(words, words[1:])):
        gap = round(b.start - a.end, 4)
        out.append({"after_word_index": i, "after_word": a.word, "before_word": b.word,
                    "gap_seconds": gap, "candidate": gap >= threshold})
    return out

def compare_breaks(reference: list[dict], generated: list[dict], tolerance: float = .12) -> dict:
    r = {x["after_word_index"] for x in reference if x.get("reviewed_label") == "break"}
    g = {x["after_word_index"] for x in generated if x.get("gap_seconds", 0) >= tolerance}
    return {"matched": sorted(r & g), "missing": sorted(r - g), "extra": sorted(g - r), "tolerance_seconds": tolerance}
