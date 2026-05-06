from __future__ import annotations
from typing import Iterable
from simhash import Simhash


def simhash_value(text: str, width: int = 64) -> Simhash:
    return Simhash(text, f=width)


def hamming_similarity(a: Simhash, b: Simhash, width: int = 64) -> float:
    return 1.0 - (a.distance(b) / width)


def is_duplicate(candidate: str, pool: Iterable[str], epsilon: float, width: int = 64) -> bool:
    h_cand = simhash_value(candidate, width=width)
    threshold = 1.0 - epsilon
    for ex in pool:
        h_ex = simhash_value(ex, width=width)
        if hamming_similarity(h_cand, h_ex, width=width) >= threshold:
            return True
    return False
