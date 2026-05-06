from __future__ import annotations
from typing import List
import numpy as np


_MODEL = None


def _get_model(name: str):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(name)
    return _MODEL


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def round_trip_similarity(
    originals: List[str],
    back_translations: List[str],
    semantic_model: str,
) -> List[float]:

    model = _get_model(semantic_model)
    emb_o = model.encode(originals, convert_to_numpy=True, normalize_embeddings=True)
    emb_b = model.encode(back_translations, convert_to_numpy=True, normalize_embeddings=True)
    return [float(np.dot(o, b)) for o, b in zip(emb_o, emb_b)]


def passes_round_trip(
    original: str,
    back_translation: str,
    semantic_model: str,
    delta: float,
) -> bool:
    sim = round_trip_similarity([original], [back_translation], semantic_model)[0]
    return sim >= delta
