from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import numpy as np

from .extract import TrajectoryRecord


def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def trajectory_magnitude(delta_h: np.ndarray) -> np.ndarray:
    return np.linalg.norm(delta_h, axis=1)


def emergence_layer(delta_h: np.ndarray, eps: float) -> int:
    mags = trajectory_magnitude(delta_h)
    where = np.where(mags > eps)[0]
    return int(where[0] + 1) if len(where) else int(len(mags))


def cumulative_shift(delta_h: np.ndarray) -> float:
    return float(np.linalg.norm(delta_h, axis=1).sum())


def trajectory_consistency(delta_h: np.ndarray, mean_traj: np.ndarray) -> float:
    L = min(delta_h.shape[0], mean_traj.shape[0])
    if L == 0:
        return 0.0
    sims = [_safe_cos(delta_h[l], mean_traj[l]) for l in range(L)]
    return float(np.mean(sims))


def evolution_divergence(mean_a: np.ndarray, mean_b: np.ndarray) -> float:
    L = min(mean_a.shape[0], mean_b.shape[0])
    if L == 0:
        return 0.0
    diffs = np.linalg.norm(mean_a[:L] - mean_b[:L], axis=1)
    return float(diffs.mean())


@dataclass
class Aggregate:
    mean_traj: np.ndarray   # (L-1, m)
    n: int


def aggregate_trajectories(records: Iterable[TrajectoryRecord]) -> Aggregate:
    arrs = [r.delta_h for r in records]
    if not arrs:
        return Aggregate(mean_traj=np.zeros((0, 0)), n=0)
    L = min(a.shape[0] for a in arrs)
    m = arrs[0].shape[1]
    stack = np.stack([a[:L] for a in arrs])     # (N, L, m)
    return Aggregate(mean_traj=stack.mean(axis=0), n=stack.shape[0])


def grouped_aggregates(
    records: List[TrajectoryRecord],
    by: Tuple[str, ...] = ("t", "c", "s", "language", "tau"),
) -> Dict[Tuple, Aggregate]:
    groups: Dict[Tuple, List[TrajectoryRecord]] = {}
    for r in records:
        key = tuple(getattr(r, k) for k in by)
        groups.setdefault(key, []).append(r)
    return {k: aggregate_trajectories(v) for k, v in groups.items()}


@dataclass
class TrajectoryScores:
    TM: float          
    TM_curve: np.ndarray  
    EL: float
    TC: float
    ED: float
    CS: float
    Avg: float
    n: int = 1


def per_record_scores(
    records: List[TrajectoryRecord],
    eps: float,
    group_by: Tuple[str, ...] = ("t", "c", "s", "language", "tau"),
) -> List[Dict]:
    aggs = grouped_aggregates(records, by=group_by)
    rows = []
    for r in records:
        key = tuple(getattr(r, k) for k in group_by)
        agg = aggs[key]
        mags = trajectory_magnitude(r.delta_h)
        rows.append({
            "model_name": r.model_name, "language": r.language, "domain": r.domain,
            "t": r.t, "c": r.c, "s": r.s, "tau": r.tau,
            "TM_curve": mags,
            "TM": float(mags.mean()),
            "EL": float(emergence_layer(r.delta_h, eps)),
            "TC": float(trajectory_consistency(r.delta_h, agg.mean_traj)),
            "CS": float(cumulative_shift(r.delta_h)),
        })
    return rows


def condition_scores(
    records: List[TrajectoryRecord],
    eps: float,
    by_for_aggregate: Tuple[str, ...] = ("t", "c", "s", "language", "tau"),
    pair_by: Tuple[str, ...] = ("t", "c", "s"),
) -> List[Dict]:
    per = per_record_scores(records, eps=eps, group_by=by_for_aggregate)
    aggs = grouped_aggregates(records, by=by_for_aggregate)

    by_pair: Dict[Tuple, List[np.ndarray]] = {}
    for k, agg in aggs.items():
        kf = tuple(k[by_for_aggregate.index(field)] for field in pair_by)
        by_pair.setdefault(kf, []).append(agg.mean_traj)

    pair_means = {}
    for kf, v in by_pair.items():
        L_min = min(x.shape[0] for x in v)
        pair_means[kf] = np.mean(np.stack([m[:L_min] for m in v]), axis=0)
    keys = list(pair_means.keys())
    eds = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            eds.append(evolution_divergence(pair_means[keys[i]], pair_means[keys[j]]))
    ED_global = float(np.mean(eds)) if eds else 0.0

    out: Dict[Tuple, List[Dict]] = {}
    for row in per:
        out.setdefault((row["model_name"], row["tau"]), []).append(row)
    summaries = []
    for (model, tau), rows in out.items():
        TM = float(np.mean([r["TM"] for r in rows]))
        EL = float(np.mean([r["EL"] for r in rows]))
        TC = float(np.mean([r["TC"] for r in rows]))
        CS = float(np.mean([r["CS"] for r in rows]))

        L_max = max(1, max(r["TM_curve"].shape[0] for r in rows))
        Avg = compute_avg(TM=TM, EL=EL, TC=TC, ED=ED_global, CS=CS, L_max=L_max)
        summaries.append({
            "model_name": model, "tau": tau, "n": len(rows),
            "TM": TM, "EL": EL, "TC": TC, "ED": ED_global, "CS": CS, "Avg": Avg,
        })
    return summaries


def compute_avg(TM: float, EL: float, TC: float, ED: float, CS: float, L_max: int) -> float:
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    TM_n = float(sigmoid(TM / 3.0))
    EL_n = float(max(0.0, 1.0 - EL / max(1, L_max)))
    TC_n = float(max(0.0, min(1.0, TC)))
    ED_n = float(sigmoid(ED / 2.0))
    CS_n = float(sigmoid(CS / max(1.0, 10.0 * L_max)))
    return float(np.mean([TM_n, EL_n, TC_n, ED_n, CS_n]))
