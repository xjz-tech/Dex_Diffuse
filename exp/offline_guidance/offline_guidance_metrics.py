"""Offline metrics for comparing DP, sim policy, guidance, and linear mix."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError("mae shape mismatch: %s vs %s" % (pred.shape, target.shape))
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError("rmse shape mismatch: %s vs %s" % (pred.shape, target.shape))
    return float(np.sqrt(np.mean(np.square(pred - target))))


def _time_diff(chunk: np.ndarray, order: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2:
        raise ValueError("expected a (T, D) action chunk, got %s" % (chunk.shape,))
    if chunk.shape[0] <= order:
        return np.zeros((0, chunk.shape[1]), dtype=np.float64)
    return np.diff(chunk, n=order, axis=0)


def max_abs_jump(chunk: np.ndarray) -> float:
    delta = _time_diff(chunk, 1)
    if delta.size == 0:
        return 0.0
    return float(np.max(np.abs(delta)))


def mean_abs_velocity(chunk: np.ndarray) -> float:
    delta = _time_diff(chunk, 1)
    if delta.size == 0:
        return 0.0
    return float(np.mean(np.abs(delta)))


def mean_abs_jerk(chunk: np.ndarray) -> float:
    delta = _time_diff(chunk, 2)
    if delta.size == 0:
        return 0.0
    return float(np.mean(np.abs(delta)))


def out_of_range_fraction(
    chunk: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    chunk = np.asarray(chunk, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if chunk.ndim != 2:
        raise ValueError("expected a (T, D) action chunk, got %s" % (chunk.shape,))
    if lower.shape != chunk.shape[-1:] or upper.shape != chunk.shape[-1:]:
        raise ValueError("bounds must have shape (%d,)" % chunk.shape[1])
    outside = (chunk < lower) | (chunk > upper)
    return float(np.mean(outside))


def linear_mix(sim: np.ndarray, reference: np.ndarray, weight: float) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if sim.shape != reference.shape:
        raise ValueError(
            "linear mix shape mismatch: %s vs %s" % (sim.shape, reference.shape)
        )
    if not np.isfinite(weight):
        raise ValueError("mix weight must be finite")
    return np.asarray((1.0 - weight) * sim + weight * reference, dtype=np.float32)


def recommend_guidance_scale(rows: Iterable[Mapping[str, float]]) -> float:
    guided = [row for row in rows if row.get("method") == "guided"]
    if not guided:
        raise ValueError("no guided rows available for scale recommendation")
    best = min(guided, key=lambda row: (float(row["demo_mae"]), float(row["scale"])))
    return float(best["scale"])
