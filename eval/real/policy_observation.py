"""Observation construction for the 22/66-D Sim-Hand policies."""

from __future__ import annotations

import numpy as np


HAND_DIM = 22
QPOS_OBSERVATION = "qpos"
QPOS_TARGET_RESIDUAL_OBSERVATION = "qpos-target-residual"
OBSERVATION_DIMS = {
    QPOS_OBSERVATION: HAND_DIM,
    QPOS_TARGET_RESIDUAL_OBSERVATION: 3 * HAND_DIM,
}


def observation_dim(mode: str) -> int:
    try:
        return OBSERVATION_DIMS[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported observation mode: {mode!r}") from exc


def compose_policy_observation(
    qpos: np.ndarray,
    target_before: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return qpos or [qpos, target_before, target_before - qpos]."""
    qpos = np.asarray(qpos, dtype=np.float32)
    target_before = np.asarray(target_before, dtype=np.float32)
    if qpos.shape != (HAND_DIM,) or target_before.shape != (HAND_DIM,):
        raise ValueError(
            f"qpos and target_before must both have shape ({HAND_DIM},), "
            f"got {qpos.shape} and {target_before.shape}"
        )
    if not np.isfinite(qpos).all() or not np.isfinite(target_before).all():
        raise ValueError("qpos/target_before contains NaN or Inf")

    if mode == QPOS_OBSERVATION:
        return np.ascontiguousarray(qpos)
    if mode == QPOS_TARGET_RESIDUAL_OBSERVATION:
        return np.ascontiguousarray(
            np.concatenate((qpos, target_before, target_before - qpos))
        )
    raise ValueError(f"unsupported observation mode: {mode!r}")
