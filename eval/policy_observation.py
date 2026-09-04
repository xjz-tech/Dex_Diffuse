"""Build low-dimensional observations for the Sim-Hand policy."""

from __future__ import annotations

import numpy as np


HAND_DIM = 22
QPOS_OBSERVATION = "qpos"
QPOS_TARGET_RESIDUAL_OBSERVATION = "qpos-target-residual"
OBSERVATION_DIMS = {
    QPOS_OBSERVATION: HAND_DIM,
    QPOS_TARGET_RESIDUAL_OBSERVATION: 3 * HAND_DIM,
}


def observation_dim(mode):
    try:
        return OBSERVATION_DIMS[mode]
    except KeyError as exc:
        raise ValueError("unsupported observation mode: %r" % (mode,)) from exc


def compose_policy_observation(qpos, target_before, mode):
    """Return ``qpos`` or ``[qpos, target_before, target_before - qpos]``."""
    qpos = np.asarray(qpos, dtype=np.float32)
    target_before = np.asarray(target_before, dtype=np.float32)
    if qpos.shape != target_before.shape:
        raise ValueError(
            "qpos and target_before shapes differ: %r vs %r"
            % (qpos.shape, target_before.shape)
        )
    if qpos.ndim < 1 or qpos.shape[-1] != HAND_DIM:
        raise ValueError(
            "expected qpos/target_before last dimension to be %d, got %r"
            % (HAND_DIM, qpos.shape)
        )

    if mode == QPOS_OBSERVATION:
        return np.ascontiguousarray(qpos)
    if mode == QPOS_TARGET_RESIDUAL_OBSERVATION:
        return np.ascontiguousarray(
            np.concatenate(
                (qpos, target_before, target_before - qpos),
                axis=-1,
            )
        )
    raise ValueError("unsupported observation mode: %r" % (mode,))
