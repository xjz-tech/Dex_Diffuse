"""Build low-dimensional observations for the Sim-Hand policy."""

from __future__ import annotations

import numpy as np


HAND_DIM = 22
ACTION_DIM = HAND_DIM
N_OBS_STEPS = 4
N_PRED_ACTION_STEPS = 9
N_ACTION_STEPS = 5
HORIZON = 12
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


def observation_mode_from_dim(obs_dim):
    obs_dim = int(obs_dim)
    for mode, width in OBSERVATION_DIMS.items():
        if width == obs_dim:
            return mode
    raise ValueError("unsupported checkpoint obs_dim=%s" % (obs_dim,))


def expected_policy_spec(obs_dim, n_action_steps=N_ACTION_STEPS):
    obs_dim = int(obs_dim)
    observation_mode_from_dim(obs_dim)
    n_action_steps = int(n_action_steps)
    if not 1 <= n_action_steps <= N_PRED_ACTION_STEPS:
        raise ValueError(
            "n_action_steps must be in [1, %d], got %s"
            % (N_PRED_ACTION_STEPS, n_action_steps)
        )
    return {
        "obs_dim": obs_dim,
        "action_dim": ACTION_DIM,
        "n_obs_steps": N_OBS_STEPS,
        "n_pred_action_steps": N_PRED_ACTION_STEPS,
        "n_action_steps": n_action_steps,
        "horizon": HORIZON,
    }


def validate_policy_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError("policy spec must be a dict, got %r" % (type(spec),))
    try:
        expected = expected_policy_spec(
            spec.get("obs_dim"),
            spec.get("n_action_steps", N_ACTION_STEPS),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported checkpoint spec: %r" % (spec,)) from exc
    if spec != expected:
        raise ValueError(
            "checkpoint spec mismatch: got %r, expected %r" % (spec, expected)
        )
    return expected


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
