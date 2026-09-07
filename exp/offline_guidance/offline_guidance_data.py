"""Read DP replay-buffer frames and build offline evaluation observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_observation import (
    HAND_DIM,
    QPOS_TARGET_RESIDUAL_OBSERVATION,
    compose_policy_observation,
)


ARM_DIM = 9
ACTION_DIM = ARM_DIM + HAND_DIM


@dataclass(frozen=True)
class ReplayIndex:
    global_idx: int
    episode_idx: int
    t_in_episode: int


def chw_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected HWC RGB image, got %s" % (image.shape,))
    return np.moveaxis(image.astype(np.float32), -1, 0) / 255.0


def episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("episode_ends must be a non-empty 1-D array")
    starts = np.concatenate(([0], ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def select_eval_indices(
    episode_ends: np.ndarray,
    n_obs_steps: int = 4,
    chunk_steps: int = 5,
    stride: int = 20,
    max_samples: int | None = None,
) -> list[ReplayIndex]:
    if n_obs_steps <= 0 or chunk_steps <= 0 or stride <= 0:
        raise ValueError("n_obs_steps, chunk_steps, and stride must be positive")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples cannot be negative")

    indices: list[ReplayIndex] = []
    for episode_idx, (start, end) in enumerate(episode_bounds(episode_ends)):
        first = start + n_obs_steps - 1
        last = end - chunk_steps
        if last < first:
            continue
        for global_idx in range(first, last + 1, stride):
            indices.append(
                ReplayIndex(
                    global_idx=int(global_idx),
                    episode_idx=int(episode_idx),
                    t_in_episode=int(global_idx - start),
                )
            )
            if max_samples is not None and len(indices) >= max_samples:
                return indices
    return indices


def hand_chunk(array: np.ndarray, start: int, steps: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError("expected (T, %d) array, got %s" % (ACTION_DIM, array.shape))
    if steps <= 0:
        raise ValueError("steps must be positive")
    stop = start + steps
    if start < 0 or stop > array.shape[0]:
        raise ValueError("hand chunk [%d:%d] is out of range for T=%d" % (start, stop, array.shape[0]))
    return np.ascontiguousarray(array[start:stop, ARM_DIM:], dtype=np.float32)


def build_dp_observation(
    front_image,
    wrist_image,
    state: np.ndarray,
    global_idx: int,
    n_obs_steps: int = 2,
) -> dict[str, np.ndarray]:
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    start = global_idx - n_obs_steps + 1
    stop = global_idx + 1
    if start < 0:
        raise ValueError("DP observation at %d needs %d frames" % (global_idx, n_obs_steps))
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[1] != ACTION_DIM:
        raise ValueError("state must have shape (T, %d)" % ACTION_DIM)
    if stop > state.shape[0]:
        raise ValueError("DP observation at %d exceeds state length %d" % (global_idx, state.shape[0]))

    front = np.asarray(front_image[start:stop])
    wrist = np.asarray(wrist_image[start:stop])
    return {
        "front_image": np.stack([chw_float_image(frame) for frame in front], axis=0),
        "wrist_image": np.stack([chw_float_image(frame) for frame in wrist], axis=0),
        "ee_pose": np.ascontiguousarray(state[start:stop, :ARM_DIM], dtype=np.float32),
        "hand_joint": np.ascontiguousarray(state[start:stop, ARM_DIM:], dtype=np.float32),
    }


def build_controller_history(
    state: np.ndarray,
    action: np.ndarray,
    episode_start: int,
    global_idx: int,
    n_obs_steps: int = 4,
) -> np.ndarray:
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    start = global_idx - n_obs_steps + 1
    if start < episode_start:
        raise ValueError(
            "controller history at %d needs %d in-episode frames"
            % (global_idx, n_obs_steps)
        )
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != ACTION_DIM:
        raise ValueError("state/action must both have shape (T, %d)" % ACTION_DIM)

    frames = []
    for t in range(start, global_idx + 1):
        qpos = state[t, ARM_DIM:]
        if t > episode_start:
            target_before = action[t - 1, ARM_DIM:]
        else:
            target_before = qpos
        frames.append(
            compose_policy_observation(
                qpos,
                target_before,
                QPOS_TARGET_RESIDUAL_OBSERVATION,
            )
        )
    return np.stack(frames, axis=0)
