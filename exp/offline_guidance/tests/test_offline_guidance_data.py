import sys
from pathlib import Path

import numpy as np
import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _EXP_DIR.parents[1]
sys.path.insert(0, str(_EXP_DIR))
sys.path.insert(0, str(_ROOT_DIR / "eval"))

from offline_guidance_data import (
    ARM_DIM,
    HAND_DIM,
    ACTION_DIM,
    ReplayIndex,
    build_controller_history,
    build_dp_observation,
    chw_float_image,
    hand_chunk,
    select_eval_indices,
)


def test_chw_float_image_converts_uint8_hwc_to_unit_interval_chw():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 127, 0)
    converted = chw_float_image(image)
    assert converted.shape == (3, 2, 3)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted[0, 0, 0], 1.0)
    np.testing.assert_allclose(converted[1, 0, 0], 127 / 255.0, atol=1e-6)
    np.testing.assert_allclose(converted[2, 0, 0], 0.0)


def test_select_eval_indices_skips_history_and_future_action_edges():
    episode_ends = np.array([12, 24], dtype=np.int64)
    indices = select_eval_indices(
        episode_ends,
        n_obs_steps=4,
        chunk_steps=5,
        stride=4,
    )
    assert indices == [
        ReplayIndex(global_idx=3, episode_idx=0, t_in_episode=3),
        ReplayIndex(global_idx=7, episode_idx=0, t_in_episode=7),
        ReplayIndex(global_idx=15, episode_idx=1, t_in_episode=3),
        ReplayIndex(global_idx=19, episode_idx=1, t_in_episode=7),
    ]


def test_select_eval_indices_keeps_only_frames_with_full_history_and_chunk():
    episode_ends = np.array([7], dtype=np.int64)
    assert select_eval_indices(episode_ends, n_obs_steps=4, chunk_steps=5, stride=1) == []


def test_hand_chunk_returns_absolute_hand_targets():
    action = np.arange(5 * ACTION_DIM, dtype=np.float32).reshape(5, ACTION_DIM)
    chunk = hand_chunk(action, start=1, steps=2)
    np.testing.assert_array_equal(chunk, action[1:3, ARM_DIM:])


def test_dp_observation_uses_absolute_ee_and_two_image_steps():
    front = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    wrist = np.zeros((6, 2, 2, 3), dtype=np.uint8)
    front[4, 0, 0] = (10, 0, 0)
    wrist[5, 0, 0] = (0, 20, 0)
    state = np.arange(6 * ACTION_DIM, dtype=np.float32).reshape(6, ACTION_DIM)

    obs = build_dp_observation(front, wrist, state, global_idx=5, n_obs_steps=2)

    assert obs["front_image"].shape == (2, 3, 2, 2)
    assert obs["wrist_image"].shape == (2, 3, 2, 2)
    np.testing.assert_allclose(obs["front_image"][0, 0, 0, 0], 10 / 255.0, atol=1e-6)
    np.testing.assert_allclose(obs["wrist_image"][1, 1, 0, 0], 20 / 255.0, atol=1e-6)
    np.testing.assert_array_equal(obs["ee_pose"], state[4:6, :ARM_DIM])
    np.testing.assert_array_equal(obs["hand_joint"], state[4:6, ARM_DIM:])


def test_controller_history_uses_previous_demo_action_as_target():
    episode_start = 10
    state = np.zeros((16, ACTION_DIM), dtype=np.float32)
    action = np.zeros((16, ACTION_DIM), dtype=np.float32)
    for t in range(10, 16):
        state[t, ARM_DIM:] = t
        action[t, ARM_DIM:] = t + 100

    history = build_controller_history(
        state,
        action,
        episode_start=episode_start,
        global_idx=14,
        n_obs_steps=4,
    )

    assert history.shape == (4, 3 * HAND_DIM)
    np.testing.assert_allclose(history[-1, :HAND_DIM], np.full(HAND_DIM, 14.0))
    np.testing.assert_allclose(history[-1, HAND_DIM:2 * HAND_DIM], np.full(HAND_DIM, 113.0))
    np.testing.assert_allclose(history[-1, 2 * HAND_DIM:], np.full(HAND_DIM, 99.0))
    np.testing.assert_allclose(history[0, HAND_DIM:2 * HAND_DIM], np.full(HAND_DIM, 110.0))


def test_controller_history_uses_qpos_as_target_on_first_episode_frame():
    state = np.zeros((4, ACTION_DIM), dtype=np.float32)
    action = np.ones((4, ACTION_DIM), dtype=np.float32)
    state[0, ARM_DIM:] = 0.3
    history = build_controller_history(
        state,
        action,
        episode_start=0,
        global_idx=0,
        n_obs_steps=1,
    )
    np.testing.assert_allclose(history[0, :HAND_DIM], np.full(HAND_DIM, 0.3))
    np.testing.assert_allclose(history[0, HAND_DIM:2 * HAND_DIM], np.full(HAND_DIM, 0.3))
    np.testing.assert_allclose(history[0, 2 * HAND_DIM:], np.zeros(HAND_DIM))
