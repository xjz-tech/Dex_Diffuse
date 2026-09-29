"""Extend the existing episode-53 reference from frame 90 through the last state."""

from pathlib import Path
import json

import numpy as np


P = Path(__file__).resolve().parent
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
OUT = P / 'corrected_direct_vs_reference/full_episode53'
START = 90


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = np.load(SOURCE / 'state.npy')
    action = np.load(SOURCE / 'action.npy')
    objects = np.load(SOURCE / 'obj_state.npy').reshape(-1, 4, 4)
    assert len(state) == len(action) == len(objects) == 448
    states = state[START:]
    wrist = np.repeat(np.eye(4)[None], len(states), axis=0)
    row1 = states[:, 3:6].astype(float)
    row1 /= np.linalg.norm(row1, axis=1, keepdims=True)
    row2 = states[:, 6:9].astype(float)
    row2 -= np.sum(row1 * row2, axis=1, keepdims=True) * row1
    row2 /= np.linalg.norm(row2, axis=1, keepdims=True)
    wrist[:, :3, :3] = np.stack([row1, row2, np.cross(row1, row2)], axis=1)
    wrist[:, :3, 3] = states[:, :3]
    object_base = objects[START:].astype(float)
    object_wrist = np.linalg.inv(wrist) @ object_base
    arrays = dict(
        hand_qpos_rad=states[None, :, 9:],
        hand_target_rad=action[None, START:-1, 9:],
        recorded_state31=states[None],
        object_pose_base=object_base[None],
        wrist_pose_base=wrist[None],
        object_pose_wrist=object_wrist[None],
        source_state_frame_indices=np.arange(START, len(state))[None],
    )
    original = np.load(P / 'reference/reference.npz')
    for key in arrays:
        assert np.allclose(arrays[key][0, :original[key].shape[1]], original[key][1], atol=1e-6), key
    np.savez_compressed(OUT / 'reference_full.npz', **arrays)
    (OUT / 'reference_metadata.json').write_text(json.dumps(dict(
        source_episode=53, source_states=[START, len(state) - 1],
        source_actions=[START, len(state) - 2],
        action_count=len(action[START:-1]),
        initial_reference_matches_previous=True,
        timing='30 Hz nominal; source has no timestamps',
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
