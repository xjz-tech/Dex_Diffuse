"""Select comparable horizontal grasps and prepare full-tail references for all 80 episodes."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull


P = Path(__file__).resolve().parent
DATA = Path('/home/carus/Data/Object_state_data')
OUT = P / 'corrected_direct_vs_reference/all_full_episodes'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
TEMPLATES = [(51, 105), (53, 90)]
HORIZONTAL_DEG = (70, 110)
MIN_CENTER_HEIGHT_M = .22
MIN_REMAINING_ACTIONS = 80
TABLE_Z_M = .2063373668564856
LIFTED_CLEARANCE_M = .01
MAX_WORLD_DISTANCE_FROM_TEMPLATE_M = .20
MESH = Path('/home/carus/Program/dex-controller/data/NOKOV-v3/object/mesh/bulb2.obj')


def frame_data(episode):
    folder = DATA / f'episode_{episode}'
    state = np.load(folder / 'state.npy')
    action = np.load(folder / 'action.npy')
    object_base = np.load(folder / 'obj_state.npy').reshape(-1, 4, 4).astype(float)
    assert len(state) == len(action) == len(object_base)
    row1 = state[:, 3:6].astype(float)
    row1 /= np.linalg.norm(row1, axis=1, keepdims=True)
    row2 = state[:, 6:9].astype(float)
    row2 -= np.sum(row1 * row2, axis=1, keepdims=True) * row1
    row2 /= np.linalg.norm(row2, axis=1, keepdims=True)
    wrist = np.repeat(np.eye(4)[None], len(state), axis=0)
    wrist[:, :3, :3] = np.stack((row1, row2, np.cross(row1, row2)), axis=1)
    wrist[:, :3, 3] = state[:, :3]
    object_wrist = np.linalg.inv(wrist) @ object_base
    angle = np.degrees(np.arccos(np.clip(object_base[:, 2, 1], -1, 1)))
    return state, action, object_base, wrist, object_wrist, angle


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh_vertices = np.asarray([[float(v) for v in line.split()[1:4]]
                                for line in MESH.read_text().splitlines() if line.startswith('v ')])
    mesh_vertices = mesh_vertices[ConvexHull(mesh_vertices).vertices]
    templates = []
    template_world_positions = []
    for episode, frame in TEMPLATES:
        state, _, object_base, _, relative, _ = frame_data(episode)
        templates.append((state[frame, 9:], relative[frame, :3, 3]))
        template_world_positions.append(object_base[frame, :3, 3])
    records = []
    action_series = []
    for episode in range(80):
        state, action, object_base, wrist, relative, angle = frame_data(episode)
        eligible = ((angle >= HORIZONTAL_DEG[0]) & (angle <= HORIZONTAL_DEG[1]) &
                    (object_base[:, 2, 3] >= MIN_CENTER_HEIGHT_M) &
                    (np.arange(len(state)) <= len(state) - 1 - MIN_REMAINING_ACTIONS))
        clearance = (np.min(mesh_vertices @ object_base[:, 2, :3].T, axis=0) +
                     object_base[:, 2, 3] - TABLE_Z_M)
        world_distance = np.min(np.stack([
            np.linalg.norm(object_base[:, :3, 3] - p0, axis=1)
            for p0 in template_world_positions]), axis=0)
        lifted_eligible = (eligible & (clearance >= LIFTED_CLEARANCE_M) &
                           (world_distance <= MAX_WORLD_DISTANCE_FROM_TEMPLATE_M))
        use_lifted = bool(lifted_eligible.any())
        eligible = lifted_eligible if use_lifted else eligible
        scores = []
        qerrors = []
        perrors = []
        for qref, pref in templates:
            qerror = np.sqrt(np.mean((state[:, 9:] - qref)**2, axis=1))
            perror = np.linalg.norm(relative[:, :3, 3] - pref, axis=1)
            score = qerror / .3 + perror / .05
            score[~eligible] = np.inf
            scores.append(score)
            qerrors.append(qerror)
            perrors.append(perror)
        best_template_per_frame = np.argmin(scores, axis=0)
        best_scores = np.min(scores, axis=0)
        start = int(np.argmin(best_scores))
        assert np.isfinite(best_scores[start]), episode
        template_index = int(best_template_per_frame[start])
        end = len(state) - 1
        sub = OUT / f'episode_{episode:02d}'
        sub.mkdir(exist_ok=True)
        np.savez_compressed(sub / 'reference_full.npz',
                            hand_qpos_rad=state[None, start:, 9:],
                            hand_target_rad=action[None, start:end, 9:],
                            recorded_state31=state[None, start:],
                            object_pose_base=object_base[None, start:],
                            wrist_pose_base=wrist[None, start:],
                            object_pose_wrist=relative[None, start:],
                            source_state_frame_indices=np.arange(start, end + 1)[None])
        action_series.append(action[start:end, 9:].copy())
        origin = relative[start]
        prior = np.load(P / 'reference/reference.npz')
        if episode == 53:
            assert start == 90
            assert np.allclose(state[start, 9:], prior['hand_qpos_rad'][1, 0])
            assert np.allclose(origin, prior['object_pose_wrist'][1, 0])
        if episode == 51:
            assert start == 105
            assert np.allclose(state[start, 9:], prior['hand_qpos_rad'][0, 0])
            assert np.allclose(origin, prior['object_pose_wrist'][0, 0])
        rec = dict(episode=episode, source_start_frame=start, source_end_state_frame=end,
                   actions=end-start, template_episode=TEMPLATES[template_index][0],
                   lifted_start=use_lifted,
                   estimated_mesh_clearance_m=float(clearance[start]),
                   object_world_distance_from_template_m=float(world_distance[start]),
                   selection_score=float(best_scores[start]),
                   hand_qpos_rmse_rad=float(qerrors[template_index][start]),
                   object_wrist_position_error_m=float(perrors[template_index][start]),
                   start_world_vertical_angle_deg=float(angle[start]),
                   end_world_vertical_angle_deg=float(angle[end]),
                   start_object_height_m=float(object_base[start, 2, 3]),
                   start_object_position_wrist_m=origin[:3, 3].tolist(),
                   real_wrist_displacement_m=float(np.linalg.norm(state[end, :3] - state[start, :3])),
                   real_wrist_rotation_deg=float(np.degrees((Rotation.from_matrix(wrist[start, :3, :3]).inv() *
                                                             Rotation.from_matrix(wrist[end, :3, :3])).magnitude())),
                   reference=str(sub / 'reference_full.npz'))
        records.append(rec)
    manifest = dict(dataset=str(DATA), episode_count=len(records), templates=TEMPLATES,
                    selection=dict(world_vertical_angle_deg=HORIZONTAL_DEG,
                                   minimum_object_center_height_m=MIN_CENTER_HEIGHT_M,
                                   minimum_remaining_actions=MIN_REMAINING_ACTIONS,
                                   estimated_table_z_m=TABLE_Z_M,
                                   lifted_clearance_m=LIFTED_CLEARANCE_M,
                                   maximum_lifted_object_world_distance_from_template_m=MAX_WORLD_DISTANCE_FROM_TEMPLATE_M,
                                   score='min over episode51 frame105 and episode53 frame90: hand qpos RMSE / 0.3 rad + object-in-wrist position error / 0.05 m; select lowest score among lifted horizontal frames if any, otherwise use the lowest-scoring horizontal frame and mark fallback',
                                   clearance='minimum transformed bulb2.obj vertex height minus approximate real table height; this is not a direct contact measurement'),
                    episodes=records)
    longest = max(len(actions) for actions in action_series)
    packed = np.stack([np.concatenate((actions,
        np.repeat(actions[-1:,:], longest-len(actions), axis=0)))
        for actions in action_series])
    np.savez_compressed(OUT / 'packed_reference.npz', hand_target_rad=packed,
                        action_lengths=np.array([len(actions) for actions in action_series]))
    (OUT / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for page in range(4):
        canvas = Image.new('RGB', (1280, 1300), (16, 23, 31))
        draw = ImageDraw.Draw(canvas)
        for slot, rec in enumerate(records[page*20:(page+1)*20]):
            x = (slot % 4) * 320
            y = (slot // 4) * 260
            image = Image.open(DATA / f'episode_{rec["episode"]}' / 'front' /
                               f'{rec["source_start_frame"]:06d}.png').convert('RGB')
            image = image.resize((320, 240))
            canvas.paste(image, (x, y + 20))
            draw.text((x + 3, y), f'ep{rec["episode"]:02d} f{rec["source_start_frame"]} '
                                   f'{rec["start_world_vertical_angle_deg"]:.0f}°',
                      font=FONT, fill='white')
        canvas.save(OUT / f'selection_page_{page+1}.png')
    print(OUT / 'selection.json')


if __name__ == '__main__':
    main()
