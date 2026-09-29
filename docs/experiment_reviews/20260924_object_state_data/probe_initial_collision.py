"""Check hand collision-mesh vertices inside the bulb collision mesh.

This is a geometric diagnostic, not a substitute for the PhysX contact solver.
"""

from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'initial_pose_diagnostic'
CONTROLLER = Path('/home/carus/Program/dex-controller')
HAND = CONTROLLER / 'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf'
BULB = CONTROLLER / 'data/NOKOV-v3/object/mesh/bulb1_col.obj'
sys.path.insert(0, '/home/carus/Program/dex_sim_bench/vendor')
from sharpa_fk_standalone.sharpa_fk import SharpAFK


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def main():
    ref = np.load(ROOT / 'reference/reference.npz')
    ini = np.load(ROOT / 'repeat/initial_state.npz')
    trace = np.load(ROOT / 'repeat/trajectory.npz')
    bulb = trimesh.load(BULB, force='mesh', process=False)
    assert bulb.is_watertight
    scale = float(ini['object_scale'][467])
    bulb.vertices *= scale
    bulb_tree = cKDTree(bulb.vertices)
    links = []
    for link in ET.parse(HAND).getroot().findall('link'):
        for collider in link.findall('collision'):
            asset = collider.find('geometry/mesh')
            if asset is None:
                continue
            mesh = trimesh.load((HAND.parent / asset.get('filename')).resolve(), force='mesh', process=False)
            links.append((link.get('name'), mesh.vertices.copy(), origin(collider.find('origin'))))
    fk = SharpAFK(HAND)
    source_pose_rows = ref['object_pose_wrist'][1, 0]
    state = ref['recorded_state31'][1, 0]
    first = state[3:6] / np.linalg.norm(state[3:6])
    second = state[6:9] - np.dot(first, state[6:9]) * first
    second /= np.linalg.norm(second)
    source_wrist_columns = np.eye(4)
    source_wrist_columns[:3, :3] = np.column_stack([first, second, np.cross(first, second)])
    source_wrist_columns[:3, 3] = state[:3]
    source_pose_columns = np.linalg.inv(source_wrist_columns) @ ref['object_pose_base'][1, 0]
    wrist = ini['wrist'][467, :7]
    wrist_rot = Rotation.from_quat(wrist[3:])
    imported = np.eye(4)
    imported[:3, 3] = wrist_rot.inv().apply(ini['object'][467, :3] - wrist[:3])
    imported[:3, :3] = (wrist_rot.inv() * Rotation.from_quat(ini['object'][467, 3:7])).as_matrix()
    settled = np.eye(4)
    settled[:3, 3] = trace['relative_position'][59, 467]
    settled[:3, :3] = Rotation.from_quat(trace['relative_quaternion'][59, 467]).as_matrix()
    cases = dict(source_rows=(ref['hand_qpos_rad'][1, 0], source_pose_rows),
                 source_columns=(ref['hand_qpos_rad'][1, 0], source_pose_columns),
                 imported=(ini['q'][467], imported), settled=(trace['q'][59, 467], settled))
    cases['source_rows_episode51_f105'] = (
        ref['hand_qpos_rad'][0, 0], ref['object_pose_wrist'][0, 0])
    for dx in [-.03, -.04, -.05, -.06]:
        pose = source_pose_rows.copy()
        pose[0, 3] += dx
        cases[f'exploratory_ep53_wrist_x_{dx:+.2f}m'] = (ref['hand_qpos_rad'][1, 0], pose)
    for relative_index in [30, 75]:
        state = ref['recorded_state31'][1, relative_index]
        first = state[3:6] / np.linalg.norm(state[3:6])
        second = state[6:9] - np.dot(first, state[6:9]) * first
        second /= np.linalg.norm(second)
        wrist_columns = np.eye(4)
        wrist_columns[:3, :3] = np.column_stack([first, second, np.cross(first, second)])
        wrist_columns[:3, 3] = state[:3]
        cases[f'source_rows_f{90+relative_index}'] = (ref['hand_qpos_rad'][1, relative_index],
                                                       ref['object_pose_wrist'][1, relative_index])
        cases[f'source_columns_f{90+relative_index}'] = (ref['hand_qpos_rad'][1, relative_index],
                                                          np.linalg.inv(wrist_columns) @ ref['object_pose_base'][1, relative_index])
    results = {}
    for state, (q, object_pose) in cases.items():
        poses = dict(zip(fk.link_names, fk.link_poses(q, fk.link_names, joint_limit_mode='ignore')))
        into_object = np.linalg.inv(object_pose)
        hits = []
        nearest_vertex_distance = float('inf')
        nearest_link = None
        for name, vertices, collider_origin in links:
            link_pose = into_object @ poses[name] @ collider_origin
            vertices_local = vertices @ link_pose[:3, :3].T + link_pose[:3, 3]
            distance = float(bulb_tree.query(vertices_local, workers=-1)[0].min())
            if distance < nearest_vertex_distance:
                nearest_vertex_distance, nearest_link = distance, name
            # Only evaluate vertices in a generously expanded object AABB.
            near = np.all((vertices_local >= bulb.bounds[0] - .005) &
                          (vertices_local <= bulb.bounds[1] + .005), axis=1)
            if not near.any():
                continue
            distances = trimesh.proximity.signed_distance(bulb, vertices_local[near])
            inside = distances > 1e-4
            if inside.any():
                hits.append(dict(link=name, sampled_vertices_inside=int(inside.sum()),
                                 max_inside_depth_m=float(distances[inside].max())))
        results[state] = dict(links_with_inside_vertices=len(hits),
                              vertices_inside=sum(x['sampled_vertices_inside'] for x in hits),
                              maximum_vertex_depth_m=max((x['max_inside_depth_m'] for x in hits), default=0.),
                              nearest_collision_mesh_vertex_distance_m=nearest_vertex_distance,
                              nearest_link=nearest_link,
                              affected_links=hits)
    (OUT / 'collision_probe.json').write_text(json.dumps(dict(
        method='hand URDF collision mesh vertices tested inside watertight bulb1_col.obj after saved transforms',
        object_scale=scale, positive_signed_distance_threshold_m=1e-4,
        caution='Vertex sampling misses some intersections; this is neither PhysX penetration depth nor contact-force measurement.',
        states=results,
    ), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
