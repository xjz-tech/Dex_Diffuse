"""Reconstruct a paired saved-state video; no policy inference is rerun."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.append('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison/eval')
from real_sim_world import World
from isaacgym import gymapi
import cv2

out = Path(sys.argv[1]).resolve()
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 65
idx = 32 + seed - 50
all_names = ['ordinary_1b', 'guided_scale0', 'guide10k_scale25']
all_labels = ['Ordinary 1B', 'Guided sampler, scale 0', '1B + 10k guide, scale 25']
names = sys.argv[3:] or all_names
labels = [all_labels[all_names.index(name)] for name in names]
dest = out / (f'video_seed{seed}' if names == all_names else f'preview_{names[0]}_seed{seed}')
dest.mkdir(exist_ok=True)
ini = np.load(out / names[0] / 'initial_state.npz')
q0, wrist, obj0 = [ini[k][idx] for k in ['q', 'wrist', 'object']]
runs = []
for name in names:
    source = out / name / 'rollouts.npz'
    if not source.exists():
        source = out / name / 'partial.npz'
    with np.load(source) as z:
        drop = int(z['drop_steps'][idx])
        runs.append(dict(q=z['qpos'][:, idx], obj=z['object'][:, idx], drop=drop))
last = max(r['drop'] if r['drop'] >= 0 else len(r['q']) - 1 for r in runs)
w = World(n=1, dt=1e-6, demos=1, camera=True)
params = w.g.get_sim_params(w.s)
params.gravity = gymapi.Vec3(0, 0, 0)
w.g.set_sim_params(w.s, params)
for ai in range(w.g.get_actor_count(w.e.envs[0])):
    actor = w.g.get_actor_handle(w.e.envs[0], ai)
    shapes = w.g.get_actor_rigid_shape_properties(w.e.envs[0], actor)
    for shape in shapes:
        shape.filter = 1
    w.g.set_actor_rigid_shape_properties(w.e.envs[0], actor, shapes)
props = gymapi.CameraProperties()
props.width, props.height, props.horizontal_fov = 640, 480, 55
cam = w.g.create_camera_sensor(w.e.envs[0], props)
w.g.set_camera_location(cam, w.e.envs[0], gymapi.Vec3(*(wrist[:3] + [.12, .55, .15])), gymapi.Vec3(*(wrist[:3] + [-.08, 0, -.12])))
width = 640 * len(names)
writer = cv2.VideoWriter(str(dest / 'raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, 600))
assert writer.isOpened()
errors = []
selected = []
steps = sorted(set([-1, last] + list(range(2, last + 1, 3))))
try:
    for count, step in enumerate(steps):
        frame = np.zeros((600, width, 3), np.uint8)
        title = f'SAVED STATE REPLAY | seed {seed} | clock {(step+1)/30:.2f}s | 1x'
        cv2.putText(frame, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .55 if len(names)==1 else .75, (245, 245, 245), 2)
        for col, (r, label) in enumerate(zip(runs, labels)):
            s = min(step, r['drop']) if r['drop'] >= 0 else step
            q = q0 if s < 0 else r['q'][s]
            obj = obj0 if s < 0 else r['obj'][s]
            render_obj = obj.copy()
            render_obj[7:] = 0
            w.reset(q, wrist, render_obj)
            w.step(1)
            qe = float(abs(w.state()[0] - q).max())
            oe = float(abs(w.objects()[0, :3] - obj[:3]).max())
            assert qe < 1e-4 and oe < 1e-5, (qe, oe)
            errors.append([qe, oe])
            w.g.step_graphics(w.s)
            w.g.render_all_camera_sensors(w.s)
            rgba = w.g.get_camera_image(w.s, w.e.envs[0], cam, gymapi.IMAGE_COLOR)
            img = cv2.cvtColor(np.asarray(rgba).reshape(480, 640, 4)[..., :3], cv2.COLOR_RGB2BGR)
            frame[120:, col*640:(col+1)*640] = img
            failed = r['drop'] >= 0 and step >= r['drop']
            color = (80, 110, 255) if failed else (170, 240, 170)
            distance = float(np.linalg.norm(obj[:3]-obj0[:3])) * 100
            cv2.putText(frame, label, (col*640+14, 61), cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 2)
            cv2.putText(frame, f't={(s+1)/30:.2f}s | object shift {distance:.2f}cm', (col*640+14, 89), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            cv2.putText(frame, '5cm threshold reached; pane frozen' if failed else '5cm displacement threshold', (col*640+14, 114), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
        for _ in range(10 if step in (-1, last) else 1):
            writer.write(frame)
        if count in (0, 3, 10, len(steps)//2, len(steps)-1):
            cv2.imwrite(str(dest / f'frame_{step+1:05d}.png'), frame)
            selected.append(cv2.resize(frame, (384*len(names), 360)))
    cv2.imwrite(str(dest / 'contact_sheet.png'), np.concatenate(selected, axis=0))
    (dest / 'render_validation.json').write_text(json.dumps({'seed':seed, 'frames':len(steps), 'max_q_error_rad':max(x[0] for x in errors), 'max_object_error_m':max(x[1] for x in errors), 'reconstruction':'saved qpos and object pose, gravity/contact disabled, one 1us physics tick to refresh graphics', 'playback':'1x, sampled at 10Hz; failure pane freezes at first >5cm object displacement'}, indent=2))
finally:
    writer.release()
    w.close()
print(dest, flush=True)
