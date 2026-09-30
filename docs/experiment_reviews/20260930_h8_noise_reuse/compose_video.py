"""Compose existing actual videos only. Never reruns a simulation to fill gaps."""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from experiment import read, write


def frame_indices(action_steps, raw_steps, progress):
    # Input frames: 0 import; 1..60 settle; 61.. actions; then 60 hold.
    yield from [(0, 0, 'import', 0., 0, 0)] * 30
    for j in range(60):
        yield 1+j, 1+j, 'settle', 0., 0, 0
    yield from [(60, 60, 'action start', 0., 0, 0)] * 30
    for j, value in enumerate(progress):
        raw_count = min(raw_steps, int(np.floor(value)))
        yield 60+raw_count, 61+j, 'action', float(value), raw_count, j+1
    for j in range(60):
        yield 61+raw_steps+j, 61+action_steps+j, 'hold', float(raw_steps), raw_steps, action_steps


class Video:
    def __init__(self, folder, name):
        self.cap = cv2.VideoCapture(str(folder / name))
        if not self.cap.isOpened():
            raise FileNotFoundError(f'Missing actual recording: {folder / name}; no automatic replay')
        trace = read(folder / 'trace.json')
        assert int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(trace) + 1
        assert abs(self.cap.get(cv2.CAP_PROP_FPS) - 30) < 1e-6
        self.index, self.frame = -1, None

    def at(self, index):
        assert index >= self.index
        while self.index < index:
            ok, self.frame = self.cap.read()
            if not ok:
                raise RuntimeError('Truncated actual recording')
            assert self.frame.shape == (544, 1280, 3)
            self.index += 1
        # Keep the front view plus a wide inset showing the table.
        panel = self.frame[:, :640].copy()
        panel[376:520, 432:624] = cv2.resize(self.frame[64:, 640:], (192, 144))
        return panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    a, b = read(args.run / 'summary.json'), read(args.raw / 'summary.json')
    assert a['video_recorded'] and b['video_recorded']
    assert b['mode'] == 'direct' and b['reference_interpolation_threshold'] is None
    for k in ('source_episode', 'source_start_frame', 'mass_kg', 'friction', 'reference',
              'native_protocol', 'object_size_multiplier', 'settle_target_source'):
        assert a[k] == b[k], k
    with np.load(args.run / 'initial_state.npz') as x, np.load(args.raw / 'initial_state.npz') as y:
        assert set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    progress = np.load(args.run / 'reference_progress.npy')
    assert len(progress) == a['steps']['action']
    assert progress[0] == 1 and progress[-1] == b['steps']['action']
    raw, run = Video(args.raw, 'direct_front.mp4'), Video(args.run, 'guided_front.mp4')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    timeline = list(frame_indices(a['steps']['action'], b['steps']['action'], progress))
    try:
        with imageio.get_writer(str(args.out), fps=30, codec='libx264', macro_block_size=16,
            ffmpeg_params=['-preset', 'fast', '-threads', '2', '-movflags', '+faststart']) as writer:
            for ri, ai, phase, reference, raw_count, count in timeline:
                canvas = np.zeros((608, 1280, 3), np.uint8)
                canvas[64:, :640] = raw.at(ri)
                canvas[64:, 640:] = run.at(ai)
                label = f'ep{a["source_episode"]} | raw reference (left) vs h8 rho={a["prior"]["editor"]["rho"]} (right)'
                cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
                cv2.putText(canvas, f'{phase} | ref progress {reference:g} | actual controls raw={raw_count}, edit={count}',
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
                writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        raw.cap.release()
        run.cap.release()
    write(args.out.with_suffix('.json'), dict(run=str(args.run), raw=str(args.raw),
        frames=len(timeline), fps=30, alignment='1-based original reference progress; floor and hold raw frame at midpoints',
        raw_actual_steps=b['steps']['action'], edit_actual_steps=a['steps']['action']))


if __name__ == '__main__':
    main()
