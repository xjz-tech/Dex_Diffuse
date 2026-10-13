"""Render four-panel, full-tail videos for selected distinct Object_state_data episodes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
ROOT = BASE / 'corrected_direct_vs_reference/all_full_episodes'
DATA = Path('/home/carus/Data/Object_state_data')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
METHODS = [('direct', '原始动作直接仿真', 'direct_front.mp4'),
           ('guide2_exec2', '10B guide2 / exec2 · scale50', 'guided_front.mp4'),
           ('guide2_exec1', '10B guide2 / exec1 · scale50', 'guided_front.mp4')]


def render(episode, root=ROOT):
    selected = json.loads((root / 'selection.json').read_text())['episodes'][episode]
    source = DATA / f'episode_{episode}' / 'front'
    folder = root / f'episode_{episode:02d}/video_full'
    outputs = []
    for method, title, filename in METHODS:
        run_folder = folder / method
        summary = json.loads((run_folder / 'summary.json').read_text())
        trace = json.loads((run_folder / 'trace.json').read_text())
        cap = cv2.VideoCapture(str(run_folder / filename))
        assert cap.isOpened(), run_folder
        assert not summary.get('stop_on_native_failure', False)
        assert summary['source_episode'] == episode
        assert summary['steps'] == summary.get('intended_steps', summary['steps'])
        assert summary['steps']['settle'] == summary['steps']['hold'] == 60
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == (summary['steps']['action'] + 60)
        outputs.append(dict(method=method, title=title, summary=summary,
                            trace=trace, cap=cap, last=-1, frame=None))
    n = selected['actions']
    guided_length = (n - 1) * 2 + 1
    assert [run['summary']['steps']['action'] for run in outputs] == [n, guided_length, guided_length]
    assert int(outputs[0]['summary']['source_start_frame']) == selected['source_start_frame']
    video = folder / f'episode_{episode:02d}_real_direct_guide2exec2_exec1_full.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 1088))
    assert writer.isOpened(), video
    last_source = None
    real = None
    try:
        for shown in range(guided_length + 60):
            in_action = shown < guided_length
            source_frame = (selected['source_start_frame'] + shown // 2
                            if in_action else selected['source_end_state_frame'])
            if source_frame != last_source:
                real = Image.open(source / f'{source_frame:06d}.png').convert('RGB')
                assert real.size == (640, 480)
                last_source = source_frame
            canvas = Image.new('RGB', (1280, 1088), (20, 28, 39))
            draw = ImageDraw.Draw(canvas)
            canvas.paste(real, (640, 64))
            draw.rectangle((640, 0, 1279, 63), fill=(25, 36, 49))
            draw.text((648, 0), f'真机原视频 · episode {episode:02d}', font=FONT, fill='white')
            draw.text((648, 36), f'源帧 {source_frame} | {"参考动作" if in_action else "末态定格"}',
                      font=SMALL, fill=(194, 229, 212))
            for run, (x, y) in zip(outputs, ((0, 0), (0, 544), (640, 544))):
                count = run['summary']['steps']['action']
                index = (min(shown // 2, count - 1) if run['method'] == 'direct' else shown)
                if not in_action:
                    index = count + shown - guided_length
                while run['last'] < index:
                    ok, frame = run['cap'].read()
                    assert ok and frame.shape == (544, 640, 3), (episode, run['method'], index)
                    run['last'] += 1
                    run['frame'] = frame
                panel = cv2.cvtColor(run['frame'], cv2.COLOR_BGR2RGB)
                canvas.paste(Image.fromarray(panel), (x, y))
                row = run['trace'][60 + index]
                nearest_vertical = min(row['vertical_error_deg'], 180-row['vertical_error_deg'])
                draw.rectangle((x, y, x + 639, y + 63), fill=(25, 36, 49))
                draw.text((x + 8, y), run['title'], font=FONT, fill='white')
                draw.text((x + 8, y + 36),
                          f'{"动作" if in_action else "末态保持"} | 仿真 {index/30:.2f}s | '
                          f'轴距竖直 {nearest_vertical:.1f}° | failure {row["native_failure"]}',
                          font=SMALL, fill=(255, 155, 135) if row['native_failure'] else (194, 229, 212))
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
            if shown in (0, guided_length - 1, guided_length + 59):
                canvas.save(folder / f'episode_{episode:02d}_frame_{shown:04d}.png')
    finally:
        writer.release()
        for run in outputs:
            run['cap'].release()
    verify = cv2.VideoCapture(str(video))
    frame_count = int(verify.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = verify.get(cv2.CAP_PROP_FPS)
    ok, first = verify.read()
    verify.release()
    assert ok and first.shape == (1088, 1280, 3)
    assert frame_count == guided_length + 60 and abs(fps-30) < 1e-6
    (folder / 'video_verification.json').write_text(json.dumps(dict(
        episode=episode, video=str(video), frames=frame_count, fps=fps,
        duration_s=frame_count/fps, source_start_frame=selected['source_start_frame'],
        source_end_state_frame=selected['source_end_state_frame'],
        panels=['direct full', 'real front', 'guide2/exec2 full', 'guide2/exec1 full'],
        alignment='By source action progress; direct and real frames repeat twice; 60-step terminal hold. '
                  'Native failure is displayed but never stops any source action.'), indent=2) + '\n')
    print(video, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('episodes', type=int, nargs='+')
    args = parser.parse_args()
    for episode in args.episodes:
        assert 0 <= episode < 80
        render(episode, args.root)


if __name__ == '__main__':
    main()
