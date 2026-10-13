"""One-shot, fail-closed handoff from this video invocation to existing queue."""
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parents[2]
UNIT = 'guidance-video-until-failure-20260914.service'
INVOCATION = '965228e70d2c45c09707dc90567b64d7'
PAUSE = MAIN / 'eval/experiment_watchdog_20260914/PAUSED'
MARKER = 'AUTO_RESUME_AFTER_VIDEO=' + INVOCATION
QUEUE = 'serial-guidance-data-scaling-seed8.service'
FFMPEG = '/home/carus/miniforge3/envs/dp/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'


def run(command):
    return subprocess.check_output(command, text=True).strip()


def state(status, **extra):
    data = dict(status=status, updated_at=time.time(), video_unit=UNIT, **extra)
    temp = ROOT / 'handoff_state.tmp'
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(ROOT / 'handoff_state.json')
    print(json.dumps(data), flush=True)


def approved():
    if not PAUSE.exists() or PAUSE.read_text().splitlines()[0] != MARKER:
        raise RuntimeError('Pause approval changed; refusing automatic resume')


def validate_terminal(properties, log, rows):
    if properties.get('InvocationID') != INVOCATION:
        raise RuntimeError('Video invocation changed')
    if properties.get('ActiveState') not in ('inactive', 'failed'):
        raise RuntimeError('Video is not stopped')
    if properties.get('ExecMainStatus') not in ('0', '139'):
        raise RuntimeError('Unexpected video exit status')
    if len(rows) != 1 or rows[0].get('reason') != 'failure' or rows[0].get('episode') != 0:
        raise RuntimeError('Expected exactly one first-episode failure, not manual close/crash')
    if '[sim] reached max failure episodes: 1' not in log:
        raise RuntimeError('Missing normal until-failure loop termination')
    paths = re.findall(r'\[record\] STOP \| frames=\d+ duration=[\d.]+s saved=(.+)', log)
    if len(paths) != 1:
        raise RuntimeError('Missing unique finalized recording')
    video = Path(paths[0]).resolve()
    if video.parent != ROOT or video.suffix != '.mp4' or not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError('Invalid finalized video path')
    return video


def main():
    state('waiting_video')
    while True:
        approved()
        props = dict(line.split('=', 1) for line in run(['systemctl', '--user', 'show', UNIT,
            '-p', 'InvocationID', '-p', 'ActiveState', '-p', 'ExecMainStatus']).splitlines())
        if props.get('InvocationID') != INVOCATION:
            raise RuntimeError('Video invocation changed')
        if props.get('ActiveState') in ('inactive', 'failed'):
            break
        time.sleep(15)
    log = (ROOT/'console.log').read_text()
    rows = [json.loads(line) for line in (ROOT/'episodes.jsonl').read_text().splitlines() if line.strip()]
    video = validate_terminal(props, log, rows)
    state('validating_video', video=str(video))
    with (ROOT/'decode_validation.log').open('w') as output:
        subprocess.run([FFMPEG, '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    state('waiting_gpu', video=str(video))
    while run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        approved()
        time.sleep(15)
    approved()
    # Validate the existing monitor's protected inputs before unpausing it.
    import sys
    sys.path.insert(0, str(MAIN/'.worktrees/parallel-guidance-1b-base/eval'))
    import experiment_watchdog as watchdog
    watchdog.verify_integrity(watchdog.fields(str(PAUSE.with_name('config.json'))))
    approved()
    PAUSE.unlink()
    subprocess.run(['systemctl', '--user', 'start', QUEUE,
                    'guidance-experiment-watchdog.timer', 'guidance-experiment-watchdog.service'], check=True)
    active = run(['systemctl', '--user', 'is-active', QUEUE])
    state('resumed', video=str(video), queue=QUEUE, queue_state=active,
          video_exit_status=props['ExecMainStatus'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('blocked', error=repr(exc))
        raise
