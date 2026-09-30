"""One-shot queued recording. Invoked by the task heartbeat; never shares GPU."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
UNIT = 'video-10k-1b-wait0-5min-20260915'
FFMPEG = '/home/carus/.cache/uv/archive-v0/dvUdjUJ8JFV3iaK2/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
PREDECESSORS = [
    PROJECT/'eval/hold_runs/20260915_serial_100k_1m_scale10_50_seed8_3k/state.json',
    PROJECT/'eval/hold_runs/20260915_serial_100k_1m_epochs20_100_seed8_3k/state.json',
    PROJECT/'eval/hold_runs/20260915_serial_1k_guide_seed8_3k/state.json',
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def state(status, **kwargs):
    data = dict(status=status, updated_at=time.time(), **kwargs)
    tmp = ROOT/'state.tmp'
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(ROOT/'state.json')
    print(json.dumps(data), flush=True)


def readiness():
    for path in PREDECESSORS:
        status = read(path).get('status', 'missing')
        if status != 'complete':
            return False, f'{path.parent.name}: {status}'
    for unit in ('serial-guide-epochs-seed8-20260915.service', 'serial-small-guide-seed8-20260915.service'):
        result = subprocess.run(['systemctl','--user','is-active',unit], capture_output=True, text=True)
        if result.stdout.strip() in ('active','activating','deactivating','reloading'):
            return False, f'{unit} has not exited'
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'], capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        return False, 'GPU compute processes still present: '+gpu.stdout.strip()
    return True, 'All predecessor queues complete; GPU idle'


def validate_manifest():
    manifest = read(ROOT/'manifest.json')
    for path, expected in manifest['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Prepared source changed: '+path)
    for path, expected in manifest['checkpoint_stat'].items():
        info = Path(path).stat()
        if [info.st_size, info.st_mtime_ns] != expected:
            raise RuntimeError('Checkpoint changed since queueing: '+path)


def run_recording():
    with (ROOT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(ROOT/'state.json').get('status') != 'queued':
            raise RuntimeError('Refusing duplicate or unreviewed retry')
        ready, reason = readiness()
        if not ready:
            print(reason, flush=True)
            return
        validate_manifest()
        state('running', pid=os.getpid(), unit=UNIT)
        try:
            with (ROOT/'console.log').open('w') as log:
                result = subprocess.run(['bash', str(ROOT/'record.sh')], stdout=log, stderr=subprocess.STDOUT)
            summary = read(ROOT/'wait0_summary.json')
            if summary.get('stop_reason') not in ('video_time_cap','first_failure'):
                raise RuntimeError('Missing normal recording stop: '+str(summary))
            if result.returncode not in (0,139):
                raise RuntimeError('Recorder exited with '+str(result.returncode))
            videos = list(ROOT.glob('*.mp4'))
            if len(videos) != 1:
                raise RuntimeError('Expected one finalized MP4')
            video = videos[0]
            decode = subprocess.run([FFMPEG,'-hide_banner','-v','error','-xerror','-i',str(video),'-map','0:v:0','-f','null','-'], capture_output=True, text=True)
            (ROOT/'decode.log').write_text(decode.stderr)
            if decode.returncode:
                raise RuntimeError('Full video decode failed')
            probe = subprocess.run([FFMPEG,'-hide_banner','-i',str(video)], capture_output=True,text=True)
            match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)',probe.stderr)
            if not match:
                raise RuntimeError('Video duration unavailable')
            h,m,s = map(float,match.groups())
            duration = h*3600+m*60+s
            if not 0 < duration <= 300.01:
                raise RuntimeError('Video exceeds five-minute cap')
            (ROOT/'video_metadata.txt').write_text(probe.stderr)
            state('complete', video=str(video), duration_seconds=duration,
                  recorder_exit=result.returncode, full_decode='passed',
                  stop_reason=summary['stop_reason'],
                  action_steps=summary['action_steps'], hold_steps=summary['hold_steps'])
        except Exception as exc:
            state('failed', error=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run_recording()
    else:
        current = read(ROOT/'state.json')
        ready, reason = readiness()
        print(json.dumps(dict(state=current, ready=ready, reason=reason)),flush=True)
        if args.start and ready and current.get('status') == 'queued':
            validate_manifest()
            subprocess.run(['systemd-run','--user','--unit='+UNIT,'--collect',
                '--property=Restart=no','--property=KillMode=control-group',
                '--property=Environment=PYTHONDONTWRITEBYTECODE=1',
                '--property=StandardOutput=append:'+str(ROOT/'runner.log'),
                '--property=StandardError=inherit',
                '/usr/bin/python3','-B',str(Path(__file__).resolve()),'--run'],check=True)
