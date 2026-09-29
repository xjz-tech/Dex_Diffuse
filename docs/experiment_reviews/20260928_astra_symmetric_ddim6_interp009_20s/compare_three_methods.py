"""Render the three requested Astra methods in one synchronized video."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from analyze import FFMPEG, LABELS, P, load, panel


METHODS = ["direct", "edit015_ddim4", "guidance50_g4_ddim4"]
LABELS.update({
    "edit015_ddim4": "Edit noise 0.15 / DDIM4",
    "guidance50_g4_ddim4": "Guidance scale50 / guide4 / DDIM4",
})


def main():
    reviews = json.loads((P / "physical_reviews.json").read_text())
    with np.load(P / "runs/left_direct/initial_state.npz") as state:
        baseline = {key: state[key].copy() for key in state.files}

    data = {
        direction: [load(direction, method, reviews, baseline) for method in METHODS]
        for direction in ("left", "right")
    }

    for direction in data:
        direct_model = json.loads((data[direction][0][0] / "model.json").read_text())
        for folder, summary, _ in data[direction]:
            model = json.loads((folder / "model.json").read_text())
            assert model["checkpoint"] == direct_model["checkpoint"]
            assert model["checkpoint_info"] == direct_model["checkpoint_info"]
            assert summary["steps"] == 600
            assert summary["execution_steps"] == 2

    output = P / "astra_direct_edit015_guidance50_left_right_1x.mp4"
    dt = data["left"][0][1]["control_dt"]
    command = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1020",
        "-r", str(1 / dt), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for step in range(1, 601):
        rows = []
        for direction in ("left", "right"):
            rows.append(cv2.hconcat([
                panel(*item, step, direction, method)
                for item, method in zip(data[direction], METHODS)
            ]))
        frame = cv2.vconcat(rows)
        process.stdin.write(frame.tobytes())
        if step in (1, 300, 600):
            cv2.imwrite(str(P / f"three_methods_step{step:03d}.jpg"), frame)
    process.stdin.close()
    assert process.wait() == 0

    capture = cv2.VideoCapture(str(output))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 600
    assert abs(frames / fps - 600 * dt) < 0.002

    analysis = {
        "layout": "top row left; bottom row right; columns Direct / Edit noise0.15 / Guidance scale50 guide4",
        "methods": METHODS,
        "video": {
            "path": str(output),
            "frames": frames,
            "fps": fps,
            "video_duration_s": frames / fps,
            "simulation_duration_s": 600 * dt,
            "speed_ratio": 600 * dt / (frames / fps),
            "frozen_frames_added": 0,
        },
        "summaries": [summary for direction in data for _, summary, _ in data[direction]],
        "reference_hashes": {
            direction: hashlib.sha256((P / f"{direction}_reference.npz").read_bytes()).hexdigest()
            for direction in data
        },
    }
    (P / "three_methods_video_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n"
    )

    page = """<!doctype html><meta charset=\"utf-8\"><title>Astra 三种方法左右转对比</title>
<style>body{margin:24px;background:#141922;color:#edf2fa;font:17px system-ui}h1{font-size:25px}video{width:100%;max-height:80vh;background:#000}p{line-height:1.6;color:#c4d1df}a{color:#9ed5ff}button{padding:10px 18px;background:#a8d5ff;border:0;border-radius:7px}</style>
<h1>Astra：Direct / Edit / Guidance 左右转同步对比</h1>
<p>上排左转，下排右转；从左到右为 Direct、Edit（noise 0.15，DDIM4，exec2）、Guidance（scale50，guide4，DDIM4，exec2）。170g、摩擦2.2，同一套 Astra reference。</p>
<video id=\"movie\" controls playsinline preload=\"metadata\" src=\"VIDEO\"></video>
<p id=\"status\">完整20秒 · 1倍速</p><button id=\"restart\">从头播放（1×）</button>
<p><a href=\"VIDEO\" download>下载六宫格录像</a> · <a href=\"COMPARISON_TABLE.md\">查看三组表格</a></p>
<script>const v=document.querySelector('#movie');function u(){document.querySelector('#status').textContent=`${v.paused?'暂停':'播放中'} · ${v.currentTime.toFixed(2)} / ${Number.isFinite(v.duration)?v.duration.toFixed(2):'--'}秒 · ${v.playbackRate}倍速`;}['loadedmetadata','timeupdate','play','pause','ratechange'].forEach(e=>v.addEventListener(e,u));document.querySelector('#restart').onclick=()=>{v.currentTime=0;v.playbackRate=1;v.play()};</script>""".replace("VIDEO", output.name)
    (P / "three_methods.html").write_text(page)

    downloads = Path("/home/carus/Downloads/astra_symmetric_ddim6_20s_20260928")
    downloads.mkdir(exist_ok=True)
    for name in (
        output.name,
        "three_methods.html",
        "three_methods_video_analysis.json",
        "COMPARISON_TABLE.md",
    ):
        shutil.copy2(P / name, downloads / name)

    print(json.dumps(analysis["video"], indent=2))


if __name__ == "__main__":
    main()
