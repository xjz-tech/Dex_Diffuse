from pathlib import Path
import json

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


P = Path(__file__).resolve().parent
GUIDED = P / "repeat"
DIRECT = P / "direct_case467"
CASE = 467
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)
small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)

guided = np.load(GUIDED / "trajectory.npz")
direct = np.load(DIRECT / "trajectory.npz")
reference = np.load(P / "reference/reference.npz")["hand_target_rad"][1]
guided_initial = np.load(GUIDED / "initial_state.npz")
direct_initial = np.load(DIRECT / "initial_state.npz")

initial_exact = {key: bool(np.array_equal(guided_initial[key], direct_initial[key])) for key in guided_initial.files}
case_static_exact = {
    key: bool(np.array_equal(guided[key][:60, CASE], direct[key][:60, CASE]))
    for key in guided.files
    if guided[key].ndim >= 2 and guided[key].shape[1] == 486
}
direct_commands_exact = bool(np.array_equal(direct["command"][60:135, CASE], reference))
random_forces_exact = bool(np.array_equal(guided["applied_forces"][:, CASE], direct["applied_forces"][:, CASE]))
assert all(initial_exact.values())
assert all(case_static_exact.values())
assert direct_commands_exact
assert random_forces_exact

caps = {
    "direct": cv2.VideoCapture(str(DIRECT / "case467_clear.mp4")),
    "guided": cv2.VideoCapture(str(GUIDED / "case467_clear.mp4")),
}
labels = {
    "direct": "原始动作直接回放（完全不采用 prior 输出）",
    "guided": "10B prior + 原始轨迹引导（scale 25）",
}
runs = {"direct": direct, "guided": guided}
output = DIRECT / "prior_vs_direct_slow.mp4"
with imageio.get_writer(str(output), fps=15, codec="libx264", quality=8) as writer:
    for video_index in range(105):
        trajectory_index = video_index + 60
        panels = []
        for name in ["direct", "guided"]:
            ok, frame = caps[name].read()
            assert ok
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1279, 63), fill=(14, 19, 25))
            data = runs[name]
            failed = bool(data["failure"][: trajectory_index + 1, CASE].any())
            angle = float(data["vertical_error_deg"][trajectory_index, CASE])
            phase = str(data["phase"][trajectory_index])
            phase_label = {"reference": "动作执行", "hold": "末态保持"}[phase]
            draw.text((12, 1), labels[name] + "  |  case467 / 170 g / 摩擦2.2 / 原生wrist", font=font, fill="white")
            draw.text(
                (12, 33),
                f"{phase_label} {(int(data['index'][trajectory_index]) + 1) / 30:.2f}s  |  距竖直 {angle:.1f}°  |  原生failure: {failed}  |  两个视角",
                font=small,
                fill=(255, 170, 150) if failed else (190, 230, 255),
            )
            panels.append(np.asarray(image))
        composite = np.concatenate(panels, axis=0)
        writer.append_data(composite)
        if trajectory_index in [60, 97, 124, 134, 164]:
            Image.fromarray(composite).save(DIRECT / f"prior_vs_direct_{trajectory_index:03d}.jpg")

for cap in caps.values():
    cap.release()

cap = cv2.VideoCapture(str(output))
frames = 0
fps = cap.get(cv2.CAP_PROP_FPS)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    assert frame.shape == (1088, 1280, 3)
    frames += 1
cap.release()
assert frames == 105 and fps == 15

verification = {
    "case": CASE,
    "initial_fields_exact": initial_exact,
    "case_static_prefix_exact": case_static_exact,
    "direct_commands_equal_recorded_actions": direct_commands_exact,
    "case_random_forces_exact_all_steps": random_forces_exact,
    "video": {"frames": frames, "fps": fps, "duration_s": frames / fps},
}
(DIRECT / "comparison_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
