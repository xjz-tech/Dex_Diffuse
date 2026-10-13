"""Audit and summarize paired 10B FP16-vs-own-FP32 bulb-turn rollouts."""

import concurrent.futures
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/experiment_reviews/20260924_object_state_data"
sys.path.insert(0, str(SOURCE))

import analyze_h8_cross_model_comparison as analysis  # noqa: E402
import run_h8_cross_model_comparison as run  # noqa: E402
from run_precision_closed_loop import SEEDS, METHODS  # noqa: E402


def audit_one(job):
    seed, physics, episode, method = job
    run.O = HERE / "fp16"
    analysis.O = run.O
    folder = run.destination(seed, "10b", physics, episode, method)
    path = folder / "audit.json"
    if path.exists():
        return json.loads(path.read_text())
    row = analysis.audit(seed, "10b", physics, episode, method)
    path.write_text(json.dumps(row, indent=2) + "\n")
    return row


def length(row):
    if row["first_separation_action_step"] is not None:
        return row["first_separation_action_step"]
    return json.loads((Path(row["folder"]) / "summary.json").read_text())["steps"]["action"]


def pair_summary(pairs):
    a = [r[0] for r in pairs]
    b = [r[1] for r in pairs]
    lengths = [length(y) - length(x) for x, y in pairs]
    return dict(n=len(pairs), fp32_ordinary=sum(r["ordinary_turn"] for r in a),
        fp16_ordinary=sum(r["ordinary_turn"] for r in b),
        fp32_stable=sum(r["stable_turn"] for r in a),
        fp16_stable=sum(r["stable_turn"] for r in b),
        fp32_mean_steps=statistics.mean(map(length, a)),
        fp16_mean_steps=statistics.mean(map(length, b)),
        mean_step_difference=statistics.mean(lengths),
        median_step_difference=statistics.median(lengths),
        stable_lost=sum(x["stable_turn"] and not y["stable_turn"] for x, y in pairs),
        stable_gained=sum(y["stable_turn"] and not x["stable_turn"] for x, y in pairs),
        ordinary_lost=sum(x["ordinary_turn"] and not y["ordinary_turn"] for x, y in pairs),
        ordinary_gained=sum(y["ordinary_turn"] and not x["ordinary_turn"] for x, y in pairs),
        steps_shorter=sum(z < 0 for z in lengths),
        steps_equal=sum(z == 0 for z in lengths),
        steps_longer=sum(z > 0 for z in lengths))


def main():
    jobs = [(seed, physics, episode, method) for seed in SEEDS
            for physics in run.PHYSICS for episode in run.EPISODES for method in METHODS]
    assert len(jobs) == 240
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        fp16 = list(pool.map(audit_one, jobs))
    assert len(fp16) == 240
    assert all(r["validation"]["initial_all_fields_exact"] and
               r["validation"]["settle_motion_exact"] and
               r["validation"]["settle_contact_force_max_delta_N"] < 1e-5
               for r in fp16)
    (HERE / "fp16_audited_results.json").write_text(json.dumps(fp16, indent=2) + "\n")

    old_path = HERE.parent / "20260930_h8_object_state_random10/audited_results.json"
    old = [r for r in json.loads(old_path.read_text()) if r["model"] == "10b"]
    assert len(old) == 240
    key = lambda r: (r["seed"], r["physics"], r["episode"], r["method"])
    old_map = {key(r): r for r in old}
    new_map = {key(r): r for r in fp16}
    assert len(old_map) == len(new_map) == 240 and old_map.keys() == new_map.keys()
    pairs = [(old_map[k], new_map[k]) for k in sorted(old_map)]
    result = {method: pair_summary([(a, b) for a, b in pairs if a["method"] == method])
              for method in METHODS}
    result["by_physics"] = {
        physics: {method: pair_summary([(a, b) for a, b in pairs
            if a["physics"] == physics and a["method"] == method])
            for method in METHODS} for physics in run.PHYSICS}
    result["by_seed"] = {
        str(seed): {method: pair_summary([(a, b) for a, b in pairs
            if a["seed"] == seed and a["method"] == method])
            for method in METHODS} for seed in SEEDS}
    (HERE / "paired_results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# 10B：FP16 相对各方法自身 FP32 的闭环损失", "",
        "相同 10 个 prior 噪声 seed × 4 个合格横抓 episode × 3 组物理参数；每种方法各有 120 对。FP32 基线复用已审计的原始轨迹；先导实验在同一整条 episode 中确认 fused FP32 发出的动作与该基线逐值相同。FP16 使用同一个 10B EMA、相同 fused DDIM 公式和 TensorRT FP16 UNet，输入输出及 DDIM 算术为 FP32。配置见 [manifest.json](manifest.json)，先导核验见 [pilot_validation.json](pilot_validation.json)。", "",
        "| 方法 | FP32 普通翻转 | FP16 普通翻转 | FP32 稳定翻转 | FP16 稳定翻转 | FP32 平均步数 | FP16 平均步数 | FP16−FP32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",]
    for method in METHODS:
        r = result[method]
        lines.append(f'| {method} | {r["fp32_ordinary"]}/120 | {r["fp16_ordinary"]}/120 | '
            f'{r["fp32_stable"]}/120 | {r["fp16_stable"]}/120 | '
            f'{r["fp32_mean_steps"]:.1f} | {r["fp16_mean_steps"]:.1f} | '
            f'{r["mean_step_difference"]:+.1f} |')
    lines += ["", "同条件逐条配对的稳定翻转变化："]
    for method in METHODS:
        r = result[method]
        lines.append(f'- {method}：FP32 成功而 FP16 失败 {r["stable_lost"]} 条；'
                     f'FP32 失败而 FP16 成功 {r["stable_gained"]} 条。'
                     f'执行长度变短／相同／变长为 {r["steps_shorter"]}／{r["steps_equal"]}／{r["steps_longer"]}。')
    lines += ["", "## 分物理参数", "",
              "| 质量／摩擦 | 方法 | FP32 稳定翻转 | FP16 稳定翻转 | FP32 平均步数 | FP16 平均步数 |",
              "|---|---|---:|---:|---:|---:|"]
    for physics in run.PHYSICS:
        for method in METHODS:
            r = result["by_physics"][physics][method]
            lines.append(f'| {physics} | {method} | {r["fp32_stable"]}/40 | '
                f'{r["fp16_stable"]}/40 | {r["fp32_mean_steps"]:.1f} | '
                f'{r["fp16_mean_steps"]:.1f} |')
    lines += ["", "稳定翻转要求在首次几何分离前连续 ≥30 个控制步保持竖直、悬空且有效接触。平均步数从动作开始计至首次连续 3 步几何分离，未分离者在录像尾段截尾。native failure 独立记录，不作为物理脱手。实验使用 edit 0.15／DDIM4／exec2 与 guidance guide4／scale50／DDIM4／exec2；先前速度基准中的 edit 为 0.20，因此不能混用其动作误差数值。", "",
              "逐条审计结果见 [fp16_audited_results.json](fp16_audited_results.json)；各 seed 的完整配对统计见 [paired_results.json](paired_results.json)。", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("AUDITED", len(fp16), "REPORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
