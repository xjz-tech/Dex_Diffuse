#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP = Path("/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260921_1b_10b_10k_paired")
SEEDS = (42, 8, 19, 25)
MODELS = ("1b", "10b")
N = 2500
HZ = 30.0
CAP = 400.0


def records(seed, model):
    path = EXP / "runs" / f"seed{seed}_{model}" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == N
    result = np.empty(N, dtype=float)
    reason = np.empty(N, dtype="U16")
    for row in rows:
        result[int(row["env"])] = min(CAP, int(row["length"]) / HZ)
        reason[int(row["env"])] = row["reason"]
    return result, reason


def describe(values):
    return {
        "n": int(values.size),
        "mean_s": float(values.mean()),
        "median_s": float(np.median(values)),
        "p10_s": float(np.quantile(values, 0.1)),
        "p25_s": float(np.quantile(values, 0.25)),
        "p75_s": float(np.quantile(values, 0.75)),
        "p90_s": float(np.quantile(values, 0.9)),
        "ge_20_fraction": float(np.mean(values >= 20)),
        "ge_40_fraction": float(np.mean(values >= 40)),
        "ge_80_fraction": float(np.mean(values >= 80)),
        "ge_400_fraction": float(np.mean(values >= CAP)),
    }


def bootstrap(deltas_by_seed, repetitions=5000):
    rng = np.random.default_rng(20260921)
    means = np.empty(repetitions)
    medians = np.empty(repetitions)
    for iteration in range(repetitions):
        sample = np.concatenate(
            [values[rng.integers(0, values.size, values.size)] for values in deltas_by_seed]
        )
        means[iteration] = sample.mean()
        medians[iteration] = np.median(sample)
    return {
        "paired_mean_difference_95ci_s": np.quantile(means, [0.025, 0.975]).tolist(),
        "paired_median_difference_95ci_s": np.quantile(medians, [0.025, 0.975]).tolist(),
        "method": "environment bootstrap stratified equally across the four simulator seeds",
    }


def main():
    values = {model: [] for model in MODELS}
    reasons = {model: [] for model in MODELS}
    pairing = {}
    initial = []
    per_seed = {}
    for seed in SEEDS:
        first = np.load(EXP / "runs" / f"seed{seed}_1b" / "initial_state.npz")
        second = np.load(EXP / "runs" / f"seed{seed}_10b" / "initial_state.npz")
        assert first.files == second.files
        equal = {key: bool(np.array_equal(first[key], second[key])) for key in first.files}
        assert all(equal.values())
        pairing[str(seed)] = {"all_27_fields_exact": True, "fields": equal}
        initial.append(
            {
                "seed": np.full(N, seed),
                "demo": first["demo"].astype(int),
                "frame": first["frame"].astype(int),
                "mass": first["object_mass"].astype(float),
                "friction": first["object_friction"][:, 0].astype(float),
                "scale": first["object_scale"].astype(float),
            }
        )
        per_seed[str(seed)] = {}
        for model in MODELS:
            times, ending = records(seed, model)
            values[model].append(times)
            reasons[model].append(ending)
            per_seed[str(seed)][model] = describe(times)

    all_values = {model: np.concatenate(values[model]) for model in MODELS}
    all_reasons = {model: np.concatenate(reasons[model]) for model in MODELS}
    meta = {key: np.concatenate([row[key] for row in initial]) for key in initial[0]}
    delta = all_values["10b"] - all_values["1b"]
    aggregate = {model: describe(all_values[model]) for model in MODELS}
    aggregate["10b_minus_1b"] = {
        "paired_mean_s": float(delta.mean()),
        "paired_median_s": float(np.median(delta)),
        "improved_fraction": float(np.mean(delta > 0)),
        "tied_fraction": float(np.mean(delta == 0)),
        "worsened_fraction": float(np.mean(delta < 0)),
        "gain_gt_20s_fraction": float(np.mean(delta > 20)),
        "loss_gt_20s_fraction": float(np.mean(delta < -20)),
        "rescue_lt20_to_ge20_count": int(np.sum((all_values["1b"] < 20) & (all_values["10b"] >= 20))),
        "harm_ge20_to_lt20_count": int(np.sum((all_values["1b"] >= 20) & (all_values["10b"] < 20))),
        **bootstrap([values["10b"][i] - values["1b"][i] for i in range(len(SEEDS))]),
    }

    baseline_bins = []
    for low, high in ((0, 20), (20, 40), (40, 80), (80, 401)):
        mask = (all_values["1b"] >= low) & (all_values["1b"] < high)
        baseline_bins.append(
            {
                "ordinary_1b_interval_s": [low, min(high, 400)],
                "n": int(mask.sum()),
                "mean_1b_s": float(all_values["1b"][mask].mean()),
                "mean_10b_s": float(all_values["10b"][mask].mean()),
                "paired_mean_change_s": float(delta[mask].mean()),
                "improved_fraction": float(np.mean(delta[mask] > 0)),
            }
        )

    mass_edges = np.quantile(meta["mass"], [0, .25, .5, .75, 1])
    friction_edges = np.quantile(meta["friction"], [0, .25, .5, .75, 1])
    domain_bins = []
    for mass_index in range(4):
        for friction_index in range(4):
            mass_mask = (meta["mass"] >= mass_edges[mass_index]) & (
                meta["mass"] <= mass_edges[mass_index + 1] if mass_index == 3 else meta["mass"] < mass_edges[mass_index + 1]
            )
            friction_mask = (meta["friction"] >= friction_edges[friction_index]) & (
                meta["friction"] <= friction_edges[friction_index + 1] if friction_index == 3 else meta["friction"] < friction_edges[friction_index + 1]
            )
            mask = mass_mask & friction_mask
            domain_bins.append(
                {
                    "mass_quartile": mass_index + 1,
                    "friction_quartile": friction_index + 1,
                    "n": int(mask.sum()),
                    "mean_1b_s": float(all_values["1b"][mask].mean()),
                    "mean_10b_s": float(all_values["10b"][mask].mean()),
                    "paired_mean_change_s": float(delta[mask].mean()),
                    "one_b_ge20_fraction": float(np.mean(all_values["1b"][mask] >= 20)),
                    "ten_b_ge20_fraction": float(np.mean(all_values["10b"][mask] >= 20)),
                }
            )

    marginal_domain = {}
    for name, source, edges in (
        ("mass_quartile", meta["mass"], mass_edges),
        ("friction_quartile", meta["friction"], friction_edges),
    ):
        marginal_domain[name] = []
        for index in range(4):
            mask = (source >= edges[index]) & (
                source <= edges[index + 1]
                if index == 3
                else source < edges[index + 1]
            )
            marginal_domain[name].append(
                {
                    "quartile": index + 1,
                    "range": [float(edges[index]), float(edges[index + 1])],
                    "n": int(mask.sum()),
                    "mean_1b_s": float(all_values["1b"][mask].mean()),
                    "mean_10b_s": float(all_values["10b"][mask].mean()),
                    "paired_mean_change_s": float(delta[mask].mean()),
                    "one_b_ge20_fraction": float(np.mean(all_values["1b"][mask] >= 20)),
                    "ten_b_ge20_fraction": float(np.mean(all_values["10b"][mask] >= 20)),
                }
            )

    summary = {
        "aggregate": aggregate,
        "per_seed": per_seed,
        "pairing": pairing,
        "initial_distribution": {
            "mass_kg": describe(meta["mass"]),
            "friction": describe(meta["friction"]),
            "scale": describe(meta["scale"]),
            "mass_quartile_edges": mass_edges.tolist(),
            "friction_quartile_edges": friction_edges.tolist(),
        },
        "effect_by_1b_baseline_duration": baseline_bins,
        "effect_by_mass_friction_quartile": domain_bins,
        "marginal_effect_by_domain_quartile": marginal_domain,
        "terminal_reasons": {
            model: {reason: int(np.sum(all_reasons[model] == reason)) for reason in np.unique(all_reasons[model])}
            for model in MODELS
        },
    }
    (EXP / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    grid = np.linspace(0, CAP, 401)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, label, color in (("1b", "ordinary 1B", "#3478bf"), ("10b", "ordinary 10B", "#e0702f")):
        survival = np.asarray([np.mean(all_values[model] >= point) for point in grid])
        axes[0].plot(grid, survival, label=label, color=color, linewidth=2)
    axes[0].set(xlabel="hold time (s)", ylabel="survival fraction", xlim=(0, CAP), ylim=(0, 1.01))
    axes[0].grid(alpha=.25)
    axes[0].legend()
    axes[1].hist(delta, bins=np.linspace(-400, 400, 81), color="#6c71c4", alpha=.85)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="paired hold-time change: 10B - 1B (s)", ylabel="initial states")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(EXP / "comparison.png", dpi=180)
    plt.close(fig)

    a, b, d = aggregate["1b"], aggregate["10b"], aggregate["10b_minus_1b"]
    rows = []
    for model, label in (("1b", "普通1B"), ("10b", "普通10B")):
        item = aggregate[model]
        rows.append(
            f"| {label} | {item['mean_s']:.2f} | {item['median_s']:.2f} | {item['ge_20_fraction']:.1%} | {item['ge_80_fraction']:.1%} | {item['ge_400_fraction']:.1%} |"
        )
    seed_rows = []
    for seed in SEEDS:
        one = per_seed[str(seed)]["1b"]
        ten = per_seed[str(seed)]["10b"]
        seed_rows.append(
            f"| {seed} | {one['mean_s']:.2f} | {ten['mean_s']:.2f} | {ten['mean_s'] - one['mean_s']:+.2f} | {one['ge_400_fraction']:.1%} | {ten['ge_400_fraction']:.1%} |"
        )
    mass_rows = []
    for item in marginal_domain["mass_quartile"]:
        low, high = item["range"]
        mass_rows.append(
            f"| Q{item['quartile']} ({low * 1000:.1f}–{high * 1000:.1f}g) | {item['mean_1b_s']:.2f} | {item['mean_10b_s']:.2f} | {item['paired_mean_change_s']:+.2f} |"
        )
    report = f"""# 普通1B与普通10B：10,000组配对初态

共10,000组一一配对的原生随机初态（4个seed × 2,500环境）。两模型使用真正普通DDIM、DDIM4、每次执行2步、400秒上限及xjz原生失败判定。四个seed的27类初态和实际物理字段均逐元素完全相同。

| 模型 | 平均/s | 中位/s | ≥20s | ≥80s | ≥400s |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

10B相对1B的配对平均变化为 **{d['paired_mean_s']:+.2f}秒**（分seed环境自助法95%区间 {d['paired_mean_difference_95ci_s'][0]:+.2f} 至 {d['paired_mean_difference_95ci_s'][1]:+.2f}秒），配对中位变化为 **{d['paired_median_s']:+.2f}秒**。10B更久、相同、更短的比例分别为 {d['improved_fraction']:.1%}、{d['tied_fraction']:.1%}、{d['worsened_fraction']:.1%}；将1B不足20秒救到至少20秒有 {d['rescue_lt20_to_ge20_count']} 组，反向降到20秒以下有 {d['harm_ge20_to_lt20_count']} 组。

10B的优势主要出现在长时保持：20秒通过率只增加 {b['ge_20_fraction'] - a['ge_20_fraction']:+.2%}，80秒通过率增加 {b['ge_80_fraction'] - a['ge_80_fraction']:+.2%}，400秒封顶率增加 {b['ge_400_fraction'] - a['ge_400_fraction']:+.2%}。四个seed的截断平均时长都提高：

| seed | 1B平均/s | 10B平均/s | 变化/s | 1B≥400s | 10B≥400s |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(seed_rows)}

按质量四分位描述时，10B的平均收益随质量升高明显缩小；这属于当前随机化分布中的相关性结果，并未单独固定其他参数做因果实验。

| 物体质量四分位 | 1B平均/s | 10B平均/s | 配对变化/s |
| --- | ---: | ---: | ---: |
{chr(10).join(mass_rows)}

![生存曲线和配对差值]({EXP / 'comparison.png'})

保持时间按每个环境首轮episode的控制步数/30计算；未在12,000步内失败的样本右删失并按400秒计入封顶统计。因此平均值是400秒截断均值，不能解释为未删失的真实期望寿命。原生failure是任务评估代理。
"""
    (EXP / "report.md").write_text(report)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
