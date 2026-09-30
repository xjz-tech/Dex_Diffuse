"""Validate and compare paired 10,000-step pure-prior first episodes."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "20260929_h8_prior3000" / "seed42_h8"
RUNS = {name: ROOT / f"seed42_{name}" for name in ("h8", "1b", "10b")}
CAP = 10000


def read_run(name):
    folder = RUNS[name]
    records = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == 3000
    assert {r["env"] for r in records} == set(range(3000))
    records.sort(key=lambda r: r["env"])
    length = np.asarray([r["length"] for r in records], dtype=np.int64)
    reasons = [r["reason"] for r in records]
    assert length.min() >= 1 and length.max() <= CAP
    assert set(reasons) <= {"failure", "timeout"}
    assert all(r == "failure" or n == CAP for r, n in zip(reasons, length))
    with np.load(BASE / "initial_state.npz") as old, np.load(folder / "initial_state.npz") as new:
        assert set(old.files) == set(new.files)
        assert all(np.array_equal(old[k], new[k]) for k in old.files)
    progress = json.loads((folder / "progress.json").read_text())
    assert 0 < progress["steps"] <= CAP
    if "timeout" in reasons:
        assert progress["steps"] == CAP
    assert progress["max_applied_external_force"] == 0
    return length, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-model", choices=RUNS)
    args = parser.parse_args()
    names = [args.validate_model] if args.validate_model else list(RUNS)
    data = {name: read_run(name) for name in names}
    result = {"protocol": "same seed42 3000 initial states as paired 900-step run; pure EMA prior, DDIM4/exec2, zero external force, first episode, 10000-step cap", "models": {}}
    for name, (length, reasons) in data.items():
        failed = np.asarray([r == "failure" for r in reasons])
        result["models"][name] = {
            "n": 3000,
            "native_failure": int(failed.sum()),
            "censored_at_cap": int((~failed).sum()),
            "restricted_mean_steps": float(length.mean()),
            "restricted_mean_seconds": float(length.mean() / 30),
            "median_observed_steps": float(np.median(length)),
            "failure_by_900": int((failed & (length <= 900)).sum()),
            "failure_by_3000": int((failed & (length <= 3000)).sum()),
            "failure_by_6000": int((failed & (length <= 6000)).sum()),
            "failure_by_10000": int(failed.sum()),
        }
    if not args.validate_model:
        for model, baseline in (("h8", "1b"), ("h8", "10b"), ("1b", "10b")):
            delta = data[model][0] - data[baseline][0]
            result[f"{model}_minus_{baseline}"] = {
                "mean_restricted_step_difference": float(delta.mean()),
                "longer": int((delta > 0).sum()),
                "shorter": int((delta < 0).sum()),
                "equal": int((delta == 0).sum()),
            }
        (ROOT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
