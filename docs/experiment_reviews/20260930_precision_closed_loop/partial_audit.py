"""Audit the completed episode 76/34 subset while the remaining runs continue."""

import concurrent.futures
import json
from pathlib import Path

from analyze_precision_closed_loop import audit_one, pair_summary
from run_precision_closed_loop import METHODS, SEEDS, PHYSICS

HERE = Path(__file__).resolve().parent


def main():
    jobs = [(s, physics, episode, method) for s in SEEDS for physics in PHYSICS
            for episode in (76, 34) for method in METHODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        for i, row in enumerate(pool.map(audit_one, jobs), 1):
            print("AUDITED", i, "/", len(jobs), row["seed"], row["physics"],
                  row["episode"], row["method"], flush=True)
    old = json.loads((HERE.parent / "20260930_h8_object_state_random10/audited_results.json").read_text())
    old = {(r["seed"], r["physics"], r["episode"], r["method"]): r
           for r in old if r["model"] == "10b"}
    new = [json.loads((HERE / "fp16" / f"seed{s}" / "10b" / physics /
            f"episode_{episode:02d}" / method / "audit.json").read_text())
           for s, physics, episode, method in jobs]
    result = {method: pair_summary([
        (old[(r["seed"], r["physics"], r["episode"], r["method"])], r)
        for r in new if r["method"] == method]) for method in METHODS}
    (HERE / "partial_results_eps76_34.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PARTIAL COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
