# Follow-on parallel guidance experiments

Code worktree: `/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/parallel-guidance-1b-base`

Code commit: `9cd8c74c32c2de5802f8a01aea7c49add960e0c0`

Initial supervisor PID list (PIDs can change on restart):

| Role | User service | PID |
|---|---|---:|
| Existing serial/fixed/dynamic experiments | parallel-guidance-1b-base-fullhorizon-3k.service | 995311 |
| New follow-on candidate queue | parallel-guidance-candidates.service | 1156473 |

The new supervisor waits for PID 995311 with process start token 208714151,
requires the existing suite's state.env to say complete, and then waits until
no GPU compute processes remain. It does not terminate any existing GPU job.

Execution order: real-checkpoint numerical checks and batch-1 latency benchmark,
then shared_x0_w10, shared_x0_w30, dual_x0_ramp, late_x0_w10, late_x0_w30.
Each configuration runs seeds 8, 19, 25 sequentially, with 3000 first episodes
per seed. All corrections cover all nine predicted actions; two are executed.
The original suite's explicit hold evaluation thresholds are retained.

Live queue status: state.json. Supervisor log: pipeline.log.
Numerical/latency results: benchmark/benchmark.json and benchmark.log.
Automatic effect/latency comparison: comparison.md and comparison.json.
Promising results trigger a local desktop notification when available; no
future message into the chat is scheduled. A failed predecessor or failed
numerical/evaluation stage stops the queue with a failed state.

Incomplete runs are not scored. The report requires all three seeds before
screening an algorithm. See docs/parallel_candidate_suite.md in the worktree
for formulas and screening criteria. Historical hold-to-cap completion is not
task success, and the old two-action-guidance benchmark is only a reference,
not an exactly matched control for the new full-horizon setup.
