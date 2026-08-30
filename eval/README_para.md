# Parallel inference/simulation evaluator

`eval_para.sh` leaves the serial evaluator unchanged. It runs Isaac Gym and
Diffusion Policy in dedicated Python processes and exchanges float32 state and
action arrays over local ZeroMQ IPC.

```bash
cd /mnt/work/dexIL/Dex_Diffuse
bash eval/eval_para.sh
```

Common overrides:

```bash
NUM_ENV=8 RECORD_ENV=0 CONTROL_HZ=30 RENDER_HZ=30 \
  bash eval/eval_para.sh

WAIT=1 bash eval/eval_para.sh

RECORDING=0 HEADLESS=1 MAX_STEPS=100 PRINT_EVERY=25 \
  bash eval/eval_para.sh
```

## Chunk-boundary modes

The checkpoint returns a five-action chunk. `WAIT=0` is the default and retains
the non-blocking behavior:

1. At startup it sends the edge-padded four-state history to the model.
2. Before the first result arrives, every control tick holds the current joint
   targets while PhysX and rendering continue.
3. A received chunk is executed in order from `action[0]` through `action[4]`.
4. After `action[4]` has completed and the observation history has been
   updated, the simulator sends that latest state to the model.
5. During inference, every control tick repeats `action[4]`. The simulator does
   not wait or pause physics. The next chunk starts at `action[0]` when it is
   received.

With `WAIT=1`, steps 1, 3 and 4 are unchanged. At startup and after each chunk,
the simulator waits for the current request instead of repeating a target.
During that wait it does not call `env.step()`, capture a recording frame, or
draw the viewer, so both simulation time and the displayed frame are frozen.
It only polls ZeroMQ and Q/Esc/viewer-close events. Once the reply arrives, the
next chunk starts immediately from `action[0]`.

The sim-side DEALER socket is polled non-blockingly and allows at most one
in-flight request. The model-side ROUTER performs blocking CUDA inference in
its own process. Messages use versioned JSON metadata and raw little-endian
float32 arrays; no pickle or Python objects cross the Python 3.8/3.10 boundary.

If any environment resets, the full-batch chunk is invalidated. Reset histories
are padded from their new qpos and those environments hold their new initial
targets. An in-flight reply from an older generation is discarded, then a
fresh latest-state request is sent. `WAIT=0` continues physics/rendering while
that request is in flight; `WAIT=1` pauses them.

## Timing and GPU isolation

- `CONTROL_HZ=30` targets PhysX/action starts. With `WAIT=0`, a model hold is
  still a normal `env.step()`. With `WAIT=1`, intentional policy waits are
  excluded from control intervals and deadline-miss accounting.
- `RENDER_HZ=30` targets viewer draws on the Isaac Gym main thread, including
  draws between slower control deadlines when time is available. `WAIT=1`
  deliberately performs no draws while waiting for the model.
- `RECORD_FPS=30` controls MP4 encoding. The recorder continues capturing
  control steps while the last action is held in `WAIT=0`; `WAIT=1` adds no
  recording frames during a model wait.
- `MODEL_CUDA_VISIBLE_DEVICES` and `SIM_CUDA_VISIBLE_DEVICES` can isolate model
  and simulation on different physical GPUs. Each process normally continues
  to address its visible device as `cuda:0`.
- `REQUEST_TIMEOUT=30` stops the evaluator if an in-flight model request has
  produced no reply for 30 seconds; this can be raised for unusually slow
  inference.

For example, on a two-GPU machine:

```bash
MODEL_CUDA_VISIBLE_DEVICES=1 SIM_CUDA_VISIBLE_DEVICES=0 \
  MODEL_DEVICE=cuda:0 SIM_DEVICE=cuda:0 RL_DEVICE=cuda:0 \
  bash eval/eval_para.sh
```

ZeroMQ removes response-wait stalls, but cannot make overloaded work meet a
deadline. On one GPU, DDPM, PhysX and viewer kernels still contend for the same
device. If `mean_env_step_ms + mean_draw_ms` exceeds 33.33 ms, real 30 Hz is
impossible without reducing work or separating GPUs.

## Progress fields

- `control_hz` / `wall_hz`: active paced control-start rate and overall
  simulated-step throughput. In `WAIT=1`, only `wall_hz` includes model waits;
- `render_hz`: actual viewer draws over wall time;
- `deadline_misses` / `max_lag_ms`: control deadlines missed by local workload;
- `mean_env_step_ms`, `mean_draw_ms`, `mean_capture_ms` and maxima: PhysX/task,
  viewer and recording-camera costs;
- `requests` / `responses`: ZeroMQ state requests and completed model replies;
- `chunks`, `chunks_completed`, `chunks_aborted`: accepted, fully executed and
  reset-invalidated action chunks;
- `mean_inference` / `mean_roundtrip`: model compute and end-to-end IPC latency;
- `wait_mode` / `policy_waits` / `policy_wait`: configured mode, number of
  chunk-boundary waits and their cumulative wall time;
- `action_steps` / `hold_steps` / `hold_ratio`: learned chunk actions versus
  repeated last-action simulation steps;
- `stale_replies`: replies discarded because a reset changed the generation;
- `pending` / `next_mode`: current in-flight request and upcoming
  `action`/`hold`/`wait` state;
- `object_drops` / `object_pose_resets`: cumulative full-batch environment
  counters.

Q/Esc, closing the viewer and Ctrl-C all finalize recording/IPC before the
dedicated Isaac Gym process exits.
