# Continuous recording until first environment failure

No recording step/time cap (MAX_STEPS=0); no trajectory-completion reset
(TRAJ_STEPS_LIMIT=0 maps to None, the task's disabled-limit branch).
MAX_FAILURE_EPISODES=1; record the terminal pose before resetting and stopping.
Normal goal changes continue; RESET_ON_REACH_GOAL=0. The stopping signal is the
existing environment failure flag, which includes pose/tip tracking errors and
invalid object position, NOT an independently validated physical-drop detector.
Original failure thresholds remain unchanged. Viewer close/Q can also stop it.

TRT original 10k -> 1B, predict9 / guide2 / execute2, scale25, DDIM4 per model,
fresh noise seed8 / guide seed100008. Corrected once-per-control-step rendering.
One environment, wall-clock video, no selection of successful episodes.
Exact configuration is record.sh; console.log and episodes.jsonl explain why
the clip stopped. A still-running MP4 is not finalized/playable until stop.

Persistent systemd user unit: guidance-video-until-failure-20260914.service.
It does not restart itself. The user subsequently authorized automatic queue
resume after recording; a separate one-shot handoff service now performs it.
The known Isaac Gym post-save teardown crash may produce exit139 after a valid
MP4; inspect logs and decode the finalized video before treating it as complete.

Automatic handoff: guidance-resume-after-video-20260914.service runs
resume_after_video.py independently of the conversation. It pins the video
invocation, requires first-episode failure + normal failure-limit exit + finalized
MP4 + successful full decode, then waits for idle GPU and validates watchdog
integrity before removing its explicit PAUSED approval and starting the existing
serial data-scaling queue and watchdog timer. Exit139 is accepted only with all
those artifact/termination checks. Manual close, changed approval, decode error,
or unknown exit leaves the queue paused. handoff_state.json reports progress.
To cancel the automatic handoff, stop guidance-resume-after-video-20260914.service
or change/remove its AUTO_RESUME_AFTER_VIDEO marker in the watchdog PAUSED file.

## User-requested stop, 2026-09-14 ~18:00:50 CST

User requested stop and preservation before a failure occurred. Handoff service
was stopped and its PAUSED approval revoked BEFORE sending SIGINT specifically
to the simulator. Recording finalized: 20260914_173511_env0.mp4, 46183 frames,
1539.433 seconds (25 min 39 s). Last progress report: 40300 control steps,
26.24 Hz, zero failures/drops/resets. This is manually stopped/censored, not a
completed until-failure trial. Isaac Gym again crashed during post-save teardown;
all GPU compute processes exited. Experiment queue and watchdog remain paused.
