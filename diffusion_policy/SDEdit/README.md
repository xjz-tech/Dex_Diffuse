# Reference-initialized SDEdit

This package is the main-code implementation of the reference edit method used
in the object-state and Astra experiments. It is a sibling of `guidance/`:

- `reference_edit.py` selects the editing noise level and performs DDIM
  transitions. The three known action-history slots are re-noised at every
  step; the future reference initializes the remaining slots and may change.
- `reference_action_editor.py` loads a Sim-Hand checkpoint, builds the
  normalized history and reference trajectory, and returns actions in radians.

Import `ReferenceActionEditor` from
`diffusion_policy.SDEdit.reference_action_editor`. The `eval/reference_action_editor.py`
module keeps the same runtime import path for evaluation scripts. Experimental
copies under `docs/experiment_reviews/` remain frozen records of their runs.

The editor supports the project's 8-step and 12-step horizons with four
observation steps. `noise_ratio=0` returns the reference exactly without a
model call. A positive ratio chooses the nearest training timestep by
`sqrt((1 - alpha_bar) / alpha_bar)` and takes deterministic DDIM steps to zero.

For online visual DP on the real Franka + SharpA setup, use
`eval/eval_dp_edit_obs66.sh`. It takes the hand portion of each live visual DP
proposal as the editable reference and keeps the visual DP arm action. The
launcher starts in hardware-free `CHECK_ONLY=1` mode; see `eval/real/README.md`.
