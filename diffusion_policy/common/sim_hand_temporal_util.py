from __future__ import annotations

from dataclasses import dataclass


SIM_HAND_ACTION_DIM = 22
SIM_HAND_OBS_DIM = 66
SIM_HAND_DIM = SIM_HAND_ACTION_DIM  # hand/action DOF width
TEMPORAL_DOWNSAMPLE_FACTOR = 4


@dataclass(frozen=True)
class SimHandTemporalConfig:
    n_obs_steps: int
    n_pred_action_steps: int
    n_action_steps: int
    horizon: int
    obs_dim: int
    action_dim: int
    oa_step_convention: bool = True

    @property
    def usable_action_slice(self) -> slice:
        start = self.n_obs_steps - 1
        return slice(start, start + self.n_pred_action_steps)

    @property
    def execution_action_slice(self) -> slice:
        start = self.n_obs_steps - 1
        return slice(start, start + self.n_action_steps)


def _config_values(config: SimHandTemporalConfig) -> str:
    return "\n".join(
        (
            f"Observation steps       : {config.n_obs_steps}",
            f"Prediction action steps : {config.n_pred_action_steps}",
            f"Execution action steps  : {config.n_action_steps}",
            f"Derived horizon         : {config.horizon}",
            f"Observation dimension   : {config.obs_dim}",
            f"Action dimension        : {config.action_dim}",
            f"OA step convention      : {config.oa_step_convention}",
        )
    )


def _invalid(config: SimHandTemporalConfig, reason: str) -> ValueError:
    return ValueError(
        "Invalid Sim-Hand DP temporal configuration.\n\n"
        f"{_config_values(config)}\n\n"
        f"{reason}\n"
        "The requested prediction length will NOT be changed automatically."
    )


def validate_sim_hand_temporal_config(
    n_obs_steps: int,
    n_pred_action_steps: int,
    n_action_steps: int,
    horizon: int,
    obs_dim: int,
    action_dim: int,
    oa_step_convention: bool = True,
) -> SimHandTemporalConfig:
    config = SimHandTemporalConfig(
        n_obs_steps=int(n_obs_steps),
        n_pred_action_steps=int(n_pred_action_steps),
        n_action_steps=int(n_action_steps),
        horizon=int(horizon),
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        oa_step_convention=bool(oa_step_convention),
    )

    if config.n_obs_steps <= 0:
        raise _invalid(config, "Observation steps must be positive.")
    if config.n_pred_action_steps <= 0:
        raise _invalid(config, "Prediction action steps must be positive.")
    if config.n_action_steps <= 0:
        raise _invalid(config, "Execution action steps must be positive.")
    if config.n_action_steps > config.n_pred_action_steps:
        raise _invalid(
            config,
            "Execution action steps must not exceed prediction action steps.",
        )

    expected_horizon = config.n_obs_steps + config.n_pred_action_steps - 1
    if config.horizon != expected_horizon:
        raise _invalid(
            config,
            "Derived horizon must equal n_obs_steps + "
            "n_pred_action_steps - 1 "
            f"({expected_horizon}), got {config.horizon}.",
        )
    if config.horizon % TEMPORAL_DOWNSAMPLE_FACTOR != 0:
        raise _invalid(
            config,
            "Diffusion horizon must be a multiple of 4 for the current "
            "ConditionalUnet1D temporal down/up-sampling structure.",
        )
    if config.obs_dim != SIM_HAND_OBS_DIM:
        raise _invalid(config, "Observation dimension must be 66.")
    if config.action_dim != SIM_HAND_ACTION_DIM:
        raise _invalid(config, "Action dimension must be 22.")
    if not config.oa_step_convention:
        raise _invalid(config, "OA step convention must be enabled.")

    return config


def _obs_history_start(n_obs_steps: int) -> str:
    offset = n_obs_steps - 1
    return "t" if offset == 0 else f"t-{offset}"


def format_sim_hand_temporal_config(config: SimHandTemporalConfig) -> str:
    return "\n".join(
        (
            "========== Sim-Hand DP Temporal Config ==========",
            f"Observation steps       : {config.n_obs_steps}",
            f"Prediction action steps : {config.n_pred_action_steps}",
            f"Execution action steps  : {config.n_action_steps}",
            f"Diffusion horizon       : {config.horizon}",
            f"Action dimension        : {config.action_dim}",
            f"Observation dimension   : {config.obs_dim}",
            f"OA step convention      : {config.oa_step_convention}",
            "",
            "Trajectory:",
            "obs condition : "
            f"s[{_obs_history_start(config.n_obs_steps)}:t+1]",
            f"usable actions: a[t:t+{config.n_pred_action_steps}]",
            f"execute       : a[t:t+{config.n_action_steps}]",
            "=================================================",
        )
    )
