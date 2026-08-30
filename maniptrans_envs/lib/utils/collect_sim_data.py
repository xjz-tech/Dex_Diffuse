from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

# This entrypoint lives below the repository root, so an editable install of a
# sibling checkout could otherwise win import resolution. Always use this tree.
REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_str = str(REPO_ROOT)
if sys.path[0] != repo_root_str:
    sys.path.insert(0, repo_root_str)

from lib.utils.reformat import omegaconf_to_dict


torch = None


def _import_isaacgym_first() -> None:
    import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch


def _load_runtime_deps():
    global torch
    _import_isaacgym_first()
    import torch as torch_mod
    from h5_sim_data import H5SimDataWriter

    torch = torch_mod
    return torch_mod, H5SimDataWriter


def _prepare_rlgames_config(cfg: DictConfig) -> Dict:
    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    if rlg_config_dict["params"]["algo"]["name"] == "ppo":
        rlg_config_dict["params"]["algo"]["name"] = "a2c_continuous"
    network_mapping = {
        "single_rh_dict_obs_actor_critic": "actor_critic",
        "single_lh_dict_obs_actor_critic": "actor_critic",
        "res_rh_dict_obs_actor_critic": "actor_critic",
        "res_lh_dict_obs_actor_critic": "actor_critic",
        "res_bih_dict_obs_actor_critic": "actor_critic",
        "sep_dict_obs_actor_critic": "actor_critic",
    }
    net_name = rlg_config_dict["params"]["network"].get("name", "")
    if net_name in network_mapping:
        rlg_config_dict["params"]["network"]["name"] = network_mapping[net_name]
    model_mapping = {
        "single_rh_my_continuous_a2c_logstd": "continuous_a2c_logstd",
        "single_lh_my_continuous_a2c_logstd": "continuous_a2c_logstd",
        "res_rh_my_continuous_a2c_logstd": "continuous_a2c_logstd",
        "res_lh_my_continuous_a2c_logstd": "continuous_a2c_logstd",
        "res_bih_my_continuous_a2c_logstd": "continuous_a2c_logstd",
        "sep_my_continuous_a2c_logstd": "continuous_a2c_logstd",
    }
    model_name = rlg_config_dict["params"]["model"].get("name", "")
    if model_name in model_mapping:
        rlg_config_dict["params"]["model"]["name"] = model_mapping[model_name]
    if rlg_config_dict["params"]["network"].get("name") == "actor_critic":
        rlg_config_dict["params"]["network"].pop("dict_feature_encoder", None)
    config_dict = rlg_config_dict["params"]["config"]
    if cfg.get("num_envs"):
        config_dict["num_actors"] = cfg.num_envs
    if cfg.get("learning_rate"):
        config_dict["learning_rate"] = cfg.learning_rate
    if cfg.get("horizon_length"):
        config_dict["horizon_length"] = cfg.horizon_length
    if cfg.get("mini_batchsize"):
        config_dict["minibatch_size"] = cfg.mini_batchsize
    if cfg.get("mini_epochs"):
        config_dict["mini_epochs"] = cfg.mini_epochs
    if cfg.get("max_iterations"):
        config_dict["max_epochs"] = cfg.max_iterations
    if cfg.get("save_frequency"):
        config_dict["save_frequency"] = cfg.save_frequency
    if cfg.get("experiment"):
        config_dict["name"] = f"1_{cfg.experiment}"
    for key in (
        "expl_type",
        "expl_coef_block_size",
        "off_policy_ratio",
        "expl_reward_type",
        "expl_reward_coef_scale",
        "entropy_schedule",
    ):
        val = cfg.get(key)
        if val is not None:
            config_dict[key] = val
    if cfg.get("fixed_sigma") is not None:
        net_space = rlg_config_dict["params"]["network"].setdefault("space", {})
        net_space.setdefault("continuous", {})["fixed_sigma"] = cfg.fixed_sigma
    config_dict["device"] = cfg.get("rl_device", "cuda")
    config_dict["seed"] = cfg.get("seed", 42)
    config_dict["normalize_input"] = True
    config_dict["normalize_value"] = True
    config_dict.setdefault("network_path", "./nn/")
    config_dict.setdefault("log_path", "runs/")
    config_dict.setdefault("full_experiment_name", config_dict.get("name"))
    config_dict.setdefault("player", {})
    config_dict["player"].setdefault("use_vecenv", True)
    config_dict["player"]["render"] = False
    config_dict["player"]["print_stats"] = False
    return rlg_config_dict


def _register_env(cfg: DictConfig) -> None:
    _import_isaacgym_first()
    import maniptrans_envs.lib as maniptrans_envs_lib
    from lib.utils.rlgames_utils import ComplexObsRLGPUEnv
    from rl_games.common import env_configurations, vecenv

    task_cfg = cfg.task
    headless = bool(cfg.get("headless", True))
    display = bool(cfg.get("display", False))

    def create_env():
        return maniptrans_envs_lib.make(
            sim_device=cfg.get("sim_device", "cuda:0"),
            rl_device=cfg.get("rl_device", "cuda:0"),
            graphics_device_id=cfg.get("graphics_device_id", 0),
            multi_gpu=cfg.get("multi_gpu", False),
            cfg=task_cfg,
            display=display,
            record=False,
            has_headless_arg=True,
            headless=headless,
        )

    env_configurations.configurations["rlgpu"] = {
        "vecenv_type": "RLGPU",
        "env_creator": create_env,
    }
    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_snapshot(
    env,
    raw_action: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    from main.dataset.transform import rotmat_to_quat

    num_envs = env.num_envs
    fingertip_indices = [env.dexhand.body_names.index(name) for name in env.dexhand.fingertip_body_names]
    fingertips_world = torch.stack(
        [env._rigid_body_state[:, env.dexhand_handles[name], :3] for name in env.dexhand.fingertip_body_names],
        dim=1,
    )
    fingertips_wrist = env.states["joints_state_rel_wrist"][:, fingertip_indices, :3]
    fingertip_contact_force = torch.stack(
        [env.net_cf[:, env.dexhand_handles[name], :] for name in env.dexhand.contact_body_names],
        dim=1,
    )
    wrist_pose = env.states["base_state"][:, :7]
    object_root_state = env._manip_obj_root_state
    target_pose_world = env.current_target_obj_pose_world[:, 0]
    target_quat_world = rotmat_to_quat(target_pose_world[:, :3, :3])[:, [1, 2, 3, 0]]
    target_quat_world = torch.where(target_quat_world[:, 3:4] < 0, -target_quat_world, target_quat_world)
    target_obs = env.current_target_obs
    n_future = int(env.obs_future_length)

    def first_future(name: str, *shape: int) -> torch.Tensor:
        return target_obs[name].reshape(num_envs, n_future, *shape)[:, 0]

    if raw_action is None:
        raw_action = torch.zeros_like(env.curr_targets)
    action_applied = env.prev_actions if env.prev_actions is not None else torch.zeros_like(env.curr_targets)
    sim_frame = int(env.gym.get_frame_count(env.sim))
    sim_frames = torch.full((num_envs,), sim_frame, device=env.device, dtype=torch.long)
    return {
        "qpos": env._q.detach().clone(),
        "qvel": env._qd.detach().clone(),
        "target": env.curr_targets.detach().clone(),
        "raw_action": raw_action.detach().clone(),
        "action_applied": action_applied.detach().clone(),
        "wrist_pose_world": wrist_pose.detach().clone(),
        "object_position_world": object_root_state[:, :3].detach().clone(),
        "object_quaternion_world": object_root_state[:, 3:7].detach().clone(),
        "object_linear_velocity_world": object_root_state[:, 7:10].detach().clone(),
        "object_angular_velocity_world": object_root_state[:, 10:13].detach().clone(),
        "target_object_position_world": target_pose_world[:, :3, 3].detach().clone(),
        "target_object_quaternion_world": target_quat_world.detach().clone(),
        "target_demo_frame_index": env.current_target_frame_idx.detach().clone(),
        "target_data_index_id": env.current_target_data_index_id.detach().clone(),
        "target_fingertips_position_wrist": first_future("fingertips_pos_rel_wrist", 5, 3).detach().clone(),
        "target_delta_fingertips_position_wrist": first_future(
            "delta_fingertips_pos_rel_wrist", 5, 3
        ).detach().clone(),
        "target_object_position_wrist": first_future("manip_obj_pos_rel_wrist", 3).detach().clone(),
        "target_delta_object_position_wrist": first_future(
            "delta_manip_obj_pos_rel_wrist", 3
        ).detach().clone(),
        "target_object_quaternion_wrist": first_future("manip_obj_quat_rel_wrist", 4).detach().clone(),
        "target_delta_object_quaternion_wrist": first_future(
            "delta_manip_obj_quat_rel_wrist", 4
        ).detach().clone(),
        "fingertips_position_world": fingertips_world.detach().clone(),
        "fingertips_position_wrist": fingertips_wrist.detach().clone(),
        "fingertips_contact_force": fingertip_contact_force.detach().clone(),
        "traj_direction": env.traj_direction.detach().clone(),
        "skip_steps": env.skip_steps_buf.detach().clone(),
        "sim_frame": sim_frames,
        "timestamp": sim_frames.double() * float(env.dt),
    }


def _to_numpy(x: torch.Tensor, dtype=None) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _data_index_strings(env, numeric_ids: torch.Tensor) -> np.ndarray:
    ids = _to_numpy(numeric_ids, np.int64)
    return np.asarray([str(env.dataIndices[int(idx)]).encode("utf-8") for idx in ids], dtype="S128")


def _apply_fixed_block_selection(
    player,
    requested_block_id: int,
    block_size: int,
    num_envs: int,
):
    """Route every player observation through one existing SAPG block.

    The player must retain the training-time number of blocks so its learned
    extra_params and coef-conditioned sigma tensors match the checkpoint.
    Changing only the per-env coefficient embedding selects a block without
    changing that model structure.  A requested id of -1 leaves the original
    per-env block assignment unchanged.
    """
    num_blocks = int(num_envs) // int(block_size)
    if requested_block_id == -1:
        return None, None
    if requested_block_id < 0 or requested_block_id >= num_blocks:
        raise ValueError(
            f"sim_data.fixed_block_id={requested_block_id} must be -1 or in "
            f"[0, {num_blocks - 1}]"
        )

    embeddings = getattr(player, "intr_reward_coef_embd", None)
    if embeddings is None:
        raise ValueError(
            "sim_data.fixed_block_id requires a mixed_expl player with block embeddings"
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != num_envs:
        raise ValueError(
            "Unexpected SAPG block embedding shape: "
            f"expected ({num_envs}, D), got {tuple(embeddings.shape)}"
        )

    selected_row = requested_block_id * int(block_size)
    selected_embedding = embeddings[selected_row].detach().clone()
    embeddings.copy_(selected_embedding.unsqueeze(0).expand_as(embeddings))
    return requested_block_id, float(selected_embedding[0].item())


def _make_transition_batch(
    env,
    before: Mapping[str, torch.Tensor],
    after_action: Mapping[str, torch.Tensor],
    rewards: torch.Tensor,
    dones: torch.Tensor,
    timeouts: torch.Tensor,
    episode_ids: torch.Tensor,
    steps_in_episode: torch.Tensor,
    qacc: torch.Tensor,
    qacc_valid: torch.Tensor,
    successes: torch.Tensor,
    failures: torch.Tensor,
    block_size: int,
    collect_mask: torch.Tensor,
    fixed_block_id: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    ids = collect_mask.nonzero(as_tuple=False).flatten()
    data_index_numeric_id = before["target_data_index_id"][ids]
    data_index_id = _data_index_strings(env, data_index_numeric_id)
    reset_reason = torch.zeros_like(dones, dtype=torch.int8)
    reset_reason = torch.where(dones.bool(), torch.full_like(reset_reason, 4), reset_reason)
    reset_reason = torch.where(dones.bool() & successes.bool(), torch.full_like(reset_reason, 1), reset_reason)
    reset_reason = torch.where(dones.bool() & failures.bool(), torch.full_like(reset_reason, 2), reset_reason)
    reset_reason = torch.where(dones.bool() & timeouts.bool(), torch.full_like(reset_reason, 3), reset_reason)
    block_ids = (
        ids // int(block_size)
        if fixed_block_id is None
        else torch.full_like(ids, int(fixed_block_id))
    )
    return {
        "index/episode_id": _to_numpy(episode_ids[ids], np.int64),
        "index/env_id": _to_numpy(ids, np.int32),
        "index/step": _to_numpy(steps_in_episode[ids], np.int64),
        "index/sim_frame": _to_numpy(before["sim_frame"][ids], np.int64),
        "index/timestamp": _to_numpy(before["timestamp"][ids], np.float64),
        "index/done": _to_numpy(dones[ids].bool(), np.bool_),
        "index/timeout": _to_numpy(timeouts[ids].bool(), np.bool_),
        "index/success": _to_numpy(successes[ids].bool(), np.bool_),
        "index/failure": _to_numpy(failures[ids].bool(), np.bool_),
        "index/reset_reason": _to_numpy(reset_reason[ids], np.int8),
        "index/data_index_id": data_index_id,
        "index/block_id": _to_numpy(block_ids, np.int32),
        "index/traj_direction": _to_numpy(before["traj_direction"][ids], np.int8),
        "index/skip_steps": _to_numpy(before["skip_steps"][ids], np.int32),
        "robot/target_before": _to_numpy(before["target"][ids], np.float32),
        "robot/target_after": _to_numpy(after_action["target"][ids], np.float32),
        "robot/wrist_pose_world": _to_numpy(before["wrist_pose_world"][ids], np.float32),
        "robot/qpos": _to_numpy(before["qpos"][ids], np.float32),
        "robot/qvel": _to_numpy(before["qvel"][ids], np.float32),
        "robot/qacc": _to_numpy(qacc[ids], np.float32),
        "robot/qacc_valid": _to_numpy(qacc_valid[ids], np.bool_),
        "object/position_world": _to_numpy(before["object_position_world"][ids], np.float32),
        "object/quaternion_world": _to_numpy(before["object_quaternion_world"][ids], np.float32),
        "object/linear_velocity_world": _to_numpy(
            before["object_linear_velocity_world"][ids], np.float32
        ),
        "object/angular_velocity_world": _to_numpy(
            before["object_angular_velocity_world"][ids], np.float32
        ),
        "target/object_position_world": _to_numpy(
            before["target_object_position_world"][ids], np.float32
        ),
        "target/object_quaternion_world": _to_numpy(
            before["target_object_quaternion_world"][ids], np.float32
        ),
        "target/demo_frame_index": _to_numpy(before["target_demo_frame_index"][ids], np.int64),
        "target/data_index_id": data_index_id,
        "target/fingertips_position_wrist": _to_numpy(
            before["target_fingertips_position_wrist"][ids], np.float32
        ),
        "target/delta_fingertips_position_wrist": _to_numpy(
            before["target_delta_fingertips_position_wrist"][ids], np.float32
        ),
        "target/object_position_wrist": _to_numpy(
            before["target_object_position_wrist"][ids], np.float32
        ),
        "target/delta_object_position_wrist": _to_numpy(
            before["target_delta_object_position_wrist"][ids], np.float32
        ),
        "target/object_quaternion_wrist": _to_numpy(
            before["target_object_quaternion_wrist"][ids], np.float32
        ),
        "target/delta_object_quaternion_wrist": _to_numpy(
            before["target_delta_object_quaternion_wrist"][ids], np.float32
        ),
        "policy/action_raw": _to_numpy(after_action["raw_action"][ids], np.float32),
        "policy/action_applied": _to_numpy(after_action["action_applied"][ids], np.float32),
        "fingertip/position_world": _to_numpy(before["fingertips_position_world"][ids], np.float32),
        "fingertip/position_wrist": _to_numpy(before["fingertips_position_wrist"][ids], np.float32),
        "fingertip/contact_force": _to_numpy(before["fingertips_contact_force"][ids], np.float32),
        "outcome/reward": _to_numpy(rewards[ids], np.float32),
    }


class LastSuccessTransitionBuffer:
    """Buffer unconfirmed per-env tails until they reach an intermediate goal.

    Committed prefixes are immutable and can be streamed to HDF5.  If an env
    later fails, only the tail accumulated since its last reach_final_goal is
    discarded, so the stored episode ends at its last successful target.  An
    episode that fails before any reach_final_goal never reaches the writer.
    """

    def __init__(self, num_envs: int, initial_capacity: int = 64):
        self.num_envs = int(num_envs)
        self.capacity = max(1, int(initial_capacity))
        self.lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._storage: Dict[str, np.ndarray] = {}

    @property
    def pending_rows(self) -> int:
        return int(self.lengths.sum())

    def _initialize(self, batch: Mapping[str, np.ndarray]) -> None:
        self._storage = {
            path: np.empty(
                (self.capacity, self.num_envs) + np.asarray(array).shape[1:],
                dtype=np.asarray(array).dtype,
            )
            for path, array in batch.items()
        }

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < required:
            new_capacity *= 2
        for path, old in self._storage.items():
            new = np.empty((new_capacity,) + old.shape[1:], dtype=old.dtype)
            new[: self.capacity] = old
            self._storage[path] = new
        self.capacity = new_capacity

    def append(self, batch: Mapping[str, np.ndarray]) -> None:
        env_ids = np.asarray(batch["index/env_id"], dtype=np.int64)
        if env_ids.size == 0:
            return
        if np.any(env_ids < 0) or np.any(env_ids >= self.num_envs):
            raise IndexError("Transition batch contains an out-of-range env_id")
        if np.unique(env_ids).size != env_ids.size:
            raise ValueError("Transition batch contains duplicate env_id rows")
        if not self._storage:
            self._initialize(batch)
        elif set(batch) != set(self._storage):
            raise KeyError("Transition batch fields changed while buffering")

        positions = self.lengths[env_ids]
        self._ensure_capacity(int(positions.max()) + 1)
        for path, array in batch.items():
            values = np.asarray(array)
            if values.shape[0] != env_ids.size:
                raise ValueError(f"Field '{path}' has inconsistent batch length")
            self._storage[path][positions, env_ids] = values
        self.lengths[env_ids] += 1

    def pop(self, env_ids: np.ndarray):
        requested = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        selected = requested[self.lengths[requested] > 0]
        if selected.size == 0:
            return None, selected, np.empty(0, dtype=np.int64)
        selected_lengths = self.lengths[selected].copy()
        batch = {
            path: np.concatenate(
                [array[: int(length), int(env_id)] for env_id, length in zip(selected, selected_lengths)],
                axis=0,
            )
            for path, array in self._storage.items()
        }
        self.lengths[selected] = 0
        return batch, selected, selected_lengths

    def discard(self, env_ids: np.ndarray) -> int:
        selected = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        discarded = int(self.lengths[selected].sum())
        self.lengths[selected] = 0
        return discarded


def collect_sim_data(cfg: DictConfig) -> None:
    torch_mod, H5SimDataWriter = _load_runtime_deps()
    from rl_games.torch_runner import Runner, _override_sigma, _restore

    with torch_mod.no_grad():
        if not cfg.get("checkpoint"):
            raise ValueError("checkpoint must be set")
        _register_env(cfg)
        rlg_config = _prepare_rlgames_config(cfg)
        runner = Runner()
        runner.load(rlg_config)
        player = runner.create_player()
        _restore(player, {"checkpoint": cfg.checkpoint})
        _override_sigma(player, {"sigma": cfg.sigma if cfg.get("sigma") else None})

        vec_env = player.env
        env = vec_env.env
        block_size = int(cfg.get("expl_coef_block_size", env.num_envs))
        if block_size <= 0 or env.num_envs % block_size != 0:
            raise ValueError(
                f"expl_coef_block_size={block_size} must be positive and divide num_envs={env.num_envs}"
            )
        requested_fixed_block_id = int(cfg.sim_data.get("fixed_block_id", -1))
        fixed_block_id, fixed_block_coef_id = _apply_fixed_block_selection(
            player,
            requested_fixed_block_id,
            block_size,
            env.num_envs,
        )
        if fixed_block_id is not None:
            print(
                f"Fixed SAPG block selection: block_id={fixed_block_id}, "
                f"coef_id={fixed_block_coef_id:g}, num_envs={env.num_envs}",
                flush=True,
            )

        player.is_deterministic = bool(cfg.sim_data.get("deterministic", True))
        if player.is_rnn:
            player.init_rnn()

        env.compute_observations()
        if env.enable_asymmetric_actor_critic:
            env.compute_extra_states()
        obs = player.env_reset(vec_env)
        player.get_batch_size(obs, 1)
        if player.is_rnn and player.states is None:
            player.init_rnn()

        cfg_yaml = OmegaConf.to_yaml(cfg)
        cfg_hash = hashlib.sha256(cfg_yaml.encode("utf-8")).hexdigest()
        data_indices = list(env.dataIndices)
        default_output_root = Path(__file__).resolve().parents[3] / "data" / "sim_data"
        output_root = Path(str(cfg.sim_data.get("output_root", default_output_root)))
        exp_name = str(cfg.sim_data.get("exp_name") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        object_names = sorted({str(idx).split("@")[0].replace("v3:", "") for idx in data_indices})
        suite_name = str(
            cfg.sim_data.get("suite_name")
            or cfg.sim_data.get("object_name")
            or ("+".join(object_names) if object_names else "suite")
        )
        object_name = object_names[0] if len(object_names) == 1 else "multi_object"
        rollout_name = str(cfg.sim_data.get("rollout_name") or object_name)
        output_dir = output_root / object_name / exp_name
        checkpoint_path = Path(str(cfg.checkpoint)).expanduser().resolve()
        policy_dt = float(env.dt) * int(env.control_freq_inv)
        truncate_failed_episodes = bool(
            cfg.sim_data.get("truncate_failed_episodes_to_last_success", False)
        )
        min_reach_goals_per_episode = int(
            cfg.sim_data.get("min_reach_goals_per_episode", 0)
        )
        if min_reach_goals_per_episode < 0:
            raise ValueError("sim_data.min_reach_goals_per_episode must be non-negative")
        metadata = {
            "created_time": datetime.now().isoformat(),
            "hand_type": "sharpa",
            "object_name": object_name,
            "suite_name": suite_name,
            "object_names": object_names,
            "rollout_name": rollout_name,
            "data_indices": data_indices,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "action_style": env.act_style,
            "action_scale": float(env.act_scale),
            "controlFrequencyInv": int(env.control_freq_inv),
            "sim_dt": float(env.dt),
            "policy_dt": policy_dt,
            "dt": float(env.dt),
            "dof_names": list(env.dexhand.dof_names),
            "joint_lower": _to_numpy(env.dexhand_dof_lower_limits, np.float32).tolist(),
            "joint_upper": _to_numpy(env.dexhand_dof_upper_limits, np.float32).tolist(),
            "fingertip_contact_body_names": list(env.dexhand.contact_body_names),
            "quaternion_order": "xyzw",
            "units": {
                "position": "meter",
                "linear_velocity": "meter/second",
                "angular_velocity": "radian/second",
                "joint_position": "radian",
                "joint_velocity": "radian/second",
                "joint_acceleration": "radian/second^2",
                "timestamp": "simulation_second",
            },
            "random_seed": int(cfg.get("seed", 42)),
            "sapg_block_size": block_size,
            "sapg_block_count": int(env.num_envs // block_size),
            "fixed_block_id": fixed_block_id,
            "fixed_block_coef_id": fixed_block_coef_id,
            "num_envs": int(env.num_envs),
            "deterministic_policy": bool(player.is_deterministic),
            "min_reach_goals_per_episode": min_reach_goals_per_episode,
            "truncate_failed_episodes_to_last_success": truncate_failed_episodes,
            "failed_episode_storage_policy": (
                "commit_through_last_reach_final_goal; discard episodes that fail before first reach"
                if truncate_failed_episodes
                else "store all transitions including failure tails"
            ),
            "reset_reason_codes": {
                "0": "none",
                "1": "success",
                "2": "failure",
                "3": "timeout",
                "4": "other_done",
            },
            "config_hash": cfg_hash,
            "schema_note": (
                "Each row is (state_before_action, target_seen_by_policy, action, result_after_action). "
                "qpos/qvel/targets/actions use policy DOF order; quaternions use xyzw; "
                "keypoints are dexhand.body_names[1:]."
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hydra_config.yaml").write_text(cfg_yaml, encoding="utf-8")

        episode_ids = torch_mod.arange(env.num_envs, device=env.device, dtype=torch_mod.long)
        next_episode_id = int(env.num_envs)
        steps_in_episode = torch_mod.zeros(env.num_envs, device=env.device, dtype=torch_mod.long)
        reach_goals_in_episode = torch_mod.zeros(
            env.num_envs, device=env.device, dtype=torch_mod.long
        )
        warmup_remaining = torch_mod.full(
            (env.num_envs,),
            int(cfg.sim_data.get("drop_reset_warmup", 2)),
            device=env.device,
            dtype=torch_mod.long,
        )
        total_target = int(cfg.sim_data.get("num_transitions", 200_000))
        max_steps = int(cfg.sim_data.get("max_steps", 10_000_000))
        print_every = int(cfg.sim_data.get("print_every", 100))
        recorded_episode_lengths = torch_mod.zeros(env.num_envs, device=env.device, dtype=torch_mod.long)
        completed_recorded_episode_count = 0
        completed_recorded_transition_sum = 0
        previous_qvel = env._qd.detach().clone()
        qacc_valid = torch_mod.zeros(env.num_envs, device=env.device, dtype=torch_mod.bool)
        success_buffer = (
            LastSuccessTransitionBuffer(env.num_envs)
            if truncate_failed_episodes or min_reach_goals_per_episode > 0
            else None
        )
        committed_success_segments = 0
        discarded_failed_episodes = 0
        discarded_insufficient_reach_episodes = 0
        discarded_unconfirmed_transitions = 0

        writer = H5SimDataWriter(
            output_dir,
            metadata=metadata,
            dof=env.dexhand.n_dofs,
            shard_transition_limit=int(cfg.sim_data.get("shard_transition_limit", 1_000_000)),
            shard_size_limit_bytes=int(cfg.sim_data.get("shard_size_limit_bytes", 4 * 1024**3)),
            chunk_size=int(cfg.sim_data.get("chunk_size", 8192)),
            compression=cfg.sim_data.get("compression", "lzf"),
        )
        with writer:
            for step in range(max_steps):
                if writer.total_rows >= total_target:
                    break
                if step > 0 and torch_mod.any(dones):
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    obs, reset_ids = vec_env.reset_done()
                    if player.is_rnn:
                        for s in player.states:
                            s[:, done_ids, :] = 0.0
                    reset_ids = reset_ids.to(env.device)
                    if success_buffer is not None:
                        discarded_unconfirmed_transitions += success_buffer.discard(
                            _to_numpy(reset_ids, np.int64)
                        )
                    if min_reach_goals_per_episode > 0:
                        discarded_insufficient_reach_episodes += int(
                            (
                                reach_goals_in_episode[reset_ids]
                                < min_reach_goals_per_episode
                            ).sum().item()
                        )
                    finished_lengths = recorded_episode_lengths[reset_ids]
                    valid_finished = finished_lengths > 0
                    if torch_mod.any(valid_finished):
                        completed_recorded_transition_sum += int(finished_lengths[valid_finished].sum().item())
                        completed_recorded_episode_count += int(valid_finished.sum().item())
                    recorded_episode_lengths[reset_ids] = 0
                    episode_ids[reset_ids] = torch_mod.arange(
                        next_episode_id,
                        next_episode_id + len(reset_ids),
                        device=env.device,
                        dtype=torch_mod.long,
                    )
                    next_episode_id += len(reset_ids)
                    steps_in_episode[reset_ids] = 0
                    reach_goals_in_episode[reset_ids] = 0
                    warmup_remaining[reset_ids] = int(cfg.sim_data.get("drop_reset_warmup", 2))
                    previous_qvel[reset_ids] = env._qd[reset_ids]
                    qacc_valid[reset_ids] = False

                before = _extract_snapshot(env)
                qacc = (before["qvel"] - previous_qvel) / policy_dt
                current_qacc_valid = qacc_valid.clone()
                if player.evaluation and step % player.update_checkpoint_freq == 0:
                    player.maybe_load_new_checkpoint()
                action_command = player.get_action(obs, player.is_deterministic)
                raw_action = getattr(player, "last_raw_action", action_command)
                action_for_env = player.preprocess_actions(action_command)
                obs, rewards, dones, infos = player.env_step(vec_env, action_for_env)
                after_action = _extract_snapshot(
                    env,
                    raw_action=raw_action,
                )
                dones_device = dones.to(env.device).bool()
                timeouts = infos.get("time_outs", torch_mod.zeros_like(dones))
                if not isinstance(timeouts, torch_mod.Tensor):
                    timeouts = torch_mod.as_tensor(timeouts, device=env.device)
                timeouts = timeouts.to(env.device).bool()
                successes = env.success_buf.detach().clone().bool()
                failures = env.failure_buf.detach().clone().bool()
                reach_final_goal = env.reach_final_goal.detach().bool()
                reach_goals_in_episode += reach_final_goal.long()
                collect_mask = warmup_remaining <= 0
                if success_buffer is None:
                    remaining = total_target - writer.total_rows
                    eligible_ids = collect_mask.nonzero(as_tuple=False).flatten()
                    if len(eligible_ids) > remaining:
                        collect_mask = torch_mod.zeros_like(collect_mask)
                        collect_mask[eligible_ids[:remaining]] = True
                if torch_mod.any(collect_mask):
                    transition_batch = _make_transition_batch(
                        env,
                        before,
                        after_action,
                        rewards.to(env.device),
                        dones_device,
                        timeouts,
                        episode_ids,
                        steps_in_episode,
                        qacc,
                        current_qacc_valid,
                        successes,
                        failures,
                        block_size,
                        collect_mask,
                        fixed_block_id,
                    )
                    if success_buffer is None:
                        writer.append(transition_batch)
                        recorded_episode_lengths[collect_mask] += 1
                    else:
                        success_buffer.append(transition_batch)
                        episode_is_eligible = (
                            reach_goals_in_episode >= min_reach_goals_per_episode
                        )
                        if truncate_failed_episodes:
                            # Do not write any part of an episode until it reaches
                            # the configured goal count. Once eligible, preserve
                            # the existing policy: commit through each successful
                            # goal and discard a later failed tail.
                            commit_mask = collect_mask & episode_is_eligible & (
                                reach_final_goal | (dones_device & ~failures)
                            ) & ~failures
                        else:
                            # Once the episode qualifies, flush its entire prefix
                            # and stream every subsequent transition, including a
                            # possible terminal failure transition.
                            commit_mask = collect_mask & episode_is_eligible
                        commit_ids = _to_numpy(
                            commit_mask.nonzero(as_tuple=False).flatten(), np.int64
                        )
                        committed_batch, committed_env_ids, committed_lengths = success_buffer.pop(
                            commit_ids
                        )
                        if committed_batch is not None:
                            writer.append(committed_batch)
                            committed_ids_t = torch_mod.as_tensor(
                                committed_env_ids, device=env.device, dtype=torch_mod.long
                            )
                            committed_lengths_t = torch_mod.as_tensor(
                                committed_lengths, device=env.device, dtype=torch_mod.long
                            )
                            recorded_episode_lengths[committed_ids_t] += committed_lengths_t
                            committed_success_segments += int(committed_env_ids.size)

                        if truncate_failed_episodes:
                            failure_ids = _to_numpy(
                                failures.nonzero(as_tuple=False).flatten(), np.int64
                            )
                            if failure_ids.size > 0:
                                discarded_unconfirmed_transitions += success_buffer.discard(failure_ids)
                                discarded_failed_episodes += int(failure_ids.size)
                if bool(cfg.sim_data.get("print_episode_end", False)) and torch_mod.any(dones_device):
                    ended_env_ids = dones_device.nonzero(as_tuple=False).flatten()
                    for ended_env_id in ended_env_ids.tolist():
                        cprint(
                            str({
                                "event": "episode_end",
                                "env_id": int(ended_env_id),
                                "episode_id": int(episode_ids[ended_env_id].item()),
                                "step_num": int(steps_in_episode[ended_env_id].item()) + 1,
                                "num_reach_goal": int(
                                    reach_goals_in_episode[ended_env_id].item()
                                ),
                                "recorded_step_num": int(
                                    recorded_episode_lengths[ended_env_id].item()
                                ),
                            }),
                            "yellow",
                            flush=True,
                        )
                warmup_remaining = torch_mod.clamp(warmup_remaining - 1, min=0)
                steps_in_episode += 1
                previous_qvel.copy_(before["qvel"])
                qacc_valid.fill_(True)
                if step % print_every == 0:
                    active_lengths = recorded_episode_lengths[recorded_episode_lengths > 0]
                    avg_completed_len = (
                        completed_recorded_transition_sum / completed_recorded_episode_count
                        if completed_recorded_episode_count > 0
                        else 0.0
                    )
                    avg_active_len = float(active_lengths.float().mean().item()) if len(active_lengths) > 0 else 0.0
                    print(
                        {
                            "step": step,
                            "written": writer.total_rows,
                            "target": total_target,
                            "completed_recorded_episodes": completed_recorded_episode_count,
                            "avg_completed_recorded_transition_len": round(avg_completed_len, 2),
                            "avg_active_recorded_transition_len": round(avg_active_len, 2),
                            "pending_unconfirmed_transitions": (
                                success_buffer.pending_rows if success_buffer is not None else 0
                            ),
                            "committed_success_segments": committed_success_segments,
                            "discarded_failed_episodes": discarded_failed_episodes,
                            "discarded_insufficient_reach_episodes": (
                                discarded_insufficient_reach_episodes
                            ),
                            "output_dir": str(output_dir),
                        },
                        flush=True,
                    )
        print(
            "Rollout collection complete: "
            f"{output_dir} ({writer.total_rows} transitions, "
            f"avg_completed_recorded_transition_len="
            f"{completed_recorded_transition_sum / completed_recorded_episode_count if completed_recorded_episode_count > 0 else 0.0:.2f}, "
            f"committed_success_segments={committed_success_segments}, "
            f"discarded_failed_episodes={discarded_failed_episodes}, "
            f"discarded_insufficient_reach_episodes="
            f"{discarded_insufficient_reach_episodes}, "
            f"discarded_unconfirmed_transitions={discarded_unconfirmed_transitions})"
        )


@hydra.main(version_base="1.1", config_name="config", config_path="../../../main/cfg")
def main(cfg: DictConfig) -> None:
    _import_isaacgym_first()
    OmegaConf.resolve(cfg)
    seed = int(cfg.get("seed", 42))
    torch_mod, _ = _load_runtime_deps()
    torch_mod.manual_seed(seed)
    torch_mod.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if bool(cfg.get("collect_data", False)):
        if "sim_data" not in cfg:
            raise ValueError("Pass collector settings under +sim_data.*")
        collect_sim_data(cfg)
    else:
        from main.rl.train_rl_games import train_rl_games

        train_rl_games(cfg)


if __name__ == "__main__":
    main()
