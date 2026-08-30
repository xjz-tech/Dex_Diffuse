from __future__ import annotations

import os
import copy
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul, quat_apply
import math
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import (
    aa_to_quat, 
    aa_to_rotmat, 
    quat_to_rotmat, 
    rotmat_to_aa, 
    rotmat_to_quat, 
    rot6d_to_aa,
    rotmat_to_rot6d,
    rot6d_to_rotmat
)
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from ...asset_root import ASSET_ROOT

from lib.utils.object_utils import MeshModelWrapper
from lib.rl.sapg_eval_metrics import pad_and_stack_time_series, repeat_static_per_demo

from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat
from ...utils.adaptive_sampling_scheduler import AdaptiveSamplingScheduler

STRICT_TEST_MODE = False # True

_OBS_NOISE_q = 0.1 # 0.02
_OBS_NOISE_manip_obj_pos =  0.005 # 0.005
_OBS_NOISE_manip_obj_rotang = 2
_OBS_NOISE_tips_pos = 0.005 # 0.005

_ACT_NOISE = 0.05 # 0.05

def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


def calculate_relative_pose(
    a_pose: torch.Tensor,  # [B, 7] (pos_xyz + quat_xyzw)
    b_pose: torch.Tensor,  # [B, 7] (pos_xyz + quat_xyzw)
) -> torch.Tensor:
    """
    Calculate a_pose expressed in b_frame (i.e., T_b_a = T_w_b^{-1} @ T_w_a).

    Args:
        a_pose: Pose in world frame [B, 7] (pos_xyz + quat_xyzw)
        b_pose: b_frame pose in world frame [B, 7] (pos_xyz + quat_xyzw)

    Returns:
        Relative pose of a in b frame [B, 7] (pos_xyz + quat_xyzw)
    """
    a_pos = a_pose[:, :3]
    a_quat = a_pose[:, 3:7]

    b_pos = b_pose[:, :3]
    b_quat = b_pose[:, 3:7]

    # q_b_a = q_w_b^{-1} @ q_w_a = conj(q_w_b) @ q_w_a
    b_quat_conj = quat_conjugate(b_quat)
    rel_quat = quat_mul(b_quat_conj, a_quat)

    # p_b_a = q_w_b^{-1} @ (p_w_a - p_w_b)
    rel_pos = quat_apply(b_quat_conj, a_pos - b_pos)

    return torch.cat([rel_pos, rel_quat], dim=-1)



def random_orientation(
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate random valid quaternions uniformly distributed on SO(3).

    Uses the subgroup algorithm: sample u1, u2, u3 uniformly in [0, 1],
    then construct quaternion components.

    Args:
        batch_size: Number of quaternions to generate
        device: Torch device
        dtype: Torch dtype

    Returns:
        Quaternions in xyzw format, shape [batch_size, 4]
    """
    u1 = torch.rand(batch_size, device=device, dtype=dtype)
    u2 = torch.rand(batch_size, device=device, dtype=dtype)
    u3 = torch.rand(batch_size, device=device, dtype=dtype)

    # Subgroup algorithm for uniform distribution on SO(3)
    # http://planning.cs.uiuc.edu/node198.html
    sqrt_u1 = torch.sqrt(u1)
    sqrt_1_minus_u1 = torch.sqrt(1.0 - u1)

    x = sqrt_1_minus_u1 * torch.sin(2 * math.pi * u2)
    y = sqrt_1_minus_u1 * torch.cos(2 * math.pi * u2)
    z = sqrt_u1 * torch.sin(2 * math.pi * u3)
    w = sqrt_u1 * torch.cos(2 * math.pi * u3)

    return torch.stack([x, y, z, w], dim=-1)  # xyzw format


def random_orientation_in_cone(
    batch_size: int,
    max_tilt_angle_deg: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample random quaternion offsets uniformly on a spherical cap.
    The resulting rotation tilts the z-axis by at most max_tilt_angle_deg degrees.
    Uniform on the cap: cos(theta) ~ U[cos(theta_max), 1], phi ~ U[0, 2*pi).

    Args:
        batch_size: Number of quaternions to generate
        max_tilt_angle_deg: Maximum tilt angle in degrees (180 = full SO(3))
        device: Torch device
        dtype: Torch dtype

    Returns:
        Quaternions in xyzw format, shape [batch_size, 4]
    """
    if max_tilt_angle_deg >= 180.0:
        return random_orientation(batch_size, device, dtype)

    theta_max = max_tilt_angle_deg * math.pi / 180.0

    # Uniform on spherical cap: cos(theta) ~ U[cos(theta_max), 1]
    cos_theta = torch.rand(batch_size, device=device, dtype=dtype) * (1.0 - math.cos(theta_max)) + math.cos(theta_max)
    theta = torch.acos(cos_theta)

    # Uniform azimuth
    phi = torch.rand(batch_size, device=device, dtype=dtype) * 2 * math.pi

    # Rotation axis in xy-plane (perpendicular to z)
    axis_x = torch.cos(phi)
    axis_y = torch.sin(phi)

    # Axis-angle -> quaternion: q = [sin(theta/2)*axis, cos(theta/2)]
    half_theta = theta / 2.0
    sin_half = torch.sin(half_theta)
    cos_half = torch.cos(half_theta)

    x = sin_half * axis_x
    y = sin_half * axis_y
    z = torch.zeros_like(phi)
    w = cos_half

    return torch.stack([x, y, z, w], dim=-1)  # xyzw format


def add_rotquat_noise(
    base_rotquat: torch.Tensor, ## [B, 4] xyzw
    noise_angle: torch.Tensor, ## [B] degree (rotate angle)
):
    """
    Add random rotation noise to quaternions via rotation matrices.

    Args:
        base_rotquat: [B, 4] (xyzw)
        noise_angle:  [B] rotation angle in degrees

    Returns:
        noised_rotquat: [B, 4] (xyzw)
    """
    B = base_rotquat.shape[0]
    device = base_rotquat.device
    dtype = base_rotquat.dtype

    base_rotmat = quat_to_rotmat(base_rotquat[..., [3, 0, 1, 2]])  # [B, 3, 3]

    axis = torch.randn(B, 3, device=device, dtype=dtype)
    axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    angle_rad = noise_angle * math.pi / 180.0  # [B]
    aa_noise = axis * angle_rad.unsqueeze(1)  # [B, 3]

    noise_rotmat = aa_to_rotmat(aa_noise)  # [B, 3, 3]
    noised_rotmat = noise_rotmat @ base_rotmat  # [B, 3, 3]
    noised_rotquat = rotmat_to_quat(noised_rotmat)[..., [1, 2, 3, 0]]  # [B, 4]

    # numerical safety
    noised_rotquat = noised_rotquat / (
        noised_rotquat.norm(dim=1, keepdim=True) + 1e-8
    )

    return noised_rotquat

@torch.jit.script
def ensure_quat_w_positive(
    base_rotquat: torch.Tensor, # [B, 4] xyzw
):
    rot_quat = base_rotquat.clone()
    mask = rot_quat[:, 3] < 0
    rot_quat[mask] = -rot_quat[mask]
    return rot_quat

@torch.jit.script
def add_gaussian_observation_noise(
    obs_tensor: torch.Tensor,
    mu: float,
    sigma: float,
    scale_factor: float,
) -> torch.Tensor:
    """
    Add gaussian noise to observation tensor.
    
    Args:
        obs_tensor: Observation tensor to add noise to
        mu: Mean of gaussian noise
        sigma: Standard deviation of gaussian noise
        scale_factor: Scaling factor for noise (0.0 to 1.0)
    
    Returns:
        Observation tensor with noise added
    """
    if scale_factor <= 0.0:
        return obs_tensor
    
    # Scale noise according to schedule
    scaled_sigma = sigma * scale_factor
    scaled_mu = mu * scale_factor
    
    # Generate gaussian noise
    noise = torch.randn_like(obs_tensor) * scaled_sigma + scaled_mu
    
    return obs_tensor + noise


def apply_observation_noise(
    obs_tensor: torch.Tensor,
    noise_config: dict,
    scale_factor: float,
) -> torch.Tensor:
    """
    Apply observation noise based on configuration.
    
    Args:
        obs_tensor: Observation tensor to add noise to
        noise_config: Dictionary containing noise configuration
        scale_factor: Scaling factor for noise (0.0 to 1.0)
    
    Returns:
        Observation tensor with noise added
    """
    if scale_factor <= 0.0 or noise_config is None:
        return obs_tensor
    
    distribution = noise_config.get("distribution", "gaussian")
    if distribution != "gaussian":
        raise ValueError(f"Unsupported noise distribution: {distribution}. Only 'gaussian' is supported.")
    
    range_val = noise_config.get("range", [0.0, 0.0])
    mu = range_val[0]
    sigma = range_val[1]
    
    return add_gaussian_observation_noise(obs_tensor, mu, sigma, scale_factor)

class SinDexHandManipRHEnv(VecTask):

    side = "right"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.traj_steps_limit = self.cfg["env"].get("trajStepsLimit", None)  # Max total steps per trajectory before reset
        self.reset_on_reach_goal = self.cfg["env"].get("resetOnReachGoal", False)
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]
        self.sim_data_collection = self.cfg["env"].get("simDataCollection", False)
        # collect_data=true restores a player rather than running PPO, but its
        # rollouts use the training-time task dynamics requested by the script.
        self.training_or_collection = self.training or self.sim_data_collection

        self.disableBackDrive = self.cfg["env"]["disableBackDrive"]

        if not hasattr(self, "dexhand"):
            self.dexhand = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")

        self.enable_asymmetric_actor_critic = self.cfg["env"]["enable_asymmetric_actor_critic"]
        self.extra_num_states = self.cfg["env"]["extra_num_states"]

        # Asymmetric actor-critic: critic gets privileged info (states) that actor doesn't see
        # When disabled (extra_num_states=0): num_states=0, no states_buf allocated, behavior unchanged
        self.num_states = self.extra_num_states

        # State keys for asymmetric critic: first clones all privileged obs, then appends extra keys
        self._state_keys = self.cfg["env"].get("stateKeys", [])

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand.Kp_rot
            self.Ki_rot = self.dexhand.Ki_rot
            self.Kd_rot = self.dexhand.Kd_rot

            self.Kp_pos = self.dexhand.Kp_pos
            self.Ki_pos = self.dexhand.Ki_pos
            self.Kd_pos = self.dexhand.Kd_pos

        self.cfg["env"]["numActions"] = self.dexhand.n_dofs
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.act_scale = self.cfg["env"]["actionScale"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # load dexhand dof kp and kd
        self.dexhand_dof_kp = self.dexhand.dof_Kp
        self.dexhand_dof_kd = self.dexhand.dof_Kd
        self.dexhand_dof_armature = self.dexhand.dof_armature if hasattr(self.dexhand, "dof_armature") else None
        self.dexhand_dof_effort = self.dexhand.dof_effort if hasattr(self.dexhand, "dof_effort") else None
        self.dexhand_dof_friction = self.dexhand.dof_friction if hasattr(self.dexhand, "dof_friction") else None

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation

        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]
        # Values to be filled in at runtime
        self.states = {}
        self.dexhand_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.objs_assets_target = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self.last_actions = None  # Last actions to be deployed

        self.last_obs_dict_queue = []
        self.last_obs_dict_queue_size = 2

        self.dataIndices = self.cfg["env"]["dataIndices"]

        self.palmFacing = self.cfg["env"]["palmFacing"]

        self.act_style = self.cfg["env"]["actStyle"]

        self.obs_enable_action = self.cfg["env"]["obsEnableAction"]
        self.obs_enable_history = self.cfg["env"]["obsEnableHistory"]
        self.n_history = self.cfg["env"]["nHistory"]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.obs_enable_geometry = self.cfg["env"]["obsEnableGeometry"]
        self.obs_enable_task_emb = self.cfg["env"]["obsEnableTaskEmb"]
        self.obs_enable_handpos = self.cfg["env"]["obsEnableHandpos"]
        self.obs_only_fingertips = self.cfg["env"]["obsOnlyFingertips"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.enable_latency_tracking = self.cfg["env"].get("enableLatencyTracking", False)
        self.skip_steps_min = self.cfg["env"].get("skipStepsMin", 15)
        self.skip_steps_max = self.cfg["env"].get("skipStepsMax", 35)
        self.reverse_target_prob = self.cfg["env"].get("reverseTargetProb", 0.15)
        self.fixed_tolerance_steps = self.cfg["env"].get("fixedToleranceSteps", 0)
        self.failure_tolerance_scale = self.cfg["env"].get("FailureToleranceScale", 1.5)
        self.num_frames_to_stay_lower_bound = self.cfg["env"].get("numFramesToStayLowerBound", 5)
        self.num_frames_to_stay_upper_bound = self.cfg["env"].get("numFramesToStayUpperBound", 15)

        self.enable_reset_pool = self.cfg["env"].get("enableResetPool", False)
        self.reset_pool_size = self.cfg["env"].get("resetPoolSize", 10)
        self.reset_pool_sample_prob = self.cfg["env"].get("resetPoolSampleProb", 0.5)
        self.reset_pool_update_prob = self.cfg["env"].get("resetPoolUpdateProb", 0.7)

        self.enable_cross_trajectory_reset = self.cfg["env"].get("enableCrossTrajectoryReset", False)
        self.cross_trajectory_goal_prob = float(
            self.cfg["env"].get("crossTrajectoryGoalProb", self.cfg["env"].get("crossTrajectoryResetProb", 0.5))
        )

        self.enable_adapt_sampling = self.cfg["env"].get("enableAdaptSampling", False)
        self.adapt_sampling_prob_lower_bound = self.cfg["env"].get("adaptSamplingProbLowerBound", 0.001)
        self.adapt_sampling_prob_upper_bound = self.cfg["env"].get("adaptSamplingProbUpperBound", 0.04)
        self.adapt_sampling_update_interval = self.cfg["env"].get("adaptSamplingUpdateInterval", 3200)
        self.adapt_sampling_scheduler = AdaptiveSamplingScheduler(
            prob_lower_bound=self.adapt_sampling_prob_lower_bound,
            prob_upper_bound=self.adapt_sampling_prob_upper_bound,
            prefer_low_metric=True,
        )

        self.enable_control_signal_mask = self.cfg["env"].get("enableControlSignalMask", False)
        self.control_signal_mask_prob = self.cfg["env"].get("controlSignalMaskProb", 0.5)
        self.control_signal_mask_dof_num = self.cfg["env"].get("controlSignalMaskDofNum", 2)
        self.control_signal_mask_frames_lower_bound = self.cfg["env"].get("controlSignalMaskFramesLowerBound", 10)
        self.control_signal_mask_frames_upper_bound = self.cfg["env"].get("controlSignalMaskFramesUpperBound", 30)
        self.prev_actions = None

        self.time_penalty = self.cfg["env"].get("timePenalty", 0.0)

        self.success_obj_pos_thres = self.cfg["env"].get("successObjPosThres", 0.004)
        self.success_thumb_tip_pos_thres = self.cfg["env"].get("successThumbTipPosThres", 0.008)
        self.success_index_tip_pos_thres = self.cfg["env"].get("successIndexTipPosThres", 0.025)
        self.success_middle_tip_pos_thres = self.cfg["env"].get("successMiddleTipPosThres", 0.025)
        self.success_pinky_tip_pos_thres = self.cfg["env"].get("successPinkyTipPosThres", 0.025)
        self.success_ring_tip_pos_thres = self.cfg["env"].get("successRingTipPosThres", 0.025)
        self.success_obj_rot_thres = self.cfg["env"].get("successObjRotThres", 6)
        self.failure_obj_pos_thres = self.cfg["env"].get(
            "failureObjPosThres", self.success_obj_pos_thres
        )
        self.failure_thumb_tip_pos_thres = self.cfg["env"].get(
            "failureThumbTipPosThres", self.success_thumb_tip_pos_thres
        )
        self.failure_index_tip_pos_thres = self.cfg["env"].get(
            "failureIndexTipPosThres", self.success_index_tip_pos_thres
        )
        self.failure_middle_tip_pos_thres = self.cfg["env"].get(
            "failureMiddleTipPosThres", self.success_middle_tip_pos_thres
        )
        self.failure_pinky_tip_pos_thres = self.cfg["env"].get(
            "failurePinkyTipPosThres", self.success_pinky_tip_pos_thres
        )
        self.failure_ring_tip_pos_thres = self.cfg["env"].get(
            "failureRingTipPosThres", self.success_ring_tip_pos_thres
        )
        self.failure_obj_rot_thres = self.cfg["env"].get(
            "failureObjRotThres", self.success_obj_rot_thres
        )
        self.invalid_obj_pos_thres = self.cfg["env"].get("invalidObjPosThres", 0.15)

        ## randomization
        self.random_obj_scales = self.cfg["env"]["randomObjectScales"]
        self.random_next_frames = self.cfg["env"]["randomNextFrames"]
        # random force
        self.random_force_scale = self.cfg["env"]["randomForceScale"]
        self.random_force_prob_range = self.cfg["env"]["randomForceProbRange"]
        self.random_force_decay = self.cfg["env"]["randomForceDecay"]
        self.random_force_decay_interval = self.cfg["env"]["randomForceDecayInterval"]
        # random_force_prob will be initialized in init_data() after self.device and self.num_envs are available
        self.random_force_prob = None

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        self.collect_data = self.cfg["env"].get("collectData", False)

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

        self.fix_dexhand_base = self.cfg["env"].get("fixDexhandBase", False)

        self.gravity_z = self.cfg["env"].get("gravityZ", -9.8)
        self.random_wrist_orientation = self.cfg["env"].get("randomWristOrientation", False)
        self.random_wrist_max_tilt_angle = self.cfg["env"].get("randomWristMaxTiltAngle", 180.0)

        self.scale_factor = 1.0

        # observation noise randomization config
        self.observation_noise_scale_factor = 0.0
        self.obs_noise_params = self.cfg["task"].get("observation_randomize_params", None)
        if self.obs_noise_params is not None and self.training:
            self.obs_noise_schedule = self.obs_noise_params.get("noise_schedule", {})
            self.obs_noise_proprioception = self.obs_noise_params.get("proprioception", {})
            self.obs_noise_privileged = self.obs_noise_params.get("privileged", {})
            self.obs_noise_target = self.obs_noise_params.get("target", {})
        else:
            self.obs_noise_schedule = {}
            self.obs_noise_proprioception = {}
            self.obs_noise_privileged = {}
            self.obs_noise_target = {}

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_effort_limits = None  # Actuator effort limits for dexhand_r
        self._dexhand_dof_speed_limits = None  # Actuator speed limits for dexhand_r
        self._global_dexhand_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        TARGET_OBS_DIM = (
            0
            + 
            (
                (3 + 3 + 4 + 4) +
                (
                    (self.dexhand.n_finger_tips * 3 * 2 if self.obs_only_fingertips else (self.dexhand.n_bodies - 1) * 3 * 2) 
                    if self.obs_enable_handpos else 0
                )
                # + self.dexhand.n_bodies
                + (
                    self.dexhand.n_dofs if self.obs_enable_action else 0
                )
            )
            * self.obs_future_length
            + (32 if self.obs_enable_task_emb else 0)
            + (4 * self.dexhand.n_finger_tips if self.obs_enable_geometry else 0)
            # + self.dexhand.n_dofs
        )
        self.obs_dict_target_keys = []
        if self.obs_enable_handpos:
            if self.obs_only_fingertips:
                self.obs_dict_target_keys.extend([
                    # "fingertips_pos", 
                    # "delta_fingertips_pos"
                    "fingertips_pos_rel_wrist",
                    "delta_fingertips_pos_rel_wrist",
                ])
            else:
                self.obs_dict_target_keys.extend([
                    "joints_pos", 
                    "delta_joints_pos"
                ])
        self.obs_dict_target_keys.extend([
            # "manip_obj_pos", 
            # "delta_manip_obj_pos", 
            # "manip_obj_quat", 
            # "delta_manip_obj_quat"
            "manip_obj_pos_rel_wrist",
            "delta_manip_obj_pos_rel_wrist",
            "manip_obj_quat_rel_wrist",
            "delta_manip_obj_quat_rel_wrist",
        ])
        if self.obs_enable_task_emb:
            self.obs_dict_target_keys.extend(["task_embedding"])
        if self.obs_enable_geometry:
            self.obs_dict_target_keys.extend(["tip2object_distance", "tip2object_normal"])
        if self.obs_enable_action:
            self.obs_dict_target_keys.extend(["curr_targets"])
        
        if self.obs_enable_history:
            self._history_buf = torch.zeros(
                (self.num_envs, self.n_history, self.dexhand.n_dofs * 2),
                device=self.device,
                dtype=torch.float,
            )
            self.obs_dict.update(
                {
                    'history': torch.zeros(
                        (self.num_envs, self.n_history, self.dexhand.n_dofs * 2),
                        device=self.device,
                        dtype=torch.float,
                    )
                }
            )
            obs_space = self.obs_space.spaces
            obs_space['history'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_history, self.dexhand.n_dofs * 2),
            )
            self.obs_space = spaces.Dict(obs_space)

        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        # Asymmetric actor-critic: allocate states_buf and state_space for privileged critic info
        if self.num_states > 0:
            self.states_buf = torch.zeros(
                (self.num_envs, self.num_states),
                device=self.device,
                dtype=torch.float,
            )
            self.state_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_states,),
            )

        default_pose = torch.ones(self.dexhand.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_default_dof_pos = default_pose.clone().detach().to(self.sim_device)
        
        # initial base dof pos (should be update)
        self.base_dof_pos = torch.zeros((self.num_envs, self.dexhand.n_dofs), device=self.device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts = self.demo_data["obj_verts"]
        self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]

        ## Init reset pool
        self._reset_pool = {}
        if self.enable_reset_pool:
            assert self.enable_latency_tracking, "self.enable_latency_tracking must be True when self.enable_reset_pool is True"
            self._reset_pool["opt_dof_pos"] = self.demo_data["opt_dof_pos"].clone().unsqueeze(2).repeat(1, 1, self.reset_pool_size, 1)
            self._reset_pool["obj_trajectory"] = self.demo_data["obj_trajectory"].clone().unsqueeze(2).repeat(1, 1, self.reset_pool_size, 1, 1)

        # Reset all environments
        self.global_cur_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stable_frames_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.num_frames_to_stay_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.consecutive_reach_goal_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.consecutive_reach_frames_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Buffer to store error at reach_final_goal (updated only when reach_final_goal is True)
        self.last_reach_error = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float)
        self.skip_steps_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int32) * self.skip_steps_min
        self.failure_progress_buf = torch.zeros(self.num_envs, device=self.device)
        self.control_signal_mask_frames_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.control_signal_mask_dof_indices = torch.zeros(self.num_envs, self.control_signal_mask_dof_num, device=self.device, dtype=torch.int32)
        # Direction tracking for forward/reverse trajectory playback
        # 1 = forward, -1 = reverse
        self.traj_direction = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # Counter for steps in current trajectory direction (forward or reverse)
        self.traj_steps_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.wrist_orientation_offset = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float32)
        self.wrist_orientation_offset = self.wrist_orientation_offset.unsqueeze(0).repeat(self.num_envs, 1)
        self.is_target_cross = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = self.gravity_z
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, -0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015

        self._table_surface_z = table_pos.z + table_half_height
        # self.dexhand_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        if self.palmFacing == 'z_down':
            mujoco2gym_transf = np.eye(4)
            mujoco2gym_transf[:3, :3] = \
                aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        elif self.palmFacing == 'z_up':
            mujoco2gym_transf = np.eye(4)
            mujoco2gym_transf[:3, :3] = \
                aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))
        else:
            raise NotImplementedError
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_dict = {}
        for dataset_type in dataset_list:
            dataset_kwargs = {}
            if dataset_type == "nokov3":
                dataset_kwargs = {
                    "data_dir": self.cfg["env"].get("nokov3DataDir", "data/NOKOV-v3"),
                    "retarget_dir": self.cfg["env"].get(
                        "nokov3RetargetDir", "data/retargeting/NOKOV-v3"
                    ),
                }
            self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side=self.side,
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand,
                embodiment=self.cfg["env"]["dexhand"],
                **dataset_kwargs,
            )

        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = self.fix_dexhand_base
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 200000
        dexhand_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        dexhand_dof_stiffness = torch.tensor(
            self.dexhand_dof_kp,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            self.dexhand_dof_kd,
            dtype=torch.float,
            device=self.sim_device,
        )
        if self.dexhand_dof_armature is not None:
            dexhand_dof_armature = torch.tensor(
                self.dexhand_dof_armature,
                dtype=torch.float,
                device=self.sim_device,
            )
        if self.dexhand_dof_effort is not None:
            dexhand_dof_effort = torch.tensor(
                self.dexhand_dof_effort,
                dtype=torch.float,
                device=self.sim_device,
            )
        if self.dexhand_dof_friction is not None:
            dexhand_dof_friction = torch.tensor(
                self.dexhand_dof_friction,
                dtype=torch.float,
                device=self.sim_device,
            )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.05
            element.torsion_friction = 0.05

        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_props_asset)

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        print(f"Num dexhand Bodies: {self.num_dexhand_bodies}")
        print(f"Num dexhand DOFs: {self.num_dexhand_dofs}")

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]
            if self.dexhand_dof_armature is not None:
                dexhand_dof_props["armature"][i] = dexhand_dof_armature[i]
            else:
                dexhand_dof_props["armature"][i] = 0.0
            if self.dexhand_dof_effort is not None:
                dexhand_dof_props["effort"][i] = dexhand_dof_effort[i]
            if self.dexhand_dof_friction is not None:
                dexhand_dof_props["friction"][i] = dexhand_dof_friction[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.current_dexhand_dof_lower_np = np.array(self.dexhand_dof_lower_limits).reshape(1, self.num_dexhand_dofs).repeat(self.num_envs, axis=0)
        self.current_dexhand_dof_upper_np = np.array(self.dexhand_dof_upper_limits).reshape(1, self.num_dexhand_dofs).repeat(self.num_envs, axis=0)
        self.global_dexhand_dof_lower_np = np.array(self.dexhand_dof_lower_limits).reshape(1, self.num_dexhand_dofs).repeat(self.num_envs, axis=0)
        self.global_dexhand_dof_upper_np = np.array(self.dexhand_dof_upper_limits).reshape(1, self.num_dexhand_dofs).repeat(self.num_envs, axis=0)
        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)

        # compute aggregate size
        num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        num_dexhand_shapes = self.gym.get_asset_rigid_shape_count(dexhand_asset)

        self.dexhand_rs = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        def segment_data(k):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        # self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
        # self.demo_data = self.pack_data(self.demo_data)
        self.demo_data = [segment_data(i) for i in tqdm(range(len(self.dataIndices)))]
        self.demo_data = self.pack_data(self.demo_data)
        self.envidx_to_demoidx = torch.tensor([i % len(self.dataIndices) for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        if self.enable_adapt_sampling:
            self.adapt_sampling_scheduler.initialize(len(self.dataIndices))
            self.adapt_sampling_scheduler.reset(
                len(self.dataIndices), 
                torch.zeros(len(self.dataIndices), device=self.device, dtype=torch.int32),
            )
            self.envidx_to_demoidx = self.adapt_sampling_scheduler.get_indices(num_envs=self.num_envs)

            ## metrics
            self.consecutive_reach_goal_max_buf = torch.zeros(len(self.dataIndices), device=self.device, dtype=torch.int32)
            self.consecutive_reach_frames_max_buf = torch.zeros(len(self.dataIndices), device=self.device, dtype=torch.int32)
        
        assert len(set(self.demo_data["obj_mesh_path"])) == 1, "object_mesh is not shared by all envs"
        self.object_model = MeshModelWrapper(
            mesh_path=self.demo_data["obj_mesh_path"][0],
            device=self.device
        )

        # Create environments
        self.manip_obj_mass = []
        self.manip_obj_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            current_asset, sum_rigid_body_count, sum_rigid_shape_count, obj_scale, obj_mass, target_asset, target_rigid_body_count, target_rigid_shape_count = self._create_obj_assets(i)
            max_agg_bodies = (
                num_dexhand_bodies
                + 1  # table
                + sum_rigid_body_count
                + target_rigid_body_count
                + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
            )
            max_agg_shapes = (
                num_dexhand_shapes
                + 1  # table
                + sum_rigid_shape_count
                + target_rigid_shape_count
                + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
                + (1 if self._record else 0)
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand_r should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )
            
            # multiview camera handler for view rendering
            if i == 0 and self.multiview_camera_handlers is not None:
                cameras = self.create_multiview_cameras(
                    env=env_ptr, 
                    isaac_gym=self.gym
                )
                for c in cameras:
                    self.multiview_camera_handlers.append(c)

            # Create dexhand_r
            dexhand_actor = self.gym.create_actor(
                env_ptr,
                dexhand_asset,
                self.dexhand_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),  # ! some hand need to allow self-collision
            )
            # self.gym.set_actor_scale(env_ptr, dexhand_actor, 1.05)

            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, dexhand_dof_props)

            # self.gym.set_actor_scale(
            #     env_ptr, dexhand_actor, 0.95
            # )

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_handle, _ = self._create_obj_actor(
                env_ptr, i, current_asset
            )  # the handle is all the same for all envs
            self.gym.set_actor_scale(env_ptr, self.obj_handle, obj_scale)
            if not self.headless:
                self.obj_target_handle, _ = self._create_obj_target_actor(
                    env_ptr, i, target_asset
                )
                self.gym.set_actor_scale(env_ptr, self.obj_target_handle, obj_scale)
            obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_handle)
            obj_props[0].mass = min(0.5, obj_props[0].mass)  # * we only consider the mass less than 500g
            # ? caculate mass by density
            if obj_mass is not None:
                obj_props[0].mass = obj_mass

            # ! Updating the mass and scale might slightly alter the inertia tensor;
            # ! however, because the magnitude of our modifications is minimal, we temporarily neglect this effect.
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_handle, obj_props)
            self.manip_obj_mass.append(obj_props[0].mass)
            self.manip_obj_com.append(torch.tensor([obj_props[0].com.x, obj_props[0].com.y, obj_props[0].com.z]))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_actor)

        self.manip_obj_mass = torch.tensor(self.manip_obj_mass, device=self.device)
        self.manip_obj_com = torch.stack(self.manip_obj_com, dim=0).to(self.device)

        # Setup data
        self.init_data()
    
    def _update_object_mass(self):
        manip_obj_mass = []
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_handle)
            manip_obj_mass.append(obj_props[0].mass)
        self.manip_obj_mass = torch.tensor(self.manip_obj_mass, device=self.device)

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
                for i in range(self.dexhand.n_bodies)
            ]
        # ? <<<

        self._manip_obj_handle = self.gym.find_actor_handle(env_ptr, "manip_obj")
        self._manip_obj_root_state = self._root_state[:, self._manip_obj_handle, :]
        if not self.headless:
            self._target_obj_handle = self.gym.find_actor_handle(env_ptr, "manip_obj_target")
            self._target_obj_root_state = self._root_state[:, self._target_obj_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_handle, "base"
        )
        self._manip_obj_cf = self.net_cf[:, self._manip_obj_rigid_body_handle, :]

        self.dexhand_root_state = self._root_state[:, dexhand_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize random_force_prob using log-uniform distribution
        prob_range = torch.tensor(self.random_force_prob_range, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(prob_range[0]) - torch.log(prob_range[1]))
            * torch.rand(self.num_envs, device=self.device) 
            + torch.log(prob_range[1])
        )

        # Initialize indices
        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        if not self.headless:
            self._global_target_obj_indices = torch.tensor(
                [self.gym.find_actor_index(env, "manip_obj_target", gymapi.DOMAIN_SIM) for env in self.envs],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 3
        self.tips_contact_history = torch.ones(self.num_envs, CONTACT_HISTORY_LEN, self.dexhand.n_finger_tips, device=self.device).bool()

    def pack_data(self, data):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            return pad_and_stack_time_series(stack_data)

        for k in list(data[0].keys()) + ['fingertips_pos']:
            if k in ["mano_joints", "opt_mano_joints", \
                "mano_joints_velocity", "opt_mano_joints_velocity"]:
                mano_joints = []
                for d in data:
                    mano_joints.append(
                        torch.concat(
                            [
                                d[k][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.body_names
                                if self.dexhand.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                    )
                packed_data[k] = fill_data(mano_joints)
            elif k == 'fingertips_pos':
                fingertips_pos = []
                for d in data:
                    fingertips_pos.append(
                        torch.concat(
                            [
                                d['mano_joints'][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.fingertip_body_names
                            ],
                            dim=-1,
                        )
                    )
                packed_data[k] = fill_data(fingertips_pos)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k == "obj_verts":
                    # obj_verts might have different structure, handle separately if needed
                    packed_data[k] = torch.stack(stack_data).squeeze()
                elif k == "task_embedding":
                    packed_data[k] = repeat_static_per_demo(stack_data, int(max_len))
                else:
                    # Use fill_data for all time-series tensors including task_embedding
                    # This pads sequences to the same length before stacking
                    packed_data[k] = fill_data(stack_data)
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]

            ## ?? >>> for checking the memory usage
            import sys
            print(f'Processing {k} demo_data')
            # print allocated memory (GB)
            if torch.is_tensor(packed_data[k]):
                print(f'Shape: {packed_data[k].shape}')
                print(f'GPU memory used: {packed_data[k].numel() * packed_data[k].element_size() / 1024 / 1024 / 1024:.2f} GB')
            else:
                print(f'Allocated memory: {sys.getsizeof(packed_data[k])} bytes')
            print('-' * 100)
            ## ?? <<<
        return packed_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i):
        obj_id = self.demo_data["obj_id"][self.envidx_to_demoidx[i]]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
            target_asset = self.objs_assets_target[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            # asset_options.max_linear_velocity = 0.2
            # asset_options.max_angular_velocity = 0.2
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            # asset_options.use_mesh_materials = False
            asset_options.density = 400  # * the average density of low-fill-rate 3D-printed models
            # asset_options.density = 1000
            current_asset = self.gym.load_asset(
                self.sim, *os.path.split(self.demo_data["obj_urdf_path"][self.envidx_to_demoidx[i]]), asset_options
            )

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 4.0 # 4.0  # * We increase the friction coefficient to compensate for missing skin deformation friction in simulation. See the Appx for details.
                element.rolling_friction = 0.05 # 0.05
                element.torsion_friction = 0.05 # 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

            target_asset_options = copy.deepcopy(asset_options)
            target_asset_options.disable_gravity = True
            target_asset = self.gym.load_asset(
                self.sim, *os.path.split(self.demo_data["obj_urdf_path"][self.envidx_to_demoidx[i]]), target_asset_options
            )
            self.objs_assets_target[obj_id] = target_asset

        # * load assigned scale and mass for the object if available
        scale = 1.0
        if self.training_or_collection:
            random_lower_scale = self.random_obj_scales[0]
            random_upper_scale = self.random_obj_scales[1]
            scale = np.random.uniform(random_lower_scale, random_upper_scale)

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        target_rigid_body_count = self.gym.get_asset_rigid_body_count(target_asset)
        target_rigid_shape_count = self.gym.get_asset_rigid_shape_count(target_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass, target_asset, target_rigid_body_count, target_rigid_shape_count

    def _create_obj_target_actor(self, env_ptr, i, current_asset):

        obj_transf = self.demo_data["obj_trajectory"][self.envidx_to_demoidx[i]][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        # ? target object is for visualization only, no collision with any objects
        obj_target_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj_target", self.num_envs + 2, 0b1)
        obj_target_index = self.gym.get_actor_index(env_ptr, obj_target_actor, gymapi.DOMAIN_SIM)

        # * set target object rigid body color
        c = gymapi.Vec3(135 / 255, 206 / 255, 235 / 255)
        self.gym.set_rigid_body_color(env_ptr, obj_target_actor, 0, gymapi.MESH_VISUAL, c)

        return obj_target_actor, obj_target_index

    def _create_obj_actor(self, env_ptr, i, current_asset):

        obj_transf = self.demo_data["obj_trajectory"][self.envidx_to_demoidx[i]][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

        # ? object actor filter bit is always 1
        obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj", i, 0)

        # Fix Isaac Gym bug: rolling_friction and torsion_friction set on asset
        # do NOT propagate to actor instances. Re-apply at actor level.
        actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        for sp in actor_shape_props:
            sp.rolling_friction = 0.05
            sp.torsion_friction = 0.05
        self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, actor_shape_props)

        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        scene_objs = self.demo_data["scene_objs"][self.envidx_to_demoidx[i]]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
        # add dummy scene object
        MAX_SCENE_OBJS = 5 + (0 if not self.headless else 0)
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
            # ? collision filter bit is always 0b11111111, never collide with anything (except the ground)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0, 0, -1.0)
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                pose,
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        # * just for visualization purposes, add a small sphere at the finger positions
        if not self.headless:
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        return obj_actor, obj_index

    def _update_states(self):
        enable_hardcoded_noise = self.training or STRICT_TEST_MODE
        rand_q = (
            _OBS_NOISE_q * torch.rand_like(self._q[:, :])
            if enable_hardcoded_noise
            else torch.zeros_like(self._q[:, :])
        )
        if self.obs_enable_history:
            current_qa = torch.cat(
                [
                    self._q[:, :] + rand_q,
                    self.curr_targets[:, :],
                ],
                dim=-1,
            )
            self._history_buf = torch.cat(
                [
                    self._history_buf[:, 1:],
                    current_qa[:, None, :],
                ],
                dim=1,
            )
            self.states.update(
                {
                    "qa_history": self._history_buf,
                }
            )

        if self.training:
            self.states.update(
                {
                    "origin_q": self._q[:, :],
                    "origin_cos_q": torch.cos(self._q[:, :]),
                    "origin_sin_q": torch.sin(self._q[:, :]),
                    "wrist_quat": self._base_state[:, 3:7],

                    "q": self._q[:, :] + rand_q,
                    "cos_q": torch.cos(self._q[:, :] + rand_q),
                    "sin_q": torch.sin(self._q[:, :] + rand_q),
                    "dq": self._qd[:, :],
                    "base_state": self._base_state[:, :],

                    "q_curr_targets": self.curr_targets[:, :],
                }
            )

        else:
            norm_q = torch_jit_utils.unscale(
                self._q[:, :],
                self.dexhand_dof_lower_limits,
                self.dexhand_dof_upper_limits,
            )

            norm_curr_targets = torch_jit_utils.unscale(
                self.curr_targets[:, :],
                self.dexhand_dof_lower_limits,
                self.dexhand_dof_upper_limits,
            )

            self.states.update(
                {
                    "norm_q": norm_q,

                    "q": self._q[:, :] + rand_q,
                    "cos_q": torch.cos(self._q[:, :] + rand_q),
                    "sin_q": torch.sin(self._q[:, :] + rand_q),
                    "dq": self._qd[:, :],
                    "base_state": self._base_state[:, :],

                    "q_curr_targets": self.curr_targets[:, :],
                    "norm_curr_targets": norm_curr_targets,
                }
            )

        self.states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :][:, :10] for k in self.dexhand.body_names],
            dim=1,
        )

        self.states["tips_pos"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :][:, :3] for k in self.dexhand.fingertip_body_names],
            dim=1,
        )
        if enable_hardcoded_noise:
            rand_tips_pos = _OBS_NOISE_tips_pos * \
                torch.rand_like(self.states["tips_pos"])
            self.states["tips_pos"] = self.states["tips_pos"] + rand_tips_pos

        if enable_hardcoded_noise:
            rand_manip_obj_pos = _OBS_NOISE_manip_obj_pos * \
                torch.rand_like(self._manip_obj_root_state[:, :3])
            rand_manip_obj_rot_angle = _OBS_NOISE_manip_obj_rotang * \
                torch.rand_like(self._manip_obj_root_state[:, 0])
            noised_manip_obj_quat = add_rotquat_noise(
                self._manip_obj_root_state[:, 3:7],
                rand_manip_obj_rot_angle
            )
            noised_manip_obj_rot6d = rotmat_to_rot6d(
                quat_to_rotmat(
                    noised_manip_obj_quat[..., [3, 0, 1, 2]]
                )
            )
            self.states.update(
                {
                    "origin_manip_obj_pos": self._manip_obj_root_state[:, :3],
                    "origin_manip_obj_quat": self._manip_obj_root_state[:, 3:7],

                    "manip_obj_pos": self._manip_obj_root_state[:, :3] + rand_manip_obj_pos,
                    "manip_obj_quat": ensure_quat_w_positive(noised_manip_obj_quat),
                    "manip_obj_rot6d": noised_manip_obj_rot6d,
                    "manip_obj_vel": self._manip_obj_root_state[:, 7:10],
                    "manip_obj_ang_vel": self._manip_obj_root_state[:, 10:],
                }
            )
        else:
            manip_obj_rot6d = rotmat_to_rot6d(
                quat_to_rotmat(
                    self._manip_obj_root_state[:, 3:7][..., [3, 0, 1, 2]]
                )
            )
            self.states.update(
                {
                    "manip_obj_pos": self._manip_obj_root_state[:, :3],
                    "manip_obj_quat": ensure_quat_w_positive(self._manip_obj_root_state[:, 3:7]),
                    "manip_obj_rot6d": manip_obj_rot6d,
                    "manip_obj_vel": self._manip_obj_root_state[:, 7:10],
                    "manip_obj_ang_vel": self._manip_obj_root_state[:, 10:],
                }
            )

        # Compute manip object pose relative to wrist frame
        wrist_pose = torch.cat(
            [self.states["base_state"][:, :3], self.states["base_state"][:, 3:7]],
            dim=-1,
        )
        obj_pose = torch.cat(
            [self.states["manip_obj_pos"], self.states["manip_obj_quat"]],
            dim=-1,
        )
        rel_pose = calculate_relative_pose(obj_pose, wrist_pose)
        self.states["manip_obj_pos_rel_wrist"] = rel_pose[:, :3]
        self.states["manip_obj_quat_rel_wrist"] = ensure_quat_w_positive(rel_pose[:, 3:7])

        # Compute joints_state relative to wrist frame
        # joints_state: [B, num_joints, 10] -> pos[:3], quat[3:7], vel[7:10]
        wrist_pos = self.states["base_state"][:, :3]  # [B, 3]
        wrist_quat = self.states["base_state"][:, 3:7]  # [B, 4] xyzw
        wrist_quat_conj = quat_conjugate(wrist_quat)  # [B, 4]

        joints_pos_world = self.states["joints_state"][:, :, :3]  # [B, num_joints, 3]
        joints_quat_world = self.states["joints_state"][:, :, 3:7]  # [B, num_joints, 4]
        joints_vel_world = self.states["joints_state"][:, :, 7:10]  # [B, num_joints, 3]

        B, num_joints, _ = joints_pos_world.shape

        # Transform positions: p_rel = q_conj @ (p_world - p_wrist)
        pos_diff = joints_pos_world - wrist_pos.unsqueeze(1)  # [B, num_joints, 3]
        joints_pos_rel = torch_jit_utils.quat_apply(
            wrist_quat_conj.unsqueeze(1).expand(-1, num_joints, -1).reshape(B * num_joints, 4),
            pos_diff.reshape(B * num_joints, 3)
        ).reshape(B, num_joints, 3)

        # Transform rotations: q_rel = q_conj @ q_world
        joints_quat_rel = quat_mul(
            wrist_quat_conj.unsqueeze(1).expand(-1, num_joints, -1).reshape(B * num_joints, 4),
            joints_quat_world.reshape(B * num_joints, 4)
        ).reshape(B, num_joints, 4)
        joints_quat_rel = ensure_quat_w_positive(joints_quat_rel.reshape(B * num_joints, 4)).reshape(B, num_joints, 4)

        # Transform velocities: v_rel = q_conj @ v_world
        joints_vel_rel = torch_jit_utils.quat_apply(
            wrist_quat_conj.unsqueeze(1).expand(-1, num_joints, -1).reshape(B * num_joints, 4),
            joints_vel_world.reshape(B * num_joints, 3)
        ).reshape(B, num_joints, 3)

        self.states["joints_state_rel_wrist"] = torch.cat(
            [joints_pos_rel, joints_quat_rel, joints_vel_rel], dim=-1
        )  # [B, num_joints, 10]
        self.states["joints_vel_rel_wrist"] = joints_vel_rel # TODO: check if this velocity is correct

        # Gravity direction in wrist frame
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=wrist_pos.device, dtype=wrist_pos.dtype)
        self.states["gravity_dir_rel_wrist"] = torch_jit_utils.quat_apply(
            wrist_quat_conj,
            gravity_world.unsqueeze(0).expand(B, -1)
        )  # [B, 3]


    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        target_state = {}
        opt_target_state = {}
        max_length = torch.clip(self.demo_data["seq_len"][self.envidx_to_demoidx], 0, self.max_episode_length).float()
        if self.enable_latency_tracking:
            cur_idx = self.global_cur_idx
        else:
            cur_idx = self.progress_buf
        cur_wrist_pos = self.demo_data["wrist_pos"][self.envidx_to_demoidx, cur_idx]
        opt_cur_wrist_pos = self.demo_data["opt_wrist_pos"][self.envidx_to_demoidx, cur_idx]
        # target_state["wrist_pos"] = cur_wrist_pos
        opt_target_state["wrist_pos"] = opt_cur_wrist_pos
        cur_wrist_rot = self.demo_data["wrist_rot"][self.envidx_to_demoidx, cur_idx]
        opt_cur_wrist_rot = self.demo_data["opt_wrist_rot"][self.envidx_to_demoidx, cur_idx]
        # target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]
        opt_target_state["wrist_quat"] = aa_to_quat(opt_cur_wrist_rot)[:, [1, 2, 3, 0]]

        # target_state["wrist_vel"] = self.demo_data["wrist_velocity"][self.envidx_to_demoidx, cur_idx]
        # target_state["wrist_ang_vel"] = self.demo_data["wrist_angular_velocity"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["wrist_vel"] = self.demo_data["opt_wrist_velocity"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["wrist_ang_vel"] = self.demo_data["opt_wrist_angular_velocity"][self.envidx_to_demoidx, cur_idx]
        
        # target_state["tips_distance"] = self.demo_data["tips_distance"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["tips_distance"] = self.demo_data["opt_tips_distance"][self.envidx_to_demoidx, cur_idx]
        
        # target_state["dof_pos"] = self.demo_data["opt_dof_pos"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["dof_pos"] = self.demo_data["opt_dof_pos"][self.envidx_to_demoidx, cur_idx]

        cur_joints_pos = self.demo_data["mano_joints"][self.envidx_to_demoidx, cur_idx]
        opt_cur_joints_pos = self.demo_data["opt_mano_joints"][self.envidx_to_demoidx, cur_idx]
        # target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        opt_target_state["joints_pos"] = opt_cur_joints_pos.reshape(self.num_envs, -1, 3)
        # target_state["joints_vel"] = self.demo_data["mano_joints_velocity"][
        #     self.envidx_to_demoidx, cur_idx
        # ].reshape(self.num_envs, -1, 3)
        opt_target_state["joints_vel"] = self.demo_data["opt_mano_joints_velocity"][
            self.envidx_to_demoidx, cur_idx
        ].reshape(self.num_envs, -1, 3)

        cur_obj_transf = self.demo_data["obj_trajectory"][self.envidx_to_demoidx, cur_idx]
        # target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        # target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        opt_target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        opt_target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]

        # Compute opt_target states relative to opt wrist frame
        opt_wrist_quat_conj = quat_conjugate(opt_target_state["wrist_quat"])  # [B, 4]

        # opt joints_pos in wrist frame: [B, num_joints, 3]
        opt_joints_pos = opt_target_state["joints_pos"]  # [B, num_joints, 3]
        B, num_joints, _ = opt_joints_pos.shape
        opt_joints_pos_diff = opt_joints_pos - opt_target_state["wrist_pos"].unsqueeze(1)  # [B, num_joints, 3]
        opt_target_state["joints_pos_rel_wrist"] = torch_jit_utils.quat_apply(
            opt_wrist_quat_conj.unsqueeze(1).expand(-1, num_joints, -1).reshape(B * num_joints, 4),
            opt_joints_pos_diff.reshape(B * num_joints, 3)
        ).reshape(B, num_joints, 3)

        # opt manip_obj pose in wrist frame
        opt_obj_pose = torch.cat(
            [opt_target_state["manip_obj_pos"], opt_target_state["manip_obj_quat"]],
            dim=-1,
        )  # [B, 7]
        opt_wrist_pose = torch.cat(
            [opt_target_state["wrist_pos"], opt_target_state["wrist_quat"]],
            dim=-1,
        )  # [B, 7]
        opt_rel_pose = calculate_relative_pose(opt_obj_pose, opt_wrist_pose)
        opt_target_state["manip_obj_pos_rel_wrist"] = opt_rel_pose[:, :3]
        opt_target_state["manip_obj_quat_rel_wrist"] = opt_rel_pose[:, 3:7]

        # target_state["manip_obj_vel"] = self.demo_data["obj_velocity"][self.envidx_to_demoidx, cur_idx]
        # target_state["manip_obj_ang_vel"] = self.demo_data["obj_angular_velocity"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["manip_obj_vel"] = self.demo_data["obj_velocity"][self.envidx_to_demoidx, cur_idx]
        opt_target_state["manip_obj_ang_vel"] = self.demo_data["obj_angular_velocity"][self.envidx_to_demoidx, cur_idx]
        
        # target_state["tip_force"] = torch.stack(
        #     [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
        #     axis=1,
        # )
        # opt_target_state["tip_force"] = torch.stack(
        #     [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
        #     axis=1,
        # )
        # self.tips_contact_history = torch.concat(
        #     [
        #         self.tips_contact_history[:, 1:],
        #         (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
        #     ],
        #     dim=1,
        # )
        # target_state["tip_contact_state"] = self.tips_contact_history
        # opt_target_state["tip_contact_state"] = self.tips_contact_history

        power = torch.abs(torch.multiply(self.dof_force, self.states["dq"])).sum(dim=-1)
        # target_state["power"] = power
        opt_target_state["power"] = power

        # wrist_power = torch.abs(
        #     torch.sum(
        #         self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
        #         * self.states["base_state"][:, 7:10],
        #         dim=-1,
        #     )
        # )  # ? linear force * linear velocity
        # wrist_power += torch.abs(
        #     torch.sum(
        #         self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
        #         * self.states["base_state"][:, 10:],
        #         dim=-1,
        #     )
        # )  # ? torque * angular velocity
        # target_state["wrist_power"] = wrist_power
        # opt_target_state["wrist_power"] = wrist_power

        if self.training_or_collection:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                self.scale_factor = 1.0
            elif self.tighten_method == "const":
                self.scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                self.scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                self.scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                self.scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            self.scale_factor = 1.0

        # assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)


        if self.enable_latency_tracking:
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.success_buf[:],
                self.failure_buf[:],
                self.reward_dict,
                self.error_buf[:],
                self.failure_progress_buf[:],
                self.global_cur_idx[:],
                self.stable_frames_buf[:],
                self.consecutive_reach_goal_buf[:],
                self.consecutive_reach_frames_buf[:],
                self.skip_steps_buf[:],
                self.reach_final_goal,
                self.num_frames_to_stay_buf,
                self.traj_direction,
                self.traj_steps_counter,
                reach_errors,
            ) = compute_imitation_reward_latency_tracking(
                self.reset_buf,
                self.progress_buf,
                self.running_progress_buf,
                self.failure_progress_buf,
                self.global_cur_idx,
                self.stable_frames_buf,
                self.consecutive_reach_goal_buf,
                self.consecutive_reach_frames_buf,
                self.skip_steps_buf,
                self.skip_steps_min,
                self.skip_steps_max,
                self.reverse_target_prob,
                self.fixed_tolerance_steps,
                self.failure_tolerance_scale,
                self.num_frames_to_stay_buf,
                self.num_frames_to_stay_lower_bound,
                self.num_frames_to_stay_upper_bound,
                self.actions,
                self.last_actions,
                self.states,
                opt_target_state,  ## original: target_state
                max_length,
                self.scale_factor,
                self.dexhand.weight_idx,
                self.dexhand.n_finger_tips,
                self.success_obj_pos_thres,
                self.success_thumb_tip_pos_thres,
                self.success_index_tip_pos_thres,
                self.success_middle_tip_pos_thres,
                self.success_pinky_tip_pos_thres,
                self.success_ring_tip_pos_thres,
                self.success_obj_rot_thres,
                self.failure_obj_pos_thres,
                self.failure_thumb_tip_pos_thres,
                self.failure_index_tip_pos_thres,
                self.failure_middle_tip_pos_thres,
                self.failure_pinky_tip_pos_thres,
                self.failure_ring_tip_pos_thres,
                self.failure_obj_rot_thres,
                self.invalid_obj_pos_thres,
                self.time_penalty,
                self.traj_direction,
                self.traj_steps_counter,
                self.traj_steps_limit,
                self.is_target_cross,
            )

            # Update last_reach_error only when reach_final_goal is True
            # reach_errors is a tuple: (obj_pos, obj_rot, thumb, index, middle, pinky)
            error_tensor = torch.stack(reach_errors, dim=-1)
            self.last_reach_error = torch.where(
                self.reach_final_goal.unsqueeze(-1),
                error_tensor,
                self.last_reach_error
            )

            if self.reset_on_reach_goal:
                reach_success = self.reach_final_goal & ~self.failure_buf.bool()
                self.success_buf[:] = torch.where(
                    reach_success,
                    torch.ones_like(self.success_buf),
                    self.success_buf,
                )
                self.reset_buf[:] = torch.where(
                    reach_success,
                    torch.ones_like(self.reset_buf),
                    self.reset_buf,
                )

            if (
                self.training_or_collection
                and self.enable_cross_trajectory_reset
                and not self.reset_on_reach_goal
                and len(self.dataIndices) > 1
                and not self.enable_adapt_sampling
            ):
                n_demo = len(self.dataIndices)
                rg = self.reach_final_goal
                cross = rg & (torch.rand(self.num_envs, device=self.device) < self.cross_trajectory_goal_prob)
                new_demo = torch.randint(0, n_demo, (self.num_envs,), device=self.device, dtype=torch.long)
                # seq_lens = self.demo_data["seq_len"][new_demo]
                # nf = (torch.rand(self.num_envs, device=self.device) * seq_lens.float()).long()
                # nf = torch.minimum(torch.maximum(nf, torch.zeros_like(nf)), seq_lens - 1)
                self.envidx_to_demoidx = torch.where(cross, new_demo, self.envidx_to_demoidx)
                self.is_target_cross = torch.where(
                    rg,
                    cross,
                    self.is_target_cross,
                )
        else:
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.success_buf[:],
                self.failure_buf[:],
                self.reward_dict,
                self.error_buf[:],
                self.failure_progress_buf[:],
            ) = compute_imitation_reward(
                self.reset_buf,
                self.progress_buf,
                self.running_progress_buf,
                self.failure_progress_buf,
                self.actions,
                self.last_actions,
                self.states,
                opt_target_state,  ## original: target_state
                max_length,
                self.scale_factor,
                self.dexhand.weight_idx,
                self.dexhand.n_finger_tips
            )
        self.total_rew_buf += self.rew_buf

        ## update metrics
        if self.enable_adapt_sampling:
            # For each demo_idx, find the max consecutive_reach_goal_buf among all envs using that demo
            # Use scatter_reduce to aggregate max values per demo_idx (vectorized)
            demo_max_values = torch.zeros_like(self.consecutive_reach_goal_max_buf)
            demo_max_values.scatter_reduce_(
                dim=0,
                index=self.envidx_to_demoidx,
                src=self.consecutive_reach_goal_buf,
                reduce="amax",
                include_self=False
            )
            # Update consecutive_reach_goal_max_buf with the maximum of current and new values
            self.consecutive_reach_goal_max_buf = torch.max(
                self.consecutive_reach_goal_max_buf,
                demo_max_values
            )
            frames_max_values = torch.zeros_like(self.consecutive_reach_frames_max_buf)
            frames_max_values.scatter_reduce_(
                dim=0,
                index=self.envidx_to_demoidx,
                src=self.consecutive_reach_frames_buf,
                reduce="amax",
                include_self=False
            )
            self.consecutive_reach_frames_max_buf = torch.max(
                self.consecutive_reach_frames_max_buf,
                frames_max_values
            )

    def compute_extra_states(self):
        """Populate states_buf for asymmetric critic.

        Structure:
            states_buf = [proprioception] + [target] + [privileged] + [extra_keys]
        in the same order as the flattened obs that the actor sees.
        """
        if self.num_states <= 0:
            return
        state_values = []
        # 1. Proprioception (actor's body perception)
        state_values.append(self.obs_dict["proprioception"])
        # 2. Target (fingertip/object targets)
        state_values.append(self.obs_dict["target"])
        # 3. Privileged (object pose — actor doesn't see this)
        state_values.append(self.obs_dict["privileged"])
        # 4. Extra state keys (critic-only info)
        if len(self._state_keys) > 0:
            extra_keys = [k for k in self._state_keys if k not in self._privileged_obs_keys]
            for ob in extra_keys:
                state_values.append(self.states[ob].reshape(self.num_envs, -1))
        self.states_buf[:] = torch.cat(state_values, dim=-1)

    def compute_observations(self):
        self._refresh()
        
        # compute observation noise scale factor
        observation_noise_scale_factor = 0.0
        if self.obs_noise_params is not None and self.training:
            noise_schedule = self.obs_noise_schedule
            if noise_schedule:
                last_step = self.gym.get_frame_count(self.sim)
                schedule_type = noise_schedule.get("schedule", "linear")
                start_step = noise_schedule.get("start_step", 0)
                end_step = noise_schedule.get("end_step", 64000)
                operation = noise_schedule.get("operation", "scaling")
                
                if schedule_type == "linear":
                    if last_step < start_step:
                        observation_noise_scale_factor = 0.0
                    elif last_step >= end_step:
                        observation_noise_scale_factor = 1.0
                    else:
                        observation_noise_scale_factor = (last_step - start_step) / (end_step - start_step)
                else:
                    raise NotImplementedError(f"Unsupported noise schedule type: {schedule_type}")
        
        # store observation_noise_scale_factor for logging
        self.observation_noise_scale_factor = observation_noise_scale_factor
        
        # obs_keys: q, cos_q, sin_q, base_state
        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(self.states[ob][:, :3]), self.states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                ## defualt is q noise
                obs_val = self.states[ob]
                # add noise to proprioception fields
                if ob in self.obs_noise_proprioception:
                    obs_val = apply_observation_noise(
                        obs_val,
                        self.obs_noise_proprioception[ob],
                        observation_noise_scale_factor,
                    )
                obs_values.append(obs_val)
        self.obs_dict["proprioception"][:] = torch.cat(obs_values, dim=-1)
        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_val = self.states[ob] - self.states["base_state"][:, :3]
                    # add noise to manip_obj_pos
                    if ob in self.obs_noise_privileged:
                        pri_obs_val = apply_observation_noise(
                            pri_obs_val,
                            self.obs_noise_privileged[ob],
                            observation_noise_scale_factor,
                        )
                    pri_obs_values.append(pri_obs_val)
                    # pri_obs_values.append(aa_to_quat(self.states[ob][:, :3])[:, [1, 2, 3, 0]])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(self.states["manip_obj_quat"][:, [3, 0, 1, 2]])
                        @ self.manip_obj_com.unsqueeze(-1)
                    ).squeeze(-1) + self.states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - self.states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((self.manip_obj_mass * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                elif ob == "critic_obs":
                    # pri_obs_values.append(self.consecutive_reach_goal_buf.reshape(self.num_envs, -1))
                    # pri_obs_values.append(self.consecutive_reach_frames_buf.reshape(self.num_envs, -1)/100.0)
                    pri_obs_values.append(torch.zeros_like(self.consecutive_reach_goal_buf).reshape(self.num_envs, -1))
                    pri_obs_values.append(torch.zeros_like(self.consecutive_reach_frames_buf).reshape(self.num_envs, -1)/100.0)
                else:
                    pri_obs_val = self.states[ob]
                    # add noise to other privileged fields (e.g., dq, manip_obj_quat)
                    if ob in self.obs_noise_privileged:
                        pri_obs_val = apply_observation_noise(
                            pri_obs_val,
                            self.obs_noise_privileged[ob],
                            observation_noise_scale_factor,
                        )
                    pri_obs_values.append(pri_obs_val)
            self.obs_dict["privileged"][:] = torch.cat(pri_obs_values, dim=-1)

        if self.obs_enable_history:
            self.obs_dict["history"][:] = self.states["qa_history"]

        next_target_state = {}

        # cur_idx = self.progress_buf + 1
        ## ! debug, show get the progress_buf + self.control_freq_inv
        if self.enable_latency_tracking:
            cur_idx = self.global_cur_idx
        else:
            cur_idx = self.progress_buf + self.control_freq_inv

        if self.training and not self.enable_latency_tracking:
            random_next_idx = torch.randint(
                low=self.random_next_frames[0], high=self.random_next_frames[1] + 1,
                size=self.progress_buf.shape,
                device=self.progress_buf.device
            )
            cur_idx = self.progress_buf + random_next_idx

        demo_seq_len = self.demo_data["seq_len"][self.envidx_to_demoidx]
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(demo_seq_len), demo_seq_len - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        exact_cur_idx = cur_idx[:, 0].clone()
        # Persist the target identity used to build this policy observation.
        # Reward computation can advance global_cur_idx later in the same step.
        self.current_target_frame_idx = exact_cur_idx.detach().clone()
        self.current_target_data_index_id = self.envidx_to_demoidx.detach().clone()
        base_dof_pos = self.demo_data["base_dof_pos"][self.envidx_to_demoidx, exact_cur_idx]
        self.base_dof_pos = torch_jit_utils.unscale(
            base_dof_pos,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )
        next_target_state["base_dof_pos"] = self.base_dof_pos
        _, nT = self.demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx, demo_idx):
            # data: [nE, nT, ...], idx: [B, K], demo_idx: [B]
            # First index data with demo_idx to get [B, nT, ...]
            indexed_data = data[demo_idx]  # [B, nT, ...]
            assert indexed_data.shape[0] == idx.shape[0] and indexed_data.shape[1] == nT
            remaining_shape = indexed_data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(indexed_data, 1, expanded_idx)

        target_wrist_pos = indicing(self.demo_data["wrist_pos"], cur_idx, self.envidx_to_demoidx)  # [B, K, 3]
        cur_wrist_pos = self.states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(self.num_envs, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], cur_idx, self.envidx_to_demoidx)
        cur_wrist_vel = self.states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(self.num_envs, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(self.num_envs, -1)

        target_wrist_rot = indicing(self.demo_data["wrist_rot"], cur_idx, self.envidx_to_demoidx)
        cur_wrist_rot = self.states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = ensure_quat_w_positive(aa_to_quat(target_wrist_rot.reshape(self.num_envs * nF, -1))[:, [1, 2, 3, 0]])
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(self.num_envs * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(self.num_envs, -1)
        next_target_state["delta_wrist_quat"] = ensure_quat_w_positive(next_target_state["delta_wrist_quat"])
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(self.num_envs, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx, self.envidx_to_demoidx)
        cur_wrist_ang_vel = self.states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(self.num_envs, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(self.num_envs, -1)

        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx, self.envidx_to_demoidx).reshape(self.num_envs, nF, -1, 3)
        next_target_state["joints_pos"] = target_joints_pos.reshape(self.num_envs, -1)
        cur_joint_pos = self.states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_fingertips_pos = indicing(self.demo_data["fingertips_pos"], cur_idx, self.envidx_to_demoidx).reshape(self.num_envs, nF, -1, 3)
        next_target_state["fingertips_pos"] = target_fingertips_pos.reshape(self.num_envs, -1)
        cur_fingertips_pos = self.states["tips_pos"][:, :, :3]
        next_target_state["delta_fingertips_pos"] = (target_fingertips_pos - cur_fingertips_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx, self.envidx_to_demoidx).reshape(self.num_envs, nF, -1, 3)
        cur_joint_vel = self.states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(self.demo_data["obj_trajectory"], cur_idx, self.envidx_to_demoidx)
        self.current_target_obj_pose_world = target_obj_transf.detach().clone()
        target_obj_transf = target_obj_transf.reshape(self.num_envs * nF, 4, 4)
        next_target_state["manip_obj_pos"] = target_obj_transf[:, :3, 3].reshape(self.num_envs, nF, -1).reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(self.num_envs, nF, -1) - self.states["manip_obj_pos"][:, None]
        ).reshape(self.num_envs, -1)

        target_obj_vel = indicing(self.demo_data["obj_velocity"], cur_idx, self.envidx_to_demoidx)
        cur_obj_vel = self.states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(self.num_envs, -1)

        next_target_state["manip_obj_quat"] = ensure_quat_w_positive(rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]])
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            self.states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(self.num_envs * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_quat"] = ensure_quat_w_positive(
            next_target_state["delta_manip_obj_quat"]
        )
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(self.num_envs, -1)

        target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], cur_idx, self.envidx_to_demoidx)
        cur_obj_ang_vel = self.states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(self.num_envs, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            self.states["manip_obj_pos"][:, None] - self.states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(self.demo_data["tips_distance"], cur_idx, self.envidx_to_demoidx).reshape(self.num_envs, -1)

        next_target_state["bps"] = self.obj_bps[self.envidx_to_demoidx]

        ## !? >>> for quick test, embed the task
        next_target_state["task_embedding"] = indicing(self.demo_data["task_embedding"], cur_idx, self.envidx_to_demoidx).reshape(self.num_envs, -1)
        ## !? <<< for quick test, embed the task

        ## !? >>> add curr_targets
        next_target_state["curr_targets"] = self.states["q_curr_targets"]

        # ! Compute manip_obj and fingertips pose relative to wrist frame
        # Current wrist pose (world frame)
        cur_wrist_pos = self.states["base_state"][:, :3]  # [B, 3]
        cur_wrist_quat = self.states["base_state"][:, 3:7]  # [B, 4] xyzw
        cur_wrist_quat_conj = quat_conjugate(cur_wrist_quat)  # [B, 4]

        # Current manip_obj pose in wrist frame
        cur_manip_obj_pos_rel_wrist = self.states["manip_obj_pos_rel_wrist"]  # [B, 3]
        cur_manip_obj_quat_rel_wrist = self.states["manip_obj_quat_rel_wrist"]  # [B, 4]

        # Current fingertips pos in wrist frame
        cur_fingertips_pos = self.states["tips_pos"][:, :, :3]  # [B, num_tips, 3]
        num_tips = cur_fingertips_pos.shape[1]
        cur_fingertips_pos_diff = cur_fingertips_pos - cur_wrist_pos.unsqueeze(1)  # [B, num_tips, 3]
        cur_fingertips_pos_rel_wrist = torch_jit_utils.quat_apply(
            cur_wrist_quat_conj.unsqueeze(1).expand(-1, num_tips, -1).reshape(self.num_envs * num_tips, 4),
            cur_fingertips_pos_diff.reshape(self.num_envs * num_tips, 3)
        ).reshape(self.num_envs, num_tips, 3)

        # Target wrist quat (world frame) from demo
        target_wrist_quat = next_target_state["wrist_quat"][:, -4:].reshape(self.num_envs, 4)  # last frame's wrist_quat

        # Target manip_obj pose relative to target wrist frame
        target_manip_obj_pos = next_target_state["manip_obj_pos"].reshape(self.num_envs, nF, 3)[:, -1, :]  # [B, 3] last frame
        target_manip_obj_quat = next_target_state["manip_obj_quat"].reshape(self.num_envs, nF, 4)[:, -1, :]  # [B, 4] last frame
        target_manip_obj_pose = torch.cat([target_manip_obj_pos, target_manip_obj_quat], dim=-1)  # [B, 7]
        target_wrist_pose = torch.cat(
            [target_wrist_pos[:, -1, :], target_wrist_quat],
            dim=-1,
        )  # [B, 7]
        target_rel_pose = calculate_relative_pose(target_manip_obj_pose, target_wrist_pose)
        target_manip_obj_pos_rel_wrist = target_rel_pose[:, :3]  # [B, 3]
        target_manip_obj_quat_rel_wrist = target_rel_pose[:, 3:7]  # [B, 4]

        # All frames: manip_obj pose relative to each frame's wrist
        target_manip_obj_pos_all = next_target_state["manip_obj_pos"].reshape(self.num_envs, nF, 3)  # [B, nF, 3]
        target_manip_obj_quat_all = next_target_state["manip_obj_quat"].reshape(self.num_envs, nF, 4)  # [B, nF, 4]
        target_wrist_quat_all = target_wrist_quat.unsqueeze(1).expand(-1, nF, -1).reshape(self.num_envs * nF, 4)  # use same wrist quat for simplicity
        target_wrist_pos_all = target_wrist_pos  # [B, nF, 3]

        # Compute relative pose for all frames
        target_manip_obj_pose_all = torch.cat(
            [target_manip_obj_pos_all.reshape(self.num_envs * nF, 3),
             target_manip_obj_quat_all.reshape(self.num_envs * nF, 4)],
            dim=-1,
        )  # [B*nF, 7]
        target_wrist_pose_all = torch.cat(
            [target_wrist_pos_all.reshape(self.num_envs * nF, 3), target_wrist_quat_all],
            dim=-1,
        )  # [B*nF, 7]
        target_rel_pose_all = calculate_relative_pose(target_manip_obj_pose_all, target_wrist_pose_all)
        target_manip_obj_pos_rel_wrist_all = target_rel_pose_all[:, :3].reshape(self.num_envs, nF, 3)  # [B, nF, 3]
        target_manip_obj_quat_rel_wrist_all = ensure_quat_w_positive(target_rel_pose_all[:, 3:7]).reshape(self.num_envs, nF, 4)  # [B, nF, 4]

        next_target_state["manip_obj_pos_rel_wrist"] = target_manip_obj_pos_rel_wrist_all.reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_pos_rel_wrist"] = (
            target_manip_obj_pos_rel_wrist_all - cur_manip_obj_pos_rel_wrist[:, None]
        ).reshape(self.num_envs, -1)

        next_target_state["manip_obj_quat_rel_wrist"] = target_manip_obj_quat_rel_wrist_all.reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_quat_rel_wrist"] = quat_mul(
            cur_manip_obj_quat_rel_wrist[:, None].repeat(1, nF, 1).reshape(self.num_envs * nF, -1),
            quat_conjugate(target_manip_obj_quat_rel_wrist_all.reshape(self.num_envs * nF, 4)),
        ).reshape(self.num_envs, -1)
        next_target_state["delta_manip_obj_quat_rel_wrist"] = ensure_quat_w_positive(next_target_state["delta_manip_obj_quat_rel_wrist"])

        # Target fingertips pos relative to target wrist frame
        target_fingertips_pos = next_target_state["fingertips_pos"].reshape(self.num_envs, nF, num_tips, 3)  # [B, nF, num_tips, 3]
        target_fingertips_pos_rel_wrist_all = []
        for t in range(nF):
            ft_pos_t = target_fingertips_pos[:, t, :, :]  # [B, num_tips, 3]
            wrist_pos_t = target_wrist_pos[:, t, :]  # [B, 3]
            wrist_quat_t = target_wrist_quat  # [B, 4] use same wrist quat
            ft_diff_t = ft_pos_t - wrist_pos_t.unsqueeze(1)  # [B, num_tips, 3]
            ft_rel_t = torch_jit_utils.quat_apply(
                quat_conjugate(wrist_quat_t).unsqueeze(1).expand(-1, num_tips, -1).reshape(self.num_envs * num_tips, 4),
                ft_diff_t.reshape(self.num_envs * num_tips, 3)
            ).reshape(self.num_envs, num_tips, 3)
            target_fingertips_pos_rel_wrist_all.append(ft_rel_t)
        target_fingertips_pos_rel_wrist_all = torch.stack(target_fingertips_pos_rel_wrist_all, dim=1)  # [B, nF, num_tips, 3]

        next_target_state["fingertips_pos_rel_wrist"] = target_fingertips_pos_rel_wrist_all.reshape(self.num_envs, -1)
        next_target_state["delta_fingertips_pos_rel_wrist"] = (
            target_fingertips_pos_rel_wrist_all - cur_fingertips_pos_rel_wrist[:, None, :, :]
        ).reshape(self.num_envs, -1)

        # ! embed the tip <-> object distance and normal
        cur_obj_pos = self.states["manip_obj_pos"]  # [NUM_ENVS, 3]
        cur_obj_quat = self.states["manip_obj_quat"]  # [NUM_ENVS, 4], xyzw
        cur_obj_rotmat = quat_to_rotmat(cur_obj_quat[..., [3, 0, 1, 2]])  # [NUM_ENVS, 3, 3]
        cur_obj_poses = torch.eye(4, device=cur_obj_pos.device, dtype=cur_obj_pos.dtype).unsqueeze(0).repeat(self.num_envs, 1, 1)  # [NUM_ENVS, 4, 4]
        cur_obj_poses[:, :3, :3] = cur_obj_rotmat
        cur_obj_poses[:, :3, 3] = cur_obj_pos
        tip2object_distance, tip2object_normal = self.object_model.distance_gradient(
            torch.tensor(self.states["tips_pos"].clone().detach(), dtype=torch.float32, device=self.device),
            signed=False,
            object_poses=cur_obj_poses
        )
        next_target_state["tip2object_distance"] = tip2object_distance.reshape(self.num_envs, -1)
        next_target_state["tip2object_normal"] = tip2object_normal.reshape(self.num_envs, -1)

        # apply observation noise to target fields
        for key in next_target_state.keys():
            if key in self.obs_noise_target:
                next_target_state[key] = apply_observation_noise(
                    next_target_state[key],
                    self.obs_noise_target[key],
                    observation_noise_scale_factor,
                )

        # Retain named target fields so collection stores exactly what the
        # policy observed, before any target index update in reward computation.
        self.current_target_obs = {
            key: next_target_state[key].detach().clone()
            for key in self.obs_dict_target_keys
        }
        self.obs_dict["target"][:] = torch.cat(
            [
                next_target_state[ob]
                for ob in self.obs_dict_target_keys
            ],
            dim=-1,
        )

        # update fields to dump
        # prop fields
        # if not self.training:

        ## Collect data
        ## 1. Observation
        ## 2. norm_q
        ## 3. q
        ## 4. norm_target
        ## 5. curr_target

        ## DEBUG: simulate delay

        ## input to queue
        # self.last_obs_dict_queue.append(copy.deepcopy(self.obs_dict))
        # if len(self.last_obs_dict_queue) > self.last_obs_dict_queue_size:
        #     self.last_obs_dict_queue.pop(0)
        # ## output from queue
        # self.obs_dict = copy.deepcopy(
        #     self.last_obs_dict_queue[
        #         # random.randint(0, len(self.last_obs_dict_queue) - 1)
        #         0
        #     ]
        # )

        # self.current_obs_dict = copy.deepcopy(self.obs_dict)
        # if self.last_obs_dict is not None:
        #     self.obs_dict = copy.deepcopy(self.last_obs_dict) 
        #     self.last_obs_dict = self.current_obs_dict
        # else:
        #     self.last_obs_dict = self.current_obs_dict
        
        if self.collect_data:
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "proprioception":
                    self.dump_fileds[prop_name][:] = self.obs_dict["proprioception"]
                elif prop_name == "privileged":
                    self.dump_fileds[prop_name][:] = self.obs_dict["privileged"]
                elif prop_name == "target":
                    self.dump_fileds[prop_name][:] = self.obs_dict["target"]
                elif prop_name == "history":
                    self.dump_fileds[prop_name][:] = self.obs_dict["history"].reshape(self.num_envs, -1)
                elif prop_name == "q":
                    self.dump_fileds[prop_name][:] = self.states["q"][:]
                elif prop_name == "norm_q":
                    self.dump_fileds[prop_name][:] = self.states["norm_q"][:]
                elif prop_name == "curr_targets":
                    self.dump_fileds[prop_name][:] = self.states["q_curr_targets"][:]
                elif prop_name == "norm_curr_targets":
                    self.dump_fileds[prop_name][:] = self.states["norm_curr_targets"][:]
                else:
                    raise ValueError(f"Unknown prop_name: {prop_name}")

        if False:
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" or prop_name == "state_lh":
                    self.dump_fileds[prop_name][:] = self.states["base_state"]
                elif prop_name == "state_manip_obj_rh" or prop_name == "state_manip_obj_lh":
                    self.dump_fileds[prop_name][:] = self._manip_obj_root_state
                elif prop_name == "joint_state_rh" or prop_name == "joint_state_lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, self.dexhand_handles[k], :] for k in self.dexhand.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "tip_force_rh" or prop_name == "tip_force_lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "q_rh" or prop_name == "q_lh":
                    self.dump_fileds[prop_name][:] = self.states["q"][:]
                elif prop_name == "dq_rh" or prop_name == "dq_lh":
                    self.dump_fileds[prop_name][:] = self.states["dq"][:]
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:  # [q, dq]
                    self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        return self.obs_dict

    def _reset_default(self, env_ids):
        demo_idx = self.envidx_to_demoidx[env_ids]
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.90 * torch.rand_like(self.demo_data["seq_len"][demo_idx].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data["seq_len"][demo_idx] * 0.90).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data["seq_len"][demo_idx]
                    * 0.9
                    * torch.rand_like(self.demo_data["seq_len"][demo_idx].float())
                ).long()
        else:
            # if self.rollout_begin is not None:
            #     seq_idx = self.rollout_begin * torch.ones_like(self.demo_data["seq_len"][demo_idx].long())
            # else:
            #     seq_idx = torch.zeros_like(self.demo_data["seq_len"][demo_idx].long())

            # << ! try random 0.10 state for rollout
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.10 * torch.rand_like(self.demo_data["seq_len"][demo_idx].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data["seq_len"][demo_idx] * 0.10).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data["seq_len"][demo_idx]
                    * 0.10
                    * torch.rand_like(self.demo_data["seq_len"][demo_idx].float())
                ).long()

        dof_pos = self.demo_data["opt_dof_pos"][demo_idx, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            self.dexhand_dof_lower_limits.unsqueeze(0),
            self.dexhand_dof_upper_limits.unsqueeze(0),
        )
        dof_vel = self.demo_data["opt_dof_velocity"][demo_idx, seq_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * self._dexhand_dof_speed_limits.unsqueeze(0),
            self._dexhand_dof_speed_limits.unsqueeze(0),
        )

        opt_wrist_pos = self.demo_data["opt_wrist_pos"][demo_idx, seq_idx]
        opt_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][demo_idx, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = self.demo_data["opt_wrist_velocity"][demo_idx, seq_idx]
        opt_wrist_ang_vel = self.demo_data["opt_wrist_angular_velocity"][demo_idx, seq_idx]

        # random wrist
        if self.random_wrist_orientation:
            origin_wrist_rot = opt_wrist_rot.clone()
            random_wrist_orientation_offset = random_orientation_in_cone(len(env_ids), self.random_wrist_max_tilt_angle, device=self.device, dtype=torch.float32) # should be [x, y, z, w]
            self.wrist_orientation_offset[env_ids, :] = random_wrist_orientation_offset
            opt_wrist_rot = quat_mul(opt_wrist_rot, random_wrist_orientation_offset) # TODO: check if this is correct

        # reset hand wrist pose
        opt_hand_base_state = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)
        self._base_state[env_ids, :] = opt_hand_base_state

        # reset manip obj
        obj_pos_init = self.demo_data["obj_trajectory"][demo_idx, seq_idx, :3, 3]
        obj_rot_init = self.demo_data["obj_trajectory"][demo_idx, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
        # [w, x, y, z] to [x, y, z, w]
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = self.demo_data["obj_velocity"][demo_idx, seq_idx]
        obj_ang_vel = self.demo_data["obj_angular_velocity"][demo_idx, seq_idx]

        if self.enable_reset_pool:
            ## override dof_pos with reset pool
            reset_pool_sample_env_indices_bool = torch.rand(env_ids.shape, device=env_ids.device) < self.reset_pool_sample_prob
            reset_pool_sample_env_indices = reset_pool_sample_env_indices_bool.nonzero(as_tuple=False).flatten()
            reset_pool_demo_idx = demo_idx[reset_pool_sample_env_indices]
            reset_pool_seq_idx = seq_idx[reset_pool_sample_env_indices]
            if len(reset_pool_sample_env_indices) > 0:
                reset_pool_sample_indices = torch.randint(0, self.reset_pool_size, (len(reset_pool_sample_env_indices),))
                dof_pos[reset_pool_sample_env_indices, :] = self._reset_pool["opt_dof_pos"][reset_pool_demo_idx, reset_pool_seq_idx, reset_pool_sample_indices, :]
                obj_pos_init[reset_pool_sample_env_indices, :] = self._reset_pool["obj_trajectory"][reset_pool_demo_idx, reset_pool_seq_idx, reset_pool_sample_indices, :3, 3]
                obj_rot_init[reset_pool_sample_env_indices, :] = rotmat_to_quat(
                    self._reset_pool["obj_trajectory"][reset_pool_demo_idx, reset_pool_seq_idx, reset_pool_sample_indices, :3, :3]
                )[..., [1, 2, 3, 0]]

        if self.random_wrist_orientation:
            obj_pose_in_wrist_frame = calculate_relative_pose(
                torch.cat([obj_pos_init, obj_rot_init], dim=-1), 
                torch.cat([opt_wrist_pos, origin_wrist_rot], dim=-1),
            )
            obj_pos_init = quat_apply(
                opt_wrist_rot,
                obj_pose_in_wrist_frame[:, :3],   
            )
            obj_rot_init = quat_mul(
                opt_wrist_rot,
                obj_pose_in_wrist_frame[:, 3:7],
            )

        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = torch.zeros_like(dof_pos)
        self._pos_control[env_ids, :] = dof_pos

        self._manip_obj_root_state[env_ids, :3] = obj_pos_init
        self._manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        self._manip_obj_root_state[env_ids, 7:10] = torch.zeros_like(obj_vel)
        self._manip_obj_root_state[env_ids, 10:13] = torch.zeros_like(obj_ang_vel)

        dexhand_multi_env_ids_int32 = self._global_dexhand_indices[env_ids].flatten()
        manip_obj_multi_env_ids_int32 = self._global_manip_obj_indices[env_ids].flatten()

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.stable_frames_buf[env_ids] = torch.zeros_like(self.stable_frames_buf[env_ids])
        self.num_frames_to_stay_buf[env_ids] = torch.randint(
            low=self.num_frames_to_stay_lower_bound,
            high=self.num_frames_to_stay_upper_bound + 1,
            size=env_ids.shape,
            device=env_ids.device,
            dtype=torch.int32,
        )
        self.consecutive_reach_goal_buf[env_ids] = torch.zeros_like(self.consecutive_reach_goal_buf[env_ids])
        self.consecutive_reach_frames_buf[env_ids] = torch.zeros_like(self.consecutive_reach_frames_buf[env_ids])
        self.skip_steps_buf[env_ids] = torch.ones_like(self.skip_steps_buf[env_ids]) * self.skip_steps_min
        self.global_cur_idx[env_ids] = seq_idx
        self.progress_buf[env_ids] = seq_idx
        # Reset trajectory direction and counter
        self.traj_direction[env_ids] = torch.ones_like(self.traj_direction[env_ids])
        self.traj_steps_counter[env_ids] = torch.zeros_like(self.traj_steps_counter[env_ids])
        self.last_reach_error[env_ids] = torch.zeros_like(self.last_reach_error[env_ids])
        self.running_progress_buf[env_ids] = torch.zeros_like(self.running_progress_buf[env_ids])
        self.failure_progress_buf[env_ids] = torch.zeros_like(self.failure_progress_buf[env_ids])
        self.is_target_cross[env_ids] = torch.zeros_like(self.is_target_cross[env_ids])
        self.reset_buf[env_ids] = torch.zeros_like(self.reset_buf[env_ids])
        self.success_buf[env_ids] = torch.zeros_like(self.success_buf[env_ids])
        self.failure_buf[env_ids] = torch.zeros_like(self.failure_buf[env_ids])
        self.error_buf[env_ids] = torch.zeros_like(self.error_buf[env_ids])
        self.total_rew_buf[env_ids] = torch.zeros_like(self.total_rew_buf[env_ids])
        self.apply_forces[env_ids] = torch.zeros_like(self.apply_forces[env_ids])
        self.apply_torque[env_ids] = torch.zeros_like(self.apply_torque[env_ids])
        self.curr_targets[env_ids] = dof_pos
        self.prev_targets[env_ids] = dof_pos

        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0

        if self.obs_enable_history:
            self._history_buf[env_ids] = torch.zeros_like(self._history_buf[env_ids])

        self.tips_contact_history[env_ids] = torch.ones_like(self.tips_contact_history[env_ids]).bool()

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)
        # apply_forces is already reset in _reset_default, but reset again to ensure clean state
        self.apply_forces[env_ids] = torch.zeros_like(self.apply_forces[env_ids])
        self._update_object_mass()

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()
            if self.enable_asymmetric_actor_critic:
                self.compute_extra_states()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        # info["reward_dict"] = self.reward_dict
        # info["total_rewards"] = self.total_rew_buf
        # info["total_steps"] = self.progress_buf
        if self.enable_latency_tracking:
            # info["global_cur_idx"] = self.global_cur_idx
            buf = self.consecutive_reach_goal_buf.detach().float()
            info["consecutive_reach_goal"] = buf

            # print(f"Mean consecutive reach goal: {buf.mean().cpu()}")
            # print(f"Max consecutive reach goal: {buf.max().cpu()}")
            # rl_games IsaacAlgoObserver 只把 float/int/0-dim 写入 TensorBoard（direct_info）
            # info["env/consecutive_reach_goal_mean"] = float(buf.mean().cpu())
            # info["env/consecutive_reach_goal_max"] = float(buf.max().cpu())

            # Log traj_steps_counter
            steps_buf = self.traj_steps_counter.detach().float()
            info["traj_steps_counter"] = steps_buf
            # info["env/traj_steps_counter_mean"] = float(steps_buf.mean().cpu())
            # info["env/traj_steps_counter_max"] = float(steps_buf.max().cpu())

            # Log last reach errors (updated only at reach_final_goal frames)
            error_names = ["diff_obj_pos_dist", "diff_obj_rot_angle", "diff_thumb_tip_pos_dist",
                           "diff_index_tip_pos_dist", "diff_middle_tip_pos_dist", "diff_pinky_tip_pos_dist"]
            for i, name in enumerate(error_names):
                valid_errors = self.last_reach_error[:, i]
                nonzero_count = (valid_errors != 0).sum()
                if nonzero_count > 0:
                    info[f"error_{name}"] = valid_errors.float()
        return obs, rew, done, info

    def pre_physics_step(self, actions):
        if self.training or STRICT_TEST_MODE:
            actions = actions + torch.randn_like(actions) * _ACT_NOISE

        ## Apply random mask on hand control signals
        if self.enable_control_signal_mask:
            control_mask_condition = (torch.rand(self.num_envs, device=self.device) < self.control_signal_mask_prob) & (self.control_signal_mask_frames_buf == 0)
            self.control_signal_mask_frames_buf = torch.where(
                control_mask_condition,
                torch.randint(
                    self.control_signal_mask_frames_lower_bound,
                    int((self.control_signal_mask_frames_upper_bound - self.control_signal_mask_frames_lower_bound) / (1 - 0.343) * (1.0 - self.scale_factor**3) + self.control_signal_mask_frames_lower_bound) + 1,
                    # (skip_steps_max - skip_steps_min) / (1.0 - 0.343) * (1.0 - scale_factor**3) + skip_steps_min + 1
                    (self.num_envs,),
                    device=self.device,
                ),
                self.control_signal_mask_frames_buf,
            )
            self.control_signal_mask_dof_indices = torch.where(
                control_mask_condition[:, None],
                torch.randint(
                    self.dexhand.n_dofs,
                    (self.num_envs, self.control_signal_mask_dof_num),
                    device=self.device,
                ),
                self.control_signal_mask_dof_indices,
            ) # (num_envs, control_signal_mask_dof_num)
            mask_indices = self.control_signal_mask_frames_buf > 0
            if mask_indices.any() and self.prev_actions is not None:
                flat_mask = mask_indices.nonzero(as_tuple=True)[0]  # (n_masked,)
                row_idx = flat_mask.unsqueeze(1).expand(
                    -1, self.control_signal_mask_dof_num
                )  # (n_masked, control_signal_mask_dof_num)
                col_idx = self.control_signal_mask_dof_indices[mask_indices]  # (n_masked, control_signal_mask_dof_num)
                actions[row_idx, col_idx] = self.prev_actions[row_idx, col_idx]
            self.control_signal_mask_frames_buf[self.control_signal_mask_frames_buf > 0] -= 1

        self.prev_actions = actions.clone()

        # ? >>> for visualization
        if not self.headless:

            if self.enable_latency_tracking:
                cur_idx = self.global_cur_idx
            else:
                cur_idx = self.progress_buf

            self.gym.clear_lines(self.viewer)

            cur_wrist_pos = self.demo_data["wrist_pos"][self.envidx_to_demoidx, cur_idx]

            cur_mano_joint_pos = self.demo_data["opt_mano_joints"][self.envidx_to_demoidx, cur_idx].reshape(
                self.num_envs, -1, 3
            )
            cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
            if self.random_wrist_orientation:
                # Demo keypoints are in world frame under the original demo wrist rotation.
                # Transform: demo world -> wrist-local -> sim world (which has randomized wrist).
                demo_wrist_pos = self.demo_data["opt_wrist_pos"][self.envidx_to_demoidx, cur_idx]
                demo_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][self.envidx_to_demoidx, cur_idx])[:, [1, 2, 3, 0]]
                sim_wrist_pos = self.states["base_state"][:, :3]
                sim_wrist_rot = self.states["base_state"][:, 3:7]

                B, N, _ = cur_mano_joint_pos.shape
                # Step 1: demo world -> wrist-local
                rel_to_demo_wrist = cur_mano_joint_pos - demo_wrist_pos[:, None, :]
                demo_rot_conj = quat_conjugate(demo_wrist_rot)
                joints_local = quat_apply(
                    demo_rot_conj[:, None, :].expand(-1, N, -1).reshape(B * N, 4),
                    rel_to_demo_wrist.reshape(B * N, 3),
                ).reshape(B, N, 3)
                # Step 2: wrist-local -> sim world
                cur_mano_joint_pos = quat_apply(
                    sim_wrist_rot[:, None, :].expand(-1, N, -1).reshape(B * N, 4),
                    joints_local.reshape(B * N, 3),
                ).reshape(B, N, 3) + sim_wrist_pos[:, None, :]
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = cur_mano_joint_pos[:, k]
            for env_id, env_ptr in enumerate(self.envs):
                for k in self.dexhand.body_names:
                    self.set_force_vis(
                        env_ptr, k, torch.norm(self.net_cf[env_id, self.dexhand_handles[k]], dim=-1) != 0
                    )

                def add_lines(viewer, env_ptr, hand_joints, color):
                    assert hand_joints.shape[0] == self.dexhand.n_bodies and hand_joints.shape[1] == 3
                    hand_joints = hand_joints.cpu().numpy()
                    lines = np.array([[hand_joints[b[0]], hand_joints[b[1]]] for b in self.dexhand.bone_links])
                    for line in lines:
                        self.gym.add_lines(viewer, env_ptr, 1, line, color)

                color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu(), color)
        # ? <<< for visualization

        # dof_pos = actions
        # curr_act_moving_average = self.act_moving_average

        if self.act_style == "hora":
            dof_action = torch.clamp(actions, -1, 1)
            self.curr_targets = self.prev_targets + self.act_scale * dof_action
        elif self.act_style == "flat_hora":
            actions = torch.where((actions < 0.1) & (actions > -0.1), 0.0, actions)
            actions = torch.where(actions > 0.1, actions - 0.1, actions)
            actions = torch.where(actions < -0.1, actions + 0.1, actions)
            self.curr_targets = self.prev_targets.clone() + self.act_scale * actions.clone()
        elif self.act_style == "abs":
            dof_action = torch.clamp(actions, -1, 1)
            self.curr_targets = torch_jit_utils.scale(
                dof_action,  # ! actions must in [-1, 1]
                self.dexhand_dof_lower_limits,
                self.dexhand_dof_upper_limits,
            )
        else:
            raise ValueError(f"Invalid action style: {self.act_style}")

        ## Apply random mask on hand control signals
        # if self.enable_control_signal_mask:
        #     control_mask_condition = (torch.rand(self.num_envs, device=self.device) < self.control_signal_mask_prob) & (self.control_signal_mask_frames_buf == 0)
        #     self.control_signal_mask_frames_buf = torch.where(
        #         control_mask_condition,
        #         torch.randint(
        #             self.control_signal_mask_frames_lower_bound,
        #             int((self.control_signal_mask_frames_upper_bound - self.control_signal_mask_frames_lower_bound) / (1 - 0.343) * (1.0 - self.scale_factor**3) + self.control_signal_mask_frames_lower_bound) + 1,
        #             # (skip_steps_max - skip_steps_min) / (1.0 - 0.343) * (1.0 - scale_factor**3) + skip_steps_min + 1
        #             (self.num_envs,),
        #             device=self.device,
        #         ),
        #         self.control_signal_mask_frames_buf,
        #     )
        #     self.control_signal_mask_dof_indices = torch.where(
        #         control_mask_condition,
        #         torch.randint(
        #             self.dexhand.n_dofs,
        #             (self.num_envs,),
        #             device=self.device,
        #         ),
        #         self.control_signal_mask_dof_indices,
        #     )
        #     mask_indices = self.control_signal_mask_frames_buf > 0
        #     if mask_indices.any():
        #         self.curr_targets[mask_indices, self.control_signal_mask_dof_indices[mask_indices]] = \
        #             self.prev_targets[mask_indices, self.control_signal_mask_dof_indices[mask_indices]]
        #     self.control_signal_mask_frames_buf -= 1
            

        # self.curr_targets = (
        #     curr_act_moving_average * self.curr_targets + (1.0 - curr_act_moving_average) * self.prev_targets
        # )
        # self.curr_targets = (
        #     curr_act_scale * dof_action + self.prev_targets
        # )
        self.curr_targets = torch_jit_utils.tensor_clamp(
            self.curr_targets,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        ##! apply random force to manip_obj
        if self.training_or_collection or STRICT_TEST_MODE:
            decay_factor = self.random_force_decay ** (self.dt / self.random_force_decay_interval)
            self.apply_forces *= decay_factor
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero(as_tuple=False).flatten()
            if len(force_indices) > 0:
                force_shape = self.apply_forces[force_indices, self._manip_obj_rigid_body_handle, :].shape
                force_dir = torch.randn(force_shape, device=self.device)
                force_dir = force_dir / (force_dir.norm(dim=-1, keepdim=True) + 1e-6)
                self.apply_forces[force_indices, self._manip_obj_rigid_body_handle, :] = \
                    force_dir * self.random_force_scale * self.manip_obj_mass[force_indices].unsqueeze(-1)
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.apply_forces),
                None,
                gymapi.LOCAL_SPACE,
            )

        self.prev_targets[:] = self.curr_targets[:]
        self._pos_control[:] = self.prev_targets[:]
        self.actions[:] = self.curr_targets[:]

        if self.disableBackDrive:
            # reset the dof limit
            states_q = self.states['q'].cpu().numpy()
            self.current_dexhand_dof_lower_np = self.global_dexhand_dof_lower_np
            self.current_dexhand_dof_upper_np = self.global_dexhand_dof_upper_np
            set_lower_limit = (self.actions > self.states['q']).cpu().numpy()
            set_upper_limit = ~set_lower_limit
            self.current_dexhand_dof_lower_np[set_lower_limit] = states_q[set_lower_limit]
            self.current_dexhand_dof_upper_np[set_upper_limit] = states_q[set_upper_limit]
            # set dof properties
            for i in range(self.num_envs):
                dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.dexhand_rs[i])
                dof_props['lower'] = self.current_dexhand_dof_lower_np[i]
                dof_props['upper'] = self.current_dexhand_dof_upper_np[i]
                self.gym.set_actor_dof_properties(self.envs[i], self.dexhand_rs[i], dof_props)

        self.gym.set_dof_position_target_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self._pos_control)
        )

    def post_physics_step(self):

        self.compute_observations()
        if self.enable_asymmetric_actor_critic:
            self.compute_extra_states()
        self.compute_reward(self.actions)

        ## update reset pool
        if self.enable_reset_pool:
            if self.reach_final_goal.any():
                reset_pool_sample_env_indices = self.reach_final_goal.nonzero(as_tuple=False).flatten()
                update_reset_pool_indices = torch.rand(len(reset_pool_sample_env_indices), device=self.device) < self.reset_pool_update_prob
                if len(update_reset_pool_indices) > 0:
                    update_reset_pool_env_indices = reset_pool_sample_env_indices[update_reset_pool_indices]
                    demo_idx = self.envidx_to_demoidx[update_reset_pool_env_indices]
                    # make demo idx unique
                    unique_demo_idx_indices = torch.unique(demo_idx, return_inverse=True)[1]
                    unique_demo_idx = demo_idx[unique_demo_idx_indices]
                    unique_update_reset_pool_env_indices = update_reset_pool_env_indices[unique_demo_idx_indices]
                    unique_seq_idx = self.global_cur_idx[unique_update_reset_pool_env_indices] - \
                        self.skip_steps_buf[unique_update_reset_pool_env_indices]
                    self._reset_pool["opt_dof_pos"][unique_demo_idx, unique_seq_idx, :, :] = torch.cat(
                        [
                            self._reset_pool["opt_dof_pos"][unique_demo_idx, unique_seq_idx, 1:, :],
                            self.states["q"][unique_update_reset_pool_env_indices, :].unsqueeze(1),
                        ],
                        dim=1,
                    )
                    current_obj_pos = self._manip_obj_root_state[unique_update_reset_pool_env_indices, :3]
                    current_obj_rot = self._manip_obj_root_state[unique_update_reset_pool_env_indices, 3:7]
                    current_obj_transf = torch.eye(4, device=self.device).unsqueeze(0).repeat(len(unique_update_reset_pool_env_indices), 1, 1)
                    current_obj_transf[:, :3, :3] = quat_to_rotmat(current_obj_rot[..., [3, 0, 1, 2]])
                    current_obj_transf[:, :3, 3] = current_obj_pos

                    self._reset_pool["obj_trajectory"][unique_demo_idx, unique_seq_idx, :, :, :] = torch.cat(
                        [
                            self._reset_pool["obj_trajectory"][unique_demo_idx, unique_seq_idx, 1:, :, :],
                            current_obj_transf.unsqueeze(1),
                        ],
                        dim=1,
                    )

                
        self.last_actions[:] = self.actions[:]

        if not self.headless:

            if self.enable_latency_tracking:
                cur_idx = self.global_cur_idx
            else:
                cur_idx = self.progress_buf
            # * for target object visualization
            cur_obj_pos = self.demo_data["obj_trajectory"][self.envidx_to_demoidx, cur_idx, :3, 3]
            cur_obj_rot = self.demo_data["obj_trajectory"][self.envidx_to_demoidx, cur_idx, :3, :3]
            cur_obj_rot = rotmat_to_quat(cur_obj_rot)
            # [w, x, y, z] to [x, y, z, w]
            cur_obj_rot = cur_obj_rot[:, [1, 2, 3, 0]]
            cur_obj_vel = self.demo_data["obj_velocity"][self.envidx_to_demoidx, cur_idx]
            cur_obj_ang_vel = self.demo_data["obj_angular_velocity"][self.envidx_to_demoidx, cur_idx]
            if self.random_wrist_orientation:
                # Transform target object pose: demo world -> wrist-local -> sim world
                demo_wrist_pos = self.demo_data["opt_wrist_pos"][self.envidx_to_demoidx, cur_idx]
                demo_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][self.envidx_to_demoidx, cur_idx])[:, [1, 2, 3, 0]]
                sim_wrist_pos = self.states["base_state"][:, :3]
                sim_wrist_rot = self.states["base_state"][:, 3:7]
                demo_rot_conj = quat_conjugate(demo_wrist_rot)
                # pos: demo world -> wrist-local -> sim world
                obj_pos_local = quat_apply(demo_rot_conj, cur_obj_pos - demo_wrist_pos)
                cur_obj_pos = quat_apply(sim_wrist_rot, obj_pos_local) + sim_wrist_pos
                # rot: demo world -> wrist-local -> sim world
                obj_rot_local = quat_mul(demo_rot_conj, cur_obj_rot)
                cur_obj_rot = quat_mul(sim_wrist_rot, obj_rot_local)

            self._target_obj_root_state[torch.arange(self.num_envs), :3] = cur_obj_pos
            self._target_obj_root_state[torch.arange(self.num_envs), 3:7] = cur_obj_rot
            self._target_obj_root_state[torch.arange(self.num_envs), 7:10] = torch.zeros_like(cur_obj_vel)
            self._target_obj_root_state[torch.arange(self.num_envs), 10:13] = torch.zeros_like(cur_obj_ang_vel)

            target_obj_multi_env_ids_int32 = self._global_target_obj_indices[torch.arange(self.num_envs)].flatten()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(target_obj_multi_env_ids_int32),
                len(target_obj_multi_env_ids_int32),
            )

        self.progress_buf += self.control_freq_inv
        self.running_progress_buf += self.control_freq_inv
        self.randomize_buf += self.control_freq_inv
        
        # add observation_noise_scale_factor to extras for logging
        self.extras["observation_noise_scale_factor"] = self.observation_noise_scale_factor

        # update adapt sampling scheduler
        if self.enable_adapt_sampling and (
            (self.last_step % self.adapt_sampling_update_interval == 0 and self.last_step > 0) or 
            False
            # (self.last_step == 1200)
        ):
            print(f"Updating adapt sampling scheduler at step {self.last_step}")
            self.adapt_sampling_scheduler.update(
                # metrics=self.consecutive_reach_goal_max_buf,
                metrics=self.consecutive_reach_frames_max_buf,
            )
            self.consecutive_reach_goal_max_buf[:] = 0
            self.consecutive_reach_frames_max_buf[:] = 0
            self.consecutive_reach_frames_buf[:] = 0
            self.envidx_to_demoidx = self.adapt_sampling_scheduler.get_indices(num_envs=self.num_envs)
            self.adapt_sampling_scheduler.print_probs(num_per_line=10, precision=6)
            self.reset_buf[:] = 1.0

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera
    
    def create_multiview_cameras(self, *, env, isaac_gym):
        camera_cfg = gymapi.CameraProperties()
        camera_cfg.enable_tensors = True
        camera_cfg.width = 640
        camera_cfg.height = 360
        camera_cfg.horizontal_fov = 69.4
        cam_target = gymapi.Vec3(0, 0, 0.0)

        # four views around the wrist
        cameras = []
        cam_pos = gymapi.Vec3(0.5, 0, 0.0)
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        cameras.append(camera)
        cam_pos = gymapi.Vec3(0, 0.5, 0.0)
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        cameras.append(camera)
        cam_pos = gymapi.Vec3(-0.5, 0, 0.0)
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        cameras.append(camera)
        cam_pos = gymapi.Vec3(0, -0.5, 0.0)
        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        cameras.append(camera)
        return cameras

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis




@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    failure_progress_buf: Tensor,
    actions: Tensor,
    last_actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    dexhand_n_finger_tips: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,  Dict[str, Tensor], Dict[str, Tensor], Tensor, float, Dict[str, List[int]], int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state_rel_wrist"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos_rel_wrist"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    if dexhand_n_finger_tips == 5:
        diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    else:
        diff_ring_tip_pos_dist = torch.zeros_like(diff_thumb_tip_pos_dist)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-90 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-90 * diff_pinky_tip_pos_dist)
    if dexhand_n_finger_tips == 5:
        reward_ring_tip_pos = torch.exp(-90 * diff_ring_tip_pos_dist)
    else:
        reward_ring_tip_pos = torch.zeros_like(reward_thumb_tip_pos)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    # object pose reward in wrist frame
    current_obj_pos = states["manip_obj_pos_rel_wrist"]
    current_obj_quat = states["manip_obj_quat_rel_wrist"]

    target_obj_pos = target_states["manip_obj_pos_rel_wrist"]
    target_obj_quat = target_states["manip_obj_quat_rel_wrist"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    # reward_power = torch.exp(-10 * target_states["power"])
    # reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    # finger_tip_force = target_states["tip_force"]
    # finger_tip_distance = target_states["tips_distance"]
    # contact_range = [0.02, 0.03]
    # finger_tip_weight = torch.clamp(
    #     (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    # )
    # finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    # reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    ## ? add action rate reqularization
    penalty_action_rate = torch.norm(actions - last_actions, dim=-1)

    ## ? add action delta regularization
    action_delta = torch.abs(states["q_curr_targets"] - states["q"])
    penalty_action_delta = torch.norm(3 * torch.relu(action_delta - 0.1), dim=-1)

    ## ? add dof pos regularization
    target_dof_pos = target_states["dof_pos"]
    diff_dof_pos_action = (target_dof_pos - states["q_curr_targets"]).abs()
    reward_dof_pos_action = torch.exp(-5 * diff_dof_pos_action.mean(dim=-1))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3)  # TODO
            # | (diff_thumb_tip_pos_dist > 0.03 / 0.7 * scale_factor)
            # | (diff_index_tip_pos_dist > 0.03 / 0.7 * scale_factor)
            # | (diff_middle_tip_pos_dist > 0.03 / 0.7 * scale_factor)
            # | (diff_ring_tip_pos_dist > 0.03 / 0.7 * scale_factor if dexhand_n_finger_tips == 5 else torch.zeros_like(diff_thumb_tip_pos_dist, dtype=torch.bool))
            # | (diff_pinky_tip_pos_dist > 0.03 / 0.7 * scale_factor)
            # | (diff_level_1_pos_dist > 0.05 / 0.7 * scale_factor)
            # | (diff_level_2_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor**3)  # TODO
            # | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
        & (running_progress_buf >= 1)
    ) | error_buf

    failure_progress_buf += torch.where(failed_execute, torch.ones_like(failure_progress_buf), torch.zeros_like(failure_progress_buf))
    failed_execute = (failure_progress_buf >= 1) | error_buf

    penalty_reward_scale = 0.0

    reward_execute = (
        # 0.1 * reward_eef_pos
        # + 0.6 * reward_eef_rot
        1.0 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.8 * reward_middle_tip_pos
        + 0.8 * reward_pinky_tip_pos
        + 0.8 * (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
        + 0.6 * reward_level_1_pos
        + 0.4 * reward_level_2_pos
        + 5.0 * reward_obj_pos * (1 - scale_factor) * 0.3
        + 5.0 * reward_obj_rot * (1 - scale_factor) * 0.3
        # + 0.5 * reward_power
        + 1.0 * reward_dof_pos_action
    ) + (
        - 0.2 * penalty_action_rate
        - 0.2 * penalty_action_delta
    ) * penalty_reward_scale

    succeeded = (
        progress_buf + 1 + 50 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_dof_pos_action": reward_dof_pos_action,
        # "reward_power": reward_power,
        # "reward_wrist_power": reward_wrist_power,
        # "reward_finger_tip_force": reward_finger_tip_force,
        "penalty_action_rate": penalty_action_rate,
        "penalty_action_delta": penalty_action_delta
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf, failure_progress_buf


def compute_imitation_reward_latency_tracking(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    failure_progress_buf: Tensor,
    global_cur_idx: Tensor,
    stable_frames_buf: Tensor,
    consecutive_reach_goal_buf: Tensor,
    consecutive_reach_frames_buf: Tensor,
    skip_steps_buf: Tensor,
    skip_steps_min: int,
    skip_steps_max: int,
    reverse_target_prob: float,
    fixed_tolerance_steps: int,
    failure_tolerance_scale: float,
    num_frames_to_stay_buf: Tensor,
    num_frames_to_stay_lower_bound: int,
    num_frames_to_stay_upper_bound: int,
    actions: Tensor,
    last_actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    dexhand_n_finger_tips: int,
    obj_pos_thres: float,
    thumb_tip_pos_thres: float,
    index_tip_pos_thres: float,
    middle_tip_pos_thres: float,
    pinky_tip_pos_thres: float,
    ring_tip_pos_thres: float,
    obj_rot_thres: float,
    failure_obj_pos_thres: float,
    failure_thumb_tip_pos_thres: float,
    failure_index_tip_pos_thres: float,
    failure_middle_tip_pos_thres: float,
    failure_pinky_tip_pos_thres: float,
    failure_ring_tip_pos_thres: float,
    failure_obj_rot_thres: float,
    invalid_obj_pos_thres: float,
    time_penalty: float,
    traj_direction: Tensor,
    traj_steps_counter: Tensor,
    traj_steps_limit: int,
    is_target_cross: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    joints_vel = states["joints_state"][:, 1:, 7:10]
    current_dof_vel = states["dq"]
    current_obj_vel = states["manip_obj_vel"]
    current_obj_ang_vel = states["manip_obj_ang_vel"]

    ## Hand reward
    # joints_pos = states["joints_state"][:, 1:, :3]
    # target_joints_pos = target_states["joints_pos"]
    # diff_joints_pos = target_joints_pos - joints_pos
    # diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # Hand diff in wrist frame
    joints_pos = states['joints_state_rel_wrist'][:, 1:, :3]
    target_joints_pos = target_states['joints_pos_rel_wrist']
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    if dexhand_n_finger_tips == 5:
        diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    else:
        diff_ring_tip_pos_dist = torch.zeros_like(diff_thumb_tip_pos_dist)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-90 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-90 * diff_pinky_tip_pos_dist)
    if dexhand_n_finger_tips == 5:
        reward_ring_tip_pos = torch.exp(-90 * diff_ring_tip_pos_dist)
    else:
        reward_ring_tip_pos = torch.zeros_like(reward_thumb_tip_pos)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    target_dof_pos = target_states["dof_pos"]
    diff_dof_pos_action = (target_dof_pos - states["q_curr_targets"]).abs()
    reward_dof_pos_action = torch.exp(-5 * diff_dof_pos_action.mean(dim=-1))

    ## Object reward
    # current_obj_pos = states["manip_obj_pos"]
    # current_obj_quat = states["manip_obj_quat"]
    # target_obj_pos = target_states["manip_obj_pos"]
    # target_obj_quat = target_states["manip_obj_quat"]

    ## Object reward in wrist frame
    current_obj_pos = states["manip_obj_pos_rel_wrist"]
    current_obj_quat = states["manip_obj_quat_rel_wrist"]
    target_obj_pos = target_states["manip_obj_pos_rel_wrist"]
    target_obj_quat = target_states["manip_obj_quat_rel_wrist"]

    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    ## Penalty
    penalty_action_rate = torch.norm(actions - last_actions, dim=-1)

    action_delta = torch.abs(states["q_curr_targets"] - states["q"])
    penalty_action_delta = torch.norm(3 * torch.relu(action_delta - 0.1), dim=-1)

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )  # sanity check

    ## Success condition
    obj_pos_success_condition = (diff_obj_pos_dist <= obj_pos_thres)
    thumb_tip_pos_success_condition = (diff_thumb_tip_pos_dist <= thumb_tip_pos_thres)
    index_tip_pos_success_condition = (diff_index_tip_pos_dist <= index_tip_pos_thres)
    middle_tip_pos_success_condition = (diff_middle_tip_pos_dist <= middle_tip_pos_thres)
    pinky_tip_pos_success_condition = (diff_pinky_tip_pos_dist <= pinky_tip_pos_thres)
    if dexhand_n_finger_tips == 5:
        ring_tip_pos_success_condition = (diff_ring_tip_pos_dist <= ring_tip_pos_thres)
    obj_rot_success_condition = (diff_obj_rot_angle.abs() / np.pi * 180 <= obj_rot_thres)

    invalid_buf = (
        diff_obj_pos_dist > invalid_obj_pos_thres
    )

    reach_goal = (
        (
            obj_pos_success_condition
            & thumb_tip_pos_success_condition
            & index_tip_pos_success_condition
            & middle_tip_pos_success_condition
            & pinky_tip_pos_success_condition
            & (ring_tip_pos_success_condition if dexhand_n_finger_tips == 5 else True)
            & obj_rot_success_condition
        )
        & (running_progress_buf >= 1)
    ) & (~error_buf)

    ## Failed Condition
    failed_execute = (
        (
            (diff_obj_pos_dist > failure_obj_pos_thres)  # TODO
            | (diff_thumb_tip_pos_dist > failure_thumb_tip_pos_thres)
            | (diff_index_tip_pos_dist > failure_index_tip_pos_thres)
            | (diff_middle_tip_pos_dist > failure_middle_tip_pos_thres)
            | ((diff_ring_tip_pos_dist > failure_ring_tip_pos_thres) if dexhand_n_finger_tips == 5 else False)
            | (diff_pinky_tip_pos_dist > failure_pinky_tip_pos_thres)
            # | (diff_level_1_pos_dist > 0.05 / 0.7 * scale_factor)
            # | (diff_level_2_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_obj_rot_angle.abs() / np.pi * 180 > failure_obj_rot_thres)  # TODO
            # | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
        )
        & (running_progress_buf >= 1)
    ) | error_buf

    failure_progress_buf += torch.where(failed_execute, torch.ones_like(failure_progress_buf), torch.zeros_like(failure_progress_buf))
    # tolerance_steps = failure_tolerance_scale * skip_steps_buf.abs() + fixed_tolerance_steps
    tolerance_steps = torch.where(
        is_target_cross,
        fixed_tolerance_steps,
        failure_tolerance_scale * skip_steps_buf.abs(),
    )
    failed_execute = (failure_progress_buf >= tolerance_steps) | error_buf | invalid_buf # set to 100 for latency tracking

    ## Extend traj logic
    reached_end = (global_cur_idx + 1 + 50 >= max_length) & ~failed_execute
    reached_start = (global_cur_idx <= 50) & (traj_direction == -1) & ~failed_execute

    hit_traj_limit = (traj_steps_counter >= traj_steps_limit) if traj_steps_limit is not None else torch.zeros_like(global_cur_idx, dtype=torch.bool)
    succeeded = hit_traj_limit & ~failed_execute

    origin_traj_direction = traj_direction.clone()
    traj_direction = torch.where(
        reached_end & (origin_traj_direction == 1),
        torch.ones_like(traj_direction) * -1,
        traj_direction
    )
    traj_direction = torch.where(
        reached_start & (origin_traj_direction == -1),
        torch.ones_like(traj_direction),
        traj_direction
    )
    
    stable_frames_buf = torch.where(
        reach_goal,
        stable_frames_buf + 1,
        torch.zeros_like(stable_frames_buf) # reset to 0 when lost goal
    )
    reach_final_goal = (stable_frames_buf >= num_frames_to_stay_buf)
    stable_frames_buf = torch.where(
        reach_final_goal,
        torch.zeros_like(stable_frames_buf),
        stable_frames_buf
    )

    step_increment = skip_steps_buf.abs()
    traj_steps_counter = torch.where(
        hit_traj_limit,
        torch.zeros_like(traj_steps_counter),
        traj_steps_counter + torch.where(
            reach_final_goal,
            step_increment,
            torch.zeros_like(step_increment)
        )
    )

    # Update num_frames_to_stay_buf when reach_final_goal
    num_frames_to_stay_buf = torch.where(
        reach_final_goal,
        torch.randint(
            low=num_frames_to_stay_lower_bound,
            high=num_frames_to_stay_upper_bound + 1,
            size=global_cur_idx.shape,
            device=global_cur_idx.device,
            dtype=torch.int32,
        ),
        num_frames_to_stay_buf,
    )
    skip_steps = torch.randint(
        low=skip_steps_min,
        high=int(
            (skip_steps_max - skip_steps_min) / (1.0 - 0.343) * (1.0 - scale_factor**3) + skip_steps_min + 1
        ),
        dtype=torch.int32,
        size=global_cur_idx.shape,
        device=global_cur_idx.device
    )
    if reverse_target_prob > 0:
        reverse_target = torch.rand(global_cur_idx.shape, device=global_cur_idx.device) < reverse_target_prob
        skip_steps = torch.where(
            reverse_target,
            -skip_steps,
            skip_steps,
        )
    consecutive_reach_goal_buf = torch.where(
        reach_final_goal,
        consecutive_reach_goal_buf + 1,
        consecutive_reach_goal_buf,
    )
    consecutive_reach_frames_buf = torch.where(
        reach_final_goal,
        consecutive_reach_frames_buf + skip_steps_buf.abs(),
        consecutive_reach_frames_buf,
    )
    
    goal_reward_step_weight = skip_steps_buf.clone().abs() + 5.0
    goal_reward_step_weight = torch.where(
        is_target_cross,
        100.0,
        goal_reward_step_weight,
    )
    skip_steps_buf = torch.where(
        (reach_final_goal | failed_execute),
        skip_steps,
        skip_steps_buf,
    )

    ## Update global_cur_idx for successful reaches
    global_cur_idx = torch.where(
        reach_final_goal & (traj_direction == 1),
        global_cur_idx + skip_steps,
        global_cur_idx
    )
    global_cur_idx = torch.where(
        reach_final_goal & (traj_direction == -1),
        global_cur_idx - skip_steps,
        global_cur_idx
    )
    failure_progress_buf = torch.where(
        reach_final_goal,
        torch.zeros_like(failure_progress_buf),
        failure_progress_buf,
    )
    global_cur_idx = torch.clamp(global_cur_idx, torch.zeros_like(max_length), max_length - 1)
    reward_reach_goal = reach_final_goal.float()

    # general mode
    # reach_score = (
    #     1.0 * reward_thumb_tip_pos
    #     + 0.8 * reward_index_tip_pos
    #     + 0.8 * reward_middle_tip_pos
    #     + 0.8 * reward_pinky_tip_pos
    #     + 2.0 * reward_obj_pos
    #     + 2.0 * reward_obj_rot
    # ) * 1.5

    # finger_tip_reward_scale = 0.5
    finger_tip_reward_scale = 0.5


    # hard object mode
    reach_score = (
        finger_tip_reward_scale * reward_thumb_tip_pos
        + finger_tip_reward_scale * reward_index_tip_pos
        + finger_tip_reward_scale * reward_middle_tip_pos
        + finger_tip_reward_scale * reward_pinky_tip_pos
        + (finger_tip_reward_scale * reward_ring_tip_pos if dexhand_n_finger_tips == 5 else 0.0)
        + 2.0 * reward_obj_pos
        + 2.0 * reward_obj_rot
    ) * 1.5
    static_reach_score = 8.0

    dense_reward_scale = 0.1
    # dense_reward_scale = 0.
    penalty_reward_scale = 0.0

    reward_execute = (
        1.0 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.8 * reward_middle_tip_pos
        + 0.8 * reward_pinky_tip_pos
        + 0.8 * (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
        + 0.6 * reward_level_1_pos
        + 0.4 * reward_level_2_pos
        + 5.0 * reward_obj_pos * (1 - scale_factor) * 0.3
        + 5.0 * reward_obj_rot * (1 - scale_factor) * 0.3
        # + 1.0 * reward_finger_tip_force
        # + 0.5 * reward_power
        # + 0.5 * reward_wrist_power
        # + 1.0 * reward_dof_pos_action
    ) * dense_reward_scale + (
        - 0.2 * penalty_action_rate
        - 0.2 * penalty_action_delta
        # - 1.0 * penalty_action_rate
        # - 1.0 * penalty_action_delta
    ) * penalty_reward_scale + \
    (
        # static_reach_score * \
        reach_score * \
        goal_reward_step_weight.float() * \
        reward_reach_goal
    ) - time_penalty

    reset_buf = torch.where(
        failed_execute | hit_traj_limit,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_dof_pos_action": reward_dof_pos_action,
    }

    # Collect errors for logging at reach_final_goal
    reach_errors = (
        diff_obj_pos_dist,
        diff_obj_rot_angle.abs() / np.pi * 180,  # Convert to degrees
        diff_thumb_tip_pos_dist,
        diff_index_tip_pos_dist,
        diff_middle_tip_pos_dist,
        diff_pinky_tip_pos_dist,
    )

    return (
        reward_execute,
        reset_buf,
        succeeded,
        failed_execute,
        reward_dict,
        error_buf,
        failure_progress_buf,
        global_cur_idx,
        stable_frames_buf,
        consecutive_reach_goal_buf,
        consecutive_reach_frames_buf,
        skip_steps_buf,
        reach_final_goal,
        num_frames_to_stay_buf,
        traj_direction,
        traj_steps_counter,
        reach_errors,
    )



class SinDexHandManipLHEnv(SinDexHandManipRHEnv):
    side = "left"

    def __init__(
        self,
        cfg,
        *,
        rl_device=0,
        sim_device=0,
        graphics_device_id=0,
        display=False,
        record=False,
        headless=True,
    ):
        self.dexhand = DexHandFactory.create_hand(cfg["env"]["dexhand"], "left")
        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )


# @torch.jit.script
# def compute_imitation_reward_latency_tracking(
#     reset_buf: Tensor,
#     progress_buf: Tensor,
#     running_progress_buf: Tensor,
#     failure_progress_buf: Tensor,
#     global_cur_idx: Tensor,
#     stable_frames_buf: Tensor,
#     consecutive_reach_goal_buf: Tensor,
#     consecutive_reach_frames_buf: Tensor,
#     skip_steps_buf: Tensor,
#     skip_steps_min: int,
#     skip_steps_max: int,
#     reverse_target_prob: float,
#     failure_tolerance_scale: float,
#     num_frames_to_stay_buf: Tensor,
#     num_frames_to_stay_lower_bound: int,
#     num_frames_to_stay_upper_bound: int,
#     actions: Tensor,
#     last_actions: Tensor,
#     states: Dict[str, Tensor],
#     target_states: Dict[str, Tensor],
#     max_length: List[int],
#     scale_factor: float,
#     dexhand_weight_idx: Dict[str, List[int]],
#     dexhand_n_finger_tips: int,
#     obj_pos_thres: float,
#     thumb_tip_pos_thres: float,
#     index_tip_pos_thres: float,
#     middle_tip_pos_thres: float,
#     pinky_tip_pos_thres: float,
#     obj_rot_thres: float,
#     time_penalty: float,
#     traj_direction: Tensor,
#     traj_steps_counter: Tensor,
#     traj_steps_limit: int,
# ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#     # end effector pose reward
#     current_eef_pos = states["base_state"][:, :3]
#     current_eef_quat = states["base_state"][:, 3:7]

#     target_eef_pos = target_states["wrist_pos"]
#     target_eef_quat = target_states["wrist_quat"]
#     diff_eef_pos = target_eef_pos - current_eef_pos
#     diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

#     current_eef_vel = states["base_state"][:, 7:10]
#     current_eef_ang_vel = states["base_state"][:, 10:13]
#     target_eef_vel = target_states["wrist_vel"]
#     target_eef_ang_vel = target_states["wrist_ang_vel"]

#     diff_eef_vel = target_eef_vel - current_eef_vel
#     diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

#     joints_pos = states["joints_state"][:, 1:, :3]
#     target_joints_pos = target_states["joints_pos"]
#     diff_joints_pos = target_joints_pos - joints_pos
#     diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

#     # ? assign different weights to different joints
#     # assert diff_joints_pos_dist.shape[1] == 17  # ignore the base joint
#     diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
#     diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
#     diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
#     diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
#     if dexhand_n_finger_tips == 5:
#         diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
#     else:
#         diff_ring_tip_pos_dist = torch.zeros_like(diff_thumb_tip_pos_dist)
#     diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
#     diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

#     joints_vel = states["joints_state"][:, 1:, 7:10]
#     target_joints_vel = target_states["joints_vel"]
#     diff_joints_vel = target_joints_vel - joints_vel

#     reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
#     reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
#     reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
#     reward_middle_tip_pos = torch.exp(-90 * diff_middle_tip_pos_dist)
#     reward_pinky_tip_pos = torch.exp(-90 * diff_pinky_tip_pos_dist)
#     if dexhand_n_finger_tips == 5:
#         reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
#     else:
#         reward_ring_tip_pos = torch.zeros_like(reward_thumb_tip_pos)
#     reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
#     reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

#     reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
#     reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
#     reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
#     current_dof_vel = states["dq"]

#     diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
#     diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
#     reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

#     # object pose reward
#     current_obj_pos = states["manip_obj_pos"]
#     current_obj_quat = states["manip_obj_quat"]

#     target_obj_pos = target_states["manip_obj_pos"]
#     target_obj_quat = target_states["manip_obj_quat"]
#     diff_obj_pos = target_obj_pos - current_obj_pos
#     diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

#     reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

#     diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
#     diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
#     reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

#     current_obj_vel = states["manip_obj_vel"]
#     target_obj_vel = target_states["manip_obj_vel"]
#     diff_obj_vel = target_obj_vel - current_obj_vel
#     reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

#     current_obj_ang_vel = states["manip_obj_ang_vel"]
#     target_obj_ang_vel = target_states["manip_obj_ang_vel"]
#     diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
#     reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

#     reward_power = torch.exp(-10 * target_states["power"])
#     reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

#     finger_tip_force = target_states["tip_force"]
#     finger_tip_distance = target_states["tips_distance"]
#     contact_range = [0.02, 0.03]
#     finger_tip_weight = torch.clamp(
#         (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
#     )
#     finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

#     reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

#     ## ? add action rate reqularization
#     penalty_action_rate = torch.norm(actions - last_actions, dim=-1)

#     ## ? add action delta regularization
#     action_delta = torch.abs(states["q_curr_targets"] - states["q"])
#     penalty_action_delta = torch.norm(3 * torch.relu(action_delta - 0.1), dim=-1)

#     ## ? add dof pos regularization
#     target_dof_pos = target_states["dof_pos"]
#     diff_dof_pos_action = (target_dof_pos - states["q_curr_targets"]).abs()
#     reward_dof_pos_action = torch.exp(-5 * diff_dof_pos_action.mean(dim=-1))

#     error_buf = (
#         (torch.norm(current_eef_vel, dim=-1) > 100)
#         | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
#         | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
#         | (torch.abs(current_dof_vel).mean(-1) > 200)
#         | (torch.norm(current_obj_vel, dim=-1) > 100)
#         | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
#     )  # sanity check

#     ## self.enable_latency_tracking:
#     # failed_execute = torch.zeros_like(reset_buf) | error_buf
#     # obj_pos_thres = 0.006 # 0.01
#     # thumb_tip_pos_thres = 0.006 # 0.008
#     # index_tip_pos_thres = 0.025 # 0.02
#     # middle_tip_pos_thres = 0.025 # 0.02
#     # pinky_tip_pos_thres = 0.025 # 0.02
#     # # thumb_tip_pos_thres = 10.01
#     # # index_tip_pos_thres = 10.025
#     # # middle_tip_pos_thres = 10.025
#     # # pinky_tip_pos_thres = 10.025
#     # obj_rot_thres = 5 # 8

#     ## for test
#     # obj_pos_thres = 0.015 # 0.01
#     # thumb_tip_pos_thres = 0.015 # 0.008
#     # index_tip_pos_thres = 0.035 # 0.02
#     # middle_tip_pos_thres = 0.035 # 0.02
#     # pinky_tip_pos_thres = 0.035 # 0.02
#     # # thumb_tip_pos_thres = 10.01
#     # # index_tip_pos_thres = 10.025
#     # # middle_tip_pos_thres = 10.025
#     # # pinky_tip_pos_thres = 10.025
#     # obj_rot_thres = 12 # 8

#     ## training simple
#     # obj_pos_thres = 0.008 # 0.01
#     # thumb_tip_pos_thres = 0.008 # 0.008
#     # index_tip_pos_thres = 0.025 # 0.02
#     # middle_tip_pos_thres = 0.025 # 0.02
#     # pinky_tip_pos_thres = 0.025 # 0.02
#     # # thumb_tip_pos_thres = 10.01
#     # # index_tip_pos_thres = 10.025
#     # # middle_tip_pos_thres = 10.025
#     # # pinky_tip_pos_thres = 10.025
#     # obj_rot_thres = 6 # 8

#     ## training simple (better for translation task)
#     # obj_pos_thres = 0.004 # 0.01
#     # thumb_tip_pos_thres = 0.008 # 0.008
#     # index_tip_pos_thres = 0.025 # 0.02
#     # middle_tip_pos_thres = 0.025 # 0.02
#     # pinky_tip_pos_thres = 0.02 # 0.02
#     # # thumb_tip_pos_thres = 10.01
#     # # index_tip_pos_thres = 10.025
#     # # middle_tip_pos_thres = 10.025
#     # # pinky_tip_pos_thres = 10.025
#     # obj_rot_thres = 6 # 8

#     obj_pos_success_condition = (diff_obj_pos_dist <= obj_pos_thres)
#     thumb_tip_pos_success_condition = (diff_thumb_tip_pos_dist <= thumb_tip_pos_thres)
#     index_tip_pos_success_condition = (diff_index_tip_pos_dist <= index_tip_pos_thres)
#     middle_tip_pos_success_condition = (diff_middle_tip_pos_dist <= middle_tip_pos_thres)
#     pinky_tip_pos_success_condition = (diff_pinky_tip_pos_dist <= pinky_tip_pos_thres)
#     obj_rot_success_condition = (diff_obj_rot_angle.abs() / np.pi * 180 <= obj_rot_thres)

#     invalid_buf = (
#         diff_obj_pos_dist > 0.15
#     )

#     # print(f"obj pos success: {obj_pos_success_condition[0]}")
#     # print(f"thumb tip pos success: {thumb_tip_pos_success_condition[0]}")
#     # print(f"index tip pos success: {index_tip_pos_success_condition[0]}")
#     # print(f"middle tip pos success: {middle_tip_pos_success_condition[0]}")
#     # print(f"pinky tip pos success: {pinky_tip_pos_success_condition[0]}")
#     # print(f"obj rot success: {obj_rot_success_condition[0]}")

#     reach_goal = (
#         (
#             obj_pos_success_condition
#             & thumb_tip_pos_success_condition
#             & index_tip_pos_success_condition
#             & middle_tip_pos_success_condition
#             & pinky_tip_pos_success_condition
#             & obj_rot_success_condition
#         )
#         & (running_progress_buf >= 1) 
#     ) & (~error_buf)

#     ## normal tracking
#     failed_execute = (
#         (
#             (diff_obj_pos_dist > obj_pos_thres)  # TODO
#             | (diff_thumb_tip_pos_dist > thumb_tip_pos_thres)
#             | (diff_index_tip_pos_dist > index_tip_pos_thres)
#             | (diff_middle_tip_pos_dist > middle_tip_pos_thres)
#             # | (diff_ring_tip_pos_dist > 0.015 / 0.7 * scale_factor if dexhand_n_finger_tips == 5 else torch.zeros_like(diff_thumb_tip_pos_dist, dtype=torch.bool))
#             | (diff_pinky_tip_pos_dist > pinky_tip_pos_thres)
#             # | (diff_level_1_pos_dist > 0.05 / 0.7 * scale_factor)
#             # | (diff_level_2_pos_dist > 0.05 / 0.7 * scale_factor)
#             | (diff_obj_rot_angle.abs() / np.pi * 180 > obj_rot_thres)  # TODO
#             # | torch.any((finger_tip_distance < 0.005) & ~(target_states["tip_contact_state"].any(1)), dim=-1)
#         )
#         & (running_progress_buf >= 1)
#     ) | error_buf

#     failure_progress_buf += torch.where(failed_execute, torch.ones_like(failure_progress_buf), torch.zeros_like(failure_progress_buf))
#     failed_execute = (failure_progress_buf >= failure_tolerance_scale * skip_steps_buf.abs()) | error_buf | invalid_buf # set to 100 for latency tracking

#     # from termcolor import cprint
#     # cprint(f"diff_obj_pos_dist: {diff_obj_pos_dist[0].item()}", "green")
#     # cprint(f"diff_obj_rot_angle: {diff_obj_rot_angle[0].item()}", "green")
#     # cprint(f"diff_thumb_tip_pos_dist: {diff_thumb_tip_pos_dist[0].item()}", "green")
#     # cprint(f"diff_index_tip_pos_dist: {diff_index_tip_pos_dist[0].item()}", "green")
#     # cprint(f"diff_middle_tip_pos_dist: {diff_middle_tip_pos_dist[0].item()}", "green")
#     # cprint(f"diff_pinky_tip_pos_dist: {diff_pinky_tip_pos_dist[0].item()}", "green")
#     # print(f"reset: {failed_execute.sum()}")

#     # Detect end of trajectory in forward direction
#     reached_end = (global_cur_idx + 1 + 50 >= max_length) & ~failed_execute
#     # Detect start of trajectory in reverse direction
#     reached_start = (global_cur_idx <= 50) & (traj_direction == -1) & ~failed_execute

#     # Handle traj_steps_limit - if exceeded, reset
#     hit_traj_limit = (traj_steps_counter >= traj_steps_limit) if traj_steps_limit is not None else torch.zeros_like(global_cur_idx, dtype=torch.bool)
#     succeeded = (reached_end | reached_start | hit_traj_limit) & ~failed_execute

#     # Save original direction for position calculation
#     orig_traj_direction = traj_direction.clone()

#     # Update traj_direction
#     # When reaching end in forward mode -> switch to reverse
#     traj_direction = torch.where(
#         reached_end & (traj_direction == 1),
#         torch.ones_like(traj_direction) * -1,
#         traj_direction
#     )
#     # When reaching start in reverse mode -> switch to forward
#     traj_direction = torch.where(
#         reached_start & (traj_direction == -1),
#         torch.ones_like(traj_direction),
#         traj_direction
#     )

#     # Calculate new position when switching direction
#     # Forward (1): idx goes from current to traj_len-1, then reverse
#     # Reverse (-1): idx goes from current to 0, then forward
#     # new_idx_forward = max_length - 1 - global_cur_idx (go back from current to start)
#     # new_idx_reverse = max_length - 1 - global_cur_idx (go forward from current to end)
#     new_idx_forward = global_cur_idx  # Continue forward from current position
#     new_idx_reverse = max_length.long() - 1 - global_cur_idx.long()  # Reverse from current position

#     # Use original direction for the check
#     global_cur_idx = torch.where(
#         reached_end & (orig_traj_direction == 1),
#         new_idx_reverse,
#         global_cur_idx
#     )
#     global_cur_idx = torch.where(
#         reached_start & (orig_traj_direction == -1),
#         new_idx_forward,
#         global_cur_idx
#     )

#     # Compute reach_final_goal before updating traj_steps_counter
#     # Randomly skip steps between SKIP_STEPS_MIN and SKIP_STEPS_MAX when reach_goal is True, else 0.
#     stable_frames_buf = torch.where(
#         reach_goal,
#         stable_frames_buf + 1,
#         torch.zeros_like(stable_frames_buf) # reset to 0 when lost goal
#     )
#     reach_final_goal = (stable_frames_buf >= num_frames_to_stay_buf)
#     stable_frames_buf = torch.where(
#         reach_final_goal,
#         torch.zeros_like(stable_frames_buf),
#         stable_frames_buf
#     )

#     # Update traj_steps_counter
#     # Only increment when reach_final_goal (actually advancing through trajectory)
#     # Forward + reverse count toward the same limit
#     step_increment = skip_steps_buf.abs()
#     # Only reset counter when hit_traj_limit (not on direction switch)
#     traj_steps_counter = torch.where(
#         hit_traj_limit,
#         torch.zeros_like(traj_steps_counter),
#         traj_steps_counter + torch.where(reach_final_goal, step_increment, torch.zeros_like(step_increment))
#     )

#     # Update global_cur_idx for successful reaches (switch direction and continue)
#     # Use original direction for the check
#     global_cur_idx = torch.where(
#         reached_end & (orig_traj_direction == 1),
#         global_cur_idx + step_increment,
#         global_cur_idx
#     )
#     global_cur_idx = torch.where(
#         reached_start & (orig_traj_direction == -1),
#         global_cur_idx + step_increment,
#         global_cur_idx
#     )

#     # Update num_frames_to_stay_buf when reach_final_goal
#     num_frames_to_stay_buf = torch.where(
#         reach_final_goal,
#         torch.randint(
#             low=num_frames_to_stay_lower_bound,
#             high=num_frames_to_stay_upper_bound + 1,
#             size=global_cur_idx.shape,
#             device=global_cur_idx.device,
#             dtype=torch.int32,
#         ),
#         num_frames_to_stay_buf,
#     )
#     skip_steps = torch.randint(
#         low=skip_steps_min,
#         high=int(
#             (skip_steps_max - skip_steps_min) / (1.0 - 0.343) * (1.0 - scale_factor**3) + skip_steps_min + 1
#         ),
#         dtype=torch.int32,
#         size=global_cur_idx.shape,
#         device=global_cur_idx.device
#     )
#     if reverse_target_prob > 0:
#         reverse_target = torch.rand(global_cur_idx.shape, device=global_cur_idx.device) < reverse_target_prob
#         skip_steps = torch.where(
#             reverse_target,
#             -skip_steps,
#             skip_steps,
#         )
#     consecutive_reach_goal_buf = torch.where(
#         reach_final_goal,
#         consecutive_reach_goal_buf + 1,
#         consecutive_reach_goal_buf,
#     )
#     consecutive_reach_frames_buf = torch.where(
#         reach_final_goal,
#         consecutive_reach_frames_buf + skip_steps_buf.abs(),
#         consecutive_reach_frames_buf,
#     )
#     goal_reward_step_weight = skip_steps_buf.clone().abs() + 5.0
#     skip_steps_buf = torch.where(
#         (reach_final_goal | failed_execute),
#         skip_steps,
#         skip_steps_buf,
#     )
#     # Apply skip_steps based on traj_direction
#     # Forward (1): add skip_steps, Reverse (-1): subtract skip_steps
#     global_cur_idx = torch.where(
#         reach_final_goal & (traj_direction == 1),
#         global_cur_idx + skip_steps,
#         global_cur_idx
#     )
#     global_cur_idx = torch.where(
#         reach_final_goal & (traj_direction == -1),
#         global_cur_idx - skip_steps,
#         global_cur_idx
#     )
#     failure_progress_buf = torch.where(
#         reach_final_goal,
#         torch.zeros_like(failure_progress_buf),
#         failure_progress_buf,
#     )
#     global_cur_idx = torch.clamp(global_cur_idx, torch.zeros_like(max_length), max_length - 1)
#     reward_reach_goal = reach_final_goal.float()

#     # general mode
#     # reach_score = (
#     #     1.0 * reward_thumb_tip_pos
#     #     + 0.8 * reward_index_tip_pos
#     #     + 0.8 * reward_middle_tip_pos
#     #     + 0.8 * reward_pinky_tip_pos
#     #     + 2.0 * reward_obj_pos
#     #     + 2.0 * reward_obj_rot
#     # ) * 1.5

#     # hard object mode
#     reach_score = (
#         1.2 * reward_thumb_tip_pos
#         + 1.2 * reward_index_tip_pos
#         + 1.2 * reward_middle_tip_pos
#         + 1.2 * reward_pinky_tip_pos
#         + 1.2 * reward_obj_pos
#         + 1.2 * reward_obj_rot
#     ) * 1.5

#     static_reach_score = 8.0

#     reward_execute = (
#         # 0.1 * reward_eef_pos
#         # + 0.6 * reward_eef_rot
#         1.0 * reward_thumb_tip_pos
#         + 0.8 * reward_index_tip_pos
#         + 0.8 * reward_middle_tip_pos
#         + 0.8 * reward_pinky_tip_pos
#         + 0.8 * (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
#         + 0.6 * reward_level_1_pos
#         + 0.4 * reward_level_2_pos
#         + 5.0 * reward_obj_pos * (1 - scale_factor) * 0.3
#         + 5.0 * reward_obj_rot * (1 - scale_factor) * 0.3
#         # + 10.0 * reward_obj_pos * (1 - scale_factor) * 0.3
#         # + 10.0 * reward_obj_rot * (1 - scale_factor) * 0.3
#         # + 0.1 * reward_eef_vel
#         # + 0.05 * reward_eef_ang_vel
#         # + 0.5 * reward_joints_vel
#         # + 0.5 * reward_obj_vel
#         # + 0.5 * reward_obj_ang_vel
#         # + 1.0 * reward_finger_tip_force
#         + 0.5 * reward_power
#         # + 0.5 * reward_wrist_power

#         + 1.0 * reward_dof_pos_action
#     ) * 0.1 + (
#         - 0.2 * penalty_action_rate
#         - 0.2 * penalty_action_delta
#         # - 0.0 * penalty_action_rate
#         # - 1.0 * penalty_action_rate
#         # - 1.0 * penalty_action_delta
#     ) * 0.1 + \
#         (
#         # static_reach_score * \
#         reach_score * \
#         goal_reward_step_weight.float() * \
#         # consecutive_reach_goal_buf.float() * \
#         reward_reach_goal
#     ) - time_penalty


#     ## normal tracking
#     # succeeded = (
#     #     progress_buf + 1 + 50 >= max_length
#     # ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps

#     reset_buf = torch.where(
#         succeeded | failed_execute,
#         torch.ones_like(reset_buf),
#         reset_buf,
#     )
#     reward_dict = {
#         # "reward_eef_pos": reward_eef_pos,
#         # "reward_eef_rot": reward_eef_rot,
#         # "reward_eef_vel": reward_eef_vel,
#         # "reward_eef_ang_vel": reward_eef_ang_vel,
#         # "reward_joints_vel": reward_joints_vel,
#         "reward_obj_pos": reward_obj_pos,
#         "reward_obj_rot": reward_obj_rot,
#         "reward_obj_vel": reward_obj_vel,
#         "reward_obj_ang_vel": reward_obj_ang_vel,
#         "reward_joints_pos": (
#             reward_thumb_tip_pos
#             + reward_index_tip_pos
#             + reward_middle_tip_pos
#             + reward_pinky_tip_pos
#             + (reward_ring_tip_pos if dexhand_n_finger_tips == 5 else torch.zeros_like(reward_thumb_tip_pos))
#             + reward_level_1_pos
#             + reward_level_2_pos
#         ),
#         "reward_dof_pos_action": reward_dof_pos_action,
#         "reward_power": reward_power,
#         "reward_wrist_power": reward_wrist_power,
#         "reward_finger_tip_force": reward_finger_tip_force,
#         "penalty_action_rate": penalty_action_rate,
#         "penalty_action_delta": penalty_action_delta
#     }

#     return (
#         reward_execute, 
#         reset_buf, 
#         succeeded, 
#         failed_execute, 
#         reward_dict, 
#         error_buf, 
#         failure_progress_buf, 
#         global_cur_idx, 
#         stable_frames_buf, 
#         consecutive_reach_goal_buf, 
#         consecutive_reach_frames_buf,
#         skip_steps_buf,
#         reach_final_goal,
#         num_frames_to_stay_buf,
#     )
