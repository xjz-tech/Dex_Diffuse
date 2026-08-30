# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import time
from datetime import datetime
from os.path import join
from typing import Dict, Any, Tuple

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from ...utils.dr_utils import (
    get_property_setter_map,
    get_property_getter_map,
    get_default_setter_args,
    apply_random_samples,
    check_buckets,
    generate_random_samples,
)
from ...utils.cv2_display import Cv2Display

import torch
import numpy as np
import operator, random
from copy import deepcopy

import sys

import abc
from abc import ABC

# comment out cause torchvision is not valid
# import torchvision

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)


def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


def save_getattr(obj, attr):
    try:
        return deepcopy(getattr(obj, attr))
    except:
        return getattr(obj, attr)


class Env(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless
        self.enable_viewer_sync = None
        self.viewer = None

        enable_camera_sensors = config["env"].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if not enable_camera_sensors and not display and not record and headless:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.prop_obs_dim = config["env"]["propObsDim"]
        self.privileged_obs_dim = config["env"].get("privilegedObsDim", 0)

        self._obs_keys = self.cfg["env"]["obsKeys"]
        self._privileged_obs_keys = self.cfg["env"].get("privilegedObsKeys", [])

        obs_space = {
            "proprioception": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.prop_obs_dim,),
            ),
        }

        if self.privileged_obs_dim > 0:
            obs_space.update(
                {
                    "privileged": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.privileged_obs_dim,),
                    ),
                }
            )
        self.obs_space = spaces.Dict(obs_space)

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.best_rollout_len = 0
        self.best_rollout_begin = -1

    @abc.abstractmethod
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        self.total_train_env_frames = env_frames
        # print(f'env_frames updated to {self.total_train_env_frames}')

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass


def get_external_sample(attr_randomization_params):
    if (
        "external_sample" in attr_randomization_params
        and attr_randomization_params["external_sample"]["type"] == "const_scale"
    ):
        return attr_randomization_params["external_sample"]["init_value"]
    return None


class VecTask(Env):
    dict_obs_cls = True
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`.
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        # super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs)
        super().__init__(config, rl_device, sim_device, graphics_device_id, display, record, headless)

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.dt: float = self.sim_params.dt

        self.sim_params.physx.max_gpu_contact_pairs = int(
            2 * self.sim_params.physx.max_gpu_contact_pairs
        )  # for complicated scenes

        self._set_renderers(display)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.randomize = self.cfg["task"]["randomize"]
        self.original_props = {}
        self.dr_randomizations = self.cfg["task"]["randomization_params"] if self.randomize else {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.camera_handlers = [] if (display or record) else None
        self.camera_obs = [] if (display or record) else None

        ## ! only for env[0], recording multiview videos
        self.multiview_camera_handlers = [] if (display or record) else None
        self.multiview_camera_obs = [] if (display or record) else None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.set_camera()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

    def _set_renderers(self, display):
        self._rgb_viewr_renderer = Cv2Display("IsaacGym") if display else None

    def set_camera(self):
        if self.camera_obs is not None:
            for env, handle in zip(self.envs, self.camera_handlers):
                self.camera_obs.append(
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_COLOR)
                    )
                )
        if self.multiview_camera_obs is not None:
            for handle in self.multiview_camera_handlers:
                self.multiview_camera_obs.append(
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], handle, gymapi.IMAGE_COLOR)
                    )
                )

    @staticmethod
    def create_camera(
        *,
        env,
        isaac_gym,
        img_size,
    ):
        """
        Only create front camera for view purpose
        """
        camera_cfg = gymapi.CameraProperties()
        camera_cfg.enable_tensors = True
        camera_cfg.width = img_size[0]
        camera_cfg.height = img_size[1]
        camera_cfg.horizontal_fov = 69.4

        camera = isaac_gym.create_camera_sensor(env, camera_cfg)
        cam_pos = gymapi.Vec3(0.75, 0.75, 2)
        cam_target = gymapi.Vec3(-0.1, -0.1, 0.75)
        isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_viewer(self):
        """Create the viewer."""
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            self.enable_viewer_sync = True
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                num_per_row = int(np.sqrt(self.num_envs))
                cam_pos = gymapi.Vec3(num_per_row + 1.0, num_per_row + 1.0, 3.0)
                cam_target = gymapi.Vec3(num_per_row - 6.0, num_per_row - 6.0, 1.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # Default: asymmetric actor-critic disabled. Task __init__ will override if enabled.
        if not hasattr(self, 'num_states'):
            self.num_states = 0

        # allocate buffers
        self.obs_dict = {
            "proprioception": torch.zeros(
                (self.num_envs, self.prop_obs_dim),
                device=self.device,
                dtype=torch.float,
            ),
        }
        if self.privileged_obs_dim > 0:
            self.obs_dict.update(
                {
                    "privileged": torch.zeros(
                        (self.num_envs, self.privileged_obs_dim),
                        device=self.device,
                        dtype=torch.float,
                    ),
                }
            )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.total_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.running_progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.failure_progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.failure_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.error_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.reward_dict = {}

    def create_sim(
        self,
        compute_device: int,
        graphics_device: int,
        physics_engine,
        sim_params: gymapi.SimParams,
    ):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        # Fallback: if states_buf hasn't been allocated yet (e.g., asymmetric not enabled), return zeros
        if not hasattr(self, 'states_buf') or self.states_buf is None:
            return torch.zeros((self.num_envs, 0), device=self.rl_device)
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self._rgb_viewr_renderer is not None or self.viewer is not None:
                self.render()
            # self.gym.set_dof_position_target_tensor(
            #     self.sim, 
            #     gymtorch.unwrap_tensor(self._pos_control)
            # )
            self.gym.simulate(self.sim)

        if self.camera_obs is not None or self.multiview_camera_obs is not None:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        if self.camera_obs is not None or self.multiview_camera_obs is not None:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.camera_obs is not None or self.multiview_camera_obs is not None:
            self.gym.end_access_image_tensors(self.sim)

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras["error_masks"] = torch.ones_like((~self.error_buf).float())  # ! TODO: never checked

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros(
            [self.num_envs, self.num_actions],
            dtype=torch.float32,
            device=self.rl_device,
        )

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx.
        Should be implemented in an environment class inherited from VecTask.
        """
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def render(self, mode="rgb_array"):
        if self._rgb_viewr_renderer is not None:
            rgbs = torch.stack(self.camera_obs)[..., :-1]  # RGBA -> RGB
            rgbs = rgbs.permute(0, 3, 1, 2)  # (n, 3, H, W)
            N = rgbs.shape[0]
            raise ValueError("torchvision is not valid")
            # rgb_to_display = torchvision.utils.make_grid(rgbs, nrow=N // 2)
            # self._rgb_viewr_renderer(rgb_to_display)

        if self.viewer is not None:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(
                    self.viewer,
                    join(self.record_frames_dir, f"frame_{self.control_steps}.png"),
                )

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(
                            sim_params.physx,
                            opt,
                            gymapi.ContactCollection(config_sim["physx"][opt]),
                        )
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == "color":
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + "_" + str(prop_idx) + "_" + attr
                        lo_hi = attr_randomization_params["range"]
                        distr = attr_randomization_params["distribution"]
                        if "uniform" not in distr:
                            lo_hi = (-1.0 * float("Inf"), float("Inf"))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + "_" + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(
                self.randomize_buf >= rand_freq,
                torch.ones_like(self.randomize_buf),
                torch.zeros_like(self.randomize_buf),
            )
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        if do_nonenv_randomize:
            param_setters_map = get_property_setter_map(self.gym)
            param_setter_defaults_map = get_default_setter_args(self.gym)
            param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = (
                    dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                )
                sched_step = (
                    dr_params[nonphysical_param]["schedule_steps"]
                    if "schedule" in dr_params[nonphysical_param]
                    else None
                )
                op = operator.add if op_type == "additive" else operator.mul

                if sched_type == "linear":
                    sched_scaling = 1.0 / sched_step * min(self.last_step, sched_step)
                elif sched_type == "constant":
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == "gaussian":
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0.0, 0.0])

                    if op_type == "additive":
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == "scaling":
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = corr * params["var_corr"] + params["mu_corr"]
                        return op(
                            tensor,
                            corr + torch.randn_like(tensor) * params["var"] + params["mu"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "mu": mu,
                        "var": var,
                        "mu_corr": mu_corr,
                        "var_corr": var_corr,
                        "noise_lambda": noise_lambda,
                    }

                elif dist == "uniform":
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0.0, 0.0])

                    if op_type == "additive":
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == "scaling":
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = corr * (params["hi_corr"] - params["lo_corr"]) + params["lo_corr"]
                        return op(
                            tensor,
                            corr + torch.rand_like(tensor) * (params["hi"] - params["lo"]) + params["lo"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "lo": lo,
                        "hi": hi,
                        "lo_corr": lo_corr,
                        "hi_corr": hi_corr,
                        "noise_lambda": noise_lambda,
                    }

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {attr: save_getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                external_sample = get_external_sample(attr_randomization_params)
                apply_random_samples(
                    prop,
                    self.original_props["sim_params"],
                    attr,
                    attr_randomization_params,
                    self.last_step,
                    external_sample,
                )

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            assert False, "Not implemented"  # ! TODO, temp disabled
            for env_id in env_ids:
                self.extern_actor_params[env_id] = self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props
        # and lastly loop over the ranges of the params

        for actor, actor_properties in dr_params["actor_params"].items():
            if not do_nonenv_randomize:
                continue
            # continue  # ! TODO: too slow during debugging
            # Loop over all envs as this part is not tensorised yet
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                if handle == -1:
                    continue
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties
                #          prop_attrs:
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == "color":
                        #     num_bodies = self.gym.get_actor_rigid_body_count(env, handle)
                        #     for n in range(num_bodies):
                        #         self.gym.set_rigid_body_color(
                        #             env,
                        #             handle,
                        #             n,
                        #             gymapi.MESH_VISUAL,
                        #             gymapi.Vec3(
                        #                 random.uniform(0, 1),
                        #                 random.uniform(0, 1),
                        #                 random.uniform(0, 1),
                        #             ),
                        #         )
                        continue

                    if prop_name == "scale":
                        setup_only = prop_attrs.get("setup_only", False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1, self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params["operation"] == "scaling":
                                new_scale = og_scale * sample
                            elif attr_randomization_params["operation"] == "additive":
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            if prop_name not in self.original_props:
                                self.original_props[prop_name] = {}
                            self.original_props[prop_name][f"{env_id}_{handle}"] = [
                                {attr: save_getattr(p, attr) for attr in dir(p)} for p in prop
                            ]
                        for p, og_p in zip(prop, self.original_props[prop_name][f"{env_id}_{handle}"]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get("setup_only", False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        (
                                            smpl,
                                            extern_offsets[env_id],
                                        ) = get_attr_val_from_sample(
                                            extern_sample,
                                            extern_offsets[env_id],
                                            p,
                                            attr,
                                        )
                                    external_sample = get_external_sample(attr_randomization_params)
                                    apply_random_samples(
                                        p,
                                        og_p,
                                        attr,
                                        attr_randomization_params,
                                        self.last_step,
                                        external_sample,
                                    )
                                else:
                                    set_random_properties = False
                    else:
                        # assert False, "Not implemented"
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get("setup_only", False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    (
                                        smpl,
                                        extern_offsets[env_id],
                                    ) = get_attr_val_from_sample(
                                        extern_sample,
                                        extern_offsets[env_id],
                                        prop,
                                        attr,
                                    )
                                apply_random_samples(
                                    prop,
                                    self.original_props[prop_name],
                                    attr,
                                    attr_randomization_params,
                                    self.last_step,
                                    smpl,
                                )
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        # DEBUG: collect stats across all envs on first randomization
                        if self.first_randomization and isinstance(prop, list):
                            if not hasattr(self, '_dr_debug_stats'):
                                self._dr_debug_stats = {}
                            actor_name = self.gym.get_actor_name(env, handle)
                            key = f"{actor_name}/{prop_name}"
                            if key not in self._dr_debug_stats:
                                self._dr_debug_stats[key] = {a: [] for a in prop_attrs.keys()}
                            for p in prop:
                                for a in prop_attrs.keys():
                                    self._dr_debug_stats[key][a].append(getattr(p, a, 0))
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            assert False, "Not implemented"  # ! TODO, temp disabled
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print(
                            "env_id",
                            env_id,
                            "extern_offset",
                            extern_offsets[env_id],
                            "vs extern_sample.shape",
                            extern_sample.shape,
                        )
                        raise Exception("Invalid extern_sample size")

        # DEBUG: print DR stats summary across all envs
        if hasattr(self, '_dr_debug_stats') and self._dr_debug_stats:
            print(f"\n[DR-DEBUG] ===== Randomization stats across {len(env_ids)} envs =====")
            for key, attrs in self._dr_debug_stats.items():
                print(f"  {key}:")
                for attr_name, vals in attrs.items():
                    import numpy as _np
                    arr = _np.array(vals)
                    print(f"    {attr_name}: min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}  std={arr.std():.6f}  n={len(arr)}")
            del self._dr_debug_stats
            print()

        self.first_randomization = False
