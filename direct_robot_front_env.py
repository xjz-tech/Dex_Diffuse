"""Front-camera environment; reuse the original robot action and cleanup methods."""

from pathlib import Path

import cv2
import numpy as np

import direct_robot_env as robot


def start_front_camera(fps, serial, resolution):
    """Open only the selected front camera, without requiring a wrist camera."""
    rs = robot._import_realsense()
    context = rs.context()
    candidates = []
    for device in context.devices:
        name = robot.get_device_info(device, rs.camera_info.name)
        device_serial = robot.get_device_info(device, rs.camera_info.serial_number)
        product_id = robot.get_device_info(device, rs.camera_info.product_id)
        if serial:
            if device_serial != serial:
                continue
        elif robot.identify_camera_label(name, product_id) != "front":
            continue
        candidates.append(robot.CameraSpec(
            label="front", serial=device_serial, name=name, product_id=product_id,
            profiles=tuple(robot.iter_color_profiles(device, fps=fps, rs=rs)),
        ))
    if not candidates:
        raise RuntimeError(
            f"Front camera not found (serial={serial or 'auto D435/D455'}). "
            "Connect a front camera or set FRONT_SERIAL / --front_serial."
        )
    if len(candidates) != 1:
        raise RuntimeError("Multiple front cameras found; set FRONT_SERIAL / --front_serial.")
    spec = candidates[0]
    if not spec.profiles:
        raise RuntimeError(f"Front camera ({spec.serial}) has no RGB/BGR profile at {fps} Hz.")
    profile = robot.select_color_profile(spec.profiles, rs, requested_resolution=resolution)
    camera = robot.RealSenseColorCamera(spec, profile, requested_resolution=resolution)
    camera.start()
    return camera


class FrontDirectRobotEnv(robot.DirectRobotEnv):
    """Use front RGB and robot state with the shared Franka/SharpA execution code.

    Initialization and observation capture are separate because DirectRobotEnv
    requires two cameras. No globals or behavior in that module are modified.
    """

    def __init__(self, args):
        if args.hz <= 0:
            raise ValueError("hz must be positive")
        self.action_dim = robot.ACTION_DIM
        self.live = bool(args.live)
        self.franka_control_mode = args.franka_control_mode
        self.sync_qpos_each_step = True
        self.max_arm_xyz_step = float(args.max_arm_xyz_step)
        self.max_hand_step = float(args.max_hand_step)
        self.disable_clamp = bool(args.disable_clamp)
        self.hand_interpolate = bool(args.hand_interpolate)
        self.action_chunk_steps = 1
        self.period = 1.0 / args.hz
        self.log_action_steps = bool(args.log_action_steps)
        self.camera_timeout_ms = args.camera_timeout_ms
        self.show_camera_input = bool(args.show_camera_input)
        self.camera_window = "Diffusion Policy front camera input"
        self.stop_on_close = bool(args.stop_on_close)
        self.use_tactile = False
        self.tactile = None
        self.obs_viz = None
        self.obs = {}
        self.reward = 0.0
        self.chunk_idx = 0
        self.previous_action = None
        self.closed = False
        self.ik_solver = None
        self.franka = None
        self.hand = None
        self.cameras = None

        # Read-only initialization; release opened resources if a later step fails.
        try:
            resolution = robot.parse_camera_resolution(args.front_resolution)
            self.cameras = start_front_camera(args.camera_fps, args.front_serial, resolution)
            self.franka = robot.FrankaArmController(
                host=args.franka_host, port=args.franka_port,
                timeout_ms=args.franka_timeout_ms,
            )
            self.hand = robot.SharpaWaveController(
                host=args.hand_host, port=args.hand_port,
                timeout_ms=args.hand_timeout_ms,
            )
            if self.franka_control_mode == "joints" and self.live:
                urdf_path = robot.resolve_fr3_urdf_path(
                    Path(args.franka_urdf) if args.franka_urdf else None
                )
                self.ik_solver = robot.FrankaJointIkSolver(
                    initial_joint_positions=self.franka.get_joint_positions(),
                    urdf_path=urdf_path,
                    max_joint_step=args.max_joint_step,
                    ik_dq_max=args.ik_dq_max,
                    damping=args.ik_damping,
                    tcp_pose_at_initial_q=self.franka.get_current_tcp_pose(),
                    sync_qpos_fn=self.franka.get_joint_positions,
                )
                print(f"[franka] control_mode=joints urdf={urdf_path}")
            if self.show_camera_input:
                try:
                    cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.camera_window, *resolution)
                except cv2.error as exc:
                    print(f"[view] camera preview disabled: {exc}")
                    self.show_camera_input = False
        except BaseException:
            for resource, method in (
                (self.cameras, "stop"), (self.hand, "close"), (self.franka, "close"),
            ):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception:
                        pass
            raise

    def _read_obs(self, *, clear=False):
        front_rgb = self.cameras.read(timeout_ms=self.camera_timeout_ms).color_rgb
        if self.show_camera_input:
            cv2.imshow(self.camera_window, cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        tcp_pose = self.franka.get_current_tcp_pose()
        hand_angles_real = self.hand.get_state()
        obs = {
            "/observe/vision/front/rgb": front_rgb,
            "/state/arm/eef_pose": robot.pose_matrix_to_arm9(tcp_pose),
            "/state/hand/joint_angle": hand_angles_real[
                robot.REAL2POLICY_DOF_INDICES
            ].astype(np.float32),
            "reward": 0.0,
        }
        if clear:
            obs["clear"] = True
        if not hasattr(self, "_printed_obs_shapes"):
            self._printed_obs_shapes = True
            print(
                f"[obs] front={front_rgb.shape} "
                f"arm={obs['/state/arm/eef_pose'].shape} "
                f"hand={obs['/state/hand/joint_angle'].shape}"
            )
        return obs

    def reset(self, reset_meta=None):
        super().reset(reset_meta=reset_meta)
        return self.obs

    def step_single(self, action, step_idx):
        self.previous_action = self._execute_action_step(
            np.asarray(action, dtype=np.float64), step_idx
        )
        self.obs = self._read_obs(clear=False)
        return self.obs
