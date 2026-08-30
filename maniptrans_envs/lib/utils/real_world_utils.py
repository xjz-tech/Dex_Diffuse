import numpy as np
import isaacgym
import torch
import math
from torch.nn.utils import clip_grad_norm_

import logging

from termcolor import cprint
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

try:
    from dex_retargeting.imitator_dexretargeting import ImitatorDexRetargeting
except ModuleNotFoundError:
    logging.error("Failed to import ImitatorDexRetargeting. \n")

from main.dataset.opt_utils import WujiHandModel, LeapHandModel, ObjectModel, EnergyFunction, extract_obj_mesh_path
from main.dataset.opt_utils import OptLogger

from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    # rotmat_to_rot6d,
)

class Retargetor:
    
    def __init__(
        self,
        hand_name,
        side,
        obj_urdf_path,
        only_fingertips,
        enable_post_opt,
        post_opt_weight_dict=None,
        post_opt_iters=100,
        post_opt_dist_surface_thres=0.04,
        post_opt_dist_surface_ratio=1.0,
        device="cuda:0",
        object_scale=1.0,
    ):
        self.device = device
        self.obj_urdf_path = obj_urdf_path
        self.object_scale = object_scale

        ## Dexhand Config
        self.hand_name = hand_name
        self.side = side
        self.dexhand = DexHandFactory.create_hand(hand_name, side)

        ## Post-opt Config
        self.only_fingertips = only_fingertips
        self.enable_post_opt = enable_post_opt
        # self.post_opt_weight_dict = post_opt_weight_dict
        self.post_opt_weight_dict = {'pen': 10.0, 'dist_surface': 5.0, 'smooth': 2.0, 'self_collision': 10.0}
        self.post_opt_iters = post_opt_iters
        self.post_opt_dist_surface_thres = post_opt_dist_surface_thres
        self.post_opt_dist_surface_ratio = post_opt_dist_surface_ratio

        self._init_dof_mapping()
        self._init_dex_retargeting()
        self._init_opt_tools()

    def retarget(
        self,
        mocap_joints: np.ndarray,
        object_pose: np.ndarray,
        return_tensor: bool = False,
    ):
        ## Assume wrist at world frame
        wrist_pos = np.zeros(3, dtype=np.float32)
        wrist_rot = np.eye(3, dtype=np.float32)

        ## Convert mocap joints to mano 21 joints
        target_mano_joints21 = self._mocap2joints21(mocap_joints)

        ## assemble joints21
        target_mano_joints21_reindex = self._assemble_joints21(
            wrist_pos,
            target_mano_joints21,
        )
        assert target_mano_joints21_reindex is not None, "target_mano_joints21_reindex is None"

        ## Dex-retargeting
        for _ in range(200):
            target_dof_pos = self.dex_retargetor(target_mano_joints21_reindex)
        target_dof_pos = torch.from_numpy(target_dof_pos).to(self.device, dtype=torch.float32).unsqueeze(0)
        # TODO: clamp target_dof_pos to joint limits

        ## Post-opt
        if self.enable_post_opt:
            ## TODO: check rotation format
            wrist_pos = torch.from_numpy(wrist_pos).to(self.device, dtype=torch.float32).unsqueeze(0)
            wrist_rot = torch.from_numpy(wrist_rot).to(self.device, dtype=torch.float32).unsqueeze(0)
            object_pose = torch.from_numpy(object_pose).to(self.device, dtype=torch.float32).unsqueeze(0)
            target_mano_joints21_reindex = torch.from_numpy(target_mano_joints21_reindex).to(self.device, dtype=torch.float32).unsqueeze(0)
            target_dof_pos = self._post_opt(
                object_pose=object_pose,
                wrist_pos=wrist_pos,
                wrist_rot=wrist_rot,
                dof_pos=target_dof_pos,
                target_mano_joints21=target_mano_joints21_reindex,
            )

        ## Data Type Conversion
        if return_tensor:
            return target_dof_pos
        else:
            return target_dof_pos.detach().cpu().numpy()

    def _post_opt(
        self, 
        object_pose: torch.Tensor,
        wrist_pos: torch.Tensor,
        wrist_rot: torch.Tensor,
        dof_pos: torch.Tensor,
        target_mano_joints21: torch.Tensor,
        ) -> torch.Tensor:

        post_opt_dof_pos = dof_pos.clone().detach().requires_grad_(True)
        post_opt_wrist_pos = wrist_pos.clone().detach().requires_grad_(False)
        post_opt_wrist_rot = wrist_rot.clone().detach().requires_grad_(False)
        object_pose = object_pose.clone().detach().requires_grad_(False)

        try:
            self.opt_logger.update_meta(
                hand=dict(
                    asset_root="maniptrans_envs/assets/wuji_hand",
                    asset_file="right-fixcol.urdf",
                    link_mesh_dir="maniptrans_envs/assets/wuji_hand",
                    contact_path="maniptrans_envs/assets/wuji_hand/config/contact_wuji_right.json",
                    pts_density=10,
                    q_lower_limits=[-math.pi] * int(self.dexhand.n_dofs),
                    q_upper_limits=[math.pi] * int(self.dexhand.n_dofs),
                    visual_as_collision=1,
                ),
                object=dict(
                    mesh_path=self.object_model.object_mesh_path,
                    urdf_path=self.obj_urdf_path,
                ),
                post_opt=dict(
                    weight_dict=self.energy_function.weight_dict,
                    contact_num=self.energy_function.contact_num,
                    dist_surface_thres=float(self.post_opt_dist_surface_thres),
                    dist_surface_ratio=float(self.post_opt_dist_surface_ratio),
                ),
            )
        except Exception:
            pass

        post_optimizer = torch.optim.Adam(
            [
                {"params": [post_opt_dof_pos], "lr": 0.003},
            ]
        )
        post_grad_clip_norm = 1.0
        # contact_indices = self.energy_function.get_contact_indices(obj_trajectory, post_opt_wrist_pos, post_opt_wrist_rot, post_opt_dof_pos[:, self.isaac2chain_order])
        ## TODO: transfer post_opt_wrist_rot to 6D representation
        post_opt_wrist_rot = rotmat_to_rot6d(post_opt_wrist_rot)
        self.energy_function._update_state(object_pose, post_opt_wrist_pos, post_opt_wrist_rot, post_opt_dof_pos)
        self.energy_function.select_surface_point_mask_global(thres=self.post_opt_dist_surface_thres, ratio=self.post_opt_dist_surface_ratio)

        for post_opt_iter in range(self.post_opt_iters):
            ## TODO: clamp dof_pos to joint limits
            post_optimizer.zero_grad()
            ## TODO: check joint order
            energy_dict = self.energy_function.compute_energy_with_breakdown(
                object_pose, post_opt_wrist_pos, post_opt_wrist_rot, post_opt_dof_pos
            )
            loss = energy_dict['total']
            # Ensure loss is a scalar or has at least one element with gradient
            if loss.numel() == 0:
                raise RuntimeError("loss is empty, cannot compute gradients")
            # Sum to get scalar loss for backward
            loss_sum = loss.sum()
            
            loss_sum.backward()
            clip_grad_norm_([post_opt_dof_pos], post_grad_clip_norm)
            post_optimizer.step()

            # Get joints21 for current frame (all frames share the same joints21 in this implementation)
            # Since we're iterating over post_opt_iters, we use the joints21 from the first frame
            # In practice, joints21_tensor is [num_envs, 21, 3], we need to select the frame
            # For now, we'll use all frames' joints21 (one per batch)
            self.opt_logger.add_log(
                energy_dict=energy_dict,
                wrist_pos=post_opt_wrist_pos,
                wrist_rot_6d=post_opt_wrist_rot,
                hand_q=post_opt_dof_pos,
                object_pose=object_pose,
                joints21=target_mano_joints21,  # [num_envs, 21, 3]
            )

            if post_opt_iter % 10 == 0:
                cprint(f"post-opt {post_opt_iter} {loss.sum().item()}", "green")

        ## overwrite opt_dof_pos_clamped with the final post-opt dof pos
        opt_dof_pos = post_opt_dof_pos.clone().detach()

        # try:
        #     self.opt_logger.vis_log(self.hand_model, self.object_model, batch_idx=0, port=6007)
        #     input("Press Enter to continue...")
        # except Exception as e:
        #     cprint(f"Opt visualization failed: {str(e)}", "red")

        # 保存日志（如提供路径）
        # try:
        #     if getattr(self.args, "log_path", None):
        #         os.makedirs(os.path.dirname(self.args.log_path), exist_ok=True)
        #         self.opt_logger.save(self.args.log_path)
        #         cprint(f"Log saved to {self.args.log_path}", "green")
        # except Exception as e:
        #     cprint(f"Save log failed: {str(e)}", "red")
        return opt_dof_pos

    def _assemble_joints21(
        self, 
        wrist_pos, 
        mano_joints_frame
    ):
        """Assemble MANO joints into 21-joint format for ImitatorDexRetargeting."""
        if wrist_pos is None or mano_joints_frame is None:
            print(f"wrist_pos: {wrist_pos}, mano_joints_frame: {mano_joints_frame}")
            return None
        if mano_joints_frame.shape[0] < len(self._mano_name_to_tensor_idx):
            print(f"mano_joints_frame.shape[0]: {mano_joints_frame.shape[0]}, len(self._mano_name_to_tensor_idx): {len(self._mano_name_to_tensor_idx)}")
            return None
        joints21 = np.zeros((21, 3), dtype=np.float32)
        joints21[0] = wrist_pos.astype(np.float32)
        for name in self._mediapipe_names[1:]:
            idx21 = self._mediapipe_name_to_idx[name]
            mano_idx = self._mano_name_to_tensor_idx.get(name)
            print(f"name: {name}, idx21: {idx21}, mano_idx: {mano_idx}")
            if mano_idx is None:
                print(f"name: {name}, mano_idx: {mano_idx}")
                # return None
                continue
            joints21[idx21] = mano_joints_frame[mano_idx].astype(np.float32)
        if not np.all(np.isfinite(joints21)):
            return None

        ## mimic that: only has fingertips joints
        if self.only_fingertips:
            for name in self._mediapipe_names[1:]:
                if "_tip" not in name:
                    joints21[self._mediapipe_name_to_idx[name]] = 0.0
            
        return joints21

    def _mocap2joints21(self, mocap_joints):
        """Convert mocap joints to mano 21 joints."""
        ## TODO: check nokov2 dataset
        mano_joints = {k: None for k in self._mano_joint2index.keys()}

        mano_joints['thumb_tip'] = mocap_joints[0, :]
        mano_joints['index_tip'] = mocap_joints[1, :]
        mano_joints['middle_tip'] = mocap_joints[2, :]
        mano_joints['ring_tip'] = mocap_joints[3, :]
        mano_joints['pinky_tip'] = mocap_joints[4, :]
        for k in self._mano_joint2index.keys():
            if 'tip' not in k:
                mano_joints[k] = np.array([0., 0., 0.])

        mano_joints = pack_mano_joints(mano_joints, self.dexhand)

        return mano_joints

    def _init_dof_mapping(self, ):
        self._mano_joint2index = {
            "thumb_proximal": 1,
            "thumb_intermediate": 2,
            "thumb_distal": 3,
            "thumb_tip": 4,
            "index_proximal": 5,
            "index_intermediate": 6,
            "index_distal": 7,
            "index_tip": 8,
            "middle_proximal": 9,
            "middle_intermediate": 10,
            "middle_distal": 11,
            "middle_tip": 12,
            "ring_proximal": 13,
            "ring_intermediate": 14,
            "ring_distal": 15,
            "ring_tip": 16,
            "pinky_proximal": 17,
            "pinky_intermediate": 18,
            "pinky_distal": 19,
            "pinky_tip": 20,
        }

        self._mediapipe_names = [
            "wrist",
            "thumb_proximal",
            "thumb_intermediate",
            "thumb_distal",
            "thumb_tip",
            "index_proximal",
            "index_intermediate",
            "index_distal",
            "index_tip",
            "middle_proximal",
            "middle_intermediate",
            "middle_distal",
            "middle_tip",
            "ring_proximal",
            "ring_intermediate",
            "ring_distal",
            "ring_tip",
            "pinky_proximal",
            "pinky_intermediate",
            "pinky_distal",
            "pinky_tip",
        ]
        self._mediapipe_name_to_idx = {
            name: idx for idx, name in enumerate(self._mediapipe_names)
        }
        self._mano_name_sequence = []
        for body_name in self.dexhand.body_names:
            hand_name = self.dexhand.to_hand(body_name)[0]
            if hand_name == "wrist":
                continue
            self._mano_name_sequence.append(hand_name)
        self._mano_name_to_tensor_idx = {}
        for idx, name in enumerate(self._mano_name_sequence):
            if name not in self._mano_name_to_tensor_idx:
                self._mano_name_to_tensor_idx[name] = idx

    def _init_dex_retargeting(self,):
        hand_name = self.hand_name
        if self.only_fingertips and 'wuji' in hand_name:
            hand_name = self.hand_name + '-onlytips'

        urdf_dict = {
            "wuji": "maniptrans_envs/assets/wuji_hand/right-fixcol.urdf",
            "wuji-onlytips": "maniptrans_envs/assets/wuji_hand/right-fixcol.urdf",
            "leap": "maniptrans_envs/assets/leap_hand/leap_hand_right_modified-newfingertip.urdf"
        }

        urdf_path = urdf_dict.get(hand_name, None)
        assert urdf_path is not None, f"urdf_path is not found for hand_name: {hand_name}"

        self.dex_retargetor = ImitatorDexRetargeting(
            hand_name=hand_name, 
            side=self.side, 
            urdf_path=urdf_path
        )

    def _init_opt_tools(self,):
        hand_model_dict = {
            "wuji": WujiHandModel,
            "leap": LeapHandModel,
        }

        hand_model_kwargs = {
            "wuji": {
                "asset_root": "maniptrans_envs/assets/wuji_hand",
                "asset_file": "right-fixcol.urdf",
                "link_mesh_dir": "maniptrans_envs/assets/wuji_hand",
                "contact_path": "maniptrans_envs/assets/wuji_hand/config/contact_wuji_right.json",
                "pts_density": 10,
                "q_lower_limits": [-np.pi] * self.dexhand.n_dofs,
                "q_upper_limits": [np.pi] * self.dexhand.n_dofs,
            },
            "leap": {
                "asset_root": "maniptrans_envs/assets/leap_hand",
                "asset_file": "leap_hand_right_modified-newfingertip.urdf",
                "link_mesh_dir": "maniptrans_envs/assets/leap_hand",
                "contact_path": "maniptrans_envs/assets/wuji_hand/config/contact_wuji_right.json", ## deprecated
                "pts_density": 10,
                "q_lower_limits": [-np.pi] * self.dexhand.n_dofs,
                "q_upper_limits": [np.pi] * self.dexhand.n_dofs,
            },
        }

        self.hand_model = hand_model_dict[self.dexhand.name](
            **hand_model_kwargs[self.dexhand.name],
            device="cuda",
            batch_size=1,
            visual_as_collision=True,
        )

        self.object_model = ObjectModel(
            object_mesh_path=extract_obj_mesh_path(self.obj_urdf_path),
            object_poses=None,
            object_scale=self.object_scale,
            device=self.device,
        )

        self.energy_function = EnergyFunction(
            object_model=self.object_model,
            hand_model=self.hand_model,
            weight_dict=self.post_opt_weight_dict,
            # weight_dict={
            #     # "dist": 1.0,
            #     "pen": 20.0,
            #     "dist_surface": 1.0,
            # },
            contact_num=12,
        )

        self.opt_logger = OptLogger() # For visualization & saving

def rotmat_to_rot6d(rotmat):
    batch_dim = rotmat.size()[:-2]
    return rotmat[..., :2, :].clone().reshape(batch_dim + (6,))

def pack_mano_joints(data, dexhand):
    mano_joints = []
    mano_joints = np.stack(
        [
            data[dexhand.to_hand(j_name)[0]]
            for j_name in dexhand.body_names
            if dexhand.to_hand(j_name)[0] != "wrist"
        ],
        axis=0,
    )

    return mano_joints

if __name__ == "__main__":
    retargetor = Retargetor(
        hand_name="wuji",
        side="right",
        obj_urdf_path="data/NOKOV-v2/object/cylinder_35x100mm.urdf",
        only_fingertips=False,
        enable_post_opt=True,
        post_opt_weight_dict={},
        post_opt_iters=100,
        post_opt_dist_surface_thres=0.04,
        post_opt_dist_surface_ratio=1.0,
    )
    mocap_joints = np.ones((5, 3)).astype(np.float32)
    mocap_joints[0, :] = np.array([-0.06521768, -0.11836934,  0.06553661])
    mocap_joints[1, :] = np.array([-0.13007844, -0.08938991,  0.04774203])
    mocap_joints[2, :] = np.array([-0.13519621, -0.09621898,  0.01629778])
    mocap_joints[3, :] = np.array([-0.1149085 , -0.11635342, -0.01873568])
    mocap_joints[4, :] = np.array([-0.07777146, -0.12773795, -0.02478589])
    object_pose = np.eye(4)
    object_pose = np.array(
        [[ 0.96056824, -0.01925298,  0.27737697, -0.0948799 ],
        [-0.27627351, -0.17850204,  0.94435691, -0.14121607],
        [ 0.03133067, -0.98375116, -0.17678248,  0.02831816],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
    target_dof_pos = retargetor.retarget(mocap_joints, object_pose)
    print(target_dof_pos)

# ipdb> mano_joints['thumb_tip'][0]
# array([-0.06521768, -0.11836934,  0.06553661])
# ipdb> mano_joints['index_tip'][0]
# array([-0.13007844, -0.08938991,  0.04774203])
# ipdb> mano_joints['middle_tip'][0]
# array([-0.13519621, -0.09621898,  0.01629778])
# ipdb> mano_joints['ring_tip'][0]
# array([-0.1149085 , -0.11635342, -0.01873568])
# ipdb> mano_joints['pinky_tip'][0]
# array([-0.07777146, -0.12773795, -0.02478589])
# ipdb> obj_trajectory[0]
# array([[ 0.96056824, -0.01925298,  0.27737697, -0.0948799 ],
#        [-0.27627351, -0.17850204,  0.94435691, -0.14121607],
#        [ 0.03133067, -0.98375116, -0.17678248,  0.02831816],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])