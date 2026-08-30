from contextlib import nullcontext
from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat

class Wuji(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "wuji"
        self.body_names = [
            'palm_link', 
            'finger1_link1', 
            'finger1_link2', 
            'finger1_link3', 
            'finger1_link4', 
            'finger1_tip_link', 
            'finger2_link1', 
            'finger2_link2', 
            'finger2_link3', 
            'finger2_link4', 
            'finger2_tip_link', 
            'finger3_link1', 
            'finger3_link2', 
            'finger3_link3', 
            'finger3_link4', 
            'finger3_tip_link', 
            'finger4_link1', 
            'finger4_link2', 
            'finger4_link3', 
            'finger4_link4', 
            'finger4_tip_link', 
            'finger5_link1', 
            'finger5_link2', 
            'finger5_link3', 
            'finger5_link4', 
            'finger5_tip_link', 
        ]
        self.dof_names = [
            'finger1_joint1', 'finger1_joint2', 'finger1_joint3', 'finger1_joint4', 
            'finger2_joint1', 'finger2_joint2', 'finger2_joint3', 'finger2_joint4', 
            'finger3_joint1', 'finger3_joint2', 'finger3_joint3', 'finger3_joint4', 
            'finger4_joint1', 'finger4_joint2', 'finger4_joint3', 'finger4_joint4', 
            'finger5_joint1', 'finger5_joint2', 'finger5_joint3', 'finger5_joint4',
        ]
        self.hand2dex_mapping = {
            'wrist': ['palm_link'],
            'thumb_proximal': ['finger1_link1', 'finger1_link2'],
            'thumb_intermediate': ['finger1_link3'],
            'thumb_distal': ['finger1_link4'],
            'thumb_tip': ['finger1_tip_link'],
            'index_proximal': ['finger2_link1', 'finger2_link2'],
            'index_intermediate': ['finger2_link3'],
            'index_distal': ['finger2_link4'],
            'index_tip': ['finger2_tip_link'],
            'middle_proximal': ['finger3_link1', 'finger3_link2'],
            'middle_intermediate': ['finger3_link3'],
            'middle_distal': ['finger3_link4'],
            'middle_tip': ['finger3_tip_link'],
            'ring_proximal': ['finger4_link1', 'finger4_link2'],
            'ring_intermediate': ['finger4_link3'],
            'ring_distal': ['finger4_link4'],
            'ring_tip': ['finger4_tip_link'],
            'pinky_proximal': ['finger5_link1', 'finger5_link2'],
            'pinky_intermediate': ['finger5_link3'],
            'pinky_distal': ['finger5_link4'],
            'pinky_tip': ['finger5_tip_link'],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            'finger1_tip_link',
            'finger2_tip_link',
            'finger3_tip_link',
            'finger4_tip_link',
            'finger5_tip_link',
        ]
        self.fingertip_body_names = [
            'finger1_tip_link',
            'finger2_tip_link',
            'finger3_tip_link',
            'finger4_tip_link',
            'finger5_tip_link',
        ]
        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 11],
            [0, 16],
            [0, 21],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
            [21, 22],
            [22, 23],
            [23, 24],
            [24, 25],
        ]
        self.weight_idx = {
            "thumb_tip": [5],      # fingertip_thumb
            "index_tip": [10],     # fingertip_index
            "middle_tip": [15],    # fingertip_middle
            "ring_tip": [20],      # fingertip_ring
            "pinky_tip": [25],     # fingertip_pinky
            "level_1_joints": [1, 2, 6, 11, 16, 21],      # MCP关节级别
            "level_2_joints": [3, 4, 7, 8, 12, 13, 17, 18, 22, 23],  # 其他关节
        }
        self.n_finger_tips = 5
        # !! real dexhand dof kp-kd
        self.dof_Kp = [
            12.90, 6.39, 5.78, 5.10,
            11.85, 6.85, 11.85, 11.85,
            11.85, 6.85, 11.85, 11.85,
            11.85, 6.85, 11.85, 11.85,
            11.85, 6.85, 11.85, 11.85,
        ]
        # self.dof_Kp = [kp * 5 for kp in self.dof_Kp]
        self.dof_Kd = [
            0.81, 0.11, 0.11, 0.09,
            0.72, 0.69, 0.72, 0.72,
            0.72, 0.69, 0.72, 0.72,
            0.72, 0.69, 0.72, 0.72,
            0.72, 0.69, 0.72, 0.72,
        ]
        self.dof_armature = 0.0
        # self.dof_effort = 10

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.8
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 80
        self.Ki_pos = 0.005
        self.Kd_pos = 3
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name

@register_dexhand("wuji_rh")
class WujiRH(Wuji):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/wuji_hand/right-fixcol_fixlim.urdf"
        # self._urdf_path = "assets/wuji_hand/right-fixcol.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0., 0., 0.]))
        # self.relative_translation = np.array([0.0, 0.0, 0.0])

    def __str__(self):
        return super().__str__() + "_rh"

@register_dexhand("wuji_lh")
class WujiLH(Wuji):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        raise NotImplementedError("WujiLH is not implemented")
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, 0, 0]))

    def __str__(self):
        return super().__str__() + "_lh"