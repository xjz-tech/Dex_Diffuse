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
            'base_link', 
            'CMC_joint', 
            'CMC_2rd', 
            'MCP_thumb', 
            'IP_thumb', 
            'fingertip_thumb', 
            'MCP_joint_index', 
            'PIPMCP_2rd_index', 
            'DIP_index', 
            'tipmount_index', 
            'fingertip_index', 
            'MCP_joint_middle', 
            'PIPMCP_2rd_middle', 
            'DIP_middle', 
            'tipmount_middle', 
            'fingertip_middle', 
            'MCP_joint_ring', 
            'PIPMCP_2rd_ring', 
            'DIP_ring', 
            'tipmount_ring', 
            'fingertip_ring', 
            'MCP_joint_pinky', 
            'PIPMCP_2rd_pinky', 
            'DIP_pinky', 
            'tipmount_pinky', 
            'fingertip_pinky'
        ]
        self.dof_names = [
            'F1J1', 'F1J2', 'F1J3', 'F1J4', 
            'F2J1', 'F2J2', 'F2J3', 'F2J4', 
            'F3J1', 'F3J2', 'F3J3', 'F3J4', 
            'F4J1', 'F4J2', 'F4J3', 'F4J4', 
            'F5J1', 'F5J2', 'F5J3', 'F5J4',
        ]
        self.hand2dex_mapping = {
            'wrist': ['base_link'],
            'thumb_proximal': ['CMC_joint', 'CMC_2rd'],
            'thumb_intermediate': ['MCP_thumb'],
            'thumb_distal': ['IP_thumb'],
            'thumb_tip': ['fingertip_thumb'],
            'index_proximal': ['MCP_joint_index', 'PIPMCP_2rd_index'],
            'index_intermediate': ['DIP_index'],
            'index_distal': ['tipmount_index'],
            'index_tip': ['fingertip_index'],
            'middle_proximal': ['MCP_joint_middle', 'PIPMCP_2rd_middle'],
            'middle_intermediate': ['DIP_middle'],
            'middle_distal': ['tipmount_middle'],
            'middle_tip': ['fingertip_middle'],
            'ring_proximal': ['MCP_joint_ring', 'PIPMCP_2rd_ring'],
            'ring_intermediate': ['DIP_ring'],
            'ring_distal': ['tipmount_ring'],
            'ring_tip': ['fingertip_ring'],
            'pinky_proximal': ['MCP_joint_pinky', 'PIPMCP_2rd_pinky'],
            'pinky_intermediate': ['DIP_pinky'],
            'pinky_distal': ['tipmount_pinky'],
            'pinky_tip': ['fingertip_pinky']
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            'fingertip_thumb',
            'fingertip_index',
            'fingertip_middle',
            'fingertip_ring',
            'fingertip_pinky',
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

@register_dexhand("legacy-wuji_rh")
class WujiRH(Wuji):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/wuji_hand/d5_right_hands_tips-fixcol.urdf"
        # self._urdf_path = "assets/wuji_hand/d5_right_hands_tips.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0., 0., 0.]))
        # self.relative_translation = np.array([0.0, 0.0, 0.0])

    def __str__(self):
        return super().__str__() + "_rh"

@register_dexhand("legacy-wuji_lh")
class WujiLH(Wuji):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        raise NotImplementedError("WujiLH is not implemented")
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, 0, 0]))

    def __str__(self):
        return super().__str__() + "_lh"