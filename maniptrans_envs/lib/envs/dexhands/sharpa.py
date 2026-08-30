from contextlib import nullcontext
from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat

class Sharpa(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "sharpa"
        """
        right_hand_C_MC (palm)
            right_thumb_CMC_VL
                right_thumb_MC
                    right_thumb_MCP_VL
                        right_thumb_PP
                            right_thumb_DP
                                right_thumb_elastomer (fixed)
                                    right_thumb_fingertip (fixed)
            right_index_MCP_VL
                right_index_PP
                    right_index_MP
                        right_index_DP
                            right_index_elastomer (fixed)
                                right_index_fingertip (fixed)
            right_middle_MCP_VL
                right_middle_PP
                    right_middle_MP
                        right_middle_DP
                            right_middle_elastomer (fixed)
                                right_middle_fingertip (fixed)
            right_ring_MCP_VL
                right_ring_PP
                    right_ring_MP
                        right_ring_DP
                            right_ring_elastomer (fixed)
                                right_ring_fingertip (fixed)
            right_pinky_MC
                right_pinky_MCP_VL
                    right_pinky_PP
                        right_pinky_MP
                            right_pinky_DP
                                right_pinky_elastomer (fixed)
                                    right_pinky_fingertip (fixed)
        """
        self.body_names = [
            "right_hand_C_MC",       # 0  palm
            # thumb chain (5 actuated DOFs)
            "right_thumb_CMC_VL",    # 1
            "right_thumb_MC",        # 2
            "right_thumb_MCP_VL",    # 3
            "right_thumb_PP",        # 4
            "right_thumb_DP",        # 5
            "right_thumb_elastomer", # 6
            "right_thumb_fingertip", # 7
            # index chain (4 actuated DOFs)
            "right_index_MCP_VL",    # 8
            "right_index_PP",        # 9
            "right_index_MP",        # 10
            "right_index_DP",        # 11
            "right_index_elastomer", # 12
            "right_index_fingertip", # 13
            # middle chain (4 actuated DOFs)
            "right_middle_MCP_VL",   # 14
            "right_middle_PP",       # 15
            "right_middle_MP",       # 16
            "right_middle_DP",       # 17
            "right_middle_elastomer",# 18
            "right_middle_fingertip",# 19
            # ring chain (4 actuated DOFs)
            "right_ring_MCP_VL",     # 20
            "right_ring_PP",         # 21
            "right_ring_MP",         # 22
            "right_ring_DP",         # 23
            "right_ring_elastomer",  # 24
            "right_ring_fingertip",  # 25
            # pinky chain (5 actuated DOFs)
            "right_pinky_MC",        # 26
            "right_pinky_MCP_VL",    # 27
            "right_pinky_PP",        # 28
            "right_pinky_MP",        # 29
            "right_pinky_DP",        # 30
            "right_pinky_elastomer", # 31
            "right_pinky_fingertip", # 32
        ]
        # 22 actuated revolute joints (isaacgym order)
        self.dof_names = [
            'right_index_MCP_FE', 'right_index_MCP_AA', 'right_index_PIP', 'right_index_DIP', 
            'right_middle_MCP_FE', 'right_middle_MCP_AA', 'right_middle_PIP', 'right_middle_DIP', 
            'right_pinky_CMC', 'right_pinky_MCP_FE', 'right_pinky_MCP_AA', 'right_pinky_PIP', 'right_pinky_DIP', 
            'right_ring_MCP_FE', 'right_ring_MCP_AA', 'right_ring_PIP', 'right_ring_DIP', 
            'right_thumb_CMC_FE', 'right_thumb_CMC_AA', 'right_thumb_MCP_FE', 'right_thumb_MCP_AA', 'right_thumb_IP'
        ]
        self.hand2dex_mapping = {
            'wrist': ['right_hand_C_MC'],
            'thumb_proximal': ['right_thumb_CMC_VL', 'right_thumb_MC'],
            'thumb_intermediate': ['right_thumb_MCP_VL', 'right_thumb_PP'],
            'thumb_distal': ['right_thumb_DP', 'right_thumb_elastomer'],
            'thumb_tip': ['right_thumb_fingertip'],
            'index_proximal': ['right_index_MCP_VL', 'right_index_PP'],
            'index_intermediate': ['right_index_MP', 'right_index_elastomer'],
            'index_distal': ['right_index_DP'],
            'index_tip': ['right_index_fingertip'],
            'middle_proximal': ['right_middle_MCP_VL', 'right_middle_PP'],
            'middle_intermediate': ['right_middle_MP', 'right_middle_elastomer'],
            'middle_distal': ['right_middle_DP'],
            'middle_tip': ['right_middle_fingertip'],
            'ring_proximal': ['right_ring_MCP_VL', 'right_ring_PP'],
            'ring_intermediate': ['right_ring_MP', 'right_ring_elastomer'],
            'ring_distal': ['right_ring_DP'],
            'ring_tip': ['right_ring_fingertip'],
            'pinky_proximal': ['right_pinky_MC', 'right_pinky_MCP_VL', 'right_pinky_PP'],
            'pinky_intermediate': ['right_pinky_MP', 'right_pinky_elastomer'],
            'pinky_distal': ['right_pinky_DP'],
            'pinky_tip': ['right_pinky_fingertip'],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "right_thumb_elastomer",
            "right_index_elastomer",
            "right_middle_elastomer",
            "right_ring_elastomer",
            "right_pinky_elastomer",
        ]
        self.fingertip_body_names = [
            'right_thumb_fingertip',
            'right_index_fingertip',
            'right_middle_fingertip',
            'right_ring_fingertip',
            'right_pinky_fingertip',
        ]

        self.n_finger_tips = 5

        self.bone_links = [
            # thumb: palm → CMC_VL → MC → MCP_VL → PP → DP → elastomer → fingertip
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            # index: palm → MCP_VL → PP → MP → DP → elastomer → fingertip
            [0, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            # middle: palm → MCP_VL → PP → MP → DP → elastomer → fingertip
            [0, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            # ring: palm → MCP_VL → PP → MP → DP → elastomer → fingertip
            [0, 20],
            [20, 21],
            [21, 22],
            [22, 23],
            [23, 24],
            [24, 25],
            # pinky: palm → MC → MCP_VL → PP → MP → DP → elastomer → fingertip
            [0, 26],
            [26, 27],
            [27, 28],
            [28, 29],
            [29, 30],
            [30, 31],
            [31, 32],
        ]

        self.weight_idx = {
            "thumb_tip": [7],       # right_thumb_fingertip
            "index_tip": [13],      # right_index_fingertip
            "middle_tip": [19],     # right_middle_fingertip
            "ring_tip": [25],       # right_ring_fingertip
            "pinky_tip": [32],      # right_pinky_fingertip
            "level_1_joints": [1, 2, 8, 14, 20, 26],  # CMC/MCP root level
            "level_2_joints": [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29, 30],
        }

        # ##! >>> simtooreal lefthand dynamics parameters
        # # Isaacgym DOF order: index(4), middle(4), pinky(5), ring(4), thumb(5)
        # # Source: sindexhandmanip_sh.py hand_stiffnesses etc. (hand order: thumb, index, middle, ring, pinky)
        # self.dof_Kp = [
        #     # index (4): MCP_FE, MCP_AA, PIP, DIP
        #     4.76, 6.62, 0.9, 0.9,
        #     # middle (4): MCP_FE, MCP_AA, PIP, DIP
        #     4.76, 6.62, 0.9, 0.9,
        #     # pinky (5): CMC, MCP_FE, MCP_AA, PIP, DIP
        #     1.38, 4.76, 6.62, 0.9, 0.9,
        #     # ring (4): MCP_FE, MCP_AA, PIP, DIP
        #     4.76, 6.62, 0.9, 0.9,
        #     # thumb (5): CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
        #     6.95, 13.2, 4.76, 6.62, 0.9,
        # ]
        # self.dof_Kd = [
        #     # index (4)
        #     0.20859232, 0.24595532, 0.04243185, 0.03504461,
        #     # middle (4)
        #     0.2085923, 0.24595532, 0.04243185, 0.03504461,
        #     # pinky (5)
        #     0.02782345, 0.20859229, 0.24595528, 0.04243183, 0.0350446,
        #     # ring (4)
        #     0.20859226, 0.24595528, 0.04243183, 0.0350446,
        #     # thumb (5)
        #     0.28676845, 0.40845109, 0.20394083, 0.24044435, 0.04190723,
        # ]
        # self.dof_effort = [
        #     # index (4)
        #     1.864, 1.864, 0.638, 0.189,
        #     # middle (4)
        #     1.864, 1.864, 0.638, 0.189,
        #     # pinky (5)
        #     0.529, 1.864, 1.864, 0.638, 0.189,
        #     # ring (4)
        #     1.864, 1.864, 0.638, 0.189,
        #     # thumb (5)
        #     3.300, 3.300, 1.864, 1.864, 0.638,
        # ]
        # self.dof_armature = [
        #     # index (4)
        #     0.00265, 0.00265, 0.0006, 0.00042,
        #     # middle (4)
        #     0.00265, 0.00265, 0.0006, 0.00042,
        #     # pinky (5)
        #     0.00012, 0.00265, 0.00265, 0.0006, 0.00042,
        #     # ring (4)
        #     0.00265, 0.00265, 0.0006, 0.00042,
        #     # thumb (5)
        #     0.0032, 0.0032, 0.00265, 0.00265, 0.0006,
        # ]
        # self.dof_friction = [
        #     # index (4)
        #     0.07456, 0.07456, 0.01276, 0.00378738,
        #     # middle (4)
        #     0.07456, 0.07456, 0.01276, 0.00378738,
        #     # pinky (5)
        #     0.012, 0.07456, 0.07456, 0.01276, 0.00378738,
        #     # ring (4)
        #     0.07456, 0.07456, 0.01276, 0.00378738,
        #     # thumb (5)
        #     0.132, 0.132, 0.07456, 0.07456, 0.01276,
        # ]
        # ##! <<< simtooreal lefthand dynamics parameters

        ##! >>> official righthand dynamics parameters
        ## Source: right_sharpa_wave.usda, converted from per-degree to per-radian (× 180/π)
        ## Armature and friction are angle-unit-independent, copied directly.
        # Isaacgym DOF order: index(4), middle(4), pinky(5), ring(4), thumb(5)
        self.dof_Kp = [
            # index (4): MCP_FE, MCP_AA, PIP, DIP
            4.7600, 6.6217, 0.9076, 0.9041,
            # middle (4): MCP_FE, MCP_AA, PIP, DIP
            4.7600, 6.6217, 0.9076, 0.9041,
            # pinky (5): CMC, MCP_FE, MCP_AA, PIP, DIP
            1.3803, 4.7600, 6.6217, 0.9076, 0.9041,
            # ring (4): MCP_FE, MCP_AA, PIP, DIP
            4.7600, 6.6217, 0.9076, 0.9041,
            # thumb (5): CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
            6.9546, 13.2009, 4.7600, 6.6217, 0.9076,
        ]
        self.dof_Kd = [
            # index (4)
            0.183003, 0.207984, 0.039992, 0.031513,
            # middle (4)
            0.183003, 0.207984, 0.039992, 0.031513,
            # pinky (5)
            0.039248, 0.183003, 0.207984, 0.039992, 0.031513,
            # ring (4)
            0.183003, 0.207984, 0.039992, 0.031513,
            # thumb (5)
            0.240986, 0.451640, 0.183003, 0.207984, 0.039992,
        ]
        self.dof_armature = [
            # index (4)
            0.00265, 0.00265, 0.00061, 0.00042,
            # middle (4)
            0.00265, 0.00265, 0.00061, 0.00042,
            # pinky (5)
            0.00012, 0.00265, 0.00265, 0.00061, 0.00042,
            # ring (4)
            0.00265, 0.00265, 0.00061, 0.00042,
            # thumb (5)
            0.00320, 0.00320, 0.00265, 0.00265, 0.00061,
        ]
        self.dof_effort = [
            # index (4)
            1.864, 1.864, 0.638, 0.189,
            # middle (4)
            1.864, 1.864, 0.638, 0.189,
            # pinky (5)
            0.529, 1.864, 1.864, 0.638, 0.189,
            # ring (4)
            1.864, 1.864, 0.638, 0.189,
            # thumb (5)
            3.300, 3.300, 1.864, 1.864, 0.638,
        ]
        self.dof_friction = [
            # index (4)
            0.104, 0.104, 0.02476, 0.000418,
            # middle (4)
            0.104, 0.104, 0.02476, 0.000418,
            # pinky (5)
            0.013, 0.104, 0.104, 0.02476, 0.000418,
            # ring (4)
            0.104, 0.104, 0.02476, 0.000418,
            # thumb (5)
            0.132, 0.132, 0.104, 0.104, 0.02476,
        ]
        ##! <<< official righthand dynamics parameters

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        self.Kp_rot = 0.8
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 80
        self.Ki_pos = 0.005
        self.Kd_pos = 3
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("sharpa_rh")
class SharpaRH(Sharpa):
    def __init__(self):
        super().__init__()
        # self._urdf_path = "assets/sharpa_hand/right_sharpa_wave.urdf"

        self._urdf_path = "assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0., 0., 0.]))

        # TODO: calibrate init_transformation for sharpa hand
        self.init_transformation = np.eye(4)

    def __str__(self):
        return super().__str__() + "_rh"
