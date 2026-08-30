from contextlib import nullcontext

from pyparsing import C
from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat

class Leap(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "leap"
        """
        palm_lower
 	 	mcp_joint
 	 	 	pip
 	 	 	 	dip
 	 	 	 	 	fingertip
 	 	 	 	 	 	index_tip_head
 	 	mcp_joint_2
 	 	 	pip_2
 	 	 	 	dip_2
 	 	 	 	 	fingertip_2
 	 	 	 	 	 	middle_tip_head
 	 	mcp_joint_3
 	 	 	pip_3
 	 	 	 	dip_3
 	 	 	 	 	fingertip_3
 	 	 	 	 	 	ring_tip_head
 	 	thumb_temp_base
 	 	 	thumb_pip
 	 	 	 	thumb_dip
 	 	 	 	 	thumb_fingertip
 	 	 	 	 	 	thumb_tip_head
        """
        self.body_names = [
            "palm_lower",
            "mcp_joint", # 1
            "pip", # 2
            "dip", # 3
            "fingertip", # 4
            "index_tip_head", # 5
            "mcp_joint_2", # 6
            "pip_2", # 7
            "dip_2", # 8
            "fingertip_2", # 9
            "middle_tip_head", # 10
            "mcp_joint_3", # 11
            "pip_3", # 12
            "dip_3", # 13
            "fingertip_3", # 14
            "ring_tip_head", # 15
            "thumb_temp_base", # 16
            "thumb_pip", # 17
            "thumb_dip", # 18
            "thumb_fingertip", # 19
            "thumb_tip_head", # 20
        ]
        self.dof_names = [
            '1', '0', '2', '3', 
            '12', '13', '14', '15', 
            '5', '4', '6', '7', 
            '9', '8', '10', '11'
        ]
        self.hand2dex_mapping = {
            'wrist': ['palm_lower'],
            'thumb_proximal': ['thumb_temp_base', 'thumb_pip'],
            'thumb_intermediate': ['thumb_dip'],
            'thumb_distal': ['thumb_fingertip'],
            'thumb_tip': ['thumb_tip_head'],
            'index_proximal': ['mcp_joint', 'pip'],
            'index_intermediate': ['dip'],
            'index_distal': ['fingertip'],
            'index_tip': ['index_tip_head'],
            'middle_proximal': ['mcp_joint_2', 'pip_2'],
            'middle_intermediate': ['dip_2'],
            'middle_distal': ['fingertip_2'],
            'middle_tip': ['middle_tip_head'],
            'pinky_proximal': ['mcp_joint_3', 'pip_3'],
            'pinky_intermediate': ['dip_3'],
            'pinky_distal': ['fingertip_3'],
            'pinky_tip': ['ring_tip_head'],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            # 'thumb_tip_head',
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
            # 'index_tip_head',
            # 'middle_tip_head',
            # 'ring_tip_head',
        ]
        self.fingertip_body_names = [
            'thumb_tip_head',
            'index_tip_head',
            'middle_tip_head',
            'ring_tip_head',
        ]

        self.n_finger_tips = 4

        self.bone_links = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [0, 6],
            [6, 7], 
            [7, 8],
            [8, 9],
            [9, 10],
            [0, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [0, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            [19, 20],
        ] # TODO: add the bone links

        self.weight_idx = {
            "thumb_tip": [20],      # fingertip_thumb
            "index_tip": [5],     # fingertip_index
            "middle_tip": [10],    # fingertip_middle
            "pinky_tip": [15],     # ring_tip_head
            "level_1_joints": [1,6,11,16],      # MCP关节级别
            "level_2_joints": [2,3,7,8,12,13,17,18],  # 其他关节
        } 

        # !! real dexhand dof kp-kd
        # self.dof_Kp = [
        #     7.758, 0.276, 0.263, 0.245,
        #     0.491, 0.226, 1.538, 0.269,
        #     0.246, 0.270, 0.258, 0.223,
        #     0.267, 1.983, 0.245, 0.248
        # ]
        # self.dof_Kd = [
        #     0.31, 0.01, 0.01, 0.01,
        #     0.01, 0.01, 0.01, 0.01,
        #     0.01, 0.01, 0.01, 0.01,
        #     0.01, 0.01, 0.01, 0.01,
        # ]
        # self.dof_friction = [
        #     1.76, 0.01, 0.01, 0.01,
        #     0.01, 0.01, 9.94, 0.01,
        #     0.01, 0.01, 0.01, 7.66,
        #     0.01, 10.0, 0.01, 3.93,
        # ]
        # self.dof_effort = [
        #     0.50, 1.42, 6.95, 7.66,
        #     0.50, 5.39, 0.95, 5.24,
        #     7.99, 0.50, 6.16, 4.75,
        #     10.0, 0.50, 6.90, 6.68
        # ]
        
        ## 0118 ssid
        # self.dof_Kp = [
        #     4.264, 4.279, 2.595, 2.389,
        #     5.0, 1.771, 1.817, 2.486,
        #     4.264, 4.279, 2.595, 2.389,
        #     4.264, 4.279, 2.595, 2.389,
        # ]
        # self.dof_Kd = [
        #     0.19, 0.19, 0.1, 0.1,
        #     0.194, 0.1, 0.1, 0.1,
        #     0.19, 0.19, 0.1, 0.1,
        #     0.19, 0.19, 0.1, 0.1,
        # ]
        # self.dof_effort = [0.95] * len(self.dof_names)

        # test_scale = 1.0
        # self.dof_Kp = [kp * test_scale for kp in self.dof_Kp]
        # self.dof_Kd = [kd * test_scale for kd in self.dof_Kd]

        ## 0211 ssid
        self.dof_Kp = [
            5.656, 1.000, 6.000, 1.525,
            2.112, 6.000, 5.496, 1.000,
            5.656, 1.000, 6.000, 1.525,
            5.656, 1.000, 6.000, 1.525,
        ]
        self.dof_Kd = [
            0.747, 0.201, 0.538, 0.287,
            0.363, 0.761, 0.769, 0.163,
            0.747, 0.201, 0.538, 0.287,
            0.747, 0.201, 0.538, 0.287,
        ]
        self.dof_effort = [
            0.210, 0.218, 0.180, 0.218,
            0.184, 0.180, 0.180, 0.216,
            0.210, 0.218, 0.180, 0.218,
            0.210, 0.218, 0.180, 0.218,
        ]

        # self.dof_Kp = [
        #     2.388, 2.239, 2.595, 2.393,
        #     5.000, 1.747, 1.817, 2.487,
        #     1.472, 2.108, 2.471, 2.498,
        #     1.709, 2.451, 2.244, 2.588,
        # ]
        # self.dof_Kd = [
        #     0.100, 0.100, 0.100, 0.100,
        #     0.194, 0.100, 0.100, 0.100,
        #     0.100, 0.100, 0.100, 0.100,
        #     0.100, 0.100, 0.100, 0.100,
        # ]

        # self.dof_Kp = [2.5] * len(self.dof_names)
        # self.dof_Kd = [0.1] * len(self.dof_names)
        # self.dof_effort = [
        #     0.95
        # ] * len(self.dof_names)

        # self.dof_Kp = [1.0] * len(self.dof_names)
        # self.dof_Kd = [0.1] * len(self.dof_names)
        # self.dof_effort = [
        #     0.95
        # ] * len(self.dof_names)

        self.dof_armature = None

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


@register_dexhand("leap_rh")
class LeapRH(Leap):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/leap_hand/leap_hand_right_cytip4cylinder.urdf"

        # self._urdf_path = "assets/leap_hand/leap_hand_right_modified-newfingertip_0127tri.urdf" ## TODO: change task
        # self._urdf_path = "assets/leap_hand/leap_hand_right_modified-newfingertip.urdf"
        # self._urdf_path = "assets/leap_hand/leap_hand_right_modified_origin.urdf" ## modified by 0120
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0., 0., 0.]))

        self.init_transformation = np.array(
            [[-0.21984620,  0.00000000, -0.97553454, -0.02200000],
             [-0.97553454,  0.00000000,  0.21984620,  0.00000000],
             [ 0.00000000,  1.00000000,  0.00000000,  0.00000000],
             [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]]
        )


    def __str__(self):
        return super().__str__() + "_rh"
