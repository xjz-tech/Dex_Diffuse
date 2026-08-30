from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class Xhand(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "xhand"
        self.self_collision = False
        self.body_names = [
            "hand_link",
            "hand_index_bend_link",
            "hand_index_rota_link1",
            "hand_index_rota_link2",
            "hand_index_rota_tip",
            "hand_mid_link1",
            "hand_mid_link2",
            "hand_mid_tip",
            "hand_pinky_link1",
            "hand_pinky_link2",
            "hand_pinky_tip",
            "hand_ring_link1",
            "hand_ring_link2",
            "hand_ring_tip",
            "hand_thumb_bend_link",
            "hand_thumb_rota_link1",
            "hand_thumb_rota_link2",
            "hand_thumb_rota_tip",
        ]
        self.dof_names = [
            "hand_index_bend_joint",
            "hand_index_joint1",
            "hand_index_joint2",
            "hand_mid_joint1",
            "hand_mid_joint2",
            "hand_pinky_joint1",
            "hand_pinky_joint2",
            "hand_ring_joint1",
            "hand_ring_joint2",
            "hand_thumb_bend_joint",
            "hand_thumb_rota_joint1",
            "hand_thumb_rota_joint2",
        ]
        self.hand2dex_mapping = {
            "wrist": ["hand_link"],  # ! MUST only have one element
            "thumb_proximal": ["hand_thumb_bend_link", "hand_thumb_rota_link1"],
            "thumb_intermediate": ["hand_thumb_rota_link2"],
            "thumb_distal": [],
            "thumb_tip": ["hand_thumb_rota_tip"],
            "index_proximal": ["hand_index_bend_link", "hand_index_rota_link1"],
            "index_intermediate": ["hand_index_rota_link2"],
            "index_distal": [],  # missing
            "index_tip": ["hand_index_rota_tip"],
            "middle_proximal": ["hand_mid_link1"],
            "middle_intermediate": ["hand_mid_link2"],
            "middle_distal": [],  # missing
            "middle_tip": ["hand_mid_tip"],
            "ring_proximal": ["hand_ring_link1"],
            "ring_intermediate": ["hand_ring_link2"],
            "ring_distal": [],  # missing
            "ring_tip": ["hand_ring_tip"],
            "pinky_proximal": ["hand_pinky_link1"],
            "pinky_intermediate": ["hand_pinky_link2"],
            "pinky_distal": [],  # missing
            "pinky_tip": ["hand_pinky_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "hand_thumb_rota_link2",
            "hand_index_rota_link2",
            "hand_mid_link2",
            "hand_ring_link2",
            "hand_pinky_link2",
        ]
        self.bone_links = [
            [0, 1],
            [0, 5],
            [0, 8],
            [0, 11],
            [0, 14],
            [14, 15],
            [15, 16],
            [16, 17],
            [1, 2],
            [2, 3],
            [4, 6],
            [5, 6],
            [6, 7],
            [8, 9],
            [9, 10],
            [11, 12],
            [12, 13],
        ]
        self.weight_idx = {
            "thumb_tip": [17],
            "index_tip": [4],
            "middle_tip": [7],
            "ring_tip": [13],
            "pinky_tip": [10],
            "level_1_joints": [1, 2, 5, 8, 11, 14, 15],
            "level_2_joints": [3, 6, 9, 12, 16],
        }

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.8
        self.Ki_rot = 0.008
        self.Kd_rot = 0.02
        self.Kp_pos = 60
        self.Ki_pos = 0.08
        self.Kd_pos = 1.0
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("xhand_rh")
class XhandRH(Xhand):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/xhand/xhand_right.urdf"
        self.side = "rh"
        self.body_names = ["right_" + name for name in self.body_names]
        self.dof_names = ["right_" + name for name in self.dof_names]
        self.relative_rotation = aa_to_rotmat(np.array([0, np.pi / 2, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi]))

        self.hand2dex_mapping = {k: ["right_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["right_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("xhand_lh")
class XhandLH(Xhand):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/xhand/xhand_left.urdf"
        self.side = "lh"
        self.body_names = ["left_" + name for name in self.body_names]
        self.dof_names = ["left_" + name for name in self.dof_names]
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))

        self.hand2dex_mapping = {k: ["left_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["left_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_lh"
