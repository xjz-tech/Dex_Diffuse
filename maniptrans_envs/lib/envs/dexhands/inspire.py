from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class Inspire(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "inspire"
        self.body_names = [
            "hand_base_link",
            "index_proximal",
            "index_intermediate",
            "index_tip",
            "middle_proximal",
            "middle_intermediate",
            "middle_tip",
            "pinky_proximal",
            "pinky_intermediate",
            "pinky_tip",
            "ring_proximal",
            "ring_intermediate",
            "ring_tip",
            "thumb_proximal_base",
            "thumb_proximal",
            "thumb_intermediate",
            "thumb_distal",
            "thumb_tip",
        ]
        self.dof_names = [
            "index_proximal_joint",
            "index_intermediate_joint",
            "middle_proximal_joint",
            "middle_intermediate_joint",
            "pinky_proximal_joint",
            "pinky_intermediate_joint",
            "ring_proximal_joint",
            "ring_intermediate_joint",
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "thumb_intermediate_joint",
            "thumb_distal_joint",
        ]
        self.hand2dex_mapping = {
            "wrist": ["hand_base_link"],
            "thumb_proximal": ["thumb_proximal", "thumb_proximal_base"],  # one-to-many mapping
            "thumb_intermediate": ["thumb_intermediate"],
            "thumb_distal": ["thumb_distal"],
            "thumb_tip": ["thumb_tip"],
            "index_proximal": ["index_proximal"],
            "index_intermediate": ["index_intermediate"],
            "index_distal": [],  # missing
            "index_tip": ["index_tip"],
            "middle_proximal": ["middle_proximal"],
            "middle_intermediate": ["middle_intermediate"],
            "middle_distal": [],  # missing
            "middle_tip": ["middle_tip"],
            "ring_proximal": ["ring_proximal"],
            "ring_intermediate": ["ring_intermediate"],
            "ring_distal": [],  # missing
            "ring_tip": ["ring_tip"],
            "pinky_proximal": ["pinky_proximal"],
            "pinky_intermediate": ["pinky_intermediate"],
            "pinky_distal": [],  # missing
            "pinky_tip": ["pinky_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "thumb_distal",
            "index_intermediate",
            "middle_intermediate",
            "ring_intermediate",
            "pinky_intermediate",
        ]
        self.bone_links = [
            [0, 1],
            [0, 4],
            [0, 7],
            [0, 10],
            [0, 13],
            [13, 14],
            [3, 2],
            [2, 1],
            [6, 5],
            [5, 4],
            [9, 8],
            [8, 7],
            [12, 11],
            [11, 10],
            [17, 16],
            [16, 15],
            [15, 14],
        ]
        self.weight_idx = {
            "thumb_tip": [17],
            "index_tip": [3],
            "middle_tip": [6],
            "ring_tip": [12],
            "pinky_tip": [9],
            "level_1_joints": [1, 4, 14],
            "level_2_joints": [2, 5, 7, 8, 10, 11, 13, 15, 16],
        }

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.5
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 20
        self.Ki_pos = 0.005
        self.Kd_pos = 0.1
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("inspire_rh")
class InspireRH(Inspire):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/inspire_hand/inspire_hand_right.urdf"
        self.side = "rh"
        self.body_names = ["R_" + name for name in self.body_names]
        self.dof_names = ["R_" + name for name in self.dof_names]
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 36, 0, 0]))
            @ aa_to_rotmat(np.array([0, 0, np.pi / 36]))
            @ aa_to_rotmat(np.array([0, 0, -np.pi / 2]))
            @ aa_to_rotmat(np.array([0, np.pi, 0]))
        )

        self.hand2dex_mapping = {k: ["R_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["R_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("inspire_lh")
class InspireLH(Inspire):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/inspire_hand/inspire_hand_left.urdf"
        self.side = "lh"
        self.body_names = ["L_" + name for name in self.body_names]
        self.dof_names = ["L_" + name for name in self.dof_names]
        self.relative_rotation = (
            aa_to_rotmat(np.array([-np.pi / 36, 0, 0]))
            @ aa_to_rotmat(np.array([0, 0, -np.pi / 36]))
            @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        )

        self.hand2dex_mapping = {k: ["L_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["L_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_lh"
