from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class InspireFTP(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "inspireftp"
        self.self_collision = True
        self.body_names = [
            "base_link",
            "index_1",
            "index_2",
            "index_tip",
            "little_1",
            "little_2",
            "little_tip",
            "middle_1",
            "middle_2",
            "middle_tip",
            "ring_1",
            "ring_2",
            "ring_tip",
            "thumb_1",
            "thumb_2",
            "thumb_3",
            "thumb_4",
            "thumb_tip",
        ]
        self.dof_names = [
            "index_1_joint",
            "index_2_joint",
            "little_1_joint",
            "little_2_joint",
            "middle_1_joint",
            "middle_2_joint",
            "ring_1_joint",
            "ring_2_joint",
            "thumb_1_joint",
            "thumb_2_joint",
            "thumb_3_joint",
            "thumb_4_joint",
        ]
        self.hand2dex_mapping = {
            "wrist": ["base_link"],
            "thumb_proximal": ["thumb_1", "thumb_2"],  # one-to-many mapping
            "thumb_intermediate": ["thumb_3"],
            "thumb_distal": ["thumb_4"],
            "thumb_tip": ["thumb_tip"],
            "index_proximal": ["index_1"],
            "index_intermediate": ["index_2"],
            "index_distal": [],  # missing
            "index_tip": ["index_tip"],
            "middle_proximal": ["middle_1"],
            "middle_intermediate": ["middle_2"],
            "middle_distal": [],  # missing
            "middle_tip": ["middle_tip"],
            "ring_proximal": ["ring_1"],
            "ring_intermediate": ["ring_2"],
            "ring_distal": [],  # missing
            "ring_tip": ["ring_tip"],
            "pinky_proximal": ["little_1"],
            "pinky_intermediate": ["little_2"],
            "pinky_distal": [],  # missing
            "pinky_tip": ["little_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "thumb_4",
            "index_2",
            "middle_2",
            "ring_2",
            "little_2",
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
            "middle_tip": [9],
            "ring_tip": [12],
            "pinky_tip": [6],
            "level_1_joints": [1, 7, 14],
            "level_2_joints": [2, 4, 5, 8, 10, 11, 13, 15, 16],
        }

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.3
        self.Ki_rot = 0.001
        self.Kd_rot = 0.005
        self.Kp_pos = 20
        self.Ki_pos = 0.01
        self.Kd_pos = 2
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("inspireftp_rh")
class InspireFTPRH(InspireFTP):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/inspireftp/urdf_right/urdf/urdf_right_12_20.urdf"
        self.side = "rh"
        self.body_names = ["right_" + name for name in self.body_names]
        self.dof_names = ["right_" + name for name in self.dof_names]
        self.relative_rotation = aa_to_rotmat(np.array([np.pi, 0, 0])) @ aa_to_rotmat(np.array([0, np.pi / 2, 0]))

        self.hand2dex_mapping = {k: ["right_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["right_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("inspireftp_lh")
class InspireFTPLH(InspireFTP):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/inspireftp/urdf_left/urdf/urdf_left_12_20.urdf"
        self.side = "lh"
        self.body_names = ["left_" + name for name in self.body_names]
        self.dof_names = ["left_" + name for name in self.dof_names]
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))

        self.hand2dex_mapping = {k: ["left_" + dex_v for dex_v in v] for k, v in self.hand2dex_mapping.items()}
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.contact_body_names = ["left_" + name for name in self.contact_body_names]

    def __str__(self):
        return super().__str__() + "_lh"
