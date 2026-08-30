from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class Artimano(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "artimano"
        self.self_collision = True
        self.body_names = [
            "palm",
            "index1y",
            "index1z",
            "index2",
            "index3",
            "index_tip",
            "middle1y",
            "middle1z",
            "middle2",
            "middle3",
            "middle_tip",
            "pinky1y",
            "pinky1z",
            "pinky2",
            "pinky3",
            "pinky_tip",
            "ring1y",
            "ring1z",
            "ring2",
            "ring3",
            "ring_tip",
            "thumb1x",
            "thumb1y",
            "thumb1z",
            "thumb2y",
            "thumb2z",
            "thumb3",
            "thumb_tip",
        ]
        self.dof_names = [
            "j_index1y",
            "j_index1z",
            "j_index2",
            "j_index3",
            "j_middle1y",
            "j_middle1z",
            "j_middle2",
            "j_middle3",
            "j_pinky1y",
            "j_pinky1z",
            "j_pinky2",
            "j_pinky3",
            "j_ring1y",
            "j_ring1z",
            "j_ring2",
            "j_ring3",
            "j_thumb1x",
            "j_thumb1y",
            "j_thumb1z",
            "j_thumb2y",
            "j_thumb2z",
            "j_thumb3",
        ]
        self.hand2dex_mapping = {
            "wrist": ["palm"],
            "thumb_proximal": ["thumb1x", "thumb1y", "thumb1z"],
            "thumb_intermediate": ["thumb2y", "thumb2z"],
            "thumb_distal": ["thumb3"],
            "thumb_tip": ["thumb_tip"],
            "index_proximal": ["index1y", "index1z"],
            "index_intermediate": ["index2"],
            "index_distal": ["index3"],
            "index_tip": ["index_tip"],
            "middle_proximal": ["middle1y", "middle1z"],
            "middle_intermediate": ["middle2"],
            "middle_distal": ["middle3"],
            "middle_tip": ["middle_tip"],
            "ring_proximal": ["ring1y", "ring1z"],
            "ring_intermediate": ["ring2"],
            "ring_distal": ["ring3"],
            "ring_tip": ["ring_tip"],
            "pinky_proximal": ["pinky1y", "pinky1z"],
            "pinky_intermediate": ["pinky2"],
            "pinky_distal": ["pinky3"],
            "pinky_tip": ["pinky_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "thumb3",
            "index3",
            "middle3",
            "ring3",
            "pinky3",
        ]

        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 11],
            [0, 16],
            [0, 21],
            [2, 3],
            [3, 4],
            [4, 5],
            [7, 8],
            [8, 9],
            [9, 10],
            [12, 13],
            [13, 14],
            [14, 15],
            [17, 18],
            [18, 19],
            [19, 20],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
        ]
        self.weight_idx = {
            "thumb_tip": [27],
            "index_tip": [5],
            "middle_tip": [10],
            "ring_tip": [20],
            "pinky_tip": [15],
            "level_1_joints": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22, 23],
            "level_2_joints": [3, 4, 8, 9, 13, 14, 18, 19, 24, 25, 26],
        }

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.3
        self.Ki_rot = 0.01
        self.Kd_rot = 0.005
        self.Kp_pos = 10
        self.Ki_pos = 0.003
        self.Kd_pos = 0.5
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("artimano_rh")
class ArtimanoRH(Artimano):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/mano_urdf/rh_mano.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0, 0, 0]))

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("artimano_lh")
class ArtimanoLH(Artimano):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/mano_urdf/lh_mano.urdf"
        self.side = "lh"

        self.relative_rotation = aa_to_rotmat(np.array([0, 0, 0]))

    def __str__(self):
        return super().__str__() + "_lh"
