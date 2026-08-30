from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class Shadow(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "shadow"
        self.body_names = [
            "palm",
            "ffknuckle",
            "ffproximal",
            "ffmiddle",
            "ffdistal",
            "fftip",
            "lfmetacarpal",
            "lfknuckle",
            "lfproximal",
            "lfmiddle",
            "lfdistal",
            "lftip",
            "mfknuckle",
            "mfproximal",
            "mfmiddle",
            "mfdistal",
            "mftip",
            "rfknuckle",
            "rfproximal",
            "rfmiddle",
            "rfdistal",
            "rftip",
            "thbase",
            "thproximal",
            "thhub",
            "thmiddle",
            "thdistal",
            "thtip",
        ]
        self.dof_names = [
            "FFJ4",
            "FFJ3",
            "FFJ2",
            "FFJ1",
            "LFJ5",
            "LFJ4",
            "LFJ3",
            "LFJ2",
            "LFJ1",
            "MFJ4",
            "MFJ3",
            "MFJ2",
            "MFJ1",
            "RFJ4",
            "RFJ3",
            "RFJ2",
            "RFJ1",
            "THJ5",
            "THJ4",
            "THJ3",
            "THJ2",
            "THJ1",
        ]
        self.hand2dex_mapping = {
            "wrist": ["palm"],
            "thumb_proximal": ["thbase", "thproximal"],  # one-to-many mapping
            "thumb_intermediate": ["thhub", "thmiddle"],
            "thumb_distal": ["thdistal"],
            "thumb_tip": ["thtip"],
            "index_proximal": ["ffknuckle", "ffproximal"],
            "index_intermediate": ["ffmiddle"],
            "index_distal": ["ffdistal"],
            "index_tip": ["fftip"],
            "middle_proximal": ["mfknuckle", "mfproximal"],
            "middle_intermediate": ["mfmiddle"],
            "middle_distal": ["mfdistal"],
            "middle_tip": ["mftip"],
            "ring_proximal": ["rfknuckle", "rfproximal"],
            "ring_intermediate": ["rfmiddle"],
            "ring_distal": ["rfdistal"],
            "ring_tip": ["rftip"],
            "pinky_proximal": ["lfmetacarpal", "lfknuckle", "lfproximal"],
            "pinky_intermediate": ["lfmiddle"],
            "pinky_distal": ["lfdistal"],
            "pinky_tip": ["lftip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "thdistal",
            "ffdistal",
            "mfdistal",
            "rfdistal",
            "lfdistal",
        ]
        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 12],
            [0, 17],
            [0, 22],
            [2, 3],
            [3, 4],
            [4, 5],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [13, 14],
            [14, 15],
            [15, 16],
            [18, 19],
            [19, 20],
            [20, 21],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
        ]
        self.weight_idx = {
            "thumb_tip": [27],
            "index_tip": [5],
            "middle_tip": [16],
            "ring_tip": [21],
            "pinky_tip": [11],
            "level_1_joints": [1, 2, 7, 8, 12, 13, 17, 18, 22, 23],
            "level_2_joints": [3, 4, 6, 9, 10, 14, 15, 19, 20, 24, 25, 26],
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


@register_dexhand("shadow_rh")
class ShadowRH(Shadow):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/shadow_hand/shadow_hand_right_woarm.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("shadow_lh")
class ShadowLH(Shadow):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/shadow_hand/shadow_hand_left_woarm.urdf"
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, np.pi / 2, 0]))

    def __str__(self):
        return super().__str__() + "_lh"
