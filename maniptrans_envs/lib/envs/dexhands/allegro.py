from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


class Allegro(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "allegro"
        self.self_collision = True

        # ? >>>>>>>>>>>
        # ? Used only in PID-controlled wrist pose mode (reference only, not our main method).
        # ? More stable in highly dynamic scenarios but requires careful tuning.
        self.Kp_rot = 0.3
        self.Ki_rot = 0.001
        self.Kd_rot = 0.005
        self.Kp_pos = 20
        self.Ki_pos = 0.05
        self.Kd_pos = 0.2
        # ? <<<<<<<<<<

    def __str__(self):
        return self.name


@register_dexhand("allegro_rh")
class AllegroRH(Allegro):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/allegro_hand/allegro_hand_right.urdf"
        self.side = "rh"
        self.relative_rotation = aa_to_rotmat(np.array([np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        self.relative_translation = np.array([0.0, 0.0, -0.095])
        self.body_names = [
            "base_link",
            # index
            "link_0.0",
            "link_1.0",
            "link_2.0",
            "link_3.0",
            "link_3.0_tip",
            # thumb
            "link_12.0",
            "link_13.0",
            "link_14.0",
            "link_15.0",
            "link_15.0_tip",
            # middle
            "link_4.0",
            "link_5.0",
            "link_6.0",
            "link_7.0",
            "link_7.0_tip",
            # pinky
            "link_8.0",
            "link_9.0",
            "link_10.0",
            "link_11.0",
            "link_11.0_tip",
        ]
        self.dof_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
        ]
        self.hand2dex_mapping = {
            "wrist": ["base_link"],
            "thumb_proximal": ["link_12.0", "link_13.0"],  # one-to-many mapping
            "thumb_intermediate": ["link_14.0"],
            "thumb_distal": ["link_15.0"],
            "thumb_tip": ["link_15.0_tip"],
            "index_proximal": ["link_0.0", "link_1.0"],
            "index_intermediate": ["link_2.0"],
            "index_distal": ["link_3.0"],
            "index_tip": ["link_3.0_tip"],
            "middle_proximal": ["link_4.0", "link_5.0"],
            "middle_intermediate": ["link_6.0"],
            "middle_distal": ["link_7.0"],
            "middle_tip": ["link_7.0_tip"],
            "ring_proximal": ["link_8.0", "link_9.0"],
            "ring_intermediate": ["link_10.0"],
            "ring_distal": ["link_11.0"],
            "ring_tip": ["link_11.0_tip"],
            "pinky_proximal": ["link_8.0", "link_9.0"],
            "pinky_intermediate": ["link_10.0"],
            "pinky_distal": ["link_11.0"],
            "pinky_tip": ["link_11.0_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "link_15.0",
            "link_3.0",
            "link_7.0",
            "link_11.0",
            "link_11.0",
        ]
        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 11],
            [0, 16],
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
        ]
        self.weight_idx = {
            "thumb_tip": [10],
            "index_tip": [5],
            "middle_tip": [15],
            "ring_tip": [20],
            "pinky_tip": [20],
            "level_1_joints": [1, 2, 6, 7, 11, 12, 16, 17],
            "level_2_joints": [3, 4, 8, 9, 13, 14, 18, 19],
        }

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("allegro_lh")
class AllegroLH(Allegro):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/allegro_hand/allegro_hand_left.urdf"
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([-np.pi / 2, 0, 0]))
        self.relative_translation = np.array([0.0, 0.0, -0.095])
        self.body_names = [
            "base_link",
            # pinky
            "link_0.0",
            "link_1.0",
            "link_2.0",
            "link_3.0",
            "link_3.0_tip",
            # thumb
            "link_12.0",
            "link_13.0",
            "link_14.0",
            "link_15.0",
            "link_15.0_tip",
            # middle
            "link_4.0",
            "link_5.0",
            "link_6.0",
            "link_7.0",
            "link_7.0_tip",
            # index
            "link_8.0",
            "link_9.0",
            "link_10.0",
            "link_11.0",
            "link_11.0_tip",
        ]
        self.dof_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
        ]
        self.hand2dex_mapping = {
            "wrist": ["base_link"],
            "thumb_proximal": ["link_12.0", "link_13.0"],  # one-to-many mapping
            "thumb_intermediate": ["link_14.0"],
            "thumb_distal": ["link_15.0"],
            "thumb_tip": ["link_15.0_tip"],
            "index_proximal": ["link_8.0", "link_9.0"],
            "index_intermediate": ["link_10.0"],
            "index_distal": ["link_11.0"],
            "index_tip": ["link_11.0_tip"],
            "middle_proximal": ["link_4.0", "link_5.0"],
            "middle_intermediate": ["link_6.0"],
            "middle_distal": ["link_7.0"],
            "middle_tip": ["link_7.0_tip"],
            "ring_proximal": ["link_0.0", "link_1.0"],
            "ring_intermediate": ["link_2.0"],
            "ring_distal": ["link_3.0"],
            "ring_tip": ["link_3.0_tip"],
            "pinky_proximal": ["link_0.0", "link_1.0"],
            "pinky_intermediate": ["link_2.0"],
            "pinky_distal": ["link_3.0"],
            "pinky_tip": ["link_3.0_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)
        self.contact_body_names = [
            "link_15.0",
            "link_11.0",
            "link_7.0",
            "link_3.0",
            "link_3.0",
        ]
        self.bone_links = [
            [0, 1],
            [0, 6],
            [0, 11],
            [0, 16],
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
        ]
        self.weight_idx = {
            "thumb_tip": [10],
            "index_tip": [20],
            "middle_tip": [15],
            "ring_tip": [5],
            "pinky_tip": [5],
            "level_1_joints": [1, 2, 6, 7, 11, 12, 16, 17],
            "level_2_joints": [3, 4, 8, 9, 13, 14, 18, 19],
        }

    def __str__(self):
        return super().__str__() + "_lh"
