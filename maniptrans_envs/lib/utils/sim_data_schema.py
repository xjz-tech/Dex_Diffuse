from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1

INDEX_DATASETS = {
    "episode_id": ((), np.int64),
    "env_id": ((), np.int32),
    "step": ((), np.int64),
    "sim_frame": ((), np.int64),
    "timestamp": ((), np.float64),
    "data_index_id": ((), np.dtype("S128")),
    "done": ((), np.bool_),
    "timeout": ((), np.bool_),
    "success": ((), np.bool_),
    "failure": ((), np.bool_),
    "reset_reason": ((), np.int8),
    "block_id": ((), np.int32),
    "traj_direction": ((), np.int8),
    "skip_steps": ((), np.int32),
}

ROBOT_DATASETS = {
    "qpos": (("dof",), np.float32),
    "qvel": (("dof",), np.float32),
    "qacc": (("dof",), np.float32),
    "qacc_valid": ((), np.bool_),
    "target_before": (("dof",), np.float32),
    "target_after": (("dof",), np.float32),
    "wrist_pose_world": ((7,), np.float32),
}

OBJECT_DATASETS = {
    "position_world": ((3,), np.float32),
    "quaternion_world": ((4,), np.float32),
    "linear_velocity_world": ((3,), np.float32),
    "angular_velocity_world": ((3,), np.float32),
}

TARGET_DATASETS = {
    "object_position_world": ((3,), np.float32),
    "object_quaternion_world": ((4,), np.float32),
    "demo_frame_index": ((), np.int64),
    "data_index_id": ((), np.dtype("S128")),
    "fingertips_position_wrist": ((5, 3), np.float32),
    "delta_fingertips_position_wrist": ((5, 3), np.float32),
    "object_position_wrist": ((3,), np.float32),
    "delta_object_position_wrist": ((3,), np.float32),
    "object_quaternion_wrist": ((4,), np.float32),
    "delta_object_quaternion_wrist": ((4,), np.float32),
}

POLICY_DATASETS = {
    "action_raw": (("dof",), np.float32),
    "action_applied": (("dof",), np.float32),
}

FINGERTIP_DATASETS = {
    "position_world": ((5, 3), np.float32),
    "position_wrist": ((5, 3), np.float32),
    "contact_force": ((5, 3), np.float32),
}

OUTCOME_DATASETS = {
    "reward": ((), np.float32),
}


@dataclass(frozen=True)
class SimDataSchema:
    dof: int

    @property
    def datasets(self) -> Dict[str, Dict[str, object]]:
        specs: Dict[str, Dict[str, object]] = {}
        for prefix, datasets in (
            ("index", INDEX_DATASETS),
            ("robot", ROBOT_DATASETS),
            ("object", OBJECT_DATASETS),
            ("target", TARGET_DATASETS),
            ("policy", POLICY_DATASETS),
            ("fingertip", FINGERTIP_DATASETS),
            ("outcome", OUTCOME_DATASETS),
        ):
            for name, (shape, dtype) in datasets.items():
                specs[f"{prefix}/{name}"] = {
                    "shape": self._resolve_shape(shape),
                    "dtype": np.dtype(dtype),
                }
        return specs

    def _resolve_shape(self, shape: Sequence[object]) -> tuple:
        return tuple(self.dof if dim == "dof" else int(dim) for dim in shape)


def validate_transition_batch(batch: Mapping[str, np.ndarray], dof: int) -> int:
    expected = SimDataSchema(dof=dof).datasets
    batch_size = None
    for path, spec in expected.items():
        if path not in batch:
            raise KeyError(f"Missing transition field '{path}'")
        array = np.asarray(batch[path])
        if batch_size is None:
            batch_size = array.shape[0]
        elif array.shape[0] != batch_size:
            raise ValueError(
                f"Field '{path}' has inconsistent length {array.shape[0]} != {batch_size}"
            )
        expected_shape = (batch_size,) + tuple(spec["shape"])
        if array.shape != expected_shape:
            raise ValueError(f"Field '{path}' shape {array.shape} != {expected_shape}")
    return int(batch_size or 0)
