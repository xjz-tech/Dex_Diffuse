#!/usr/bin/env python3
"""Convert the TacMP bulb demonstrations to Diffusion Policy replay-buffer zarr.

The source directory is expected to contain ``episode_<id>`` directories with
the following layout::

    front/000000.png ...
    wrist/000000.png ...
    state.npy              # (T, 31), arm pose (9) + hand joints (22)
    action.npy             # (T, 31), absolute commanded state
    hand_joint_torque.npy  # (T, 22)
    tactile_obs.npy        # dict containing tactile_f6 with shape (T, 5, 6)

The output is a standard Diffusion Policy ``ReplayBuffer``::

    <output>/replay_buffer.zarr/
      data/
        front_image        # (N, H, W, 3), uint8 RGB
        wrist_image        # (N, H, W, 3), uint8 RGB
        state              # (N, 31), float32
        hand_joint_torque  # (N, 22), float32
        tactile_f6         # (N, 5, 6), float32
        tactile_force      # (N, 30), float32; flattened tactile_f6
        action             # (N, 31), float32
      meta/
        episode_ends       # exclusive cumulative episode ends, int64

Example:
    /home/wty/miniconda3/envs/rdp/bin/python data2dp.py

The generated key names can be used directly in a task's ``shape_meta``.  RGB
arrays are stored channel-last, as expected by Diffusion Policy datasets.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import os.path as osp
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_SRC = "/mnt/work/dexIL/Dex_Diffuse/data/real_data/bulb_tac_80"
DEFAULT_OUT = "/mnt/work/dexIL/Dex_Diffuse/data/real_data/bulb_tac_80_dp_"

ARM_DIM = 9
HAND_DIM = 22
STATE_DIM = ARM_DIM + HAND_DIM
TACTILE_SHAPE = (5, 6)


# tactile_obs.npy was created with NumPy >= 2.  A pickle-free cache lets the
# NumPy 1.x Diffusion Policy environment load it safely.
_F6_CONVERTER = r"""
import os
import sys
import numpy as np

src, cache_dir = sys.argv[1], sys.argv[2]
os.makedirs(cache_dir, exist_ok=True)
obs = np.load(src, allow_pickle=True).item()
f6 = np.ascontiguousarray(obs["tactile_f6"], dtype=np.float32)
channels = np.asarray(obs.get("tactile_channels", np.arange(f6.shape[1])))
np.save(os.path.join(cache_dir, "f6.npy"), f6)
np.save(os.path.join(cache_dir, "channels.npy"), channels)
"""


@dataclass
class Episode:
    path: str
    front_paths: list[str]
    wrist_paths: list[str]
    keep: np.ndarray
    source_length: int
    dropped_image_frames: int
    dropped_stalled_frames: int
    dropped_npy_frames: int

    @property
    def length(self) -> int:
        return int(self.keep.size)


def _numpy_major(python: str) -> int | None:
    try:
        version = subprocess.check_output(
            [python, "-c", "import numpy; print(numpy.__version__)"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(version.split(".", 1)[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _find_numpy2_python() -> str | None:
    candidates = [
        os.environ.get("NUMPY2_PYTHON"),
        sys.executable,
        "/home/bighand/miniconda3/envs/tianji/bin/python",
        "/home/wty/miniconda3/bin/python",
        "python3",
        "python",
    ]
    seen: set[str] = set()
    for python in candidates:
        if not python:
            continue
        if python in seen:
            continue
        seen.add(python)
        major = _numpy_major(python)
        if major is not None and major >= 2:
            return python
    return None


def load_tactile_f6(episode_dir: str) -> np.ndarray:
    """Load (T, 5, 6) tactile data, creating a portable cache if necessary."""
    cache_dir = osp.join(episode_dir, "tac_cache")
    cache_path = osp.join(cache_dir, "f6.npy")
    if not osp.isfile(cache_path):
        source_path = osp.join(episode_dir, "tactile_obs.npy")
        if not osp.isfile(source_path):
            raise FileNotFoundError(source_path)

        python = _find_numpy2_python()
        if python is None:
            raise RuntimeError(
                f"{source_path} requires NumPy >= 2 to decode, but no suitable "
                "Python interpreter was found"
            )
        print(f"  building tactile cache for {osp.basename(episode_dir)}")
        subprocess.run(
            [python, "-c", _F6_CONVERTER, source_path, cache_dir], check=True
        )

    f6 = np.load(cache_path, mmap_mode="r")
    if f6.ndim != 3 or tuple(f6.shape[1:]) != TACTILE_SHAPE:
        raise ValueError(
            f"{cache_path}: expected shape (T, {TACTILE_SHAPE[0]}, "
            f"{TACTILE_SHAPE[1]}), got {f6.shape}"
        )
    return f6


def list_episode_dirs(src: str, max_episodes: int | None = None) -> list[str]:
    if not osp.isdir(src):
        raise FileNotFoundError(f"source directory does not exist: {src}")

    result = []
    for name in os.listdir(src):
        path = osp.join(src, name)
        if not name.startswith("episode_") or not osp.isdir(path):
            continue
        try:
            index = int(name.removeprefix("episode_"))
        except ValueError:
            continue
        result.append((index, path))
    result.sort(key=lambda item: item[0])
    paths = [path for _, path in result]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise RuntimeError(f"no episode_<integer> directories found under {src}")
    return paths


def downsample_indices(length: int, ratio: int) -> np.ndarray:
    """Decimate an episode while always retaining its first and last frames."""
    if ratio <= 1 or length <= 2:
        return np.arange(length, dtype=np.int64)
    middle = np.arange(1, length - 1, ratio, dtype=np.int64)
    return np.concatenate(
        (np.array([0], dtype=np.int64), middle, np.array([length - 1], dtype=np.int64))
    )


def index_image_paths(directory: str) -> dict[int, str]:
    """Index PNG files by their integer filename stem."""
    result: dict[int, str] = {}
    for path in glob.glob(osp.join(directory, "*.png")):
        stem = osp.splitext(osp.basename(path))[0]
        try:
            frame_index = int(stem)
        except ValueError:
            continue
        if frame_index in result:
            raise ValueError(
                f"{directory}: duplicate image index {frame_index}: "
                f"{result[frame_index]} and {path}"
            )
        result[frame_index] = path
    return result


def image_signature(path: str | None) -> bytes | None:
    """Return a decoded RGB fingerprint, or None for a missing/broken image."""
    if path is None or not osp.isfile(path) or osp.getsize(path) == 0:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            digest = hashlib.blake2b(digest_size=16)
            digest.update(f"{rgb.width}x{rgb.height}:".encode("ascii"))
            digest.update(rgb.tobytes())
        return digest.digest()
    except (OSError, SyntaxError, ValueError):
        return None


def inspect_camera_frames(
    paths_by_index: dict[int, str], length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return readable and stalled masks for one camera sequence."""
    readable = np.zeros(length, dtype=bool)
    stalled = np.zeros(length, dtype=bool)
    previous_signature: bytes | None = None
    for frame_index in range(length):
        signature = image_signature(paths_by_index.get(frame_index))
        if signature is None:
            # Keep the last readable signature so a camera that remains frozen
            # across one or more missing frames is still detected afterwards.
            continue
        readable[frame_index] = True
        if previous_signature is not None and signature == previous_signature:
            stalled[frame_index] = True
        previous_signature = signature
    return readable, stalled


def inspect_episode(path: str, downsample: int) -> Episode:
    try:
        state = np.load(osp.join(path, "state.npy"), mmap_mode="r")
        action = np.load(osp.join(path, "action.npy"), mmap_mode="r")
        torque = np.load(osp.join(path, "hand_joint_torque.npy"), mmap_mode="r")
        f6 = load_tactile_f6(path)
    except Exception as exc:
        raise ValueError(f"failed to load required NPY data: {exc}") from exc

    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"{path}/state.npy: expected (T, {STATE_DIM}), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != STATE_DIM:
        raise ValueError(f"{path}/action.npy: expected (T, {STATE_DIM}), got {action.shape}")
    if torque.ndim != 2 or torque.shape[1] != HAND_DIM:
        raise ValueError(
            f"{path}/hand_joint_torque.npy: expected (T, {HAND_DIM}), got {torque.shape}"
        )

    lengths = {
        "state": state.shape[0],
        "action": action.shape[0],
        "hand_joint_torque": torque.shape[0],
        "tactile_f6": f6.shape[0],
    }
    if len(set(lengths.values())) != 1:
        details = ", ".join(f"{key}={value}" for key, value in lengths.items())
        raise ValueError(f"{path}: modality lengths do not match: {details}")

    length = state.shape[0]
    if length == 0:
        raise ValueError(f"{path}: empty episode")

    try:
        finite_mask = (
            np.isfinite(state).all(axis=1)
            & np.isfinite(action).all(axis=1)
            & np.isfinite(torque).all(axis=1)
            & np.isfinite(f6).all(axis=(1, 2))
        )
    except TypeError as exc:
        raise ValueError(f"{path}: NPY data must contain numeric values") from exc

    front_by_index = index_image_paths(osp.join(path, "front"))
    wrist_by_index = index_image_paths(osp.join(path, "wrist"))
    front_readable, front_stalled = inspect_camera_frames(front_by_index, length)
    wrist_readable, wrist_stalled = inspect_camera_frames(wrist_by_index, length)
    image_mask = front_readable & wrist_readable
    stalled_mask = image_mask & (front_stalled | wrist_stalled)

    valid_indices = np.flatnonzero(finite_mask & image_mask & ~stalled_mask)
    if valid_indices.size == 0:
        raise ValueError(f"{path}: no frame has valid NPY data and both camera images")

    selected_positions = downsample_indices(valid_indices.size, downsample)
    keep = valid_indices[selected_positions]
    return Episode(
        path=path,
        front_paths=[front_by_index[int(i)] for i in keep],
        wrist_paths=[wrist_by_index[int(i)] for i in keep],
        keep=keep,
        source_length=length,
        dropped_image_frames=int((~image_mask).sum()),
        dropped_stalled_frames=int(stalled_mask.sum()),
        dropped_npy_frames=int((~finite_mask).sum()),
    )


def load_rgb_batch(paths: Sequence[str], width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        from PIL import Image

        result = np.empty((len(paths), height, width, 3), dtype=np.uint8)
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.BOX)
                    result[index] = np.asarray(image, dtype=np.uint8)
            except (OSError, SyntaxError, ValueError) as exc:
                raise RuntimeError(f"failed to read image: {path}") from exc
        return result

    result = np.empty((len(paths), height, width, 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        result[index] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return result


def create_array(group, name: str, shape: tuple[int, ...], dtype: str, chunk0: int, compressor):
    chunks = (min(chunk0, shape[0]),) + shape[1:]
    return group.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        overwrite=True,
    )


def convert(args: argparse.Namespace) -> str | None:
    episode_dirs = list_episode_dirs(args.src, args.max_episodes)
    episodes = []
    total = 0
    print(f"[data2dp] inspecting {len(episode_dirs)} episode(s) under {args.src}")
    for path in episode_dirs:
        try:
            episode = inspect_episode(path, args.downsample)
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"  WARNING: skipping {osp.basename(path)}: {exc}",
                file=sys.stderr,
            )
            continue
        episodes.append(episode)
        total += episode.length
        print(
            f"  {osp.basename(path)}: {episode.source_length} -> "
            f"{episode.length} frame(s); dropped for missing/broken image="
            f"{episode.dropped_image_frames}, stalled image="
            f"{episode.dropped_stalled_frames}, "
            f"NPY={episode.dropped_npy_frames}"
        )
    if not episodes:
        raise RuntimeError("no valid episodes remain after input validation")
    print(f"[data2dp] total output frames: {total}")
    if args.dry_run:
        print("[data2dp] dry-run complete; no replay-buffer output was written")
        return None

    # Import these here so --dry-run can validate source data in a lightweight
    # environment that does not have the Diffusion Policy zarr dependencies.
    import zarr
    from numcodecs import Blosc

    output_dir = osp.abspath(args.out)
    protected_paths = {
        osp.abspath(args.src),
        osp.abspath(os.getcwd()),
        osp.abspath(osp.expanduser("~")),
        osp.abspath(os.sep),
    }
    if output_dir in protected_paths:
        raise ValueError(f"refusing to use protected path as output directory: {output_dir}")
    zarr_path = osp.join(output_dir, "replay_buffer.zarr")
    if osp.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"output already exists: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

    try:
        root = zarr.open_group(zarr_path, mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        compressor = Blosc(cname="zstd", clevel=args.compression_level, shuffle=1)

        image_shape = (total, args.height, args.width, 3)
        arrays = {
            "front_image": create_array(
                data, "front_image", image_shape, "uint8", args.image_chunk, compressor
            ),
            "wrist_image": create_array(
                data, "wrist_image", image_shape, "uint8", args.image_chunk, compressor
            ),
            "state": create_array(data, "state", (total, STATE_DIM), "float32", 4096, compressor),
            "hand_joint_torque": create_array(
                data, "hand_joint_torque", (total, HAND_DIM), "float32", 4096, compressor
            ),
            "tactile_f6": create_array(
                data, "tactile_f6", (total,) + TACTILE_SHAPE, "float32", 4096, compressor
            ),
            "tactile_force": create_array(
                data, "tactile_force", (total, 30), "float32", 4096, compressor
            ),
            "action": create_array(
                data, "action", (total, STATE_DIM), "float32", 4096, compressor
            ),
        }

        cursor = 0
        episode_ends = []
        for episode in episodes:
            start, end = cursor, cursor + episode.length
            keep = episode.keep
            print(f"[data2dp] writing {osp.basename(episode.path)} -> [{start}:{end}]")

            state = np.load(osp.join(episode.path, "state.npy"), mmap_mode="r")
            action = np.load(osp.join(episode.path, "action.npy"), mmap_mode="r")
            torque = np.load(
                osp.join(episode.path, "hand_joint_torque.npy"), mmap_mode="r"
            )
            f6 = load_tactile_f6(episode.path)
            arrays["state"][start:end] = np.asarray(state[keep], dtype=np.float32)
            arrays["action"][start:end] = np.asarray(action[keep], dtype=np.float32)
            arrays["hand_joint_torque"][start:end] = np.asarray(
                torque[keep], dtype=np.float32
            )
            f6_kept = np.asarray(f6[keep], dtype=np.float32)
            arrays["tactile_f6"][start:end] = f6_kept
            arrays["tactile_force"][start:end] = f6_kept.reshape(episode.length, -1)

            # Decode a small batch at a time to keep peak memory bounded.
            for local_start in range(0, episode.length, args.image_batch_size):
                local_end = min(local_start + args.image_batch_size, episode.length)
                target = slice(start + local_start, start + local_end)
                arrays["front_image"][target] = load_rgb_batch(
                    episode.front_paths[local_start:local_end], args.width, args.height
                )
                arrays["wrist_image"][target] = load_rgb_batch(
                    episode.wrist_paths[local_start:local_end], args.width, args.height
                )

            cursor = end
            episode_ends.append(end)

        episode_ends_array = np.asarray(episode_ends, dtype=np.int64)
        meta.create_dataset(
            "episode_ends",
            data=episode_ends_array,
            shape=episode_ends_array.shape,
            chunks=episode_ends_array.shape,
            dtype="int64",
            compressor=None,
            overwrite=True,
        )
    except BaseException:
        # Do not leave an output directory that looks like a complete dataset.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(f"[data2dp] done: {zarr_path}")
    for name, array in sorted(root["data"].arrays()):
        print(f"  data/{name}: shape={array.shape}, dtype={array.dtype}")
    print(f"  meta/episode_ends: {episode_ends}")
    return zarr_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="source dataset directory")
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="output directory containing replay_buffer.zarr"
    )
    parser.add_argument("--width", type=int, default=320, help="output image width")
    parser.add_argument("--height", type=int, default=240, help="output image height")
    parser.add_argument(
        "--downsample", type=int, default=1, help="temporal stride; first/last frames are retained"
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--image-chunk", type=int, default=16)
    parser.add_argument("--compression-level", type=int, default=3, choices=range(0, 10))
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="convert only the first N episodes (testing)"
    )
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without writing")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    for name in ("width", "height", "downsample", "image_batch_size", "image_chunk"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
