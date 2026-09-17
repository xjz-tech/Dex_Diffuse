#!/usr/bin/env python3
"""Convert TacMP LeRobot v2.1 Parquet recordings for train_bulb_dino.sh.

Input can be a collection (rank_*/id_*/meta/info.json), a rank directory,
or one LeRobot dataset with meta/info.json and meta/episodes.jsonl.

Output: <out>/replay_buffer.zarr/{data,meta}
  data/front_image, wrist_image: (N, 240, 320, 3), uint8 RGB
  data/state, action:            (N, 31), float32 (absolute source values)
  data/tactile_f6:               (N, 5, 6), float32
  data/tactile_force:            (N, 30), float32 (flattened tactile_f6)
  meta/episode_ends:             exclusive cumulative ends, int64

The DINO training task uses the images, state and action only. Tactile force
is retained for other consumers; deformation/contact points are not exported.
No hand torques are synthesized. Frames, joint order, and action timing are
preserved. Relative EE conversion is handled by BulbImageDataset at training
time, not here. Empty, uncommitted recordings (e.g. id_36) are reported and
skipped; corruption in a committed episode raises an error.

Usage (from the project root):
  /home/bighand/miniconda3/envs/dp/bin/python -m pip install pyarrow
  /home/bighand/miniconda3/envs/dp/bin/python scripts/lerobot2dp.py --dry-run
  /home/bighand/miniconda3/envs/dp/bin/python scripts/lerobot2dp.py
  DATASET_PATH="$PWD/data/real_data/realworld_bulb_sft_260909_dp" \
    task_name=bulb_sft_260909 bash train_bulb_dino.sh

The source is read-only. Output must not already exist. A temporary sibling
directory is published only after the whole conversion succeeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = "/home/bighand/wangtianyu/projects/TacMP/data/realworld_bulb_sft_260909"
DEFAULT_OUT = str(PROJECT_ROOT / "data/real_data/realworld_bulb_sft_260909_dp")
STATE_DIM = 31
CAMERAS = {"front_image": "image", "wrist_image": "extra_view_image"}
NUMERIC_COLUMNS = (
    "state", "actions", "tactile_f6", "frame_index", "episode_index",
    "timestamp", "done",
)


@dataclass(frozen=True)
class Episode:
    dataset_dir: Path
    parquet_path: Path
    index: int
    length: int
    fps: float
    tasks: list[str]


def natural_key(path: Path) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(path))]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def discover_episodes(source: Path) -> tuple[list[Episode], list[dict]]:
    """Use committed metadata; never ingest pending files or leftover PNGs."""
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    infos = sorted(source.glob("**/meta/info.json"), key=natural_key)
    if not infos:
        raise ValueError(f"no LeRobot meta/info.json found under {source}")
    episodes, skipped = [], []
    for info_path in infos:
        dataset = info_path.parent.parent
        info = json.loads(info_path.read_text())
        n_episodes, n_frames = int(info["total_episodes"]), int(info["total_frames"])
        if n_episodes == 0 and n_frames == 0:
            if any(dataset.glob("data/**/*.parquet")):
                raise ValueError(f"{dataset}: zero-episode metadata has committed data files")
            skipped.append({"dataset": str(dataset.relative_to(source)),
                            "reason": "no committed episodes (total_episodes=0, total_frames=0)"})
            continue
        if info.get("codebase_version") != "v2.1":
            raise ValueError(f"{dataset}: expected LeRobot v2.1 metadata")
        fps = float(info["fps"])
        chunk_size = int(info["chunks_size"])
        if not np.isfinite(fps) or fps <= 0 or chunk_size <= 0:
            raise ValueError(f"{dataset}: invalid fps or chunks_size")
        rows = sorted(read_jsonl(dataset / "meta/episodes.jsonl"), key=lambda r: int(r["episode_index"]))
        if len(rows) != n_episodes or sum(int(r["length"]) for r in rows) != n_frames:
            raise ValueError(f"{dataset}: info.json / episodes.jsonl counts disagree")
        indices = [int(r["episode_index"]) for r in rows]
        if len(set(indices)) != len(indices):
            raise ValueError(f"{dataset}: duplicate episode_index in episodes.jsonl")
        for row in rows:
            index, length = int(row["episode_index"]), int(row["length"])
            if index < 0 or length <= 0:
                raise ValueError(f"{dataset}: invalid episode index or length")
            parquet = (dataset / info["data_path"].format(
                episode_chunk=index // chunk_size, episode_index=index
            )).resolve()
            if not parquet.is_relative_to(dataset.resolve()):
                raise ValueError(f"{dataset}: data_path escapes the dataset directory")
            if not parquet.is_file():
                raise FileNotFoundError(f"missing committed episode: {parquet}")
            episodes.append(Episode(dataset, parquet, index, length, fps, row.get("tasks", [])))
    if not episodes:
        raise ValueError("no complete episodes found")
    if len({e.fps for e in episodes}) != 1:
        raise ValueError("mixed source frame rates; resample explicitly before combining")
    return episodes, skipped


def read_numeric(episode: Episode) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(episode.parquet_path)
    missing = set(NUMERIC_COLUMNS).union(CAMERAS.values()) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"{episode.parquet_path}: missing columns {sorted(missing)}")
    if pf.metadata.num_rows != episode.length:
        raise ValueError(f"{episode.parquet_path}: row count does not match metadata")
    table = pf.read(columns=list(NUMERIC_COLUMNS))
    result = {}
    for key, shape in {"state": (31,), "actions": (31,), "tactile_f6": (5, 6)}.items():
        if table[key].null_count:
            raise ValueError(f"{episode.parquet_path}: null {key}")
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (episode.length,) + shape or not np.isfinite(values).all():
            raise ValueError(f"{episode.parquet_path}: invalid shape or nonfinite {key}")
        result[key] = values
    for key in ("frame_index", "episode_index", "timestamp", "done"):
        if table[key].null_count:
            raise ValueError(f"{episode.parquet_path}: null {key}")
        result[key] = table[key].to_numpy()
    if not np.array_equal(result["frame_index"], np.arange(episode.length)):
        raise ValueError(f"{episode.parquet_path}: frame_index is not consecutive from zero")
    if not (result["episode_index"] == episode.index).all():
        raise ValueError(f"{episode.parquet_path}: episode_index does not match metadata")
    if not np.allclose(result["timestamp"], np.arange(episode.length) / episode.fps,
                       atol=1e-4, rtol=0):
        raise ValueError(f"{episode.parquet_path}: timestamp does not match declared fps")
    if not np.array_equal(np.flatnonzero(result["done"]), [episode.length - 1]):
        raise ValueError(f"{episode.parquet_path}: done must be true only at the last frame")
    return result


def decode_image(record: dict, *, dataset: Path, width: int, height: int) -> np.ndarray:
    """Prefer embedded bytes; an external path is relative to the dataset root."""
    import cv2

    if not isinstance(record, dict):
        raise ValueError("expected a LeRobot image struct with bytes/path")
    encoded = record.get("bytes")
    if not encoded:
        path = record.get("path")
        if not path:
            raise ValueError("image has neither bytes nor path")
        encoded = (dataset / path).read_bytes()
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image")
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_array(group, key: str, shape: tuple, dtype: str, chunk: int, compressor):
    return group.create_dataset(key, shape=shape, dtype=dtype,
                                chunks=(min(chunk, shape[0]),) + shape[1:], compressor=compressor)


def convert(args: argparse.Namespace) -> Path | None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(f"PyArrow is required. Install with: {sys.executable} -m pip install pyarrow") from exc

    source, output = Path(args.src).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must be separate directories, neither inside the other")
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"output already exists: {output}; choose a new --out (existing data is never overwritten)")
    episodes, skipped = discover_episodes(source)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    for item in skipped:
        print(f"[lerobot2dp] SKIP {item['dataset']}: {item['reason']}", flush=True)
    for episode in episodes:
        read_numeric(episode)
    total = sum(e.length for e in episodes)
    print(f"[lerobot2dp] validated {len(episodes)} episodes, {total} frames at {episodes[0].fps:g} Hz", flush=True)
    if args.dry_run:
        print("[lerobot2dp] dry-run: metadata/numeric fields checked; images are decoded during conversion")
        return None

    import cv2
    import zarr
    from numcodecs import Blosc

    if int(zarr.__version__.split('.')[0]) != 2:
        raise RuntimeError('This project requires Zarr 2; install "zarr<3" and compatible numcodecs')
    cv2.setNumThreads(1)  # Parallelize images, not each image's resize operation.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.converting-", dir=output.parent))
    manifest = {
        "source": str(source), "source_format": "LeRobot v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fps": episodes[0].fps, "total_episodes": len(episodes), "total_frames": total,
        "camera_mapping": CAMERAS, "image_shape_hwc": [args.height, args.width, 3],
        "image_resize": "OpenCV INTER_AREA, RGB", "image_chunk": args.image_chunk,
        "action_mode": "absolute source actions, no shifting or joint reordering",
        "temporal_processing": "all committed frames retained, no resampling or phase trimming",
        "extra_fields": "tactile_f6 and flattened tactile_force; not consumed by current bulb_image task",
        "omitted_fields": ["tactile_deform", "tactile_contact_points", "done", "segment_id", "timestamp", "timing"],
        "skipped_datasets": skipped, "episodes": [],
    }
    try:
        root = zarr.open_group(str(staging / "replay_buffer.zarr"), mode="w")
        data, meta = root.create_group("data"), root.create_group("meta")
        compressor = Blosc(cname="zstd", clevel=args.compression_level, shuffle=Blosc.SHUFFLE)
        arrays = {key: create_array(data, key, (total, args.height, args.width, 3), "uint8", args.image_chunk, compressor)
                  for key in CAMERAS}
        for key, shape in {"state": (STATE_DIM,), "action": (STATE_DIM,), "tactile_f6": (5, 6), "tactile_force": (30,)}.items():
            arrays[key] = create_array(data, key, (total,) + shape, "float32", 4096, compressor)

        cursor, ends = 0, []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, episode in enumerate(episodes):
                end = cursor + episode.length
                label = str(episode.dataset_dir.relative_to(source))
                numeric = read_numeric(episode)
                arrays["state"][cursor:end] = numeric["state"]
                arrays["action"][cursor:end] = numeric["actions"]
                arrays["tactile_f6"][cursor:end] = numeric["tactile_f6"]
                arrays["tactile_force"][cursor:end] = numeric["tactile_f6"].reshape(episode.length, 30)
                pf = pq.ParquetFile(episode.parquet_path)
                decode = partial(decode_image, dataset=episode.dataset_dir, width=args.width, height=args.height)
                local = 0
                for batch in pf.iter_batches(batch_size=args.image_batch_size, columns=list(CAMERAS.values())):
                    size = batch.num_rows
                    for target, key in CAMERAS.items():
                        records = batch.column(batch.schema.get_field_index(key)).to_pylist()
                        try:
                            images = np.stack(list(pool.map(decode, records)))
                        except Exception as exc:
                            raise ValueError(f"{episode.parquet_path}: {key} decode failed in frames {local}:{local+size}: {exc}") from exc
                        arrays[target][cursor + local:cursor + local + size] = images
                    local += size
                if local != episode.length:
                    raise ValueError(f"{episode.parquet_path}: image/numeric row count mismatch")
                manifest["episodes"].append({
                    "output_episode_index": i, "source_dataset": label,
                    "source_episode_index": episode.index,
                    "source_parquet": str(episode.parquet_path.relative_to(source)),
                    "frames": episode.length, "start": cursor, "end": end, "tasks": episode.tasks,
                })
                ends.append(end)
                cursor = end
                print(f"[lerobot2dp] {i+1}/{len(episodes)} {label}/episode_{episode.index}: [{end-episode.length}:{end}]", flush=True)

        meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64), compressor=None)
        root.attrs.update({"source": str(source), "fps": episodes[0].fps, "camera_mapping": CAMERAS,
                           "action_mode": "absolute", "conversion_manifest": "../conversion_manifest.json"})
        if int(meta["episode_ends"][-1]) != total or any(a.shape[0] != total for a in arrays.values()):
            raise ValueError("output arrays and episode_ends disagree")
        (staging / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        if output.exists():
            raise FileExistsError(f"output appeared during conversion: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"[lerobot2dp] done: {output / 'replay_buffer.zarr'}")
    print(f"DATASET_PATH={shlex.quote(str(output))} task_name=bulb_sft_260909 bash train_bulb_dino.sh")
    return output / "replay_buffer.zarr"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=DEFAULT_SRC, help="LeRobot dataset or collection directory")
    parser.add_argument("--out", default=DEFAULT_OUT, help="new output directory containing replay_buffer.zarr")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--image-batch-size", type=int, default=64, help="maximum images decoded per camera batch")
    parser.add_argument("--image-chunk", type=int, default=1, help="frames per image chunk; 1 suits the launcher's n_obs_steps=1")
    parser.add_argument("--workers", type=int, default=4, help="image decode threads")
    parser.add_argument("--compression-level", type=int, default=3, choices=range(10))
    parser.add_argument("--max-episodes", type=int, default=None, help="convert the first N complete episodes")
    parser.add_argument("--dry-run", action="store_true", help="check metadata and numeric fields without writing output")
    args = parser.parse_args(argv)
    for name in ("width", "height", "image_batch_size", "image_chunk", "workers", "max_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
