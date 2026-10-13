#!/usr/bin/env python3
"""Export LeRobot v2.1 episodes to the explicit TacMP data layout.

The output is one ``episode_<n>`` directory per committed LeRobot episode::

    episode_0/{front,wrist}/000000.png ...
    episode_0/{state,action}.npy       # (T, 31), float32
    episode_0/hand_joint_torque.npy    # (T, 22), zero-filled (not in LeRobot)
    episode_0/obj_state.npy            # (T, 16), zero-filled (not in LeRobot)
    episode_0/tactile_obs.npy          # dict, including tactile_f6/deform

The source is never modified and the output is published atomically.  Images
are decoded from the embedded LeRobot image records and written as PNG files.
The source 6D rotations are converted from ``[R[:,0], R[:,1]]`` to the
row-major layout used by the explicit TacMP files.

Example::

    python scripts/lerobot2data.py \
      --src data/real_data/realworld_bulb_sft_20261001 \
      --out data/real_data/bulb_tac_261001
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


STATE_DIM = 31


def natural_key(value: Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def rotation_6d_to_rows(values: np.ndarray) -> np.ndarray:
    """Convert LeRobot column-major 6D rotations to explicit row-major 6D."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError(f"expected (N, {STATE_DIM}), got {values.shape}")
    first, second = values[:, 3:6], values[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm <= 1e-8):
        raise ValueError("cannot convert a rotation with a zero first column")
    first = first / first_norm
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm <= 1e-8):
        raise ValueError("cannot convert a rotation with collinear first two columns")
    second = second / second_norm
    third = np.cross(first, second)
    converted = values.copy()
    matrix = np.stack((first, second, third), axis=-1)
    converted[:, 3:9] = matrix[:, :2, :].reshape(-1, 6)
    return converted


def image_bytes(record: dict, dataset: Path) -> bytes:
    if not isinstance(record, dict):
        raise ValueError(f"invalid image record: {type(record)!r}")
    if record.get("bytes"):
        return record["bytes"]
    if record.get("path"):
        return (dataset / record["path"]).read_bytes()
    raise ValueError("image record has neither bytes nor path")


def discover(source: Path) -> list[tuple[Path, dict, Path, int]]:
    """Return (dataset, info, parquet, output_id) for committed episodes."""
    result = []
    for info_path in sorted(source.glob("**/meta/info.json"), key=natural_key):
        dataset = info_path.parent.parent
        info = json.loads(info_path.read_text())
        if int(info.get("total_episodes", 0)) == 0:
            continue
        rows = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        chunk_size = int(info["chunks_size"])
        id_match = re.fullmatch(r"id_(\d+)", dataset.name)
        for row in sorted(rows, key=lambda item: int(item["episode_index"])):
            index = int(row["episode_index"])
            parquet = dataset / info["data_path"].format(
                episode_chunk=index // chunk_size, episode_index=index
            )
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
            if int(row["length"]) > 0:
                output_id = int(id_match.group(1)) if id_match else index
                result.append((dataset, info, parquet, output_id))
    if not result:
        raise ValueError(f"no committed episodes found under {source}")
    return result


def convert(args: argparse.Namespace) -> Path:
    try:
        import cv2
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("this script requires opencv-python and pyarrow") from exc

    source = Path(args.src).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must be separate directories")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    episodes = discover(source)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.converting-", dir=output.parent))
    manifest = {
        "source": str(source), "source_format": "LeRobot v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": [], "missing_fields_filled_with_zeros": ["obj_state", "hand_joint_torque"],
    }
    try:
        used_ids = set()
        for output_index, (dataset, info, parquet_path, output_id) in enumerate(episodes):
            if output_id in used_ids:
                raise ValueError(f"duplicate output episode id: {output_id}")
            used_ids.add(output_id)
            pf = pq.ParquetFile(parquet_path)
            names = set(pf.schema_arrow.names)
            required = {"state", "actions", "image", "extra_view_image", "tactile_f6", "tactile_deform", "tactile_contact_points"}
            missing = required - names
            if missing:
                raise ValueError(f"{parquet_path}: missing columns {sorted(missing)}")
            numeric = pf.read(columns=["state", "actions", "tactile_f6", "tactile_deform", "tactile_contact_points"])
            state = np.asarray(numeric["state"].to_pylist(), dtype=np.float32)
            action = np.asarray(numeric["actions"].to_pylist(), dtype=np.float32)
            tactile_f6 = np.asarray(numeric["tactile_f6"].to_pylist(), dtype=np.float32)
            tactile_deform = np.asarray(numeric["tactile_deform"].to_pylist(), dtype=np.uint8)
            length = state.shape[0]
            if state.shape != (length, STATE_DIM) or action.shape != (length, STATE_DIM):
                raise ValueError(f"{parquet_path}: invalid state/action shape")
            if tactile_f6.shape != (length, 5, 6) or tactile_deform.shape != (length, 5, 240, 240):
                raise ValueError(f"{parquet_path}: invalid tactile shape")
            episode_dir = staging / f"episode_{output_id}"
            (episode_dir / "front").mkdir(parents=True)
            (episode_dir / "wrist").mkdir()
            np.save(episode_dir / "state.npy", rotation_6d_to_rows(state))
            np.save(episode_dir / "action.npy", rotation_6d_to_rows(action))
            np.save(episode_dir / "hand_joint_torque.npy", np.zeros((length, 22), dtype=np.float32))
            np.save(episode_dir / "obj_state.npy", np.zeros((length, 16), dtype=np.float32))

            contacts = []
            for value in numeric["tactile_contact_points"].to_pylist():
                try:
                    contacts.append(json.loads(value) if isinstance(value, str) else value)
                except json.JSONDecodeError:
                    contacts.append(value)
            tactile = {
                "tactile_f6": tactile_f6, "tactile_deform": tactile_deform,
                "tactile_timestamps": np.zeros((length, 5), dtype=np.float64),
                "tactile_frame_ids": np.zeros((length, 5), dtype=np.int64),
                "tactile_channels": np.arange(5, dtype=np.int64),
                "sample_time_ns": np.zeros(length, dtype=np.int64),
                "tactile_contact_points": contacts,
            }
            timing = dataset / "timing" / parquet_path.name
            if timing.is_file():
                timing_table = pq.read_table(timing, columns=["tactile_timestamps", "tactile_frame_ids", "sample_time_ns"])
                tactile["tactile_timestamps"] = np.asarray(timing_table["tactile_timestamps"].to_pylist(), dtype=np.float64)
                tactile["tactile_frame_ids"] = np.asarray(timing_table["tactile_frame_ids"].to_pylist(), dtype=np.int64)
                tactile["sample_time_ns"] = np.asarray(timing_table["sample_time_ns"].to_pylist(), dtype=np.int64)
            np.save(episode_dir / "tactile_obs.npy", tactile, allow_pickle=True)

            local = 0
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=["image", "extra_view_image"]):
                for row in range(batch.num_rows):
                    for column, folder in (("image", "front"), ("extra_view_image", "wrist")):
                        encoded = image_bytes(batch[column][row].as_py(), dataset)
                        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None or not cv2.imwrite(str(episode_dir / folder / f"{local:06d}.png"), image):
                            raise ValueError(f"{parquet_path}: failed to decode/write {column} frame {local}")
                    local += 1
            if local != length:
                raise ValueError(f"{parquet_path}: image rows {local} != numeric rows {length}")
            manifest["episodes"].append({"output_episode_index": output_id, "source_dataset": str(dataset.relative_to(source)), "source_parquet": str(parquet_path.relative_to(source)), "frames": length})
            print(f"[lerobot2data] {output_index + 1}/{len(episodes)} episode_{output_id}: {length} frames", flush=True)
        (staging / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[lerobot2data] done: {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="LeRobot collection/dataset directory")
    parser.add_argument("--out", required=True, help="new explicit output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or (args.max_episodes is not None and args.max_episodes <= 0):
        parser.error("batch size and max episodes must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
