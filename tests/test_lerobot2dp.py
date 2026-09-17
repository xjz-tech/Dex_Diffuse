"""Conversion contracts: RGB mapping, boundaries, source values and safe failure."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import zarr

from scripts.lerobot2dp import convert, parse_args


def write_episode(dataset, length=18, index=0, marker=2.0, bad_image=False, bad_state=False):
    (dataset / "meta").mkdir(parents=True, exist_ok=True)
    path = dataset / f"data/chunk-000/episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.full((length, 31), marker, dtype=np.float32)
    state[:, 3:9] = [1, 0, 0, 0, 1, 0]
    action = state.copy()
    action[:, :3] += 0.1
    action[:, 9:] += 0.2
    tactile = np.arange(length * 30, dtype=np.float32).reshape(length, 5, 6)
    if bad_state:
        state[2, 0] = np.nan
    def png(color):
        buff = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(buff, format="PNG")
        return buff.getvalue()
    front = b"broken PNG" if bad_image else png((220, 30, 10))
    wrist = png((10, 40, 210))
    table = pa.table({
        "state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 31)),
        "actions": pa.array(action.tolist(), type=pa.list_(pa.float32(), 31)),
        "tactile_f6": pa.array(tactile.tolist(), type=pa.list_(pa.list_(pa.float32()))),
        "image": [{"bytes": front, "path": "nonexistent.png"}] * length,
        "extra_view_image": [{"bytes": wrist, "path": "nonexistent.png"}] * length,
        "frame_index": np.arange(length), "episode_index": np.full(length, index),
        "timestamp": (np.arange(length) / 30).astype(np.float32),
        "done": [False] * (length - 1) + [True],
    })
    pq.write_table(table, path, row_group_size=7)
    (dataset / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "total_episodes": 1, "total_frames": length,
        "fps": 30, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }))
    (dataset / "meta/episodes.jsonl").write_text(json.dumps({
        "episode_index": index, "length": length, "tasks": ["screw the bulb"],
    }) + "\n")
    return state, action, tactile


class LeRobotConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "output"

    def args(self, *extra):
        return parse_args(["--src", str(self.source), "--out", str(self.output),
                           "--width", "4", "--height", "3", "--image-batch-size", "5", *extra])

    def test_rgb_values_numeric_order_episode_boundaries_and_skip(self):
        later = write_episode(self.source / "rank_0/id_10", length=19, marker=10)
        first = write_episode(self.source / "rank_0/id_2", length=18, marker=2)
        empty = self.source / "rank_0/id_3/meta"
        empty.mkdir(parents=True)
        (empty / "info.json").write_text('{"total_episodes": 0, "total_frames": 0}')
        convert(self.args())
        z = zarr.open_group(str(self.output / "replay_buffer.zarr"), mode="r")
        np.testing.assert_array_equal(z["meta/episode_ends"][:], [18, 37])
        for key, column in [("state", 0), ("action", 1), ("tactile_f6", 2)]:
            np.testing.assert_array_equal(z[f"data/{key}"][:], np.concatenate([first[column], later[column]]))
        np.testing.assert_array_equal(z["data/tactile_force"][:], z["data/tactile_f6"][:].reshape(37, 30))
        self.assertEqual(z["data/front_image"].shape, (37, 3, 4, 3))
        self.assertEqual(z["data/front_image"].dtype, np.dtype('uint8'))
        self.assertTrue(np.all(z["data/front_image"][:] == [220, 30, 10]))
        self.assertTrue(np.all(z["data/wrist_image"][:] == [10, 40, 210]))
        manifest = json.loads((self.output / "conversion_manifest.json").read_text())
        self.assertEqual(manifest["skipped_datasets"][0]["dataset"], "rank_0/id_3")
        self.assertEqual([r["source_dataset"] for r in manifest["episodes"]], ["rank_0/id_2", "rank_0/id_10"])

    def test_committed_numeric_corruption_fails_without_output(self):
        write_episode(self.source / "rank_0/id_0", bad_state=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            convert(self.args())
        self.assertFalse(self.output.exists())

    def test_late_decode_failure_removes_partial_output(self):
        write_episode(self.source / "rank_0/id_0")
        write_episode(self.source / "rank_0/id_1", bad_image=True)
        with self.assertRaisesRegex(ValueError, "decode failed"):
            convert(self.args())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.output.converting-*')), [])

    def test_existing_output_and_source_are_never_overwritten(self):
        write_episode(self.source / "rank_0/id_0")
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            convert(self.args())
        self.assertEqual(marker.read_text(), 'keep')
        with self.assertRaisesRegex(ValueError, "separate directories"):
            convert(self.args("--out", str(self.source / "converted")))

    def test_dry_run_and_single_dataset_nonzero_episode_index(self):
        write_episode(self.source, index=4)
        self.assertIsNone(convert(self.args("--dry-run")))
        self.assertFalse(self.output.exists())
        convert(self.args())
        manifest = json.loads((self.output / "conversion_manifest.json").read_text())
        self.assertEqual(manifest["episodes"][0]["source_episode_index"], 4)

    def test_missing_committed_parquet_is_not_silently_skipped(self):
        write_episode(self.source)
        next(self.source.glob('data/**/*.parquet')).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing committed"):
            convert(self.args())

    def test_training_dataset_reads_absolute_and_relative_windows(self):
        from diffusion_policy.dataset.bulb_image_dataset import BulbImageDataset

        write_episode(self.source / 'rank_0/id_0', marker=2)
        write_episode(self.source / 'rank_0/id_1', marker=10)
        convert(self.args())
        for relative in (False, True):
            dataset = BulbImageDataset(str(self.output), horizon=16, n_obs_steps=1,
                                       pad_before=0, pad_after=7, val_ratio=0, relative=relative)
            for sample_index in range(len(dataset)):
                sample = dataset[sample_index]
                self.assertEqual(tuple(sample['obs']['front_image'].shape), (1, 3, 3, 4))
                self.assertEqual(tuple(sample['action'].shape), (16, 31))
                hand = sample['action'][:, 9:].numpy()
                # Each action window stays inside one episode, including padding.
                self.assertTrue(np.allclose(hand, 2.2) or np.allclose(hand, 10.2))
                if relative:
                    np.testing.assert_allclose(sample['obs']['ee_pose'][0, :3], 0, atol=1e-6)
                    np.testing.assert_allclose(sample['action'][:, :3], .1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
