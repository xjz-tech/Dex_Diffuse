from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import h5py
import numpy as np

from sim_data_schema import SCHEMA_VERSION, SimDataSchema, validate_transition_batch


class H5SimDataWriter:
    def __init__(
        self,
        output_dir: str | Path,
        metadata: Mapping[str, object],
        dof: int,
        shard_transition_limit: int = 1_000_000,
        shard_size_limit_bytes: int = 4 * 1024**3,
        chunk_size: int = 8192,
        compression: Optional[str] = "lzf",
    ):
        self.output_dir = Path(output_dir)
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.metadata = dict(metadata)
        self.schema = SimDataSchema(dof=dof)
        self.shard_transition_limit = int(shard_transition_limit)
        self.shard_size_limit_bytes = int(shard_size_limit_bytes)
        self.chunk_size = int(chunk_size)
        self.compression = compression
        self._file: Optional[h5py.File] = None
        self._datasets: Dict[str, h5py.Dataset] = {}
        self._shard_index = 0
        self._shard_rows = 0
        self._total_rows = 0
        self._shards: List[Dict[str, object]] = []

    def __enter__(self) -> "H5SimDataWriter":
        self._open_next_shard()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def append(self, batch: Mapping[str, np.ndarray]) -> None:
        num_rows = validate_transition_batch(batch, dof=self.schema.dof)
        start = 0
        while start < num_rows:
            if self._file is None:
                self._open_next_shard()
            remaining = self.shard_transition_limit - self._shard_rows
            take = min(num_rows - start, remaining)
            if take <= 0:
                self._open_next_shard()
                continue
            self._append_slice(batch, start, start + take)
            start += take
            if self._should_rotate():
                self._open_next_shard()

    def _append_slice(self, batch: Mapping[str, np.ndarray], start: int, stop: int) -> None:
        old_size = self._shard_rows
        new_size = old_size + stop - start
        for path, dataset in self._datasets.items():
            spec = self.schema.datasets[path]
            dataset.resize((new_size,) + tuple(spec["shape"]))
            dataset[old_size:new_size] = np.asarray(
                batch[path][start:stop], dtype=spec["dtype"]
            )
        self._shard_rows = new_size
        self._total_rows += stop - start

    def _should_rotate(self) -> bool:
        if self._file is None:
            return False
        if self._shard_rows >= self.shard_transition_limit:
            return True
        try:
            self._file.flush()
            return Path(self._file.filename).stat().st_size >= self.shard_size_limit_bytes
        except OSError:
            return False

    def _open_next_shard(self) -> None:
        if self._file is not None:
            self._close_current_shard()
            self._shard_index += 1
        shard_path = self.shard_dir / f"shard_{self._shard_index:06d}.h5"
        self._file = h5py.File(shard_path, "w")
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["created_time"] = time.time()
        self._file.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
        for group_name in ("index", "robot", "object", "target", "policy", "fingertip", "outcome"):
            self._file.require_group(group_name)
        self._datasets = {}
        for path, spec in self.schema.datasets.items():
            shape_tail = tuple(spec["shape"])
            chunks = (min(self.chunk_size, self.shard_transition_limit),) + shape_tail
            self._datasets[path] = self._file.create_dataset(
                path,
                shape=(0,) + shape_tail,
                maxshape=(None,) + shape_tail,
                chunks=chunks,
                dtype=spec["dtype"],
                compression=self.compression,
            )
        self._shard_rows = 0

    def _close_current_shard(self) -> None:
        self._file.flush()
        path = Path(self._file.filename)
        row_count = self._shard_rows
        self._file.close()
        if row_count == 0:
            path.unlink(missing_ok=True)
        else:
            self._shards.append(
                {
                    "path": str(path.relative_to(self.output_dir)),
                    "num_transitions": row_count,
                    "size_bytes": path.stat().st_size,
                }
            )
            self._write_manifest()
        self._file = None
        self._datasets = {}
        self._shard_rows = 0

    def close(self) -> None:
        if self._file is not None:
            self._close_current_shard()
        else:
            self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_time": self.metadata.get("created_time", time.time()),
            "total_transitions": self._total_rows,
            "dof": self.schema.dof,
            "metadata": self.metadata,
            "shards": self._shards,
        }
        temporary_path = self.manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_path.replace(self.manifest_path)
