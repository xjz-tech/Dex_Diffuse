"""Load the Sim-Hand Diffusion Policy checkpoint used by this evaluator.

Normal checkpoints are loaded with ``torch.load`` and use EMA weights.  The
provided step_01700000.ckpt was interrupted/corrupted after its base model was
fully written but before its EMA/optimizer and ZIP directory were complete.
For that specific failure mode this module can recover only the fully intact
base-model records.  Every recovered record is size-checked and the resulting
policy is loaded strictly; partial EMA weights are never mixed in.
"""

from __future__ import annotations

import io
import mmap
import pickle
import re
import struct
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import dill
import torch


_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_DATA_DESCRIPTOR = struct.Struct("<IIII")
_LOCAL_SIGNATURE = 0x04034B50
_DESCRIPTOR_SIGNATURE = 0x08074B50
_STORAGE_NAME = re.compile(r"archive/data/(\d+)\Z")
_STORAGE_DTYPES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "HalfStorage": torch.float16,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


@dataclass(frozen=True)
class StorageRef:
    key: str
    dtype: object
    location: str
    numel: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    requires_grad: bool


@dataclass(frozen=True)
class LocalRecord:
    header_offset: int
    data_offset: int
    flags: int
    compression: int
    name: str


@dataclass
class LoadedCheckpoint:
    cfg: object
    state_dict: Mapping[str, torch.Tensor]
    weight_source: str
    global_step: Optional[int]
    epoch: Optional[int]
    salvaged: bool


def _rebuild_tensor_ref(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad=False,
    backward_hooks=None,
    *unused,
):
    del backward_hooks, unused
    return TensorRef(
        storage=storage,
        offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
        requires_grad=bool(requires_grad),
    )


def _rebuild_parameter_ref(data, *unused):
    del unused
    return data


class _MetadataUnpickler(pickle.Unpickler):
    """Unpickle metadata without materializing any tensor storage."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in (
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        ):
            return _rebuild_tensor_ref
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return _rebuild_parameter_ref
        return super().find_class(module, name)

    def persistent_load(self, persistent_id):
        if not isinstance(persistent_id, tuple) or len(persistent_id) != 5:
            raise pickle.UnpicklingError(
                "unexpected persistent id in checkpoint metadata: %r"
                % (persistent_id,)
            )
        tag, storage_type, key, location, numel = persistent_id
        if tag != "storage":
            raise pickle.UnpicklingError(
                "unsupported checkpoint persistent id: %r" % (persistent_id,)
            )
        dtype = _STORAGE_DTYPES.get(getattr(storage_type, "__name__", ""))
        return StorageRef(
            key=str(key),
            dtype=dtype,
            location=str(location),
            numel=int(numel),
        )


def _parse_local_header(buffer, offset: int) -> LocalRecord:
    if offset < 0 or offset + _LOCAL_HEADER.size > len(buffer):
        raise ValueError("truncated ZIP local header at byte %d" % offset)
    fields = _LOCAL_HEADER.unpack_from(buffer, offset)
    (
        signature,
        _version,
        flags,
        compression,
        _mtime,
        _mdate,
        _crc,
        _compressed_size,
        _uncompressed_size,
        name_length,
        extra_length,
    ) = fields
    if signature != _LOCAL_SIGNATURE:
        raise ValueError("invalid ZIP local signature at byte %d" % offset)
    name_start = offset + _LOCAL_HEADER.size
    name_end = name_start + name_length
    data_offset = name_end + extra_length
    if data_offset > len(buffer):
        raise ValueError("truncated ZIP local record at byte %d" % offset)
    try:
        name = bytes(buffer[name_start:name_end]).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid ZIP record name at byte %d" % offset) from exc
    return LocalRecord(
        header_offset=offset,
        data_offset=data_offset,
        flags=int(flags),
        compression=int(compression),
        name=name,
    )


def _find_named_header(buffer, start: int, expected_name: str) -> LocalRecord:
    signature = b"PK\x03\x04"
    offset = int(start)
    while True:
        offset = buffer.find(signature, offset)
        if offset < 0:
            raise ValueError("checkpoint record not found: %s" % expected_name)
        try:
            record = _parse_local_header(buffer, offset)
        except ValueError:
            offset += len(signature)
            continue
        if record.name == expected_name:
            return record
        offset += len(signature)


def _read_incomplete_payload_metadata(buffer):
    first = _parse_local_header(buffer, 0)
    if first.name != "archive/data.pkl" or first.compression != 0:
        raise ValueError(
            "unsupported damaged checkpoint layout: first record is %r "
            "with compression=%d" % (first.name, first.compression)
        )
    next_record = _find_named_header(
        buffer,
        first.data_offset,
        "archive/.format_version",
    )
    pickle_end = next_record.header_offset
    if first.flags & 0x0008:
        if pickle_end < _DATA_DESCRIPTOR.size:
            raise ValueError("truncated data.pkl descriptor")
        descriptor_offset = pickle_end - _DATA_DESCRIPTOR.size
        signature, _crc, compressed_size, uncompressed_size = (
            _DATA_DESCRIPTOR.unpack_from(buffer, descriptor_offset)
        )
        if signature != _DESCRIPTOR_SIGNATURE:
            raise ValueError("data.pkl is missing its ZIP data descriptor")
        if compressed_size != uncompressed_size:
            raise ValueError("compressed data.pkl salvage is unsupported")
        pickle_end = descriptor_offset
        actual_size = pickle_end - first.data_offset
        if uncompressed_size != actual_size:
            raise ValueError(
                "data.pkl size mismatch: descriptor=%d, actual=%d"
                % (uncompressed_size, actual_size)
            )
    payload = _MetadataUnpickler(
        io.BytesIO(bytes(buffer[first.data_offset:pickle_end]))
    ).load()
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload metadata is not a dictionary")
    return payload


def _scan_storage_records(buffer, required_keys):
    required = {str(key) for key in required_keys}
    found: Dict[str, LocalRecord] = {}
    signature = b"PK\x03\x04"
    offset = 0
    while required - set(found):
        offset = buffer.find(signature, offset)
        if offset < 0:
            break
        try:
            record = _parse_local_header(buffer, offset)
        except ValueError:
            offset += len(signature)
            continue
        match = _STORAGE_NAME.fullmatch(record.name)
        if match is not None and match.group(1) in required:
            key = match.group(1)
            if key in found:
                raise ValueError("duplicate checkpoint storage record %s" % key)
            if record.compression != 0:
                raise ValueError("compressed tensor storage is unsupported")
            found[key] = record
        offset += len(signature)
    missing = sorted(required - set(found), key=int)
    if missing:
        raise ValueError(
            "damaged checkpoint is missing base-model storages: %s"
            % ", ".join(missing[:20])
        )
    return found


def _validate_storage_record(buffer, storage: StorageRef, record: LocalRecord):
    if storage.dtype != torch.float32:
        raise ValueError(
            "unsupported salvage dtype for storage %s: %s"
            % (storage.key, storage.dtype)
        )
    if storage.numel < 0:
        raise ValueError("negative numel for storage %s" % storage.key)
    nbytes = storage.numel * torch.tensor([], dtype=storage.dtype).element_size()
    data_end = record.data_offset + nbytes
    if data_end > len(buffer):
        raise ValueError(
            "checkpoint storage %s is truncated (%d bytes required)"
            % (storage.key, nbytes)
        )
    if record.flags & 0x0008:
        descriptor_end = data_end + _DATA_DESCRIPTOR.size
        if descriptor_end > len(buffer):
            raise ValueError(
                "checkpoint storage %s has a truncated descriptor" % storage.key
            )
        signature, _crc, compressed_size, uncompressed_size = (
            _DATA_DESCRIPTOR.unpack_from(buffer, data_end)
        )
        if signature != _DESCRIPTOR_SIGNATURE:
            raise ValueError(
                "checkpoint storage %s failed descriptor validation" % storage.key
            )
        if compressed_size != nbytes or uncompressed_size != nbytes:
            raise ValueError(
                "checkpoint storage %s size mismatch: expected %d, "
                "descriptor=(%d,%d)"
                % (storage.key, nbytes, compressed_size, uncompressed_size)
            )
    elif nbytes != 0:
        raise ValueError(
            "non-empty checkpoint storage %s has no data descriptor" % storage.key
        )


def _recover_model_state(buffer, model_metadata):
    if not isinstance(model_metadata, (dict, OrderedDict)):
        raise ValueError("checkpoint model state metadata is not a mapping")

    refs = []
    for name, value in model_metadata.items():
        if not isinstance(value, TensorRef):
            raise ValueError(
                "unsupported non-tensor model state entry %r: %s"
                % (name, type(value).__name__)
            )
        refs.append(value)
    if not refs:
        raise ValueError("checkpoint model state is empty")

    storages = {ref.storage.key: ref.storage for ref in refs}
    numeric_keys = sorted(int(key) for key in storages)
    # This is a safety boundary, not merely a fact about the current file: EMA
    # begins at 162.  Never salvage beyond the independently complete model.
    if numeric_keys[0] != 0 or numeric_keys[-1] > 161:
        raise ValueError(
            "unexpected base-model storage range %d..%d; refusing salvage"
            % (numeric_keys[0], numeric_keys[-1])
        )
    records = _scan_storage_records(buffer, storages)
    for key, storage in storages.items():
        _validate_storage_record(buffer, storage, records[key])

    base_tensors = {}
    recovered = OrderedDict()
    for name, ref in model_metadata.items():
        key = ref.storage.key
        if key not in base_tensors:
            if ref.storage.numel == 0:
                base = torch.empty(0, dtype=ref.storage.dtype)
            else:
                source = torch.frombuffer(
                    buffer,
                    dtype=ref.storage.dtype,
                    count=ref.storage.numel,
                    offset=records[key].data_offset,
                )
                # Own the bytes so the mmap can be closed after this function.
                base = source.clone()
            base_tensors[key] = base
        base = base_tensors[key]
        tensor = torch.as_strided(
            base,
            size=ref.size,
            stride=ref.stride,
            storage_offset=ref.offset,
        )
        if ref.requires_grad:
            tensor = tensor.requires_grad_(True)
        recovered[name] = tensor
    return recovered


def _decode_pickled_scalar(payload, name):
    encoded = payload.get("pickles", {}).get(name)
    if encoded is None:
        return None
    value = dill.loads(encoded)
    return int(value) if value is not None else None


def _salvage_incomplete_checkpoint(path: Path) -> LoadedCheckpoint:
    with path.open("rb") as stream:
        # ACCESS_COPY is a private, writable mapping.  ``torch.frombuffer`` can
        # view it without warning, while the checkpoint file remains read-only.
        buffer = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            payload = _read_incomplete_payload_metadata(buffer)
            state_dicts = payload.get("state_dicts")
            if not isinstance(state_dicts, dict) or "model" not in state_dicts:
                raise ValueError("damaged checkpoint has no base model state")
            state_dict = _recover_model_state(buffer, state_dicts["model"])
        finally:
            buffer.close()

    return LoadedCheckpoint(
        cfg=payload.get("cfg"),
        state_dict=state_dict,
        weight_source="base model (strictly salvaged; EMA is incomplete)",
        global_step=_decode_pickled_scalar(payload, "global_step"),
        epoch=_decode_pickled_scalar(payload, "epoch"),
        salvaged=True,
    )


def load_checkpoint(path, allow_salvage=True) -> LoadedCheckpoint:
    """Load a complete checkpoint, or safely salvage its intact base model."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)
    try:
        payload = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            pickle_module=dill,
            weights_only=False,
        )
    except RuntimeError as exc:
        if not allow_salvage or "failed finding central directory" not in str(exc):
            raise
        warnings.warn(
            "Checkpoint ZIP is incomplete. Recovering only the fully written "
            "base model; the damaged EMA and optimizer records will not be used.",
            RuntimeWarning,
        )
        return _salvage_incomplete_checkpoint(checkpoint_path)

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, dict) or "model" not in state_dicts:
        raise ValueError("checkpoint does not contain a model state")
    cfg = payload.get("cfg")
    use_ema = bool(getattr(getattr(cfg, "training", None), "use_ema", False))
    if use_ema:
        if "ema_model" not in state_dicts:
            raise ValueError("complete checkpoint requests EMA but has no EMA state")
        state_dict = state_dicts["ema_model"]
        weight_source = "EMA model"
    else:
        state_dict = state_dicts["model"]
        weight_source = "base model"
    return LoadedCheckpoint(
        cfg=cfg,
        state_dict=state_dict,
        weight_source=weight_source,
        global_step=_decode_pickled_scalar(payload, "global_step"),
        epoch=_decode_pickled_scalar(payload, "epoch"),
        salvaged=False,
    )


def _identity_field_normalizer(dim):
    from diffusion_policy.model.common.normalizer import (
        SingleFieldLinearNormalizer,
    )

    ones = torch.ones(dim, dtype=torch.float32)
    zeros = torch.zeros(dim, dtype=torch.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=ones,
        offset=zeros,
        input_stats_dict={
            "min": -ones,
            "max": ones,
            "mean": zeros,
            "std": ones,
        },
    )


def build_policy(loaded: LoadedCheckpoint):
    """Construct the main-branch low-dim policy equivalent to Sim-Hand DP."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from omegaconf import OmegaConf

    from diffusion_policy.model.diffusion.conditional_unet1d import (
        ConditionalUnet1D,
    )
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
        DiffusionUnetLowdimPolicy,
    )

    cfg = loaded.cfg
    if cfg is None:
        raise ValueError("checkpoint has no embedded configuration")
    obs_dim = int(cfg["obs_dim"])
    if obs_dim not in (22, 66):
        raise ValueError(
            "unsupported Sim-Hand checkpoint: obs_dim=%d, expected 22 or 66"
            % obs_dim
        )
    expected = {
        "obs_dim": obs_dim,
        "action_dim": 22,
        "n_obs_steps": 4,
        "n_pred_action_steps": 9,
        "n_action_steps": 5,
        "horizon": 12,
    }
    for key, value in expected.items():
        actual = int(cfg[key])
        if actual != value:
            raise ValueError(
                "unsupported Sim-Hand checkpoint: %s=%d, expected %d"
                % (key, actual, value)
            )
    if not bool(cfg.obs_as_global_cond) or not bool(cfg.oa_step_convention):
        raise ValueError(
            "Sim-Hand checkpoint must use global observation conditioning "
            "and OA-step convention"
        )

    model_kwargs = OmegaConf.to_container(cfg.policy.model, resolve=True)
    model_kwargs.pop("_target_", None)
    scheduler_kwargs = OmegaConf.to_container(
        cfg.policy.noise_scheduler,
        resolve=True,
    )
    scheduler_kwargs.pop("_target_", None)
    model = ConditionalUnet1D(**model_kwargs)
    scheduler = DDPMScheduler(**scheduler_kwargs)
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=expected["horizon"],
        obs_dim=expected["obs_dim"],
        action_dim=expected["action_dim"],
        n_action_steps=expected["n_action_steps"],
        n_obs_steps=expected["n_obs_steps"],
        num_inference_steps=int(cfg.policy.num_inference_steps),
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=True,
    )
    # LinearNormalizer starts empty; create its expected parameter tree before
    # strict-loading the checkpoint's learned statistics.
    policy.normalizer["obs"] = _identity_field_normalizer(expected["obs_dim"])
    policy.normalizer["action"] = _identity_field_normalizer(
        expected["action_dim"]
    )
    incompatible = policy.load_state_dict(loaded.state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load unexpectedly returned key errors")
    return policy, expected


def configure_policy_execution_steps(policy, spec, n_action_steps):
    """Execute a prefix of the predicted action window, then replan.

    The checkpoint still generates the full horizon. ``n_action_steps`` only
    changes how many of those predicted steps the closed loop applies.
    """
    from policy_observation import expected_policy_spec

    updated = expected_policy_spec(spec["obs_dim"], n_action_steps)
    layout_keys = ("obs_dim", "action_dim", "n_obs_steps", "n_pred_action_steps", "horizon")
    for key in layout_keys:
        if int(spec[key]) != int(updated[key]):
            raise ValueError(
                "cannot override n_action_steps: %s=%s, expected %s"
                % (key, spec[key], updated[key])
            )
    policy.n_action_steps = updated["n_action_steps"]
    spec["n_action_steps"] = updated["n_action_steps"]
    return spec


def configure_policy_sampler(policy, sampler, inference_steps=None):
    """Switch the loaded policy onto DDPM or DDIM and optionally override steps."""
    sampler = str(sampler)
    if sampler == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        policy.noise_scheduler = DDIMScheduler.from_config(
            policy.noise_scheduler.config,
            set_alpha_to_one=True,
            steps_offset=0,
            timestep_spacing="leading",
        )
    elif sampler != "ddpm":
        raise ValueError("unsupported sampler: %r" % (sampler,))
    if inference_steps is not None:
        if int(inference_steps) <= 0:
            raise ValueError("inference_steps must be positive")
        policy.num_inference_steps = int(inference_steps)
    return policy
