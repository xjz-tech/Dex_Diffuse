"""Versioned JSON + float32 NumPy messages over ZeroMQ.

The simulator uses Python 3.8 and the policy server uses Python 3.10.  Keeping
the protocol to JSON metadata and raw little-endian float32 frames avoids
sharing Python or PyTorch objects between the two environments.
"""

from __future__ import annotations

import json

import numpy as np


PROTOCOL_VERSION = 1
_FLOAT32_LE = np.dtype("<f4")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_ARRAY_BYTES = 1024 * 1024 * 1024


def endpoint_for_path(path):
    text = str(path)
    if "://" in text:
        return text
    if not text.startswith("/"):
        raise ValueError("ZeroMQ IPC socket path must be absolute: %s" % text)
    return "ipc://" + text


def _encode(message, array):
    metadata = dict(message)
    metadata["protocol_version"] = PROTOCOL_VERSION
    frames = []
    if array is None:
        metadata["array"] = None
    else:
        array = np.asarray(array, dtype=_FLOAT32_LE, order="C")
        if array.nbytes > _MAX_ARRAY_BYTES:
            raise ValueError("IPC array payload exceeds the safety limit")
        metadata["array"] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
        }
        frames.append(array.tobytes(order="C"))

    encoded = json.dumps(
        metadata,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("IPC JSON header is too large")
    return [encoded] + frames


def _decode(frames):
    if len(frames) not in (1, 2):
        raise ValueError("invalid ZeroMQ message frame count: %d" % len(frames))
    if len(frames[0]) > _MAX_JSON_BYTES:
        raise ValueError("IPC JSON header exceeds the safety limit")
    metadata = json.loads(frames[0].decode("utf-8"))
    if int(metadata.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            "unsupported IPC protocol version: %r"
            % metadata.get("protocol_version")
        )

    array_meta = metadata.pop("array", None)
    if array_meta is None:
        if len(frames) != 1:
            raise ValueError("message has an unexpected array frame")
        return metadata, None
    if len(frames) != 2:
        raise ValueError("message array frame is missing")

    dtype = np.dtype(array_meta.get("dtype"))
    if dtype != _FLOAT32_LE:
        raise TypeError("IPC array must be little-endian float32, got %s" % dtype)
    shape = tuple(int(value) for value in array_meta.get("shape", ()))
    if any(value < 0 for value in shape):
        raise ValueError("IPC array shape contains a negative dimension")
    nbytes = int(array_meta.get("nbytes", -1))
    if nbytes < 0 or nbytes > _MAX_ARRAY_BYTES:
        raise ValueError("IPC array payload exceeds the safety limit")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected != nbytes or len(frames[1]) != nbytes:
        raise ValueError(
            "IPC array metadata is inconsistent: expected %d bytes, got %d"
            % (expected, len(frames[1]))
        )
    array = np.frombuffer(frames[1], dtype=dtype).reshape(shape).copy()
    return metadata, array


def send_dealer(socket, message, array=None, flags=0):
    socket.send_multipart(_encode(message, array), flags=flags, copy=True)


def recv_dealer(socket, flags=0):
    return _decode(socket.recv_multipart(flags=flags, copy=True))


def send_router(socket, identity, message, array=None, flags=0):
    socket.send_multipart(
        [identity] + _encode(message, array),
        flags=flags,
        copy=True,
    )


def recv_router(socket, flags=0):
    frames = socket.recv_multipart(flags=flags, copy=True)
    if len(frames) < 2:
        raise ValueError("ROUTER message has no payload")
    metadata, array = _decode(frames[1:])
    return frames[0], metadata, array
