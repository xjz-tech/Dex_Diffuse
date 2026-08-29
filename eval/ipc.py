"""Small NumPy-over-UNIX-socket protocol shared by both eval processes.

The simulator runs under Python 3.8 (Isaac Gym) while Diffusion Policy runs
under Python 3.10.  Keeping the wire format to JSON plus raw float32 arrays
avoids sharing Python or PyTorch objects across those environments.
"""

import json
import socket
import struct

import numpy as np


_HEADER_SIZE = struct.Struct("!I")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_ARRAY_BYTES = 1024 * 1024 * 1024


def _recv_exact(sock, size):
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the inference socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, message, array=None):
    """Send one metadata object and an optional contiguous NumPy array."""
    metadata = dict(message)
    payload = b""
    if array is not None:
        array = np.ascontiguousarray(array)
        payload = memoryview(array).cast("B")
        metadata["array"] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": array.nbytes,
        }
    else:
        metadata["array"] = None

    encoded = json.dumps(
        metadata,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("IPC JSON header is too large")
    if len(payload) > _MAX_ARRAY_BYTES:
        raise ValueError("IPC array payload is too large")

    sock.sendall(_HEADER_SIZE.pack(len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def recv_message(sock):
    """Receive one metadata object and its optional NumPy array."""
    header_size = _HEADER_SIZE.unpack(_recv_exact(sock, _HEADER_SIZE.size))[0]
    if header_size > _MAX_JSON_BYTES:
        raise ValueError("IPC JSON header exceeds the safety limit")
    metadata = json.loads(_recv_exact(sock, header_size).decode("utf-8"))

    array_meta = metadata.pop("array", None)
    array = None
    if array_meta is not None:
        nbytes = int(array_meta["nbytes"])
        if nbytes < 0 or nbytes > _MAX_ARRAY_BYTES:
            raise ValueError("IPC array payload exceeds the safety limit")
        dtype = np.dtype(array_meta["dtype"])
        shape = tuple(int(value) for value in array_meta["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != nbytes:
            raise ValueError(
                "IPC array metadata is inconsistent: "
                "shape/dtype imply %d bytes, header says %d" % (expected, nbytes)
            )
        payload = _recv_exact(sock, nbytes)
        # Copy so callers receive writable, independently owned memory.
        array = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    return metadata, array


def connect_unix(path, timeout_seconds=120.0):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(float(timeout_seconds))
    sock.connect(str(path))
    return sock

