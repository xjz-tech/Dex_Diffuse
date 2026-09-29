"""Resample recorded absolute joint targets while preserving all originals."""

import numpy as np


def interpolate_actions(actions, inserted):
    actions = np.asarray(actions)
    inserted = int(inserted)
    if actions.ndim != 3 or actions.shape[-1] != 22:
        raise ValueError('expected (episodes, actions, 22)')
    if inserted < 0:
        raise ValueError('inserted must be nonnegative')
    if inserted == 0:
        return actions.copy()
    t = np.arange((actions.shape[1]-1)*(inserted+1)+1, dtype=np.float64)/(inserted+1)
    lo = np.minimum(np.floor(t).astype(int), actions.shape[1]-1)
    hi = np.minimum(lo+1, actions.shape[1]-1)
    frac = (t-lo)[None, :, None]
    return ((1-frac)*actions[:,lo]+frac*actions[:,hi]).astype(actions.dtype)


def interpolate_large_jumps(actions, threshold):
    """Insert one midpoint only where any joint changes by more than threshold.

    Returns expanded absolute targets and their 1-based fractional source-action
    progress. Adaptive timing is defined for one recorded episode at a time.
    """
    actions = np.asarray(actions)
    threshold = float(threshold)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 22:
        raise ValueError('expected one episode of shape (1, actions, 22)')
    if actions.shape[1] < 1 or not np.isfinite(actions).all():
        raise ValueError('expected nonempty finite actions')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be positive and finite')
    source = actions[0]
    # Compare the exact stored float32 targets in float64. NumPy 1.x and 2.x
    # otherwise cast a Python threshold differently at values such as 0.09,
    # which can desynchronize the simulator and the policy server.
    source64 = source.astype(np.float64)
    expanded = [source[0]]
    progress = [1.0]
    for i in range(len(source) - 1):
        if np.max(np.abs(source64[i + 1] - source64[i])) > threshold:
            expanded.append(((source64[i] + source64[i + 1]) / 2).astype(actions.dtype))
            progress.append(i + 1.5)
        expanded.append(source[i + 1])
        progress.append(float(i + 2))
    return (np.asarray(expanded, dtype=actions.dtype)[None],
            np.asarray(progress, dtype=np.float64))


def interpolate_equal_jumps(actions, jump, atol=1e-6):
    """Insert one midpoint where the maximum joint jump equals ``jump``.

    The source targets are stored as float32, so mathematical 0.18 rad jumps
    may differ from 0.18 by several 1e-8 after subtraction. The explicit
    absolute tolerance only handles this representation error; larger jumps
    are not selected.
    """
    actions = np.asarray(actions)
    jump, atol = float(jump), float(atol)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 22:
        raise ValueError('expected one episode of shape (1, actions, 22)')
    if actions.shape[1] < 1 or not np.isfinite(actions).all():
        raise ValueError('expected nonempty finite actions')
    if not np.isfinite(jump) or jump <= 0 or not np.isfinite(atol) or atol < 0:
        raise ValueError('jump must be positive and atol nonnegative')
    source = actions[0]
    source64 = source.astype(np.float64)
    expanded, progress = [source[0]], [1.0]
    for i in range(len(source) - 1):
        maximum = np.max(np.abs(source64[i + 1] - source64[i]))
        if abs(maximum - jump) <= atol:
            expanded.append(((source64[i] + source64[i + 1]) / 2).astype(actions.dtype))
            progress.append(i + 1.5)
        expanded.append(source[i + 1])
        progress.append(float(i + 2))
    return (np.asarray(expanded, dtype=actions.dtype)[None],
            np.asarray(progress, dtype=np.float64))
