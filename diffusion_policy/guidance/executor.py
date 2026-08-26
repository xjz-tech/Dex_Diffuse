"""Segment executors for closed-loop guided inference.

FakeSegmentExecutor treats commanded hand actions as post-execution states
(command-is-state): ``execute(segment)`` returns ``segment[..., 9:31]`` as
``(1, E, 22)`` post_states. This is sufficient for hardware-free dry-runs.
"""

from __future__ import annotations

from typing import Protocol

import torch

HAND_SLICE = slice(9, 31)


class ExecutionError(RuntimeError):
    """Raised when a segment fails partially or completely during execution."""

    def __init__(
        self,
        message: str | None = None,
        *,
        segment_id: int,
        executed_count: int,
        partial_states: torch.Tensor | None = None,
    ) -> None:
        if message is None:
            message = (
                f"Segment {segment_id} failed after executing "
                f"{executed_count} step(s)"
            )
        super().__init__(message)
        self.segment_id = segment_id
        self.executed_count = executed_count
        self.partial_states = partial_states


class SegmentExecutor(Protocol):
    def reset(self) -> torch.Tensor:
        """Return initial hand-state history ``(1, n_obs_steps, 22)``."""

    def execute(
        self,
        segment: torch.Tensor,
        *,
        segment_id: int,
    ) -> torch.Tensor:
        """Execute a guided segment ``(1, E, 31)`` and return post states ``(1, E, 22)``."""


class FakeSegmentExecutor:
    """Deterministic fake that treats commanded hand as the resulting state."""

    def __init__(self, initial_history: torch.Tensor) -> None:
        if initial_history.ndim != 3 or initial_history.shape[0] != 1:
            raise ValueError(
                "initial_history must have shape (1, n_obs_steps, 22), "
                f"got {tuple(initial_history.shape)}"
            )
        if initial_history.shape[-1] != 22:
            raise ValueError(
                "initial_history hand dim must be 22, "
                f"got {initial_history.shape[-1]}"
            )
        if not torch.isfinite(initial_history).all():
            raise ValueError("initial_history is non-finite")
        self._initial_history = initial_history.detach().clone()

    def reset(self) -> torch.Tensor:
        # Shape contract: (1, n_obs_steps, 22).
        return self._initial_history.clone()

    def execute(
        self,
        segment: torch.Tensor,
        *,
        segment_id: int,
    ) -> torch.Tensor:
        del segment_id  # unused; present for Protocol parity
        if segment.ndim != 3 or segment.shape[0] != 1 or segment.shape[-1] != 31:
            raise ValueError(
                f"segment must have shape (1, E, 31), got {tuple(segment.shape)}"
            )
        if not torch.isfinite(segment).all():
            raise ValueError("segment is non-finite")
        # Command-is-state: commanded hand [9:31] is treated as post state.
        return segment[..., HAND_SLICE].clone()
