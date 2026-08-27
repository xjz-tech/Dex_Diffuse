"""Dynamic closed-loop coordinator for Real proposal hand guidance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_policy.guidance.executor import ExecutionError, SegmentExecutor
from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

REAL_ACTION_DIM = 31


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: int
    start: int
    stop: int
    real_hand_reference: torch.Tensor
    guided_hand: torch.Tensor
    executed_action: torch.Tensor
    post_states: torch.Tensor
    history_before: torch.Tensor
    history_after: torch.Tensor


@dataclass(frozen=True)
class GuidedRunResult:
    proposal: torch.Tensor
    guided_action: torch.Tensor
    records: tuple[SegmentRecord, ...]
    final_history: torch.Tensor


class GuidedCoordinator:
    def __init__(
        self,
        guidance: SimHandGuidance,
        hand_slice: slice = slice(9, 31),
    ) -> None:
        self.guidance = guidance
        self.hand_slice = hand_slice

    def run(
        self,
        proposal: torch.Tensor,
        executor: SegmentExecutor,
        *,
        hand_reference: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> GuidedRunResult:
        if (
            proposal.ndim != 3
            or proposal.shape[0] != 1
            or proposal.shape[-1] != REAL_ACTION_DIM
        ):
            raise ValueError(
                f"Real proposal must have shape (1,T,31), got {tuple(proposal.shape)}"
            )
        if not torch.isfinite(proposal).all():
            raise ValueError("Real proposal is non-finite")

        execution_steps = int(self.guidance.config.execution_steps)
        total_steps = int(proposal.shape[1])
        if total_steps % execution_steps != 0:
            raise ValueError(
                f"Real proposal length {total_steps} is not divisible by "
                f"execution_steps {execution_steps}"
            )
        # Dynamic: never hardcode 50/10 — count comes from proposal length.
        segment_count = total_steps // execution_steps

        n_obs_steps = int(self.guidance.adapter.n_obs_steps)
        guidance_steps = int(self.guidance.adapter.n_pred_action_steps)
        hand_dim = _hand_dim(self.hand_slice)
        required_reference_steps = total_steps - execution_steps + guidance_steps
        _validate_hand_reference(
            hand_reference,
            required_steps=required_reference_steps,
            hand_dim=hand_dim,
            expected_device=proposal.device,
            expected_dtype=proposal.dtype,
        )

        guided_action = proposal.clone()
        history = executor.reset()
        _validate_history(history, n_obs_steps=n_obs_steps, hand_dim=hand_dim)

        records: list[SegmentRecord] = []
        for segment_id in range(segment_count):
            start = segment_id * execution_steps
            stop = start + execution_steps

            # Reference is a full Sim prediction horizon. Consecutive windows
            # overlap when guidance_steps > execution_steps.
            real_hand_reference = hand_reference[
                :, start : start + guidance_steps, :
            ].clone()
            history_before = history.clone()
            guided_hand = self.guidance.guide_segment(
                history_before,
                real_hand_reference,
                generator=generator,
            )
            _validate_guided_hand(
                guided_hand,
                execution_steps=execution_steps,
                hand_dim=hand_dim,
                expected_device=proposal.device,
                expected_dtype=proposal.dtype,
            )

            # Replace only Real [9:31]; keep [0:9] elementwise.
            guided_action[:, start:stop, self.hand_slice] = guided_hand
            executed_action = guided_action[:, start:stop, :].clone()

            try:
                post_states = executor.execute(
                    executed_action,
                    segment_id=segment_id,
                )
            except ExecutionError:
                # Stop immediately — do not fabricate history or continue.
                raise

            _validate_post_states(
                post_states,
                execution_steps=execution_steps,
                hand_dim=hand_dim,
            )

            history_after = torch.cat(
                (history_before, post_states),
                dim=1,
            )[:, -n_obs_steps:, :]
            history = history_after

            records.append(
                SegmentRecord(
                    segment_id=segment_id,
                    start=start,
                    stop=stop,
                    real_hand_reference=real_hand_reference,
                    guided_hand=guided_hand.detach().clone(),
                    executed_action=executed_action,
                    post_states=post_states.detach().clone(),
                    history_before=history_before,
                    history_after=history_after.clone(),
                )
            )

        return GuidedRunResult(
            proposal=proposal.clone(),
            guided_action=guided_action,
            records=tuple(records),
            final_history=history.clone(),
        )


def _hand_dim(hand_slice: slice) -> int:
    if hand_slice.start is None or hand_slice.stop is None:
        raise ValueError(f"hand_slice must have explicit start/stop, got {hand_slice}")
    return int(hand_slice.stop) - int(hand_slice.start)


def _validate_hand_reference(
    hand_reference: torch.Tensor,
    *,
    required_steps: int,
    hand_dim: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    if (
        hand_reference.ndim != 3
        or hand_reference.shape[0] != 1
        or hand_reference.shape[-1] != hand_dim
    ):
        raise ValueError(
            "hand_reference must have shape "
            f"(1,R,{hand_dim}), got {tuple(hand_reference.shape)}"
        )
    if hand_reference.shape[1] < required_steps:
        raise ValueError(
            f"hand_reference has {hand_reference.shape[1]} step(s), but "
            f"{required_steps} are required for full-trajectory guidance"
        )
    if not torch.isfinite(hand_reference).all():
        raise ValueError("hand_reference is non-finite")
    _validate_tensor_contract(
        hand_reference,
        name="hand_reference",
        expected_device=expected_device,
        expected_dtype=expected_dtype,
    )


def _validate_history(
    history: torch.Tensor,
    *,
    n_obs_steps: int,
    hand_dim: int,
) -> None:
    expected = (1, n_obs_steps, hand_dim)
    if tuple(history.shape) != expected:
        raise ValueError(
            f"executor.reset() must return shape {expected}, got {tuple(history.shape)}"
        )
    if not torch.isfinite(history).all():
        raise ValueError("executor.reset() history is non-finite")


def _validate_guided_hand(
    guided_hand: torch.Tensor,
    *,
    execution_steps: int,
    hand_dim: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    expected = (1, execution_steps, hand_dim)
    if tuple(guided_hand.shape) != expected:
        raise ValueError(
            f"guided_hand must have shape {expected}, got {tuple(guided_hand.shape)}"
        )
    if not torch.isfinite(guided_hand).all():
        raise ValueError("guided_hand is non-finite")
    _validate_tensor_contract(
        guided_hand,
        name="guided_hand",
        expected_device=expected_device,
        expected_dtype=expected_dtype,
    )


def _validate_tensor_contract(
    value: torch.Tensor,
    *,
    name: str,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    if value.device != expected_device:
        raise ValueError(
            f"{name} device {value.device} does not match "
            f"proposal device {expected_device}"
        )
    if value.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype {value.dtype} does not match "
            f"proposal dtype {expected_dtype}"
        )


def _validate_post_states(
    post_states: torch.Tensor,
    *,
    execution_steps: int,
    hand_dim: int,
) -> None:
    expected = (1, execution_steps, hand_dim)
    if tuple(post_states.shape) != expected:
        raise ValueError(
            f"executor post_states must have shape {expected}, "
            f"got {tuple(post_states.shape)}"
        )
    if not torch.isfinite(post_states).all():
        raise ValueError("executor post_states are non-finite")
