from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from diffusion_policy.guidance.coordinator import (
    GuidedCoordinator,
    GuidedRunResult,
    SegmentRecord,
)
from diffusion_policy.guidance.executor import (
    ExecutionError,
    FakeSegmentExecutor,
)
from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidanceConfig

HAND_DIM = 22
REAL_ACTION_DIM = 31
HAND_SLICE = slice(9, 31)


@dataclass
class RecordingGuidance:
    """Stub SimHandGuidance that records calls and returns deterministic hands."""

    config: SimHandGuidanceConfig
    adapter: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    guided_offset: float = 100.0
    output_dtype: torch.dtype | None = None

    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "history": hand_state_history.detach().clone(),
                "reference": hand_reference.detach().clone(),
                "generator": generator,
            }
        )
        # Deterministic guided hand that differs from the Real reference.
        execution_steps = int(self.config.execution_steps)
        output = (
            hand_reference[:, :execution_steps]
            + self.guided_offset
            + float(len(self.calls))
        )
        if self.output_dtype is not None:
            output = output.to(dtype=self.output_dtype)
        return output


@dataclass
class SeededRecordingGuidance:
    """Guidance stub whose output depends on a seeded generator draw."""

    config: SimHandGuidanceConfig
    adapter: Any
    calls: list[dict[str, Any]] = field(default_factory=list)

    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            hand_reference.shape,
            dtype=hand_reference.dtype,
            device=hand_reference.device,
            generator=generator,
        )
        self.calls.append(
            {
                "history": hand_state_history.detach().clone(),
                "reference": hand_reference.detach().clone(),
                "noise": noise.detach().clone(),
            }
        )
        execution_steps = int(self.config.execution_steps)
        return hand_reference[:, :execution_steps] + noise[:, :execution_steps]


@dataclass
class FakeAdapter:
    n_obs_steps: int = 4
    n_pred_action_steps: int = 9


class RecordingExecutor:
    def __init__(
        self,
        initial_history: torch.Tensor,
        *,
        fail_at: int | None = None,
        malformed_at: int | None = None,
        malformed_shape: tuple[int, ...] = (1, 3, 22),
    ) -> None:
        self.initial_history = initial_history
        self.fail_at = fail_at
        self.malformed_at = malformed_at
        self.malformed_shape = malformed_shape
        self.reset_calls = 0
        self.execute_calls: list[dict[str, Any]] = []
        self._inner = FakeSegmentExecutor(initial_history)

    def reset(self) -> torch.Tensor:
        self.reset_calls += 1
        return self._inner.reset()

    def execute(
        self,
        segment: torch.Tensor,
        *,
        segment_id: int,
    ) -> torch.Tensor:
        self.execute_calls.append(
            {
                "segment": segment.detach().clone(),
                "segment_id": segment_id,
            }
        )
        if self.fail_at is not None and segment_id == self.fail_at:
            raise ExecutionError(
                segment_id=segment_id,
                executed_count=2,
                partial_states=segment[:, :2, HAND_SLICE],
            )
        if self.malformed_at is not None and segment_id == self.malformed_at:
            return torch.zeros(self.malformed_shape, dtype=segment.dtype)
        return self._inner.execute(segment, segment_id=segment_id)


def make_proposal(
    total_steps: int,
    *,
    batch: int = 1,
    action_dim: int = REAL_ACTION_DIM,
) -> torch.Tensor:
    # Distinct values so segment boundaries and hand/arm slices are easy to check.
    base = torch.arange(
        batch * total_steps * action_dim,
        dtype=torch.float32,
    ).reshape(batch, total_steps, action_dim)
    return base


def make_history(n_obs_steps: int = 4) -> torch.Tensor:
    return torch.arange(
        1 * n_obs_steps * HAND_DIM,
        dtype=torch.float32,
    ).reshape(1, n_obs_steps, HAND_DIM)


def make_hand_reference(
    total_execution_steps: int,
    *,
    execution_steps: int = 5,
    guidance_steps: int = 9,
) -> torch.Tensor:
    required_steps = total_execution_steps - execution_steps + guidance_steps
    return make_proposal(required_steps)[..., HAND_SLICE]


def make_guidance(
    *,
    execution_steps: int = 5,
    n_obs_steps: int = 4,
    guided_offset: float = 100.0,
) -> RecordingGuidance:
    return RecordingGuidance(
        config=SimHandGuidanceConfig(execution_steps=execution_steps),
        adapter=FakeAdapter(n_obs_steps=n_obs_steps),
        guided_offset=guided_offset,
    )


def assert_segment_boundaries(
    result: GuidedRunResult,
    proposal: torch.Tensor,
    hand_reference: torch.Tensor,
    execution_steps: int,
    guidance_steps: int = 9,
) -> None:
    total_steps = proposal.shape[1]
    expected_count = total_steps // execution_steps
    assert len(result.records) == expected_count
    for i, record in enumerate(result.records):
        start = i * execution_steps
        stop = start + execution_steps
        assert record.segment_id == i
        assert record.start == start
        assert record.stop == stop
        torch.testing.assert_close(
            record.real_hand_reference,
            hand_reference[:, start : start + guidance_steps],
        )
        torch.testing.assert_close(
            record.executed_action[:, :, :9],
            proposal[:, start:stop, :9],
        )
        torch.testing.assert_close(
            record.executed_action[:, :, HAND_SLICE],
            record.guided_hand,
        )


# ---------------------------------------------------------------------------
# Step 1: dynamic segmentation
# ---------------------------------------------------------------------------


def test_coordinator_segments_fifty_step_proposal_into_ten_boundaries():
    guidance = make_guidance(execution_steps=5, n_obs_steps=4)
    proposal = make_proposal(50)
    hand_reference = make_hand_reference(50)
    history = make_history(4)
    coordinator = GuidedCoordinator(guidance)
    result = coordinator.run(
        proposal,
        FakeSegmentExecutor(history),
        hand_reference=hand_reference,
    )

    assert_segment_boundaries(
        result,
        proposal,
        hand_reference,
        execution_steps=5,
    )
    assert len(result.records) == 10
    assert [(r.start, r.stop) for r in result.records] == [
        (i * 5, i * 5 + 5) for i in range(10)
    ]
    torch.testing.assert_close(result.proposal, proposal)
    assert result.guided_action.shape == proposal.shape


def test_coordinator_guides_nine_step_overlapping_windows_and_executes_five():
    guidance = make_guidance(execution_steps=5, n_obs_steps=4)
    proposal = make_proposal(50)
    hand_reference = make_hand_reference(50)

    result = GuidedCoordinator(guidance).run(
        proposal,
        FakeSegmentExecutor(make_history(4)),
        hand_reference=hand_reference,
    )

    assert len(result.records) == 10
    assert hand_reference.shape == (1, 54, HAND_DIM)
    for segment_id, call in enumerate(guidance.calls):
        start = segment_id * 5
        torch.testing.assert_close(
            call["reference"],
            hand_reference[:, start : start + 9],
        )
        assert call["reference"].shape == (1, 9, HAND_DIM)
        assert result.records[segment_id].guided_hand.shape == (1, 5, HAND_DIM)
        assert result.records[segment_id].executed_action.shape == (
            1,
            5,
            REAL_ACTION_DIM,
        )


def test_coordinator_segments_fifteen_step_proposal_without_hardcoded_counts():
    """Proves production code does not hardcode 50/10; uses T // E dynamically."""
    guidance = make_guidance(execution_steps=5, n_obs_steps=4)
    proposal = make_proposal(15)
    hand_reference = make_hand_reference(15)
    history = make_history(4)
    coordinator = GuidedCoordinator(guidance)
    result = coordinator.run(
        proposal,
        FakeSegmentExecutor(history),
        hand_reference=hand_reference,
    )

    assert_segment_boundaries(
        result,
        proposal,
        hand_reference,
        execution_steps=5,
    )
    assert len(result.records) == 3
    assert [(r.start, r.stop) for r in result.records] == [
        (0, 5),
        (5, 10),
        (10, 15),
    ]


def test_coordinator_rejects_wrong_rank_batch_action_dim_and_nonfinite():
    guidance = make_guidance()
    coordinator = GuidedCoordinator(guidance)
    history = make_history()

    with pytest.raises(ValueError, match=r"\(1,T,31\)"):
        coordinator.run(
            make_proposal(10, batch=1).squeeze(0),
            FakeSegmentExecutor(history),
            hand_reference=make_hand_reference(10),
        )

    with pytest.raises(ValueError, match=r"\(1,T,31\)"):
        coordinator.run(
            make_proposal(10, batch=2),
            FakeSegmentExecutor(history),
            hand_reference=make_hand_reference(10),
        )

    with pytest.raises(ValueError, match=r"\(1,T,31\)"):
        coordinator.run(
            make_proposal(10, action_dim=30),
            FakeSegmentExecutor(history),
            hand_reference=make_hand_reference(10),
        )

    bad = make_proposal(10)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        coordinator.run(
            bad,
            FakeSegmentExecutor(history),
            hand_reference=make_hand_reference(10),
        )


def test_coordinator_rejects_proposal_length_not_divisible_by_execution_steps():
    guidance = make_guidance(execution_steps=5)
    coordinator = GuidedCoordinator(guidance)
    history = make_history()
    with pytest.raises(ValueError, match="not divisible"):
        coordinator.run(
            make_proposal(12),
            FakeSegmentExecutor(history),
            hand_reference=make_hand_reference(12),
        )


def test_coordinator_rejects_guided_hand_dtype_before_assignment():
    guidance = make_guidance(execution_steps=5)
    guidance.output_dtype = torch.float64

    with pytest.raises(ValueError, match="guided_hand.*dtype"):
        GuidedCoordinator(guidance).run(
            make_proposal(10),
            FakeSegmentExecutor(make_history()),
            hand_reference=make_hand_reference(10),
        )


# ---------------------------------------------------------------------------
# Step 2: closed-loop / executor
# ---------------------------------------------------------------------------


def test_fake_executor_reset_returns_initial_history_shape():
    history = make_history(n_obs_steps=4)
    executor = FakeSegmentExecutor(history)
    reset = executor.reset()
    assert reset.shape == (1, 4, 22)
    torch.testing.assert_close(reset, history)


def test_fake_executor_command_is_state_returns_hand_slice():
    history = make_history()
    executor = FakeSegmentExecutor(history)
    segment = make_proposal(5)
    post = executor.execute(segment, segment_id=0)
    assert post.shape == (1, 5, 22)
    torch.testing.assert_close(post, segment[..., HAND_SLICE])


def test_coordinator_updates_history_with_post_states_window():
    n_obs_steps = 4
    execution_steps = 5
    guidance = make_guidance(execution_steps=execution_steps, n_obs_steps=n_obs_steps)
    proposal = make_proposal(10)
    history = make_history(n_obs_steps)
    executor = RecordingExecutor(history)
    coordinator = GuidedCoordinator(guidance)

    result = coordinator.run(
        proposal,
        executor,
        hand_reference=make_hand_reference(10),
    )

    assert len(result.records) == 2
    assert executor.reset_calls == 1

    previous = history
    for record in result.records:
        torch.testing.assert_close(record.history_before, previous)
        expected_history = torch.cat(
            (previous, record.post_states),
            dim=1,
        )[:, -n_obs_steps:, :]
        torch.testing.assert_close(record.history_after, expected_history)
        previous = record.history_after

    torch.testing.assert_close(result.final_history, previous)

    # First guidance call sees reset history; second sees updated window.
    torch.testing.assert_close(guidance.calls[0]["history"], history)
    torch.testing.assert_close(
        guidance.calls[1]["history"],
        result.records[0].history_after,
    )


def test_coordinator_passes_exact_command_length_and_fresh_guidance_per_segment():
    execution_steps = 5
    guidance = make_guidance(execution_steps=execution_steps)
    proposal = make_proposal(15)
    executor = RecordingExecutor(make_history())
    hand_reference = make_hand_reference(15)
    result = GuidedCoordinator(guidance).run(
        proposal,
        executor,
        hand_reference=hand_reference,
    )

    assert len(guidance.calls) == 3
    assert len(executor.execute_calls) == 3
    for i, call in enumerate(executor.execute_calls):
        assert call["segment_id"] == i
        assert call["segment"].shape == (1, execution_steps, REAL_ACTION_DIM)
        torch.testing.assert_close(
            call["segment"],
            result.records[i].executed_action,
        )

    # Fresh guidance call each segment with an overlapping full-horizon reference.
    for i, call in enumerate(guidance.calls):
        start = i * execution_steps
        torch.testing.assert_close(
            call["reference"],
            hand_reference[:, start : start + 9],
        )


def test_coordinator_replaces_only_hand_slice_in_guided_action():
    guidance = make_guidance(execution_steps=5, guided_offset=50.0)
    proposal = make_proposal(10)
    result = GuidedCoordinator(guidance).run(
        proposal,
        FakeSegmentExecutor(make_history()),
        hand_reference=make_hand_reference(10),
    )

    torch.testing.assert_close(
        result.guided_action[:, :, :9],
        proposal[:, :, :9],
    )
    for record in result.records:
        torch.testing.assert_close(
            result.guided_action[:, record.start:record.stop, HAND_SLICE],
            record.guided_hand,
        )
        # Guided hand must differ from the executable prefix of the reference.
        assert not torch.allclose(
            record.guided_hand,
            record.real_hand_reference[:, :5],
        )


def test_coordinator_reproduces_with_fixed_seed_generator():
    config = SimHandGuidanceConfig(execution_steps=5)
    adapter = FakeAdapter(n_obs_steps=4)
    proposal = make_proposal(10)
    history = make_history()

    def run_with_seed(seed: int) -> GuidedRunResult:
        guidance = SeededRecordingGuidance(config=config, adapter=adapter)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return GuidedCoordinator(guidance).run(
            proposal,
            FakeSegmentExecutor(history.clone()),
            hand_reference=make_hand_reference(10),
            generator=generator,
        )

    first = run_with_seed(7)
    second = run_with_seed(7)
    third = run_with_seed(8)

    torch.testing.assert_close(first.guided_action, second.guided_action)
    assert not torch.allclose(first.guided_action, third.guided_action)


def test_coordinator_rejects_malformed_executor_post_states():
    guidance = make_guidance(execution_steps=5)
    executor = RecordingExecutor(
        make_history(),
        malformed_at=0,
        malformed_shape=(1, 3, 22),
    )
    with pytest.raises(ValueError, match="post.?state|shape"):
        GuidedCoordinator(guidance).run(
            make_proposal(10),
            executor,
            hand_reference=make_hand_reference(10),
        )
    # Failed on first segment; second segment never executed.
    assert len(executor.execute_calls) == 1


def test_coordinator_rejects_batch_greater_than_one():
    guidance = make_guidance()
    with pytest.raises(ValueError, match=r"\(1,T,31\)"):
        GuidedCoordinator(guidance).run(
            make_proposal(10, batch=2),
            FakeSegmentExecutor(make_history()),
            hand_reference=make_hand_reference(10),
        )


def test_execution_error_stops_immediately_without_later_segments():
    guidance = make_guidance(execution_steps=5)
    executor = RecordingExecutor(make_history(), fail_at=1)
    coordinator = GuidedCoordinator(guidance)

    with pytest.raises(ExecutionError) as exc_info:
        coordinator.run(
            make_proposal(15),
            executor,
            hand_reference=make_hand_reference(15),
        )

    err = exc_info.value
    assert err.segment_id == 1
    assert err.executed_count == 2
    assert err.partial_states is not None
    assert err.partial_states.shape == (1, 2, 22)

    # Segment 0 executed; segment 1 raised; segment 2 never reached.
    assert [c["segment_id"] for c in executor.execute_calls] == [0, 1]
    assert len(guidance.calls) == 2


def test_execution_error_does_not_fabricate_history_after_failure():
    guidance = make_guidance(execution_steps=5)
    executor = RecordingExecutor(make_history(), fail_at=0)

    with pytest.raises(ExecutionError):
        GuidedCoordinator(guidance).run(
            make_proposal(10),
            executor,
            hand_reference=make_hand_reference(10),
        )

    # Only the failing segment was attempted; no second guidance/history update.
    assert len(guidance.calls) == 1
    assert len(executor.execute_calls) == 1
