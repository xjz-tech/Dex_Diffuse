"""Four reference actions plus a time-aligned tail of the previous edited plan."""
import numpy as np
from reference_action_editor import ReferenceActionEditor


def initialize_future(reference4, previous_plan, advance=2):
    reference4 = np.asarray(reference4, dtype=np.float32)
    assert reference4.ndim == 3 and reference4.shape[1:] == (4, 22)
    if previous_plan is None:
        tail = np.repeat(reference4[:, -1:], 5, axis=1)
        indices = None
    else:
        previous_plan = np.asarray(previous_plan, dtype=np.float32)
        assert previous_plan.shape == (len(reference4), 9, 22)
        assert 1 <= advance <= 4
        # New future slot k corresponds to previous future slot k + advance.
        # For exec2 the five new tail slots use [6, 7, 8, 8, 8].
        indices = np.minimum(np.arange(4, 9) + advance, 8)
        tail = previous_plan[:, indices].copy()
    return np.concatenate([reference4, tail], axis=1), indices


class ReferencePlanTailEditor(ReferenceActionEditor):
    def __init__(self, checkpoint, noise_ratio, steps=4, execution_steps=2):
        super().__init__(checkpoint, noise_ratio, steps, execution_steps)
        assert execution_steps == 2
        self.reset()
        self.metadata.update(
            algorithm='reference4_previous_plan_tail_sdedit', future_reference_steps=4,
            generated_future_steps=9, initialized_tail_steps=5,
            tail_initialization='previous full edited plan shifted by actual execution count; last-action padding',
            bootstrap='hold reference action 3 across tail',
            previous_plan_tail_indices=[6, 7, 8, 8, 8],
            extra_network_calls=0, total_network_calls=steps,
            reference_prefix_editable=True, tail_editable=True,
            reference_after_four_used=False)

    def reset(self):
        self.previous_plan = None
        self.previous_index = None
        self.previous_reference_ids = None

    def predict(self, history, reference4, seeds, reference_index, reference_ids):
        reference4 = np.asarray(reference4, dtype=np.float32)
        assert reference4.shape == (len(history), 4, 22)
        reference_ids = tuple(reference_ids)
        assert len(reference_ids) == len(history)
        if reference_index == 0:
            self.reset()
        else:
            assert self.previous_plan is not None, 'nonzero reference index requires previous plan'
            assert reference_index == self.previous_index + self.execution_steps, 'cached plan must advance by actual execution count'
            assert reference_ids == self.previous_reference_ids, 'episode change requires reset'
        future, indices = initialize_future(reference4, self.previous_plan, self.execution_steps)
        plan, stats = super().predict(history, future, seeds, return_full_plan=True)
        assert plan.shape == future.shape and np.isfinite(plan).all()
        delta = plan[:, :4] - reference4
        stats.update(edit_rmse_rad=float(np.sqrt(np.mean(delta ** 2))),
            edit_max_abs_rad=float(np.max(np.abs(delta))),
            executed_prefix_edit_rmse_rad=float(np.sqrt(np.mean(delta[:, :self.execution_steps] ** 2))),
            tail_edit_rmse_rad=float(np.sqrt(np.mean((plan[:, 4:] - future[:, 4:]) ** 2))),
            network_calls=0 if self.noise_ratio == 0 else len(self.timesteps),
            future_reference_steps_used=4,
            tail_source='hold_bootstrap' if indices is None else 'shifted_previous_plan',
            previous_plan_tail_indices=None if indices is None else indices.tolist(),
            previous_plan_reference_index=self.previous_index,
            initialized_tail=future[:, 4:].tolist(),
            reference_prefix=reference4.tolist(), generated_future_plan=plan.tolist())
        self.previous_plan = plan.copy()
        self.previous_index = reference_index
        self.previous_reference_ids = reference_ids
        return plan[:, :self.execution_steps].copy(), stats
