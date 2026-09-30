"""Offline synthetic data checks, with no model or simulator imports."""
import copy
import unittest
import numpy as np
from analyze import continuity, noise_audit
from experiment import jobs, episodes_for, commands
from compose_video import frame_indices


def prediction(j, plan):
    reference = np.repeat(np.arange(j, j+5)[:, None], 22, axis=1).astype(float)
    return dict(reference_index=j, full_plan_rad=plan.tolist(), reference_rad=reference.tolist(), valid_future_steps=5)


class AnalysisTests(unittest.TestCase):
    def test_only_direct_qualified_conditions_are_scheduled(self):
        self.assertEqual(len(jobs()), 220)
        self.assertEqual(len(set(jobs())), 220)
        self.assertNotIn(2, episodes_for('m170_mu20'))
        self.assertIn(2, episodes_for('m044_mu11'))
        server, sim, _ = commands(jobs()[0])
        self.assertIn('--baseline-trace', server)
        self.assertIn('--audit-recording', sim)
        self.assertNotIn('--stop-on-native-failure', sim)

    def test_overlap_and_boundary_are_distinct(self):
        plans = [prediction(j, np.repeat(np.arange(j,j+5)[:,None],22,axis=1)) for j in (0,2)]
        stats = continuity(plans, 7)
        self.assertEqual(stats['overlap_plan_rmse_rad'], 0)
        self.assertEqual(stats['executed_boundary_rmse_rad'], 1)
        plans[1]['full_plan_rad'][0] = [3.] * 22
        stats = continuity(plans, 3)
        self.assertEqual(stats['overlap_plan_rmse_rad'], 1)
        self.assertEqual(stats['aligned_joint_values'], 22)
        self.assertIsNone(continuity(plans, 2)['overlap_plan_rmse_rad'])

    def test_padding_and_misalignment(self):
        plans = [prediction(j, np.zeros((5,22))) for j in (0,2)]
        plans[1]['valid_future_steps'] = 1
        plans[1]['full_plan_rad'][1:] = [[999.] * 22] * 4
        self.assertEqual(continuity(plans, 9)['overlap_plan_rmse_rad'], 0)
        plans[1]['reference_rad'][0][0] += 1
        with self.assertRaises(AssertionError):
            continuity(plans, 9)

    def test_noise_audit_rejects_wrong_shift(self):
        rng = np.random.RandomState(1)
        xi = rng.randn(5, 8, 22).astype(np.float32)
        noise = xi.copy()
        for j in range(1, len(noise)):
            noise[j, 3:6] = .8 * noise[j-1, 5:8] + .6 * xi[j, 3:6]
        rows = [dict(raw_noise=n.tolist(), innovation=x.tolist()) for n,x in zip(noise,xi)]
        noise_audit(rows, .8)
        broken = copy.deepcopy(rows)
        broken[1]['raw_noise'][3] = noise[0, 3].tolist()
        with self.assertRaises(AssertionError):
            noise_audit(broken, .8)

    def test_video_reference_progress_and_full_timeline(self):
        timeline = list(frame_indices(5, 3, [1, 1.5, 2, 2.5, 3]))
        actions = [x for x in timeline if x[2] == 'action']
        self.assertEqual([x[0] for x in actions], [61,61,62,62,63])
        self.assertEqual([x[1] for x in actions], list(range(61,66)))
        self.assertEqual(len(timeline), 30+60+30+5+60)
        self.assertEqual(timeline[-1][:2], (123,125))


if __name__ == '__main__':
    unittest.main(verbosity=2)
