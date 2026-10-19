from copy import deepcopy
import unittest
import numpy as np
import torch
from survival_agent.hybrid.diagnostics import episode_predictions, prediction_scores
from tests.test_hybrid_stage_b import replay_fixture, small_model


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_future_observation_changes_do_not_enter_open_loop_predictions(self):
        replay = replay_fixture()
        rows = replay.sample(1, unroll=4)
        model = small_model(replay.spec)
        original = episode_predictions(model, rows[0], context=1)
        changed = deepcopy(rows[0]); changed['observations'][2:] += 100
        actual = episode_predictions(model, changed, context=1)
        for a, b in zip(original, actual):
            self.assertEqual(a['prior']['reward'], b['prior']['reward'])
            self.assertEqual(a['prior']['continuation'], b['prior']['continuation'])
        self.assertTrue(any(a['posterior']['reward'] != b['posterior']['reward'] for a, b in zip(original, actual)))

    def test_final_transition_labels_indices_and_no_reset_observation(self):
        replay = replay_fixture()
        for episode in replay.episodes:
            row = deepcopy(episode)
            row.update(loss_mask=np.ones(4, bool), context_is_episode_start=True, episode_offset=0)
            predictions = episode_predictions(small_model(replay.spec), row, context=1)
            self.assertEqual([p['transition'] for p in predictions], [1, 2, 3])
            self.assertEqual([p['reward'] for p in predictions], [2., 3., 4.])
            self.assertEqual(predictions[-1]['continuation'], float(not episode['terminated'][-1]))
            self.assertEqual(predictions[-1]['truncated'], bool(episode['truncated'][-1]))
        with self.assertRaises(ValueError):episode_predictions(small_model(replay.spec), row, context=4)

    def test_baseline_scores_calibration_and_class_counts_have_numeric_oracle(self):
        records = [{'reward': y, 'continuation': y, 'terminated': not bool(y), 'truncated': bool(y)} for y in [0., 1.]]
        scores = prediction_scores(records, constant=(.5, .5))
        self.assertEqual(scores['count'], 2)
        self.assertEqual(scores['death_targets'], 1)
        self.assertEqual(scores['exclusive_truncation_targets'], 1)
        self.assertEqual(scores['constant']['reward_mse'], .25)
        self.assertEqual(scores['constant']['continuation_brier'], .25)
        self.assertAlmostEqual(scores['constant']['continuation_bce'], np.log(2))
        self.assertEqual(scores['constant']['calibration_ece'], 0.)
        self.assertEqual(sum(b['count'] for b in scores['constant']['calibration_bins']), 2)
        extreme = prediction_scores(records, constant=(1., 1.))
        self.assertTrue(np.isfinite(extreme['constant']['continuation_bce']))
        with self.assertRaises(ValueError):prediction_scores(records, constant=(0., 2.))


if __name__ == '__main__':
    unittest.main()
