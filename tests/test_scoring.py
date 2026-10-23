import unittest

import numpy as np

from behavior_ap_sim.scoring import evaluate_approach, mesh_pixels


class ScoringTests(unittest.TestCase):
    def test_exact_target_subtree_not_similar_object_prefix(self):
        labels = {'2': '/World/radio/link/mesh', '3': '/World/radio_other/link/mesh',
                  '4': '/World/radio'}
        mask = np.array([[2, 3, 0, 1, 4]] * 8, dtype=np.uint32)
        result = mesh_pixels(mask, labels, '/World/radio')
        self.assertEqual(result['pixel_count'], 16)
        self.assertTrue(result['visible'])
        self.assertEqual(set(result['matched_renderer_ids']), {2, 4})
        result = mesh_pixels(mask[:7], labels, '/World/radio')
        self.assertEqual(result['pixel_count'], 14)
        self.assertFalse(result['visible'])

    def test_empty_or_invalid_mapping_is_not_invisibility_evidence(self):
        mask = np.zeros((2, 2), dtype=np.uint32)
        for labels, prim in [({}, '/World/radio'), ({'2': 'radio'}, '/World/radio'),
                             ({'2': '/World/other'}, '/World/radio'),
                             ({'2': '/World/radio'}, '/World/radio/')]:
            with self.subTest(labels=labels, prim=prim), self.assertRaises(ValueError):
                mesh_pixels(mask, labels, prim)
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            mesh_pixels(np.full((2, 2), 99), {'2': '/World/radio'}, '/World/radio')
        with self.assertRaisesRegex(ValueError, 'integer'):
            mesh_pixels(mask.astype(float), {'2': '/World/radio'}, '/World/radio')

    def test_geometry_pass_is_neither_planner_claim_nor_official_success(self):
        evidence = {'target_visible': True, 'base_position': [0, 0, 0], 'target_position': [1, 0, 8]}
        result = evaluate_approach(evidence)
        self.assertTrue(result['geometry_and_visibility_passed'])
        self.assertEqual(result['horizontal_distance_m'], 1)
        self.assertIsNone(result['planner_reached_claim'])
        self.assertFalse(result['task_completion_implied'])
        self.assertFalse(result['physical_reachability_verified'])
        self.assertNotIn('success', result)
        for change in ({'target_visible': False}, {'target_position': [1.001, 0, 0]}):
            self.assertFalse(evaluate_approach({**evidence, **change})['geometry_and_visibility_passed'])
        with self.assertRaises(ValueError):
            evaluate_approach({**evidence, 'base_position': [float('nan'), 0, 0]})


if __name__ == '__main__':
    unittest.main()
