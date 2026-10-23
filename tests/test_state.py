from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from behavior_ap_sim.environment import BehaviorEnvironment, RadioEnvironment
from behavior_ap_sim.state import fingerprint, match_fingerprint, read_serialized_state
from test_environment import Backend


class StateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.backend_patch = patch('behavior_ap_sim.backend.SimulationBackend', Backend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.env = RadioEnvironment(self.path / 'sim', variant='original').start()
        self.state = np.arange(16, dtype=np.float32)
        self.env.backend.og.sim.dump_state = Mock(return_value=self.state)
        self.env.backend.og.sim.load_state = Mock()
        self.env.backend.og.sim.device = 'cpu'

    def export(self):
        return self.env.save_handoff(self.path / 'handoff')

    def test_generic_no_target_handoff_still_checks_full_state_and_object_order(self):
        self.env = BehaviorEnvironment(self.path / 'generic', task='assembling_gift_baskets').start()
        self.env.backend.og.sim.dump_state = Mock(return_value=self.state)
        self.env.backend.og.sim.load_state = Mock()
        self.env.backend.og.sim.device = 'cpu'
        metadata = self.export()
        self.assertEqual(metadata['fingerprint']['objects'], {})
        self.assertEqual(metadata['objects'], ['robot_r1', 'radio'])
        with patch('behavior_ap_sim.state._restore_derived_caches'):
            restored = self.env.restore_handoff(self.path / 'handoff')
        self.assertTrue(restored['full_state_checked'])
        self.assertEqual(restored['physics_steps_added'], 0)
        self.env.backend.og.sim.load_state.assert_called_once()
        self.env.backend.og.sim.step.assert_not_called()

    def test_generic_extra_target_fingerprint_rejected_before_world_load(self):
        self.env = BehaviorEnvironment(self.path / 'generic', task='assembling_gift_baskets').start()
        self.env.backend.og.sim.dump_state = Mock(return_value=self.state)
        self.env.backend.og.sim.load_state = Mock()
        metadata = self.export()
        metadata['fingerprint']['objects']['invented_target'] = [[0, 0, 1], [0, 0, 0, 1]]
        (self.path / 'handoff/metadata.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'fingerprint identity'):
            self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()

    def test_generic_fingerprint_must_explicitly_record_an_object_mapping(self):
        self.env = BehaviorEnvironment(self.path / 'generic', task='assembling_gift_baskets').start()
        self.env.backend.og.sim.dump_state = Mock(return_value=self.state)
        self.env.backend.og.sim.load_state = Mock()
        metadata = self.export()
        for invalid_objects in (None, [], ''):
            metadata['fingerprint']['objects'] = invalid_objects
            (self.path / 'handoff/metadata.json').write_text(json.dumps(metadata))
            with self.subTest(objects=invalid_objects), self.assertRaisesRegex(ValueError, 'fingerprint identity'):
                self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()

    def test_fingerprint_rejects_identity_pose_and_nonfinite_drift(self):
        expected = fingerprint(self.env)
        opposite = deepcopy(expected)
        opposite['robot_pose'][1] = [0, 0, 0, -1]
        match_fingerprint(opposite, expected)
        for changed in ({'task_name': 'other'}, {'instance_id': 302}, {'objects': {}},
                        {'robot_pose': [[.03, 0, 1], [0, 0, 0, 1]]},
                        {'robot_pose': [[float('nan'), 0, 1], [0, 0, 0, 1]]},
                        {'robot_pose': [[0, 0, 1], [0, 0, 0, 0]]},
                        {'robot_pose': [[0, 0, 1], [1, 0, 0, 0]]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                match_fingerprint({**expected, **changed}, expected)

    def test_export_hashes_full_state_and_preserves_private_identity(self):
        self.env.step(np.zeros(23))
        meta = self.export()
        self.assertEqual(meta['source_steps'], 1)
        self.assertEqual(meta['objects'], ['robot_r1', 'radio'])
        self.assertEqual(meta['registered_scene']['variant'], 'original')
        self.assertTrue(meta['saved_before_terminal_segmentation'])
        self.assertIs(meta['source_task_done'], False)
        self.assertIs(meta['source_official_success'], False)
        np.testing.assert_array_equal(read_serialized_state(self.path / 'handoff', meta), self.state)
        with self.assertRaises(FileExistsError):
            self.export()

    def test_restore_rejects_cross_variant_before_world_mutation(self):
        meta = self.export()
        meta['registered_scene']['variant'] = 'ood'
        (self.path / 'handoff/metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'scene mismatch'):
            self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()

    def test_terminal_or_unknown_source_cannot_be_exported(self):
        for done, success in ((True, False), (False, True), (True, True), (False, None)):
            self.env.backend.done, self.env.backend.success = done, success
            with self.subTest(done=done, success=success), self.assertRaisesRegex(ValueError, 'nonterminal'):
                self.export()
        self.assertFalse((self.path / 'handoff').exists())
        self.env.backend.og.sim.dump_state.assert_not_called()

    def test_terminal_destination_is_rejected_before_world_mutation(self):
        self.export()
        for done, success in ((True, False), (False, True), (True, True), (False, None)):
            self.env.backend.done, self.env.backend.success = done, success
            with self.subTest(done=done, success=success), self.assertRaisesRegex(ValueError, 'destination'):
                self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()
        self.assertFalse(self.env._handoff_restore_attempted)

    def test_missing_or_terminal_source_flags_rejected_before_world_mutation(self):
        metadata = self.export()
        for key in ('source_task_done', 'source_official_success'):
            for value in (None, True, 0, 'false'):
                invalid = dict(metadata)
                if value is None:
                    invalid.pop(key)
                else:
                    invalid[key] = value
                (self.path / 'handoff/metadata.json').write_text(json.dumps(invalid))
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'source evidence'):
                    self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()
        self.assertFalse(self.env._handoff_restore_attempted)

    def test_restore_requires_fresh_episode_and_never_adds_task_steps(self):
        self.export()
        with patch('behavior_ap_sim.state._restore_derived_caches') as caches:
            result = self.env.restore_handoff(self.path / 'handoff')
        self.assertEqual(result['physics_steps_added'], 0)
        self.assertTrue(result['full_state_checked'])
        self.assertEqual(self.env.steps, 0)
        self.env.backend.og.sim.step.assert_not_called()
        caches.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'allowed once'):
            self.env.restore_handoff(self.path / 'handoff')
        self.assertEqual(self.env.backend.og.sim.load_state.call_count, 1)

    def test_nonzero_steps_or_wrong_object_order_cannot_restore(self):
        meta = self.export()
        self.env.step(np.zeros(23))
        with self.assertRaisesRegex(ValueError, 'fresh initialized'):
            self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.steps = 0
        meta['objects'].reverse()
        (self.path / 'handoff/metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'object order'):
            self.env.restore_handoff(self.path / 'handoff')
        self.env.backend.og.sim.load_state.assert_not_called()

    def test_full_state_mismatch_is_persisted_and_continuation_forbidden(self):
        self.export()
        self.env.backend.og.sim.dump_state.return_value = self.state + 0.01
        with patch('behavior_ap_sim.state._restore_derived_caches'), self.assertRaisesRegex(ValueError, 'full simulator state differs'):
            self.env.restore_handoff(self.path / 'handoff')
        self.assertTrue((self.path / 'sim/handoff_restored_actual.npy').exists())
        self.assertTrue((self.path / 'sim/handoff_mismatch.json').exists())
        for call in (lambda: self.env.step(np.zeros(23)),
                     lambda: self.env.save_handoff(self.path / 'invalid_handoff')):
            with self.assertRaisesRegex(RuntimeError, 'restoration failed'):
                call()

    def test_invalid_flat_state_or_pickle_never_loads_into_simulator(self):
        meta = self.export()
        path = self.path / 'handoff/world_state.npy'
        for values in (np.full(16, np.nan), np.zeros((2, 8)), np.array([], dtype=float),
                       np.array([{'object': 'pickle'}], dtype=object)):
            np.save(path, values)
            meta['state_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.subTest(dtype=values.dtype, shape=values.shape), self.assertRaises(ValueError):
                read_serialized_state(path.parent, meta)
        self.env.backend.og.sim.load_state.assert_not_called()


if __name__ == '__main__':
    unittest.main()
