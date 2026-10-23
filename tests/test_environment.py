import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from behavior_ap_sim.environment import RadioEnvironment


class Body:
    def __init__(self, name):
        self.name = name
        self.position = np.array([0., 0., 1.])
        self.orientation = np.array([0., 0., 0., 1.])
        self.keep_still = Mock()
        self.apply_action = Mock()

    def get_position_orientation(self):
        return self.position.copy(), self.orientation.copy()

    def set_position_orientation(self, position, orientation):
        self.position, self.orientation = np.array(position), np.array(orientation)


class Backend:
    """No GPU, Isaac, policy, network, or physical execution."""
    def __init__(self, config):
        self.config = config
        self.task_name, self.instance_id = config['task_name'], config['instance_id']
        self.output_dir = Path(config['output_dir'])
        self.steps, self.done, self.success, self.termination_reason = 0, False, False, None
        self._stop_event = threading.Event()
        self._assert_owner = Mock()
        self._writers = {}
        self.robot, self.target = Body('robot_r1'), Body('radio')
        self._action_idx = {key: np.array(value) for key, value in {
            'base': range(3), 'trunk': range(3, 7), 'arm_left': range(7, 14), 'gripper_left': [14],
            'arm_right': range(15, 22), 'gripper_right': [22]}.items()}
        self.state = SimpleNamespace(obs={'robot_r1::proprio': np.zeros(61)}, env_accessor=SimpleNamespace(
            object_scope={config['target_key']: self.target}, scene=SimpleNamespace(objects=[self.robot, self.target])))
        self.og = SimpleNamespace(sim=SimpleNamespace(get_sim_step_dt=lambda: 1 / 30, step=Mock()))
        self.Rotation = Rotation
        self.th = SimpleNamespace(as_tensor=lambda value, **kwargs: np.array(value), float32=np.float32)
        self._make_hold_action = Mock(return_value=np.zeros(23))
        self._refresh_sensor_observations = Mock()
        self.synchronize_observation = Mock()
        self._frames = Mock(return_value={role: np.zeros((4, 4, 3), dtype=np.uint8)
                                         for role in ('head', 'left_wrist', 'right_wrist')})
        self.close = Mock(return_value={'errors': []})
        self.shutdown_runtime = Mock()

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _step(self, action, source):
        assert source == 'external'
        if self.done:
            return False
        self.steps += 1
        return True


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.backend_patch = patch('behavior_ap_sim.backend.SimulationBackend', Backend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    def make(self, **kwargs):
        return RadioEnvironment(self.path / 'sim', **kwargs)

    def test_original_and_ood_initialize_with_fixed_interface_and_heading(self):
        original = RadioEnvironment(self.path / 'original', variant='original').start()
        original.backend.og.sim.step.assert_not_called()
        self.assertEqual(original.steps, 0)
        ood = self.make().start()
        self.assertEqual(ood.backend.og.sim.step.call_count, 30)
        self.assertEqual(ood.backend.robot.apply_action.call_count, 30)
        self.assertEqual(ood.steps, 0)
        np.testing.assert_allclose(ood.backend.robot.orientation, [0, 0, 1, 0], atol=1e-8)
        evidence = json.loads((ood.backend.output_dir / 'initial_heading_validation.json').read_text())
        self.assertEqual(evidence['setup_only_physics_steps'], 30)
        self.assertLess(evidence['orientation_error_rad'], 1e-8)
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            ood.start()

    def test_constructor_rejects_nonregistered_variants_and_invalid_budget(self):
        for kwargs in ({'variant': 'cabinet'}, {'max_steps': 0}, {'max_steps': True},
                       {'headless': 'true'}, {'render_profile': 'unverified'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_step_checks_lifecycle_and_action_shape_before_backend(self):
        env = self.make(variant='original')
        with self.assertRaisesRegex(RuntimeError, 'started'):
            env.step(np.zeros(23))
        env.start()
        for action in (np.zeros(22), np.zeros((1, 23)), np.full(23, np.nan), np.full(23, np.inf)):
            with self.subTest(shape=action.shape), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.steps, 0)
        self.assertTrue(env.step(np.zeros(23)))
        self.assertEqual(env.steps, 1)
        env.close('done')
        with self.assertRaisesRegex(RuntimeError, 'open'):
            env.step(np.zeros(23))

    def test_capture_exposes_images_and_state_but_private_ground_truth_is_separate(self):
        env = self.make(variant='original').start()
        observation = env.capture(self.path / 'initial')
        self.assertEqual(set(observation), {'state', 'image_paths', 'step', 'captured_ns'})
        self.assertEqual(len(observation['state']), 61)
        self.assertTrue(all(Path(path).is_file() for path in observation['image_paths']))
        persisted = json.loads((self.path / 'initial/observation.json').read_text())
        self.assertEqual(persisted, observation)
        private = json.loads((self.path / 'initial/scorer_private.json').read_text())
        self.assertIn('target_position', private)
        self.assertIn('official_task_success', private)
        env.backend.synchronize_observation.assert_called_once()
        with self.assertRaises(FileExistsError):
            env.capture(self.path / 'initial')

    def test_failed_state_restore_and_terminal_audit_forbid_physics_or_export(self):
        env = self.make(variant='original').start()
        for attribute in ('_state_restore_failed', '_terminal_audit_started'):
            setattr(env, attribute, True)
            with self.subTest(attribute=attribute):
                with self.assertRaises(RuntimeError):
                    env.step(np.zeros(23))
                with self.assertRaises(RuntimeError):
                    env.save_handoff(self.path / 'handoff')
            setattr(env, attribute, False)
        self.assertEqual(env.steps, 0)
        self.assertFalse((self.path / 'handoff').exists())

    def test_cross_thread_access_is_rejected_before_gpu_or_filesystem_operations(self):
        env = self.make(variant='original').start()
        env.backend._assert_owner.side_effect = RuntimeError('owner thread required')
        calls = (lambda: env.capture(self.path / 'capture'),
                 lambda: env.save_handoff(self.path / 'handoff'),
                 lambda: env.restore_handoff(self.path / 'handoff'),
                 lambda: env.verify_approach(self.path / 'audit'),
                 lambda: env.step(np.zeros(23)), env.refresh)
        for call in calls:
            with self.subTest(call=call), self.assertRaisesRegex(RuntimeError, 'owner thread'):
                call()
        self.assertEqual(env.steps, 0)
        self.assertFalse((self.path / 'capture').exists())
        self.assertFalse((self.path / 'handoff').exists())
        self.assertFalse((self.path / 'audit').exists())

    def test_initial_visibility_audit_is_terminal_even_at_zero_pixels(self):
        env = self.make(initial_visibility_audit=True)
        scorer = Mock(return_value={'visibility_evidence': {'cameras': {
            role: {'pixel_count': 0} for role in ('head', 'left_wrist', 'right_wrist')}}})
        with patch.object(env, 'make_scorer', return_value=scorer):
            env.start()
        with self.assertRaisesRegex(RuntimeError, 'after visibility audit'):
            env.step(np.zeros(23))
        with self.assertRaisesRegex(RuntimeError, 'after visibility audit'):
            env.save_handoff(self.path / 'handoff')

    def test_initial_ood_audit_rejects_any_target_pixel(self):
        env = self.make(initial_visibility_audit=True)
        scorer = Mock(return_value={'visibility_evidence': {'cameras': {
            role: {'pixel_count': int(role == 'head')} for role in ('head', 'left_wrist', 'right_wrist')}}})
        with patch.object(env, 'make_scorer', return_value=scorer), self.assertRaisesRegex(RuntimeError, 'preflight rejected'):
            env.start()
        self.assertTrue(env._terminal_audit_started)

    def test_terminal_audit_failure_still_forbids_later_actions(self):
        env = self.make(variant='original').start()
        scorer = Mock()
        scorer.enable.side_effect = RuntimeError('sensor attach failed')
        with patch.object(env, 'make_scorer', return_value=scorer), self.assertRaisesRegex(RuntimeError, 'sensor attach'):
            env.verify_approach(self.path / 'audit')
        with self.assertRaisesRegex(RuntimeError, 'after visibility audit'):
            env.step(np.zeros(23))

    def test_close_failure_still_closes_environment_and_shutdown_is_available(self):
        env = self.make(variant='original').start()
        env.backend.close.side_effect = RuntimeError('video failure')
        with self.assertRaisesRegex(RuntimeError, 'video failure'):
            env.close()
        self.assertTrue(env._closed)
        env.shutdown()
        env.backend.shutdown_runtime.assert_called_once()


if __name__ == '__main__':
    unittest.main()
