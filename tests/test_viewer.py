import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from behavior_ap_sim import viewer
from behavior_ap_sim.registry import resolve_registered


class ViewerTests(unittest.TestCase):
    def test_default_general_task_and_instance_selection(self):
        for variant in (None, 'id'):
            text = io.StringIO()
            args = ['bringing_paper_to_recycling'] + ([variant] if variant else [])
            with contextlib.redirect_stdout(text), \
                    patch('behavior_ap_sim.viewer.gpu_status', return_value=(True, '', 24000)):
                self.assertEqual(viewer.main(args + ['--split', 'train', '--instance-id', '1', '--dry-run']), 0)
            report = json.loads(text.getvalue())
            self.assertEqual(report['scene']['task_name'], 'bringing_paper_to_recycling')
            self.assertEqual(report['scene']['variant'], 'original')
            self.assertEqual(report['scene']['split'], 'train')
            self.assertEqual(report['scene']['instance_id'], 1)
            self.assertIn('bringing_paper_to_recycling_original_train_instance1_', report['output'])

    def test_general_focus_does_not_access_target_scope(self):
        backend = types.SimpleNamespace(
            robot=types.SimpleNamespace(get_position_orientation=lambda: ([1., 2., 0.], [0., 0., 0., 1.])),
            _array=np.asarray,
        )
        center, target = viewer.focus_reference(backend, {'target_key': None}, 'scene')
        np.testing.assert_array_equal(center, [1., 2., .8])
        self.assertIsNone(target)

    def test_radio_focus_preserves_registered_target(self):
        target = types.SimpleNamespace(get_position_orientation=lambda: ([4., 5., 1.], [0., 0., 0., 1.]))
        backend = types.SimpleNamespace(
            state=types.SimpleNamespace(env_accessor=types.SimpleNamespace(object_scope={'radio': target})),
            _array=np.asarray,
        )
        center, selected = viewer.focus_reference(backend, {'target_key': 'radio'}, 'target')
        np.testing.assert_array_equal(center, [4., 5., 1.])
        self.assertIs(selected, target)

    def test_canonical_task_and_instance_survive_privilege_forwarding(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('behavior_ap_sim.viewer.needs_privilege', return_value=True), \
                patch('behavior_ap_sim.viewer.runtime_env', return_value={}), \
                patch('behavior_ap_sim.viewer.privilege_command', side_effect=lambda module, args, env: args) as command, \
                patch('behavior_ap_sim.viewer.subprocess.call', return_value=0):
            self.assertEqual(viewer.main(['bringing_paper_to_recycling', 'id', '--split', 'train',
                                          '--instance-id', '1', '--output', str(Path(directory) / 'new')]), 0)
        argv = command.call_args.args[1]
        self.assertEqual(argv[:2], ['bringing_paper_to_recycling', 'original'])
        self.assertEqual(argv[argv.index('--split') + 1], 'train')
        self.assertEqual(argv[argv.index('--instance-id') + 1], '1')

    def test_internal_gui_rejects_public_scene_selection(self):
        with patch('behavior_ap_sim.viewer.gui') as launch:
            for extra in (['--split', 'train'], ['--instance-id', '1']):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    viewer.main(['--_gui-child', '/unused', *extra])
            launch.assert_not_called()

    def test_dry_run_two_variants_without_gpu_or_privilege_launch(self):
        for variant in ('original', 'ood'):
            text = io.StringIO()
            with contextlib.redirect_stdout(text), \
                    patch('behavior_ap_sim.viewer.gpu_status', return_value=(True, '', 24000)), \
                    patch('behavior_ap_sim.viewer.subprocess.Popen') as spawn, \
                    patch('behavior_ap_sim.viewer.subprocess.call') as elevate:
                self.assertEqual(viewer.main(['radio', variant, '--dry-run']), 0)
            record = json.loads(text.getvalue())
            self.assertEqual(record['scene']['variant'], variant)
            self.assertFalse(record['will_start_simulation'])
            self.assertTrue(record['native_gui'])
            self.assertFalse(Path(record['output']).exists())
            spawn.assert_not_called()
            elevate.assert_not_called()

    def test_unregistered_task_and_invalid_duration_rejected(self):
        for args in (['other-task'], ['radio', '--smoke-seconds', '-1'],
                     ['bringing_paper_to_recycling', 'ood'], ['radio', '--split', 'unknown'],
                     ['radio', '--smoke-seconds', 'nan'], ['radio', '--spec', 'unregistered.json']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                viewer.main([*args, '--dry-run'])

    def test_gui_child_requires_inherited_lease_before_import(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('behavior_ap_sim.viewer.gui') as gui:
            with self.assertRaises(ValueError):
                viewer.main(['--_gui-child', directory])
            gui.assert_not_called()

    def test_start_failure_always_closes_and_preserves_original_error(self):
        for close_error, shutdown_error in ((None, None), (None, SystemExit(0)),
                                            (OSError('flush failed'), RuntimeError('shutdown failed'))):
            with self.subTest(close_error=close_error, shutdown_error=shutdown_error), \
                    tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / 'viewer_config.json').write_text(json.dumps(resolve_registered(
                    'bringing_paper_to_recycling', 'original', instance_id=1, split='train')))
                calls = []
                constructor = {}

                class PartialEnvironment:
                    def __init__(self, *args, **kwargs):
                        # Isaac has started but start() never returns to gui().
                        self.backend = types.SimpleNamespace(og=object())
                        constructor.update(kwargs)

                    def start(self):
                        calls.append('start')
                        raise RuntimeError('heading validation failed')

                    def close(self, reason):
                        calls.append(('close', reason))
                        if close_error is not None:
                            raise close_error

                    def shutdown(self):
                        calls.append('shutdown')
                        if shutdown_error is not None:
                            raise shutdown_error

                environment = types.ModuleType('behavior_ap_sim.environment')
                environment.BehaviorEnvironment = PartialEnvironment
                omnigibson = types.ModuleType('omnigibson')
                macros = types.ModuleType('omnigibson.macros')
                macros.gm = types.SimpleNamespace()
                with patch.dict(sys.modules, {'behavior_ap_sim.environment': environment,
                                              'omnigibson': omnigibson, 'omnigibson.macros': macros}):
                    with self.assertRaisesRegex(RuntimeError, 'heading validation failed'):
                        viewer.gui(output)
                self.assertEqual(calls, ['start', ('close', 'viewer_error'), 'shutdown'])
                self.assertEqual(constructor['task'], 'bringing_paper_to_recycling')
                self.assertEqual(constructor['variant'], 'original')
                self.assertEqual(constructor['instance_id'], 1)
                self.assertEqual(constructor['split'], 'train')
                error = json.loads((output / 'error.json').read_text())
                self.assertEqual(error['stage'], 'viewer')
                self.assertEqual(error['message'], 'heading validation failed')
                self.assertFalse((output / 'viewer_ready.json').exists())
                if close_error is not None:
                    errors = json.loads((output / 'finalization_errors.json').read_text())['errors']
                    self.assertEqual([row['stage'] for row in errors], ['close', 'shutdown'])
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(viewer.check_exit(output, 0, viewer=True), 1)


if __name__ == '__main__':
    unittest.main()
