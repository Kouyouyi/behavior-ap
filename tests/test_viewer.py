import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from behavior_ap_sim import viewer
from behavior_ap_sim.registry import resolve_registered


class ViewerTests(unittest.TestCase):
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
                (output / 'viewer_config.json').write_text(json.dumps(resolve_registered('radio', 'ood')))
                calls = []

                class PartialEnvironment:
                    def __init__(self, *args, **kwargs):
                        # Isaac has started but start() never returns to gui().
                        self.backend = types.SimpleNamespace(og=object())

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
                environment.RadioEnvironment = PartialEnvironment
                omnigibson = types.ModuleType('omnigibson')
                macros = types.ModuleType('omnigibson.macros')
                macros.gm = types.SimpleNamespace()
                with patch.dict(sys.modules, {'behavior_ap_sim.environment': environment,
                                              'omnigibson': omnigibson, 'omnigibson.macros': macros}):
                    with self.assertRaisesRegex(RuntimeError, 'heading validation failed'):
                        viewer.gui(output)
                self.assertEqual(calls, ['start', ('close', 'viewer_error'), 'shutdown'])
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
