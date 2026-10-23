import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from behavior_ap_sim import resources


class ResourceTests(unittest.TestCase):
    def test_default_lock_independent_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resources.gpu_lock_path(), Path('/tmp/behavior-ap-gpu.lock'))
        with patch.dict(os.environ, {'BEHAVIOR_GPU_LOCK_PATH': '/tmp/shared-test-gpu.lock'}):
            self.assertEqual(resources.gpu_lock_path(), Path('/tmp/shared-test-gpu.lock'))

    def test_busy_lock_exits_before_gpu_query(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            with path.open('a+') as holder, patch('behavior_ap_sim.resources.gpu_lock_path', return_value=path):
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('behavior_ap_sim.resources.gpu_status') as query, self.assertRaises(resources.GpuBusy):
                    with resources.gpu_lease(wait=False):
                        self.fail('busy lease entered')
                query.assert_not_called()

    def test_busy_gpu_exits_without_spawning_or_stealing(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('behavior_ap_sim.resources.gpu_lock_path', return_value=Path(directory) / 'gpu.lock'), \
                patch('behavior_ap_sim.resources.gpu_status', return_value=(False, 'another-job', 19000)), \
                patch('behavior_ap_sim.resources.subprocess.Popen') as spawn:
            with self.assertRaises(resources.GpuBusy):
                with resources.gpu_lease(wait=False):
                    self.fail('busy GPU entered')
            spawn.assert_not_called()

    def test_lease_validation_requires_exact_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            expected, other = Path(directory) / 'gpu.lock', Path(directory) / 'other.lock'
            with expected.open('a+') as lease, other.open('a+') as wrong, \
                    patch('behavior_ap_sim.resources.gpu_lock_path', return_value=expected):
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                resources.verify_inherited_lease(lease.fileno())
                with self.assertRaises(ValueError):
                    resources.verify_inherited_lease(None)
                with self.assertRaises(ValueError):
                    resources.verify_inherited_lease(wrong.fileno())

    def test_gpu_query_checks_all_processes_and_memory(self):
        for procs, memory, expected in [('', '24000\n', True), ('42, unrelated\n', '24000\n', False),
                                        ('', '19999\n', False)]:
            replies = [subprocess.CompletedProcess([], 0, procs, ''),
                       subprocess.CompletedProcess([], 0, memory, '')]
            with self.subTest(procs=procs, memory=memory), \
                    patch('behavior_ap_sim.resources.subprocess.run', side_effect=replies):
                self.assertEqual(resources.gpu_status()[0], expected)

    def test_privilege_command_preserves_only_scoped_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = Path(directory) / 'helper.sh'
            helper.touch()
            with patch.dict(os.environ, {'BEHAVIOR_SUDO_RUN': str(helper)}):
                command = resources.privilege_command('behavior_ap_sim.cli', ['radio'],
                    {'BEHAVIOR_ASSET_ROOT': '/data/assets', 'PYTHONPATH': '/source',
                     'XAUTHORITY': '/desktop/auth', 'UNRELATED_SECRET': 'should-not-forward'})
            self.assertIn('BEHAVIOR_ASSET_ROOT=/data/assets', command)
            self.assertIn('XAUTHORITY=/desktop/auth', command)
            self.assertFalse(any('UNRELATED_SECRET' in item for item in command))

    def test_exit_zero_requires_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'sim').mkdir()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(resources.check_exit(output, 0), 1)
            (output / 'sim/result.json').write_text(json.dumps({'success': False, 'errors': []}))
            (output / 'episode_closed.json').write_text(json.dumps({'status': 'closed'}))
            self.assertEqual(resources.check_exit(output, 0), 0)
            resources.record_error(output, 'close', OSError('flush failed'), finalization=True)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(resources.check_exit(output, 0), 1)


if __name__ == '__main__':
    unittest.main()
