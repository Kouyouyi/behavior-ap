import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from behavior_ap_sim import cli
from behavior_ap_sim.registry import resolve_registered


class CliTests(unittest.TestCase):
    def dry(self, *args):
        text = io.StringIO()
        with contextlib.redirect_stdout(text), patch('behavior_ap_sim.cli.gpu_status', return_value=(True, '', 24000)), \
                patch('behavior_ap_sim.cli.subprocess.Popen') as spawn, \
                patch('behavior_ap_sim.cli.subprocess.call') as elevate:
            self.assertEqual(cli.main([*args, '--dry-run']), 0)
        spawn.assert_not_called()
        elevate.assert_not_called()
        return json.loads(text.getvalue())

    def test_default_dry_run_initializes_only_and_has_no_policy(self):
        for variant in ('original', 'ood'):
            report = self.dry('radio', variant)
            self.assertTrue(report['job']['initialize_only'])
            self.assertFalse(report['job']['model_inference'])
            self.assertEqual(report['job']['scene']['variant'], variant)
            self.assertIsNone(report['job']['actions'])
            self.assertEqual(report['job']['capture_every'], 8)
            self.assertFalse(report['will_start_simulation'])
            self.assertFalse(Path(report['output']).exists())

    def test_general_task_default_id_alias_and_selected_split(self):
        report = self.dry('bringing_paper_to_recycling')
        scene = report['job']['scene']
        self.assertEqual(scene['task_name'], 'bringing_paper_to_recycling')
        self.assertEqual(scene['variant'], 'original')
        self.assertEqual(scene['split'], 'public_test')
        self.assertEqual(scene['instance_id'], 301)
        self.assertIn('bringing_paper_to_recycling_original_public_test_instance301_', report['output'])
        by_index = self.dry(str(scene['task_index']), 'id', '--split', 'train', '--instance-id', '1')
        self.assertEqual(by_index['job']['scene']['task_name'], scene['task_name'])
        self.assertEqual(by_index['job']['scene']['variant'], 'original')
        self.assertEqual(by_index['job']['scene']['split'], 'train')
        self.assertEqual(by_index['job']['scene']['instance_id'], 1)

    def test_catalog_distinguishes_tasks_instances_and_layouts(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(['--list']), 0)
        self.assertIn('官方任务: 100', output.getvalue())
        self.assertIn('ID 实例: 32000', output.getvalue())
        self.assertIn('房屋布局: 7', output.getvalue())
        self.assertIn('已注册 OOD: 1', output.getvalue())

    def test_canonical_task_and_instance_survive_privilege_forwarding(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('behavior_ap_sim.cli.needs_privilege', return_value=True), \
                patch('behavior_ap_sim.cli.runtime_env', return_value={}), \
                patch('behavior_ap_sim.cli.privilege_command', side_effect=lambda module, args, env: args) as command, \
                patch('behavior_ap_sim.cli.subprocess.call', return_value=0):
            self.assertEqual(cli.main(['bringing_paper_to_recycling', 'id', '--split', 'train',
                                       '--instance-id', '1', '--output', str(Path(directory) / 'new')]), 0)
        argv = command.call_args.args[1]
        self.assertEqual(argv[:2], ['bringing_paper_to_recycling', 'original'])
        self.assertEqual(argv[argv.index('--split') + 1], 'train')
        self.assertEqual(argv[argv.index('--instance-id') + 1], '1')

    def test_worker_revalidates_scene_split_and_instance(self):
        job = self.dry('bringing_paper_to_recycling', '--split', 'train', '--instance-id', '1')['job']
        job['scene']['instance_id'] = 301
        with self.assertRaises(ValueError):
            cli.validate_job(job)

    def test_internal_worker_rejects_public_scene_selection(self):
        with patch('behavior_ap_sim.cli.execute_job') as execute:
            for extra in (['--split', 'train'], ['--instance-id', '1']):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(['--_worker', '/unused', *extra])
            execute.assert_not_called()

    def test_external_actions_validated_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actions.npy'
            np.save(path, np.zeros((5, 23), dtype=np.float64))
            report = self.dry('radio', 'ood', '--actions', str(path), '--render-profile', 'fast',
                              '--capture-every', '4')
            self.assertFalse(report['job']['initialize_only'])
            self.assertEqual(report['job']['max_steps'], 5)
            self.assertEqual(report['job']['actions_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report['job']['render_profile'], 'fast')
            self.assertEqual(report['job']['capture_every'], 4)
            for values in (np.zeros((5, 22)), np.zeros((0, 23)), np.full((1, 23), np.nan),
                           np.full((1, 23), 1e100), np.full((1, 23), 'text')):
                np.save(path, values)
                with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                    cli.load_actions(path)

    def test_budget_mode_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actions.npy'
            np.save(path, np.zeros((5, 23)))
            for args in (['radio', '--output', directory],
                         ['radio', '--actions', str(path), '--initialize-only'],
                         ['radio', '--actions', str(path), '--initial-visibility-audit'],
                         ['radio', '--initial-visibility-audit', '--export-handoff'],
                         ['radio', '--actions', str(path), '--max-steps', '4'],
                         ['radio', '--capture-every', '0'], ['radio', '--capture-every', '-1'],
                         ['radio', '--capture-every', '1.5'],
                         ['bringing_paper_to_recycling', 'ood'],
                         ['bringing_paper_to_recycling', '--split', 'unknown'],
                         ['bringing_paper_to_recycling', '--instance-id', '99999'],
                         ['bringing_paper_to_recycling', '--initial-visibility-audit'],
                         ['radio', '--max-steps', '0'], ['other-task'],
                         ['radio', '--planner-model', 'model']):
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main([*args, '--dry-run'])

    def test_worker_rejects_terminal_audit_export_and_invalid_capture_interval(self):
        job = self.dry('radio')['job']
        invalid_audit = dict(job, initial_visibility_audit=True, export_handoff=True)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            cli.validate_job(invalid_audit)
        for value in (0, -1, True, 1.5, '8', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'capture_every'):
                cli.validate_job(dict(job, capture_every=value))

    def test_source_change_is_rejected_at_worker_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actions.npy'
            np.save(path, np.zeros((2, 23)))
            job = self.dry('radio', '--actions', str(path))['job']
            np.save(path, np.ones((2, 23)))
            with self.assertRaisesRegex(ValueError, 'source changed'):
                cli.validate_job(job)

    def test_execution_ack_and_shutdown_without_isaac(self):
        instances = []

        class Environment:
            def __init__(self, output, **kwargs):
                self.output, self.steps, self.done, self.success = output, 0, False, False
                self.termination_reason = 'host_budget'
                self.shutdown_called = False
                self.captures = []
                self.kwargs = kwargs
                instances.append(self)

            def start(self):
                self.output.mkdir()

            def capture(self, path):
                path.mkdir(parents=True)
                self.captures.append((self.steps, path.relative_to(self.output.parent).as_posix()))

            def step(self, action):
                self.steps += 1
                return True

            def close(self, reason):
                result = {'success': False, 'steps': self.steps, 'errors': []}
                (self.output / 'result.json').write_text(json.dumps(result))
                return result

            def shutdown(self):
                self.shutdown_called = True

        for count in (8, 10):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                path = output / 'external.npy'
                np.save(path, np.arange(count * 23).reshape(count, 23))
                _, digest = cli.load_actions(path)
                job = {'scene': resolve_registered('bringing_paper_to_recycling', 'original',
                                                   instance_id=1, split='train'), 'max_steps': count,
                       'render_profile': 'baseline', 'initialize_only': False, 'capture_every': 4,
                       'initial_visibility_audit': False, 'export_handoff': False,
                       'actions': str(path), 'actions_sha256': digest}
                (output / 'run_config.json').write_text(json.dumps(job))
                fake = types.ModuleType('behavior_ap_sim.environment')
                fake.BehaviorEnvironment = Environment
                with patch.dict(sys.modules, {'behavior_ap_sim.environment': fake}):
                    cli.execute_job(output)
                events = [json.loads(line) for line in (output / 'execution.jsonl').read_text().splitlines()]
                self.assertEqual([row['event'] for row in events], ['submitted', 'ack'] * count)
                self.assertEqual([row['step'] for row in events],
                                 [step for offset in range(count) for step in (offset, offset + 1)])
                instance = instances[-1]
                self.assertEqual(instance.kwargs['task'], 'bringing_paper_to_recycling')
                self.assertEqual(instance.kwargs['variant'], 'original')
                self.assertEqual(instance.kwargs['instance_id'], 1)
                self.assertEqual(instance.kwargs['split'], 'train')
                expected_captures = [(0, 'initial'), (4, 'capture/00000004'), (8, 'capture/00000008')]
                if count == 10:
                    expected_captures.append((10, 'final'))
                self.assertEqual(instance.captures, expected_captures)
                self.assertTrue(instance.shutdown_called)
                summary = json.loads((output / 'execution_summary.json').read_text())
                self.assertEqual(summary['acknowledged'], count)
                self.assertEqual(summary['capture_every'], 4)
                self.assertEqual(summary['capture_steps'], [row[0] for row in expected_captures])
                self.assertEqual(cli.check_exit(output, 0), 0)

    def test_finalization_failure_persists_marker_before_native_shutdown(self):
        class BrokenEnvironment:
            steps, done, success = 0, False, False

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

            def capture(self, output):
                pass

            def close(self, reason):
                raise OSError('encoder flush failed')

            def shutdown(self):
                raise SystemExit(0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job = self.dry('radio')['job']
            (output / 'run_config.json').write_text(json.dumps(job))
            fake = types.ModuleType('behavior_ap_sim.environment')
            fake.BehaviorEnvironment = BrokenEnvironment
            with patch.dict(sys.modules, {'behavior_ap_sim.environment': fake}), self.assertRaises(SystemExit):
                cli.execute_job(output)
            self.assertIn('encoder flush failed', (output / 'error.json').read_text())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.check_exit(output, 0), 1)


if __name__ == '__main__':
    unittest.main()
