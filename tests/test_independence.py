"""The environment must never pull a planner or an old checkout into its runtime."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'ap_harness', 'behavior_harness', 'openpi', 'openpi_client', 'codex'}


class IndependenceTests(unittest.TestCase):
    def test_runtime_has_no_legacy_or_model_imports(self):
        for path in (ROOT / 'behavior_ap_sim').rglob('*.py'):
            with self.subTest(path=path):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or '']
                    else:
                        continue
                    self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in names))
                self.assertNotIn('sys.path', path.read_text())
                self.assertNotIn('/behavior_harness/', path.read_text())

    def test_no_dependency_symlinks(self):
        self.assertFalse(any(p.is_symlink() for p in (ROOT / 'behavior_ap_sim').rglob('*')))

    def test_catalog_import_does_not_load_gpu_or_model_modules(self):
        code = '''
import sys
from behavior_ap_sim.registry import resolve_registered
assert resolve_registered('radio', 'ood')['task_name'] == 'turning_on_radio'
assert not {'torch', 'omnigibson', 'openpi', 'ap_harness', 'behavior_harness'} & set(sys.modules)
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == '__main__':
    unittest.main()
