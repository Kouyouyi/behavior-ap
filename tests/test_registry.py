import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from behavior_ap_sim.registry import (catalog, registered_scene_identity, registered_tasks,
                                     require_registered, resolve_registered, validate_registered_handoff)


class RegistryTests(unittest.TestCase):
    def test_catalog_contains_all_official_id_and_only_approved_ood(self):
        rows = catalog()
        self.assertEqual(len(rows), 100)
        self.assertEqual([row['task_index'] for row in rows], list(range(100)))
        self.assertEqual([row['task_name'] for row in rows if 'ood' in row['variants']], ['turning_on_radio'])
        self.assertTrue(all('original' in row['variants'] for row in rows))
        for alias in ('radio', '收音机', 'turning_on_radio', 0):
            config = resolve_registered(alias, 'ood')
            self.assertEqual(config['instance_id'], 301)
            self.assertEqual(config['seed'], 20261020)
            self.assertEqual(config['initial_yaw_offset_deg'], 180)
            self.assertEqual(config['ood'], 'radio_on_tv_cabinet_facing_room')
        original = resolve_registered('radio', 'original')
        self.assertEqual((original['ood'], original['initial_yaw_offset_deg']), ('none', 0))

    def test_unknown_task_variant_and_custom_scene_cannot_enter(self):
        for task, variant in [('bringing_paper_to_recycling', 'ood'), ('radio', 'closed_container'),
                              ('radio', 'radio_on_tv_cabinet_facing_room')]:
            with self.subTest(task=task, variant=variant), self.assertRaises(ValueError):
                resolve_registered(task, variant)
        config = resolve_registered('radio', 'ood')
        for changes in ({'instance_id': 302}, {'target_key': 'other'}, {'ood': 'closed_container'},
                        {'scenario_spec': {}}, {'seed': 0}, {'initial_yaw_offset_deg': 0},
                        {'initial_offset_x': 1}, {'container_name': 'cabinet'}, {'variant': 'original'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                registered_scene_identity({**config, **changes})
        with self.assertRaises(ValueError):
            require_registered('radio', 301, config['target_key'], config['ood'])

    def test_registry_provenance_is_portable_and_return_values_are_isolated(self):
        rows = registered_tasks()
        provenance = rows[0]['variants']['ood']['provenance']
        self.assertEqual(provenance['document'], 'docs/PROVENANCE.md')
        self.assertNotIn('/media/', json.dumps(rows))
        self.assertNotIn('/home/', json.dumps(rows))
        rows[0]['variants'].clear()
        self.assertEqual(len(registered_tasks()[0]['variants']), 2)

    def test_every_official_instance_resolves_without_new_ood(self):
        total = 0
        rows = registered_tasks()
        self.assertEqual(len({row['scene_model'] for row in rows}), 7)
        for row in rows:
            for split, instances in row['instances'].items():
                self.assertEqual(len(instances), 300 if split == 'train' else 20)
                for number in instances:
                    config = resolve_registered(row['task_name'], 'id', split=split, instance_id=number)
                    self.assertEqual(config['ood'], 'none')
                    self.assertEqual(config['initial_yaw_offset_deg'], 0)
                    self.assertEqual(config['instance_id'], number)
                    self.assertEqual(config['mode'], split)
                    total += 1
                for invalid in (min(instances)-1, max(instances)+1, True, 301.0):
                    with self.assertRaises(ValueError):
                        resolve_registered(row['task_name'], 'original', split=split, instance_id=invalid)
            if row['task_index'] != 0:
                with self.assertRaises(ValueError):
                    resolve_registered(row['task_index'], 'ood')
        self.assertEqual(total, 32000)
        self.assertEqual(resolve_registered(0, 'original', split='train')['instance_id'], 0)
        self.assertEqual(resolve_registered(50, 'original', split='train')['instance_id'], 1)

    def test_split_identity_and_layout_overrides_are_rejected(self):
        config = resolve_registered(61, 'original', split='train', instance_id=1)
        self.assertEqual(registered_scene_identity(config)['split'], 'train')
        for key, value in [('mode', 'public_test'), ('split', 'public_test'), ('scene_model', 'other'),
                           ('activity_definition_id', 1), ('partial_scene_load', False), ('target_key', 'fake'),
                           ('task_index', 0), ('initial_offset_x', 2.0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                registered_scene_identity({**config, key: value})
        for split in ('hidden_test', 'val', '../train'):
            with self.assertRaises(ValueError):
                resolve_registered(0, 'original', split=split)
        with self.assertRaises(ValueError):
            resolve_registered('radio', 'ood', split='train', instance_id=1)
        with self.assertRaises(ValueError):
            resolve_registered('radio', 'ood', split='public_test', instance_id=302)

    def fixture(self, directory):
        config = resolve_registered('radio', 'ood')
        path = directory / 'world_state.npy'
        np.save(path, np.arange(4, dtype=np.float32), allow_pickle=False)
        metadata = {'registered_scene': registered_scene_identity(config),
                    'task_name': config['task_name'], 'instance_id': 301,
                    'saved_before_terminal_segmentation': True, 'source_steps': 8,
                    'source_task_done': False, 'source_official_success': False,
                    'state_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return config, metadata

    def test_handoff_identity_hash_and_before_audit_gate(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            config, metadata = self.fixture(path)
            (path / 'metadata.json').write_text(json.dumps(metadata))
            self.assertEqual(validate_registered_handoff(path, config), metadata)
            with self.assertRaisesRegex(ValueError, 'scene mismatch'):
                validate_registered_handoff(path, resolve_registered('radio', 'original'))
            for changes in ({'registered_scene': None}, {'instance_id': 302},
                            {'saved_before_terminal_segmentation': False}, {'source_steps': True},
                            {'source_task_done': True}, {'source_task_done': None},
                            {'source_official_success': True}, {'source_official_success': 0},
                            {'source_steps': -1}, {'state_sha256': 'tampered'}):
                (path / 'metadata.json').write_text(json.dumps({**metadata, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_registered_handoff(path, config)
            (path / 'metadata.json').write_text(json.dumps(metadata))
            with (path / 'world_state.npy').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate_registered_handoff(path, config)


if __name__ == '__main__':
    unittest.main()
