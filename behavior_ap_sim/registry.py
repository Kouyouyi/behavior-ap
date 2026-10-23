"""Official ID instance catalog and explicitly approved OOD variants, without GPU imports."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / 'configs/task_registry.json'
OFFICIAL_PATH = REGISTRY_PATH.parent / 'official_tasks.json'


@lru_cache(maxsize=1)
def _registered_tasks():
    registry = json.loads(REGISTRY_PATH.read_text())
    official = json.loads(OFFICIAL_PATH.read_text())
    if registry.get('schema_version') != 2 or registry.get('official_catalog') != OFFICIAL_PATH.name:
        raise ValueError('Unsupported task registry schema')
    if official.get('schema_version') != 1 or not isinstance(official.get('tasks'), list):
        raise ValueError('Unsupported official catalog schema')
    entries = official['tasks']
    if len(entries) != 100 or [row['task_index'] for row in entries] != list(range(100)):
        raise ValueError('Official catalog must contain exactly task indices 0 through 99')
    if len({row['task_name'] for row in entries}) != len(entries):
        raise ValueError('Duplicate official task name')
    overrides = {row['task_name']: row for row in registry['tasks']}
    if set(overrides) - {row['task_name'] for row in entries}:
        raise ValueError('OOD registry contains a task outside the official catalog')
    rows = []
    for entry in entries:
        instances = entry['instances']
        if set(instances) != {'train', 'public_test'}:
            raise ValueError('Only installed official train and public_test splits are supported')
        for split, ids in instances.items():
            if (not ids or any(type(number) is not int or number < 0 for number in ids)
                    or ids != sorted(set(ids))):
                raise ValueError('Invalid official instance IDs')
        override = overrides.get(entry['task_name'], {})
        if override and override['task_index'] != entry['task_index']:
            raise ValueError('Task index differs between official and tuned registries')
        original = {'ood': 'none', 'seed': 20261020, 'initial_yaw_offset_deg': 0.0,
                    'initial_offset_x': 0.0, 'initial_offset_y': 0.0, 'scenario_spec': None,
                    'validation_scope': 'Official unmodified task instance; registration is not a task-success claim.'}
        variants = {'original': original, **deepcopy(override.get('variants', {}))}
        rows.append({'task_index': entry['task_index'], 'task_name': entry['task_name'],
                     'scene_model': entry['scene_model'], 'activity_definition_id': entry['activity_definition_id'],
                     'instance_id': 301, 'aliases': override.get('aliases', []),
                     'target_key': override.get('target_key'), 'instances': instances,
                     'variants': variants,
                     'official_success': 'Official BehaviorTask.success / BDDL, independent of search visibility.'})
    return rows


def registered_tasks():
    return deepcopy(_registered_tasks())


def catalog():
    return [{'task_index': row['task_index'], 'task_name': row['task_name'],
             'variants': list(row['variants'])} for row in _registered_tasks()]


def resolve_registered(task, variant='original', *, instance_id=None, split=None):
    if isinstance(task, bool):
        raise ValueError('Task must be its official index or name')
    variant = 'original' if variant == 'id' else variant
    split = 'public_test' if split is None else split
    for row in _registered_tasks():
        if str(task) not in [row['task_name'], str(row['task_index']), *row.get('aliases', [])]:
            continue
        if variant not in row['variants']:
            raise ValueError(f'Unregistered variant {variant!r} for {row["task_name"]}; OOD requires explicit approval')
        if split not in row['instances']:
            raise ValueError(f'Unregistered split {split!r}; only train and public_test are installed')
        ids = row['instances'][split]
        selected = ids[0] if instance_id is None else instance_id
        if type(selected) is not int or selected not in ids:
            raise ValueError(f'Unregistered instance {selected!r} for {row["task_name"]}/{split}')
        if variant != 'original' and (split != 'public_test' or selected != row['instance_id']):
            raise ValueError('Tuned OOD is registered only for public_test instance 301')
        return {'task_name': row['task_name'], 'task_index': row['task_index'],
                'scene_model': row['scene_model'], 'activity_definition_id': row['activity_definition_id'],
                'variant': variant, 'instance_id': selected, 'target_key': row['target_key'],
                'split': split, 'mode': split, 'partial_scene_load': True,
                **deepcopy(row['variants'][variant]), 'configuration_source': str(REGISTRY_PATH)}
    raise ValueError(f'Unregistered task {task!r}; use --list to see installed official tasks')


def require_registered(task_name, instance_id, target_key, ood, scenario_spec=None, *, split=None):
    if scenario_spec is not None:
        raise ValueError('Custom scenario specifications are not registered')
    for row in _registered_tasks():
        if row['task_name'] != task_name:
            continue
        if target_key != row['target_key']:
            raise ValueError(f'Unregistered target {target_key!r}')
        for variant, config in row['variants'].items():
            if config['ood'] == ood:
                return resolve_registered(task_name, variant, instance_id=instance_id, split=split)
        raise ValueError(f'Unregistered fixture {ood!r}')
    raise ValueError(f'Unregistered task {task_name!r}')


def registered_scene_identity(config):
    try:
        expected = require_registered(config['task_name'], config['instance_id'], config['target_key'],
                                      config['ood'], config.get('scenario_spec'), split=config['split'])
    except KeyError as exc:
        raise ValueError(f'Missing registered scene field: {exc.args[0]}') from exc
    for key in ('seed', 'initial_yaw_offset_deg', 'initial_offset_x', 'initial_offset_y',
                'mode', 'split', 'scene_model', 'activity_definition_id', 'partial_scene_load'):
        if config.get(key) != expected[key] or type(config.get(key)) is not type(expected[key]):
            raise ValueError(f'Unregistered scene override: {key}')
    if config.get('container_name') is not None:
        raise ValueError('Container override is not registered')
    if config.get('variant', expected['variant']) != expected['variant']:
        raise ValueError('Variant does not match its registered fixture')
    if config.get('task_index', expected['task_index']) != expected['task_index']:
        raise ValueError('Task index does not match its official identity')
    keys = ('task_name', 'instance_id', 'target_key', 'variant', 'ood', 'seed',
            'initial_yaw_offset_deg', 'initial_offset_x', 'initial_offset_y', 'scenario_spec',
            'mode', 'split', 'scene_model', 'activity_definition_id', 'partial_scene_load')
    return {key: expected[key] for key in keys}


def validate_registered_handoff(directory, config):
    """Validate provenance and bytes before any simulator-state mutation."""
    directory = Path(directory)
    expected = registered_scene_identity(config)
    try:
        metadata = json.loads((directory / 'metadata.json').read_text())
        if not isinstance(metadata, dict) or metadata.get('registered_scene') != expected:
            raise ValueError('Handoff registered scene mismatch or missing provenance')
        if metadata.get('task_name') != expected['task_name'] or metadata.get('instance_id') != expected['instance_id']:
            raise ValueError('Handoff task identity mismatch')
        if metadata.get('saved_before_terminal_segmentation') is not True:
            raise ValueError('Handoff must be saved before visibility audit')
        if (metadata.get('source_task_done') is not False
                or metadata.get('source_official_success') is not False):
            raise ValueError('Handoff requires explicit nonterminal and officially unsuccessful source evidence')
        steps = metadata.get('source_steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Invalid handoff source step count')
        if hashlib.sha256((directory / 'world_state.npy').read_bytes()).hexdigest() != metadata.get('state_sha256'):
            raise ValueError('Handoff state checksum mismatch')
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid handoff evidence at {directory}: {exc}') from exc
    return metadata
