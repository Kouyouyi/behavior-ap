"""Pure-Python registry of tuned radio scenes; no simulation or policy imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / 'configs/task_registry.json'


def registered_tasks():
    registry = json.loads(REGISTRY_PATH.read_text())
    if registry.get('schema_version') != 1 or not isinstance(registry.get('tasks'), list):
        raise ValueError('Unsupported task registry schema')
    return deepcopy(registry['tasks'])


def catalog():
    return [{'task_index': row['task_index'], 'task_name': row['task_name'],
             'variants': list(row['variants'])} for row in registered_tasks()]


def resolve_registered(task, variant):
    for row in registered_tasks():
        if str(task) not in [row['task_name'], str(row['task_index']), *row.get('aliases', [])]:
            continue
        if variant not in row['variants']:
            raise ValueError(f'Unregistered variant {variant!r} for {row["task_name"]}')
        return {'task_name': row['task_name'], 'task_index': row['task_index'],
                'variant': variant, 'instance_id': row['instance_id'], 'target_key': row['target_key'],
                **deepcopy(row['variants'][variant]), 'configuration_source': str(REGISTRY_PATH)}
    raise ValueError(f'Unregistered task {task!r}; use --list to see tuned tasks')


def require_registered(task_name, instance_id, target_key, ood, scenario_spec=None):
    if scenario_spec is not None:
        raise ValueError('Custom scenario specifications are not registered')
    for row in registered_tasks():
        if row['task_name'] != task_name:
            continue
        if isinstance(instance_id, bool) or instance_id != row['instance_id']:
            raise ValueError(f'Unregistered instance {instance_id!r}')
        if target_key != row['target_key']:
            raise ValueError(f'Unregistered target {target_key!r}')
        for variant, config in row['variants'].items():
            if config['ood'] == ood:
                return resolve_registered(task_name, variant)
        raise ValueError(f'Unregistered fixture {ood!r}')
    raise ValueError(f'Unregistered task {task_name!r}')


def registered_scene_identity(config):
    try:
        expected = require_registered(config['task_name'], config['instance_id'], config['target_key'],
                                      config['ood'], config.get('scenario_spec'))
    except KeyError as exc:
        raise ValueError(f'Missing registered scene field: {exc.args[0]}') from exc
    for key in ('seed', 'initial_yaw_offset_deg', 'initial_offset_x', 'initial_offset_y'):
        if config.get(key) != expected[key]:
            raise ValueError(f'Unregistered initial-state override: {key}')
    if config.get('container_name') is not None:
        raise ValueError('Container override is not registered')
    if config.get('variant', expected['variant']) != expected['variant']:
        raise ValueError('Variant does not match its registered fixture')
    keys = ('task_name', 'instance_id', 'target_key', 'variant', 'ood', 'seed',
            'initial_yaw_offset_deg', 'initial_offset_x', 'initial_offset_y', 'scenario_spec')
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
