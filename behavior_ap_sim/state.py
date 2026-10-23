"""Private scene fingerprints and strictly checked complete-state handoffs."""
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np

from .registry import registered_scene_identity, validate_registered_handoff
from .resources import save_json
from .scoring import array


def fingerprint(environment):
    backend = environment.backend
    target = environment.target
    return {'task_name': backend.task_name, 'instance_id': backend.instance_id,
            'objects': {target.name: [array(value).tolist() for value in target.get_position_orientation()]},
            'robot_pose': [array(value).tolist() for value in backend.robot.get_position_orientation()]}


def _valid_pose(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) != 2:
        raise ValueError('Fingerprint must contain position and quaternion')
    position, quaternion = np.asarray(pose[0], dtype=float), np.asarray(pose[1], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError('Invalid fingerprint position/quaternion')
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3):
        raise ValueError('Fingerprint quaternion is not normalized')
    return position, quaternion


def match_fingerprint(current, expected, tolerance=0.02):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive finite fingerprint tolerance required')
    if (current['task_name'] != expected['task_name'] or current['instance_id'] != expected['instance_id']
            or set(current['objects']) != set(expected['objects'])):
        raise ValueError('Scene fingerprint identity mismatch')
    for name, pose in {**current['objects'], 'robot': current['robot_pose']}.items():
        old = expected['robot_pose'] if name == 'robot' else expected['objects'][name]
        position, quaternion = _valid_pose(pose)
        old_position, old_quaternion = _valid_pose(old)
        if np.linalg.norm(position - old_position) > tolerance:
            raise ValueError('Scene fingerprint position mismatch: ' + name)
        if min(np.linalg.norm(quaternion - old_quaternion), np.linalg.norm(quaternion + old_quaternion)) > 0.02:
            raise ValueError('Scene fingerprint orientation mismatch: ' + name)


def read_serialized_state(directory, metadata):
    """Hash and load exactly the same bytes, with pickle disabled."""
    payload = (Path(directory) / 'world_state.npy').read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata['state_sha256']:
        raise ValueError('Handoff state checksum mismatch')
    state = np.load(BytesIO(payload), allow_pickle=False)
    if state.ndim != 1 or not state.size or state.dtype.kind not in 'fiu' or not np.isfinite(state).all():
        raise ValueError('Handoff must contain a finite flat numeric simulator state')
    return state


def save_handoff(environment, directory):
    environment._require_ready(allow_audit=False)
    if environment.done is not False or environment.success is not False:
        raise ValueError('Handoff export requires a nonterminal, officially unsuccessful source episode')
    backend = environment.backend
    identity = registered_scene_identity(backend.config)
    state = array(backend.og.sim.dump_state(serialized=True))
    if state.ndim != 1 or not state.size or state.dtype.kind not in 'fiu' or not np.isfinite(state).all():
        raise ValueError('Cannot save invalid serialized simulator state')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / 'world_state.npy', state, allow_pickle=False)
    metadata = {'task_name': backend.task_name, 'instance_id': backend.instance_id,
                'registered_scene': identity, 'source_steps': environment.steps,
                'source_task_done': False, 'source_official_success': False,
                'objects': [obj.name for obj in backend.state.env_accessor.scene.objects],
                'fingerprint': fingerprint(environment),
                'state_sha256': hashlib.sha256((directory / 'world_state.npy').read_bytes()).hexdigest(),
                'saved_before_terminal_segmentation': True}
    save_json(directory / 'metadata.json', metadata)
    return metadata


def _restore_derived_caches(backend, state):
    """Restore sleep flags and recompute OG's intentionally unloaded AABB cache."""
    expected, _ = backend.og.sim.deserialize(backend.th.as_tensor(state, device=backend.og.sim.device))
    for index, scene in enumerate(backend.og.sim.scenes):
        saved_objects = expected[index]['registry']['object_registry']
        for obj in scene.objects:
            if saved_objects.get(obj.name, {}).get('is_asleep', False):
                obj.sleep()
    from omnigibson.utils.usd_utils import RigidBodyViewAPI
    from omnigibson.object_states import AABB
    import warp as wp
    RigidBodyViewAPI.invalidate_kinematic([link for scene in backend.og.sim.scenes
                                         for obj in scene.objects for link in obj.links.values()])
    wp.copy(RigidBodyViewAPI.POSES_GPU, RigidBodyViewAPI.POSES)
    RigidBodyViewAPI.update()
    AABB._update_values(AABB.VALUES)
    wp.synchronize()
    AABB.VALUES_CPU.copy_(AABB.VALUES)


def restore_handoff(environment, directory):
    environment._require_ready(allow_audit=False)
    if environment.done is not False or environment.success is not False:
        raise ValueError('Handoff restore requires a nonterminal, officially unsuccessful destination episode')
    if environment.steps != 0 or environment._handoff_restore_attempted:
        raise ValueError('Handoff restore is allowed once, before task actions in a fresh initialized scene')
    backend = environment.backend
    directory = Path(directory)
    metadata = validate_registered_handoff(directory, backend.config)
    if metadata.get('objects') != [obj.name for obj in backend.state.env_accessor.scene.objects]:
        raise ValueError('Handoff scene object order differs')
    expected_fingerprint = metadata.get('fingerprint')
    if (not isinstance(expected_fingerprint, dict)
            or expected_fingerprint.get('task_name') != backend.task_name
            or expected_fingerprint.get('instance_id') != backend.instance_id
            or set(expected_fingerprint.get('objects', {})) != {environment.target.name}):
        raise ValueError('Handoff fingerprint identity is invalid')
    _valid_pose(expected_fingerprint['robot_pose'])
    _valid_pose(expected_fingerprint['objects'][environment.target.name])
    state = read_serialized_state(directory, metadata)
    environment._handoff_restore_attempted = True
    environment._state_restore_failed = True
    backend.og.sim.load_state(backend.th.as_tensor(state, device=backend.og.sim.device), serialized=True)
    environment.refresh()
    _restore_derived_caches(backend, state)
    restored = array(backend.og.sim.dump_state(serialized=True))
    if restored.shape != state.shape or not np.allclose(restored, state, atol=1e-4, rtol=1e-5):
        np.save(backend.output_dir / 'handoff_restored_actual.npy', restored, allow_pickle=False)
        save_json(backend.output_dir / 'handoff_mismatch.json', {
            'shape_expected': list(state.shape), 'shape_actual': list(restored.shape),
            'full_state_checked': True, 'physics_steps_added': 0, 'continuation_allowed': False})
        raise ValueError('Restored full simulator state differs from saved handoff')
    match_fingerprint(fingerprint(environment), expected_fingerprint)
    if environment.steps != 0:
        raise RuntimeError('Task steps changed during handoff restoration')
    environment._state_restore_failed = False
    record = {'verified': True, 'source': str(directory.resolve()), 'source_steps': metadata['source_steps'],
              'registered_scene': metadata['registered_scene'], 'full_state_checked': True,
              'physics_steps_added': 0}
    save_json(backend.output_dir / 'handoff_restore.json', record)
    return record
