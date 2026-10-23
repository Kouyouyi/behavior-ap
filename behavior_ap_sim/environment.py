"""Standalone registered radio simulation, observations, actions, and private audits."""
import time
from pathlib import Path

import numpy as np

from .registry import resolve_registered
from .resources import save_json
from .scoring import ROLES, RawMeshTargetScorer, array
from .state import fingerprint


class RadioEnvironment:
    def __init__(self, output, *, variant='ood', max_steps=800, render_profile='baseline',
                 headless=True, write_video=True, initial_visibility_audit=False):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if render_profile not in ('baseline', 'async_video', 'fast'):
            raise ValueError('Unknown render profile')
        if any(not isinstance(value, bool) for value in (headless, write_video, initial_visibility_audit)):
            raise ValueError('headless, write_video and initial_visibility_audit must be booleans')
        registered = resolve_registered('radio', variant)
        self.variant = variant
        self.initial_visibility_audit = initial_visibility_audit
        self._started = self._start_attempted = self._terminal_audit_started = False
        self._handoff_restore_attempted = self._state_restore_failed = False
        self._closed = False
        self.scorer = None
        from .backend import SimulationBackend
        self.backend = SimulationBackend({
            **registered, 'output_dir': str(Path(output).resolve()), 'max_steps': max_steps,
            'mode': 'public_test', 'run_mode': 'external', 'head_mode': 'physical_trunk',
            'headless': headless, 'write_video': write_video, 'render_profile': render_profile,
            'robot_config': str(Path(__file__).resolve().parent / 'configs/r1pro.yaml')})

    def _setup_initial_heading(self):
        backend = self.backend
        robot = backend.robot
        offset = float(backend.config['initial_yaw_offset_deg'])
        position, quaternion = robot.get_position_orientation()
        rotated = (backend.Rotation.from_euler('z', offset, degrees=True)
                   * backend.Rotation.from_quat(array(quaternion))).as_quat()
        robot.set_position_orientation(position=position, orientation=backend.th.as_tensor(rotated, dtype=backend.th.float32))
        robot.keep_still()
        save_json(backend.output_dir / 'initial_heading_setup.json', {
            'initial_setup_only': True, 'yaw_offset_deg': offset,
            'position': array(position).tolist(), 'original_orientation': array(quaternion).tolist(),
            'requested_orientation': rotated.tolist()})
        hold = backend._make_hold_action(robot)
        for _ in range(30):
            robot.apply_action(hold)
            backend.og.sim.step()
        self.refresh()
        actual = array(robot.get_position_orientation()[1])
        error = (backend.Rotation.from_quat(actual) * backend.Rotation.from_quat(rotated).inv()).magnitude()
        save_json(backend.output_dir / 'initial_heading_validation.json', {
            'actual_orientation': actual.tolist(), 'orientation_error_rad': float(error),
            'setup_only_physics_steps': 30, 'task_steps': self.steps})
        if error > 0.05:
            raise RuntimeError('Initial search heading failed to persist through settling')

    def start(self):
        if self._start_attempted:
            raise RuntimeError('An environment instance cannot restart or replay initialization')
        self._start_attempted = True
        self.backend.start()
        if self.backend.config['initial_yaw_offset_deg']:
            self._setup_initial_heading()
        actual = {key: value.tolist() for key, value in self.backend._action_idx.items() if len(value)}
        expected = {'base': list(range(3)), 'trunk': list(range(3, 7)), 'arm_left': list(range(7, 14)),
                    'gripper_left': [14], 'arm_right': list(range(15, 22)), 'gripper_right': [22]}
        if actual != expected:
            raise ValueError(f'Action ordering mismatch: {actual}')
        dt = float(self.backend.og.sim.get_sim_step_dt())
        if not np.isclose(dt, 1 / 30):
            raise ValueError(f'Expected 30 Hz simulation, got dt={dt}')
        save_json(self.backend.output_dir / 'interface_validation.json', {
            'action_indices': actual, 'sim_step_dt': dt, 'robot_config': self.backend.config['robot_config'],
            'observation_source': 'official robot_r1::proprio and native robot sensors',
            'joint_axis_physical_probe_passed': False})
        self.target = self.backend.state.env_accessor.object_scope[self.backend.config['target_key']]
        self._started = True
        if self.initial_visibility_audit:
            # This OG version can crash when stepping after attaching these graphs.
            self._terminal_audit_started = True
            self.scorer = self.make_scorer()
            self.scorer.enable()
            self.refresh()
            evidence = self.scorer()
            save_json(self.backend.output_dir / 'initial_occlusion_private.json', evidence)
            self.scorer.save_raw(self.backend.output_dir / 'initial_masks_private')
            save_json(self.backend.output_dir / 'initial_fingerprint_private.json', fingerprint(self))
            if self.variant == 'ood' and any(camera['pixel_count'] for camera in evidence['visibility_evidence']['cameras'].values()):
                raise RuntimeError('Registered OOD initial target is rendered; scene preflight rejected')
        return self

    def _require_ready(self, *, allow_audit=True):
        self.backend._assert_owner()
        if not self._started or self._closed:
            raise RuntimeError('Environment must be started and open')
        if self._state_restore_failed:
            raise RuntimeError('State restoration failed; physics continuation is forbidden')
        if not allow_audit and self._terminal_audit_started:
            raise RuntimeError('Physics and handoff export cannot resume after visibility audit')

    @property
    def steps(self):
        return self.backend.steps

    @property
    def done(self):
        return self.backend.done or self.backend._stop_event.is_set()

    @property
    def success(self):
        return self.backend.success

    @property
    def termination_reason(self):
        return 'stop_requested' if self.backend._stop_event.is_set() else self.backend.termination_reason

    def refresh(self):
        self.backend._assert_owner()
        self.backend._refresh_sensor_observations()

    def make_scorer(self):
        return RawMeshTargetScorer(self.backend.robot, self.target, self.backend.evaluator.robot_camera_names)

    def step(self, action):
        self._require_ready(allow_audit=False)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError('Action must be a finite 23-dimensional array')
        return self.backend._step(action, 'external')

    def capture(self, directory):
        self._require_ready()
        import cv2
        self.backend.synchronize_observation()
        state = array(self.backend.state.obs['robot_r1::proprio'])
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError('Invalid official raw proprioception state')
        frames = self.backend._frames()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        paths = []
        for role in ROLES:
            rgb = frames[role]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f'Invalid current RGB: {role}')
            path = directory / (role + '.png')
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise IOError(path)
            paths.append(str(path.resolve()))
        observation = {'state': state.astype(np.float32).tolist(), 'image_paths': paths,
                       'step': self.steps, 'captured_ns': time.time_ns()}
        save_json(directory / 'observation.json', observation)
        private = self.scorer() if self.scorer else {
            'target_visible': None, 'visibility_measurement': 'not instrumented during physics rollout',
            'target_position': array(self.target.get_position_orientation()[0]).tolist(),
            'base_position': array(self.backend.robot.get_position_orientation()[0]).tolist()}
        private.update(step=self.steps, official_task_success=bool(self.success))
        save_json(directory / 'scorer_private.json', private)
        return observation

    def verify_approach(self, directory):
        self._require_ready(allow_audit=False)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self._terminal_audit_started = True
        save_json(directory / 'started.json', {'step': self.steps, 'physics_must_not_resume': True})
        for writer in self.backend._writers.values():
            writer.close()
        save_json(directory / 'videos_finalized.json', {key: writer.metadata() for key, writer in self.backend._writers.items()})
        self.scorer = self.make_scorer()
        self.scorer.enable()
        self.refresh()
        evidence = self.scorer()
        self.scorer.save_raw(directory / 'masks_private')
        evidence.update(step=self.steps, official_task_success=bool(self.success))
        save_json(directory / 'result.json', evidence)
        return evidence

    def save_handoff(self, directory):
        from .state import save_handoff
        return save_handoff(self, directory)

    def restore_handoff(self, directory):
        from .state import restore_handoff
        return restore_handoff(self, directory)

    def close(self, reason='stopped'):
        try:
            return self.backend.close(reason)
        finally:
            self._closed = True

    def shutdown(self):
        return self.backend.shutdown_runtime()
