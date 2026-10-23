"""Single-environment BEHAVIOR runtime controlled by external numeric actions.

The simulator is owned by the thread which calls :meth:`start`.  Network
handlers must queue work to that thread.  Privileged object access is confined
to the initial OOD fixture; observations expose only cameras and robot state.
Importing this module does not import Isaac Sim, Torch, or a GPU driver.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import signal
import threading
import time


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class _VideoWriter:
    """Stream H264 directly into an MP4 whose index is finalized on close."""

    def __init__(self, path, frame, fps=30):
        import av

        self.path = Path(path)
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = int(frame.shape[1])
        self.stream.height = int(frame.shape[0])
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.options = {"profile": "high", "crf": "18", "preset": "fast"}
        self.stream.codec_context.codec_tag = "avc1"
        self.frames = 0
        self.fps = fps
        self.closed = False
        self.finalized = False

    def write(self, frame):
        import av

        if frame.shape[:2] != (self.stream.height, self.stream.width):
            raise RuntimeError(f"Video resolution changed for {self.path}: {frame.shape}")
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        if self.closed:
            return
        try:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
            self.finalized = True
        finally:
            self.closed = True

    def metadata(self):
        return {
            "path": str(self.path), "frames": self.frames, "fps": self.fps,
            "duration_seconds": self.frames / self.fps,
            "width": self.stream.width, "height": self.stream.height,
            "codec": "h264", "profile": "High", "pixel_format": "yuv420p",
            "faststart_requested": True, "finalized": self.finalized,
        }


class SimulationBackend:
    """Own an official registered scene, apply 23-D actions, and persist evidence.

    Controllers live outside this package. The evaluator's LocalPolicy is an
    initialization placeholder only; this backend never calls its forward().
    """

    def __init__(self, config: dict):
        self.config = dict(config)
        self.output_dir = Path(config["output_dir"]).expanduser().resolve()
        self.task_name = str(config.get("task_name", "turning_on_radio"))
        self.instance_id = int(config.get("instance_id", 301))
        self.max_steps = int(config.get("max_steps", 2000))
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.phase = "external"
        self.manipulation_proprio_enabled = config.get("manipulation_proprio_enabled", False)
        if type(self.manipulation_proprio_enabled) is not bool:
            raise ValueError("manipulation_proprio_enabled must be a boolean")
        self.steps = 0
        self.done = False
        self.success = None
        self.termination_reason = None
        self.terminated = False
        self.truncated = False
        self.environment_truncated = False
        self.host_budget_exhausted = False
        self._ready = False
        self.evaluator = None
        self.started = False
        self.closed = False
        self._runtime_shutdown = False
        self._owner = None
        self._observation_sequence = 0
        self._latest_observation = None
        self._hold_action = None
        self._writers = {}
        self._third_eye = None
        self._third_reference = None
        self._last_result = None
        self._started_at = None
        self._stop_event = config.get("stop_event") or threading.Event()
        self._signal_handlers = {}
        self.render_profile = self.config.get("render_profile", "baseline")
        if self.render_profile not in {"baseline", "async_video", "fast"}:
            raise ValueError("Unknown render profile")
        self._timings = {}
        self._capture_sync_step = None

    def _timed(self, name, elapsed):
        row = self._timings.setdefault(name, {"seconds": 0.0, "calls": 0})
        row["seconds"] += elapsed
        row["calls"] += 1

    def synchronize_observation(self):
        """Flush native sensor transforms at observation boundaries, without physics.

        Fast mode removes the audit camera's per-action extra renders. The
        external controller receives two explicitly refreshed render passes at
        each new observation step. Repeated captures at that step reuse them.
        """
        if self.render_profile == "fast" and self._capture_sync_step != self.steps:
            started = time.perf_counter()
            self._refresh_sensor_observations()
            self._capture_sync_step = self.steps
            self._timed("observation_rgb_sync", time.perf_counter() - started)

    def _assert_owner(self):
        if self._owner is not None and threading.get_ident() != self._owner:
            raise RuntimeError("All simulator calls must run on the thread that started SimulationBackend")

    def _audit(self, kind, **data):
        row = {"event": kind, "time_unix": time.time(), "step": self.steps, "phase": self.phase, **data}
        with (self.output_dir / "backend_audit.jsonl").open("a") as stream:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")

    def start(self):
        self._assert_owner()
        if self.started or self.closed:
            raise RuntimeError("A backend can only start once; use a fresh process for another episode")
        # Validate the low-level entrypoint too: callers must not bypass the
        # public environment registry by constructing this class directly.
        # This runs before creating output files or importing the simulator.
        from .registry import registered_scene_identity
        identity = registered_scene_identity(self.config)
        registered_mode = identity["mode"]
        if self.config.get("mode") != registered_mode:
            raise ValueError("Evaluation mode does not match the registered scene split")
        package_robot_config = (Path(__file__).resolve().parent / "configs" / "r1pro.yaml").resolve()
        robot_path = Path(self.config.get("robot_config", package_robot_config)).expanduser().resolve()
        if robot_path != package_robot_config:
            raise ValueError("Only the package's registered configs/r1pro.yaml robot configuration is supported")
        self._owner = threading.get_ident()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "observations").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        if (self.output_dir / "backend_audit.jsonl").exists():
            raise FileExistsError(f"Refusing to mix episodes in existing output directory: {self.output_dir}")
        self._started_at = time.time()
        self._audit("initializing", config={k: v for k, v in self.config.items() if k != "stop_event"})

        import cv2
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from omegaconf import OmegaConf
        import omnigibson as og
        from omnigibson.macros import gm
        from omnigibson.controllers import ControllerView
        from omnigibson.eval.evaluator import BatchedEvaluator
        from omnigibson.eval.utils.eval_utils import seed_everything
        from omnigibson.eval.wrappers import DefaultWrapper
        from omnigibson.object_states import OnTop, ToggledOn

        self.np, self.th, self.cv2, self.Rotation = np, torch, cv2, Rotation
        self.og, self.ControllerView = og, ControllerView
        self.OnTop, self.ToggledOn = OnTop, ToggledOn
        gm.HEADLESS = bool(self.config.get("headless", True))
        gm.RENDER_VIEWER_CAMERA = bool(self.config.get("write_video", True))
        seed_everything(int(self.config.get("seed", 0)))
        backend = self

        class BehaviorRGBWrapper(DefaultWrapper):
            @classmethod
            def camera_spec(cls):
                return {"modalities": ["rgb"], "resolution": {
                    "head": (448, 448), "left_wrist": (224, 224), "right_wrist": (224, 224)}}

        # Hydra resolves this named module attribute when the evaluator starts.
        globals()["BehaviorRGBWrapper"] = BehaviorRGBWrapper
        globals()["RadioRGBWrapper"] = BehaviorRGBWrapper  # Existing Hydra references remain valid.

        class SetupEvaluator(BatchedEvaluator):
            def _preprocess_obs(inner, obs, state):
                processed = super()._preprocess_obs(obs, state)
                # Retain the evaluator's 224-pixel processed observation while
                # exposing the native 448-pixel head sensor externally.
                for camera_name in inner.robot_camera_names.values():
                    key = camera_name + "::rgb"
                    if key in processed and processed[key].shape[:2] != (224, 224):
                        value = processed[key]
                        rgb = cv2.resize(backend._array(value), (224, 224), interpolation=cv2.INTER_AREA)
                        processed[key] = torch.as_tensor(rgb, device=value.device, dtype=value.dtype)
                return processed

            def _load_instance_state(inner, instance_id, state):
                super()._load_instance_state(instance_id, state)
                backend._setup_fixture(state)

            def _settle_and_finalize(inner, env_indices):
                if backend.config.get("ood", "none") != "none":
                    # No task env.step is consumed during fixture settling.
                    # The original robot configuration is held through joint controllers.
                    robot = inner.instance_eval_states[0].env_accessor.robot
                    hold = backend._make_hold_action(robot)
                    for _ in range(30):
                        robot.apply_action(hold)
                        og.sim.step()
                    backend._validate_fixture(inner.instance_eval_states[0])
                super()._settle_and_finalize(env_indices)

        model = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}
        cfg = OmegaConf.create({
            "env_wrapper": {"_target_": "behavior_ap_sim.backend.BehaviorRGBWrapper"},
            "policy_name": "local", "model": model,
            "headless": gm.HEADLESS, "partial_scene_load": identity["partial_scene_load"],
            "max_steps": max(0, self.max_steps - 1), "write_video": False,
            "mode": registered_mode,
            "seed": int(self.config.get("seed", 0)), "num_envs": 1,
            "task": {"name": self.task_name}, "robot": OmegaConf.load(str(robot_path)),
        })
        self.evaluator = SetupEvaluator(cfg)
        self.evaluator.load_batch({0: self.instance_id}, write_video=False)
        self.state = self.evaluator.instance_eval_states[0]
        self._validate_loaded_scene(identity)
        self.robot = self.state.env_accessor.robot
        if self.robot.action_dim != 23:
            raise RuntimeError(f"Expected the R1Pro 23-D action interface, got {self.robot.action_dim}")
        self._action_idx = self.robot.controller_action_idx
        expected = {"base": 3, "trunk": 4, "arm_left": 7, "gripper_left": 1,
                    "arm_right": 7, "gripper_right": 1}
        if {k: len(v) for k, v in self._action_idx.items()} != expected:
            raise RuntimeError(f"Unexpected controller layout: {self._action_idx}")
        self._hold_action = self._make_hold_action(self.robot)
        if self.config.get("ood", "none") != "none":
            self._validate_fixture(self.state)
        _write_json(self.output_dir / "initial_robot_state.json", self._robot_state())
        self.started = True
        self.success = bool(self.state.env_accessor.success)
        if self.success:
            raise RuntimeError("Episode unexpectedly starts already successful")
        if self.config.get("write_video", True) and og.sim.viewer_camera is None:
            raise RuntimeError("Third-person viewer camera was requested but not created")
        self._warm_third_person_camera()
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda _sig, _frame: self._stop_event.set())
        self._audit("started", action_indices={k: v.tolist() for k, v in self._action_idx.items()},
                    action_frequency_hz=30, physical_head_joint=False, controller="external")
        observation = self.observe()
        self._ready = True
        return observation

    def _validate_loaded_scene(self, identity):
        """Reject external evaluator metadata drift before enabling task actions."""
        task = self.evaluator.env.task
        expected = {
            "scene_model": identity["scene_model"],
            "activity_name": identity["task_name"],
            "activity_definition_id": identity["activity_definition_id"],
            "instance_id": identity["instance_id"],
        }
        actual = {
            "scene_model": getattr(self.state.env_accessor.scene, "scene_model", None),
            "activity_name": getattr(task, "activity_name", None),
            "activity_definition_id": getattr(task, "activity_definition_id", None),
            "instance_id": getattr(self.state, "instance_id", None),
        }
        mismatches = [key for key in expected
                      if actual[key] != expected[key] or type(actual[key]) is not type(expected[key])]
        _write_json(self.output_dir / "scene_identity_validation.json", {
            "verified": not mismatches, "expected": expected, "actual": actual,
            "mismatches": mismatches, "split": identity["split"], "mode": identity["mode"],
            "task_steps": self.steps,
        })
        if mismatches:
            raise ValueError("Loaded official scene identity differs from registered scene: " + ", ".join(mismatches))

    def _setup_fixture(self, state):
        """The only method that may move a task object; called before episode reset."""
        ood = self.config.get("ood", "none")
        if ood == "none":
            _write_json(self.output_dir / "setup.json", {"ood": "none", "privileged_setup_only": True})
            return
        if ood not in {"radio_on_tv_cabinet", "radio_on_tv_cabinet_facing_room"} or self.task_name != "turning_on_radio":
            raise ValueError(f"Unsupported OOD fixture: {ood!r} for task {self.task_name!r}")
        radio = state.env_accessor.object_scope["radio_receiver.n.01_1"]
        support = state.env_accessor.scene.object_registry("name", "bottom_cabinet_rhdbzv_0")
        if support is None:
            raise RuntimeError("TV cabinet bottom_cabinet_rhdbzv_0 is absent from the loaded scene")
        np = self.np
        before_pos, before_quat = radio.get_position_orientation()
        old_pos = self._array(before_pos).copy()
        original_quat = self._array(before_quat).copy()
        requested_quat = original_quat.copy()
        facing_room = ood == "radio_on_tv_cabinet_facing_room"
        if facing_room:
            if radio.category != "radio" or radio.model != "wxnicr":
                raise RuntimeError("Room-facing fixture requires radio/wxnicr with its verified local +X toggle face")
            # wxnicr's toggle marker lies on its local +X face. The cabinet's
            # room-facing edge is world -X. Rotate in WORLD coordinates; keep
            # the original tilt, then recompute the translated collision AABB.
            requested_quat = (self.Rotation.from_euler("z", math.pi)
                              * self.Rotation.from_quat(original_quat)).as_quat()
            radio.set_position_orientation(position=before_pos,
                                           orientation=self.th.as_tensor(requested_quat, dtype=self.th.float32))
        lower, upper = [self._array(a) for a in support.aabb]
        radio_lower, radio_upper = [self._array(a) for a in radio.aabb]
        radio_center = (radio_lower + radio_upper) / 2
        half = (radio_upper - radio_lower) / 2
        if np.any((upper - lower)[:2] < 2 * half[:2] + .02):
            raise RuntimeError("Radio does not fit onto the TV cabinet's horizontal AABB")
        desired_center = (lower + upper) / 2
        # Front is -X in this scene; retain a small edge margin.
        desired_center[0] = lower[0] + half[0] + .035
        desired_center[1] = (lower[1] + upper[1]) / 2
        desired_center[2] = upper[2] + half[2] + .005
        new_pos = old_pos + desired_center - radio_center
        radio.set_position_orientation(position=self.th.as_tensor(new_pos, dtype=self.th.float32),
                                       orientation=self.th.as_tensor(requested_quat, dtype=self.th.float32)
                                       if facing_room else before_quat)
        radio.keep_still()
        if bool(radio.states[self.ToggledOn].get_value()):
            raise RuntimeError("Official source instance radio unexpectedly toggled on")
        self._fixture = (radio, support)
        self._setup_record = {
            "ood": ood, "privileged_setup_only": True,
            "original_benchmark_initial_distribution": False,
            "task_goal_modified": False, "radio_name": radio.name, "support_name": support.name,
            "original_radio_position_world": old_pos.tolist(),
            "requested_radio_position_world": new_pos.tolist(),
            "radio_orientation_xyzw": requested_quat.tolist(),
            "support_aabb_world": [lower.tolist(), upper.tolist()],
        }
        if facing_room:
            self._setup_record.update({
                "radio_category": radio.category, "radio_model": radio.model,
                "original_radio_orientation_xyzw": original_quat.tolist(),
                "requested_radio_orientation_xyzw": requested_quat.tolist(),
                "room_facing_rotation": {
                    "axis_world": [0., 0., 1.], "angle_rad": math.pi,
                    "composition": "world_Rz_pi_left_multiply_original",
                    "control_surface_axis_local": [1., 0., 0.],
                    "cabinet_front_axis_world": [-1., 0., 0.],
                    "settled_front_alignment_min_dot": .95,
                    "angular_speed_limit_rps": .1,
                },
                "requested_control_surface_normal_world": self.Rotation.from_quat(requested_quat).apply(
                    [1., 0., 0.]).tolist(),
            })
        _write_json(self.output_dir / "setup.json", self._setup_record)

    def _validate_fixture(self, state):
        radio, support = self._fixture
        on_top = bool(radio.states[self.OnTop].get_value(support))
        toggled_on = bool(radio.states[self.ToggledOn].get_value())
        self._setup_record.update({
            "settled_on_support": on_top, "initial_toggled_on": toggled_on,
            "settled_radio_position_world": self._array(radio.get_position_orientation()[0]).tolist(),
            "linear_speed_mps": float(self.np.linalg.norm(self._array(radio.get_linear_velocity()))),
        })
        orientation_valid = True
        if self.config.get("ood") == "radio_on_tv_cabinet_facing_room":
            settled_quat = self._array(radio.get_position_orientation()[1])
            normal = self.Rotation.from_quat(settled_quat).apply([1., 0., 0.])
            alignment = float(self.np.dot(normal, [-1., 0., 0.]))
            angular_speed = float(self.np.linalg.norm(self._array(radio.get_angular_velocity())))
            orientation_valid = alignment > .95 and angular_speed <= .1
            self._setup_record.update({
                "settled_radio_orientation_xyzw": settled_quat.tolist(),
                "settled_control_surface_normal_world": normal.tolist(),
                "settled_front_alignment_dot": alignment, "settled_facing_room": alignment > .95,
                "angular_speed_rps": angular_speed,
                "settled_stable": self._setup_record["linear_speed_mps"] <= .05 and angular_speed <= .1,
            })
        _write_json(self.output_dir / "setup.json", self._setup_record)
        if not on_top or toggled_on or self._setup_record["linear_speed_mps"] > .05 or not orientation_valid:
            raise RuntimeError(f"Invalid OOD fixture; inspect {self.output_dir / 'setup.json'}")

    def _array(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return self.np.asarray(value)

    def _make_hold_action(self, robot):
        action = self.th.zeros(robot.action_dim, dtype=self.th.float32)
        for name, (group_key, member_index) in robot._controllers.items():
            action[robot.controller_action_idx[name]] = self.ControllerView.compute_no_op_action(
                group_key, member_index).detach().cpu()
        return action

    @property
    def trunk_qpos(self):
        self._assert_owner()
        return self._array(self.robot.get_joint_positions()[self.robot.trunk_control_idx]).astype(float).tolist()

    @property
    def trunk_limits(self):
        self._assert_owner()
        indices = self.robot.trunk_control_idx
        return self.np.stack([self._array(self.robot.joint_lower_limits[indices]),
                              self._array(self.robot.joint_upper_limits[indices])], axis=1).astype(float).tolist()

    def _frames(self):
        frames = {}
        for role, key in self.evaluator.robot_camera_names.items():
            # Read native sensors rather than upscaling the evaluator's
            # deliberately resized 224-pixel observation.
            sensor_name = key.split("::", 1)[1]
            sensor_obs, _ = self.robot.sensors[sensor_name].get_obs()
            rgb = self._array(sensor_obs["rgb"])[..., :3]
            if role == "head" and rgb.shape[:2] != (448, 448):
                raise RuntimeError(f"Head sensor is {rgb.shape[:2]}, expected native 448x448; no upscaling is allowed")
            frames[role] = self.np.ascontiguousarray(rgb, dtype=self.np.uint8)
        if set(frames) != {"head", "left_wrist", "right_wrist"}:
            raise RuntimeError(f"Expected three camera roles, got {sorted(frames)}")
        return frames

    def _robot_state(self):
        pos, quat = self.robot.get_position_orientation()
        q = self._array(quat)
        rotation = self.Rotation.from_quat(q)
        local_linear = rotation.inv().apply(self._array(self.robot.get_linear_velocity()))
        local_angular = rotation.inv().apply(self._array(self.robot.get_angular_velocity()))
        state = {
            "base_position": self._array(pos).astype(float).tolist(),
            "base_orientation_xyzw": q.astype(float).tolist(),
            "base_yaw_rad": float(rotation.as_euler("xyz")[2]),
            "base_velocity_local": [float(local_linear[0]), float(local_linear[1]), float(local_angular[2])],
            "trunk_qpos": self.trunk_qpos, "trunk_limits": self.trunk_limits,
            "trunk_joint_names": list(self.robot.trunk_joint_names),
        }
        if self.manipulation_proprio_enabled:
            def measured_vector(value, length, label):
                array = self._array(value)
                if (array.shape != (length,) or array.dtype.kind not in "fiu"
                        or not self.np.isfinite(array).all()):
                    raise ValueError(f"Measured {label} must contain exactly {length} finite numbers")
                return array.astype(float).tolist()

            # These are measured articulation joint indices, not the 23-D
            # controller action indices or the frozen hold-command targets.
            joints = self.robot.get_joint_positions(normalized=False)
            state["arm_qpos"], state["gripper_qpos"] = {}, {}
            for arm in ("left", "right"):
                state["arm_qpos"][arm] = measured_vector(
                    joints[self.robot.arm_control_idx[arm]], 7, f"{arm} arm qpos")
                state["gripper_qpos"][arm] = measured_vector(
                    joints[self.robot.gripper_control_idx[arm]], 2, f"{arm} gripper qpos")
                position, orientation = self.robot.get_relative_eef_pose(arm=arm)
                state[f"eef_{arm}"] = {
                    "position": measured_vector(position, 3, f"{arm} EEF position"),
                    "orientation_xyzw": measured_vector(orientation, 4, f"{arm} EEF orientation"),
                    "frame": "robot_base_link",
                }
        return state

    def observe(self):
        self._assert_owner()
        if self.closed:
            if self._latest_observation is None:
                raise RuntimeError("No observation exists; backend did not start")
            return self._latest_observation
        if not self.started:
            raise RuntimeError("Backend has not started")
        self.synchronize_observation()
        self._observation_sequence += 1
        observation_id = f"step-{self.steps:08d}-obs-{self._observation_sequence:06d}"
        camera_dir = self.output_dir / "observations" / observation_id
        camera_dir.mkdir()
        cameras = {}
        for role, rgb in self._frames().items():
            path = camera_dir / f"{role}.png"
            if not self.cv2.imwrite(str(path), self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)):
                raise IOError(f"Failed to save camera image: {path}")
            cameras[role] = {"path": str(path), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                             "source": "robot_sensor", "color_space": "RGB", "step": self.steps}
        obs = {
            "observation_id": observation_id, "step": self.steps, "phase": self.phase,
            "remaining_steps": max(0, self.max_steps - self.steps),
            "task_instruction": self.config.get("task_instruction", self.task_name.replace("_", " ")),
            "robot": self._robot_state(), "cameras": cameras,
            "terminal": {"done": self.done, "success": self.success,
                         "termination_reason": self.termination_reason},
            "action_interface": {
                "action_dimension": 23,
                "action_frequency_hz": 30,
                "physical_head_joint": False,
                "base_frame": "robot_local: +x forward, +y left, +yaw counterclockwise",
                "trunk_joint_names": list(self.robot.trunk_joint_names),
            },
        }
        _write_json(camera_dir / "observation.json", obs)
        _write_json(self.output_dir / "observation.json", obs)
        self._latest_observation = obs
        self._audit("observation", observation_id=observation_id)
        return obs

    def _refresh_sensor_observations(self):
        """Synchronize sensor prim edits without advancing physics or metrics.

        Isaac needs two render passes after a pose edit to propagate the new
        transform. Refresh the evaluator observation from those same sensors
        before returning a synchronized external observation.
        """
        for _ in range(2):
            self.og.sim.render()
        obs_list, _ = self.evaluator.env.get_obs()
        self.state.obs = self.evaluator._preprocess_obs(obs_list[0], self.state)

    def _step(self, action, source="external"):
        self._assert_owner()
        if not self.started or self.closed:
            raise RuntimeError("Backend must be started and open before applying actions")
        if source != "external":
            raise ValueError("SimulationBackend accepts external numeric actions only")
        if self.done:
            return False
        if self._stop_event.is_set():
            self.done, self.termination_reason = True, "stop_requested"
            return False
        if self.steps >= self.max_steps:
            self.done, self.truncated, self.host_budget_exhausted = True, True, True
            self.termination_reason = "host_budget"
            return False
        action = self.th.as_tensor(action, dtype=self.th.float32).detach().cpu().reshape(-1)
        if action.shape != (23,) or not bool(self.th.isfinite(action).all()):
            raise ValueError("Simulator actions must contain exactly 23 finite values")
        started = time.perf_counter()
        phase_started = started
        term, trunc, _ = self.evaluator._apply_actions(action[None, :], [0])
        self._timed("physics_and_native_observation", time.perf_counter() - phase_started)
        self.steps += 1
        self.success = bool(self.state.env_accessor.success)
        self.terminated, self.environment_truncated = bool(term[0]), bool(trunc[0])
        self.host_budget_exhausted = self.steps >= self.max_steps
        self.truncated = self.environment_truncated or self.host_budget_exhausted
        if self.success or self.terminated or self.truncated:
            self.done = True
            self.termination_reason = ("success" if self.success else "terminated" if self.terminated
                                       else "environment_timeout" if self.environment_truncated else "host_budget")
        phase_started = time.perf_counter()
        self._record_frame()
        self._timed("video_capture_and_submit", time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        self._audit("step", source=source, action=action.tolist(), success=self.success,
                    terminated=self.terminated, truncated=self.truncated)
        self._write_progress()
        self._timed("audit_io", time.perf_counter() - phase_started)
        self._timed("action_step", time.perf_counter() - started)
        return True

    def _write_progress(self):
        _write_json(self.output_dir / "progress.json", {
            "step": self.steps, "max_steps": self.max_steps, "phase": self.phase,
            "closed": self.closed, "done": self.done, "success": self.success,
            "termination_reason": self.termination_reason})

    def _record_frame(self):
        if not self.config.get("write_video", True):
            return
        frames = self._frames()
        head = self.cv2.resize(frames["head"], (448, 448))
        left = self.cv2.resize(frames["left_wrist"], (224, 224))
        right = self.cv2.resize(frames["right_wrist"], (224, 224))
        composite = self.np.hstack([self.np.vstack([left, right]), head])
        started = time.perf_counter()
        third = self._third_person_frame(render_updates=0 if self.render_profile == "fast" else 2)
        self._timed("third_person_capture", time.perf_counter() - started)
        for role, frame in {"robot_cameras": composite, "third_person": third}.items():
            if role not in self._writers:
                if self.render_profile == "baseline":
                    self._writers[role] = _VideoWriter(self.output_dir / "videos" / f"{role}.mp4", frame)
                else:
                    from .async_video import AsyncVideoWriter
                    self._writers[role] = AsyncVideoWriter(
                        self.output_dir / "videos" / f"{role}.mp4", frame,
                        queue_size=8, writer_factory=_VideoWriter)
            started = time.perf_counter()
            self._writers[role].write(frame)
            self._timed("video_encode_or_enqueue", time.perf_counter() - started)

    def _warm_third_person_camera(self):
        if self.config.get("write_video", True):
            # Replicator returns its latest buffer without rendering on get_obs.
            # Match VisionSensor's four-update camera initialization before the
            # first physics step; discard this frame instead of recording it.
            self._third_person_frame(render_updates=4)

    def _third_person_frame(self, *, render_updates=2):
        np = self.np
        center = self._array(self.robot.get_position_orientation()[0])
        if self._third_eye is None:
            # This is an audit-only camera. Its privileged fixture reference is
            # never returned as a robot observation to an external controller.
            target_key = self.config.get("target_key")
            target_obj = self.state.env_accessor.object_scope.get(target_key) if target_key is not None else None
            reference = self._array(target_obj.get_position_orientation()[0]) if target_obj is not None else center.copy()
            direction = center - reference
            direction[2] = 0
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > .05 else np.array([1., 0., 0.])
            side = np.array([-direction[1], direction[0], 0.])
            if self.config.get("ood", "none") in {"radio_on_tv_cabinet", "radio_on_tv_cabinet_facing_room"}:
                # The cabinet backs onto a wall. Observe it from inside the
                # room, rather than putting an audit camera behind that wall.
                self._third_eye = reference + 1.5 * direction + .8 * side
            else:
                self._third_eye = reference - 1.2 * direction + .4 * side
            self._third_eye[2] = 2.15
            self._third_reference = reference
            self.og.sim.viewer_camera.focal_length = 10.0
            _write_json(self.output_dir / "third_person_camera.json", {
                "camera_position_world": self._third_eye.tolist(), "position_fixed": True,
                "auto_pan": self.render_profile != "fast", "audit_only": True,
                "render_profile": self.render_profile,
                "frame_timing": "native action render; external captures separately synchronized" if self.render_profile == "fast" else "two render-only updates after action"})
        if self.render_profile == "fast" and getattr(self, "_fixed_audit_camera_ready", False):
            obs, _ = self.og.sim.viewer_camera.get_obs()
            return np.ascontiguousarray(self._array(obs["rgb"])[..., :3], dtype=np.uint8)
        target = .75 * center + .25 * self._third_reference
        target[2] = .8
        forward = target - self._third_eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        quat = self.Rotation.from_matrix(np.column_stack([right, up, -forward])).as_quat()
        self.og.sim.viewer_camera.set_position_orientation(position=self._third_eye.tolist(), orientation=quat.tolist())
        # USD camera pose changes need two renders to reach the RGB buffer.
        # These are render-only updates: no physical steps or video frames accrue.
        for _ in range(render_updates):
            self.og.sim.render()
        self._fixed_audit_camera_ready = self.render_profile == "fast"
        obs, _ = self.og.sim.viewer_camera.get_obs()
        return np.ascontiguousarray(self._array(obs["rgb"])[..., :3], dtype=np.uint8)

    def close(self, reason="stopped"):
        """Finalize episode evidence without terminating the simulator process."""
        self._assert_owner()
        if self.closed:
            return self._last_result
        errors = []
        if not self.done:
            self.done = True
            self.termination_reason = reason
        valid_episode = self.started and self._ready
        if valid_episode:
            self.success = bool(self.state.env_accessor.success)
            try:
                self.observe()
            except Exception as exc:
                errors.append(f"final_observation: {type(exc).__name__}: {exc}")
        else:
            self.success = None
        metrics = {}
        if valid_episode:
            for metric in self.state.metrics:
                try:
                    metrics.update(_jsonable(metric.aggregate()))
                except Exception as exc:
                    errors.append(f"metric {type(metric).__name__}: {type(exc).__name__}: {exc}")
        drain_started = time.perf_counter()
        videos = {}
        for role, writer in self._writers.items():
            try:
                writer.close()
            except Exception as exc:
                errors.append(f"video {role}: {type(exc).__name__}: {exc}")
            try:
                videos[role] = writer.metadata()
                meta = videos[role]
                if not meta.get("finalized") or meta.get("frames") != self.steps:
                    errors.append(f"video {role}: incomplete finalization or frame count differs from {self.steps} actions")
                for key in ("frames_submitted", "frames_encoded"):
                    if key in meta and meta[key] != self.steps:
                        errors.append(f"video {role}: {key} differs from {self.steps} actions")
            except Exception as exc:
                errors.append(f"video {role} metadata: {type(exc).__name__}: {exc}")
        if self.config.get("write_video", True) and self.steps > 0:
            for role in ("robot_cameras", "third_person"):
                if role not in videos:
                    errors.append(f"video {role}: missing recording for {self.steps} actions")
        self._timed("video_close_drain", time.perf_counter() - drain_started)
        result = {
            "task": self.task_name, "instance_id": self.instance_id, "steps": self.steps,
            "variant": self.config.get("variant"), "split": self.config.get("split"),
            "mode": self.config.get("mode"), "scene_model": self.config.get("scene_model"),
            "partial_scene_load": self.config.get("partial_scene_load"),
            "max_steps": self.max_steps, "success": self.success, "done": self.done,
            "phase": self.phase, "termination_reason": self.termination_reason,
            "terminated": self.terminated, "truncated": self.truncated,
            "run_type": "external_numeric_actions", "official_baseline": False,
            "success_source": "BehaviorTask.success" if valid_episode else "unavailable_initialization_failed",
            "environment_terminated": self.terminated, "environment_truncated": self.environment_truncated,
            "host_budget_exhausted": self.host_budget_exhausted,
            "complete_environment_episode": self.terminated or self.environment_truncated,
            "complete_requested_rollout": self.terminated or self.truncated,
            "external_control_steps": self.steps,
            "ood": self.config.get("ood", "none"), "metrics": metrics,
            "physical_head_joint": False, "embodiment_modified": False,
            "elapsed_wall_seconds": time.time() - self._started_at if self._started_at else 0,
            "videos": videos,
            "errors": errors,
            "render_profile": self.render_profile,
            "performance": self._timings,
        }
        self._last_result = result
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "result.json", result)
        self._audit("closed", reason=self.termination_reason, success=self.success, errors=errors)
        self.closed = True
        self._write_progress()
        # Environment.close is a no-op in this OG version. Global Isaac shutdown
        # is deferred until the service has persisted its terminal RPC receipt.
        if self.evaluator is not None:
            try:
                self.evaluator.env.close()
            except Exception as exc:
                errors.append(f"env.close: {type(exc).__name__}: {exc}")
        _write_json(self.output_dir / "result.json", result)
        if errors:
            raise RuntimeError("Episode evidence finalization failed: " + "; ".join(errors))
        return result

    def shutdown_runtime(self):
        """Release Isaac only after the caller has finalized its own receipts.

        SimulationApp.close can exit natively without unwinding Python, so the
        service must close HTTP and persist all evidence before calling here.
        """
        self._assert_owner()
        if self._runtime_shutdown:
            return
        if not self.closed:
            raise RuntimeError("Call close() and persist service receipts before shutdown_runtime()")
        self._runtime_shutdown = True
        if hasattr(self, "og"):
            try:
                self.og.shutdown()
            except SystemExit:
                pass
            except Exception as exc:
                self._last_result["errors"].append(f"og.shutdown: {type(exc).__name__}: {exc}")
                _write_json(self.output_dir / "result.json", self._last_result)
