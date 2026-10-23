import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from behavior_ap_sim.backend import SimulationBackend
from behavior_ap_sim.registry import resolve_registered


class Writer:
    def __init__(self, frames=3, failure=None, finalized=True):
        self.frames, self.failure, self.finalized = frames, failure, finalized

    def close(self):
        if self.failure:
            raise OSError(self.failure)

    def metadata(self):
        return {
            "frames": self.frames, "frames_submitted": 3,
            "frames_encoded": self.frames, "finalized": self.finalized,
        }


class EvidenceFinalizationTests(unittest.TestCase):
    def backend(self, output, writer=None, steps=3):
        backend = SimulationBackend({"output_dir": output})
        backend.steps = steps
        if writer is not None:
            backend._writers = {"robot_cameras": writer, "third_person": Writer()}
        return backend

    def test_flush_failure_is_persisted_and_raised_after_closing(self):
        with tempfile.TemporaryDirectory() as output:
            backend = self.backend(output, Writer(failure="disk full", finalized=False))
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                backend.close()
            self.assertTrue(backend.closed)
            result = json.loads((Path(output) / "result.json").read_text())
            self.assertFalse(result["videos"]["robot_cameras"]["finalized"])
            self.assertTrue(any("disk full" in error for error in result["errors"]))
            self.assertIs(backend.close(), backend._last_result)

    def test_missing_video_and_dropped_frame_cannot_pass(self):
        for writer in (None, Writer(frames=2)):
            with self.subTest(writer=writer), tempfile.TemporaryDirectory() as output:
                backend = self.backend(output, writer)
                with self.assertRaisesRegex(RuntimeError, "video"):
                    backend.close()

    def test_zero_action_preflight_and_complete_videos_pass(self):
        for steps, writer in ((0, None), (3, Writer())):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as output:
                result = self.backend(output, writer, steps).close()
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["external_control_steps"], steps)

    def test_final_observation_failure_remains_an_evidence_failure(self):
        with tempfile.TemporaryDirectory() as output:
            backend = self.backend(output, Writer())
            backend.started = backend._ready = True
            backend.state = SimpleNamespace(
                env_accessor=SimpleNamespace(success=True), metrics=[],
            )

            def failed_observation():
                raise OSError("sensor read failed")

            backend.observe = failed_observation
            with self.assertRaisesRegex(RuntimeError, "sensor read failed"):
                backend.close()
            result = json.loads((Path(output) / "result.json").read_text())
            self.assertTrue(result["success"])
            self.assertTrue(any("final_observation" in error for error in result["errors"]))

    def test_observation_sync_runs_once_per_step_without_stepping(self):
        with tempfile.TemporaryDirectory() as output:
            backend = SimulationBackend({"output_dir": output, "render_profile": "fast"})
            refreshed = []
            backend._refresh_sensor_observations = lambda: refreshed.append(backend.steps)
            backend.synchronize_observation()
            backend.synchronize_observation()
            self.assertEqual(backend.steps, 0)
            backend.steps = 4
            backend.synchronize_observation()
            backend.synchronize_observation()
            self.assertEqual(refreshed, [0, 4])
            self.assertEqual(backend.steps, 4)

    def test_non_external_control_source_is_rejected_before_simulation(self):
        with tempfile.TemporaryDirectory() as output:
            backend = SimulationBackend({"output_dir": output})
            backend.started = True
            with self.assertRaisesRegex(ValueError, "external numeric actions"):
                backend._step([0.0] * 23, source="semantic_command")
            self.assertEqual(backend.steps, 0)

    def test_loaded_scene_identity_checked_without_actions_and_persisted(self):
        with tempfile.TemporaryDirectory() as output:
            config = resolve_registered("assembling_gift_baskets", "original", split="train", instance_id=42)
            backend = SimulationBackend({**config, "output_dir": output})
            backend.state = SimpleNamespace(instance_id=42, env_accessor=SimpleNamespace(
                scene=SimpleNamespace(scene_model=config["scene_model"])))
            backend.evaluator = SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(
                activity_name=config["task_name"], activity_definition_id=config["activity_definition_id"])))
            backend._validate_loaded_scene(config)
            record = json.loads((Path(output) / "scene_identity_validation.json").read_text())
            self.assertTrue(record["verified"])
            self.assertEqual(record["expected"], record["actual"])
            self.assertEqual(record["split"], "train")
            self.assertEqual(backend.steps, 0)
            self.assertFalse(backend.started)

    def test_loaded_scene_identity_drift_is_rejected_and_recorded(self):
        for key, value in (("scene_model", "wrong_house"), ("activity_name", "wrong_activity"),
                           ("activity_definition_id", 1), ("activity_definition_id", False),
                           ("instance_id", 302)):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as output:
                config = resolve_registered("radio", "ood")
                backend = SimulationBackend({**config, "output_dir": output})
                scene = SimpleNamespace(scene_model=config["scene_model"])
                task = SimpleNamespace(activity_name=config["task_name"],
                                       activity_definition_id=config["activity_definition_id"])
                backend.state = SimpleNamespace(instance_id=config["instance_id"],
                                                env_accessor=SimpleNamespace(scene=scene))
                backend.evaluator = SimpleNamespace(env=SimpleNamespace(task=task))
                target = scene if key == "scene_model" else backend.state if key == "instance_id" else task
                setattr(target, key, value)
                with self.assertRaisesRegex(ValueError, "Loaded official scene identity"):
                    backend._validate_loaded_scene(config)
                record = json.loads((Path(output) / "scene_identity_validation.json").read_text())
                self.assertFalse(record["verified"])
                self.assertEqual(record["mismatches"], [key])
                self.assertFalse(backend.started)
                self.assertEqual(backend.steps, 0)

    def test_generic_low_level_start_rejects_unregistered_original_overrides(self):
        overrides = (
            {"task_name": "not_an_official_task"}, {"instance_id": 99999},
            {"target_key": "invented_target"}, {"ood": "invented_ood"},
            {"mode": "hidden_test"}, {"initial_offset_x": 1.0},
        )
        for override in overrides:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "unstarted"
                config = {**resolve_registered("assembling_gift_baskets", "original"),
                          "output_dir": output, **override}
                backend = SimulationBackend(config)
                with self.assertRaises(ValueError):
                    backend.start()
                self.assertFalse(output.exists())
                self.assertIsNone(backend.evaluator)
                self.assertFalse(backend.started)

    def test_low_level_start_cannot_bypass_registered_scene_or_runtime(self):
        overrides = (
            {"instance_id": 302}, {"target_key": "other_target"},
            {"seed": 1}, {"initial_yaw_offset_deg": 90.0},
            {"initial_offset_x": 1.0}, {"ood": "radio_on_tv_cabinet"},
            {"scenario_spec": {}}, {"container_name": "custom_container"},
            {"variant": "original"}, {"mode": "train"},
            {"robot_config": "/tmp/unregistered_robot.yaml"},
        )
        for override in overrides:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "unstarted"
                config = {**resolve_registered("radio", "ood"), "output_dir": output, **override}
                backend = SimulationBackend(config)
                with self.assertRaises(ValueError):
                    backend.start()
                self.assertFalse(output.exists())
                self.assertIsNone(backend.evaluator)
                self.assertFalse(backend.started)


if __name__ == "__main__":
    unittest.main()
