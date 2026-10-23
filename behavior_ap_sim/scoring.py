"""Private rendered-pixel audits; ground truth never enters policy observations."""
from pathlib import Path

import numpy as np

from .resources import save_json

ROLES = ('head', 'left_wrist', 'right_wrist')
PIXEL_THRESHOLD = 16


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def mesh_pixels(mask, labels, prim_path):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in 'iu':
        raise ValueError('Expected a nonempty integer rendered mesh instance mask')
    if not isinstance(prim_path, str) or not prim_path.startswith('/World/') or prim_path.endswith('/'):
        raise ValueError('Expected an exact object prim path under /World')
    if not isinstance(labels, dict) or not labels or not all(isinstance(path, str) for path in labels.values()):
        raise ValueError('Missing renderer prim-path mapping')
    mapping = {int(key): value for key, value in labels.items()}
    unknown = set(map(int, np.unique(mask))) - set(mapping) - {0, 1}
    if unknown:
        raise ValueError('Unmapped renderer instance IDs')
    if not any(path.startswith('/World/') for path in mapping.values()):
        raise ValueError('Renderer mapping has no world prim paths')
    ids = [key for key, path in mapping.items() if path == prim_path or path.startswith(prim_path + '/')]
    if not ids:
        raise ValueError('Target mesh absent from renderer mapping: ' + prim_path)
    pixels = int(np.isin(mask, ids).sum())
    return {'pixel_count': pixels, 'visible': pixels >= PIXEL_THRESHOLD, 'image_pixels': int(mask.size),
            'pixel_fraction': pixels / mask.size, 'threshold_pixels': PIXEL_THRESHOLD,
            'matched_renderer_ids': ids, 'target_prim_path': prim_path}


def evaluate_approach(evidence):
    base, target = [np.asarray(evidence[key], dtype=float) for key in ('base_position', 'target_position')]
    if base.shape != (3,) or target.shape != (3,) or not np.isfinite([base, target]).all():
        raise ValueError('Finite 3D positions required for approach audit')
    distance = float(np.linalg.norm((target - base)[:2]))
    passed = evidence.get('target_visible') is True and distance <= 1.0
    return {**evidence, 'horizontal_distance_m': distance, 'max_horizontal_distance_m': 1.0,
            'geometry_and_visibility_passed': passed, 'find_and_approach_success': passed,
            'criterion': 'exact radio >=16 rendered pixels in an onboard camera and horizontal base distance <=1m',
            'planner_reached_claim': None, 'physical_reachability_verified': False,
            'task_completion_implied': False}


class RawMeshTargetScorer:
    """Single registered target, exact rendered USD subtree, three robot cameras."""
    def __init__(self, robot, target, camera_names):
        if set(camera_names) != set(ROLES):
            raise ValueError('Exactly head and both wrist cameras are required')
        self.robot, self.target = robot, target
        self.sensors = {}
        for role in ROLES:
            robot_name, sensor_name = camera_names[role].split('::', 1)
            if robot_name != robot.name:
                raise ValueError('Audit camera must belong to this robot')
            self.sensors[role] = robot.sensors[sensor_name]
        self.annotators, self.raw = {}, {}

    def enable(self):
        if self.annotators:
            raise RuntimeError('Visibility annotators are already enabled')
        import omnigibson as og
        import omni.replicator.core as rep
        for role, sensor in self.sensors.items():
            with og.sim.editing_usd():
                annotator = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast', device='cpu')
                annotator.attach([sensor._render_product])
            self.annotators[role] = annotator
            for _ in range(4):
                og.sim.render()

    def __call__(self):
        if set(self.annotators) != set(ROLES):
            raise RuntimeError('Visibility audit annotators are not initialized')
        self.raw = {role: annotator.get_data() for role, annotator in self.annotators.items()}
        cameras = {role: mesh_pixels(value['data'], value['info']['idToLabels'], self.target.prim_path)
                   for role, value in self.raw.items()}
        row = {'object_name': self.target.name,
               'base_position': array(self.robot.get_position_orientation()[0]).tolist(),
               'target_position': array(self.target.get_position_orientation()[0]).tolist(),
               'target_visible': any(camera['visible'] for camera in cameras.values()),
               'visibility_evidence': {'method': 'rendered_mesh_instance_ids_exact_USD_object_subtree',
                                       'annotator': 'instance_id_segmentation_fast', 'cameras': cameras}}
        return evaluate_approach(row)

    def save_raw(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for role, value in self.raw.items():
            np.save(directory / (role + '_mesh_ids.npy'), value['data'], allow_pickle=False)
            save_json(directory / (role + '_id_to_prim.json'), value['info']['idToLabels'])
