"""Initialize a registered scene or execute an externally supplied action array."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .registry import registered_tasks, resolve_registered, registered_scene_identity
from .resources import (GpuBusy, ROOT, check_exit, gpu_lease, gpu_lock_path, gpu_status,
                        needs_privilege, output_root, privilege_command, record_error,
                        runtime_env, runtime_python, save_json, verify_inherited_lease, wait_child)


def load_actions(path):
    import numpy as np
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.npy' or not path.is_file():
        raise ValueError('动作文件必须是已存在的 .npy 文件')
    source = path.read_bytes()
    actions = np.load(io.BytesIO(source), allow_pickle=False)
    if (actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 23
            or actions.dtype.kind not in 'fiu' or not np.isfinite(actions).all()):
        raise ValueError('动作必须是非空、有限数值的 N×23 数组')
    with np.errstate(over='ignore', invalid='ignore'):
        actions = np.ascontiguousarray(actions, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise ValueError('动作转换为 float32 后超出有限数值范围')
    digest = hashlib.sha256(source).hexdigest()
    return actions, digest


def positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('必须为正整数') from None
    if number < 1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return number


def parser():
    p = argparse.ArgumentParser(description='独立 BEHAVIOR 仿真：初始化官方 ID 场景，或执行外部 N×23 动作文件。', allow_abbrev=False)
    p.add_argument('task', nargs='?')
    p.add_argument('variant', nargs='?', choices=['original', 'id', 'ood'], default='original')
    p.add_argument('--instance-id', type=int, help='官方实例编号，必须属于任务与 split 的官方列表')
    p.add_argument('--split', choices=['train', 'public_test'], help='官方数据划分，省略时使用注册任务的默认划分')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--actions', type=Path)
    p.add_argument('--initialize-only', action='store_true', help='无 --actions 时默认只初始化')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--capture-every', type=positive_int, default=8,
                   help='外部动作执行期间每 N 步保存观测，默认 8；结束时补存未保存的最终观测')
    terminal = p.add_mutually_exclusive_group()
    terminal.add_argument('--initial-visibility-audit', action='store_true')
    terminal.add_argument('--export-handoff', action='store_true')
    p.add_argument('--render-profile', choices=['baseline', 'async_video', 'fast'], default='baseline')
    p.add_argument('--no-wait', action='store_false', dest='wait', default=True)
    p.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--_lease-fd', type=int, help=argparse.SUPPRESS)
    return p


def scene_output_name(scene):
    """Include the complete selected instance identity in generated paths."""
    return f"{scene['task_name']}_{scene['variant']}_{scene.get('split') or 'registered'}_instance{scene['instance_id']}"


def print_catalog():
    tasks = registered_tasks()
    split_counts = {
        split: sum(len(task['instances'][split]) for task in tasks)
        for split in ('train', 'public_test')
    }
    ood_count = sum('ood' in task['variants'] for task in tasks)
    print(f"官方任务: {len(tasks)} | ID 任务: {sum('original' in task['variants'] for task in tasks)} | "
          f"ID 实例: {sum(split_counts.values())} "
          f"(train {split_counts['train']}, public_test {split_counts['public_test']}) | "
          f"房屋布局: {len({task['scene_model'] for task in tasks})} | 已注册 OOD: {ood_count}")
    for task in tasks:
        counts = ', '.join(f"{split}={len(ids)}" for split, ids in task['instances'].items())
        print(f"{task['task_index']:>2}  {task['task_name']}  {task['scene_model']}  "
              f"{counts}  variants={','.join(task['variants'])}")


def validate_job(job):
    registered_scene_identity(job['scene'])
    if job['render_profile'] not in ('baseline', 'async_video', 'fast'):
        raise ValueError('Unknown render profile')
    if type(job['max_steps']) is not int or job['max_steps'] < 1:
        raise ValueError('max_steps must be positive')
    if type(job.get('capture_every')) is not int or job['capture_every'] < 1:
        raise ValueError('capture_every must be a positive integer')
    for key in ('initialize_only', 'initial_visibility_audit', 'export_handoff'):
        if type(job[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if bool(job['actions']) == job['initialize_only']:
        raise ValueError('Choose initialization or external actions')
    if job['initial_visibility_audit'] and not job['initialize_only']:
        raise ValueError('Visibility auditing requires initialize-only mode')
    if job['initial_visibility_audit'] and job['export_handoff']:
        raise ValueError('Visibility auditing is terminal and cannot export a handoff')
    if job['initial_visibility_audit'] and job['scene'].get('target_key') is None:
        raise ValueError('Visibility auditing requires a separately registered target; general ID scenes have none')
    actions = None
    if job['actions']:
        actions, digest = load_actions(job['actions'])
        if digest != job['actions_sha256'] or len(actions) > job['max_steps']:
            raise ValueError('Action source changed or exceeds the explicit step budget')
    return actions


def execute_job(output):
    """GPU child only. Caller already verified the inherited lock descriptor."""
    output = Path(output)
    job = json.loads((output / 'run_config.json').read_text())
    actions = validate_job(job)
    from .environment import BehaviorEnvironment
    env = BehaviorEnvironment(output / 'sim', task=job['scene']['task_name'],
                           variant=job['scene']['variant'], instance_id=job['scene']['instance_id'],
                           split=job['scene'].get('split'),
                           max_steps=job['max_steps'], render_profile=job['render_profile'],
                           headless=True, write_video=True,
                           initial_visibility_audit=job['initial_visibility_audit'])
    reason = 'infrastructure_error'
    submitted = acknowledged = 0
    try:
        env.start()
        env.capture(output / 'initial')
        last_capture_step = env.steps
        capture_steps = [env.steps]
        if job['export_handoff']:
            env.save_handoff(output / 'initial_handoff')
        if actions is None:
            reason = 'initialize_only'
        else:
            import numpy as np
            np.save(output / 'actions.npy', actions, allow_pickle=False)
            with (output / 'execution.jsonl').open('w', buffering=1) as stream:
                for offset, action in enumerate(actions):
                    if env.done:
                        break
                    before = env.steps
                    stream.write(json.dumps({'event': 'submitted', 'offset': offset, 'step': before,
                                             'action': action.tolist(), 'time_ns': time.time_ns()}) + '\n')
                    submitted += 1
                    if not env.step(action):
                        break
                    stream.write(json.dumps({'event': 'ack', 'offset': offset, 'step': env.steps,
                                             'time_ns': time.time_ns()}) + '\n')
                    acknowledged += 1
                    if env.steps != before + 1:
                        raise RuntimeError('One submitted action did not advance exactly one task step')
                    if env.steps % job['capture_every'] == 0:
                        env.capture(output / 'capture' / f'{env.steps:08d}')
                        last_capture_step = env.steps
                        capture_steps.append(env.steps)
            if env.steps != last_capture_step:
                env.capture(output / 'final')
                capture_steps.append(env.steps)
            reason = env.termination_reason if env.done else 'external_actions_complete'
        save_json(output / 'execution_summary.json', {
            'external_actions': job['actions'], 'requested_actions': 0 if actions is None else len(actions),
            'submitted': submitted, 'acknowledged': acknowledged,
            'task_steps': env.steps, 'official_task_success': bool(env.success),
            'capture_every': job['capture_every'], 'capture_steps': capture_steps,
            'reason': reason, 'automatic_replay': False, 'model_inference': False,
        })
        if submitted != acknowledged:
            raise RuntimeError('An action lacks acknowledgement; refusing automatic replay')
    except BaseException as exc:
        record_error(output, 'execution', exc)
        raise
    finally:
        try:
            try:
                result = env.close(reason)
                save_json(output / 'episode_closed.json', {
                    'status': 'closed', 'task_steps': env.steps,
                    'official_task_success': bool(env.success),
                    'errors': result.get('errors', []),
                })
            except BaseException as exc:
                record_error(output, 'close', exc, finalization=True)
                raise
        finally:
            try:
                env.shutdown()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    record_error(output, 'shutdown', exc, finalization=True)
                raise
            except BaseException as exc:
                record_error(output, 'shutdown', exc, finalization=True)
                raise


def launch_job(args, job, output):
    with gpu_lease(wait=args.wait) as lease:
        if output.exists():
            raise ValueError(f'输出目录已存在，拒绝覆盖或重放：{output}')
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'run_config.json', job)
        print(f"{job['scene']['task_name']}/{job['scene']['variant']}\n输出：{output}", flush=True)
        command = [runtime_python(), '-m', 'behavior_ap_sim.cli', '--_worker', str(output),
                   '--_lease-fd', str(lease.fileno())]
        with (output / 'run.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=runtime_env(headless=True),
                                       pass_fds=(lease.fileno(),), start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            returncode = wait_child(process)
        return check_exit(output, returncode)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if args._worker:
        if (args.task or args.list or args.dry_run or args.actions
                or args.instance_id is not None or args.split is not None):
            p.error('内部 worker 参数不能混用公共入口选项')
        try:
            verify_inherited_lease(args._lease_fd)
            return execute_job(args._worker)
        except SystemExit as exc:
            if exc.code not in (None, 0) and args._worker.is_dir():
                record_error(args._worker, 'worker', exc)
            raise
        except BaseException as exc:
            if args._worker.is_dir():
                record_error(args._worker, 'worker', exc)
            raise
    if args._lease_fd is not None:
        p.error('--_lease-fd 仅允许内部 worker 使用')
    if args.list:
        print_catalog()
        return 0
    if not args.task:
        p.error('请指定已注册任务，如 ./simulate radio ood')
    try:
        scene = resolve_registered(args.task, 'original' if args.variant == 'id' else args.variant,
                                   instance_id=args.instance_id, split=args.split)
        if args.actions and args.initialize_only:
            raise ValueError('--actions 与 --initialize-only 不能同时使用')
        if args.actions and args.initial_visibility_audit:
            raise ValueError('--initial-visibility-audit 只能与初始化模式一起使用')
        actions, digest = load_actions(args.actions) if args.actions else (None, None)
        max_steps = args.max_steps if args.max_steps is not None else (len(actions) if actions is not None else 800)
        if max_steps < 1 or (actions is not None and len(actions) > max_steps):
            raise ValueError('--max-steps 必须为正且不少于外部动作数')
        suffix = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        output = (args.output or output_root() / scene['task_name'] / f'{scene_output_name(scene)}_{suffix}').expanduser().resolve()
        if output.exists():
            raise ValueError(f'输出目录已存在，拒绝覆盖或重放：{output}')
        job = {'scene': scene, 'max_steps': max_steps, 'render_profile': args.render_profile,
               'capture_every': args.capture_every,
               'initialize_only': actions is None, 'initial_visibility_audit': args.initial_visibility_audit,
               'export_handoff': args.export_handoff,
               'actions': str(args.actions.expanduser().resolve()) if args.actions else None,
               'actions_sha256': digest, 'model_inference': False}
        validate_job(job)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if args.dry_run:
        available, processes, memory = gpu_status()
        print(json.dumps({'job': job, 'output': str(output), 'will_start_simulation': False,
                          'gpu_lock': str(gpu_lock_path()), 'gpu_available': available,
                          'gpu_memory_free_mib': memory, 'gpu_processes': processes}, ensure_ascii=False, indent=2))
        return 0
    if needs_privilege():
        normalized = [scene['task_name'], scene['variant'], '--instance-id', str(scene['instance_id']),
                      '--output', str(output), '--max-steps', str(max_steps),
                      '--render-profile', args.render_profile, '--capture-every', str(args.capture_every)]
        if scene.get('split') is not None:
            normalized += ['--split', scene['split']]
        if args.actions:
            normalized += ['--actions', str(args.actions.expanduser().resolve())]
        for key in ('initialize_only', 'initial_visibility_audit', 'export_handoff'):
            if getattr(args, key):
                normalized.append('--' + key.replace('_', '-'))
        if not args.wait:
            normalized.append('--no-wait')
        return subprocess.call(privilege_command('behavior_ap_sim.cli', normalized, runtime_env(headless=True)), cwd=ROOT)
    def cancel(_signal, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        return launch_job(args, job, output)
    except GpuBusy as exc:
        print(str(exc), file=sys.stderr)
        return 75
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
