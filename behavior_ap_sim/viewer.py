"""Native mouse-interactive Isaac inspector for registered radio scenes."""
import argparse
import json
import math
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


def gui(output, seconds=0):
    output = Path(output)
    config = json.loads((output / 'viewer_config.json').read_text())
    registered_scene_identity(config)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('GUI duration must be finite and nonnegative')
    from .environment import RadioEnvironment
    from omnigibson.macros import gm
    gm.GUI_VIEWPORT_ONLY = False
    gm.REMOTE_STREAMING = None
    env = RadioEnvironment(output / 'sim', variant=config['variant'], max_steps=10000,
                           headless=False, write_video=True, render_profile='baseline')
    primary_error = None
    try:
        env.start()
        b = env.backend
        og = b.og
        og.sim.pause()
        # Intentional manual editing is isolated to this inspector process and
        # the USD session layer. Original scene assets are not saved.
        og.sim._disable_usd_guard()
        og.sim.stage.SetEditTarget(og.sim.stage.GetSessionLayer())
        import omni.ui as ui
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError('Isaac did not create a native viewport')
        viewport.camera_path = og.sim.viewer_camera.prim_path
        og.app.set_setting('/app/window/title', f"BEHAVIOR AP | radio | {config['variant'].upper()}")
        for name in ('Viewport', 'Stage', 'Property', 'Main ToolBar'):
            window = ui.Workspace.get_window(name)
            if window is not None:
                window.visible = True
        request = {'focus': None, 'close': False}

        def focus(kind):
            camera = og.sim.viewer_camera
            if kind == 'robot':
                name = b.evaluator.robot_camera_names['head'].split('::', 1)[1]
                position, orientation = b.robot.sensors[name].get_position_orientation()
                camera.set_position_orientation(position=position, orientation=orientation)
                return
            target = b.state.env_accessor.object_scope[config['target_key']]
            ref = b._array(target.get_position_orientation()[0]).copy()
            robot = b._array(b.robot.get_position_orientation()[0])
            direction = robot - ref
            direction[2] = 0
            length = b.np.linalg.norm(direction)
            direction = direction / length if length > .05 else b.np.array([1., 0., 0.])
            eye = ref + 2.6 * direction
            eye[2] = max(1.3, min(float(ref[2]) + 1.1, 2.3))
            forward = ref - eye
            forward /= b.np.linalg.norm(forward)
            right = b.np.cross(forward, [0., 0., 1.])
            right /= b.np.linalg.norm(right)
            up = b.np.cross(right, forward)
            quat = b.Rotation.from_matrix(b.np.column_stack([right, up, -forward])).as_quat()
            camera.set_position_orientation(position=eye.tolist(), orientation=quat.tolist())
            omni.usd.get_context().get_selection().set_selected_prim_paths([target.prim_path], False)

        panel = ui.Window('Radio Scene Inspector', width=440, height=220)
        with panel.frame:
            with ui.VStack(spacing=6):
                ui.Label(config['task_name'], word_wrap=True)
                ui.Label(f"{config['variant'].upper()} | instance {config['instance_id']} | seed {config['seed']}")
                ui.Label('Right-drag: look; middle-drag: pan; wheel: zoom.\nSelect object: W move, E rotate, F frame.\nStarts paused. Edits remain in the session layer.', word_wrap=True)
                with ui.HStack(spacing=8):
                    ui.Button('Focus target', clicked_fn=lambda: request.update(focus='target'))
                    ui.Button('Robot viewpoint', clicked_fn=lambda: request.update(focus='robot'))
                    ui.Button('Close', clicked_fn=lambda: request.update(close=True))
        focus('target')
        for _ in range(8):
            og.app.update()
        save_json(output / 'viewer_ready.json', {
            'status': 'ready', 'scene': config, 'native_gui': True, 'paused_initially': og.sim.is_paused(),
            'manual_usd_editing_enabled': True, 'asset_edits_saved': False,
            'edit_layer': og.sim.stage.GetSessionLayer().identifier,
            'evaluation_run': False, 'model_inference': False,
        })
        started = time.monotonic()
        while og.app.is_running() and not b._stop_event.is_set() and not request['close']:
            tick = time.monotonic()
            if request['focus']:
                focus(request['focus'])
                request['focus'] = None
            og.app.update()
            if og.sim.is_paused():
                og.sim.usdrt_stage.SynchronizeToFabric()
            if seconds and time.monotonic() - started >= seconds:
                break
            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        save_json(output / 'viewer_closed.json', {'status': 'closed', 'evaluation_run': False,
                                                 'manual_interaction': True, 'asset_edits_saved': False})
    except BaseException as exc:
        primary_error = exc
        record_error(output, 'viewer', exc)
        raise
    finally:
        # start() may launch Isaac and then fail before returning a backend.
        # Always finalize the environment, even before the local og assignment
        # or after the native window has stopped. Preserve the original error
        # if shutdown raises SystemExit(0), as some Isaac versions do.
        cleanup_error = None
        try:
            try:
                env.close('viewer_error' if primary_error is not None else 'viewer_closed')
            except BaseException as exc:
                cleanup_error = exc
                record_error(output, 'close', exc, finalization=True)
        finally:
            try:
                env.shutdown()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    record_error(output, 'shutdown', exc, finalization=True)
                if cleanup_error is None:
                    cleanup_error = exc
            except BaseException as exc:
                record_error(output, 'shutdown', exc, finalization=True)
                if cleanup_error is None:
                    cleanup_error = exc
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def launch(args, config, output):
    env = runtime_env(headless=False)
    if not env.get('DISPLAY') and not env.get('WAYLAND_DISPLAY'):
        raise ValueError('请在本地图形桌面运行，或设置 DISPLAY/XAUTHORITY')
    if env.get('XAUTHORITY') and not Path(env['XAUTHORITY']).is_file():
        raise ValueError('配置的 XAUTHORITY 文件不存在')
    with gpu_lease(wait=args.wait) as lease:
        if output.exists():
            raise ValueError(f'输出目录已存在：{output}')
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'viewer_config.json', config)
        print(f"打开 {config['task_name']}/{config['variant']}\n日志：{output / 'viewer.log'}", flush=True)
        command = [runtime_python(), '-m', 'behavior_ap_sim.viewer', '--_gui-child', str(output),
                   '--_lease-fd', str(lease.fileno()), '--smoke-seconds', str(args.smoke_seconds)]
        with (output / 'viewer.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, pass_fds=(lease.fileno(),),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            returncode = wait_child(child)
        return check_exit(output, returncode, viewer=True)


def parser():
    p = argparse.ArgumentParser(description='打开原生可交互 Isaac 窗口，默认暂停物理。', allow_abbrev=False)
    p.add_argument('task', nargs='?')
    p.add_argument('variant', nargs='?', choices=['original', 'ood'], default='ood')
    p.add_argument('--list', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--no-wait', action='store_false', dest='wait', default=True)
    p.add_argument('--smoke-seconds', type=float, default=0)
    p.add_argument('--_gui-child', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--_lease-fd', type=int, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.smoke_seconds) or args.smoke_seconds < 0:
        p.error('--smoke-seconds 必须是非负有限数')
    if args._gui_child:
        if args.task or args.list or args.dry_run:
            p.error('内部 GUI 参数不能混用公共入口选项')
        verify_inherited_lease(args._lease_fd)
        return gui(args._gui_child, args.smoke_seconds)
    if args._lease_fd is not None:
        p.error('--_lease-fd 仅允许内部 GUI 子进程使用')
    if args.list:
        for row in registered_tasks():
            print(f"{row['task_index']}  {row['task_name']}  {', '.join(row['variants'])}")
        return 0
    if not args.task:
        p.error('请指定已注册任务，如 ./scene radio original')
    try:
        config = resolve_registered(args.task, args.variant)
        suffix = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        output = (args.output or output_root() / 'viewer' / f'{args.variant}_{suffix}').expanduser().resolve()
        if output.exists():
            raise ValueError(f'输出目录已存在：{output}')
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        available, processes, memory = gpu_status()
        print(json.dumps({'scene': config, 'output': str(output), 'will_start_simulation': False,
                          'native_gui': True, 'gpu_lock': str(gpu_lock_path()),
                          'gpu_available': available, 'gpu_memory_free_mib': memory,
                          'gpu_processes': processes}, ensure_ascii=False, indent=2))
        return 0
    if needs_privilege():
        normalized = [args.task, args.variant, '--output', str(output), '--smoke-seconds', str(args.smoke_seconds)]
        if not args.wait:
            normalized.append('--no-wait')
        return subprocess.call(privilege_command('behavior_ap_sim.viewer', normalized, runtime_env(headless=False)), cwd=ROOT)
    def cancel(_signal, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        return launch(args, config, output)
    except GpuBusy as exc:
        print(str(exc), file=sys.stderr)
        return 75
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
