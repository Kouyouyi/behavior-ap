"""Platform helpers for the standalone simulator; no Isaac or policy imports."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def runtime_python():
    return os.environ.get('BEHAVIOR_PYTHON', sys.executable)


def asset_root():
    return Path(os.environ.get('BEHAVIOR_ASSET_ROOT', os.environ.get(
        'OMNIGIBSON_DATA_PATH', '/root/download/datasets/BEHAVIOR-1K'))).expanduser()


def output_root():
    return Path(os.environ.get('BEHAVIOR_OUTPUT_ROOT', str(Path.cwd() / 'outputs'))).expanduser().resolve()


def gpu_lock_path():
    return Path(os.environ.get('BEHAVIOR_GPU_LOCK_PATH', '/tmp/behavior-ap-gpu.lock')).expanduser().resolve()


def runtime_env(*, headless):
    env = dict(os.environ)
    env.update(OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_HEADLESS='1' if headless else '0',
               PYTHONUNBUFFERED='1', OMNIGIBSON_DATA_PATH=str(asset_root()))
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    if not headless:
        env.setdefault('DISPLAY', ':1')
        candidate = Path(f'/run/user/{os.getuid()}/gdm/Xauthority')
        if 'XAUTHORITY' not in env and candidate.is_file():
            env['XAUTHORITY'] = str(candidate)
    return env


def needs_privilege():
    return (os.geteuid() != 0 and os.environ.get('BEHAVIOR_NO_SUDO') != '1'
            and not os.access(asset_root(), os.R_OK | os.X_OK))


def privilege_command(module, argv, env):
    """Use an explicitly configured/local controlled helper, otherwise sudo.

    No credentials are read here. Environment variables are passed as argv to
    env, not interpolated as shell code. GUI authentication is inherited.
    """
    helper = os.environ.get('BEHAVIOR_SUDO_RUN')
    if helper is None:
        local = Path.home() / '.codex/skills/environment/scripts/sudo_run.sh'
        helper = str(local) if local.is_file() else None
    command = [runtime_python(), '-m', module, *argv]
    admitted = ('DISPLAY', 'XAUTHORITY', 'PYTHONPATH', 'OMNI_KIT_ACCEPT_EULA',
                'OMNIGIBSON_HEADLESS', 'OMNIGIBSON_DATA_PATH', 'BEHAVIOR_ASSET_ROOT',
                'BEHAVIOR_PYTHON', 'BEHAVIOR_OUTPUT_ROOT', 'BEHAVIOR_GPU_LOCK_PATH',
                'BEHAVIOR_NO_SUDO')
    assignments = [key + '=' + env[key] for key in admitted if key in env]
    if helper:
        if not Path(helper).is_file():
            raise ValueError(f'Configured privilege helper does not exist: {helper}')
        return ['bash', helper, 'env', *assignments, *command]
    return ['sudo', 'env', *assignments, *command]


def gpu_status():
    """Check all GPU compute processes and GPU 0 memory, without changing them."""
    try:
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        free = int(memory.stdout.splitlines()[0]) if memory.returncode == 0 else 0
        details = processes.stdout.strip() or processes.stderr.strip()
        return processes.returncode == 0 and not processes.stdout.strip() and free >= 20000, details, free
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), 0


class GpuBusy(RuntimeError):
    pass


@contextmanager
def gpu_lease(*, wait=True):
    lock = gpu_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as lease:
        announced = False
        while True:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not wait:
                    raise GpuBusy('共享 GPU 锁正在使用，未启动仿真。')
                if not announced:
                    print('等待共享 GPU 锁；Ctrl+C 可取消。', flush=True)
                    announced = True
                time.sleep(2)
        announced = False
        while True:
            available, processes, memory = gpu_status()
            if available:
                break
            if not wait:
                raise GpuBusy(f'GPU 忙或空闲显存不足 20000 MiB（当前 {memory} MiB），未启动仿真。')
            if not announced:
                print(f'等待 GPU 空闲（当前 {memory} MiB）。\n{processes}', flush=True)
                announced = True
            time.sleep(10)
        yield lease


def verify_inherited_lease(fd):
    if fd is None:
        raise ValueError('Simulator child must inherit its parent GPU lease')
    try:
        inherited, expected = os.fstat(fd), gpu_lock_path().stat()
        if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Inherited GPU lease refers to a different file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise ValueError('Invalid inherited GPU lease') from exc


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def record_error(output, stage, error, *, finalization=False):
    output = Path(output)
    record = {'stage': stage, 'type': type(error).__name__, 'message': str(error),
              'automatic_replay': False}
    target = output / ('finalization_errors.json' if finalization else 'error.json')
    if finalization:
        previous = json.loads(target.read_text()) if target.exists() else {'errors': []}
        previous['errors'].append(record)
        save_json(target, previous)
    elif not target.exists():
        save_json(target, record)
    if finalization and not (output / 'error.json').exists():
        save_json(output / 'error.json', record)


def wait_child(process):
    """Cancel only our child process group; never terminate other GPU owners."""
    try:
        return process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return process.wait()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 130


def check_exit(output, returncode, *, viewer=False):
    output = Path(output)
    if returncode:
        return returncode
    for name in ('error.json', 'finalization_errors.json', 'viewer_error.json'):
        if (output / name).exists():
            print(f'运行记录包含错误：{output / name}', file=sys.stderr)
            return 1
    try:
        if viewer:
            ready = json.loads((output / 'viewer_ready.json').read_text())
            closed = json.loads((output / 'viewer_closed.json').read_text())
            if ready.get('status') != 'ready' or closed.get('status') != 'closed':
                raise ValueError('Incomplete GUI initialization/finalization')
        else:
            result = json.loads((output / 'sim/result.json').read_text())
            closed = json.loads((output / 'episode_closed.json').read_text())
            if result.get('errors') or not isinstance(result.get('success'), bool) or closed.get('status') != 'closed':
                raise ValueError('Simulator initialization/finalization failed')
    except (OSError, ValueError, TypeError) as exc:
        print(f'缺少有效终止证据：{output}: {exc}', file=sys.stderr)
        return 1
    return 0
