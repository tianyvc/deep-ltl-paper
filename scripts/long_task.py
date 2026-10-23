"""Linux background jobs in the active wty_LTL environment; no shell execution."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def supervise(job):
    metadata = json.loads((job / 'job.json').read_text())
    stopped = []
    child = None
    def stop(signum, frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state = dict(state='starting', pid=os.getpid(), identity=identity(os.getpid()), started=now())
    try:
        with (job / 'output.log').open('a', buffering=1) as log:
            child = subprocess.Popen(metadata['command'], cwd=metadata['cwd'], env=dict(os.environ, **metadata['overrides']), stdout=log, stderr=subprocess.STDOUT)
            state.update(state='running', child_pid=child.pid)
            if stopped:
                child.terminate()
            while child.poll() is None:
                state['heartbeat'] = now()
                write(job / 'status.json', state)
                time.sleep(2)
            state.update(state='interrupted' if stopped else ('completed' if child.returncode == 0 else 'failed'), exit_code=child.returncode, ended=now())
    except Exception as exc:
        state.update(state='failed', error=repr(exc), ended=now(), exit_code=1)
    write(job / 'status.json', state)
    return 0 if state['state'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'stop', '_supervise'])
    parser.add_argument('--job-dir', required=True, type=Path)
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES, e.g. 2 or a GPU UUID; omit for CPU')
    arguments = sys.argv[1:]
    split = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:split])
    command = arguments[split + 1:]
    if sys.platform != 'linux' or Path(sys.prefix).name != 'wty_LTL':
        parser.error('Run on the remote Linux server in conda environment wty_LTL')
    job = args.job_dir.resolve()
    if args.action == '_supervise':
        return supervise(job)
    if args.action == 'start':
        if not command:
            parser.error('Provide a Python script/arguments after --')
        if command[0] in ['python', 'python3']:
            command = command[1:]
        command = [sys.executable, *command]
        job.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(PYTHONHASHSEED='0', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONPATH=os.pathsep.join([str(Path('src').resolve()), str(Path('src/envs/zones/safety-gymnasium').resolve())]), MUJOCO_GL='egl', OMP_NUM_THREADS='2')
        if args.gpu is not None:
            environment['CUDA_VISIBLE_DEVICES'] = args.gpu
        overrides = {key: environment[key] for key in ['PYTHONHASHSEED','PYTHONNOUSERSITE','PYTHONUNBUFFERED','PYTHONPATH','MUJOCO_GL','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES'] if key in environment}
        write(job / 'job.json', dict(command=command, cwd=str(Path.cwd()), created=now(), python=sys.executable, conda_prefix=sys.prefix, overrides=overrides))
        with (job / 'supervisor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_supervise', '--job-dir', str(job)], env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(job / 'launch.json', dict(pid=process.pid, identity=identity(process.pid), created=now()))
        print(json.dumps(dict(job_dir=str(job), supervisor_pid=process.pid, log=str(job / 'output.log'))))
        return 0
    launch = json.loads((job / 'launch.json').read_text())
    state = json.loads((job / 'status.json').read_text()) if (job / 'status.json').exists() else dict(state='starting')
    alive = identity(launch['pid']) is not None and identity(launch['pid']) == launch['identity']
    if args.action == 'stop':
        if not alive:
            parser.error('Supervisor is no longer the recorded live process; no signal sent')
        os.kill(launch['pid'], signal.SIGTERM)
        print('Stop requested; wait for the current episode/update and check status again.')
    else:
        if not alive and state['state'] in ['running', 'starting']:
            state['state'] = 'stale_or_crashed'
        print(json.dumps(dict(**state, supervisor_alive=alive, job_dir=str(job)), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
