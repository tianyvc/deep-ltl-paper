"""Resource-limited, restartable experiment DAG; invoke through long_task.py.

The manifest and frozen source hashes are immutable. SIGTERM pauses dispatch,
allowing already started workers to finish and save their results.
"""
import argparse
import datetime as dt
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from long_task import identity, now, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate(task):
    """Require scientific outputs, not merely a zero process exit code."""
    output = Path(task['output'])
    if task['kind'] in ('train', 'pilot'):
        progress = read(output / 'progress.json')
        assert progress['state'] == 'completed' and progress['steps'] >= task['steps']
        assert (output / 'status.pt').stat().st_size > 0
        metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        import math
        assert metrics and all(math.isfinite(m[k]) for m in metrics for k in ('loss','value_loss','gradient_norm'))
        if task['kind'] == 'pilot':
            assert len(metrics) >= 2, 'Pilot needs at least two real updates'
        return {'steps': progress['steps'], 'updates': progress['updates'], 'checkpoint_sha256': digest(output / 'status.pt'), 'peak_cuda_bytes': max(m.get('cuda_peak_allocated_bytes') or 0 for m in metrics)}
    if task['kind'] == 'evaluation':
        result = read(output / task['result'])
        config = read(output / 'config.json')
        assert result['weights_unchanged'] and result['checkpoint_steps'] >= task['steps']
        assert result['checkpoint_sha256'] == digest(output / 'status.pt')
        assert result['split'] == task['split'] and result['deterministic']
        assert result['task_manifest_sha256'] == config['task_manifest']['sha256']
        assert set(result['per_formula']) == set(config['task_manifest'][task['split']])
        for formula in result['per_formula']:
            rows = [r for r in result['episodes'] if r['formula'] == formula]
            assert len(rows) == task['episodes'] and len({r['seed'] for r in rows}) == task['episodes']
        return {'result_sha256': digest(output / task['result']), 'episodes': len(result['episodes'])}
    if task['kind'] == 'minimal_evaluation':
        result = read(output / 'evaluation-minimal.json')
        spec = read(task['spec'])
        assert result['weights_unchanged'] and result['deterministic']
        assert result['checkpoint_steps'] >= task['steps']
        assert result['checkpoint_sha256'] == digest(task['checkpoint'])
        assert result['spec_sha256'] == digest(task['spec'])
        assert result['environment'] == task['env'] and result['configuration'] == task['configuration'] and result['training_seed'] == task['seed']
        assert set(result['per_formula']) == {item['formula'] for item in spec['tasks']}
        for item in spec['tasks']:
            rows = [row for row in result['episodes'] if row['formula'] == item['formula']]
            assert len(rows) == item['episodes'] and len({row['seed'] for row in rows}) == item['episodes']
        return {'result_sha256': digest(output / 'evaluation-minimal.json'), 'episodes': len(result['episodes'])}
    if task['kind'] == 'curve':
        result = read(output / 'validation-curve-index.json')
        assert result['points']
        for point in result['points']:
            assert digest(point['result']) == point['sha256']
        return {'validation_curve_points': len(result['points'])}
    if task['kind'] == 'summary':
        result = read(output)
        assert result['groups']
        return {'result_sha256': digest(output)}
    raise ValueError(task['kind'])


def worker_state(job):
    launch = read(job / 'launch.json')
    state = read(job / 'status.json') if (job / 'status.json').exists() else {'state':'starting'}
    alive = identity(launch['pid']) == launch['identity'] and launch['identity'] is not None
    if not alive and state['state'] in ('starting', 'running'):
        state['state'] = 'stale_or_crashed'
    return state


def free_gpus():
    command = ['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    rows = subprocess.check_output(command,text=True,timeout=15).strip().splitlines()
    return {int(v[0]): {'free_mib':int(v[1]),'utilization':int(v[2])} for v in (row.split(',') for row in rows)}


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    root = manifest_path.parent
    ids = [task['id'] for task in manifest['tasks']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs in manifest; no worker started')
    known = set(ids)
    if any(dep not in known or dep == task['id'] for task in manifest['tasks'] for dep in task['depends']):
        raise ValueError('Invalid/self dependency in manifest')
    lock = (root / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'state.json'
    state = read(state_path) if state_path.exists() else {'tasks':{}}
    manifest_hash = digest(manifest_path)
    assert state.get('manifest_sha256', manifest_hash) == manifest_hash, 'Manifest changed'
    state.update(manifest_sha256=manifest_hash, pid=os.getpid(), identity=identity(os.getpid()), started=now())
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    for task in manifest['tasks']:
        state['tasks'].setdefault(task['id'], {'state':'pending'})
    def event(kind, **fields):
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=now(), event=kind, **fields))+'\n')
    event('scheduler_started', pid=os.getpid())
    while True:
        active = []
        for task in manifest['tasks']:
            status = state['tasks'][task['id']]
            if status['state'] != 'running':
                continue
            worker = None
            try:
                worker = worker_state(Path(status['job_dir']))
                status['worker_state'] = worker['state']
                if worker['state'] == 'completed':
                    status.update(state='completed', ended=now(), evidence=validate(task))
                    event('task_completed', task=task['id'], evidence=status['evidence'])
                elif worker['state'] in ('failed', 'interrupted', 'stale_or_crashed'):
                    status.update(state='failed', ended=now(), error=worker)
                    event('task_failed', task=task['id'], error=worker)
                else:
                    active.append(status)
            except Exception as exc:
                # A missing output is a failure; a transient status-file read is retried.
                if worker is not None and worker.get('state') == 'completed':
                    status.update(state='failed', error=repr(exc), ended=now())
                    event('output_validation_failed', task=task['id'], error=repr(exc))
                else:
                    active.append(status)
                    status['last_poll_error'] = repr(exc)
        integrity = all(Path(path).is_file() and digest(path) == checksum for path,checksum in manifest['frozen_hashes'].items())
        if manifest.get('dispatch_deadline_utc') and dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(manifest['dispatch_deadline_utc']):
            stopping.append(True)
            state['time_budget_reached'] = True
            state['error'] = 'Minimal-study wall-time dispatch budget reached; no new tasks. Active tasks may finish.'
        if not integrity:
            stopping.append(True)
            state['error'] = 'Frozen source integrity mismatch; dispatch halted'
        for task in manifest['tasks']:
            status = state['tasks'][task['id']]
            if status['state'] != 'pending':
                continue
            deps = [state['tasks'][d]['state'] for d in task['depends']]
            if any(d in ('failed','blocked') for d in deps):
                status.update(state='blocked', reason='Dependency failed')
        counts = Counter(v['state'] for v in state['tasks'].values())
        if stopping or (root / 'PAUSE').exists():
            state['state'] = 'pausing' if active else 'paused'
        elif not counts['pending'] and not counts['running']:
            state['state'] = 'completed' if not counts['failed'] and not counts['blocked'] else 'finished_with_failures'
        else:
            state['state'] = 'running'
        state.update(heartbeat=now(), counts=dict(counts))
        write(state_path,state)
        if state['state'] in ('completed','finished_with_failures','paused'):
            event('scheduler_stopped',state=state['state'])
            return 0 if state['state'] in ('completed','paused') else 1
        if state['state'] == 'running':
            try:
                resources = free_gpus()
                state['resources'] = resources
                occupied = {s['gpu'] for s in active}
                available = [g for g in manifest['gpus'] if g not in occupied and resources[g]['free_mib'] >= manifest['min_free_mib']]
                for task in manifest['tasks']:
                    if len(active) >= manifest['max_workers'] or not available:
                        break
                    status = state['tasks'][task['id']]
                    if status['state'] != 'pending' or any(state['tasks'][d]['state'] != 'completed' for d in task['depends']):
                        continue
                    gpu = available.pop(0)
                    attempt = status.get('attempt',0)+1
                    job = root / 'jobs' / f"{task['id']}-attempt-{attempt}"
                    command = [sys.executable, str(Path(__file__).with_name('long_task.py')), 'start','--job-dir',str(job),'--gpu',str(gpu),'--',*task['argv'],*(['--resume'] if status.get('resume') else [])]
                    result = subprocess.run(command,cwd=manifest['source_root'],text=True,capture_output=True,timeout=60)
                    if result.returncode:
                        status.update(state='failed', error=result.stderr, attempt=attempt)
                        event('launch_failed',task=task['id'],error=result.stderr)
                        continue
                    status.update(state='running',gpu=gpu,job_dir=str(job),started=now(),attempt=attempt)
                    active.append(status)
                    event('task_started',task=task['id'],gpu=gpu,job_dir=str(job))
                    write(state_path,state)
                if not active:
                    state['waiting_reason'] = 'Ready tasks wait for >= min_free_mib on an eligible GPU, or dependencies'
            except Exception as exc:
                state['resource_error'] = repr(exc)
                event('dispatch_error',error=repr(exc))
        write(state_path,state)
        time.sleep(manifest.get('poll_seconds',10))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','status','pause','unpause','retry-failed'])
    parser.add_argument('--manifest',required=True,type=Path)
    args=parser.parse_args()
    if sys.platform != 'linux' or Path(sys.prefix).name != 'wty_LTL':
        parser.error('Use remote Linux conda wty_LTL')
    root=args.manifest.resolve().parent
    if args.action == 'run':
        return run(args.manifest)
    if args.action == 'pause':
        (root/'PAUSE').touch(exist_ok=True)
        print('Dispatch paused; active workers will finish. Check status for paused.')
    elif args.action == 'unpause':
        (root/'PAUSE').unlink(missing_ok=True)
        print('Pause flag removed; restart scheduler if it already exited.')
    elif args.action == 'retry-failed':
        lock=(root/'scheduler.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state=read(root/'state.json')
        for value in state['tasks'].values():
            if value['state']=='running':
                raise RuntimeError('Active workers remain; pause and wait before retrying')
        for task in read(args.manifest)['tasks']:
            value=state['tasks'][task['id']]
            if value['state'] in ('failed','blocked'):
                if value.get('job_dir'):
                    worker=worker_state(Path(value['job_dir']))
                    assert worker['state'] not in ('running','starting'), 'Worker is still active'
                    old=read(Path(value['job_dir'])/'status.json')
                    child=old.get('child_pid')
                    assert child is None or identity(child) is None, 'Possible orphan child remains; inspect manually'
                try:
                    evidence=validate(task)
                    value.update(state='completed', evidence=evidence)
                except Exception:
                    if task['kind'] in ('train','pilot') and (Path(task['output'])/'status.pt').exists():
                        value['resume']=True
                    value['state']='pending'
        state['state']='ready_to_restart'
        write(root/'state.json',state)
        print('Retry state prepared; use a new long_task job directory to restart the same manifest.')
    else:
        state=read(root/'state.json')
        alive=identity(state['pid']) == state['identity'] and state['identity'] is not None
        print(json.dumps({k:v for k,v in state.items() if k!='tasks'},indent=2))
        print('scheduler_alive=',alive)
        for key,value in state['tasks'].items():
            if value['state'] in ('running','failed'):
                print(key,json.dumps(value))
    return 0


if __name__=='__main__':
    sys.exit(main())
