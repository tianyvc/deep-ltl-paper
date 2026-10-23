"""Read-only campaign health check. Does not launch, stop, or edit experiments."""
import argparse
from collections import Counter
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
from long_task import identity


def read(path):
    return json.loads(Path(path).read_text())


def age(timestamp):
    return (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(timestamp)).total_seconds()


def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'ticks':int(fields[11])+int(fields[12]),'identity':fields[19], 'state':fields[0]}
    except (FileNotFoundError,ProcessLookupError):
        return None


def tail_metrics(path):
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines()[-60:]:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows


def snapshot(manifest_path):
    manifest=read(manifest_path);root=manifest_path.parent;state=read(root/'state.json');tasks={t['id']:t for t in manifest['tasks']};active={}
    for key,value in state['tasks'].items():
        if value['state']!='running':continue
        task=tasks[key];job=Path(value['job_dir']);worker=read(job/'status.json') if (job/'status.json').exists() else {};launch=read(job/'launch.json');proc=process(worker.get('child_pid',0));rows=tail_metrics(Path(task['output'])/'metrics.jsonl') if task['kind'] in ('train','pilot') else []
        active[key]={'task':task,'status':value,'worker':worker,'supervisor_alive':identity(launch['pid'])==launch['identity'] and launch['identity'] is not None,'process':proc,'rows':rows,'metrics_age':time.time()-(Path(task['output'])/'metrics.jsonl').stat().st_mtime if rows else None}
    return manifest,state,active


def budget_progress(manifest,state):
    """Count capped training steps and evaluation resets, including audited imports."""
    trained=training_target=evaluated=evaluation_target=0
    for task in manifest['tasks']:
        status=state['tasks'][task['id']];output=Path(task['output'])
        if task['kind']=='train':
            target=task['steps'];training_target+=target
            if status['state']=='completed':done=target
            else:
                rows=tail_metrics(output/'metrics.jsonl');done=rows[-1]['steps'] if rows else 0
            trained+=min(target,max(0,done))
        elif task['kind']=='minimal_evaluation':
            target=sum(t['episodes'] for t in read(task['spec'])['tasks']);evaluation_target+=target
            progress=output/'evaluation-progress.json'
            done=target if status['state']=='completed' else (read(progress).get('episodes_done',0) if progress.exists() else 0)
            evaluated+=min(target,max(0,done))
    return trained,training_target,evaluated,evaluation_target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('experiments/reproduction/20261021-v3/manifest.json'))
    parser.add_argument('--sample-seconds',type=float,default=10)
    args=parser.parse_args()
    if sys.platform!='linux' or Path(sys.prefix).name!='wty_LTL':parser.error('Run remotely in wty_LTL')
    if not 0<=args.sample_seconds<=60:parser.error('Sample seconds must be 0..60')
    active_pointer=Path('experiments/reproduction/ACTIVE.json')
    if '--manifest' not in sys.argv and active_pointer.exists():
        args.manifest=Path(read(active_pointer)['manifest'])
    print('Manifest:',str(args.manifest),flush=True)
    _,_,first=snapshot(args.manifest)
    if args.sample_seconds:
        print(f'Sampling worker activity for {args.sample_seconds:g}s ...',flush=True);time.sleep(args.sample_seconds)
    manifest,state,active=snapshot(args.manifest);counts=Counter(v['state'] for v in state['tasks'].values());issues=[]
    scheduler_alive=identity(state['pid'])==state['identity'] and state['identity'] is not None
    heartbeat_age=age(state['heartbeat'])
    print('Checked at:',dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat())
    print(f"Campaign={state['state']} | scheduler_alive={scheduler_alive} | heartbeat_age={heartbeat_age:.0f}s")
    print('Tasks:', ' / '.join(f'{k}={counts[k]}' for k in ['completed','running','pending','failed','blocked']),f"/ total={len(manifest['tasks'])}")
    formal=[t for t in manifest['tasks'] if t['kind']=='train'];formal_done=sum(state['tasks'][t['id']]['state']=='completed' for t in formal)
    print(f'Formal training completed={formal_done}/{len(formal)}; task counts are NOT a wall-time percentage.')
    trained,target,evaluated,eval_target=budget_progress(manifest,state)
    if target:print(f'Training budget: completed_steps={trained:,}/{target:,} ({100*trained/target:.2f}%) | remaining_steps={target-trained:,}; capped per-run, includes imported checkpoints.')
    if eval_target:print(f'Evaluation budget: completed_episodes={evaluated:,}/{eval_target:,} ({100*evaluated/eval_target:.2f}%) | remaining_episodes={eval_target-evaluated:,}.')
    if manifest.get('dispatch_deadline_utc'):
        deadline=dt.datetime.fromisoformat(manifest['dispatch_deadline_utc']);hours=(deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()/3600
        print(f'Dispatch deadline (Beijing): {deadline.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()} | hours_until_deadline={max(0,hours):.1f}; stops NEW tasks, not a completion ETA.')
    if state['state'] == 'completed' and counts['completed'] == len(manifest['tasks']):
        print('Whole-batch ETA: 0h; all planned stages completed.')
    else:
        print('Whole-batch ETA: not yet calibrated; worker remaining times below exclude queued runs, evaluation and GPU waiting.')
    if counts['failed'] or counts['blocked']:issues.append('Failed/blocked tasks exist; inspect their logs.')
    if state['state']=='running' and (not scheduler_alive or heartbeat_age>120):issues.append('Scheduler missing or heartbeat older than 120s.')
    if state['state'] in ('paused','pausing','finished_with_failures'):issues.append('Campaign is paused/stopping or finished with failures.')
    if state.get('error'):issues.append(str(state['error']))
    for key,item in active.items():
        task=item['task'];worker=item['worker'];proc=item['process'];old=first.get(key);cpu_delta=None
        if proc and old and old['process'] and proc['identity']==old['process']['identity']:cpu_delta=proc['ticks']-old['process']['ticks']
        print('\n'+key+f" | physical_GPU={item['status'].get('gpu')} | worker={worker.get('state')} | supervisor_alive={item['supervisor_alive']} | child_alive={proc is not None and proc['state']!='Z'} | CPU_ticks_delta={cpu_delta}")
        if not item['supervisor_alive'] or worker.get('state') in ('failed','interrupted','stale_or_crashed'):issues.append(key+': worker unavailable or has ended; allow one dispatch cycle then recheck.')
        if worker.get('state')=='running' and (proc is None or proc['state']=='Z'):issues.append(key+': child process is missing.')
        if item['rows']:
            last=item['rows'][-1];step_delta=last['steps']-old['rows'][-1]['steps'] if old and old['rows'] else None
            print(f"  steps={last['steps']}/{task['steps']} ({100*last['steps']/task['steps']:.2f}%) | updates={last['update']} | metrics_age={item['metrics_age']:.0f}s | sample_steps_delta={step_delta}")
            print(f"  loss={last['loss']:.6g} | gradient_norm={last['gradient_norm']:.6g}")
            if not all(math.isfinite(last[k]) for k in ('loss','gradient_norm','value_loss')):issues.append(key+': nonfinite metrics.')
            compatible=[r for r in item['rows'] if r['steps']<=last['steps'] and r['elapsed_seconds']<=last['elapsed_seconds']]
            # Only use a suffix from this training process, across no elapsed reset.
            start=len(item['rows'])-1
            while start>0 and item['rows'][start-1]['elapsed_seconds']<item['rows'][start]['elapsed_seconds']:start-=1
            window=item['rows'][max(start,len(item['rows'])-21):]
            if len(window)>1:
                elapsed=last['elapsed_seconds']-window[0]['elapsed_seconds'];rate=(last['steps']-window[0]['steps'])/elapsed if elapsed>0 else 0
                if rate>0:print(f"  recent_speed={rate:.2f} product_steps/s | training_only_remaining={(task['steps']-last['steps'])/rate/3600:.2f}h")
            if item['metrics_age']>1800:issues.append(key+': no completed update for over 30min; inspect logs/process activity. Long episodes can take time.')
        elif task['kind']=='minimal_evaluation':
            progress_path=Path(task['output'])/'evaluation-progress.json'
            if progress_path.exists():
                evaluation=read(progress_path)
                print(f"  evaluation_episodes={evaluation['episodes_done']}/{evaluation['episodes_target']} | task={evaluation.get('task_id','finished')} | progress_age={time.time()-progress_path.stat().st_mtime:.0f}s")
                elapsed=evaluation.get('elapsed_seconds',0);done=evaluation['episodes_done'];remaining=evaluation['episodes_target']-done
                if elapsed>0 and done>0:print(f'  evaluation_only_remaining_rough={max(0,remaining)*elapsed/done/3600:.2f}h; task lengths may differ.')
            else:
                print('  Evaluation starting: strict checkpoint validation and environment compilation in progress.')
        else:
            print('  Phase has no completed-update metrics yet (startup/evaluation/summary). CPU activity is liveness, not proof of scientific progress.')
        print('  log='+str(Path(item['status']['job_dir'])/'output.log'))
    if state.get('time_budget_reached'):issues.append('Configured wall-time budget reached; no automatic full-study extension.')
    if state['state']=='running' and not active:print('\nWAIT: no active worker; check free GPU memory and pending dependencies.')
    for key,value in state['tasks'].items():
        if value['state']=='failed':print('FAILED:',key,'log=',str(Path(value.get('job_dir','.'))/'output.log'))
    if state['state']=='completed':
        for task in manifest['tasks']:
            if task['kind']=='summary' and not Path(task['output']).is_file():issues.append('Completed campaign summary is missing: '+task['output'])
            if task['kind']=='summary' and manifest.get('study')=='zone-main-minimal-v1' and not Path(task['output']).with_suffix('.csv').is_file():issues.append('Completed campaign summary CSV is missing.')
    if state['state']=='completed' and counts['completed']==len(manifest['tasks']) and not issues:
        print('\nALL_COMPLETED: all planned stages passed scheduler output checks. Ready for result review.');return 0
    if issues:
        print('\nCHECK_REQUIRED:')
        for issue in issues:print(' -',issue)
        return 2
    print('\nOK: no current health-check failure detected. Compare steps/updates across checks; heartbeat alone does not prove learning quality.')
    return 0


if __name__=='__main__':sys.exit(main())
