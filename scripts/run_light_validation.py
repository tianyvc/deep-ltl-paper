"""Bounded, predeclared baseline/hv_track pilot; does not certify paper gains."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback

def dump(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def execute(command,log,timeout,environment):
    started=time.monotonic()
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise TimeoutError('Hard time limit: '+str(command))
    if code:raise RuntimeError('Exit '+str(code)+': '+str(log))
    return time.monotonic()-started

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--devices',nargs=2,default=['0','1'])
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--train-seconds',type=int,default=360)
    p.add_argument('--eval-seconds',type=int,default=240)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH='src:src/envs/zones/safety-gymnasium',MUJOCO_GL='egl',PYTHONHASHSEED='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    plan=dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),purpose='Fixed-budget pilot. Report every seed; no guarantee or selection of positive gains.',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in pathlib.Path('src').rglob('*.py') if 'safety-gymnasium' not in x.parts},settings=vars(args)|{'output':str(args.output)},environments=['PointLtl2-v0','LetterEnv-v0'],seeds=[1,2,3],configurations=['baseline','hv_track'],training_changes='Only bounded budget/batch/process settings. Original reward, curriculum and episode horizons retained.',evaluation='Three finite paper templates, 10 episodes each, reset seeds10001..10010; same seeds per method. Training seeds are statistical units.',criteria='Primary: matched-step Zone paired SR; learning and transfer diagnostics secondary. Unequal steps/incomplete evaluation cannot support a controlled gain claim.')
    dump(args.output/'plan.json',plan)
    (args.output/'gpu-before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    for name,command in [('transition',[sys.executable,'tests/test_curriculum_transition.py']),('contracts',[sys.executable,'tests/test_tlt_curriculum.py']),('mapping',[sys.executable,'tests/test_paper_mapping.py','--output',str(args.output/'mapping.json')]),('upstream',[sys.executable,'scripts/verify_tlt_upstream.py','--output',str(args.output/'upstream.json')])]:
        execute(command,args.output/(name+'.log'),180,env)
    def worker(configuration,physical):
        local=dict(env,CUDA_VISIBLE_DEVICES=physical);rows=[]
        for environment in plan['environments']:
            for seed in plan['seeds']:
                slug=('zone' if environment=='PointLtl2-v0' else 'letter')+'-'+configuration+'-s'+str(seed)
                run=args.output/slug;row=dict(id=slug,configuration=configuration,environment=environment,seed=seed,physical_gpu=physical,state='running')
                dump(args.output/(slug+'.state.json'),row)
                try:
                    command=[sys.executable,'src/train/train_tlt.py','--env',environment,'--configuration',configuration,'--output',str(run),'--steps',str(args.steps),'--seed',str(seed),'--device','cuda:0','--num-procs','4','--rollout-steps','512','--epochs','2','--batch-size','256','--threads','2','--max-seconds',str(args.train_seconds)]
                    row['train_command']=command;dump(args.output/(slug+'.state.json'),row)
                    row['train_seconds']=execute(command,args.output/(slug+'.train.log'),args.train_seconds+180,local)
                    row['progress']=json.loads((run/'progress.json').read_text())
                    metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
                    row['training_updates']=len(metrics);row['training_metrics_first']=metrics[0];row['training_metrics_last']=metrics[-1]
                    command=[sys.executable,'scripts/evaluate_tlt.py','--checkpoint',str(run/'status.pth'),'--output',str(run/'evaluation'),'--episodes','10','--seed','10001','--device','cuda:0','--max-seconds',str(args.eval_seconds)]
                    row['eval_command']=command
                    row['eval_seconds']=execute(command,args.output/(slug+'.eval.log'),args.eval_seconds+120,local)
                    row['evaluation']=json.loads((run/'evaluation/summary.json').read_text());row['state']='completed'
                except Exception as exc:row.update(state='failed',error=repr(exc),traceback=traceback.format_exc())
                dump(args.output/(slug+'.state.json'),row);rows.append(row);print(json.dumps({k:row[k] for k in ['id','state']}),flush=True)
        return rows
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,configuration,physical) for configuration,physical in zip(plan['configurations'],args.devices)]
        for future in concurrent.futures.as_completed(futures):rows.extend(future.result())
    rows.sort(key=lambda r:r['id']);dump(args.output/'runs.json',rows)
    dump(args.output/'status.json',dict(state='completed' if all(r['state']=='completed' for r in rows) else 'failed',runs=len(rows),failures=sum(r['state']=='failed' for r in rows),finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('ALL_RUNS_FINISHED',flush=True)
if __name__=='__main__':main()
