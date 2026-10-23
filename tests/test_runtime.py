"""Bounded integration matrix for maintained paper configurations; no SR claims."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--main-only',action='store_true',help='Refresh only the two primary-environment smoke checkpoints')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    configurations=['baseline','hv_only','msgnn_only','hv_track','msgnn_track','full_no_options']
    cases=[('PointLtl2-v0',c,128,64) for c in ['baseline','hv_track']]
    if not args.main_only:
        cases += [('FlatWorld-v0',c,32,16) for c in configurations]
        cases += [('LetterEnv-v0','hv_track',32,16)]
    for env,configuration,steps,rollout in cases:
        target=args.output/(env+'-'+configuration)
        command=[sys.executable,'src/train/train_tlt.py','--env',env,'--configuration',configuration,'--output',str(target),'--steps',str(steps),'--num-procs','1','--rollout-steps',str(rollout),'--epochs','2','--batch-size',str(min(32,rollout)),'--seed','1','--device','cpu','--threads','2']
        started=time.monotonic()
        log=args.output/(target.name+'.log')
        with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
        if result.returncode:raise RuntimeError('Failed: '+str(log)+'\n'+log.read_text()[-4000:])
        progress=json.loads((target/'progress.json').read_text());assert progress['state']=='completed' and progress['steps']>=steps
        checkpoint=torch.load(target/'status.pth',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(value).all() for value in checkpoint['model_state'].values())
        row=dict(env=env,configuration=configuration,steps=progress['steps'],updates=progress['updates'],elapsed_seconds=time.monotonic()-started,returncode=0,all_tensors_finite=True,command=command)
        rows.append(row);print(json.dumps(row),flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(dict(complete=False,cases=rows),indent=2))
        # Only two main-environment smoke checkpoints remain useful for evaluator checks.
        if env!='PointLtl2-v0':shutil.rmtree(target);log.unlink()
    args.report.write_text(json.dumps(dict(complete=True,purpose='runtime integration, not performance evidence',cases=rows),indent=2)+'\n')

if __name__=='__main__':main()
