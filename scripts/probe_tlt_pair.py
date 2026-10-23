"""Bounded paired integration probe; does not establish manuscript performance."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=512);p.add_argument('--rollout-steps',type=int,default=128);p.add_argument('--device',default='cpu');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20)
    (args.output/'gpu.txt').write_text(gpu.stdout+gpu.stderr)
    rows=[]
    for configuration in ['baseline','hv_track']:
        run=args.output/configuration
        command=[sys.executable,'src/train/train_tlt.py','--env','PointLtl2-v0','--configuration',configuration,'--output',str(run),'--steps',str(args.steps),'--seed','1','--device',args.device,'--num-procs','2','--rollout-steps',str(args.rollout_steps),'--epochs','2','--batch-size','64','--threads','2','--max-seconds','180']
        started=time.monotonic()
        with (args.output/(configuration+'.log')).open('w') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=360)
        row=dict(configuration=configuration,command=command,returncode=child.returncode,wall_seconds=time.monotonic()-started)
        if child.returncode:raise RuntimeError(row)
        row['progress']=json.loads((run/'progress.json').read_text());rows.append(row)
        print(json.dumps(row),flush=True)
        (args.output/'summary.json').write_text(json.dumps(dict(purpose='paired integration only, not evidence of SR improvement',runs=rows),indent=2))

if __name__=='__main__':main()
