"""Evaluate retained checkpoints on validation only; never choose by test score."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('base',nargs=argparse.REMAINDER)
    a=p.parse_args(); base=a.base[1:] if a.base[:1]==['--'] else a.base
    snapshots=sorted((a.output/'checkpoints').glob('step-*'),key=lambda x:int(x.name.split('-')[1]))
    assert snapshots,'No retained checkpoints'
    rows=[]
    final_steps=json.loads((a.output/'progress.json').read_text())['steps']
    snapshots = [p for p in snapshots if int(p.name.split('-')[1]) < final_steps] + [a.output]
    for snapshot in snapshots:
        steps=final_steps if snapshot == a.output else int(snapshot.name.split('-')[1])
        result=snapshot/f'evaluation-validation-20001-{a.episodes}-det.json'
        if not result.exists():
            command=[sys.executable,*base,'--output',str(snapshot),'--evaluate',str(a.episodes),'--eval-split','validation','--eval-seed','20001','--deterministic']
            print(json.dumps({'snapshot':str(snapshot),'command':command}),flush=True)
            subprocess.run(command,check=True)
        evaluation=json.loads(result.read_text())
        assert evaluation['weights_unchanged'] and evaluation['split']=='validation' and evaluation['checkpoint_steps']==steps
        assert evaluation['checkpoint_sha256']==hashlib.sha256((snapshot/'status.pt').read_bytes()).hexdigest()
        rows.append({'steps':steps,'result':str(result),'sha256':hashlib.sha256(result.read_bytes()).hexdigest()})
    dest=a.output/'validation-curve-index.json'
    temp=dest.with_suffix('.tmp'); temp.write_text(json.dumps({'selection_rule':'Final budget checkpoint only; intermediate validation for learning curves, not checkpoint selection.','points':rows},indent=2)); temp.replace(dest)

if __name__=='__main__':
    main()
