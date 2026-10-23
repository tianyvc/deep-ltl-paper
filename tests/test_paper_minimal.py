"""Acceptance checks for the short original-task study; fixtures are not research data."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    repo=Path.cwd().resolve();old=repo/'experiments/reproduction/20261021-v3';source=old/'source';evidence=repo/'docs/maintenance/validation-runs/20261021/minimal';checks=[]
    environment=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONHASHSEED='0',PYTHONPATH=os.pathsep.join([str(source/'src'),str(source/'src/envs/zones/safety-gymnasium')]),MUJOCO_GL='egl',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0')
    with tempfile.TemporaryDirectory(prefix='minimal-study-check-') as temp:
        root=Path(temp);spec=json.loads((repo/'experiments/reproduction/20261021-minimal-v3/specs/flatworld.json').read_text());spec['minimum_training_steps']=1000
        for t in spec['tasks']:t['episodes']=1
        specpath=root/'spec.json';specpath.write_text(json.dumps(spec))
        for cfg in ['gnn_single','baseline','full_all']:
            checkpoint=old/'runs/flatworld-gnn_single-fixed-s1/checkpoints/step-200532/status.pt' if cfg=='gnn_single' else old/'pilots'/f'pilot-flatworld-{cfg}'/'status.pt'
            local_spec=dict(spec);local_spec['minimum_training_steps']=1
            sp=root/(cfg+'-spec.json');sp.write_text(json.dumps(local_spec));output=root/cfg
            command=[sys.executable,str(repo/'scripts/evaluate_paper_minimal.py'),'--checkpoint',str(checkpoint),'--spec',str(sp),'--output',str(output),'--device','cuda:0']
            r=subprocess.run(command,cwd=source,env=environment,text=True,capture_output=True)
            (evidence/f'evaluator-{cfg}.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
            data=json.loads((output/'evaluation-minimal.json').read_text());assert len(data['episodes'])==4 and data['weights_unchanged'];assert sorted(g['task_id'] for g in data['per_formula'].values())==['phi1','phi2','phi3','recurrent'];assert next(g for g in data['per_formula'].values() if g['task_id']=='recurrent')['success_rate'] is None
            r=subprocess.run(command,cwd=source,env=environment,text=True,capture_output=True);assert r.returncode!=0 and 'FileExistsError' in r.stderr
            checks.append('real_'+cfg+'_original_tasks_frozen_weights_and_no_overwrite')
            print(checks[-1],flush=True)
        # Verify wall-time budget stops dispatch even if tasks are pending.
        deadline=root/'deadline';deadline.mkdir();m=deadline/'manifest.json';m.write_text(json.dumps(dict(source_root=str(repo),dispatch_deadline_utc=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=1)).isoformat(),gpus=[2],max_workers=1,min_free_mib=4096,frozen_hashes={},tasks=[dict(id='never',kind='train',output=str(deadline/'never'),depends=[],argv=['-c','raise AssertionError("must not run")'])])))
        r=subprocess.run([sys.executable,'scripts/paper_campaign.py','run','--manifest',str(m)],capture_output=True,text=True);assert r.returncode==0,r.stderr
        state=json.loads((deadline/'state.json').read_text());assert state['state']=='paused' and state['time_budget_reached'] and state['tasks']['never']['state']=='pending' and not (deadline/'jobs').exists();checks.append('wall_time_deadline_prevents_dispatch')
        # Synthetic complete matrix with a negative treatment effect must be retained.
        plan=json.loads((repo/'experiments/reproduction/20261021-minimal-v3/manifest.json').read_text());fixtures=root/'statistics';fixtures.mkdir()
        for task in plan['tasks']:
            if task['kind']!='minimal_evaluation':continue
            output=fixtures/task['id'];output.mkdir();task['output']=str(output);s=json.loads(Path(task['spec']).read_text())
            for t in s['tasks']:t['episodes']=1
            sp=output/'spec.json';sp.write_text(json.dumps(s));task['spec']=str(sp)
            import hashlib
            per={};rows=[]
            for t in s['tasks']:
                finite=t['id']!='recurrent';sr=.6 if task['configuration']=='hv_track' else .7
                per[t['formula']]=dict(task_id=t['id'],finite=finite,seen_training_formula=t['id']=='phi1',success_rate=sr if finite else None,violation_rate=.1,mean_steps=20,successful_episodes=1,successful_episode_mean_steps=20,mean_reward=1,mean_discounted_return=.9,mean_accepting_visits=1,mean_accepting_visits_per_step=.05)
                rows.append(dict(formula=t['formula'],seed=10001))
            data=dict(weights_unchanged=True,deterministic=True,checkpoint_steps=200000,spec_sha256=hashlib.sha256(sp.read_bytes()).hexdigest(),training_seed=task['seed'],configuration=task['configuration'],environment=task['env'],per_formula=per,episodes=rows)
            (output/'evaluation-minimal.json').write_text(json.dumps(data))
        m=fixtures/'manifest.json';m.write_text(json.dumps(plan));dest=fixtures/'summary.json';r=subprocess.run([sys.executable,'scripts/summarize_paper_minimal.py','--manifest',str(m),'--output',str(dest)],text=True,capture_output=True);assert r.returncode==0,r.stderr
        data=json.loads(dest.read_text());assert len(data['inputs_sha256'])==31
        contrast=next(c for c in data['comparisons'] if c['environment']=='FlatWorld-v0' and c['treatment']=='hv_track' and c['reference']=='gnn_track');assert abs(contrast['statistics']['mean']+.1)<1e-8 and contrast['statistics']['n']==3
        reference=next(g for g in data['groups'] if g['configuration']=='baseline');assert reference['finite_macro_sr']['ci95'] is None
        assert dest.with_suffix('.csv').exists();checks.extend(['complete_31_run_matrix_and_csv','negative_paired_effect_preserved','single_seed_no_ci'])
    result=dict(passed=checks,count=len(checks));(evidence/'minimal-acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
