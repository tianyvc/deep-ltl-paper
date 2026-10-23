"""Synthetic aggregation acceptance only; fixtures are not experimental evidence."""
from pathlib import Path
import tempfile,subprocess,sys,json

def main():
    repo=Path.cwd().resolve();checks=[]
    evidence=repo/'docs/maintenance/validation-runs/20261021/zone-minimal'
    with tempfile.TemporaryDirectory(prefix='zone-summary-check-') as temp:
        root=Path(temp)
        # Synthetic complete matrix with a negative treatment effect must be retained.
        plan=json.loads((repo/'experiments/reproduction/20261021-zone-minimal-v3/manifest.json').read_text());fixtures=root/'statistics';fixtures.mkdir()
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
        data=json.loads(dest.read_text());assert len(data['inputs_sha256'])==25
        contrast=next(c for c in data['comparisons'] if c['environment']=='PointLtl2-v0' and c['treatment']=='hv_track' and c['reference']=='gnn_track');assert abs(contrast['statistics']['mean']+.1)<1e-8 and contrast['statistics']['n']==3
        reference=next(g for g in data['groups'] if g['environment']=='FlatWorld-v0');assert reference['finite_macro_sr']['ci95'] is None
        assert len(data['per_task_comparisons'])>0 and data['main_environment']=='PointLtl2-v0'
        core=next(g for g in data['groups'] if g['environment']=='PointLtl2-v0' and g['configuration']=='hv_track');assert core['finite_macro_sr']['n']==3
        assert dest.with_suffix('.csv').exists();checks.extend(['complete_25_run_zone_matrix_and_csv','negative_paired_effect_preserved','single_seed_no_ci'])
    result=dict(passed=checks,count=len(checks));(evidence/'summary-acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
