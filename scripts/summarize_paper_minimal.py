"""Summarize all predetermined minimal-study outcomes, including negative effects."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from summarize_paper_runs import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();manifest=json.loads(a.manifest.read_text());tasks=[t for t in manifest['tasks'] if t['kind']=='minimal_evaluation'];results=[];index={}
    for task in tasks:
        path=Path(task['output'])/'evaluation-minimal.json';data=json.loads(path.read_text());spec=json.loads(Path(task['spec']).read_text());assert data['weights_unchanged'] and data['deterministic'];assert data['checkpoint_steps']>=200000;assert data['spec_sha256']==hashlib.sha256(Path(task['spec']).read_bytes()).hexdigest();assert data['training_seed']==task['seed'] and data['configuration']==task['configuration'] and data['environment']==task['env']
        assert set(data['per_formula'])=={x['formula'] for x in spec['tasks']}
        for item in spec['tasks']:
            rows=[r for r in data['episodes'] if r['formula']==item['formula']];assert len(rows)==item['episodes'] and len({r['seed'] for r in rows})==item['episodes']
        results.append((task,path,data));key=(task['env'],task['configuration']);values=index.setdefault(key,{});assert task['seed'] not in values;values[task['seed']]=data
    groups=[];macro={};csv_rows=[]
    for (env,cfg),seeds in index.items():
        expected=next(t['seeds'] for t in manifest['run_groups'] if t['env']==env and t['configuration']==cfg);assert sorted(seeds)==expected
        metrics={};macro[(env,cfg)]={}
        for seed,data in seeds.items():
            finite=[g for g in data['per_formula'].values() if g['finite'] and g['task_id'] in manifest.get('primary_task_ids',['phi1','phi2','phi3'])];macro[(env,cfg)][seed]=sum(g['success_rate'] for g in finite)/len(finite)
            for formula,g in data['per_formula'].items():
                csv_rows.append(dict(environment=env,configuration=cfg,training_seed=seed,task_id=g['task_id'],formula=formula,seen_training_formula=g['seen_training_formula'],checkpoint_steps=data['checkpoint_steps'],success_rate=g['success_rate'],violation_rate=g['violation_rate'],mean_steps=g['mean_steps'],successful_episodes=g['successful_episodes'],successful_episode_mean_steps=g['successful_episode_mean_steps'],mean_reward=g['mean_reward'],mean_discounted_return=g['mean_discounted_return'],mean_accepting_visits=g['mean_accepting_visits'],mean_accepting_visits_per_step=g['mean_accepting_visits_per_step']))
        for formula,g in next(iter(seeds.values()))['per_formula'].items():
            metrics[formula]={}
            for metric in ['success_rate','violation_rate','mean_steps','mean_reward','mean_discounted_return','mean_accepting_visits','mean_accepting_visits_per_step']:
                values=[d['per_formula'][formula][metric] for d in seeds.values()];metrics[formula][metric]=None if all(v is None for v in values) else summarize(values)
        groups.append(dict(environment=env,configuration=cfg,seeds=sorted(seeds),scope='core_three_seed' if len(seeds)==3 else 'single_seed_exploratory',finite_macro_sr=summarize(list(macro[(env,cfg)].values())),per_formula=metrics,actual_steps={s:d['checkpoint_steps'] for s,d in seeds.items()}))
    pairs=[]
    for env in sorted({k[0] for k in index}):
        contrasts=[('hv_track','gnn_track')]
        if env==manifest.get('main_environment','FlatWorld-v0'):contrasts += [('hv_only','gnn_single'),('hv_track','hv_only'),('gnn_track','gnn_single'),('full_no_options','hv_track'),('full_all','full_no_options'),('msgnn_only','gnn_single'),('msgnn_track','gnn_track'),('hv_track','baseline')]
        else:contrasts+=[('hv_track','baseline')]
        for treatment,reference in contrasts:
            if (env,treatment) not in macro or (env,reference) not in macro:continue
            x,y=macro[(env,treatment)],macro[(env,reference)];seeds=sorted(set(x)&set(y));differences=[x[s]-y[s] for s in seeds]
            pairs.append(dict(environment=env,treatment=treatment,reference=reference,seeds=seeds,metric='paired_finite_macro_sr_difference',statistics=summarize(differences),percentage_points=summarize([100*d for d in differences]),scope='core_three_seed' if len(seeds)==3 else 'single_seed_exploratory'))
    task_pairs=[]
    for pair in pairs:
        env,treatment,reference=pair['environment'],pair['treatment'],pair['reference']
        for formula,g in next(iter(index[(env,treatment)].values()))['per_formula'].items():
            for metric in (['success_rate','violation_rate','mean_discounted_return'] if g['finite'] else ['mean_accepting_visits_per_step','violation_rate','mean_discounted_return']):
                values=[index[(env,treatment)][seed]['per_formula'][formula][metric]-index[(env,reference)][seed]['per_formula'][formula][metric] for seed in pair['seeds']]
                task_pairs.append(dict(environment=env,treatment=treatment,reference=reference,task_id=g['task_id'],formula=formula,metric=metric,seeds=pair['seeds'],statistics=summarize(values),scope=pair['scope']))
    output=dict(main_environment=manifest.get('main_environment','FlatWorld-v0'),primary_task_ids=manifest.get('primary_task_ids',['phi1','phi2','phi3']),per_task_comparisons=task_pairs,study='minimal-original-tasks-v1',groups=groups,comparisons=pairs,inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,p,_ in results},manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),notes=['200k-budget initial evidence, not replication of original 1M/5seed/500episode results.','All fixed runs and tasks included; no positive-effect filtering.','Three-seed intervals are descriptive and can be wide; single-seed controls have no between-training CI.','Finite phi1/2/3 macro SR is separate from recurrent accepting visits. Seen phi1 is not zero-shot.','Task set retained for checkpoint reuse; original formulas and actual automaton counts are explicit.','No external SOTA claim or inference of infinite satisfaction probability.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(output,f,indent=2)
    with a.output.with_suffix('.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print('Saved',a.output,flush=True)

if __name__=='__main__':main()
