"""Independent-seed statistics and prespecified matched contrasts for a campaign."""
import argparse
import hashlib
import json
from pathlib import Path
from summarize_paper_runs import aggregate,summarize


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--env',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); manifest=json.loads(args.manifest.read_text())
    tasks=[t for t in manifest['tasks'] if t['kind']=='evaluation' and t['env']==args.env and t['split']==args.split]
    assert len(tasks)==65, 'Require all 13 configurations x 5 independent seeds'
    files=[Path(t['output'])/t['result'] for t in tasks]
    episodes=tasks[0]['episodes']
    result=aggregate(files,[1,2,3,4,5],episodes,'gnn_single')
    seed_values={}; costs=[]
    for task,path in zip(tasks,files):
        evaluation=json.loads(path.read_text());config=json.loads((path.parent/'config.json').read_text())
        finite=[v for v in evaluation['per_formula'].values() if v['finite']]
        recurrent=[v for v in evaluation['per_formula'].values() if not v['finite']]
        key=(config['configuration'],config['gate'])
        values=seed_values.setdefault(key,{})
        values[config['seed']]={
          'finite_formula_macro_sr':sum(v['success_rate'] for v in finite)/len(finite) if finite else None,
          'recurrent_macro_accepting_visits':sum(v['mean_accepting_visits'] for v in recurrent)/len(recurrent) if recurrent else None,
          'per_formula':evaluation['per_formula']}
        metrics=[json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
        costs.append(dict(configuration=key[0],gate=key[1],seed=config['seed'],actual_steps=evaluation['checkpoint_steps'],parameter_count=config['parameter_count'],peak_cuda_allocated_bytes=max(m.get('cuda_peak_allocated_bytes') or 0 for m in metrics),training_loop_seconds=metrics[-1]['elapsed_seconds'],source=str(path)))
    result['macro_metrics']=[]
    for (configuration,gate),values in seed_values.items():
        record={'configuration':configuration,'gate':gate,'per_seed':values,'statistics':{}}
        for metric in ['finite_formula_macro_sr','recurrent_macro_accepting_visits']:
            observations=[values[s][metric] for s in range(1,6)]
            record['statistics'][metric]=summarize(observations) if all(x is not None for x in observations) else None
        result['macro_metrics'].append(record)
    pairs=[('hv_only','gnn_single'),('hv_track','gnn_track'),('hv_only','gnn_capacity'),('hv_track','gnn_track_capacity'),('gnn_track','gnn_single'),('hv_track','hv_only'),('msgnn_only','gnn_single'),('msgnn_track','gnn_track'),('full_no_options','hv_track'),('full_all','full_no_options')]
    contrasts=[((a,'fixed'),(b,'fixed')) for a,b in pairs]+[(('hv_track',gate),('hv_track','fixed')) for gate in ['constant','learned']]
    result['prespecified_paired_macro_contrasts']=[]
    for a,b in contrasts:
        for metric in ['finite_formula_macro_sr','recurrent_macro_accepting_visits']:
            x=[seed_values[a][s][metric] for s in range(1,6)];y=[seed_values[b][s][metric] for s in range(1,6)]
            if any(v is None for v in x+y): continue
            result['prespecified_paired_macro_contrasts'].append(dict(treatment=a,reference=b,metric=metric,statistics=summarize([l-r for l,r in zip(x,y)])))
    interaction=[]
    for s in range(1,6):
        v=lambda c:seed_values[(c,'fixed')][s]['finite_formula_macro_sr']
        interaction.append((v('hv_track')-v('gnn_track'))-(v('hv_only')-v('gnn_single')))
    result['hv_history_factorial_interaction']=summarize(interaction)
    result['costs']=costs
    result['provenance']={'manifest_sha256':sha(args.manifest),'input_sha256':{str(p):sha(p) for p in files},'checkpoint_selection':'final budget only','note':'Intervals are descriptive t intervals over training seeds, conditional on this fixed formula set; no multiple-comparison significance claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2)
    print(args.output)

if __name__=='__main__': main()
