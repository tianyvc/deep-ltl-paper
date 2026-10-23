"""Synthetic independent-seed oracle for the campaign's prespecified statistics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    variants=[(c,'fixed') for c in ['baseline','gnn_single','gnn_track','gnn_capacity','gnn_track_capacity','hv_only','msgnn_only','hv_track','msgnn_track','full_no_options','full_all']]+[('hv_track',g) for g in ['learned','constant']]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp); tasks=[]
        for i,(configuration,gate) in enumerate(variants):
            for seed in range(1,6):
                run=root/f'{i}-{seed}';run.mkdir()
                config=dict(env='synthetic',seed=seed,configuration=configuration,gate=gate,max_episode_steps=4,gamma=.998,protocol='test-only',task_manifest={'sha256':'synthetic'},rollout_steps=4,epochs=1,batch_size=4,lr=.0003,option_max_steps=50,parameter_count=100+i)
                (run/'config.json').write_text(json.dumps(config))
                groups={};rows=[]
                for formula,finite in [('F a',True),('G F a',False)]:
                    sr=(seed+i)/100
                    groups[formula]=dict(finite=finite,success_rate=sr if finite else None,mean_accepting_visits=seed,mean_steps=4,mean_reward=seed,mean_discounted_return=seed)
                    rows.extend(dict(formula=formula,seed=j) for j in [10001,10002])
                name='evaluation.json';(run/name).write_text(json.dumps(dict(protocol='test-only',weights_unchanged=True,task_manifest_sha256='synthetic',split='held_out',deterministic=True,checkpoint_sha256='synthetic',checkpoint_steps=100,per_formula=groups,episodes=rows)))
                (run/'metrics.jsonl').write_text(json.dumps(dict(cuda_peak_allocated_bytes=1000,elapsed_seconds=2))+'\n')
                tasks.append(dict(kind='evaluation',env='synthetic',split='held_out',output=str(run),result=name,episodes=2))
        manifest=root/'manifest.json';manifest.write_text(json.dumps({'tasks':tasks}))
        output=root/'summary.json'
        subprocess.run([sys.executable,'scripts/analyze_paper_campaign.py','--manifest',str(manifest),'--env','synthetic','--split','held_out','--output',str(output)],check=True)
        data=json.loads(output.read_text()); assert len(data['groups'])==26 and len(data['costs'])==65
        baseline=next(v for v in data['macro_metrics'] if v['configuration']=='baseline')
        assert abs(baseline['statistics']['finite_formula_macro_sr']['mean']-.03)<1e-10
        assert baseline['statistics']['finite_formula_macro_sr']['n']==5
        assert len(data['prespecified_paired_macro_contrasts'])==24
        assert len(data['provenance']['input_sha256'])==65
        print(json.dumps({'passed':['full_13_by_5_matrix','known_seed_macro_mean','prespecified_matched_contrasts','separate_recurrent_metrics','input_hash_provenance']}))

if __name__=='__main__': main()
