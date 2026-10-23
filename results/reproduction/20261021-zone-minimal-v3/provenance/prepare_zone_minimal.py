from pathlib import Path
import datetime as dt
import hashlib,json,shutil,subprocess,sys,os
sys.path.insert(0,'scripts');from long_task import identity
repo=Path.cwd().resolve();old=repo/'experiments/reproduction/20261021-v3';oldstate=json.loads((old/'state.json').read_text());assert oldstate['state']=='paused' and identity(oldstate['pid']) is None
for v in oldstate['tasks'].values():
    if v.get('job_dir'):
        p=Path(v['job_dir'])/'status.json'
        if p.exists():
            worker=json.loads(p.read_text());assert worker['state'] not in ('running','starting')
root=repo/'experiments/reproduction/20261021-zone-minimal-v3';root.mkdir(exist_ok=False)
source=root/'source';shutil.copytree(old/'source',source,ignore=shutil.ignore_patterns('__pycache__','.git'))
for name in ['paper_campaign.py','evaluate_paper_minimal.py','summarize_paper_minimal.py','check_paper_campaign.py']:
    shutil.copy2(repo/'scripts'/name,source/'scripts'/name)
(root/'specs').mkdir();evidence=repo/'docs/maintenance/validation-runs/20261021/zone-minimal';evidence.mkdir(exist_ok=True)
resources=subprocess.check_output(['nvidia-smi'],text=True);(root/'resource-audit.txt').write_text(resources);(evidence/'resource-audit.txt').write_text(resources)
envs=[('flatworld','FlatWorld-v0',500),('letter','LetterEnv-v0',75),('zones','PointLtl2-v0',1000)]
for slug,env,h in envs:
    a,b,c=['a','b','c'] if slug=='letter' else ['blue','green','yellow']
    formulas=[('phi1',f'F {a} & F {b}'),('phi2',f'F {a} & F {b} & F {c}'),('phi3',f'(!{a} U {b}) & F {c}'),('recurrent',f'G F {a} & G F {c}')]
    spec=dict(env=env,horizon=h,minimum_training_steps=200000,eval_seed=10001,tasks=[dict(id=i,formula=f,episodes=50) for i,f in formulas],origin='Original long-manuscript phi1/phi2/phi3; colors mapped to a/b/c in Letter. Recurrence reported separately.',selection='Fixed before minimal-study results. No positive-effect filtering.')
    if slug=='zones':
        spec['tasks'] += [dict(id='sequence3',formula='F (blue & F (green & F yellow))',episodes=50),dict(id='original_phi7',formula='F blue & (!blue U (green & F yellow))',episodes=50)]
        spec['origin'] += ' Zone sequence3 follows Appendix G.3 reachability template at depth 3; original_phi7 is the exact ZoneEnv Table 2 / Appendix F.2 phi7.'
    (root/'specs'/f'{slug}.json').write_text(json.dumps(spec,indent=2))
groups=[]
# The order puts the main method and its fair control first for every seed.
for slug,env,h in [envs[2],envs[0],envs[1]]:
    configurations=['hv_track','gnn_track','gnn_single','hv_only','baseline'] if slug=='zones' else (['gnn_single','hv_only','gnn_track','hv_track'] if slug=='flatworld' else ['gnn_track','hv_track'])
    for cfg in configurations:groups.append(dict(slug=slug,env=env,horizon=h,configuration=cfg,seeds=[1,2,3] if slug=='zones' else [1],scope='core_three_seed' if slug=='zones' else 'single_seed_transfer'))
for cfg in ['msgnn_only','msgnn_track','full_no_options','full_all']:groups.append(dict(slug='zones',env='PointLtl2-v0',horizon=1000,configuration=cfg,seeds=[1],scope='single_seed_extension'))

tasks=[];imports=[];states={}
# Core runs first, supplementary runs only after all core tasks have been enumerated.
runs=[(g,seed) for seed in [1,2,3] for g in groups if g['scope']=='core_three_seed' for _ in [0] if seed in g['seeds']]+[(g,seed) for g in groups if g['scope']!='core_three_seed' for seed in g['seeds']]
for g,seed in runs:
    ident=f"{g['slug']}-{g['configuration']}-s{seed}";out=root/'runs'/ident;train=ident+'-train'
    base=['src/train/train_paper.py','--env',g['env'],'--task-set',str(source/f"configurations/{g['slug']}_paper_v3.json"),'--configuration',g['configuration'],'--gate','fixed','--seed',str(seed),'--device','cuda:0','--max-episode-steps',str(g['horizon']),'--rollout-steps','512','--epochs','4','--batch-size','64','--gamma','.998','--lr','.0003','--option-max-steps','50','--steps','200000','--checkpoint-every','0','--output',str(out)]
    t=dict(id=train,kind='train',env=g['env'],configuration=g['configuration'],seed=seed,output=str(out),steps=200000,depends=[],argv=base,scope=g['scope']);tasks.append(t)
    if g['slug']=='flatworld' and seed==1 and g['configuration'] in ('gnn_single','hv_only'):
        original=old/'runs'/f"flatworld-{g['configuration']}-fixed-s1";snapshots=sorted((original/'checkpoints').glob('step-*'),key=lambda p:int(p.name.split('-')[1]));snap=next(p for p in snapshots if int(p.name.split('-')[1])>=200000);actual=int(snap.name.split('-')[1]);out.mkdir(parents=True)
        shutil.copy2(snap/'status.pt',out/'status.pt');shutil.copy2(snap/'config.json',out/'config.json')
        metrics=[json.loads(l) for l in (original/'metrics.jsonl').read_text().splitlines()];metrics=[r for r in metrics if r['steps']<=actual];assert metrics[-1]['steps']==actual
        (out/'metrics.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metrics));last_update=metrics[-1]['update']
        episodes=[json.loads(l) for l in (original/'episodes.jsonl').read_text().splitlines()];(out/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes if e['update']<=last_update))
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONHASHSEED='0',PYTHONPATH=os.pathsep.join([str(source/'src'),str(source/'src/envs/zones/safety-gymnasium')]),MUJOCO_GL='egl',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
        cpu=[x if x!='cuda:0' else 'cpu' for x in base]
        r=subprocess.run([sys.executable,*cpu,'--resume'],cwd=source,env=env,capture_output=True,text=True);(evidence/f'import-{ident}.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
        states[train]=dict(state='completed',evidence=dict(steps=actual,updates=last_update,checkpoint_sha256=hashlib.sha256((out/'status.pt').read_bytes()).hexdigest(),imported=True))
        imports.append(dict(run=ident,source_checkpoint=str(snap/'status.pt'),actual_steps=actual,strict_resume_verified=True,reason='Reuse predetermined earliest saved checkpoint at/above 200k; no selection by evaluation.'))
    spec=root/'specs'/f"{g['slug']}.json";evaluation=dict(id=ident+'-eval',kind='minimal_evaluation',env=g['env'],configuration=g['configuration'],seed=seed,scope=g['scope'],output=str(out),checkpoint=str(out/'status.pt'),spec=str(spec),steps=200000,depends=[train],argv=['scripts/evaluate_paper_minimal.py','--checkpoint',str(out/'status.pt'),'--spec',str(spec),'--output',str(out),'--device','cuda:0']);tasks.append(evaluation)
summary=root/'summary.json';tasks.append(dict(id='minimal-summary',kind='summary',output=str(summary),depends=[t['id'] for t in tasks if t['kind']=='minimal_evaluation'],argv=['scripts/summarize_paper_minimal.py','--manifest',str(root/'manifest.json'),'--output',str(summary)]))
assert len(tasks)==51 and len({t['id'] for t in tasks})==51
# Freeze specs as well as source. The source is an unchanged training implementation.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={str(p):sha(p) for directory in [source,root/'specs'] for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
now=dt.datetime.now(dt.timezone.utc);deadline=now+dt.timedelta(days=5)
manifest=dict(name='20261021-zone-minimal-v3',main_environment='PointLtl2-v0',primary_task_ids=['phi1','phi2','phi3'],study='zone-main-minimal-v1',protocol='hiltl-product-v3',source_root=str(source),created_utc=now.isoformat(),dispatch_deadline_utc=deadline.isoformat(),gpus=[2,1],max_workers=2,min_free_mib=4096,poll_seconds=10,frozen_hashes=hashes,training_budget=200000,run_groups=groups,imported_checkpoints=imports,tasks=tasks,notes=['Zone main: fair HV x history 2x2 and same-protocol DeepSets/GRU reference, each 3 seeds = 15. Zone original MSGNN/Options configurations each 1 exploratory seed = 4.', 'Flat 4 controls x 1 seed (2 checkpoints reused); Letter main/control x 1 seed. Total 25 models, 23 new 200k training runs.', 'Primary macro is original manuscript phi1/phi2/phi3 only; sequence3, original DeepLTL phi7 and recurrence reported separately. All outcomes retained.', 'No capacity-matched or external-official SOTA evidence. 200k finite-budget evidence is not 15M original-curriculum reproduction.', 'Five-day dispatch limit; active tasks finish naturally; incomplete work is disclosed.'])
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'state.json').write_text(json.dumps(dict(manifest_sha256=sha(root/'manifest.json'),tasks=states),indent=2))
(root/'imports.json').write_text(json.dumps(imports,indent=2));shutil.copy2(old/'provenance.json',root/'training-source-provenance.json');shutil.copy2(old/'pip-freeze.txt',root/'pip-freeze.txt')
import shlex
(root/'commands.txt').write_text('\n\n'.join(t['id']+'\n  '+shlex.join([sys.executable,*t['argv']]) for t in tasks)+'\n')
for name in ['manifest.json','imports.json','commands.txt']:shutil.copy2(root/name,evidence/name)
(old/'SUPERSEDED_BY.json').write_text(json.dumps({'minimal_manifest':str(root/'manifest.json'),'reason':'User reduced wall-time budget to days/weeks; old batch remains paused and must not auto-resume.','utc':now.isoformat()},indent=2))
print(json.dumps(dict(root=str(root),training_runs=25,imported=imports,tasks=len(tasks),evaluation_episodes=19*6*50+6*4*50,deadline=deadline.isoformat())),flush=True)
