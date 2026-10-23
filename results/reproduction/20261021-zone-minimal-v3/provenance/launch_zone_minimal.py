from pathlib import Path
import datetime as dt,hashlib,json,os,subprocess,sys,time,shutil
repo=Path.cwd().resolve();root=repo/'experiments/reproduction/20261021-zone-minimal-v3';previous=repo/'experiments/reproduction/20261021-minimal-v3';big=repo/'experiments/reproduction/20261021-v3';evidence=repo/'docs/maintenance/validation-runs/20261021/zone-minimal'
sys.path.insert(0,str(repo/'scripts'));from paper_campaign import validate;from long_task import identity
assert not (root/'launch.json').exists(),'Already launched'
for batch in [big,previous]:
    state=json.loads((batch/'state.json').read_text());assert state['state']=='paused' and identity(state['pid']) is None,(str(batch),state['state'])
    assert not any(t['state']=='running' for t in state['tasks'].values())
accept=json.loads((evidence/'acceptance.json').read_text());assert len(accept['checks'])==4
manifest=json.loads((root/'manifest.json').read_text());assert len({t['id'] for t in manifest['tasks']})==51
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==checksum for p,checksum in manifest['frozen_hashes'].items())
# Import only completed, contract-identical evaluations. No selection on performance.
state=json.loads((root/'state.json').read_text());imports=[]
for task in manifest['tasks']:
    if task['kind']!='minimal_evaluation' or task['env']!='FlatWorld-v0' or task['configuration'] not in ('gnn_single','hv_only'):continue
    original=previous/'runs'/Path(task['output']).name;prior=json.loads((original/'evaluation-minimal.json').read_text())
    assert prior['spec_sha256']==hashlib.sha256(Path(task['spec']).read_bytes()).hexdigest()
    assert prior['checkpoint_sha256']==hashlib.sha256(Path(task['checkpoint']).read_bytes()).hexdigest()
    out=Path(task['output']);assert not (out/'evaluation-minimal.json').exists()
    for name in ['evaluation-minimal.json','evaluation-progress.json']:shutil.copy2(original/name,out/name)
    verified=validate(task);state['tasks'][task['id']]=dict(state='completed',evidence=dict(**verified,imported=True),source=str(original/'evaluation-minimal.json'))
    imports.append(dict(task=task['id'],source=str(original/'evaluation-minimal.json'),evidence=verified,selection='All completed identity-matched evaluations; no outcome filtering.'))
(root/'state.json').write_text(json.dumps(state,indent=2));(root/'imported-evaluations.json').write_text(json.dumps(imports,indent=2));shutil.copy2(evidence/'acceptance.json',root/'acceptance.json')
(previous/'SUPERSEDED_BY.json').write_text(json.dumps(dict(manifest=str(root/'manifest.json'),reason='User specified ZoneEnv as main environment; previous contract is permanently paused, data retained.',utc=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2))
source=root/'source';env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONHASHSEED='0',PYTHONPATH=os.pathsep.join([str(source/'src'),str(source/'src/envs/zones/safety-gymnasium')]),MUJOCO_GL='egl',OMP_NUM_THREADS='2')
command=[sys.executable,str(source/'scripts/long_task.py'),'start','--job-dir',str(root/'scheduler-01'),'--',str(source/'scripts/paper_campaign.py'),'run','--manifest',str(root/'manifest.json')]
r=subprocess.run(command,cwd=source,env=env,capture_output=True,text=True);print(r.stdout+r.stderr,flush=True);assert r.returncode==0
for _ in range(30):
    state=json.loads((root/'state.json').read_text())
    if 'pid' in state and any(v['state']=='running' for v in state['tasks'].values()):break
    time.sleep(1)
assert state.get('state')=='running',state
pointer=repo/'experiments/reproduction/ACTIVE.json';temp=pointer.with_suffix('.tmp');temp.write_text(json.dumps(dict(manifest=str(root/'manifest.json'),name=manifest['name'],supersedes=str(previous/'manifest.json'),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat()),indent=2));temp.replace(pointer)
report=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),command=command,stdout=r.stdout,active_manifest=str(root/'manifest.json'),prior_batch_states={str(b):json.loads((b/'state.json').read_text())['state'] for b in [big,previous]},current_state={k:v for k,v in state.items() if k!='tasks'},active_tasks={k:v for k,v in state['tasks'].items() if v['state']=='running'},imported_evaluations=imports)
for folder in [evidence,root]:(folder/'launch.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
r=subprocess.run([sys.executable,'scripts/check_paper_campaign.py','--sample-seconds','10'],cwd=repo,env=dict(os.environ,PYTHONNOUSERSITE='1'),capture_output=True,text=True);(evidence/'initial-health.txt').write_text(r.stdout+r.stderr);print(r.stdout+r.stderr,flush=True);assert r.returncode==0
