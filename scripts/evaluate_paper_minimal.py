"""Budgeted original-paper task evaluation with strict frozen-checkpoint validation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from envs import make_env
from ltl.samplers import FixedSampler
from model.paper_hiltl import PaperHiLTL,ProductEpisode,PROTOCOL_VERSION
from model.deepltl_reference import PaperDeepLTL
from train.train_paper import collect_episode,atomic_json
from train.paper_tasks import load_tasks,graph_fingerprint,normalize_formula
from ltl.automata import ltl2ldba
from config import model_configs

IMPLEMENTATION=['src/model/paper_hiltl.py','src/model/paper_options.py','src/model/deepltl_reference.py','src/train/train_paper.py','src/train/paper_tasks.py']


def evaluate(checkpoint_path,spec_path,output,device):
    torch.set_num_threads(2);random.seed(10001);np.random.seed(10001);torch.manual_seed(10001)
    spec=json.loads(spec_path.read_text());status=torch.load(checkpoint_path,map_location=device,weights_only=False);config=status['config']
    assert status['protocol']==PROTOCOL_VERSION==config['protocol']
    assert config['implementation_sha256']==hashlib.sha256(b''.join(Path(p).read_bytes() for p in IMPLEMENTATION)).hexdigest(),'Implementation mismatch'
    manifest=load_tasks(Path(config['task_set']) if config['task_set'] else None,config['formula'],config['env'],config['max_episode_steps'])
    assert manifest==config['task_manifest'],'Training manifest mismatch'
    assert config['env']==spec['env'] and status['steps']>=spec['minimum_training_steps']
    result_path=output/'evaluation-minimal.json'
    if result_path.exists():raise FileExistsError(result_path)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    before_sha=checkpoint_sha
    rows=[];groups={};started=time.monotonic()
    train_records=[r for r in manifest['automata'] if r['split']=='train']
    for index,item in enumerate(spec['tasks']):
        formula=item['formula'];count=item['episodes']
        env=ProductEpisode(make_env(config['env'],FixedSampler.partial(formula),max_steps=config['max_episode_steps']))
        try:
            obs=env.reset(seed=spec['eval_seed'])
            if index==0:
                if config['configuration']=='baseline':model=PaperDeepLTL(env,model_configs[config['env']]).to(device)
                else:model=PaperHiLTL(obs['features'].shape,env.action_space,config['configuration'],propositions=obs['ldba'].propositions,gate=config['gate']).to(device)
                model.load_state_dict(status['model'],strict=True);model.eval()
                before={k:v.detach().clone() for k,v in model.state_dict().items()}
            raw=ltl2ldba(formula,obs['ldba'].propositions,simplify_labels=False)
            raw_sha,pruned_sha=graph_fingerprint(raw),graph_fingerprint(obs['ldba'])
            seen=any(normalize_formula(r['formula'])==normalize_formula(formula) or r['raw_sha256']==raw_sha or r['pruned_sha256']==pruned_sha for r in train_records)
            task_rows=[]
            for ep in range(count):
                episode=collect_episode(env,model,seed=spec['eval_seed']+ep,deterministic=True,option_max_steps=config['option_max_steps'])
                reward=episode[4];info=episode[5];row=dict(task_id=item['id'],formula=formula,seed=spec['eval_seed']+ep,episode=ep,finite=env.terminate_on_acceptance,seen_training_formula=seen,steps=len(reward),reward=sum(reward),discounted_return=sum(config['gamma']**t*r for t,r in enumerate(reward)),success=bool(info.get('success')),violation=bool(info.get('violation')),accepting_visits=int(info.get('num_accepting_visits',0)),option_decisions=len(episode[6]['segments']))
                rows.append(row);task_rows.append(row)
                if (ep+1)%10==0:
                    atomic_json(output/'evaluation-progress.json',dict(state='running',task_id=item['id'],task_index=index+1,tasks=len(spec['tasks']),episodes_done=len(rows),episodes_target=sum(t['episodes'] for t in spec['tasks']),elapsed_seconds=time.monotonic()-started))
                    print(json.dumps(dict(task=item['id'],episodes=ep+1,total=count)),flush=True)
            finite=bool(env.terminate_on_acceptance);successes=[r for r in task_rows if r['success']]
            groups[formula]=dict(task_id=item['id'],finite=finite,seen_training_formula=seen,zero_shot=not seen,episodes=count,raw_sha256=raw_sha,pruned_sha256=pruned_sha,states=obs['ldba'].num_states,edges=obs['ldba'].num_transitions,success_rate=float(np.mean([r['success'] for r in task_rows])) if finite else None,violation_rate=float(np.mean([r['violation'] for r in task_rows])),mean_steps=float(np.mean([r['steps'] for r in task_rows])),successful_episodes=len(successes),successful_episode_mean_steps=float(np.mean([r['steps'] for r in successes])) if successes and finite else None,mean_reward=float(np.mean([r['reward'] for r in task_rows])),mean_discounted_return=float(np.mean([r['discounted_return'] for r in task_rows])),mean_accepting_visits=float(np.mean([r['accepting_visits'] for r in task_rows])),mean_accepting_visits_per_step=float(np.mean([r['accepting_visits']/r['steps'] for r in task_rows])))
        finally:env.close()
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Weights changed during evaluation'
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==before_sha,'Checkpoint changed'
    result=dict(protocol=PROTOCOL_VERSION,study='minimal-original-tasks-v1',environment=config['env'],configuration=config['configuration'],training_seed=config['seed'],gate=config['gate'],horizon=config['max_episode_steps'],checkpoint_steps=status['steps'],checkpoint_sha256=checkpoint_sha,training_manifest_sha256=manifest['sha256'],implementation_sha256=config['implementation_sha256'],spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),weights_unchanged=True,deterministic=True,episodes=rows,per_formula=groups,elapsed_seconds=time.monotonic()-started,notes=['phi1 is seen in the retained training distribution; do not label all tasks zero-shot.','Original paper formulas are evaluated with actual compiled state counts; no reuse of old reported SR.','Reset episodes per formula are not independent training seeds; exact counts are in spec.'])
    atomic_json(result_path,result);atomic_json(output/'evaluation-progress.json',dict(state='completed',episodes_done=len(rows),episodes_target=len(rows),elapsed_seconds=time.monotonic()-started));print('Saved',result_path,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(a.checkpoint,a.spec,a.output,a.device)

if __name__=='__main__':main()
