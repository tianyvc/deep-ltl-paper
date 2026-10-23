"""Strict formula evaluation using original exhaustive search and environment horizons."""
import argparse,hashlib,json,pickle,random,time
from pathlib import Path
import numpy as np
import torch
from config import model_configs
from envs import make_env
from envs.tlt_context import TLTContext,preprocess_tlt
from ltl.samplers import FixedSampler
from model.agent import Agent
from model.tlt_sequence import TLTSequenceModel,PROTOCOL
from preprocessing import reset_vocab,VOCAB
from sequence.search.exhaustive_search import ExhaustiveSearch

class StrictSearch(ExhaustiveSearch):
    def get_value(self,seq,obs):
        if seq is None or not len(seq):return float('-inf')
        batch=preprocess_tlt([dict(obs,goal=seq)],self.propositions,device=next(self.model.parameters()).device)
        with torch.no_grad():_,value=self.model(batch)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite search value')
        return value.item()

class ContextAgent(Agent):
    def forward(self,obs,deterministic=False):
        batch=preprocess_tlt([obs],self.propositions,device=next(self.model.parameters()).device)
        with torch.no_grad():dist,_=self.model(batch);action=dist.mode if deterministic else dist.sample()
        return action.cpu().numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--formulas',nargs='+')
    p.add_argument('--task-set',type=Path,default=Path('configurations/paper_tasks.json'))
    p.add_argument('--task-group',choices=['finite','recurrent','all'],default='finite')
    p.add_argument('--episodes',type=int,default=2);p.add_argument('--seed',type=int,default=10001)
    p.add_argument('--device',default='cpu');p.add_argument('--max-seconds',type=float,default=180.)
    args=p.parse_args()
    if args.episodes<1 or args.max_seconds<=0:p.error('positive budgets required')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    status=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if status.get('protocol')!=PROTOCOL:raise ValueError('This evaluator requires the original-curriculum TLT protocol')
    config=status['config'];rows=[];started=time.monotonic();stop=False
    if args.formulas is None:
        groups=json.loads(args.task_set.read_text())['environments'][config['env']]
        args.formulas=[item['formula'] for group,items in groups.items() if args.task_group=='all' or group==args.task_group for item in items]
    if not args.formulas:raise ValueError('No evaluation formulas selected')
    for formula in args.formulas:
        env=make_env(config['env'],FixedSampler.partial(formula))
        try:
            reset_vocab();VOCAB.update(pickle.loads((args.checkpoint.parent/'vocab.pkl').read_bytes()))
            model=TLTSequenceModel(env,model_configs[config['env']],status['configuration']).to(args.device)
            model.load_state_dict(status['model_state'],strict=True);model.eval()
            agent=ContextAgent(model,StrictSearch(model,env.get_propositions(),num_loops=2),env.get_propositions())
            for episode in range(args.episodes):
                seed=args.seed+episode;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                obs=env.reset(seed=seed);info={'ldba_state_changed':True};agent.reset();history=();done=False;rewards=[];length=0
                while not done:
                    if time.monotonic()-started>=args.max_seconds:stop=True;break
                    history=history+(obs['ldba_state'],)
                    contextual=dict(obs,tlt_context=TLTContext(obs['ldba'],obs['ldba_state'],history))
                    action=agent.get_action(contextual,info,deterministic=config['env']=='PointLtl2-v0')[0]
                    obs,reward,done,info=env.step(np.array(action,copy=True));rewards.append(float(reward));length+=1
                row=dict(formula=formula,seed=seed,completed=bool(done),steps=length,success=bool(info.get('success')) if done else None,violation=bool(info.get('violation')) if done else None,accepting_visits=info.get('num_accepting_visits'),physical_return=sum(rewards))
                rows.append(row);print(json.dumps(row),flush=True)
                (args.output/'episodes.json').write_text(json.dumps(rows,indent=2))
                if stop:break
        finally:env.close()
        if stop:break
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==digest
    completed=[r for r in rows if r['completed']]
    result=dict(evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formula_list=args.formulas,reset_seed=args.seed,episodes_per_formula=args.episodes,protocol=PROTOCOL,configuration=status['configuration'],checkpoint_sha256=digest,training_steps=status['num_steps'],episodes=rows,completed_episodes=len(completed),requested_episodes=len(args.formulas)*args.episodes,success_rate=float(np.mean([r['success'] for r in completed])) if completed else None,elapsed_seconds=time.monotonic()-started,budget_stopped=stop,mode='deterministic' if config['env']=='PointLtl2-v0' else 'sample',note='Small diagnostic only; truncated episodes excluded. Recurrent acceptance visits are a finite-horizon proxy, not infinite LTL satisfaction.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
