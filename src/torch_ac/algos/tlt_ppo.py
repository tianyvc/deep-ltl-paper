"""Original DeepLTL PPO with the TLT appendix's TD / full-episode MC auxiliaries.

Rollouts, GAE bootstrapping and task curriculum are kept upstream-compatible.
Incomplete episodes never receive fabricated zero MC labels. Completed-episode
MC examples are consumed only in the update immediately following collection.
"""
import numpy as np
import torch
from torch import nn
from torch_ac.algos.ppo import PPO
from envs.tlt_context import preprocess_tlt


class TLTPPO(PPO):
    def __init__(self,*args,lambda_short=1.,lambda_long=.3,**kwargs):
        super().__init__(*args,**kwargs)
        self.lambda_short=lambda_short;self.lambda_long=lambda_long
        self.completed_mc=[]

    def collect_experiences(self):
        exps,logs=super().collect_experiences()
        episodes=[]
        for env in self.env.envs:
            episodes.extend(env.episode_results);env.episode_results=[]
        logs['completed_episodes']=len(episodes)
        logs['actual_success_rate']=float(np.mean([e['success'] for e in episodes])) if episodes else None
        if self.model.configuration=='baseline':
            for env in self.env.envs:env.drain_completed()
            return exps,logs
        # Evaluate one-step short targets with collection-time parameters,
        # bootstrapping across rollouts but never across episode reset.
        following=[(self.obss[t+1][p] if t+1<self.num_steps_per_proc else self.obs[p])
                   for p in range(self.num_procs) for t in range(self.num_steps_per_proc)]
        values=[]
        with torch.no_grad():
            for start in range(0,len(following),256):
                batch=preprocess_tlt(following[start:start+256],self.propositions,device=self.device)
                values.append(self.model.components(batch)[1]['short'])
        next_values=torch.cat(values)
        next_masks=torch.cat([self.masks[1:],self.mask.unsqueeze(0)],dim=0).transpose(0,1).reshape(-1)
        exps.short_target=exps.reward+self.discount*next_values*next_masks
        completed=[]
        for env in self.env.envs:
            for episode in env.drain_completed():
                running=0.;targets=[]
                for _,reward in reversed(episode):
                    running=reward+self.discount*running;targets.append(running)
                completed.extend((obs,target) for (obs,_),target in zip(episode,reversed(targets)))
        self.completed_mc=completed
        logs['completed_mc_examples']=len(completed)
        return exps,logs

    def update_parameters(self,exps):
        if self.model.configuration=='baseline':
            result=super().update_parameters(exps);self.model.invalidate_cache();return result
        stats=[];grads={};mc=self.completed_mc
        for _ in range(self.epochs):
            for indices in self._get_batches_starting_indexes():
                batch=exps[indices]
                distribution,parts=self.model.components(batch.obs)
                ratio=(distribution.log_prob(batch.action.clone())-batch.log_prob).exp()
                policy=-torch.minimum(ratio*batch.advantage,ratio.clamp(1-self.clip_eps,1+self.clip_eps)*batch.advantage).mean()
                clipped=batch.value+(parts['total']-batch.value).clamp(-self.clip_eps,self.clip_eps)
                value=torch.maximum((parts['total']-batch.returnn).square(),(clipped-batch.returnn).square()).mean()
                entropy=distribution.entropy().mean()
                short=(parts['short']-batch.short_target).square().mean() if self.model.hv else value*0.
                long=value*0.
                if self.model.hv and mc:
                    selected=np.random.choice(len(mc),size=min(len(indices),len(mc)),replace=False)
                    auxiliary=preprocess_tlt([mc[i][0] for i in selected],self.propositions,device=self.device)
                    targets=torch.tensor([mc[i][1] for i in selected],device=self.device,dtype=parts['total'].dtype)
                    long=(self.model.components(auxiliary)[1]['long']-targets).square().mean()
                loss=policy-self.entropy_coef*entropy+self.value_loss_coef*value+self.lambda_short*short+self.lambda_long*long
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TLT PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                for name,param in self.model.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():raise FloatingPointError(name)
                        group=name.split('.')[0];grads[group]=grads.get(group,0.)+float(param.grad.norm())**2
                norm=nn.utils.clip_grad_norm_(self.model.parameters(),self.max_grad_norm)
                self.optimizer.step();self.model.invalidate_cache()
                if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError('Nonfinite parameter')
                stats.append([float(loss.detach()),float(policy.detach()),float(value.detach()),float(short.detach()),float(long.detach()),float(norm),float(entropy.detach())])
        result=dict(zip(['loss','policy_loss','value_loss','short_loss','long_loss','grad_norm','entropy'],np.mean(stats,axis=0).tolist()))
        result['gradient_groups']={k:v**.5 for k,v in grads.items()};result['mc_examples']=len(mc)
        self.completed_mc=[]
        return result
