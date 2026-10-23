"""Fair encoders, capacity, history gradients and validation leakage checks."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from envs import make_env
from ltl.samplers import FixedSampler
from model.paper_hiltl import PaperHiLTL, ProductEpisode
from train.paper_tasks import load_tasks


def main():
    torch.set_num_threads(2)
    env=ProductEpisode(make_env('FlatWorld-v0',FixedSampler.partial('F (blue & F green)'),max_steps=8))
    obs=env.reset(seed=1)
    models={}
    for name in ['gnn_single','hv_only','gnn_track','hv_track','gnn_capacity','gnn_track_capacity']:
        torch.manual_seed(17)
        models[name]=PaperHiLTL(obs['features'].shape,env.action_space,name,propositions=obs['ldba'].propositions)
    for a,b in [('gnn_single','hv_only'),('gnn_track','hv_track'),('gnn_single','gnn_capacity'),('gnn_track','gnn_track_capacity')]:
        left,right=models[a].state_dict(),models[b].state_dict()
        for key,value in left.items():
            if key.startswith(('convs.','guard_','edge_encoder.','env_encoder.','node_encoder.','norms.','transforms.','pool_gates.','actor.','history_encoder.')):
                assert torch.equal(value,right[key]),(a,b,key)
        assert torch.allclose(models[a].representations([obs])[0],models[b].representations([obs])[0])
    counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    for a,b in [('gnn_capacity','hv_only'),('gnn_track_capacity','hv_track')]:
        assert abs(counts[a]-counts[b])/counts[b] < .01,(a,b,counts)
    model=models['gnn_track']
    _,values=model([obs])
    values['total'].sum().backward()
    assert sum(float(p.grad.square().sum()) for p in model.history_encoder.parameters() if p.grad is not None)>0
    # Preserve q but change history: the history-only critic must observe it.
    altered=dict(obs,history=obs['history']+(obs['ldba_state'],)*3)
    with torch.no_grad():
        assert not torch.allclose(model([obs])[1]['total'],model([altered])[1]['total'],atol=1e-8,rtol=0)
    with tempfile.TemporaryDirectory() as temp:
        task=Path(temp)/'tasks.json'
        task.write_text(json.dumps({'train':['F blue'],'validation':['F green'],'held_out':['F yellow']}))
        result=load_tasks(task,None,'FlatWorld-v0',8)
        assert result['validation']==['F green']
        task.write_text(json.dumps({'train':['F blue & F green'],'validation':['F green & F blue'],'held_out':['F yellow']}))
        try:
            load_tasks(task,None,'FlatWorld-v0',8)
        except ValueError as exc:
            assert 'overlap' in str(exc)
        else:
            raise AssertionError('Validation leakage accepted')
    env.close()
    print(json.dumps({'passed':['identical_encoder_actor_initialization','identical_actor_inputs','capacity_difference_below_one_percent','single_critic_history_gradient_and_dependence','independent_validation_and_overlap_rejection'],'parameter_counts':counts}))

if __name__=='__main__':
    main()
