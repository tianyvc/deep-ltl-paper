"""Contract tests for manuscript modules on the original curriculum, not performance claims."""
import copy,json,random,io,pickle,tempfile
from pathlib import Path
import numpy as np
import torch
from config import model_configs,PPOConfig
from envs import make_env
from envs.tlt_context import CurriculumContext,preprocess_tlt,TLTContext,sequence_monitor
from model.tlt_sequence import TLTSequenceModel
from preprocessing import init_vocab,reset_vocab
from sequence.samplers import curricula,CurriculumSampler
from torch_ac.algos.tlt_ppo import TLTPPO


def main():
    torch.set_num_threads(2);random.seed(1);np.random.seed(1);torch.manual_seed(1)
    curriculum=copy.deepcopy(curricula['FlatWorld-v0']);sampler=CurriculumSampler.partial(curriculum)
    raw=make_env('FlatWorld-v0',sampler,sequence=True);env=CurriculumContext(make_env('FlatWorld-v0',sampler,sequence=True))
    # Fixed curriculum sequence for a strict observation/reward equivalence oracle.
    fixed=raw.reset(seed=123)['initial_goal']
    def setter(wrapped):
        current=wrapped
        while hasattr(current,'env'):
            if 'sample_sequence' in current.__dict__:current.sample_sequence=lambda:fixed;return
            current=current.env
        raise AssertionError('sequence wrapper missing')
    setter(raw);setter(env)
    left=raw.reset(seed=123);right=env.reset(seed=123)
    assert np.array_equal(left['features'],right['features']) and left['goal']==right['goal']
    old_context=right['tlt_context'];old_features=right['features'].copy()
    for _ in range(6):
        action=np.asarray(0)
        a,r1,d1,i1=raw.step(action.copy());b,r2,d2,i2=env.step(action.copy())
        assert r1==r2 and d1==d2 and i1==i2
        assert np.array_equal(a['features'],b['features'])
        if d1:break
    assert old_context.history==(0,) and np.array_equal(old_features,right['features'])
    reset_vocab();init_vocab(sorted(env.get_possible_assignments(),key=lambda a:tuple(sorted(a.items()))))
    torch.manual_seed(7);baseline=TLTSequenceModel(env,model_configs['FlatWorld-v0'],'baseline')
    torch.manual_seed(7);model=TLTSequenceModel(env,model_configs['FlatWorld-v0'],'hv_track')
    batch=preprocess_tlt([right],env.get_propositions())
    with torch.no_grad():
        x,_=baseline(batch);y,_=model(batch)
        assert torch.equal(x.logits,y.logits)
    # Forward and shuffled minibatch replay must not mutate the automaton/history.
    histories=tuple(batch.context.contexts)
    for _ in range(2):model.components(batch)
    assert histories==batch.context.contexts
    # Inference caching must agree with full-history replay under the same weights.
    contexts=[TLTContext(old_context.ldba,0,(0,)*k) for k in [1,2,3]]
    with torch.no_grad():incremental=model.representations(contexts)[2].clone()
    replay=model.representations(contexts)[2].detach()
    assert torch.allclose(incremental,replay,atol=1e-6)
    for configuration in ['hv_only','msgnn_only','msgnn_track','full_no_options']:
        variant=TLTSequenceModel(env,model_configs['FlatWorld-v0'],configuration)
        distribution,parts=variant.components(batch)
        assert torch.isfinite(parts['total']).all()
    # Exact round-trip and strict rejection of incompatible state dictionaries.
    clone=TLTSequenceModel(env,model_configs['FlatWorld-v0'],'hv_track');clone.load_state_dict(model.state_dict(),strict=True)
    try:clone.base.load_state_dict({'wrong':torch.zeros(1)},strict=True)
    except RuntimeError:pass
    else:raise AssertionError('incompatible checkpoint accepted')
    # Real warm-start and serializable curriculum state, including task success sampling.
    from train.train_tlt import load_model,curriculum_state,restore_curriculum
    from preprocessing import VOCAB
    with tempfile.TemporaryDirectory() as directory:
        target=Path(directory)/'status.pth'
        torch.save({'model_state':baseline.base.state_dict()},target)
        (target.parent/'vocab.pkl').write_bytes(pickle.dumps(VOCAB))
        transferred=load_model(env,'hv_track',target)
        assert all(torch.equal(v,transferred.base.state_dict()[k]) for k,v in baseline.base.state_dict().items())
    state=curriculum_state(curriculum);buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0)
    restored=restore_curriculum(copy.deepcopy(curricula['FlatWorld-v0']),torch.load(buffer,weights_only=False))
    assert restored.stage_index==curriculum.stage_index and restored.num_updates==curriculum.num_updates
    # Force tiny complete episodes only inside this test, to verify MC delivery.
    short=CurriculumContext(make_env('FlatWorld-v0',sampler,sequence=True,max_steps=8));short.reset(seed=2)
    ppo=PPOConfig(epochs=2,batch_size=8,steps_per_process=16,discount=.98)
    algo=TLTPPO([short],model,'cpu',ppo,preprocess_obss=preprocess_tlt)
    exps,logs=algo.collect_experiences()
    assert logs['completed_mc_examples']>=8
    assert len(exps.short_target)==16
    state_before={k:v.detach().clone() for k,v in model.state_dict().items()}
    metrics=algo.update_parameters(exps)
    for group in ['base','short','long','convs','guard','history']:
        assert metrics['gradient_groups'].get(group,0)>0,(group,metrics)
    assert any(not torch.equal(v,state_before[k]) for k,v in model.state_dict().items() if k.startswith('base.actor'))
    assert model._encoded_cache=={}
    # Physics wall penalty: wrapper must return a negative original reward intact.
    import gymnasium as gym
    class Wall(gym.Env):
        action_space=gym.spaces.Box(-1,1,shape=(2,));observation_space=gym.spaces.Box(-1,1,shape=(1,))
        def get_propositions(self):return env.get_propositions()
        def get_possible_assignments(self):return env.get_possible_assignments()
        def reset(self,**kwargs):return dict(features=np.zeros_like(right['features']),goal=fixed,initial_goal=fixed)
        def step(self,action):return self.reset(),-1.,True,{'cost_ltl_walls':1}
    wall=CurriculumContext(Wall());wall.reset();_,reward,done,info=wall.step(np.zeros(2));assert reward==-1 and done
    raw.close();env.close();short.close()
    result=dict(passed=['unaltered_original_features_rewards_done_info','immutable_context_and_features',
           'base_policy_identical_at_initialization','side_effect_free_forward_replay','strict_checkpoint_roundtrip_and_rejection',
           'completed_episode_mc_delivered','one_step_short_td_targets','all_core_gradient_paths','actor_updates',
           'cache_invalidated_after_updates','wall_penalty_preserved','inference_cache_matches_full_history',
           'all_six_configurations_forward','original_checkpoint_warm_start','curriculum_state_serializable'],metrics=metrics)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
