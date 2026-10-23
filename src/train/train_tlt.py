"""Train manuscript variants on the unchanged official DeepLTL sequence curriculum.

Environment rewards and the original sequence curriculum remain unchanged.
Default Zone hyperparameters are copied from upstream run_zones.py. Every run
has explicit budget/output, strict checkpoint import, and validation-only gates.
"""
import argparse,copy,hashlib,json,os,pickle,random,time
from pathlib import Path
import numpy as np
import torch
from config import model_configs,PPOConfig
from envs import make_env,get_env_attr
from envs.tlt_context import CurriculumContext,preprocess_tlt
from model.tlt_sequence import TLTSequenceModel,CONFIGURATIONS,PROTOCOL
from preprocessing import init_vocab,reset_vocab,VOCAB
from sequence.samplers import curricula,CurriculumSampler
from torch_ac.algos.tlt_ppo import TLTPPO


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def load_model(env,configuration,warm_start=None):
    if warm_start:
        vocab=warm_start.parent/'vocab.pkl'
        if not vocab.is_file():raise FileNotFoundError('Original vocabulary required: '+str(vocab))
        reset_vocab();VOCAB.update(pickle.loads(vocab.read_bytes()))
    else:
        reset_vocab();init_vocab(sorted(env.get_possible_assignments(),key=lambda a:tuple(sorted(a.items()))))
    model=TLTSequenceModel(env,model_configs[env.spec.id if env.spec is not None else 'PointLtl2-v0'],configuration)
    if warm_start:
        status=torch.load(warm_start,map_location='cpu',weights_only=False)
        if status.get('protocol')==PROTOCOL:
            if status['configuration']!=configuration:raise ValueError('Use a baseline original checkpoint for cross-variant initialization')
            model.load_state_dict(status['model_state'],strict=True)
        else:
            state=status.get('model_state')
            if state is None:raise ValueError('Expected an original sequence model_state, not a product status.pt')
            model.base.load_state_dict(state,strict=True)
    return model


def curriculum_state(curriculum):
    def stage_state(stage):
        result={k:copy.deepcopy(getattr(stage,k)) for k in ['_tasks','_task_success'] if hasattr(stage,k)}
        if hasattr(stage,'stages'):result['children']=[stage_state(s) for s in stage.stages]
        return result
    return dict(stage_index=curriculum.stage_index,num_updates=curriculum.num_updates,stages=[stage_state(s) for s in curriculum.stages])


def restore_curriculum(curriculum,state):
    def restore(stage,data):
        for key in ['_tasks','_task_success']:
            if key in data:setattr(stage,key,data[key])
        for child,child_state in zip(getattr(stage,'stages',[]),data.get('children',[])):restore(child,child_state)
    curriculum.stage_index=state['stage_index'];curriculum.num_updates=state['num_updates']
    for stage,data in zip(curriculum.stages,state['stages']):restore(stage,data)
    return curriculum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',choices=['PointLtl2-v0','LetterEnv-v0','FlatWorld-v0'],default='PointLtl2-v0')
    parser.add_argument('--configuration',choices=list(CONFIGURATIONS),default='hv_track')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--num-procs',type=int,default=16)
    parser.add_argument('--rollout-steps',type=int,default=4096)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-seconds',type=float,default=0.,help='Optional stop between updates; not a kill timeout')
    args=parser.parse_args()
    if min(args.steps,args.num_procs,args.rollout_steps,args.epochs,args.batch_size,args.threads)<=0:parser.error('Training sizes must be positive')
    if args.resume and args.warm_start:parser.error('resume and warm-start are distinct operations')
    torch.set_num_threads(args.threads);seed_all(args.seed)
    if args.resume:
        saved=torch.load(args.output/'status.pth',map_location='cpu',weights_only=False)
        if saved.get('protocol')!=PROTOCOL or saved['configuration']!=args.configuration:raise ValueError('Resume protocol/configuration mismatch')
        previous=json.loads((args.output/'config.json').read_text())
        for key in ['env','seed','num_procs','rollout_steps','epochs','batch_size']:
            if previous[key]!=getattr(args,key):raise ValueError('Resume setting changed: '+key)
    else:
        args.output.mkdir(parents=True,exist_ok=False);saved=None
    curriculum=copy.deepcopy(curricula[args.env])
    if saved:restore_curriculum(curriculum,saved['curriculum_state'])
    if curriculum.finished:raise ValueError('Curriculum already finished; evaluate the saved checkpoint')
    sampler=CurriculumSampler.partial(curriculum);envs=[]
    for i in range(args.num_procs):
        env=CurriculumContext(make_env(args.env,sampler,sequence=True));env.reset(seed=100*args.seed+i);envs.append(env)
    model=load_model(envs[0],args.configuration,args.warm_start)
    if saved:
        reset_vocab();VOCAB.update(pickle.loads((args.output/'vocab.pkl').read_bytes()))
        model=TLTSequenceModel(envs[0],model_configs[args.env],args.configuration);model.load_state_dict(saved['model_state'],strict=True)
    (args.output/'vocab.pkl').write_bytes(pickle.dumps(VOCAB))
    discount=.998 if args.env=='PointLtl2-v0' else .98 if args.env.startswith('FlatWorld') else .94
    ppo=PPOConfig(epochs=args.epochs,batch_size=args.batch_size,steps_per_process=args.rollout_steps,discount=discount,lr=.0003,entropy_coef=.003 if args.env=='PointLtl2-v0' else .01)
    # Match random stream after architecture initialization for paired runs.
    seed_all(args.seed)
    algo=TLTPPO(envs,model,args.device,ppo,preprocess_obss=preprocess_tlt,parallel=False)
    steps=saved['num_steps'] if saved else 0;updates=saved['num_updates'] if saved else 0
    if saved:
        algo.optimizer.load_state_dict(saved['optimizer_state'])
        random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
        if torch.cuda.is_available() and saved.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(saved['rng_cuda'])
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    paths=sorted(str(p) for p in Path('src').rglob('*.py') if 'safety-gymnasium' not in p.parts)
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    if saved and saved['implementation_sha256']!=hashes:raise ValueError('Resume implementation changed')
    config.update(protocol=PROTOCOL,implementation_sha256=hashes,ppo=vars(ppo),parameters=sum(p.numel() for p in model.parameters()),warm_start_sha256=hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None)
    atomic_json(args.output/'config.json',config)
    started=time.monotonic()
    try:
        with (args.output/'metrics.jsonl').open('a') as stream:
            while steps<args.steps and not curriculum.finished:
                stage=curriculum.stage_index
                experiences,logs=algo.collect_experiences()
                curriculum.update_task_success(logs['avg_goal_success'],verbose=False)
                values=algo.update_parameters(experiences);steps+=logs['num_steps'];updates+=1
                values.update(completed_episodes=logs.get('completed_episodes',None),steps=steps,update=updates,curriculum_stage_collected=stage,curriculum_stage_next=curriculum.stage_index,elapsed_seconds=time.monotonic()-started,success_rate=logs['actual_success_rate'],goals_observed=len(logs['avg_goal_success']))
                stream.write(json.dumps(values)+'\n');stream.flush();print(json.dumps(values),flush=True)
                status=dict(protocol=PROTOCOL,configuration=args.configuration,model_state=model.state_dict(),optimizer_state=algo.optimizer.state_dict(),num_steps=steps,num_updates=updates,curriculum_stage=curriculum.stage_index,curriculum_state=curriculum_state(curriculum),rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,implementation_sha256=hashes,config=config)
                temp=args.output/'status.tmp';torch.save(status,temp);temp.replace(args.output/'status.pth')
                atomic_json(args.output/'progress.json',dict(state='running',steps=steps,target_steps=args.steps,updates=updates,curriculum_stage=curriculum.stage_index,elapsed_seconds=time.monotonic()-started))
                if args.max_seconds and time.monotonic()-started>=args.max_seconds:break
        atomic_json(args.output/'progress.json',dict(state='completed' if steps>=args.steps or curriculum.finished else 'budget_stopped',steps=steps,target_steps=args.steps,updates=updates,curriculum_stage=curriculum.stage_index,elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        atomic_json(args.output/'progress.json',dict(state='failed',steps=steps,updates=updates,error=repr(exc)))
        raise
    finally:
        for env in envs:env.close()


if __name__=='__main__':main()
