"""Compare the pinned upstream builder with the compatible sequence base (read-only)."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from config import model_configs
from envs import make_env
from envs.tlt_context import CurriculumContext,preprocess_tlt
from model.tlt_sequence import TLTSequenceModel
from preprocessing import init_vocab,reset_vocab
from sequence.samplers import curricula,CurriculumSampler

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,default=Path('tests/fixtures/deepltl'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    original=load('pinned_upstream_model',args.source_dir/'model.py')
    configs=load('pinned_upstream_config',args.source_dir/'model_config.py')
    rows=[];torch.set_num_threads(2)
    for name,upstream_config in [('PointLtl2-v0',configs.zones),('FlatWorld-v0',configs.flatworld)]:
        current=model_configs[name]
        for key in ['actor','critic','ltl_embedding_dim','num_rnn_layers','env_net','set_net']:
            assert repr(getattr(current,key))==repr(getattr(upstream_config,key)),(name,key)
        sampler=CurriculumSampler.partial(copy.deepcopy(curricula[name]));env=CurriculumContext(make_env(name,sampler,sequence=True))
        try:
            observation=env.reset(seed=123);reset_vocab();init_vocab(sorted(env.get_possible_assignments(),key=lambda a:tuple(sorted(a.items()))))
            torch.manual_seed(19);expected=original.build_model(env,{},upstream_config)
            torch.manual_seed(19);actual=TLTSequenceModel(env,current,'baseline')
            assert expected.state_dict().keys()==actual.base.state_dict().keys()
            assert all(torch.equal(v,actual.base.state_dict()[k]) for k,v in expected.state_dict().items())
            batch=preprocess_tlt([observation],env.get_propositions())
            with torch.no_grad():
                left,lv=expected(batch);right,rv=actual(batch)
                assert torch.equal(lv,rv)
                if hasattr(left,'dist'):assert torch.equal(left.dist.mean,right.dist.mean) and torch.equal(left.dist.stddev,right.dist.stddev)
                else:assert torch.equal(left.logits,right.logits)
            rows.append(dict(env=name,configuration_equal_excluding_unused_gnn=True,state_keys_and_seeded_tensors_equal=True,action_distribution_and_value_equal=True,parameters=sum(x.numel() for x in expected.parameters())))
        finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True);result=dict(commit=json.loads((args.source_dir/'revision.json').read_text())['commit'],checks=rows,scope='Upstream builder/config with current shared layers; not a complete byte-identical upstream dependency audit.')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
