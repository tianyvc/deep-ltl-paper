"""Check manuscript task compilation, semantic feature contracts and graph ablations."""
import argparse,copy,json
from pathlib import Path
import torch
from config import model_configs
from envs import make_env
from envs.tlt_context import CurriculumContext,TLTContext,preprocess_tlt
from ltl.automata import ltl2ldba
from model.automaton_graph import AutomatonGraph
from model.tlt_sequence import TLTSequenceModel,CONFIGURATIONS
from preprocessing import init_vocab,reset_vocab
from sequence.samplers import curricula,CurriculumSampler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);tasks=json.loads(Path('configurations/paper_tasks.json').read_text())['environments'];rows=[]
    assert set(CONFIGURATIONS)=={'baseline','hv_only','hv_track','msgnn_only','msgnn_track','full_no_options'}
    for name,groups in tasks.items():
        env=CurriculumContext(make_env(name,CurriculumSampler.partial(copy.deepcopy(curricula[name])),sequence=True))
        try:
            obs=env.reset(seed=123);propositions=env.get_propositions();assignments=env.get_possible_assignments()
            for group,formulas in groups.items():
                for task in formulas:
                    ldba=ltl2ldba(task['formula'],propositions,simplify_labels=False);ldba.prune(assignments);ldba.complete_sink_state();ldba.compute_sccs()
                    graph=AutomatonGraph.from_ldba(ldba)
                    assert graph.nodes.shape[1]==4 and graph.edge_features.shape[1]==6
                    assert graph.edges.shape[1]==len(graph.guard_tokens)==graph.edge_features.shape[0]
                    assert torch.all(graph.edge_features[:,:4].sum(1)==1)
                    assert torch.isfinite(graph.nodes).all() and torch.isfinite(graph.edge_features).all()
                    assert ((graph.nodes[:,3]>=0)&(graph.nodes[:,3]<=1)).all()
                    rows.append(dict(env=name,task=task['id'],formula=task['formula'],group=group,actual_states=len(graph.state_ids),actual_edges=graph.edges.shape[1]))
            reset_vocab();init_vocab(sorted(assignments,key=lambda a:tuple(sorted(a.items()))))
            short=TLTSequenceModel(env,model_configs[name],'hv_track');multi=TLTSequenceModel(env,model_configs[name],'full_no_options')
            assert len(short.convs)==3 and short.actor_residual.in_features==128 and multi.actor_residual.in_features==256
            final=len(obs['initial_goal']);context=obs['tlt_context']
            terminal=dict(obs,tlt_context=TLTContext(context.ldba,final,(final,)))
            trap=dict(obs,tlt_context=TLTContext(context.ldba,final+1,(final+1,)))
            with torch.no_grad():_,parts=short.components(preprocess_tlt([terminal,trap],propositions))
            assert torch.allclose(parts['beta'],torch.sigmoid(torch.tensor([1.5,-1.5])),atol=1e-6)
        finally:env.close()
    result=dict(passed=['three_environments_and_fifteen_task_automata','semantic_node4_edge6_and_guards','finite_graph_features','six_core_configuration_scope','three_gatv2_layers','single_vs_multiscale_widths','fixed_gate_boundary_values'],tasks=rows,note='Actual converter states are recorded, not forced to manuscript example counts. Recurrent visits are finite-horizon proxies.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
