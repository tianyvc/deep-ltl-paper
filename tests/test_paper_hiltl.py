"""Independent semantic and training-oracle checks; run as a script in wty_LTL."""
import copy
import json
import numpy as np
import torch
from torch import nn

from envs import make_env
from ltl.automata import LDBA, LDBASequence
from ltl.samplers import FixedSampler
from model.discrete_epsilon_distribution import DiscreteEpsilonDistribution
from model.gnn_net import SimpleGlobalAttention
from model.paper_hiltl import AutomatonGraph, PaperHiLTL, ProductEpisode, discounted_returns
from train.train_paper import episode_targets


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    checks = []
    from sequence.samplers.curriculum import Curriculum, RandomCurriculumStage
    curriculum = Curriculum([RandomCurriculumStage(.8, 'min', lambda _: None)])
    curriculum.update_task_success({})
    assert curriculum.stage_index == 0 and curriculum.num_updates == 0
    curriculum.update_task_success({'completed': 1.})
    assert curriculum.finished
    curriculum.update_task_success({'completed': 1.})
    checks.append('curriculum_empty_statistics_and_finished_guard')
    distribution = DiscreteEpsilonDistribution(torch.zeros(2, 3))
    distribution.set_epsilon_mask(torch.tensor([True, False]))
    actions = torch.tensor([LDBASequence.EPSILON, 0])
    saved = actions.clone()
    distribution.log_prob(actions)
    assert torch.equal(actions, saved), 'log_prob mutated rollout actions'
    checks.append('epsilon_action_input_immutable')

    pooling = SimpleGlobalAttention(nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        pooling.gate_nn.weight.zero_()
    features = torch.tensor([[1., 3.], [3., 5.], [10., 20.]])
    assert torch.allclose(pooling(features, torch.tensor([0, 0, 1])), torch.tensor([[2., 4.], [10., 20.]]))
    checks.append('attention_uniform_logits_equals_per_graph_mean')

    ldba = LDBA({'a'}, simplify_labels=False)
    for state in range(4):
        ldba.add_state(state, initial=state == 0)
    ldba.add_transition(0, 1, 'a', False)
    ldba.add_transition(0, 3, '!a', False)
    ldba.add_transition(1, 2, 't', False)
    ldba.add_transition(2, 2, 't', True)
    ldba.add_transition(3, 3, 't', False)
    ldba.compute_sccs()
    graph = AutomatonGraph.from_ldba(ldba)
    assert graph.distance[0] == 2 and graph.distance[1] == 1 and graph.distance[2] == 0
    assert graph.nodes[:, 3].tolist() == [1., .5, 0., 1.]
    assert graph.nodes[:, 0].tolist() == [0., 0., 1., 0.]
    assert graph.nodes[:, 1].tolist() == [0., 0., 1., 1.]
    assert graph.edges.shape[1] == ldba.num_transitions
    assert torch.equal(graph.edge_features[:, :4].sum(1), torch.ones(ldba.num_transitions))
    assert torch.isfinite(graph.nodes).all()
    checks.append('hand_graph_distance_acceptance_trap_scc_edges')

    no_accept = LDBA({'a'}, simplify_labels=False)
    no_accept.add_state(0, initial=True)
    no_accept.add_transition(0, 0, 't', False)
    assert AutomatonGraph.from_ldba(no_accept).nodes[0, 3] == 1
    zero_distance = LDBA({'a'}, simplify_labels=False)
    zero_distance.add_state(0, initial=True)
    zero_distance.add_transition(0, 0, 't', True)
    assert AutomatonGraph.from_ldba(zero_distance).nodes[0, 3] == 0
    checks.append('no_accepting_and_zero_denominator_finite')
    from ltl.automata import ltl2ldba
    assert not no_accept.is_finite_specification()
    assert zero_distance.is_finite_specification()
    assert not ltl2ldba('G F blue').is_finite_specification()
    assert ltl2ldba('F blue').is_finite_specification()
    recurring = ProductEpisode(make_env('FlatWorld-v0', FixedSampler.partial('G F blue'), max_steps=4))
    recurring.reset(seed=1)
    assert not recurring.terminate_on_acceptance
    for step in range(4):
        _, _, ended, _ = recurring.step(recurring.action_space.sample())
        assert ended == (step == 3)
    recurring.close()
    checks.append('recurrence_not_finite_and_runs_to_defined_horizon')

    values = torch.tensor([.2, .4])
    td, returns, adv, mc = episode_targets([1., 2.], values, .5, 1.)
    assert torch.allclose(td, torch.tensor([1.2, 2.]))
    assert torch.allclose(returns, torch.tensor([2., 2.]))
    assert torch.allclose(mc, torch.tensor([2., 2.]))
    assert torch.allclose(discounted_returns([1., 2., 3.], .5), torch.tensor([2.75, 3.5, 3.]))
    checks.append('independent_two_step_td_gae_mc_oracle')

    env = ProductEpisode(make_env('FlatWorld-v0', FixedSampler.partial('F blue'), max_steps=4))
    observation = env.reset(seed=1)
    model = PaperHiLTL(observation['features'].shape, env.action_space, 'full_no_options', propositions=observation['ldba'].propositions)
    model.eval()
    history = observation['history']
    state = observation['ldba_state']
    dist, predicted = model([observation])
    _, duplicate = model([observation, observation])
    assert torch.allclose(predicted['total'].expand(2), duplicate['total'], atol=1e-6)
    assert observation['history'] == history and observation['ldba_state'] == state
    assert torch.allclose(predicted['total'], (1-predicted['beta'])*predicted['short'] + predicted['beta']*predicted['long'])
    assert predicted['beta'].min() >= torch.sigmoid(torch.tensor(-1.5)) - 1e-6
    assert predicted['beta'].max() <= torch.sigmoid(torch.tensor(1.5)) + 1e-6
    checks.append('batch_equivalence_pure_forward_fixed_gate')

    altered = dict(observation, features=np.asarray(observation['features']) + 2.)
    _, predicted_altered = model([altered])
    assert torch.allclose(predicted['long'], predicted_altered['long'])
    checks.append('long_head_environment_independent')
    loss = predicted['short'].square().mean() + (predicted['long'] - 1).square().mean() - dist.log_prob(dist.sample().clone()).mean()
    loss.backward()
    for group in ['short_head', 'long_head', 'convs', 'history_encoder', 'actor', 'pool_gates']:
        norm = sum(parameter.grad.square().sum().item() for name, parameter in model.named_parameters() if name.startswith(group + '.') and parameter.grad is not None)
        assert norm > 0 and np.isfinite(norm), group
    checks.append('actual_gradients_both_heads_gnn_history_actor_pooling')

    saved_obs = copy.copy(observation)
    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(np.asarray(action))
    expected_state, accepting = observation['ldba'].get_next_state(state, next_observation['propositions'])
    assert next_observation['ldba_state'] == expected_state
    assert next_observation['history'] == history + (expected_state,)
    assert saved_obs['history'] == history and saved_obs['ldba_state'] == state
    checks.append('real_transition_q_matches_automaton_snapshot_immutable')
    restored = PaperHiLTL(observation['features'].shape, env.action_space, 'full_no_options', propositions=observation['ldba'].propositions)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        _, reloaded = restored([observation])
    assert torch.allclose(predicted['total'], reloaded['total'])
    checks.append('strict_state_dict_roundtrip')
    env.close()
    print(json.dumps({'passed': checks, 'count': len(checks)}))


if __name__ == '__main__':
    main()
