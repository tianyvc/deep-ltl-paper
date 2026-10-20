"""Independent acceptance checks for guard conditioning, options and task isolation."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from envs import make_env
from ltl.samplers import FixedSampler
from ltl.automata import LDBA
from model.paper_hiltl import PaperHiLTL, ProductEpisode
from model.paper_options import available_options, option_transition, smdp_targets
from train.paper_tasks import load_tasks
from train.train_paper import collect_episode, update


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    checks = []
    import importlib.util
    spec = importlib.util.spec_from_file_location('statistics_tool', 'scripts/summarize_paper_runs.py')
    statistics_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statistics_tool)
    statistics = statistics_tool.summarize([.1, .2, .3])
    assert abs(statistics['mean'] - .2) < 1e-8
    assert abs(statistics['sample_std'] - .1) < 1e-8
    assert statistics['ci95'][0] < .2 < statistics['ci95'][1]
    assert statistics_tool.summarize([.2])['ci95'] is None
    checks.append('independent_seed_mean_std_and_ci_oracle')
    environment = ProductEpisode(make_env('FlatWorld-v0', FixedSampler.partial('F blue & F green'), max_steps=8))
    obs = environment.reset(seed=1)
    model = PaperHiLTL(obs['features'].shape, environment.action_space, 'full_all', propositions=obs['ldba'].propositions)
    candidates = available_options(obs)
    assert len(candidates) >= 2
    edges = [edge for q in sorted(obs['ldba'].state_to_transitions) for edge in obs['ldba'].state_to_transitions[q]]
    assert all(edges[index].target != obs['ldba_state'] or edges[index].accepting or edges[index].is_epsilon() for index in candidates)
    meta, available = model.option_distribution(obs)
    assert torch.allclose(meta.probs.sum(), torch.tensor(1.))
    assert candidates == available
    saved_history = obs['history']
    configured = dict(obs, option_edge=candidates[0])
    _, first = model([configured])
    _, duplicate = model([configured, configured])
    assert torch.allclose(first['total'].expand(2), duplicate['total'], atol=1e-6)
    assert obs['history'] == saved_history
    checks.append('topology_mask_variable_option_distribution_and_pure_replay')

    td, returns, advantages = smdp_targets([2., 3.], [2, 1], torch.tensor([.2, .4]), .5, 1.)
    assert torch.allclose(td, torch.tensor([2.1, 3.]))
    assert torch.allclose(returns, torch.tensor([2.75, 3.]))
    assert torch.allclose(advantages, torch.tensor([2.55, 2.6]))
    checks.append('smdp_gamma_power_duration_independent_oracle')

    recurring = LDBA({'blue'}, simplify_labels=False)
    recurring.add_state(0, initial=True)
    recurring.add_transition(0, 0, 'blue', True)
    recurring.add_transition(0, 0, '!blue', False)
    recurring.compute_sccs()
    before = dict(ldba=recurring, ldba_state=0, propositions=set())
    reward, reason = option_transition(before, before, {}, 0, 1, 4)
    assert reward == 0 and reason is None
    after = dict(before, propositions={'blue'})
    reward, reason = option_transition(before, after, {}, 0, 1, 4)
    assert reward == 1 and reason == 'target'
    reward, reason = option_transition(before, before, {}, 0, 4, 4)
    assert reason == 'timeout'
    epsilon = LDBA({'blue'}, simplify_labels=False)
    epsilon.add_state(0, initial=True)
    epsilon.add_state(1)
    epsilon.add_transition(0, 1, None, False)
    epsilon.add_transition(1, 1, 't', True)
    epsilon.compute_sccs()
    old, new = dict(ldba=epsilon, ldba_state=0), dict(ldba_state=1)
    assert option_transition(old, new, {'took_epsilon': True}, 0, 1, 4) == (1.05, 'target')
    checks.append('recurrence_guard_timeout_and_epsilon_option_termination')

    other = ProductEpisode(make_env('FlatWorld-v0', FixedSampler.partial('F green'), max_steps=4))
    blue = ProductEpisode(make_env('FlatWorld-v0', FixedSampler.partial('F blue'), max_steps=4))
    obs_green, obs_blue = other.reset(seed=3), blue.reset(seed=3)
    guard_model = PaperHiLTL(obs_blue['features'].shape, blue.action_space, 'hv_track', propositions=obs_blue['ldba'].propositions)
    graph_blue, graph_green = guard_model.graph(obs_blue['ldba']), guard_model.graph(obs_green['ldba'])
    assert torch.equal(graph_blue.nodes, graph_green.nodes) and torch.equal(graph_blue.edges, graph_green.edges)
    assert graph_blue.guard_tokens != graph_green.guard_tokens
    _, values_blue = guard_model([obs_blue])
    _, values_green = guard_model([dict(obs_green, features=obs_blue['features'])])
    representation_blue = guard_model.representations([obs_blue])[0]
    representation_green = guard_model.representations([dict(obs_green, features=obs_blue['features'])])[0]
    assert float((representation_blue - representation_green).norm()) > 1e-4
    assert not torch.allclose(values_blue['total'], values_green['total'], atol=1e-8, rtol=0)
    checks.append('isomorphic_formulas_distinguished_by_guard_encoder')

    manifest = load_tasks('configurations/flatworld_tasks.json', None, 'FlatWorld-v0', 4)
    assert not set(manifest['train']) & set(manifest['held_out'])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'leak.json'
        path.write_text(json.dumps(dict(train=['F blue & F green'], held_out=['(F green) & (F blue)'])))
        try:
            load_tasks(path, None, 'FlatWorld-v0', 4)
        except ValueError as exc:
            assert 'overlap' in str(exc)
        else:
            raise AssertionError('Equivalent compiled held-out task was accepted')
    checks.append('task_manifest_rejects_compiled_training_heldout_overlap')

    episode = collect_episode(environment, model, seed=2, option_max_steps=3)
    segments = episode[6]['segments']
    assert sum(item['duration'] for item in segments) == len(episode[4])
    assert all(item['end'] - item['start'] == item['duration'] for item in segments)
    assert all('option_edge' in observation for observation in episode[0])
    before_weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = update(model, optimizer, [episode], Namespace(gamma=.998, epochs=1, batch_size=8))
    for group in ['meta_controller', 'option_encoder', 'option_critic', 'actor', 'guard_encoder']:
        changed = any(not torch.equal(before_weights[name], parameter) for name, parameter in model.named_parameters() if name.startswith(group + '.'))
        assert changed, group
    assert metrics['meta_gradient_norm'] > 0
    assert all(np.isfinite(float(metrics[key])) for key in ['loss', 'meta_loss', 'meta_gradient_norm'])
    checks.append('option_rollout_segments_and_actual_parameter_updates_all_levels')

    from config import model_configs
    from model.deepltl_reference import PaperDeepLTL
    baseline = PaperDeepLTL(blue, model_configs['FlatWorld-v0'])
    prepared = baseline.prepare_observation(obs_blue)
    assert 'baseline_sequence' in prepared
    _, value = baseline([prepared])
    loss = (value['total'] - 1).square().mean()
    loss.backward()
    assert not any('gnn' in name or 'convs' in name for name, _ in baseline.named_parameters())
    for group in ['reference.ltl_net.set_network', 'reference.ltl_net.rnn', 'reference.critic']:
        assert sum(float(p.grad.square().sum()) for name, p in baseline.named_parameters() if name.startswith(group) and p.grad is not None) > 0, group
    _, repeated = baseline([prepared])
    assert torch.allclose(value['total'], repeated['total'])
    checks.append('reference_baseline_deepsets_gru_gradients_without_gnn')
    environment.close()
    other.close()
    blue.close()
    print(json.dumps(dict(passed=checks, count=len(checks))))


if __name__ == '__main__':
    main()
