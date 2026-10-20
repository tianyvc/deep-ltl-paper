import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn

from envs import make_env
from ltl.samplers import FixedSampler
from model.paper_hiltl import CONFIGURATIONS, PROTOCOL_VERSION, PaperHiLTL, ProductEpisode, discounted_returns
from model.paper_options import option_transition, smdp_targets
from train.paper_tasks import load_tasks, training_sampler


def arguments():
    parser = argparse.ArgumentParser(description='Audited product-LDBA PPO, sequence-reference baseline, options and held-out formula evaluation; not evidence for published performance.')
    parser.add_argument('--env', required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument('--formula')
    tasks.add_argument('--task-set', type=Path, help='JSON with train and held_out LTL lists')
    parser.add_argument('--eval-formula', action='append', default=[])
    parser.add_argument('--eval-split', choices=['train', 'held_out'], default=None)
    parser.add_argument('--gate', choices=['fixed', 'learned', 'constant'], default='fixed')
    parser.add_argument('--option-max-steps', type=int, default=50)
    parser.add_argument('--configuration', choices=CONFIGURATIONS, default='hv_track')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--rollout-steps', type=int, default=512)
    parser.add_argument('--max-episode-steps', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gamma', type=float, default=.998)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate', type=int, default=0, metavar='EPISODES')
    parser.add_argument('--eval-seed', type=int, default=10001)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.steps, args.rollout_steps, args.epochs, args.batch_size) <= 0 or args.evaluate < 0:
        parser.error('Training sizes must be positive and evaluation count nonnegative')
    if args.option_max_steps <= 0 or (args.max_episode_steps is not None and args.max_episode_steps <= 0):
        parser.error("Episode and option horizons must be positive")
    if (args.eval_formula or args.eval_split) and not args.evaluate:
        parser.error("Evaluation selectors require --evaluate")
    if not 0 < args.gamma <= 1 or args.lr <= 0:
        parser.error('Invalid discount or learning rate')
    return args


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checkpoint(path, model, optimizer, steps, updates, config):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), steps=steps, updates=updates,
                    config=config, protocol=PROTOCOL_VERSION, random=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
    temporary.replace(path)


def collect_episode(env, model, seed=None, deterministic=False, option_max_steps=50):
    obs = env.reset(seed=seed) if seed is not None else env.reset()
    observations, actions, log_probs, values, rewards = [], [], [], [], []
    metadata = dict(segments=[], intrinsic=[], option_values=[], formula=obs['goal'])
    option_age, segment = 0, None
    with torch.no_grad():
        while True:
            if hasattr(model, 'prepare_observation'):
                obs = model.prepare_observation(obs)
            if model.options and segment is None:
                meta, candidates = model.option_distribution(obs)
                chosen = meta.probs.argmax() if deterministic else meta.sample()
                edge = candidates[int(chosen)]
                obs = dict(obs, option_edge=edge)
                segment = dict(start=len(rewards), observation=obs, option=edge,
                               log_prob=meta.log_prob(chosen).clone())
                option_age = 0
            elif model.options:
                obs = dict(obs, option_edge=segment['option'])
            distribution, value = model([obs])
            action = distribution.mode if deterministic else distribution.sample()
            observations.append(obs)
            actions.append(action[0].clone())
            log_probs.append(distribution.log_prob(action.clone())[0])
            values.append(value['total'][0])
            following, reward, done, info = env.step(action[0].cpu().numpy())
            rewards.append(reward)
            if model.options:
                option_age += 1
                intrinsic, reason = option_transition(obs, following, info, segment['option'], option_age, option_max_steps)
                metadata['intrinsic'].append(intrinsic)
                metadata['option_values'].append(value['option'][0])
                if done or reason:
                    segment.update(end=len(rewards), duration=option_age, reason='episode_end' if done else reason)
                    metadata['segments'].append(segment)
                    segment = None
            obs = following
            if done:
                break
    return observations, torch.stack(actions), torch.stack(log_probs), torch.stack(values), rewards, info, metadata


def episode_targets(rewards, values, gamma, gae_lambda=.95):
    rewards = torch.as_tensor(rewards, device=values.device, dtype=values.dtype)
    following = torch.cat([values[1:], values.new_zeros(1)])
    td_target = rewards + gamma * following
    advantages = torch.empty_like(values)
    running = values.new_zeros(())
    for index in reversed(range(len(rewards))):
        running = rewards[index] + gamma * following[index] - values[index] + gamma * gae_lambda * running
        advantages[index] = running
    return td_target.detach(), (values + advantages).detach(), advantages.detach(), discounted_returns(rewards.cpu(), gamma).to(values.device)


def update(model, optimizer, episodes, args):
    observations = [obs for episode in episodes for obs in episode[0]]
    actions = torch.cat([episode[1] for episode in episodes])
    old_probs = torch.cat([episode[2] for episode in episodes]).detach()
    old_values = torch.cat([episode[3] for episode in episodes]).detach()
    targets = [episode_targets(episode[4], episode[3], args.gamma) for episode in episodes]
    td, returns, advantages, mc = [torch.cat([target[index] for target in targets]) for index in range(4)]
    advantages = ((advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)).clamp(-1, 1)
    stats = []
    gradient_groups = {key: 0. for key in ['reference', 'short_head', 'long_head', 'convs', 'guard_encoder', 'history_encoder', 'actor', 'meta_controller', 'option_encoder', 'option_critic', 'beta_net']}
    intrinsic_advantages, intrinsic_returns = None, None
    if model.options:
        intrinsic_targets = []
        for episode in episodes:
            for segment in episode[6]['segments']:
                a, b = segment['start'], segment['end']
                option_values = torch.stack(episode[6]['option_values'][a:b])
                intrinsic_targets.append(episode_targets(episode[6]['intrinsic'][a:b], option_values, args.gamma))
        intrinsic_returns = torch.cat([target[1] for target in intrinsic_targets])
        intrinsic_advantages = torch.cat([target[2] for target in intrinsic_targets])
        intrinsic_advantages = ((intrinsic_advantages - intrinsic_advantages.mean()) / intrinsic_advantages.std(unbiased=False).clamp_min(1e-8)).clamp(-1, 1)
        advantages = advantages + intrinsic_advantages
    for _ in range(args.epochs):
        for indices in torch.randperm(len(observations)).split(args.batch_size):
            batch = [observations[index] for index in indices.tolist()]
            indices = indices.to(old_values.device)
            distribution, predicted = model(batch)
            ratio = (distribution.log_prob(actions[indices].clone()) - old_probs[indices]).exp()
            policy_loss = -torch.minimum(ratio * advantages[indices], ratio.clamp(.8, 1.2) * advantages[indices]).mean()
            clipped = old_values[indices] + (predicted['total'] - old_values[indices]).clamp(-.2, .2)
            value_loss = torch.maximum((predicted['total'] - returns[indices]).square(), (clipped - returns[indices]).square()).mean()
            short_loss = nn.functional.mse_loss(predicted['short'], td[indices]) if model.hv else predicted['short'].sum() * 0
            long_loss = nn.functional.mse_loss(predicted['long'], mc[indices]) if model.hv else predicted['short'].sum() * 0
            option_value_loss = nn.functional.mse_loss(predicted['option'], intrinsic_returns[indices]) if model.options else value_loss * 0
            loss = policy_loss + .5 * value_loss + short_loss + .3 * long_loss + .5 * option_value_loss - .01 * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite gradient: ' + name)
                    for group in gradient_groups:
                        if name.startswith(group + '.'):
                            gradient_groups[group] += parameter.grad.norm().item() ** 2
            norm = nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError('Nonfinite parameters')
            stats.append([loss.item(), policy_loss.item(), value_loss.item(), short_loss.item(), long_loss.item(), float(norm)])
    metrics = dict(zip(['loss', 'policy_loss', 'value_loss', 'short_loss', 'long_loss', 'gradient_norm'], np.mean(stats, axis=0).tolist()))
    metrics['gradient_groups'] = {key: value ** .5 for key, value in gradient_groups.items()}
    if model.options:
        metrics.update(update_meta(model, optimizer, episodes, args))
        metrics['gradient_groups']['meta_controller'] = metrics['meta_gradient_norm']
    return metrics


def update_meta(model, optimizer, episodes, args):
    decisions = []
    for episode in episodes:
        segments = episode[6]['segments']
        rewards, durations, initial_values = [], [], []
        for segment in segments:
            a, b = segment['start'], segment['end']
            rewards.append(sum(args.gamma ** i * r for i, r in enumerate(episode[4][a:b])))
            durations.append(b - a)
            initial_values.append(episode[3][a])
        td, returns, advantages = smdp_targets(rewards, durations, torch.stack(initial_values), args.gamma)
        for segment, target, advantage in zip(segments, returns, advantages):
            decisions.append((segment, target, advantage))
    stats, norm_sum = [], 0.
    for _ in range(args.epochs):
        for selection in torch.randperm(len(decisions)).split(args.batch_size):
            losses = []
            for index in selection.tolist():
                segment, target, advantage = decisions[index]
                obs = segment['observation']
                distribution, candidates = model.option_distribution(obs)
                position = torch.tensor(candidates.index(segment['option']), device=target.device)
                ratio = (distribution.log_prob(position) - segment['log_prob']).exp()
                # Do not center away all learning signal when there is one SMDP decision.
                advantage = advantage.clamp(-10, 10)
                policy = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                _, value = model([obs])
                losses.append(policy + .5 * (value['total'][0] - target).square() - .01 * distribution.entropy())
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SMDP loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite SMDP gradient: ' + name)
            norm_sum += sum(p.grad.norm().item() ** 2 for p in model.meta_controller.parameters() if p.grad is not None)
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            stats.append(float(loss.detach()))
    return dict(meta_loss=float(np.mean(stats)), meta_gradient_norm=norm_sum ** .5,
                option_decisions=len(decisions), option_mean_duration=float(np.mean([x[0]['duration'] for x in decisions])))


def evaluate(args, training_env, model, manifest, checkpoint_path, checkpoint_steps):
    model.eval()
    split = args.eval_split or ('held_out' if manifest['held_out'] else 'train')
    formulas = args.eval_formula or manifest[split]
    if not formulas or len(set(formulas)) != len(formulas) or any(f not in manifest[split] for f in formulas):
        raise ValueError('Requested evaluation tasks must be unique members of the selected saved split')
    destination = args.output / f'evaluation-{split}-{args.eval_seed}-{args.evaluate}-{"det" if args.deterministic else "sample"}.json'
    if destination.exists():
        raise FileExistsError(destination)
    results = []
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for formula in formulas:
        env = ProductEpisode(make_env(args.env, FixedSampler.partial(formula), max_steps=args.max_episode_steps))
        try:
            for index in range(args.evaluate):
                episode = collect_episode(env, model, args.eval_seed + index, args.deterministic, args.option_max_steps)
                results.append(dict(formula=formula, split=split, episode=index, seed=args.eval_seed + index,
                                    finite=bool(env.terminate_on_acceptance), steps=len(episode[4]), reward=sum(episode[4]),
                                    discounted_return=discounted_returns(episode[4], args.gamma)[0].item(),
                                    success=bool(episode[5].get('success')), violation=bool(episode[5].get('violation')),
                                    accepting_visits=int(episode[5].get('num_accepting_visits', 0)),
                                    option_decisions=len(episode[6]['segments'])))
        finally:
            env.close()
    if not all(torch.equal(before[key], value) for key, value in model.state_dict().items()):
        raise RuntimeError('Evaluation mutated model weights')
    groups = {}
    for formula in formulas:
        rows = [row for row in results if row['formula'] == formula]
        finite = rows[0]['finite']
        groups[formula] = dict(finite=finite, success_rate=float(np.mean([row['success'] for row in rows])) if finite else None,
                               mean_accepting_visits=float(np.mean([row['accepting_visits'] for row in rows])),
                               mean_steps=float(np.mean([row['steps'] for row in rows])),
                               mean_reward=float(np.mean([row['reward'] for row in rows])),
                               mean_discounted_return=float(np.mean([row['discounted_return'] for row in rows])))
    atomic_json(destination, dict(episodes=results, per_formula=groups, split=split,
                                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                                  protocol=PROTOCOL_VERSION, task_manifest_sha256=manifest['sha256'],
                                  checkpoint_steps=checkpoint_steps, environment=args.env, configuration=args.configuration, training_seed=args.seed,
                                  gate=args.gate, horizon=args.max_episode_steps, discount=args.gamma,
                                  deterministic=args.deterministic, weights_unchanged=True,
                                  zero_shot=split == 'held_out'))
    print('Evaluation saved:', destination)
    training_env.close()


def main():
    args = arguments()
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    task_manifest = load_tasks(args.task_set, args.formula, args.env, args.max_episode_steps)
    env = ProductEpisode(make_env(args.env, training_sampler(task_manifest['train']), max_steps=args.max_episode_steps))
    initial = env.reset(seed=args.seed)
    unknown = set(initial['ldba'].propositions) - set(env.get_propositions())
    if unknown:
        raise ValueError('Formula has propositions outside environment: ' + repr(unknown))
    if args.configuration == 'baseline':
        from model.deepltl_reference import PaperDeepLTL
        from config import model_configs
        configuration_name = 'FlatWorld-v0' if args.env.startswith('FlatWorld') else args.env
        model = PaperDeepLTL(env, model_configs[configuration_name]).to(args.device)
    else:
        model = PaperHiLTL(initial['features'].shape, env.action_space, args.configuration, propositions=initial['ldba'].propositions, gate=args.gate).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['protocol'] = PROTOCOL_VERSION
    implementation_paths = ['src/model/paper_hiltl.py', 'src/model/paper_options.py', 'src/model/deepltl_reference.py', 'src/train/train_paper.py', 'src/train/paper_tasks.py']
    config['implementation_sha256'] = hashlib.sha256(b''.join(Path(path).read_bytes() for path in implementation_paths)).hexdigest()
    config['task_manifest'] = task_manifest
    config['propositions'] = list(initial['ldba'].propositions)
    graph = model.graph(initial['ldba'])
    config['automaton_states'] = list(graph.state_ids)
    config['automaton_edges'] = graph.edges.t().tolist()
    config['automaton_node_features'] = graph.nodes.tolist()
    config['automaton_edge_features'] = graph.edge_features.tolist()
    config['protocol_notes'] = ['Complete finite-horizon episodes per PPO batch; time limit is the defined horizon, with zero bootstrap.',
                              'Long head is a discounted MC return proxy, not a calibrated satisfaction probability.',
                              'Track enables history; product q is always observed for correct reward and task state.',
                              'SCC bit denotes a cyclic SCC; accepting nodes are targets of accepting transitions.',
                              'No artificial skip edges: shortest-hop progress decreases by at most one.',
                              'Guard GRU conditions edge messages on propositions; graph semantic feature dimensions remain 4/6.',
                              'full_all uses target-edge options, topology/timeout termination, intrinsic PPO and extrinsic SMDP PPO.',
                              'Task-set held_out formulas are never sampled for training. Structural overlap is rejected, without claiming language-equivalence checking.',
                              'Architecture and operational choices are explicit, not evidence for the historical paper performance numbers.']
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'status.pt'
    existing_run = status_path.exists() or (args.output / 'config.json').exists()
    if existing_run and not (args.resume or args.evaluate):
        raise FileExistsError('Existing run; use --resume or a new output directory')
    steps, updates = 0, 0
    if args.resume or args.evaluate:
        status = torch.load(status_path, map_location=args.device, weights_only=False)
        for key in ['protocol', 'env', 'formula', 'configuration', 'max_episode_steps', 'gamma', 'lr', 'seed', 'batch_size', 'epochs', 'rollout_steps', 'gate', 'option_max_steps', 'task_manifest', 'propositions', 'implementation_sha256']:
            if status['config'].get(key) != config[key]:
                raise ValueError('Checkpoint configuration mismatch: ' + key)
        model.load_state_dict(status['model'], strict=True)
        optimizer.load_state_dict(status['optimizer'])
        steps, updates = status['steps'], status['updates']
        random.setstate(status['random'])
        np.random.set_state(status['numpy'])
        torch.set_rng_state(status['torch'].cpu())
        if status['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in status['cuda']])
    if args.evaluate:
        evaluate(args, env, model, task_manifest, status_path, steps)
        return
    atomic_json(args.output / 'config.json', config)
    stopped = []
    def request_stop(signum, frame):
        stopped.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    model.train()
    with (args.output / 'metrics.jsonl').open('a') as metrics_file, (args.output / 'episodes.jsonl').open('a') as episode_file:
        while steps < args.steps and not stopped:
            episodes = []
            collected = 0
            while collected < args.rollout_steps and not stopped:
                episode = collect_episode(env, model, option_max_steps=args.option_max_steps)
                episodes.append(episode)
                collected += len(episode[4])
                episode_file.write(json.dumps(dict(update=updates + 1, formula=episode[6]['formula'], option_segments=[{k: v for k, v in segment.items() if k not in ['observation', 'log_prob']} for segment in episode[6]['segments']], steps=len(episode[4]), rewards=episode[4], success=bool(episode[5].get('success')), violation=bool(episode[5].get('violation')), accepting_visits=int(episode[5].get('num_accepting_visits', 0)))) + '\n')
                episode_file.flush()
            if not episodes:
                break
            metrics = update(model, optimizer, episodes, args)
            steps += collected
            updates += 1
            metrics.update(steps=steps, update=updates, elapsed_seconds=time.monotonic() - started, samples=collected, completed_episodes=len(episodes))
            metrics_file.write(json.dumps(metrics) + '\n')
            metrics_file.flush()
            checkpoint(status_path, model, optimizer, steps, updates, config)
            atomic_json(args.output / 'progress.json', dict(state='running', steps=steps, updates=updates, target_steps=args.steps, pid=os.getpid()))
            print(json.dumps(metrics), flush=True)
    atomic_json(args.output / 'progress.json', dict(state='interrupted' if stopped else 'completed', steps=steps, updates=updates, target_steps=args.steps, pid=os.getpid()))
    env.close()


if __name__ == '__main__':
    main()
