import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from ltl import FixedSampler
from ltl.automata import ltl2ldba
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from model.longshort_integration import (
    LongShortConfig,
    enhance_model_with_multiscale,
    load_checkpoint_compatible,
)
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'], default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='deepset')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow))')
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    gamma = 0.94 if args.env == 'LetterEnv-v0' else 0.998 if args.env == 'PointLtl2-v0' else 0.98
    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render, args.deterministic)


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic, device='cpu'):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed)
    training_status = model_store.load_training_status(map_location=device)
    model_store.load_vocab()
    if training_status.get('architecture', 'sequence_gnn') == 'deepltl':
        from model.deepltl_reference import DeepLTLSequenceModel
        model = DeepLTLSequenceModel(env, config)
    else:
        model = build_model(env, training_status, config)

    # 加载训练好的参数（新版 build_model 默认不再自动 load_state_dict）
    if isinstance(training_status, dict) and 'model_state' in training_status:
        state_dict = training_status['model_state']

        # 检测是否为长短期增强模型的权重（state_dict 以 base_model.* 前缀存储）
        has_base_prefix = any(k.startswith('base_model.') for k in state_dict)
        has_hv = any('hierarchical_value_net' in k for k in state_dict)
        has_multiscale = any('multiscale' in k for k in state_dict)
        has_automaton = any('automaton_tracker' in k for k in state_dict)

        if has_base_prefix:
            ldba = ltl2ldba(formula)
            config_ls = LongShortConfig(
                enable_hierarchical_value=has_hv,
                enable_multiscale_gnn=has_multiscale,
                enable_automaton_tracking=has_automaton,
                enable_options=False,
            )
            model = enhance_model_with_multiscale(model, ldba, config_ls)
            load_checkpoint_compatible(model, state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset(), {}
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return success_rate, average_steps
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits


if __name__ == '__main__':
    main()
