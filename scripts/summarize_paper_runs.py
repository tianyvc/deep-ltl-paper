"""Aggregate evaluation JSON by independent training seeds, never by pooled episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Statistics require finite observations')
    mean = float(values.mean())
    if len(values) == 1:
        return dict(n=1, mean=mean, sample_std=None, ci95=None)
    std = float(values.std(ddof=1))
    half = float(t.ppf(.975, len(values) - 1) * std / np.sqrt(len(values)))
    return dict(n=len(values), mean=mean, sample_std=std, ci95=[mean-half, mean+half])


def aggregate(files, expected_seeds, episodes):
    groups = {}
    for filename in files:
        path = Path(filename)
        evaluation = json.loads(path.read_text())
        config = json.loads((path.parent / 'config.json').read_text())
        if evaluation['protocol'] != config['protocol'] or not evaluation['weights_unchanged']:
            raise ValueError('Invalid evaluation protocol or mutable weights: ' + str(path))
        if evaluation['task_manifest_sha256'] != config['task_manifest']['sha256']:
            raise ValueError('Task manifest mismatch: ' + str(path))
        seed = config['seed']
        shared = (config['env'], evaluation['split'], evaluation['deterministic'], config['gate'],
                  config['max_episode_steps'], config['gamma'], config['protocol'], config['task_manifest']['sha256'],
                  config['rollout_steps'], config['epochs'], config['batch_size'], config['lr'], config['option_max_steps'])
        for formula, values in evaluation['per_formula'].items():
            rows = [row for row in evaluation['episodes'] if row['formula'] == formula]
            if len(rows) != episodes or len({row['seed'] for row in rows}) != episodes:
                raise ValueError('Wrong number of evaluation episodes or duplicate seeds: ' + str(path))
            key = shared + (formula, config['configuration'])
            group = groups.setdefault(key, {})
            if seed in group:
                raise ValueError('Duplicate independent training seed: ' + str(path))
            group[seed] = dict(values=values, checkpoint=evaluation['checkpoint_sha256'], source=str(path),
                               steps=evaluation['checkpoint_steps'])
    output, index = [], {}
    for key, seeds in groups.items():
        if set(seeds) != set(expected_seeds):
            raise ValueError(f'Missing or unexpected independent training seeds for {key[-2:]}: {sorted(seeds)}')
        metrics = {}
        for metric in ['success_rate', 'mean_accepting_visits', 'mean_steps', 'mean_reward', 'mean_discounted_return']:
            values = [seeds[seed]['values'][metric] for seed in sorted(seeds)]
            metrics[metric] = None if all(value is None for value in values) else summarize(values)
        record = dict(environment=key[0], split=key[1], deterministic=key[2], gate=key[3], horizon=key[4],
                      gamma=key[5], protocol=key[6], task_manifest_sha256=key[7], formula=key[-2], configuration=key[-1],
                      independent_training_seeds=sorted(seeds), episodes_per_seed=episodes,
                      actual_step_counts={seed: value['steps'] for seed, value in seeds.items()}, metrics=metrics,
                      sources=[value['source'] for value in seeds.values()])
        output.append(record)
        index[key] = seeds
    comparisons = []
    for key, seeds in index.items():
        baseline_key = key[:-1] + ('baseline',)
        if key[-1] == 'baseline' or baseline_key not in index:
            continue
        baseline = index[baseline_key]
        differences = []
        for seed in sorted(seeds):
            ours = seeds[seed]['values']['success_rate']
            reference = baseline[seed]['values']['success_rate']
            if ours is not None and reference is not None:
                differences.append(ours-reference)
        if differences:
            comparisons.append(dict(environment=key[0], formula=key[-2], configuration=key[-1],
                                    metric='paired_success_rate_difference', statistics=summarize(differences)))
    return dict(groups=output, comparisons=comparisons,
                notes=['Training seeds are independent units; episodes are averaged within each training seed.',
                       'n=1 gives no variance/CI. Infinite specifications have no exact satisfaction SR.',
                       'Paired differences are absolute proportions, not relative percentages.',
                       'Actual step counts are reported; budget/horizon/protocol differences require review.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluations', nargs='+', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.episodes <= 0:
        parser.error('Seeds must be unique and episode count positive')
    result = aggregate(args.evaluations, args.seeds, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
