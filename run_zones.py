#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from utils import kill_all_wandb_processes


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True
    # --- Long/Short RL extension flags (all optional) ---
    enable_hierarchical_value: bool = False
    enable_multiscale_gnn: bool = False
    enable_automaton_tracking: bool = False
    enable_options: bool = False
    ltl_formula: str | None = None  # If not provided will attempt env formula
    lambda_short: float = 1.0
    lambda_mid: float = 0.5
    lambda_long: float = 0.3


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        command = [
            sys.executable, 'src/train/train_ppo.py',
            '--env', 'PointLtl2-v0',
            '--steps_per_process', '4096',
            '--batch_size', '2048',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '15_000_000',
            '--model_config', 'PointLtl2-v0',
            '--curriculum', 'PointLtl2-v0',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
            # Pass lambda weights explicitly (harmless if unused)
            '--lambda_short', str(args.lambda_short),
            '--lambda_mid', str(args.lambda_mid),
            '--lambda_long', str(args.lambda_long),
        ]
        # Optional formula
        if args.ltl_formula:
            command += ['--ltl_formula', args.ltl_formula]
        # Long/short flags
        if args.enable_hierarchical_value:
            command.append('--enable_hierarchical_value')
        if args.enable_multiscale_gnn:
            command.append('--enable_multiscale_gnn')
        if args.enable_automaton_tracking:
            command.append('--enable_automaton_tracking')
        if args.enable_options:
            command.append('--enable_options')
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 2 --device cpu --name demo --seed 1 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)
