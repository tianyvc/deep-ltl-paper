import argparse
from typing import Optional

import gymnasium
import simple_parsing
import time
import datetime

import torch

import config
import preprocessing
import torch_ac

import utils
from model.model import build_model
from envs import make_env, get_env_attr
from model.longshort_integration import (
    enhance_model_with_multiscale, 
    LongShortConfig,
    load_checkpoint_compatible
)
from ltl.automata import ltl2ldba

from sequence.samplers import CurriculumSampler, curricula
from utils import torch_utils
from utils.logging.file_logger import FileLogger
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore
from config import *


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.text_logger = TextLogger(args)
        self.model_store = ModelStore.from_config(args)

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        training_status, resuming = self.get_training_status()
        envs = self.make_envs(training_status["curriculum_stage"])
        if resuming:
            self.model_store.load_vocab()
        else:
            assignments = envs[0].get_possible_assignments()
            print(f'Number of assignments: {len(assignments)}')
            preprocessing.init_vocab(assignments)
            self.model_store.save_vocab()
        
        # ---- 检测是否需要 Long/Short RL integration ----
        enable_longshort = any([
            self.args.enable_hierarchical_value,
            self.args.enable_multiscale_gnn,
            self.args.enable_automaton_tracking,
            self.args.enable_options
        ])
        
        # 获取 LDBA（如果需要）
        ldba = None
        if enable_longshort:
            formula = self.args.ltl_formula
            if formula is None:
                if hasattr(envs[0], 'formula'):
                    formula = envs[0].formula
                else:
                    try:
                        formula = get_env_attr(envs[0], 'formula')
                    except Exception:
                        formula = 'F a'  # 回退默认
            ldba = ltl2ldba(formula)
        
        # 构建基础模型（不加载 checkpoint）
        model = build_model(envs[0], {}, model_configs[self.args.model_config])
        
        # ---- 先增强模型结构（如果需要）----
        if enable_longshort:
            config_ls = LongShortConfig(
                enable_hierarchical_value=self.args.enable_hierarchical_value,
                enable_multiscale_gnn=self.args.enable_multiscale_gnn,
                enable_automaton_tracking=self.args.enable_automaton_tracking,
                enable_options=self.args.enable_options,
                lambda_short=self.args.lambda_short,
                lambda_mid=self.args.lambda_mid,
                lambda_long=self.args.lambda_long,
            )
            self.text_logger.important_info(
                f"[LongShort] Enhancing model: HV={config_ls.enable_hierarchical_value} MSGNN={config_ls.enable_multiscale_gnn} AT={config_ls.enable_automaton_tracking} OPT={config_ls.enable_options}"
            )
            model = enhance_model_with_multiscale(model, ldba, config_ls)
        
        # ---- 再加载 checkpoint（如果存在）----
        if "model_state" in training_status:
            try:
                load_checkpoint_compatible(model, training_status["model_state"], strict=False)
                self.text_logger.info("Successfully loaded model from checkpoint.")
            except Exception as e:
                self.text_logger.info(f"WARNING: Failed to load checkpoint: {e}. Starting from scratch.")
        
        model.to(self.args.experiment.device)
        algo = torch_ac.PPO(envs, model, self.args.experiment.device, self.args.ppo,
                            preprocess_obss=preprocessing.preprocess_obss, parallel=False)
        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.")
        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}')
        num_steps = training_status["num_steps"]
        num_updates = training_status["num_updates"]
        num_eval_steps = training_status["num_eval_steps"]
        while num_steps < self.args.experiment.num_steps:
            if self.args.save and (num_updates == 0 or num_eval_steps >= self.args.experiment.eval_interval):
                num_eval_steps = 0
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict()}
                self.model_store.save_eval_training_status(training_status)
            start = time.time()
            exps, logs = algo.collect_experiences()
            curriculum = get_env_attr(envs[0], 'sample_sequence').curriculum
            curriculum.update_task_success(logs['avg_goal_success'], verbose=True)
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)
            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_eval_steps += logs["num_steps"]
            num_updates += 1
            if num_updates % self.args.experiment.log_interval == 0 or curriculum.finished:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)
            if (
                    curriculum.finished or self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0) \
                    and self.args.save:
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict(),
                                   "optimizer_state": algo.optimizer.state_dict(),
                                   "curriculum_stage": curriculum.stage_index,
                                   "num_eval_steps": num_eval_steps,
                                   }
                self.model_store.save_training_status(training_status)
                self.model_store.save_ltl_net(algo.model.ltl_net.state_dict())
                self.text_logger.info("Saved training status")
            if curriculum.finished:
                self.text_logger.important_info("Finished curriculum.")
                break

    def make_envs(self, curriculum_stage: int) -> list[gymnasium.Env]:
        utils.set_seed(self.args.experiment.seed)
        envs = []
        for i in range(self.args.experiment.num_procs):
            curriculum = curricula[self.args.curriculum]
            curriculum.stage_index = curriculum_stage
            self.text_logger.important_info(f"Curriculum stage: {curriculum.stage_index}")
            sampler = CurriculumSampler.partial(curriculum)
            envs.append(make_env(self.args.experiment.env, sampler, sequence=True))
        # Set different seeds for each environment. The seed offset is used to ensure that the seeds do not overlap.
        seed_offset = 100 * self.args.experiment.seed
        seeds = [seed_offset + i for i in range(self.args.experiment.num_procs)]
        self.text_logger.info(f"Using seeds: {seeds}")
        for env, seed in zip(envs, seeds):
            env.reset(seed=seed)
        self.text_logger.info("Environments loaded.")
        return envs

    def get_training_status(self) -> tuple[dict, bool]:
        resuming = False
        try:
            training_status = self.model_store.load_training_status()
            self.text_logger.important_info("Resuming training from existing run.")
            resuming = True
        except FileNotFoundError:
            training_status = {"num_steps": 0, "num_updates": 0, "curriculum_stage": 0, "num_eval_steps": 0}
        return training_status, resuming

    def make_logger(self, log_csv: bool, log_wandb: bool, resuming: bool) -> MultiLogger:
        loggers = [self.text_logger]
        if log_csv:
            loggers.append(FileLogger(self.args, resuming=resuming))
        if log_wandb:
            loggers.append(WandbLogger(self.args, project_name='deep-ltl', resuming=resuming))
        return MultiLogger(*loggers)

    def augment_logs(self, logs: dict, update_time: float, num_steps: int) -> dict:
        sps = logs["num_steps"] / update_time
        remaining_duration = int((self.args.experiment.num_steps - num_steps) / sps)
        remaining_duration = 0 if remaining_duration < 0 else remaining_duration
        remaining_time = str(datetime.timedelta(seconds=remaining_duration))

        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"],
                                                                logs["num_steps_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"],
                                                                    logs["num_steps_per_episode"],
                                                                    self.args.ppo.discount)
        logs.update({
            "arps": average_reward_per_step,
            "adr": average_discounted_return,
            'sps': sps,
            'remaining': remaining_time,
            'num_steps': num_steps  # set num_steps to the total number of steps
        })
        return logs


# noinspection PyTypeChecker
def parse_arguments() -> argparse.Namespace:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(config.ExperimentConfig, dest="experiment")
    parser.add_arguments(config.PPOConfig, dest="ppo")
    parser.add_argument("--model_config", type=str, default="default", choices=model_configs.keys(),
                        required=True)
    parser.add_argument("--curriculum", type=str, choices=curricula.keys(), required=True)
    parser.add_argument("--log_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)
    # ---- Long/Short RL flags ----
    parser.add_argument('--enable_hierarchical_value', action='store_true', help='启用层次价值网络')
    parser.add_argument('--enable_multiscale_gnn', action='store_true', help='启用多尺度GNN')
    parser.add_argument('--enable_automaton_tracking', action='store_true', help='启用自动机状态追踪')
    parser.add_argument('--enable_options', action='store_true', help='启用Options子目标框架')
    parser.add_argument('--ltl_formula', type=str, default=None, help='覆盖环境内的LTL公式（可选）')
    parser.add_argument('--lambda_short', type=float, default=1.0, help='短期价值损失权重')
    parser.add_argument('--lambda_mid', type=float, default=0.5, help='中期价值损失权重')
    parser.add_argument('--lambda_long', type=float, default=0.3, help='长期价值损失权重')
    args = parser.parse_args()

    if args.experiment.device == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."
        args.experiment.device = 'cuda'

    return args


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    args = parse_arguments()
    trainer = Trainer(args)
    start_time = time.time()
    trainer.train(log_csv=args.log_csv, log_wandb=args.log_wandb)
    training_time = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Training took {training_time}.")


if __name__ == '__main__':
    main()
