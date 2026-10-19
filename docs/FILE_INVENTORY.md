# 文件与代码清单

生成时间：2026-10-16T04:13:07.175469-04:00

本清单从实际文件生成。类/函数和模块说明来自源码 AST/文档字符串，不代表该模块已执行或测试。第三方完整文件列表保存在清理前原始清单，正文按目录汇总。

## 顶层资产统计

| 目录/文件 | 文件数 | 总字节 |
|---|---:|---:|
| .dockerignore | 1 | 112 |
| .gitignore | 1 | 290 |
| AGENTS.md | 1 | 1413 |
| Dockerfile | 1 | 426 |
| LICENSE | 1 | 1075 |
| README.md | 1 | 6825 |
| docs | 27 | 777299 |
| eval_datasets | 114 | 475518 |
| paper | 48 | 9257853 |
| rabinizer4 | 851 | 12576815 |
| requirements.txt | 1 | 153 |
| results | 7 | 356210 |
| results_infinite | 3 | 5429 |
| run_flatworld.py | 1 | 1924 |
| run_flatworld_big.py | 1 | 2067 |
| run_letter.py | 1 | 1897 |
| run_zones.py | 1 | 3020 |
| scripts | 6 | 74679 |
| src | 852 | 105951256 |
| tests | 7 | 21033 |

## 项目自有 Python 源码逐文件索引

### docs/archive/config/enhanced_gnn_config.py

- 行数：65；字节：1677。
- 顶层导入：config, torch.nn

### docs/archive/config/legacy_config.py

- 行数：70；字节：1950。
- 顶层导入：dataclasses, typing, torch.nn
- L6 类 GNNConfig
- L18 类 NetworkConfig
- L23 类 ModelConfig
- L33 类 PPOConfig

### paper/scripts/generate_automaton_figure.py

- 行数：234；字节：10777。
- 模块说明：生成自动机流程图，展示HiLTL方法的优化点
- 顶层导入：pathlib, os, matplotlib.pyplot, matplotlib.patches, matplotlib.patches, numpy
- L21 函数 create_automaton_diagram：创建展示优化的自动机流程图

### paper/scripts/generate_figures.py

- 行数：651；字节：30182。
- 模块说明：Generate publication-quality figures for HiLTL paper Creates: SE-GNN architecture, long-short term fusion, automaton hierarchy, fusion weight dynamics Upgrades: - Unified theme (fonts, line widths, colors) for top-tier publication quality - Helper primitives for consistent boxes/arrows/labels - Carefully tuned layouts to eliminate overlaps and uneven spacing - Export both PNG (high DPI) and PDF (vector) assets
- 顶层导入：os, matplotlib.pyplot, matplotlib.patches, numpy, matplotlib.patheffects, seaborn, matplotlib
- L73 函数 _ensure_output_dir
- L77 函数 save_figure：Save both PNG (high DPI) and PDF (vector) with tight bounding box.
- L87 函数 draw_box：Draw a rounded box with centered text using axes coordinates. xy, w, h are in data coordinates of the axis.
- L100 函数 draw_arrow
- L107 函数 add_title：Simple panel title without background boxes.
- L113 函数 figure_1_se_gnn_architecture：SE-GNN architecture with clean, symmetric layout and no overlaps.
- L180 函数 figure_2_long_short_fusion：Long-short term fusion with symmetric routing and balanced spacing.
- L229 函数 figure_3_ldba_transitions：Four transition types in LDBA with consistent framing and spacing.
- L298 函数 figure_4_fusion_weight_dynamics：Fusion weight β(q) dynamics with clean shading and labels.
- L367 函数 figure_5_performance_improvement：Performance improvement visuals with refined labels and spacing.
- L426 函数 figure_6_architecture_overview：High-level architecture overview with balanced grid-like layout.
- L508 函数 figure_6_architecture_compact：Compact version: larger text, tighter layout, reduced whitespace.
- L564 函数 figure_6_architecture_flow：Expressive flow version matching paper narrative with four stages.

### paper/scripts/generate_limitations_figure.py

- 行数：235；字节：9390。
- 模块说明：Generate the 'Two Core Limitations' figure for the paper. This script creates a three-panel visualization showing: (a) Standard LDBA with semantic information (b) Semantic Flattening problem (c) Temporal Myopia problem
- 顶层导入：pathlib, os, matplotlib.pyplot, matplotlib.patches, matplotlib.patches, matplotlib.lines, matplotlib, numpy

### run_flatworld.py

- 行数：67；字节：1924。
- 顶层导入：os, subprocess, sys, dataclasses, simple_parsing, wandb, utils
- L13 类 Args
- L23 函数 main

### run_flatworld_big.py

- 行数：70；字节：2067。
- 顶层导入：os, subprocess, sys, dataclasses, simple_parsing, wandb, utils
- L13 类 Args
- L24 函数 main

### run_letter.py

- 行数：67；字节：1897。
- 顶层导入：os, subprocess, sys, dataclasses, simple_parsing, wandb, utils
- L13 类 Args
- L23 函数 main

### run_zones.py

- 行数：92；字节：3020。
- 顶层导入：os, subprocess, sys, dataclasses, simple_parsing, wandb, utils
- L13 类 Args
- L32 函数 main

### scripts/batch_evaluation.py

- 行数：276；字节：11005。
- 模块说明：自动化批量评估脚本 用于第一层（Tier 1）和第三层（Tier 3）数据收集 支持：环境选择 -> 模型选择 -> 公式选择 -> 自动生成命令 -> 结果导入 CSV
- 顶层导入：os, sys, csv, datetime, pathlib
- L62 类 BatchEvaluator
- L63 函数 BatchEvaluator.__init__
- L68 函数 BatchEvaluator.create_csv_if_needed：创建 CSV 文件（如果不存在）
- L77 函数 BatchEvaluator.save_result：保存单个结果到 CSV
- L101 函数 BatchEvaluator.print_command：生成并打印评估命令
- L117 函数 BatchEvaluator.run_interactive_session：交互式评估会话
- L220 函数 BatchEvaluator.print_summary：打印已保存结果的摘要
- L249 函数 main

### scripts/benchmarks/benchmark_gnn_optimized.py

- 行数：280；字节：10119。
- 模块说明：测试GNN优化后的性能
- 顶层导入：torch, time, sys, pathlib, model.gnn_net
- L12 函数 benchmark_gnn：基准测试GNN性能
- L110 函数 test_graph_building：测试图构建性能

### scripts/generate_eval_commands.py

- 行数：165；字节：5927。
- 模块说明：生成批量评估命令列表 支持快速生成第一层和第三层的所有评估命令
- 顶层导入：os, pathlib
- L67 函数 generate_commands：生成命令列表 tier: 'tier1' 或 'tier3' seed: 种子编号
- L112 函数 main

### scripts/generate_paper_figures.py

- 行数：449；字节：19637。
- 模块说明：生成论文级别的图表和可视化 Generates publication-quality figures for the LSTI paper. 使用方法: python generate_paper_figures.py --results_file results/PointLtl2-v0_results_full.csv --output_dir figures/
- 顶层导入：os, sys, argparse, pathlib, json, typing, numpy, pandas, matplotlib.pyplot, seaborn, scipy.stats, warnings
- L37 类 PaperFigureGenerator：论文级别图表生成器
- L62 函数 PaperFigureGenerator.__init__：初始化 Args: results_df: 评估结果DataFrame output_dir: 输出目录
- L73 函数 PaperFigureGenerator.generate_all_figures：生成所有论文图表
- L94 函数 PaperFigureGenerator.figure1_performance_comparison：Figure 1: 按难度级别的性能对比 显示: 4个难度级别 × 7个配置的成功率对比
- L162 函数 PaperFigureGenerator.figure2_learning_curves：Figure 2: 学习曲线对比（模拟） 由于没有实时训练日志，生成基于种子差异的学习曲线
- L199 函数 PaperFigureGenerator.figure3_complexity_analysis：Figure 3: 自动机复杂度 vs 性能 显示: 随LDBA状态数增加，不同配置的性能衰退曲线
- L263 函数 PaperFigureGenerator.figure4_ablation_heatmap：Figure 4: 消融热力图 显示: 7个配置 × 4个难度级别的性能矩阵
- L302 函数 PaperFigureGenerator.figure5_statistical_significance：Figure 5: 统计显著性对比 显示: baseline vs LSTI的显著性检验结果
- L361 函数 PaperFigureGenerator.generate_summary_tables：生成汇总表格文本
- L414 函数 main

### scripts/quick_analysis.py

- 行数：328；字节：12651。
- 模块说明：快速统计分析脚本 快速分析评估结果，生成论文所需的所有统计指标 使用方法: python quick_analysis.py --results results/PointLtl2-v0_results_full.csv
- 顶层导入：os, sys, argparse, pandas, numpy, scipy.stats, pathlib
- L19 类 QuickAnalyzer：快速统计分析器
- L22 函数 QuickAnalyzer.__init__：初始化 Args: results_file: 评估结果CSV文件
- L36 函数 QuickAnalyzer.analyze_all：执行完整分析
- L54 函数 QuickAnalyzer.analyze_main_performance：主性能分析
- L122 函数 QuickAnalyzer.analyze_ablation：消融分析
- L187 函数 QuickAnalyzer.analyze_significance：统计显著性分析
- L224 函数 QuickAnalyzer.analyze_complexity：复杂度影响分析
- L277 函数 generate_latex_tables：生成LaTeX格式的表格
- L296 函数 main

### scripts/quick_data_entry.py

- 行数：409；字节：15340。
- 模块说明：快速数据录入和可视化脚本 使用方法： # 步骤1：运行测试命令并记录结果 python scripts/quick_data_entry.py --record # 步骤2：生成图表 python scripts/quick_data_entry.py --visualize
- 顶层导入：argparse, csv, os, matplotlib.pyplot, numpy, pathlib
- L49 函数 record_data：交互式数据录入
- L228 函数 show_status：显示当前数据收集状态
- L263 函数 visualize_data：生成可视化图表
- L387 函数 main

### src/config/__init__.py

- 行数：15；字节：488。
- 顶层导入：.experiment_config, .ppo_config, .model_config

### src/config/experiment_config.py

- 行数：18；字节：810。
- 顶层导入：dataclasses
- L5 类 ExperimentConfig：Experiment configuration.

### src/config/model_config.py

- 行数：172；字节：4006。
- 顶层导入：abc, typing, dataclasses, torch, model.env, model.ltl.set_network
- L12 类 AbstractModelConfig
- L14 函数 AbstractModelConfig.build
- L19 类 StandardNetConfig
- L23 函数 StandardNetConfig.build
- L28 类 ConvNetConfig
- L33 函数 ConvNetConfig.build
- L38 类 SetNetConfig
- L42 函数 SetNetConfig.build
- L47 类 ActorConfig
- L54 类 GNNConfig
- L66 类 ModelConfig

### src/config/ppo_config.py

- 行数：17；字节：702。
- 顶层导入：dataclasses
- L5 类 PPOConfig：Configuration for PPO.

### src/debug_zones.py

- 行数：19；字节：558。
- 顶层导入：envs, ltl.samplers

### src/envs/__init__.py

- 行数：1；字节：46。
- 顶层导入：.env_utils

### src/envs/env_utils.py

- 行数：80；字节：2371。
- 顶层导入：typing, gymnasium, gymnasium.wrappers, envs.remove_trunc_wrapper
- L9 函数 get_env_attr
- L18 函数 make_env
- L54 函数 is_safety_gym_env
- L58 函数 make_safety_gym_env
- L69 函数 make_letter_env
- L76 函数 make_flatworld_env

### src/envs/flatworld/__init__.py

- 行数：21；字节：467。
- 顶层导入：.flatworld, gymnasium.envs.registration

### src/envs/flatworld/flatworld.py

- 行数：165；字节：6418。
- 顶层导入：dataclasses, typing, numpy, gymnasium, gymnasium.spaces, matplotlib.pyplot, seaborn, gymnasium.core, ltl.logic
- L17 类 Circle
- L23 类 FlatWorld
- L36 函数 FlatWorld.__init__
- L49 函数 FlatWorld.get_active_propositions
- L56 函数 FlatWorld.reset
- L69 函数 FlatWorld.step
- L101 函数 FlatWorld.get_propositions
- L104 函数 FlatWorld.get_possible_assignments
- L123 函数 FlatWorld.render
- L147 函数 hide_ticks

### src/envs/flatworld/flatworld_big.py

- 行数：166；字节：6106。
- 顶层导入：math, dataclasses, typing, numpy, gymnasium, gymnasium.spaces, matplotlib.pyplot, seaborn, gymnasium.core, ltl.logic
- L18 类 Circle
- L24 类 FlatWorldBig
- L28 函数 FlatWorldBig.__init__
- L43 函数 FlatWorldBig.init_circles
- L57 函数 FlatWorldBig.get_active_propositions
- L65 函数 FlatWorldBig.reset
- L78 函数 FlatWorldBig.step
- L110 函数 FlatWorldBig.get_propositions
- L113 函数 FlatWorldBig.get_possible_assignments
- L120 函数 FlatWorldBig.render
- L144 函数 hide_ticks

### src/envs/ldba_wrapper.py

- 行数：86；字节：3675。
- 顶层导入：functools, typing, gymnasium, gymnasium.core, envs, ltl.automata
- L11 类 LDBAWrapper：Wrapper that keeps track of LTL goal satisfaction using an LDBA, which is added to the observation space.
- L16 函数 LDBAWrapper.__init__
- L29 函数 LDBAWrapper.step
- L57 函数 LDBAWrapper.reset
- L70 函数 LDBAWrapper.complete_observation
- L76 函数 LDBAWrapper.construct_ldba

### src/envs/letter_world/__init__.py

- 行数：52；字节：1157。
- 顶层导入：gymnasium.envs.registration

### src/envs/letter_world/letter_env.py

- 行数：364；字节：14273。
- 顶层导入：pickle, random, typing, numpy, gymnasium, pygame, gymnasium, gymnasium.core, gymnasium.wrappers, ltl.logic
- L17 类 LetterEnv：This environment is a grid with randomly located letters on it. We ensure that there is a clean path to any of the letters (a path that includes no passing by any letter). Note that stepping outside the map causes the ag
- L25 函数 LetterEnv.__init__
- L55 函数 LetterEnv.step
- L66 函数 LetterEnv.wait_for_input
- L71 函数 LetterEnv._get_observation
- L88 函数 LetterEnv._get_centric_map
- L99 函数 LetterEnv.reset
- L125 函数 LetterEnv.print
- L143 函数 LetterEnv.print_features
- L156 函数 LetterEnv.render
- L160 函数 LetterEnv.render_path
- L163 函数 LetterEnv._render_frame
- L166 函数 LetterEnv.close
- L170 函数 LetterEnv.get_active_propositions
- L176 函数 LetterEnv.get_propositions
- L179 函数 LetterEnv.get_possible_assignments
- L182 函数 LetterEnv.save_world_info
- L186 函数 LetterEnv.load_world_info
- L192 函数 _is_valid_map
- L207 类 LetterEnvRenderer
- L208 函数 LetterEnvRenderer.__init__
- L228 函数 LetterEnvRenderer.render
- L232 函数 LetterEnvRenderer.draw_canvas
- L254 函数 LetterEnvRenderer.update
- L265 函数 LetterEnvRenderer.render_path
- L279 函数 LetterEnvRenderer.wait_for_input
- L286 函数 LetterEnvRenderer.draw_arrow
- L328 函数 LetterEnvRenderer.close
- L334 函数 main

### src/envs/ltl_wrapper.py

- 行数：39；字节：1569。
- 顶层导入：typing, gymnasium, gymnasium, gymnasium.core
- L8 类 LTLWrapper
- L11 函数 LTLWrapper.__init__
- L21 函数 LTLWrapper.set_goal
- L25 函数 LTLWrapper.step
- L30 函数 LTLWrapper.reset

### src/envs/remove_trunc_wrapper.py

- 行数：20；字节：717。
- 顶层导入：typing, gymnasium, gymnasium.core
- L7 类 RemoveTruncWrapper
- L8 函数 RemoveTruncWrapper.__init__
- L11 函数 RemoveTruncWrapper.step
- L16 函数 RemoveTruncWrapper.reset

### src/envs/seq_wrapper.py

- 行数：78；字节：3101。
- 顶层导入：typing, gymnasium, gymnasium, gymnasium.core, ltl.automata, ltl.logic
- L11 类 SequenceWrapper：Wrapper that adds a reach-avoid sequence of propositions to the observation space.
- L16 函数 SequenceWrapper.__init__
- L29 函数 SequenceWrapper.step
- L57 函数 SequenceWrapper.apply_epsilon_action
- L62 函数 SequenceWrapper.reset
- L72 函数 SequenceWrapper.complete_observation

### src/envs/zones/__init__.py

- 行数：1；字节：48。
- 顶层导入：.safety_gym_wrapper

### src/envs/zones/safety_gym_wrapper.py

- 行数：56；字节：2185。
- 顶层导入：typing, gymnasium, numpy, gymnasium, gymnasium.core, gymnasium.spaces, ltl.logic
- L12 类 SafetyGymWrapper：A wrapper from safety gymnasium LTL environments to the gymnasium API.
- L17 函数 SafetyGymWrapper.__init__
- L34 函数 SafetyGymWrapper.step
- L44 函数 SafetyGymWrapper.reset
- L52 函数 SafetyGymWrapper.get_propositions
- L55 函数 SafetyGymWrapper.get_possible_assignments

### src/evaluation/__init__.py

- 行数：0；字节：0。

### src/evaluation/create_eval_dataset.py

- 行数：43；字节：1259。
- 顶层导入：os, random, numpy, torch, tqdm, envs, ltl, ltl.samplers.reach_sampler, ltl.samplers.super_sampler
- L20 函数 main

### src/evaluation/create_flatworld_eval_dataset.py

- 行数：44；字节：1385。
- 顶层导入：os, random, numpy, torch, tqdm, envs, ltl, ltl.samplers.flatworld_avoid_sampler, ltl.samplers.flatworld_reach_sampler, ltl.samplers.reach_sampler, ltl.samplers.super_sampler
- L21 函数 main

### src/evaluation/draw_flatworld_trajectories.py

- 行数：86；字节：2451。
- 顶层导入：random, numpy, torch, matplotlib, tqdm, envs, envs.flatworld, ltl, model.model, model.agent, config, sequence.search, utils.model_store, visualize.zones, seaborn

### src/evaluation/draw_zone_trajectories.py

- 行数：66；字节：1818。
- 顶层导入：random, numpy, torch, matplotlib, tqdm, envs, ltl, model.model, model.agent, config, sequence.search, utils.model_store, visualize.zones

### src/evaluation/eval_flatworld_big.py

- 行数：50；字节：1634。
- 顶层导入：argparse, os, pandas, tqdm, config, envs, evaluation.simulate, multiprocessing, ltl, model.model, utils.model_store
- L17 函数 main
- L45 函数 eval_task

### src/evaluation/eval_over_time.py

- 行数：126；字节：4058。
- 顶层导入：os, random, sys, time, numpy, pandas, torch, tqdm, envs, evaluation.eval_sync_env, ltl, model.model, config, model.parallel_agent, sequence.search, utils.model_store, multiprocessing
- L23 函数 set_env
- L49 函数 aux
- L56 函数 main
- L82 函数 eval_model

### src/evaluation/eval_sync_env.py

- 行数：53；字节：2017。
- 顶层导入：gymnasium
- L4 类 EvalSyncEnv：A sequential execution of multiple environments, with predefined zones and tasks.
- L7 函数 EvalSyncEnv.__init__
- L20 函数 EvalSyncEnv.reset
- L32 函数 EvalSyncEnv.step
- L52 函数 EvalSyncEnv.render

### src/evaluation/eval_test_tasks_finite.py

- 行数：78；字节：2929。
- 顶层导入：argparse, os, time, pandas, colorama, tqdm, evaluation.simulate, multiprocessing
- L39 函数 main
- L65 函数 eval_task

### src/evaluation/eval_test_tasks_infinite.py

- 行数：67；字节：2209。
- 顶层导入：os, pandas, tqdm, evaluation.simulate, argparse, multiprocessing
- L29 函数 main
- L57 函数 eval_task

### src/evaluation/plot.py

- 行数：54；字节：1749。
- 顶层导入：os, numpy, pandas, seaborn, matplotlib
- L11 函数 main
- L22 函数 process_logs
- L44 函数 smooth：Computes the moving average over the given row of data. Returns an array of the same shape as the original row.

### src/evaluation/plot_training_curves.py

- 行数：70；字节：2789。
- 顶层导入：os, numpy, pandas, seaborn, matplotlib
- L11 函数 main
- L38 函数 process_eval_results
- L60 函数 smooth：Computes the moving average over the given row of data. Returns an array of the same shape as the original row.

### src/evaluation/simulate.py

- 行数：142；字节：5580。
- 顶层导入：random, numpy, torch, tqdm, envs, ltl, ltl.automata, model.model, model.agent, config, sequence.search, utils.model_store, model.longshort_integration, argparse
- L23 函数 main
- L38 函数 simulate

### src/evaluation/simulate_letter.py

- 行数：131；字节：4499。
- 顶层导入：random, numpy, torch, tqdm, envs, ltl, ltl.samplers.avoid_multiple_sampler, ltl.samplers.avoid_sampler, ltl.samplers.fixed_sampler, model.model, model.agent, config, sequence.search, utils.model_store

### src/ltl/__init__.py

- 行数：1；字节：24。
- 顶层导入：.samplers

### src/ltl/automata/__init__.py

- 行数：21；字节：871。
- 顶层导入：contextlib, utils, .rabinizer, .ldba, .ldba_sequence
- L10 函数 ltl2ldba：Converts an LTL formula to an LDBA using the rabinizer tool.

### src/ltl/automata/ldba.py

- 行数：330；字节：14584。
- 顶层导入：functools, collections, dataclasses, typing, ltl.logic, utils
- L10 类 LDBA
- L11 函数 LDBA.__init__
- L26 函数 LDBA.states
- L29 函数 LDBA.add_state
- L42 函数 LDBA.get_next_state：Returns the next state and whether the taken transition is accepting, given the current state and the propositions that are true.
- L58 函数 LDBA.contains_state
- L61 函数 LDBA.is_finite_specification：Checks if the LDBA represents a finite specification. Note: this is not an exhaustive check, but a sufficient heuristic for LDBAs constructed by rabinizer.
- L74 函数 LDBA.add_transition
- L90 函数 LDBA.check_valid：Checks that the LDBA satisfies the following conditions: - It has a deterministic first component - It has a deterministic second component - All transitions from the first to the second component are epsilon transitions
- L147 函数 LDBA.check_deterministic_transitions：Checks that the transitions from a state are deterministic.
- L154 函数 LDBA.complete_sink_state：Adds a sink state and transitions to the LDBA if any transitions are missing.
- L182 函数 LDBA.has_sink_state
- L185 函数 LDBA.valid_assignments_to_label
- L193 函数 LDBA.prune：Prunes transitions that involve impossible assignments. Impossible assignments may be derived from knowledge of the underlying MDP.
- L213 函数 LDBA.compute_sccs：Computes the strongly connected components of the LDBA using Tarjan's algorithm.
- L259 函数 LDBA.parity_automaton
- L280 类 LDBATransition
- L288 函数 LDBATransition.is_epsilon
- L292 函数 LDBATransition.valid_assignments
- L301 函数 LDBATransition.positive_label
- L314 函数 LDBATransition.__eq__
- L322 函数 LDBATransition.__hash__
- L327 类 SCC

### src/ltl/automata/ldba_sequence.py

- 行数：52；字节：2037。
- 顶层导入：ltl.logic
- L6 类 LDBASequence
- L9 函数 LDBASequence.__init__：:param reach_avoid: A list of pairs of reach and avoid assignments. :param repeat_last: Number of times to how often the last reach-avoid pair should be repeated. :param state_info: Optional list of state metadata (accep
- L24 函数 LDBASequence.__hash__
- L27 函数 LDBASequence.__eq__
- L32 函数 LDBASequence.__len__
- L35 函数 LDBASequence.__iter__
- L38 函数 LDBASequence.__getitem__
- L51 函数 LDBASequence.__repr__

### src/ltl/automata/rabinizer.py

- 行数：21；字节：673。
- 顶层导入：subprocess
- L6 函数 run_rabinizer：Convert an LTL formula to a LDBA in the HOA format.

### src/ltl/hoa/__init__.py

- 行数：2；字节：68。
- 顶层导入：.hoa_parser, .hoa_writer

### src/ltl/hoa/hoa_parser.py

- 行数：138；字节：5586。
- 顶层导入：typing, re, ltl.automata
- L7 类 HOAParser：A parser for LDBAs given in the HOA format. Handles epsilon transitions as output by rabinizer.
- L12 函数 HOAParser.__init__
- L21 函数 HOAParser.parse_hoa
- L30 函数 HOAParser.parse_propositions
- L39 函数 HOAParser.find_and_parse_ap_line
- L46 函数 HOAParser.parse_ap_line
- L54 函数 HOAParser.parse_header
- L73 函数 HOAParser.parse_header_line
- L80 函数 HOAParser.parse_body
- L86 函数 HOAParser.parse_state
- L95 函数 HOAParser.parse_transition
- L107 函数 HOAParser.parse_label
- L116 函数 HOAParser.replace_numeric_propositions
- L121 函数 HOAParser.peek
- L126 函数 HOAParser.consume
- L131 函数 HOAParser.expect_line
- L136 函数 HOAParser.expect

### src/ltl/hoa/hoa_writer.py

- 行数：51；字节：1848。
- 顶层导入：re, ltl.automata
- L6 类 HOAWriter
- L7 函数 HOAWriter.__init__
- L11 函数 HOAWriter.get_hoa
- L17 函数 HOAWriter.write_header
- L28 函数 HOAWriter.write_body
- L39 函数 HOAWriter.label_to_index_label
- L45 函数 HOAWriter.proposition_to_index
- L50 函数 HOAWriter.write_line

### src/ltl/logic/__init__.py

- 行数：1；字节：53。
- 顶层导入：.assignment

### src/ltl/logic/assignment.py

- 行数：123；字节：4018。
- 顶层导入：functools, typing, ltl.logic.boolean_parser
- L7 类 Assignment：An assignment of truth values to propositions.
- L10 函数 Assignment.__init__
- L16 函数 Assignment.all_possible_assignments：Returns all possible assignments for a given set of propositions. Guarantees a deterministic order.
- L28 函数 Assignment.more_than_one_true_proposition
- L36 函数 Assignment.zero_or_one_propositions
- L45 函数 Assignment.single_proposition
- L49 函数 Assignment.zero_propositions
- L53 函数 Assignment.where
- L56 函数 Assignment.satisfies
- L62 函数 Assignment.get_true_propositions
- L65 函数 Assignment.to_frozen
- L68 函数 Assignment.__str__
- L71 函数 Assignment.__repr__
- L74 函数 Assignment.__setitem__
- L77 函数 Assignment.__delitem__
- L80 函数 Assignment.__getitem__
- L83 函数 Assignment.__len__
- L86 函数 Assignment.__iter__
- L89 函数 Assignment.__or__
- L92 函数 Assignment.__eq__
- L96 类 FrozenAssignment：An immutable assignment of truth values to propositions. Used for hashing.
- L99 函数 FrozenAssignment.__init__
- L102 函数 FrozenAssignment.to_label
- L106 函数 FrozenAssignment.get_true_propositions
- L109 函数 FrozenAssignment.__eq__
- L112 函数 FrozenAssignment.__hash__
- L115 函数 FrozenAssignment.__str__
- L118 函数 FrozenAssignment.__repr__
- L122 函数 FrozenAssignment.__iter__

### src/ltl/logic/boolean_lexer.py

- 行数：83；字节：2411。
- 顶层导入：dataclasses, enum, typing
- L6 类 TokenType
- L17 类 Token
- L21 函数 Token.__repr__
- L25 类 Lexer
- L26 函数 Lexer.__init__
- L31 函数 Lexer.lex
- L65 函数 Lexer.tokenize_variable
- L72 函数 Lexer.peek

### src/ltl/logic/boolean_parser.py

- 行数：143；字节：4202。
- 顶层导入：functools, abc, typing, ltl.logic.boolean_lexer
- L7 类 Parser
- L8 函数 Parser.__init__
- L14 函数 Parser.parse
- L20 函数 Parser.parse_expression
- L27 函数 Parser.parse_implication
- L34 函数 Parser.parse_or
- L41 函数 Parser.parse_and
- L48 函数 Parser.parse_primary
- L59 函数 Parser.parse_variable
- L67 函数 Parser.match
- L73 函数 Parser.next_token
- L81 类 Node
- L83 函数 Node.eval
- L87 类 AndNode
- L88 函数 AndNode.__init__
- L92 函数 AndNode.eval
- L96 类 OrNode
- L97 函数 OrNode.__init__
- L101 函数 OrNode.eval
- L105 类 NotNode
- L106 函数 NotNode.__init__
- L109 函数 NotNode.eval
- L113 类 VarNode
- L114 函数 VarNode.__init__
- L117 函数 VarNode.eval
- L121 类 ImplicationNode
- L122 函数 ImplicationNode.__init__
- L126 函数 ImplicationNode.eval
- L131 函数 parse

### src/ltl/samplers/__init__.py

- 行数：3；字节：99。
- 顶层导入：.partially_ordered_sampler, .avoid_sampler, .fixed_sampler

### src/ltl/samplers/avoid_multiple_sampler.py

- 行数：36；字节：1253。
- 顶层导入：random, pprint
- L5 类 AvoidMultipleSampler
- L8 函数 AvoidMultipleSampler.partial
- L11 函数 AvoidMultipleSampler.__init__
- L16 函数 AvoidMultipleSampler.__call__

### src/ltl/samplers/avoid_sampler.py

- 行数：57；字节：2246。
- 顶层导入：copy, random, pprint
- L6 类 AvoidSampler
- L9 函数 AvoidSampler.partial
- L12 函数 AvoidSampler.__init__：:param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
- L21 函数 AvoidSampler.__call__

### src/ltl/samplers/fixed_sampler.py

- 行数：15；字节：313。
- 顶层导入：random, pprint
- L5 类 FixedSampler
- L8 函数 FixedSampler.partial
- L11 函数 FixedSampler.__init__
- L14 函数 FixedSampler.__call__

### src/ltl/samplers/flatworld_avoid_sampler.py

- 行数：66；字节：2752。
- 顶层导入：copy, random, pprint
- L6 类 FlatWorldAvoidSampler
- L9 函数 FlatWorldAvoidSampler.partial
- L12 函数 FlatWorldAvoidSampler.__init__：:param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
- L20 函数 FlatWorldAvoidSampler.__call__

### src/ltl/samplers/flatworld_reach_sampler.py

- 行数：56；字节：2178。
- 顶层导入：random, pprint
- L5 类 FlatWorldReachSampler
- L8 函数 FlatWorldReachSampler.partial
- L11 函数 FlatWorldReachSampler.__init__：:param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
- L19 函数 FlatWorldReachSampler.__call__
- L36 函数 FlatWorldReachSampler.prop_to_str

### src/ltl/samplers/partially_ordered_sampler.py

- 行数：58；字节：2383。
- 顶层导入：random
- L3 类 PartiallyOrderedSampler
- L6 函数 PartiallyOrderedSampler.partial
- L9 函数 PartiallyOrderedSampler.__init__
- L17 函数 PartiallyOrderedSampler.__call__
- L25 函数 PartiallyOrderedSampler.sample_as_list
- L32 函数 PartiallyOrderedSampler.sample_sequence
- L41 函数 PartiallyOrderedSampler.sequence_to_formula

### src/ltl/samplers/reach_sampler.py

- 行数：43；字节：1487。
- 顶层导入：random, pprint
- L5 类 ReachSampler
- L8 函数 ReachSampler.partial
- L11 函数 ReachSampler.__init__：:param ltl2action_format: if true, also returns the sampled LTL formula in the format of ltl2action
- L19 函数 ReachSampler.__call__

### src/ltl/samplers/reach_stay_sampler.py

- 行数：18；字节：356。
- 顶层导入：random, pprint
- L5 类 ReachStaySampler
- L8 函数 ReachStaySampler.partial
- L11 函数 ReachStaySampler.__init__
- L14 函数 ReachStaySampler.__call__

### src/ltl/samplers/super_sampler.py

- 行数：26；字节：749。
- 顶层导入：random, pprint, ltl, ltl.samplers.reach_sampler
- L8 类 SuperSampler
- L11 函数 SuperSampler.partial
- L14 函数 SuperSampler.__init__
- L17 函数 SuperSampler.__call__

### src/model/__init__.py

- 行数：0；字节：0。

### src/model/agent.py

- 行数：39；字节：1508。
- 顶层导入：numpy, torch, preprocessing, model.model, sequence.search
- L9 类 Agent
- L10 函数 Agent.__init__
- L17 函数 Agent.reset
- L20 函数 Agent.get_action
- L29 函数 Agent.forward

### src/model/discrete_epsilon_distribution.py

- 行数：53；字节：1815。
- 顶层导入：torch, torch.distributions, ltl.automata
- L7 类 DiscreteEpsilonDistribution：A class to represent categorical distribution with an extra category for taking an epsilon transition.
- L13 函数 DiscreteEpsilonDistribution.__init__：:param dist: Categorical distribution with n+1 categories, where n is the number of actions.
- L22 函数 DiscreteEpsilonDistribution.set_epsilon_mask
- L25 函数 DiscreteEpsilonDistribution.sample
- L32 函数 DiscreteEpsilonDistribution.log_prob
- L39 函数 DiscreteEpsilonDistribution.entropy
- L44 函数 DiscreteEpsilonDistribution.mode
- L50 函数 DiscreteEpsilonDistribution.create_normalized_distribution

### src/model/env/__init__.py

- 行数：2；字节：82。
- 顶层导入：.conv_env_net, .standard_env_net

### src/model/env/conv_env_net.py

- 行数：29；字节：881。
- 顶层导入：typing, torch, torch
- L7 类 ConvEnvNet
- L8 函数 ConvEnvNet.__init__
- L27 函数 ConvEnvNet.forward

### src/model/env/standard_env_net.py

- 行数：16；字节：488。
- 顶层导入：typing, torch, torch, utils
- L9 类 StandardEnvNet
- L10 函数 StandardEnvNet.__init__
- L15 函数 StandardEnvNet.forward

### src/model/gnn_net.py

- 行数：231；字节：9514。
- 顶层导入：torch, torch.nn, torch_geometric.nn, torch_geometric.data
- L18 类 SimpleGlobalAttention：自定义全局注意力池化，避免 PyG 版本差异导致的接口不兼容。 公式与 GlobalAttention 等价：score = gate_nn(x); 输出为 sum(score * x) 按 batch 聚合。
- L22 函数 SimpleGlobalAttention.__init__
- L26 函数 SimpleGlobalAttention.forward
- L32 类 GNNNet
- L33 函数 GNNNet.__init__：增强的GNN网络，支持： 1. 边特征（类型：链式、跳跃、epsilon、接受转移等） 2. 节点状态特征（接受状态标记、SCC信息等） Args: input_dim: 输入节点特征维度 hidden_dim: 隐藏层维度 num_layers: GNN层数 dropout: Dropout率 heads: 注意力头数 edge_dim: 边特征维度（原始维度） use_edge_attr: 是否使用边特征 node_state_di
- L106 函数 GNNNet.forward：前向传播 Args: x: 节点特征 (num_nodes, input_dim) edge_index: 边的连接关系 (2, num_edges) edge_attr: 可选的边特征 (num_edges, edge_dim) node_state_feat: 可选的节点状态特征 (num_nodes, node_state_dim) 包含接受状态标记、SCC信息等 batch: 批处理向量，指示节点属于哪个图 (num_nodes
- L149 函数 process_reach_avoid_to_graph：将reach/avoid序列对转换为图结构 Args: reach_data: reach集合的特征 (batch, max_len, feat_dim) 或 (batch, max_len) avoid_data: avoid集合的特征 (batch, max_len, feat_dim) 或 (batch, max_len) reach_lens: 每个序列的实际长度 (batch,) Returns: Data: PyTorch 

### src/model/hierarchical_gnn.py

- 行数：133；字节：4833。
- 顶层导入：torch, torch.nn, torch_geometric.nn, torch_geometric.data
- L6 类 HierarchicalGNNNet：分层GNN网络：同时在LDBA状态图和序列图上进行处理
- L9 函数 HierarchicalGNNNet.__init__
- L81 函数 HierarchicalGNNNet.forward：前向传播 Args: seq_data: (x, edge_index, edge_attr, batch) - 序列图数据 ldba_data: (x, edge_index, edge_attr, batch) - LDBA图数据

### src/model/hierarchical_options.py

- 行数：426；字节：13575。
- 模块说明：Options Framework：基于自动机状态的层次化强化学习 核心思想： - 每个自动机状态 q 对应一个 Option（技能） - Meta-Controller 选择目标状态 - Option 策略执行直到到达目标或终止
- 顶层导入：torch, torch.nn, typing, dataclasses, enum
- L17 类 OptionStatus：Option 执行状态
- L26 类 OptionTransition：Option 执行的转移
- L39 类 AutomatonOption：单个 Option：对应一个自动机目标状态 - Initiation: 从哪些状态可以开始 - Policy: 如何行动 - Termination: 何时停止
- L47 函数 AutomatonOption.__init__
- L67 函数 AutomatonOption.can_initiate：判断是否可以从当前自动机状态启动该 Option
- L72 函数 AutomatonOption.select_action：Option 策略选择动作
- L95 函数 AutomatonOption.compute_intrinsic_reward：计算内在奖励（引导 Option 向目标前进） 奖励结构： - 到达目标：+1.0 - 接近目标：+0.1 * (距离减少量) - 远离目标：-0.1 * (距离增加量)
- L118 函数 AutomatonOption.check_termination：检查终止条件
- L135 函数 AutomatonOption._is_reachable：BFS 检查可达性
- L156 函数 AutomatonOption._distance_to_target：到目标状态的最短距离（BFS）
- L177 函数 AutomatonOption._onehot：One-hot 编码
- L184 类 MetaController：高层元控制器：选择要执行的 Option Q(q, o) = 期望从状态 q 执行 option o 的价值
- L190 函数 MetaController.__init__
- L210 函数 MetaController.select_option：选择 Option（ε-greedy） Args: current_q: 当前自动机状态 available_options: 可用 Option ID 列表 formula_emb: 公式嵌入 epsilon: 探索率 Returns: selected_option_id
- L246 函数 MetaController.compute_q_value：计算 Q(q, o)
- L258 函数 MetaController._onehot
- L264 类 HierarchicalController：层次化控制器：协调 Meta-Controller 和 Options 训练流程： 1. Meta-Controller 选择 Option 2. Option 执行直到终止 3. 用内在奖励更新 Option 策略 4. 用环境奖励更新 Meta-Controller
- L274 函数 HierarchicalController.__init__
- L295 函数 HierarchicalController.train_step：训练一个 episode 的 transitions Args: transitions: OptionTransition 列表 formula_emb: 公式嵌入
- L313 函数 HierarchicalController._update_options：用内在奖励更新 Option 策略
- L356 函数 HierarchicalController._update_meta_controller：用环境奖励更新 Meta-Controller
- L409 函数 HierarchicalController._get_available_options：获取从状态 q 可用的 Options

### src/model/hierarchical_value.py

- 行数：368；字节：12127。
- 模块说明：层次化价值函数：结合自动机状态追踪与短期环境奖励 核心思想： - 短期价值：估计即时环境回报（V_short） - 长期价值：估计距离满足公式的进度（V_long） - 动态融合：根据自动机阶段调整权重
- 顶层导入：torch, torch.nn, typing, dataclasses
- L17 类 AutomatonState：自动机状态的增强表示
- L26 类 AutomatonStateTracker：实时追踪智能体在 LDBA 中的位置
- L30 函数 AutomatonStateTracker.__init__
- L38 函数 AutomatonStateTracker.reset：重置为初始状态
- L47 函数 AutomatonStateTracker.step：根据环境命题更新自动机状态 Args: propositions: [batch_size, n_props] 布尔张量 Returns: next_states: [batch_size] 新状态 ID is_accepting: [batch_size] 是否在接受状态
- L79 函数 AutomatonStateTracker.get_state_features：提取当前自动机状态的特征 Returns: features: [batch_size, 4] [is_accepting, in_scc, dist_norm, progress]
- L109 函数 AutomatonStateTracker._compute_distances：BFS 计算每个状态到接受状态的最短距离
- L149 函数 AutomatonStateTracker._tensor_to_prop_set：将布尔张量转为命题集合
- L158 类 HierarchicalValueNetwork：双时间尺度价值网络 V = α * V_long(q, z_φ) + (1-α) * V_short(s) 其中 α 根据自动机阶段动态调整
- L166 函数 HierarchicalValueNetwork.__init__
- L201 函数 HierarchicalValueNetwork.forward：Args: env_state: [batch, env_dim] automaton_features: [batch, 4] formula_embedding: [batch, emb_dim] Returns: dict with keys: v_total, v_short, v_long, alpha
- L237 类 MultiTimescaleLoss：多时间尺度联合损失 L = λ_short * L_short + λ_mid * L_mid + λ_long * L_long
- L243 函数 MultiTimescaleLoss.__init__
- L254 函数 MultiTimescaleLoss.forward：Args: v_pred: [batch, seq_len] 预测价值 rewards: [batch, seq_len] 实际奖励 dones: [batch, seq_len] 终止标记 gamma: 折扣因子 Returns: dict with total_loss, loss_short, loss_mid, loss_long
- L300 函数 MultiTimescaleLoss._compute_nstep_return：计算 n-step return
- L316 函数 MultiTimescaleLoss._compute_mc_return：计算 Monte Carlo return

### src/model/longshort_integration.py

- 行数：489；字节：19151。
- 模块说明：集成模块：将长短期RL组件嵌入现有 deep-ltl 模型 使用方式： from model.longshort_integration import enhance_model_with_multiscale model = build_model(...) model = enhance_model_with_multiscale( model, enable_hierarchical_value=True, enable_multiscale_gnn=True, enable_automaton_tracking=True )
- 顶层导入：torch, torch.nn, typing, dataclasses, model.hierarchical_value, model.multiscale_gnn
- L33 类 LongShortConfig：长短期RL配置
- L52 类 LongShortEnhancedModel：增强模型：在原有 Model 基础上添加长短期RL组件 保持向后兼容： - 如果所有开关关闭，行为与原模型一致 - 渐进式启用：可单独开启任意组件
- L60 函数 LongShortEnhancedModel.__init__
- L160 函数 LongShortEnhancedModel.forward：前向传播（兼容原接口） Args: obs: 观测字典 return_multiscale: 是否返回多尺度信息 Returns: 如果 return_multiscale=False: 与原模型相同的输出 如果 return_multiscale=True: 增强输出（包含长短期信息）
- L222 函数 LongShortEnhancedModel.compute_loss：计算多时间尺度损失 Args: predictions: forward() 返回的增强输出 rewards: [batch, seq_len] 奖励序列 dones: [batch, seq_len] 终止标记 returns: 可选的预计算回报 Returns: 损失字典
- L280 函数 LongShortEnhancedModel.reset_automaton_tracker：重置自动机追踪器（新 episode 开始时）
- L287 函数 LongShortEnhancedModel._infer_env_dim：从 base_model 推断环境状态维度
- L306 函数 LongShortEnhancedModel._extract_propositions：从观测中提取命题真值
- L329 函数 LongShortEnhancedModel._replace_gnn_with_multiscale：用多尺度GNN替换原有GNN
- L387 函数 enhance_model_with_multiscale：便捷函数：增强现有模型 Args: model: 原始 deep-ltl 模型 ldba: LDBA 对象 config: LongShortConfig 对象（如果提供则优先使用） enable_*: 各组件开关（当 config=None 时使用） **kwargs: 其他配置参数 Returns: 增强后的模型
- L430 函数 load_checkpoint_compatible：向后兼容的 checkpoint 加载器 处理两种情况： 1. 基础模型 checkpoint -> 加载到 base_model 2. 增强模型 checkpoint -> 直接加载 Args: model: 当前模型（可能是基础模型或增强模型） state_dict: 保存的模型权重 strict: 是否严格匹配（默认 False）

### src/model/ltl/__init__.py

- 行数：0；字节：0。

### src/model/ltl/ltl_net.py

- 行数：59；字节：2424。
- 顶层导入：torch, torch, model.ltl.transformer_model, preprocessing
- L8 类 LTLNet
- L9 函数 LTLNet.__init__
- L34 函数 LTLNet.forward

### src/model/ltl/set_network.py

- 行数：19；字节：570。
- 顶层导入：typing, torch, torch, utils
- L9 类 SetNetwork
- L11 函数 SetNetwork.__init__
- L17 函数 SetNetwork.forward

### src/model/ltl/transformer_model.py

- 行数：54；字节：2163。
- 顶层导入：torch, torch
- L4 类 PositionalEncoding
- L5 函数 PositionalEncoding.__init__
- L18 函数 PositionalEncoding.forward
- L23 类 TransformerModel
- L24 函数 TransformerModel.__init__
- L41 函数 TransformerModel.forward

### src/model/mixed_distribution.py

- 行数：51；字节：2039。
- 顶层导入：torch, torch.distributions, ltl.automata
- L7 类 MixedDistribution：A class to represent a mixed (categorical and continuous) distribution. We only have a single category (whether to take an epsilon transition) and possibly multiple continuous actions.
- L14 函数 MixedDistribution.__init__
- L20 函数 MixedDistribution.set_epsilon_mask
- L23 函数 MixedDistribution.sample
- L29 函数 MixedDistribution.log_prob
- L41 函数 MixedDistribution.entropy
- L47 函数 MixedDistribution.mode

### src/model/model.py

- 行数：398；字节：19859。
- 顶层导入：typing, gymnasium, torch, torch.nn, config, model.ltl.ltl_net, model.gnn_net, preprocessing.batched_sequences, model.mixed_distribution, preprocessing.vocab, model.policy, model.policy, utils, utils.torch_utils
- L19 类 Model
- L20 函数 Model.__init__
- L38 函数 Model._build_sequence_graph：高度向量化的图构建函数：将序列特征(batch, max_len, feat_dim) 按实际长度 reach_lens 构造成每个序列一条链式双向图（i <-> i+1）。 增强功能： - 支持从LDBASequence提取自动机状态和转移信息 - 支持接受状态节点特征标记 - 支持边类型特征（链式、跳跃、epsilon、接受转移等） 相比旧实现： - 去除 Python for 循环中的张量写入（除去 prefix 计算外完全使用批
- L146 函数 Model._extract_node_state_features：从LDBASequence提取节点状态特征（接受状态、SCC信息等）
- L169 函数 Model._extract_edge_features：从LDBASequence提取边特征（边类型、是否接受转移等）
- L204 函数 Model.compute_embedding
- L311 函数 Model.forward
- L319 函数 build_model

### src/model/multiscale_gnn.py

- 行数：339；字节：12772。
- 模块说明：多尺度 GNN：从不同层提取短期/中期/长期语义特征 架构： - Layer 0: 1-hop 邻域 → 短期子目标 - Layer 1: 2-hop 邻域 → 中期规划 - Layer 2: 全局结构 → 长期策略
- 顶层导入：torch, torch.nn, torch_geometric.nn, torch_geometric.data, typing, dataclasses
- L24 类 MultiScaleEmbeddings：多尺度嵌入容器
- L35 类 MultiScaleSemanticGNN：多尺度语义 GNN（兼容旧 `GNNNet` 接口） 与原模型的集成需求： - 旧模型调用签名：forward(x, edge_index, edge_attr=None, node_state_feat=None, batch=None) - 需要属性：output_dim, use_edge_attr, node_state_dim - 权重迁移代码期望存在 `convs`（此处提供别名指向 `gat_layers`） 当前实现： 
- L49 函数 MultiScaleSemanticGNN.__init__
- L97 函数 MultiScaleSemanticGNN.forward
- L166 类 GlobalSemanticAttentionPooling：可学习的全局注意力池化 类似 Transformer 的 [CLS] token 机制
- L172 函数 GlobalSemanticAttentionPooling.__init__
- L179 函数 GlobalSemanticAttentionPooling.forward：全局注意力池化 node_features: [num_nodes, hidden_dim] batch_idx: [num_nodes] 所属图索引 返回: pooled [batch_size, hidden_dim], attention_weights(占位)
- L213 类 MultiTimescaleValueLoss：多时间尺度价值损失（配合 MultiScaleSemanticGNN） L = Σ_i λ_i * L_i(V_layer_i, Return_timescale_i)
- L219 函数 MultiTimescaleValueLoss.__init__
- L229 函数 MultiTimescaleValueLoss.forward：计算每层对应时间尺度的价值损失
- L264 函数 MultiTimescaleValueLoss._compute_nstep_return：n-step return
- L279 函数 MultiTimescaleValueLoss._compute_mc_return：Monte Carlo return

### src/model/parallel_agent.py

- 行数：34；字节：1329。
- 顶层导入：numpy, torch, preprocessing, model.model, sequence.search
- L9 类 ParallelAgent
- L10 函数 ParallelAgent.__init__
- L16 函数 ParallelAgent.update_dones
- L21 函数 ParallelAgent.get_action
- L28 函数 ParallelAgent.forward

### src/model/policy/__init__.py

- 行数：2；字节：88。
- 顶层导入：.continuous_actor, .discrete_actor

### src/model/policy/continuous_actor.py

- 行数：44；字节：1774。
- 顶层导入：numpy, torch, torch.nn, torch.distributions, model.mixed_distribution, utils
- L10 类 ContinuousActor
- L11 函数 ContinuousActor.__init__
- L34 函数 ContinuousActor.forward

### src/model/policy/discrete_actor.py

- 行数：26；字节：847。
- 顶层导入：typing, torch, torch.nn, torch.distributions, model.discrete_epsilon_distribution, utils
- L11 类 DiscreteActor
- L12 函数 DiscreteActor.__init__
- L24 函数 DiscreteActor.forward

### src/model/seq_agent.py

- 行数：34；字节：1039。
- 顶层导入：copy, numpy, preprocessing, model.model
- L9 类 SequenceAgent
- L10 函数 SequenceAgent.__init__
- L14 函数 SequenceAgent.get_action
- L18 函数 SequenceAgent.reset
- L21 函数 SequenceAgent.forward
- L29 函数 SequenceAgent.get_value

### src/preprocessing/__init__.py

- 行数：3；字节：125。
- 顶层导入：.preprocessing, .batched_sequences, .vocab

### src/preprocessing/batched_sequences.py

- 行数：82；字节：3121。
- 顶层导入：dataclasses, torch, numpy
- L10 类 BatchedSequences
- L16 函数 BatchedSequences.from_seqs
- L31 函数 BatchedSequences.__getitem__
- L34 函数 BatchedSequences.all
- L38 类 BatchedReachAvoidSequences
- L39 函数 BatchedReachAvoidSequences.__init__：Args: seqs: list of reach-avoid sequences device: torch device ldba_sequences: optional list of原始LDBASequence对象，用于提取状态和转移信息
- L51 函数 BatchedReachAvoidSequences.batch
- L63 函数 BatchedReachAvoidSequences.split_seq
- L75 函数 BatchedReachAvoidSequences.__getitem__：Returns a sub-batch of the given sequences.
- L81 函数 BatchedReachAvoidSequences.all

### src/preprocessing/graph_processor.py

- 行数：189；字节：8321。
- 顶层导入：torch, torch_geometric.data, numpy, typing, ltl.automata
- L7 类 GraphDataProcessor：图数据增强与预处理工具
- L11 函数 GraphDataProcessor.add_virtual_nodes：添加虚拟节点以增强图的全局信息传递
- L46 函数 GraphDataProcessor.add_subgraph_edges：添加子图级别的边连接以捕获局部结构
- L76 函数 GraphDataProcessor.add_semantic_edges：基于语义信息添加边
- L112 函数 GraphDataProcessor.augment_features：使用图论特征增强节点表示
- L174 函数 GraphDataProcessor.process_graph：应用完整的图处理流程

### src/preprocessing/ltl_graph.py

- 行数：55；字节：1672。
- 顶层导入：torch, torch_geometric.data, preprocessing.vocab
- L5 函数 preprocess_ltl_to_graph：将LTL tokens转换为图结构 Args: ltl_tokens: LTL公式的token序列 Returns: Data: PyTorch Geometric的Data对象，包含图结构信息
- L44 函数 preprocess_batch_to_graphs：处理批量的LTL公式 Args: batch_ltl_tokens: 批量的LTL token序列 Returns: Batch: PyTorch Geometric的Batch对象

### src/preprocessing/preprocessing.py

- 行数：62；字节：2378。
- 顶层导入：typing, torch, torch_ac, numpy, ltl.automata, ltl.logic, preprocessing.vocab, preprocessing.batched_sequences
- L14 函数 preprocess_obss
- L47 函数 preprocess_features
- L51 函数 preprocess_sequence
- L57 函数 preprocess_assignments

### src/preprocessing/vocab.py

- 行数：15；字节：340。
- 顶层导入：ltl.logic
- L7 函数 reset_vocab
- L13 函数 init_vocab

### src/sequence/__init__.py

- 行数：0；字节：0。

### src/sequence/samplers/__init__.py

- 行数：9；字节：344。
- 顶层导入：.curriculum, .curriculum_sampler

### src/sequence/samplers/curriculum.py

- 行数：278；字节：8988。
- 顶层导入：random, abc, dataclasses, typing, numpy, torch, ltl.automata, sequence.samplers.flatworld_sequence_samplers, sequence.samplers.sequence_samplers
- L17 类 CurriculumStage
- L22 函数 CurriculumStage.sample
- L26 函数 CurriculumStage.update_task_success
- L31 类 ExplicitCurriculumStage：A curriculum stage in which all tasks are explicitly listed, and sampled from according to previous success.
- L39 函数 ExplicitCurriculumStage.sample
- L52 函数 ExplicitCurriculumStage.compute_sampling_prob
- L59 函数 ExplicitCurriculumStage.update_task_success
- L70 类 RandomCurriculumStage：A curriculum stage in which tasks are sampled randomly.
- L74 函数 RandomCurriculumStage.sample
- L77 函数 RandomCurriculumStage.update_task_success
- L82 类 MultiRandomStage：A combination of multiple RandomCurriculumStages with associated sampling probabilities.
- L87 函数 MultiRandomStage.sample
- L91 函数 MultiRandomStage.update_task_success
- L95 类 Curriculum
- L96 函数 Curriculum.__init__
- L102 函数 Curriculum.current_stage
- L106 函数 Curriculum.finished
- L109 函数 Curriculum.sample
- L112 函数 Curriculum.update_task_success

### src/sequence/samplers/curriculum_sampler.py

- 行数：18；字节：556。
- 顶层导入：typing, ltl.automata, sequence.samplers.curriculum
- L7 类 CurriculumSampler
- L10 函数 CurriculumSampler.partial
- L13 函数 CurriculumSampler.__init__
- L17 函数 CurriculumSampler.__call__

### src/sequence/samplers/flatworld_sequence_samplers.py

- 行数：132；字节：5398。
- 顶层导入：random, pprint, typing, ltl.automata, ltl.logic, envs.flatworld
- L15 函数 get_complete_color_assignments
- L27 函数 flatworld_all_reach_tasks
- L51 函数 flatworld_sample_reach
- L64 函数 flatworld_all_reach_avoid
- L75 函数 flatworld_sample_reach_avoid
- L115 函数 flatworld_sample_reach_stay

### src/sequence/samplers/sequence_samplers.py

- 行数：139；字节：5746。
- 顶层导入：random, pprint, typing, ltl.automata, ltl.logic
- L9 函数 sample_reach_avoid
- L45 函数 all_reach_avoid_tasks
- L71 函数 all_reach_tasks
- L97 函数 all_reach_stay_tasks
- L113 函数 sample_reach_stay
- L131 函数 fixed

### src/sequence/search/__init__.py

- 行数：3；字节：113。
- 顶层导入：.sequence_search, .exhaustive_search, .bfs

### src/sequence/search/bfs.py

- 行数：56；字节：2257。
- 顶层导入：dataclasses, torch, ltl.automata, sequence.search
- L10 类 SearchNode
- L16 类 BFS
- L17 函数 BFS.__init__
- L20 函数 BFS.__call__
- L26 函数 BFS.bfs

### src/sequence/search/exhaustive_search.py

- 行数：159；字节：7451。
- 顶层导入：dataclasses, typing, torch, ltl.automata, sequence.search
- L11 类 Path
- L15 函数 Path.__len__
- L18 函数 Path.__getitem__
- L23 函数 Path.__str__
- L28 函数 Path.prepend
- L31 函数 Path.to_sequence
- L38 函数 Path.reach_avoid_to_assignments
- L45 类 ExhaustiveSearch
- L46 函数 ExhaustiveSearch.__init__
- L51 函数 ExhaustiveSearch.__call__
- L58 函数 ExhaustiveSearch.all_sequences
- L63 函数 ExhaustiveSearch.dfs：Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
- L129 函数 ExhaustiveSearch.prune_paths
- L147 函数 ExhaustiveSearch.check_path_contained

### src/sequence/search/sequence_search.py

- 行数：96；字节：3673。
- 顶层导入：abc, torch, preprocessing, ltl.automata
- L9 类 SequenceSearch：A search that can be performed on an LDBA and yields the optimal sequence according to the model.
- L12 函数 SequenceSearch.__init__
- L17 函数 SequenceSearch.__call__
- L20 函数 SequenceSearch.get_value
- L40 函数 SequenceSearch.collect_avoid_transitions
- L50 函数 SequenceSearch.augment_sequence：Augments the sequence to avoid transitions that lead to non-accepting loops.
- L76 函数 SequenceSearch.only_non_accepting_loops

### src/torch_ac/__init__.py

- 行数：2；字节：64。
- 顶层导入：.algos, .utils

### src/torch_ac/algos/__init__.py

- 行数：1；字节：21。
- 顶层导入：.ppo

### src/torch_ac/algos/base.py

- 行数：258；字节：10654。
- 顶层导入：abc, collections, torch, torch_ac.utils, numpy, torch_ac.utils.sync_env
- L13 类 BaseAlgo：The base class for RL algorithms.
- L16 函数 BaseAlgo.__init__：Initializes a `BaseAlgo` instance. Parameters: ---------- envs : list a list of environments that will be run in parallel acmodel : torch.Module the model num_steps_per_proc : int the number of steps collected by every p
- L117 函数 BaseAlgo.collect_experiences：Collects rollouts and computes advantages. Runs several environments concurrently. The next actions are computed in a batch mode for all environments at the same time. The rollouts and advantages from all environments ar
- L257 函数 BaseAlgo.update_parameters

### src/torch_ac/algos/ppo.py

- 行数：170；字节：6713。
- 顶层导入：typing, numpy, torch, config, torch_ac.algos.base, math
- L10 类 PPO：The Proximal Policy Optimization algorithm ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347)).
- L14 函数 PPO.__init__
- L31 函数 PPO.update_parameters
- L144 函数 PPO._get_batches_starting_indexes：Gives, for each batch, the indexes of the observations given to the model and the experiences used to compute the loss at first. First, the indexes are the integers from 0 to `self.num_frames` with a step of `self.recurr

### src/torch_ac/utils/__init__.py

- 行数：3；字节：99。
- 顶层导入：.dictlist, .parallel_env, .sync_env

### src/torch_ac/utils/dictlist.py

- 行数：24；字节：735。
- L1 类 DictList：A dictionary of lists of same size. Dictionary items can be accessed using `.` notation and list items using `[]` notation. Example: >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]}) >>> d.a [[1, 2], [3, 4]] >>>
- L16 函数 DictList.__len__
- L19 函数 DictList.__getitem__
- L22 函数 DictList.__setitem__

### src/torch_ac/utils/parallel_env.py

- 行数：59；字节：1829。
- 顶层导入：multiprocessing, gymnasium
- L4 函数 worker
- L20 类 ParallelEnv：A concurrent execution of environments in multiple processes.
- L23 函数 ParallelEnv.__init__
- L39 函数 ParallelEnv.__del__
- L43 函数 ParallelEnv.reset
- L49 函数 ParallelEnv.step
- L58 函数 ParallelEnv.render

### src/torch_ac/utils/sync_env.py

- 行数：30；字节：854。
- 顶层导入：multiprocessing, gymnasium
- L5 类 SyncEnv：A sequential execution of multiple environments.
- L8 函数 SyncEnv.__init__
- L15 函数 SyncEnv.reset
- L19 函数 SyncEnv.step
- L29 函数 SyncEnv.render

### src/train/train_ppo.py

- 行数：248；字节：11867。
- 顶层导入：argparse, typing, gymnasium, simple_parsing, time, datetime, torch, config, preprocessing, torch_ac, utils, model.model, envs, model.longshort_integration, ltl.automata, sequence.samplers, utils, utils.logging.file_logger, utils.logging.multi_logger, utils.logging.text_logger, utils.logging.wandb_logger, utils.model_store, config
- L35 类 Trainer
- L36 函数 Trainer.__init__
- L41 函数 Trainer.train
- L150 函数 Trainer.make_envs
- L168 函数 Trainer.get_training_status
- L178 函数 Trainer.make_logger
- L186 函数 Trainer.augment_logs
- L208 函数 parse_arguments
- L236 函数 main

### src/training/hierarchical_trainer.py

- 行数：180；字节：6775。
- 顶层导入：torch, torch.nn, torch.nn.functional, typing
- L6 类 GraphContrastiveLoss：图对比学习损失函数
- L9 函数 GraphContrastiveLoss.__init__
- L13 函数 GraphContrastiveLoss.forward：计算图对比损失 Args: features: 图表示 [num_graphs, hidden_dim] batch: batch索引 [num_graphs]
- L38 类 CurriculumScheduler：课程学习调度器
- L41 函数 CurriculumScheduler.__init__
- L51 函数 CurriculumScheduler.get_complexity：获取当前epoch的任务复杂度
- L60 函数 CurriculumScheduler.should_increase_complexity：基于性能指标决定是否增加复杂度
- L64 类 HierarchicalTrainer：分层训练管理器
- L67 函数 HierarchicalTrainer.__init__
- L80 函数 HierarchicalTrainer.compute_hierarchical_loss：计算分层损失
- L110 函数 HierarchicalTrainer.compute_structure_loss：计算结构正则化损失
- L124 函数 HierarchicalTrainer.train_epoch：训练一个epoch
- L173 函数 HierarchicalTrainer.sample_by_complexity：根据复杂度采样任务

### src/utils/__init__.py

- 行数：8；字节：166。
- 顶层导入：.file_utils, .utils, .reward_utils, .sympy_utils, joblib

### src/utils/file_utils.py

- 行数：29；字节：1110。
- 顶层导入：argparse, os
- L5 函数 get_experiment_path
- L19 函数 get_eval_results_path
- L23 函数 get_experiment_path_from_config
- L28 函数 get_pretraining_experiment_path

### src/utils/logging/__init__.py

- 行数：0；字节：0。

### src/utils/logging/file_logger.py

- 行数：65；字节：2420。
- 顶层导入：argparse, copy, csv, fcntl, json, os, utils, config, utils.logging.json_encoder, utils.logging.logger
- L14 类 FileLogger：A simple logger that writes a CSV file.
- L19 函数 FileLogger.__init__
- L32 函数 FileLogger.log_config
- L47 函数 FileLogger.config_as_dict
- L57 函数 FileLogger.log

### src/utils/logging/json_encoder.py

- 行数：13；字节：303。
- 顶层导入：dataclasses, json, typing, torch.nn
- L8 类 JsonEncoder
- L9 函数 JsonEncoder.default

### src/utils/logging/logger.py

- 行数：51；字节：1387。
- 顶层导入：argparse, abc, numpy
- L7 类 Logger
- L8 函数 Logger.__init__
- L13 函数 Logger.log_config：Logs the given config.
- L20 函数 Logger.log：Logs the given data.
- L26 函数 Logger.finish：Finishes logging for the current experiment.
- L33 函数 Logger.aggregate：Aggregates values for keys if there are multiple values for a key.
- L46 函数 Logger.check_keys_valid

### src/utils/logging/multi_logger.py

- 行数：27；字节：750。
- 顶层导入：utils.logging.logger
- L4 类 MultiLogger：A logger that logs to multiple loggers.
- L9 函数 MultiLogger.__init__
- L17 函数 MultiLogger.log_config
- L21 函数 MultiLogger.log
- L25 函数 MultiLogger.finish

### src/utils/logging/text_logger.py

- 行数：63；字节：1781。
- 顶层导入：argparse, colorama, utils.logging.logger
- L7 类 TextLogger：A logger that logs to standard output.
- L12 函数 TextLogger.__init__
- L15 函数 TextLogger.log_config
- L18 函数 TextLogger.log
- L35 函数 TextLogger.get_short_name
- L58 函数 TextLogger.info
- L62 函数 TextLogger.important_info

### src/utils/logging/wandb_logger.py

- 行数：51；字节：1573。
- 顶层导入：argparse, os, wandb, utils, utils.logging.logger
- L10 类 WandbLogger：A logger that logs to Weights & Biases.
- L17 函数 WandbLogger.__init__
- L28 函数 WandbLogger.log_config
- L44 函数 WandbLogger.log
- L50 函数 WandbLogger.finish

### src/utils/model_store/__init__.py

- 行数：1；字节：27。
- 顶层导入：.model_store

### src/utils/model_store/model_store.py

- 行数：68；字节：2716。
- 顶层导入：argparse, json, os, pickle, pandas, torch, utils, preprocessing
- L13 类 ModelStore
- L14 函数 ModelStore.__init__
- L19 函数 ModelStore.from_config
- L23 函数 ModelStore.save_training_status
- L26 函数 ModelStore.save_eval_training_status
- L29 函数 ModelStore.save_ltl_net
- L32 函数 ModelStore.save_vocab
- L36 函数 ModelStore.load_vocab
- L42 函数 ModelStore.load_training_status
- L47 函数 ModelStore.load_best_model
- L58 函数 ModelStore.load_eval_training_statuses

### src/utils/reward_utils.py

- 行数：21；字节：917。
- 顶层导入：numpy
- L4 函数 average_reward_per_step
- L14 函数 average_discounted_return

### src/utils/sympy_utils.py

- 行数：22；字节：521。
- 顶层导入：functools, sympy
- L7 函数 to_sympy
- L13 函数 sympy_to_str
- L21 函数 simplify

### src/utils/torch_utils.py

- 行数：43；字节：1399。
- 顶层导入：functools, typing, torch, torch
- L10 函数 get_device
- L21 函数 make_mlp_layers
- L42 函数 get_number_of_params

### src/utils/utils.py

- 行数：35；字节：903。
- 顶层导入：random, subprocess, contextlib, typing, time, numpy, torch
- L11 函数 set_seed
- L19 类 timeit
- L20 函数 timeit.__init__
- L23 函数 timeit.__enter__
- L27 函数 timeit.__exit__
- L34 函数 kill_all_wandb_processes

### src/visualize/__init__.py

- 行数：0；字节：0。

### src/visualize/automata.py

- 行数：68；字节：2368。
- 顶层导入：enum, graphviz, ltl.automata, ltl.logic, sequence.search
- L10 类 Color
- L15 函数 Color.__str__
- L19 函数 draw_ldba：Draw an LDBA as a graph using Graphviz.
- L45 函数 construct_ldba

### src/visualize/zones.py

- 行数：149；字节：4960。
- 顶层导入：matplotlib.pyplot, numpy, matplotlib.colors, matplotlib.patches, matplotlib.axes
- L9 类 StaticColorAxisBBox
- L10 函数 StaticColorAxisBBox.set_edgecolor
- L16 函数 StaticColorAxisBBox.set_linewidth
- L20 类 FancyAxes
- L24 函数 FancyAxes.__init__
- L29 函数 FancyAxes._gen_axes_patch
- L47 函数 draw_circle
- L52 函数 draw_zones
- L60 函数 draw_path
- L70 函数 draw_diamond
- L82 函数 draw_trajectories
- L98 函数 draw_multiple_trajectories
- L109 函数 setup_axis
- L144 函数 hide_ticks

### tests/legacy/test_enhanced_gnn.py

- 行数：116；字节：3976。
- 模块说明：增强GNN性能对比测试脚本 对比三种配置： 1. 无GNN（仅RNN/Transformer） 2. 基础GNN（仅链式边） 3. 增强GNN（边特征 + 节点状态特征） 运行方式： python test_enhanced_gnn.py --env PointLtl2-v0 --formula "(!blue U green) & F yellow" --seeds 5
- 顶层导入：argparse, sys, os, torch, numpy, pathlib
- L24 函数 run_evaluation：运行评估并返回结果
- L46 函数 compare_results：对比基线和增强版本的结果
- L79 函数 main

### tests/legacy/test_longshort_components.py

- 行数：150；字节：4879。
- 模块说明：长短期RL集成测试脚本 快速验证各组件是否正常工作
- 顶层导入：torch, sys

### tests/legacy/test_model_loading.py

- 行数：73；字节：2252。
- 顶层导入：sys, torch, utils.model_store, config, model.model, envs, ltl

### tests/test_gnn_embedding_quick.py

- 行数：99；字节：3854。
- 顶层导入：torch, torch.nn, random, preprocessing.batched_sequences, model.gnn_net, model.model
- L10 类 FakeLTLNet
- L11 函数 FakeLTLNet.__init__
- L23 函数 FakeLTLNet.forward
- L60 函数 build_minimal_model
- L71 函数 make_dummy_batched_seq

### tests/test_gnn_fix.py

- 行数：146；字节：4293。
- 模块说明：测试GNN相关代码的修复
- 顶层导入：torch, sys, pathlib, model.gnn_net
- L11 函数 test_gnn_initialization：测试GNN初始化
- L33 函数 test_gnn_forward：测试GNN前向传播
- L73 函数 test_simple_sequence：测试简单序列的图构建

## 结果 CSV 实际结构

| 文件 | 数据行 | 表头 | 方法标签 |
|---|---:|---|---|
| results/FlatWorld-v0.csv | 75 | method, task, seed, success_rate, mean_steps | DeepLTL, GCRL-LTL, LTL2Action |
| results/LetterEnv-v0.csv | 75 | method, task, seed, success_rate, mean_steps | DeepLTL, GCRL-LTL, LTL2Action |
| results/PointLtl2-v0.csv | 178 | method, task, seed, success_rate, mean_steps | DeepLTL, GCRL-LTL, LTL2Action, RAD, RADd, Transformer |
| results/PointLtl2-v0_results_paper.csv | 0 | config, task, task_type, difficulty, seed, success_rate, violation_rate, avg_return, avg_steps, status |  |
| results/generalisation.csv | 30 | method, task, seed, success_rate, mean_steps | DeepLTL, GCRL-LTL, LTL2Action |
| results/safety.csv | 20 | method, seed, success_rate | DeepLTL, GCRL-LTL, LTL2Action, LTL2Action2 |
| results_infinite/FlatWorld-v0.csv | 24 | method, task, seed, accepting_visits | DeepLTL, GCRL-LTL |
| results_infinite/LetterEnv-v0.csv | 20 | method, task, seed, accepting_visits | DeepLTL, GCRL-LTL |
| results_infinite/PointLtl2-v0.csv | 60 | method, task, seed, accepting_visits | DeepLTL, GCRL-LTL |

## 文档、论文、图片与评估任务清单

| 文件 | 字节 | 文本行数（适用时） |
|---|---:|---:|
| docs/PROJECT_STATUS.md | 17739 | 196 |
| docs/README.md | 1251 | 14 |
| docs/archive/EXPERIMENT_HISTORY.md | 106361 | 3360 |
| docs/archive/PAPER_REVISION_HISTORY.md | 128387 | 2316 |
| docs/archive/TECHNICAL_HISTORY.md | 95047 | 2660 |
| docs/archive/config/enhanced_gnn_config.py | 1677 | 65 |
| docs/archive/config/hierarchical_gnn_config.yaml | 1032 | 53 |
| docs/archive/config/legacy_config.py | 1950 | 70 |
| docs/archive/papers/DeepLTL_LS_Complete_Paper.md | 33852 | 891 |
| docs/archive/papers/DeepLTL_LS_Final_Paper.md | 36686 | 869 |
| docs/archive/papers/DeepLTL_LS_Final_Paper.tex | 61910 | 1144 |
| docs/archive/papers/DeepLTL_LS_Final_Paper_EN.md | 41632 | 871 |
| docs/archive/papers/DeepLTL_LS_Paper_English.md | 37472 | 891 |
| docs/archive/papers/New_Paper_Draft.md | 15398 | 234 |
| docs/archive/papers/New_Paper_Draft_EN.md | 7053 | 129 |
| docs/archive/papers/Paper_Complete_CN.md | 17156 | 187 |
| docs/archive/papers/Paper_Complete_EN.md | 10866 | 95 |
| docs/archive/papers/paper.tex | 12224 | 144 |
| docs/maintenance/CLEANUP.md | 2774 | 46 |
| docs/maintenance/VALIDATION.md | 3012 | 46 |
| eval_datasets/FlatWorld-big-12-v0/tasks.txt | 17548 | 500 |
| eval_datasets/FlatWorld-big-12-v0/tasks_ltl2action.txt | 37156 | 500 |
| eval_datasets/FlatWorld-big-16-v0/tasks.txt | 17252 | 500 |
| eval_datasets/FlatWorld-big-16-v0/tasks_ltl2action.txt | 37144 | 500 |
| eval_datasets/FlatWorld-big-20-v0/tasks.txt | 17801 | 500 |
| eval_datasets/FlatWorld-big-20-v0/tasks_ltl2action.txt | 37000 | 500 |
| eval_datasets/FlatWorld-big-8-v0/tasks.txt | 18322 | 500 |
| eval_datasets/FlatWorld-big-8-v0/tasks_ltl2action.txt | 36784 | 500 |
| eval_datasets/FlatWorld-v0/tasks.txt | 2504 | 49 |
| eval_datasets/FlatWorld-v0/tasks_ltl2action.txt | 5020 | 50 |
| eval_datasets/LetterEnv-v0/tasks.txt | 1054 | 50 |
| eval_datasets/LetterEnv-v0/tasks_ltl2action.txt | 3676 | 50 |
| eval_datasets/LetterEnv-v0/worlds/world_info_0.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_1.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_10.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_11.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_12.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_13.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_14.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_15.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_16.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_17.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_18.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_19.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_2.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_20.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_21.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_22.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_23.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_24.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_25.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_26.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_27.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_28.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_29.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_3.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_30.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_31.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_32.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_33.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_34.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_35.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_36.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_37.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_38.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_39.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_4.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_40.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_41.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_42.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_43.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_44.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_45.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_46.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_47.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_48.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_49.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_5.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_6.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_7.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_8.pkl | 232 |  |
| eval_datasets/LetterEnv-v0/worlds/world_info_9.pkl | 232 |  |
| eval_datasets/PointLtl2-v0/tasks.txt | 1867 | 50 |
| eval_datasets/PointLtl2-v0/tasks_ltl2action.txt | 3640 | 50 |
| eval_datasets/PointLtl2-v0/worlds/world_info_0.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_1.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_10.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_11.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_12.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_13.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_14.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_15.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_16.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_17.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_18.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_19.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_2.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_20.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_21.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_22.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_23.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_24.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_25.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_26.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_27.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_28.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_29.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_3.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_30.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_31.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_32.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_33.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_34.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_35.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_36.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_37.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_38.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_39.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_4.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_40.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_41.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_42.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_43.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_44.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_45.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_46.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_47.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_48.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_49.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_5.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_6.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_7.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_8.pkl | 4543 |  |
| eval_datasets/PointLtl2-v0/worlds/world_info_9.pkl | 4543 |  |
| paper/2025-05-07 DEEPLTL LEARNING TO EFFICIENTLY SATISFY COMPLEX LTL SPECIFICATIONS FOR MULTI-TASK RL.pdf | 1067176 |  |
| paper/README.md | 2450 | 42 |
| paper/figures/figure0_intro_comparison.pdf | 21816 |  |
| paper/figures/figure0_intro_comparison.png | 195509 |  |
| paper/figures/figure1_architecture_new.png | 440627 |  |
| paper/figures/figure1_se_gnn.pdf | 28793 |  |
| paper/figures/figure1_se_gnn.png | 313426 |  |
| paper/figures/figure2_fusion.pdf | 22042 |  |
| paper/figures/figure2_fusion.png | 354665 |  |
| paper/figures/figure3_ldba_transitions.pdf | 21482 |  |
| paper/figures/figure3_ldba_transitions.png | 614261 |  |
| paper/figures/figure4_fusion_new.png | 500379 |  |
| paper/figures/figure4_fusion_weight.pdf | 32477 |  |
| paper/figures/figure4_fusion_weight.png | 354072 |  |
| paper/figures/figure4_fusion_weight_paper.png | 354063 |  |
| paper/figures/figure5_performance.pdf | 26585 |  |
| paper/figures/figure5_performance.png | 224101 |  |
| paper/figures/figure5_performance_paper.pdf | 26677 |  |
| paper/figures/figure6_architecture.pdf | 23570 |  |
| paper/figures/figure6_architecture.png | 250452 |  |
| paper/figures/figure6_architecture_compact.pdf | 24355 |  |
| paper/figures/figure6_architecture_compact.png | 332155 |  |
| paper/figures/figure6_architecture_flow.pdf | 26137 |  |
| paper/figures/figure6_architecture_flow.png | 420281 |  |
| paper/figures/figure6_architecture_paper.png | 250452 |  |
| paper/figures/figure_automaton_enhanced.pdf | 40942 |  |
| paper/figures/figure_automaton_enhanced.png | 541213 |  |
| paper/figures/figure_automaton_optimized.pdf | 31532 |  |
| paper/figures/figure_automaton_optimized.png | 433680 |  |
| paper/figures/figure_complete_system.pdf | 31025 |  |
| paper/figures/figure_complete_system.png | 335276 |  |
| paper/figures/figure_detailed_fusion.pdf | 33640 |  |
| paper/figures/figure_detailed_fusion.png | 400322 |  |
| paper/figures/figure_ldba_flow.png | 341974 |  |
| paper/figures/figure_limitations.pdf | 36531 |  |
| paper/figures/figure_limitations.png | 224264 |  |
| paper/figures/hiltl_architecture.pdf | 40848 |  |
| paper/figures/hiltl_architecture.png | 523766 |  |
| paper/figures/hiltl_architecture.svg | 94801 |  |
| paper/paper-new.tex | 38065 | 588 |
| paper/paper.tex | 67249 | 912 |
| paper/references.bib | 25572 | 688 |
| paper/scripts/generate_automaton_figure.py | 10777 | 234 |
| paper/scripts/generate_figures.py | 30182 | 651 |
| paper/scripts/generate_limitations_figure.py | 9390 | 235 |
| paper/submission/COVER_LETTER_TNNLS.txt | 6850 | 159 |
| paper/submission/TITLE_PAGE_TEMPLATE.txt | 5053 | 147 |
| paper/templates/IEEE_journal_example.tex | 26898 | 716 |

## 第三方与资源保护

Safety Gymnasium 含大量模型、纹理、环境源码、测试与许可；Rabinizer 含 bin/lib/src/doc。未尝试“瘦身”这些依赖。下表按第三方一级子目录汇总。

| 路径 | 文件数 | 字节 |
|---|---:|---:|
| rabinizer4/bin | 26 | 96730 |
| rabinizer4/doc | 468 | 8645065 |
| rabinizer4/lib | 1 | 2469983 |
| rabinizer4/src | 356 | 1365037 |
| src/envs/zones/safety-gymnasium/.editorconfig | 1 | 453 |
| src/envs/zones/safety-gymnasium/.flake8 | 1 | 631 |
| src/envs/zones/safety-gymnasium/.gitattributes | 1 | 169 |
| src/envs/zones/safety-gymnasium/.github | 8 | 12278 |
| src/envs/zones/safety-gymnasium/.gitignore | 1 | 7287 |
| src/envs/zones/safety-gymnasium/.pre-commit-config.yaml | 1 | 3212 |
| src/envs/zones/safety-gymnasium/.pylintrc | 1 | 20954 |
| src/envs/zones/safety-gymnasium/.readthedocs.yaml | 1 | 471 |
| src/envs/zones/safety-gymnasium/CHANGELOG.md | 1 | 3612 |
| src/envs/zones/safety-gymnasium/CITATION.cff | 1 | 938 |
| src/envs/zones/safety-gymnasium/CODE_OF_CONDUCT.md | 1 | 5184 |
| src/envs/zones/safety-gymnasium/CONTRIBUTING.md | 1 | 3961 |
| src/envs/zones/safety-gymnasium/LICENSE | 1 | 11375 |
| src/envs/zones/safety-gymnasium/MANIFEST.in | 1 | 584 |
| src/envs/zones/safety-gymnasium/Makefile | 1 | 7545 |
| src/envs/zones/safety-gymnasium/README.md | 1 | 29833 |
| src/envs/zones/safety-gymnasium/examples | 4 | 7886 |
| src/envs/zones/safety-gymnasium/pyproject.toml | 1 | 5342 |
| src/envs/zones/safety-gymnasium/safety_gym_bug_report.md | 1 | 5359 |
| src/envs/zones/safety-gymnasium/safety_gymnasium | 693 | 105449246 |
| src/envs/zones/safety-gymnasium/setup.py | 1 | 1704 |
| src/envs/zones/safety-gymnasium/tests | 7 | 16625 |
