# 方法与当前实现

论文基准：`overleaf/HiLTL-final/`。本文描述代码的实际行为；论文与实现存在差异时，不将当前代码反向当成修改论文结论的依据，也不将测试通过表述为完整复现。

## 1. 主线

唯一维护训练入口为 `src/train/train_tlt.py`，协议 `tlt-upstream-curriculum-v1`。保留 DeepLTL 环境编码器、DeepSets/GRU、reach–avoid 序列课程和 PPO/GAE，加入自动机图、双头及历史。

上游参照提交 `3157200b5910cd7f5c493ffba8913994f2207eaf`，来源 <https://github.com/mathiasj33/deep-ltl>；离线参照位于 `tests/fixtures/deepltl/`。兼容检查覆盖 Zone/FlatWorld 构建器参数键、同seed初始化和输出，不等于整个依赖树逐字节复现。

|模块|位置与作用|
|---|---|
|图特征|`src/model/automaton_graph.py`、`src/ltl/automata/`；Rabinizer转换，4维节点、6维边，guard另编码|
|图表示|`src/model/tlt_sequence.py`；三层GATv2，64维、4 heads、残差与归一化|
|多尺度|各层attention pooling；开启MSGNN拼接三层，否则使用末层|
|短头|原critic加残差，环境及逻辑上下文，两层256隐藏单元|
|长头|proximity、当前逻辑节点、可选history，两层128隐藏单元；当前不输入物理状态|
|融合|固定 `beta=sigmoid(3*(proximity-0.5))`，`total=(1-beta)*short+beta*long`|
|历史|`src/envs/tlt_context.py`及LSTM；真实交互推进，前向/小批重放不推进|
|训练|`src/torch_ac/algos/tlt_ppo.py`；融合critic用于PPO，额外一步TD与完整回合MC loss，权重1和0.3|
|评估|`scripts/evaluate_tlt.py`；测试公式真实LDBA、继承的序列搜索、冻结模型|

图残差零初始化接入actor，原模型迁移后初始actor保持兼容；融合critic仍会因长头加入而改变。未结束回合跨rollout保存，不能伪造零MC标签；完整MC轨迹可能跨策略更新，并非全由同一冻结策略采集。

课程统计已修复跨阶段污染：并行旧阶段回合结束时，仅当前阶段任务进入当前统计。原任务、阈值和有效证据聚合规则保留，回归测试为 `tests/test_curriculum_transition.py`。

## 2. 配置

|配置|HV双头|多层池化|历史|
|---|---|---|---|
|`baseline`|否|否|否|
|`hv_only`|是|否|否|
|`msgnn_only`|否|是|否|
|`hv_track`|是|否|是|
|`msgnn_track`|否|是|是|
|`full_no_options`|是|是|是|

增强配置保留基础语义图，关闭MSGNN不等于移除GNN。`full_all/Options` 为论文历史参考，未纳入当前维护协议；论文尚未给出足以唯一重建的启动集、终止条件、高层策略与损失。旧实现可由Git历史恢复，不能把六配置可运行写成七配置复现。

## 3. 环境与任务

|环境|动作|回合上限|默认折扣|
|---|---|---:|---:|
|`PointLtl2-v0`|连续|1000|0.998|
|`LetterEnv-v0`|离散|75|0.94|
|`FlatWorld-v0`|离散|500|0.98|

Zone默认16环境、每环境4096 rollout、batch2048、10 epoch、lr0.0003；短测必须显式覆盖预算。它不等于论文所述1M步、batch64的实验配置。

任务见 `configurations/paper_tasks.json`：phi1=`F blue & F green`；phi2=`F blue & F green & F yellow`；phi3=`(!blue U green) & F yellow`。Letter以a/b/c替代blue/green/yellow。循环诊断为 `G F green`、`G F blue & G F yellow` 及Letter代换。

训练图是采样序列monitor，测试图为完整公式LDBA。`eval_datasets/` 原布局保留，但当前评估入口未直接重放全部原pkl布局，不能称已重跑原泛化/安全基准。Zone默认确定性动作，Letter/FlatWorld默认采样。评估分列成功、逻辑违例、物理回报和接受访问；中断回合不进入完整回合成功率分母。循环有限窗口代理不能证明无限时域满足或安全。

## 4. 与终稿的覆盖差异

|终稿行号（LTL/manuscript.tex）|论文描述|代码/证据与后续处理|
|---|---|---|
|298、614、616|长头含环境s|当前主要为q/逻辑历史；以论文为目标规范，完整复现版本需实现并重验，不能称当前已覆盖|
|298、649|actor advantage clip `[-1,1]`|当前无该额外裁剪；后续需显式实现/开关，不与PPO ratio clipping混淆|
|315、320、325、330|历史5/12/18状态，当前4/8/5|当前真实转换为4/8/5；论文已区分artifact，需历史HOA/配置验证旧计数|
|276、279|七配置，Options仅历史参考|当前六配置；先规范Options再复现，或保留清楚边界|
|298、332|1M步、5seed、500episodes|当前短测/32k先导非该实验；需成熟checkpoint、逐seed明细和聚合脚本连接主张与产物|

本次整理不通过修改论文数字、改变模型或启动长训练消除差异。投稿取向建议集中在[投稿指南](SUBMISSION_GUIDE.md)。
