# 当前项目现状

更新：2026-10-23。本文件只维护现状；详细数值以 [实验结果报告](EXPERIMENT_RESULTS.md) 为唯一解释入口。

## 已完成

- 当前研究协议 hiltl-product-v3：真实 LDBA 图、guard 编码、双头 TD/MC、进展门控、历史状态与 Options/SMDP 已实现。gnn_single/gnn_track 为核心匹配对照；旧序列课程训练路径保留。
- 精简批次 20261021-zone-minimal-v3 完成51阶段、25模型、约5M预算和6900评估回合；所有阶段状态及产物通过审计。
- 独立检查25检查点、冻结实现/规格哈希、模型与优化器数值、逐回合九项公式统计及汇总重生成。actual_checkpoint_steps=5,011,670。
- 轻量证据发布在 results/reproduction/20261021-zone-minimal-v3，完整模型与源码快照留在服务器 experiments/。GitHub不等于模型异地备份。

## 结论与缺口

ZoneEnv主环境未支持改进优于参考：hv_track三个种子的主macro均零；gnn_track均值9.11%，gnn_single12.00%，baseline27.78%。Flat/Letter只有单种子局部收益，同时有指标冲突；不能写成稳定跨环境优势。工程测试通过不能替代效果实验。

三种子预算与50回合是精简初步证据，不是原论文大规模课程/五种子/大评估的复制。单种子模块不能作强归因。循环任务仅报告有限时域代理。外部LTL2Action/GCRL-LTL为原论文引用数据，不是同协议重跑。

下一步优先诊断Zone奖励、命题访问、策略动作、HV目标和门控、Options终止，而不是重启大矩阵。诊断结果出来前不承诺模型必有提升。[验证路线](CODE_VALIDATION_ROADMAP.md)列出顺序与验收条件。

## 论文与目录

网站长稿overleaf/TLT保持原字节；Copy短稿为前轮授权修订工作稿，本轮未修改。原短稿在paper/archive/pre-revision。论文实验文字仍需与本轮负结果及限制同步，不自动成为可投稿终稿。

核心阅读路径：README → EXPERIMENT_RESULTS → CODE_VALIDATION_ROADMAP。运行/监视查LONG_RUNNING_TASKS；论文路径查paper/README；历史实施与检查查maintenance/RECORDS。不新增重复的状态报告。

## 环境和历史

所有命令使用远端wty_LTL、PYTHONNOUSERSITE=1；定制Safety Gymnasium源码为运行依赖，不可清理掉。pygame版本约束仍非完全一致，不宣称环境完全锁定。早前195模型及31模型计划已停用，保留原始记录与导入模型需要的源码路径，不能按旧文档恢复长训。

本轮清理只处理一次性维护脚本、重复Git bundle、可重建LaTeX缓存和旧中间备份；完整已完成实验、历史科研结果、论文源文件及用户.vscode设置保留。删除清单见 maintenance/cleanup-20261023.json。
