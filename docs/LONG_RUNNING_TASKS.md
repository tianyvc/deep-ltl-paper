# 实验运行、监视与归档手册

更新：2026-10-23。当前精简批次已经完成，不需要继续等待，也不要重启旧大队列。研究结论以 [结果报告](EXPERIMENT_RESULTS.md) 为准。

## 本地最简单命令

在工作区PowerShell执行：

```powershell
.\scripts\check-training.ps1 -SampleSeconds 0
```

完整等价命令：

```powershell
cf.exe run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL env PYTHONNOUSERSITE=1 python scripts/check_paper_campaign.py --sample-seconds 0
```

使用cf.exe避免cf.ps1解析丢失`--`。普通运行中可省略SampleSeconds，工具采样10秒比较worker活动；检查是只读的，不会启动/暂停训练。

## 输出怎么判断

|字段|含义与本轮完成状态|
|---|---|
|Campaign|completed 为调度完成；running 只是调度状态，不等于学会|
|scheduler_alive|完成后False正常；运行中False需排查|
|heartbeat_age|完成后不断变大正常；运行中超过120秒需排查|
|Tasks|本轮completed=51，running/pending/failed/blocked均0|
|Formal training|25/25；导入的2个检查点计入|
|Training budget|5M/5M，remaining=0；按每run上限封顶，实际完整回合略超|
|Evaluation budget|6900/6900，remaining=0|
|Dispatch deadline|仅停止新派发的上限，不是完成ETA；完成后无需关注|
|ALL_COMPLETED|调度产物检查通过，可以做结果审查；不是证明性能提升|
|CHECK_REQUIRED|检查失败/阻塞、失联worker或缺少产物，按日志处理|

运行中判断进展应比较两次steps/updates或episodes是否增加，结合CPU_ticks_delta、child_alive、progress_age、loss/gradient有限性；GPU利用率低不必然异常，MuJoCo和公式评估可能CPU密集。单任务ETA不含排队与后续评估，任务计数百分比不等于剩余墙钟时间。

本轮51阶段从北京时间10/21 15:50至10/23 00:59，约33小时9分。当前剩余任务和时间均为0。独立结果审计另检查了25检查点、6900回合统计及汇总重算。

## 真实计划与输出

本轮manifest为experiments/reproduction/20261021-zone-minimal-v3/manifest.json。主环境Zone19模型（核心四因子与baseline各3种子、其余模块单种子）；Flat4、Letter2模型，约200k/run，25训练+25评估+1汇总。每公式50回合；主表phi1/2/3，循环单列，Zone额外sequence3/original_phi7。精确任务与预算以manifest/specs为准，不按历史大队列文档再跑。

|位置|内容|
|---|---|
|experiments/reproduction/20261021-zone-minimal-v3/runs/<run>/status.pt|完整checkpoint、参数与优化器状态|
|同目录config.json/metrics.jsonl/progress.json|配置、训练更新曲线与完成进度|
|evaluation-minimal.json|逐回合结果与公式汇总|
|jobs/<task>-attempt-*/output.log|任务stdout/stderr与故障诊断|
|manifest/state/events|冻结计划、调度状态与事件|
|source/|冻结源码，恢复/溯源依赖，不删除|
|summary.json/csv|按训练seed汇总，不代替原始回合|
|results/reproduction/20261021-zone-minimal-v3/|已发布轻量证据，不含模型|

早前20261021-v3与20261021-minimal-v3（如存在）为停用历史计划；源快照与部分检查点仍有导入依赖，保留但不要恢复调度。确切原始批次名称以experiments/reproduction下目录为准。

## 审计与重算（远端wty_LTL）

```bash
PYTHONNOUSERSITE=1 python scripts/audit_paper_results.py
PYTHONNOUSERSITE=1 python scripts/summarize_paper_minimal.py \
  --manifest experiments/reproduction/20261021-zone-minimal-v3/manifest.json \
  --output /tmp/hiltl-review-new/summary.json
```

输出文件必须不存在，避免覆盖证据。audit完整模型检查需原服务器路径；发布包可校验SHA清单/原始回合，但没有模型就不能重新做策略推理。导入模型原始配置与归档配置差异记录在audit.json。

## 后续后台任务原则与命令

本轮未启动新训练。先完成验证路线中的失败诊断，再锁定下一轮manifest、预算、源码hash与GPU资源，不能直接复用本批次输出目录或启动历史195/31模型计划。

未来冻结新manifest后的调度入口（模板，需要替换所有NEW路径，非本轮已执行命令）：

```bash
nohup /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL \
  env PYTHONNOUSERSITE=1 python experiments/reproduction/NEW/source/scripts/paper_campaign.py \
  run --manifest experiments/reproduction/NEW/manifest.json \
  > experiments/reproduction/NEW/scheduler.log 2>&1 < /dev/null &
```

状态/暂停使用scripts/paper_campaign.py的status/pause子命令及同一个manifest；仅在明确需要时恢复，不kill其他服务器用户进程。GPU先用nvidia-smi确认空闲显存与现有进程，新批次不得抢占他人任务。断开本机不会终止nohup后台任务。

## 保存与恢复

服务器experiments保留全部完成数据、source与模型；GitHub只提交轻量证据及代码/文档。需要异地模型备份时另外复制完整批次并校验checkpoint SHA。记录里的绝对路径是原始溯源，不是跨机器无修改执行承诺。共享服务器不存个人GitHub token，个人电脑Git/GCM负责HTTPS发布。
