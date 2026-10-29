# 工作区维护与恢复

## 1. 目录职责

论文只以 `overleaf/HiLTL-final/` 为基准；投稿适配从它新建目标版本，避免覆盖作者确认稿。`src/`、`configurations/`、`scripts/`、`tests/` 为维护主线；`eval_datasets/`、`results/reference/` 为原始资源。`results/validation/` 存紧凑运行证据，`results/submission/` 存终稿哈希、编译结果、投稿官网快照和文献核查证据。

只维护方法、验证、投稿、本说明四份项目文档，README负责入口和使用；排查不再新增长期MD。AGENTS.md与第三方README/许可证为必要元信息。新实验放忽略的 `experiments/<唯一名称>/`，保留配置、seed、checkpoint哈希、逐回合和失败记录；精选紧凑证据再进入results，不提交权重、缓存或成套编译中间文件。

## 2. 可恢复归档

本次档案在仓库**同级** `deep-ltl-paper-archive-20261029-submission/`，远端路径：

```text
/home/dell/workspace/Tianyvc/deep-ltl-paper-archive-20261029-submission/
  manifest.json                 # 原路径、目标、SHA256、大小
  protected-before.json        # 终稿、原评估资源及CSV保护清单
  working-tree-before.patch    # 整理前已跟踪文件未提交补丁
  status-before.txt
  snapshot/                    # 原README及未提交源码补充快照
  files/                       # 按原相对路径保存的旧稿、文档、实验与缓存
```

移出：旧 `overleaf/TLT/`、`overleaf/HiLTL-new/`、原docs审校历史、旧review及已提取关键证据的light-validation结果、旧campaign/probe、experiments历史模型和日志、缓存。旧 `TLT__Copy_` 此前已删除，未恢复。范围和manifest哈希见 [archive.json](../results/validation/archive.json)。Git历史未重写，其他日期的同级档案未清除。

档案不进入GitHub，新克隆只含活动代码与共享证据。恢复时在 `files/<原路径>` 查找，核对manifest SHA256后复制到新目录，避免覆盖当前版本；历史JSON中的experiments路径也如此恢复。自动机jar、定制环境mesh/XML/纹理及许可证为运行资源，已保留。135份保护文件（终稿、原评估资源及8份CSV）整理前后哈希一致。

## 3. 发布与认证

构建、检查、Git提交在远端 `wty_LTL`；推送通过本地Windows Git和既有Credential Manager：远端生成提交bundle，本地临时bare仓库导入并正常推送main，再核对GitHub SHA。禁止force覆盖他人提交。

服务器仅保留无凭证的 `https://github.com/tianyvc/deep-ltl-paper.git` URL，不新增token、credential helper或凭证文件，不复制本地凭证到服务器。发布完成后移除临时发布仓库与bundle，服务器origin/main按已确认GitHub SHA同步。

公开实名仓库与双盲稿分别处理：PAKDD审稿稿及附件避免实名GitHub URL和作者标识，需要代码时另准备匿名材料。公开仓库不自动成为双盲合规附件。
