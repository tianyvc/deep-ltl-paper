# 工作区整理与恢复

## 范围

本次以保留研究资产和运行路径为原则：src、第三方依赖、评估公式及结果 CSV 保持原路径；整理文档、论文图片、测试和杂项。未提交、推送或发布。

原仓库没有任何 Git 提交，所有现有文件均为未跟踪状态，不能依赖 Git 恢复。清理前先创建：

- `.maintenance/before-cleanup.tar.gz`：2,934 个原始文件，约 47.2 MB。
- `.maintenance/original-inventory.json`：原路径、字节数、SHA-256。
- 备份排除了 .git、缓存、临时检查文件和检查产生的 experiments；不是 Git 历史备份。

以上备份已加入 .gitignore，仅供本地恢复。重要研究资产仍建议另做独立存储备份。

## 整理内容

- 删除 __MACOSX、.DS_Store、过期 build、硬编码编辑器配置、引用缺失脚本的 test_speed.sh、空且未引用的 graph_builder.py。
- GNN/LongShort 技术文档、实验计划及四份根目录论文指南分别汇编；原文完整保留，历史失效链接不冒充当前可用入口。
- 不同论文草稿全部保留在 docs/archive/papers；旧配置集中到 docs/archive/config。
- 论文图片统一到 paper/figures，内容一致才去重；保留不同变体。
- 生成器集中到 paper/scripts；投稿材料归 submission，IEEE 空示例归 templates。
- 可运行 GNN 检查归 tests，旧接口及占位脚本归 tests/legacy；微基准归 scripts/benchmarks。
- parallel.log 有真实训练记录，保存为 results/logs/flatworld_gnn_test_seed5.log。
- 对移动脚本与论文资源路径作配套更新。

逐项操作见 [cleanup-manifest.json](cleanup-manifest.json)，图片映射见 [figure-map.json](figure-map.json)。原始文件内容可从备份取回。

## 必要代码修复

- 无正长度回合时 ARPS/ADR 返回 NaN；跳过零长度初始化占位，防止日志除零，不将未完成回合伪装成有效样本。
- checkpoint 失败分支使用已有 info 方法输出 WARNING，避免调用不存在的 warning。
- Rabinizer bin 恢复执行位。
- run_* 使用 sys.executable 并传播子进程失败。
- 大地图启动器新增 size 参数，仅接受 8/12/16/20，环境 ID 与注册保持一致。
- 主模型、PPO 优化公式和已有 CSV 未作算法性修改。

## 恢复方法

建议先在仓库外的独立目录解压并核对清单，再选择恢复文件，避免直接覆盖当前成果。例如 Linux：

```bash
mkdir -p /tmp/deep-ltl-before-cleanup
tar -xzf .maintenance/before-cleanup.tar.gz -C /tmp/deep-ltl-before-cleanup
```

结合清理清单恢复所需原始路径。恢复前保留当前新增文档；不要将解压操作当作自动安全合并。隐藏备份不属于论文附件或发布内容。
