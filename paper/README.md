# HiLTL 论文统一入口

更新：2026-10-23。本轮精简实验已完成，主环境没有证实性能提升，完整结果见 [EXPERIMENT_RESULTS.md](../docs/EXPERIMENT_RESULTS.md)。论文源码本轮未改，实验段待按真实结果同步。

此前2026-10-21修订说明：用户已明确授权在 `overleaf/TLT__Copy_` 修改短稿；它现在是修订工作稿，不再是网站原样快照。`overleaf/TLT` 长稿保持原样。原短稿tex/bib在修改前归档，可恢复和比较。

| 内容 | 位置 | 使用规则 |
|---|---|---|
| 当前修订短稿 | [manuscript.tex](../overleaf/TLT__Copy_/manuscript.tex) | IEEE conference；按真实实现重写，工作稿，实验段待同步本轮真实结果 |
| 修订短稿PDF | [HiLTL-short-revised.pdf](../docs/maintenance/validation-runs/20261021/short-revision/HiLTL-short-revised.pdf) | 本轮隔离构建的6页预览，含参考文献 |
| 网站长版 | [manuscript.tex](../overleaf/TLT/manuscript.tex) | 原journal快照，未修改；其旧主张不覆盖修订短稿 |
| 修改前短稿 | [short-manuscript-20261021.zip](archive/pre-revision/short-manuscript-20261021.zip) | 原tex/bib及全部原资源哈希；其余资源原位保留 |
| 历史比较、投稿审查与八点落实 | [FINAL_VERSION_COMPARISON.md](review/FINAL_VERSION_COMPARISON.md) | 第1–5节原快照比较，第6节审查，第7节修订记录 |
| 后续代码与实验验证 | [CODE_VALIDATION_ROADMAP.md](../docs/CODE_VALIDATION_ROADMAP.md) | 区分已落地七配置与尚需添加的公平对照 |
| 执行与构建记录 | [RECORDS.md](../docs/maintenance/RECORDS.md) | 命令、哈希、构建证据和剩余任务 |
| 旧候选稿 | [archive/pre-overleaf/](archive/pre-overleaf/) | 历史tex及文献库 |
| 图与生成器 | [figures/](figures/)、[scripts/](scripts/) | 旧示意/硬编码资产不能充当新实验结果 |
| 投稿历史材料 | [submission/](submission/)、[templates/](templates/) | 不代表当前已投稿或符合某会议模板 |
| 更早Markdown资料 | [document-history.zip](../docs/archive/document-history.zip) | 历史原字节快照 |

## 当前修订范围

短稿主线为 **Progress-Conditioned Dual-Target Critics**：明确TD/MC监督、有限时域回报代理、固定门控、guard编码、接受转移、历史与Options/SMDP。删除skip有效类别、精确概率分解、GNN层数等同物理时间尺度等不成立表述。纠正DeepLTL架构与GCRL-LTL引用；外部方法尚未本仓库重跑，不再以不同任务/预算的引用数字作同协议优越性结论。

现阶段只有实现验证，无足以支撑论文性能提升的正式实验。原13张性能及重复表、旧性能/融合曲线均退出修订稿，改为2张诊断任务/工程检查表和1张LaTeX结构图。旧资源未删除，以保留历史；修订稿不引用这些旧图。

匹配单critic/history对照与独立validation已在v3落实，精简多种子实验已完成；结果没有证实主环境提升。大预算确认与外部同协议对照仍未完成。论文明确将其列为待完成，不把计划写成实验事实。当前稿不是可以省略正式实验直接投稿的终稿。

## 编译

本轮全部项目命令经cf在远端conda `wty_LTL` 执行，不安装依赖。源短稿已使用numbers模式natbib；隔离目录复制tex/bib后执行 `pdflatex → bibtex → pdflatex → pdflatex`，不需要旧图或路径别名。PDF为6页，最终无缺失引用/未定义标签、无Overfull；少量Underfull行排版提示保留并记录。

编译输出和精确日志位于 `docs/maintenance/validation-runs/20261021/short-revision/`，临时构建缓存已清理；持久PDF与日志仍在上述validation目录。不在Overleaf源码目录写入编译产物。历史长13页/短9页是此前兼容副本结果，不能与当前修订稿混淆。

## 历史边界

原两目录各12文件、只有tex不同的结论仅适用于修订前。现在短稿tex和ref.bib均已更新，其余10个图资源及长稿保持原字节。旧稿复现时在隔离目录恢复对应tex、bib及资源；不要覆盖当前工作稿。原始DeepLTL PDF继续留在本目录。
