# HiLTL 论文统一入口

当前论文以用户保存的两个 **Overleaf 网站终稿快照**为准。长短两版均保留，不替用户选择唯一投稿版本；后续代码整理应先遵循其共同方法，再明确两版之间的歧义。

| 内容 | 位置 | 使用规则 |
|---|---|---|
| 网站长版 | [../overleaf/TLT/manuscript.tex](../overleaf/TLT/manuscript.tex) | IEEE journal；原文件禁止改动 |
| 网站短版 | [../overleaf/TLT__Copy_/manuscript.tex](../overleaf/TLT__Copy_/manuscript.tex) | IEEE conference；原文件禁止改动 |
| 长短差异与论文内部问题 | [review/FINAL_VERSION_COMPARISON.md](review/FINAL_VERSION_COMPARISON.md) | 章节、公式、实验、排版和待核对项 |
| 后续项目对齐依据 | [../docs/CODE_VALIDATION_ROADMAP.md](../docs/CODE_VALIDATION_ROADMAP.md) | 论文方法、配置、指标、代码缺口与优先顺序 |
| 验证与保护记录 | [../docs/maintenance/RECORDS.md](../docs/maintenance/RECORDS.md) | 原样编译与兼容性副本的结果分开报告 |
| 旧候选稿及旧文献库 | [archive/pre-overleaf/](archive/pre-overleaf/) | 原字节归档，退出当前稿入口 |
| 旧图片、变体与绘图输出 | [figures/](figures/) | 不用旧同名文件覆盖网站终稿图 |
| 绘图生成器 | [scripts/](scripts/) | 输出仍在 `paper/figures/`，本轮不改脚本路径 |
| 投稿准备材料 | [submission/](submission/) | cover letter / title page 历史材料，不代表已提交 |
| 样式模板 | [templates/](templates/) | 示例，不是正式论文 |
| 更早的中英文稿 | [../docs/archive/document-history.zip](../docs/archive/document-history.zip) | 历史资料 |

原始 DeepLTL PDF 保留在本目录。整合采取“统一导航、原稿保护、旧稿归档”，没有复制出第三套当前终稿。

## 资源与去重边界

两个网站目录各有 12 个文件，只有主 `.tex` 不同，其余 11 个文件逐字节相同。保护约束优先于跨快照去重，故各自保留完整资源。

旧 `figures/` 中多个同名图片与终稿实际内容不同，不能因文件名相同删除。`figure5_performance.png` 虽与网站图一致，但旧稿仍依赖它，保留以维持历史资源包。旧短稿与终稿短版只有资源引用差异，已归档，不再并列展示为独立当前稿。没有删除包含独特内容的旧稿、投稿材料或原始论文。

绘图脚本部分曲线来自硬编码或模拟数据，仅可用于示意，不能代替真实实验日志；也不能据其存在认定论文图表已复现。重新运行前检查输出，避免覆盖旧资源。

## 编译状态

本轮通过 `cf run` 在远端 `wty_LTL` 环境发起检查，所有 LaTeX 命令均在 `.maintenance/overleaf-review/` 的副本执行。

- 原样副本：首轮 pdflatex 和 BibTeX 可运行；后续 pdflatex 报 natbib author-year 与数字参考文献不兼容，完整构建未通过。
- 仅在副本将 `natbib` 改为 `[numbers]`：两版完整通过 `pdflatex → bibtex → pdflatex → pdflatex`，长 13 页、短 9 页。
- 两张环境图原名含 `.png.png`；本次 graphicx 能自动解析，没有因此缺图。跨工具导入时仍须关注命名，但本轮未改名。
- 不在受保护目录直接运行编译命令。具体日志、哈希和验证边界见 [维护与验证记录](../docs/maintenance/RECORDS.md)。

## 后续使用

先阅读差异报告与对齐文档。长版提供更多实现细节，短版澄清 `κ`、有限距离归一化和 Options 配置，但两版都有待核对之处；不能仅按“长版覆盖一切”或“短版最新所以全部正确”处理。任何修订应在用户后续明确授权的新工作副本中进行。本轮不修改算法实现、不改终稿、不提交或推送。


## 旧稿复现与历史访问

`archive/pre-overleaf/` 中三个文件保留原字节与原相对路径，不能直接从归档目录编译。历史复现应在 `.maintenance/` 建立隔离目录，复制所需 tex 与 references.bib，把 `paper/figures/` 复制为隔离目录下的 `figures/` 后再编译。旧 ICML 稿缺 icml2025.sty；不能据其缺样式或旧图问题判断网站终稿。旧短稿和网站短版正文相同，差别在 graphicspath、融合图文件名和 bibliography 入口，保留用于追溯旧图使用情况。

中英文 Markdown 旧稿与先前分散的导航/验证说明已原字节收入 `docs/archive/document-history.zip`，清单按原路径索引；更早的 tex 旧稿仍在 `docs/archive/papers/`。原始文件历史链接可能反映当时结构，不作为当前有效导航。
