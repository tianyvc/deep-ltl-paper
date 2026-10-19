# 论文工作区

## 版本关系

| 文件 | 内容与用途 | 状态 |
|---|---|---|
| paper.tex | HiLTL，ICML 风格，含较完整附录 | 候选稿；缺 icml2025.sty |
| paper-new.tex | HiLTL，IEEE conference，合并调整后的后续稿 | 候选稿；参考文献入口已改为 references |
| references.bib | 当前共享参考文献库 | 本轮未逐条在线核验真实性和完整性 |
| templates/IEEE_journal_example.tex | bare_jrnl 示例 | 模板，不是期刊终稿 |
| submission/ | TNNLS cover letter 和 title page 原材料 | 投稿准备记录，未发送 |
| ../docs/archive/papers/ | DeepLTL-LS 等中英文早期草稿 | 历史，不覆盖当前候选稿 |
| 原始 DeepLTL PDF | 原论文材料 | 保留原文件名 |

整理只修正资源定位和 bibliography 文件名，不擅自决定哪份是正式终稿。

## 图片与生成

全部 PNG/PDF/SVG 集中在 [figures/](figures/)。不同内容的同名图片保留变体，相同 SHA-256 的增强自动机图合并为一份。原位置映射见 [figure-map.json](../docs/maintenance/figure-map.json)。

生成器位于 [scripts/](scripts/)，输出定位到 figures。生成器中部分性能曲线与权重曲线使用硬编码或模拟数据，只能作为设计示意；没有追溯到真实日志前不能用于实验论证。重新生成可能覆盖同名图片，应先检查脚本。

目前全仓缺少 `zoneenv_gFblue_yellow.png` 和 `zoneenv_FGgreen.png`。本轮没有虚构或替换缺失实验图。

## 编译

从本目录执行，例如：

```bash
latexmk -pdf paper-new.tex
latexmk -pdf paper.tex
```

需要完整 LaTeX 依赖、文献工具、图片和所用会议样式。IEEEtran.cls 在本次服务器中可找到，icml2025.sty 找不到。具体尝试和失败信息以[验证报告](../docs/maintenance/VALIDATION.md)为准；本轮不声称论文已完整编译通过。

## 内容审查重点

- DeepLTL 原方法为 DeepSets/RNN 序列编码，不能将其描述为原本采用本项目 GNN。
- “+6.4%”、其他百分比与加速倍数必须补足可复现实验、种子、任务、原始日志和统计方法。
- 主 PPO 路径尚未完整使用层次价值、自动机追踪和多尺度辅助损失，论文算法陈述应与代码实际状态对齐。
- 有限时域 accepting visits 不能直接等同于无限时间满足概率的数学保证。
- 历史稿之间的贡献命名、基线和符号尚需作者统一。
