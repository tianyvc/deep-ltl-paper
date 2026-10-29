# 验证与证据边界

更新：2026-10-29。检查均在SSH服务器conda `wty_LTL`执行，未安装/变更依赖；环境变量见README。

## 1. 当前检查

|检查|结果|证据|
|---|---|---|
|`tests/test_curriculum_transition.py`|跨阶段结果隔离、合法统计与阈值推进通过|[test-0.log](../results/validation/current/test-0.log)|
|`tests/test_tlt_curriculum.py`|15项合同通过：奖励、MC、梯度、更新、缓存、严格加载等|[test-1.log](../results/validation/current/test-1.log)|
|`tests/test_paper_mapping.py`|三环境15任务转换、特征及六配置等通过|[mapping.json](../results/validation/current/mapping.json)|
|`scripts/verify_tlt_upstream.py`|固定上游构建器兼容检查通过|[upstream.json](../results/validation/current/upstream.json)|
|`tests/test_runtime.py`|9/9微型训练返回0，参数有限并成功保存|[runtime.json](../results/validation/current/runtime.json)|
|LaTeX/BibTeX四步编译|13页，40条引用，无悬空引用、重复label、Overfull|[paper-build.json](../results/submission/paper-build.json)|

执行命令与返回码见 [tests.json](../results/validation/current/tests.json)。runtime覆盖Zone baseline/hv_track各128步、FlatWorld六配置各32步、Letter hv_track32步，每组两次更新，不是论文性能实验。

作者PDF `TLT.pdf` 与重新编译PDF均13页，提取文本完全一致；不等于所有图片的PDF像素级比较。仅有两处 `h` → `ht` 浮动警告。终稿所有文件哈希见 [final-manifest.json](../results/submission/final-manifest.json)，本次未修改终稿。编译中间产物留在外部档案，避免重复PDF和日志进入终稿包。

## 2. 先导实验：保留负面结果

2026-10-25的12组短训、360个完整有限评估回合和90回合冻结actor搜索诊断，见 [report.json](../results/validation/pilot-20261025/report.json)。训练seed1/2/3，每任务每seed评估10回合；Zone/Letter、baseline/hv_track，每组目标32768步、360秒训练软上限。

|环境|seed|baseline步数|hv_track步数|baseline SR|hv_track SR|
|---|---:|---:|---:|---:|---:|
|Zone|1|32768|32768|3.33%|0.00%|
|Zone|2|32768|32768|3.33%|0.00%|
|Zone|3|32768|32768|10.00%|0.00%|
|Letter|1|32768|30720|56.67%|60.00%|
|Letter|2|32768|32768|76.67%|76.67%|
|Letter|3|32768|24576|73.33%|56.67%|

Zone等步数baseline 5/90、hv_track 0/90，没有正向提升信号；独立训练单位为3个seed，不能把90回合当90次训练。Letter两组预算不等，描述性平均68.89%和64.44%不能作为严格等预算差异；唯一等步数seed2均76.67%。早期模型不能验证或推翻历史成熟模型结果。

首轮Zone hv_track seed2在26624步触发课程统计异常。修复跨阶段污染后全部12组从头重跑；失败和修复前后证据见 [curriculum-regression.json](../results/validation/pilot-20261025/curriculum-regression.json)。没有删除失败seed或负面结果。

冻结actor、仅换搜索排序critic后，Zone hv_track各seed仍0/30；换回base搜索值没有修复该批模型，但不能证明critic完全无影响。正式长训前应找回成熟原模型与词表，先验证冻结基线，再进行固定预算和seed对照。

## 3. 文献与原始数据

40个引用key均有BibTeX条目并进入编译。上一轮逐条联网记录保留在 [reference-live.json](../results/submission/reference-live.json)、[reference-extra.json](../results/submission/reference-extra.json)：多数有出版社、会议录、Crossref或arXiv元数据；Baier书籍官网403、DeepLTL OpenReview为浏览器验证页，另有补充来源。HTTP 200不等于条目正确，也不等于文献支持正文的具体论断；本轮未把旧访问记录表述为重新逐篇核读。

`results/reference/finite/`、`infinite/` 的8份原始CSV逐字节保留，来源和哈希见 [provenance.json](../results/reference/provenance.json)。外部基线未在当前管线重跑，旧CSV、短测和终稿主表不得混为同一批实验。

程序执行、核心梯度、保存和编译正常，不代表终稿99.3%成功率或6.4%相对增益已复现。覆盖差异见[方法文档](METHOD_AND_IMPLEMENTATION.md)，投稿问题见[投稿指南](SUBMISSION_GUIDE.md)。历史JSON内路径和哈希保留原样，归档恢复见[工作区维护](WORKSPACE.md)。
