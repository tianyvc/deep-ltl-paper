# HiLTL 终稿评估与投稿指南

核查日期：2026-10-29；对象：`overleaf/HiLTL-final/LTL/manuscript.tex`、`ref.bib`、图片和随附 `TLT.pdf`。下文行号均指该原始tex；本次只评估、整理文档，没有修改作者确认的终稿。官方页面快照及失败访问记录见 [sources.json](../results/submission/sources.json)，提交前请重新核对官网与投稿系统。主题适配判断和修改建议属于本仓库评估，不是主办方的录用承诺。

## 1. 是否基本达到可尝试投稿水平

**已形成完整、可以进入同行评审的研究稿，但不建议将当前PDF原样立即提交。** 文章有明确问题、方法、算法、跨环境均值和消融，核心术语比旧稿一致；区分了折扣回报与满足概率、有限窗口行为与无限性质、内部对照与历史外部参考。重新编译通过，13页、40条引用，无悬空引用、重复label或版面溢出，随附PDF与新编译PDF的提取文本一致。

目前更大的风险来自实验口径、方法贡献证据以及目标范围，不能靠语言润色解决。用户已核实的历史结果仍作为论文结果保留；现有包中若缺少可供审稿人理解的批次和分母解释，仍需补齐。本轮微型训练通过不等于重现99.3% SR。先导实验与实现差异见 [验证](VALIDATION.md)、[方法与实现](METHOD_AND_IMPLEMENTATION.md)。

|维度|判断|含义|
|---|---|---|
|结构和可读性|基本达到研究稿水平|方法、实验、算法齐备；仍可压缩重复与展望|
|创新论证|有可评审想法，证据尚偏窄|应突出TD/MC与进展融合；当前MSGNN组合不稳定，不宜宣传全组件普遍更优|
|实验可信度与可解释性|需优先补齐|聚合/分任务口径、seed角色、误差和历史产物未完整连接|
|可复现性|部分达到|六配置可运行；不能称终稿全部实验、Options及s输入长头已复现|
|IEEE Systems Journal适配|偏低，当前有页数硬问题|不仅是格式问题，范围有明显风险|
|PAKDD Research适配|较好，但需格式和政策处理|12页LNCS、双盲、AI使用范围是提交前关卡|

## 2. 投稿前优先解决的具体位置

### 2.1 图5的两种口径必须能追溯

位置：tex第40、343–346、364–374、393、468–474、528行；图文件 `figure5_performance.png`。

主表0.933→0.993对应相对增益 `(0.993/0.933-1)*100=6.4309%`，写6.4%本身正确。图右侧标注是1.2%、2.9%、4.3%。**若三个任务、方法、基线、批次和权重相同，聚合相对增益应是各任务相对增益的非负加权平均，不能超过4.3%。** 用右侧可读柱高约baseline=.988/.970/.940、hv_track=1/.998/.980估算，三任务均值约.966/.993，相对增益约2.76%；这是图像读数诊断，不是可代替原始数据的精确新实验结果。

第393行已说明两个panel分别报告，但仍没说明“分别”的具体数据来源。正文任务集目前只明确三个模板，审稿人仍会自然把它们理解为同一评估集。需要采取一种有证据的处理：

1. 若确为不同批次/任务集：给两个panel各自的任务集合、训练checkpoint、seed、每任务回合数、分母、权重和聚合脚本；表题与图注同步标注批次。只写“不能直接平均”不足以解释。
2. 若实际为同一批：从已核实逐seed/逐任务结果重建图表，并更新摘要、结论及所有派生值；不要把估读的2.76%直接写成正式结果。
3. 若无法恢复可核查口径：删去有冲突的一组图示/数字及对应结论，保留可追溯的一组。不能为保留所有数值编造隐藏任务或新批次。

当前不提供含未知批次的“可直接替换”图注，避免把待核信息写成事实。可先用原始结果制作 `task × training_seed × reset_seed × success × steps × reward × checkpoint_sha` 明细和聚合输出，再填写最终图注。

### 2.2 门控图的端点仍与公式不一致

位置：tex第285–291、624行与 `figure2_fusion.pdf` 图内β≈0.1/0.9标注。

正文 `beta=sigmoid(3*(p-.5))` 的端点0.1824/0.8176正确。保留该固定规则时，应将图内两端标签改为 **β≈0.182 / β≈0.818**，中点保留0.5。不要仅因图内近似标签把公式改成另一参数，也不要把图7趋势曲线正确当作图2标签已正确。此次未改原图。

### 2.3 seed、分母、奖励与训练设置

位置：tex第298、330、332、461、483–498行。

- 明确1–5是独立训练seed，还是同一模型的评估reset seed；二者不能互换。训练seed内的500回合也不是500个独立训练重复。
- 明确500是“每seed每任务”还是“每seed在全部任务上合计”，任务如何分配、环境如何重置；说明确定性/采样动作、超时和中断处理。
- Avg Steps究竟仅成功回合、所有完整回合，还是成功完成时间？失败/超时如何计入？Avg Reward是环境累计回报、逻辑回报还是折扣回报？其平均分母需与表题一致。
- 给独立训练seed的均值与离散度/适当区间，避免用大量相关episode替代训练重复。若只有一模型，诚实报告该局限，不能补造误差条。
- 明确总采样步数、并行数、rollout、epoch、batch、gamma、PPO ratio clip、额外advantage clip、checkpoint选择规则。1M预算与成熟模型、是否warm-start的关系需可追溯。
- 表3历史5/12/18与当前4/8/5已有区别说明，但需历史HOA或转换器版本、参数、合法赋值裁剪配置；如不能提供，可删除无关状态数行，不删除公式任务本身。

### 2.4 外部基线与核心组件证据

位置：tex第279、302、370–388、461–474、503–521、680–692行。

外部LTL2Action/GCRL-LTL已明确为不同任务协议的历史参考，这是必要改进；其数值差不能作为同协议SOTA胜出证据。期刊扩展时建议同任务、同预算重跑，或把它们放独立背景表，主结论依赖内部受控比较。

`hv_track` 最优而 `full_no_options` 低于baseline，适合围绕价值学习写论文。SE-GNN独立特征贡献尚未被当前配置表隔离；“关闭MSGNN”不是“去掉语义图”。建议至少增加语义特征有/无、history有/无、固定常数融合与进展融合、单头容量匹配等对照，并报告参数/训练时间/推理成本。它们是增强说服力的建议，不能假称为所有venue明文必做项。

Options仅保留历史行，不能解释其改进来自已完整定义的option机制。压缩会议版时，可移除该历史行和相关非受控比较；保留六配置及明确方法。代码长头不含s、无额外advantage clip的差异见方法文档，不能把仓库链接包装成终稿逐字复现。

### 2.5 图与附录的精简

位置：tex第100、173图注、439、553–670、673–692行。

自动机图是局部示意，正文已限定，仍应统一图内α/正文β、图内 `V_long(q)` /正文 `(s,q)`、Layer0/1/2与实际message passing深度的命名，避免给人“局部图等于φ3完整自动机”的印象。架构图dense/sparse等旧标签应直接对应TD/MC目标；示意图标签多套并存会增加审稿负担。

附录A算法有复现价值，建议保留图构建与训练算法中当前正文未交代的关键步骤，删重复解释。附录B目前主要是未来工作，不能补充已完成实验的证据；会议版建议删第673–692行整节，结论保留1–2句即可。第443–457行“贡献汇总”表与正文、主表重复，可删以节约版面。未使用的模板注释删掉不会减少PDF页数，真正压缩应针对正文、图表和附录内容，不能缩小字体或页边距规避上限。

## 3. IEEE Systems Journal

### 3.1 身份、官方来源与范围

用户提供的 [IEEE Xplore链接](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4267003) 对应 **IEEE Systems Journal**。

- [期刊主页及scope](https://ieeesystemscouncil.org/publication/ieee-systems-journal)
- [Instructions for Authors](https://ieeesystemscouncil.org/publication/ieee-systems-journal/instructions-for-authors)
- [Regular Issues CFP](https://ieeesystemscouncil.org/publication/iEEE-systems-journal/call-papers-regular-issues)
- [IEEE Author Portal](https://ieee.atyponrex.com/journal/ieee-sj)

官网定位为application-oriented、systems-level的复杂系统/系统之系统研究。它明确把“一个反馈控制系统，即使控制器很复杂”列为范围外；神经网络/模糊系统只有用于互联系统等问题才适合。单agent导航仿真加图编码和critic，目前难以体现独立实体之间的耦合、协作、互操作或系统级作用。

**不建议将它作为当前稿件首选。** 把标题和引言加入“systems”不能解决范围问题。若坚持该刊，需要真正加入互联/多agent或网络化系统任务、系统组件耦合与约束分析，测量系统级可靠性、安全或资源协调，并围绕该问题重写贡献与实验。这已是实质扩展，不是删一页即可。

### 3.2 要求与当前状态

|官方要求|当前状态|需要做什么|
|---|---|---|
|Regular paper每轮评审≤12页，最终≤12页|**不满足：当前13页**|压缩至少一页；恢复真实作者等也可能增页，须重新编译计数；不要假设附录/文献不算|
|IEEE journal双栏、10号Times字体及规范模板|已使用IEEEtran journal，基本具备|最终检查缩放图表后的字体、尺寸及可读性|
|英文，摘要约150–250词，自含、单段、无缩写/引文/展示公式|当前按空白分词110词，且含LTL/TD/MC/HiLTL等缩写|重写150–200词，可兼容PAKDD词数上限；先修实质口径再保留数字|
|至少5个keywords/index terms|当前6项，数量满足|按术语统一用词，系统中选对应分类|
|single-anonymous评审，评审人可见作者|当前Anonymous占位|第28–29行填真实作者/单位/通讯信息；不得伪填作者；核对全部作者贡献与授权|
|全体作者ORCID|仓库无法判断|提前为全部作者准备并绑定；系统填写所有作者及邮箱|
|PDF在线提交，英文、可读、材料完整|PDF可编译，其他材料待准备|作者信息、摘要、关键词、可选cover letter、必要补充材料|
|原创、独占投稿；扩展会议稿须详细说明新增|作者需核查发表/投稿历史|若先投PAKDD，不能同时在该刊审稿；若将来扩展会议稿须真实扩展并披露|
|图表清晰、字足够大、规范参考文献|技术引用检查通过，图内仍有口径/符号问题|修第2节问题；逐页人工看最终尺寸，不以编译通过代替图表合规|
|录用后源文件、独立图、作者简介和照片等|源码和原图具备，作者材料未准备|按生产阶段要求提交，当前PNG/PDF不自动代表满足所有最终图格式要求|

官网允许补充代码、数据和视频等，并要求给出编译/运行说明。不能据此推断可用补充材料绕开主文12页限制，主稿仍应独立可理解。

### 3.3 费用与时间

官网写regular前8个最终出版页免费，第9–12页每页或不足一页 **US$150强制超页费**。若最终12页，依该页面规则为US$600；传统订阅模式无OA费。**支付超页费不允许超过12页。**

该作者指南同页OA费用有两套金额：提交部分US$2,495，后部US$2,045，存在官方页面内部不一致。因此此处不把任一金额当成已确认的现行报价；选择OA前以IEEE当期价目/投稿系统及正式账单核实，OA费与超页费分开。

Regular issues未列统一会议式截止日；主页past special issue截止日不适用于常规投稿。页面给首轮平均约3个月、二轮约1.5个月，仅为官网描述，不是保证，也不应据此安排必定录用日期。

## 4. PAKDD 2027

### 4.1 官方入口与时点

- [主页](https://pakdd2027.org/)
- [Research Track CFP](https://pakdd2027.org/pages/calls/research)
- [Author's Kit](https://pakdd2027.org/pages/calls/authors-kit)
- [Important Dates](https://pakdd2027.org/pages/dates)
- [CMT投稿](https://cmt3.research.microsoft.com/PAKDD2027)
- [Springer LNCS/LNAI模板与指南](https://link.springer.com/series/558/information-for-authors-and-editors)

|事件|官网日期|北京时间（官网23:59 AoE折算）|
|---|---|---|
|投稿截止|2026-11-20 23:59 AoE|**2026-11-21 19:59**|
|录用通知|2027-02-26|若按全站AoE规则，为2027-02-27 19:59；通知实际发布时间以邮件为准|
|Camera-ready|2027-03-19 23:59 AoE|2027-03-20 19:59|
|会议|2027-06-29至07-02|新西兰Wellington，官网为当地会议日期|

2026-10-29距官网投稿日期约三周。会议**仅线下**，至少一名作者须regular registration并到场报告；no-show论文不进入会议录。注册费用、签证和具体报告安排当前不能从已核页面确认。

### 4.2 Track选择

Research CFP明确包含reinforcement learning and bandits、Graph NNs、neuro-symbolic learning and reasoning，当前论文匹配。建议 **Research Track**。

[Applied Data Science Track](https://pakdd2027.org/pages/calls/applied) 强调现实应用、操作约束、验证和领域影响；虽然允许现实实验室验证，当前主要仿真RL基准还不足以支撑该叙事。不要因有RL“agent”就投LLM and Agentic AI专项，当前工作没有相应LLM/数据科学agent贡献。

### 4.3 硬要求与处理

|要求|当前状态|处理|
|---|---|---|
|英文PDF、Springer LNCS/LNAI模板|英文PDF已有，**模板不满足**|从终稿新建会议适配稿，不覆盖HiLTL-final；采用官方模板|
|Research全文≤12页，包含references、appendices等|当前13页IEEE，**不能据此推算LNCS页数**|换单栏模板后重新编译；通常需要实质精简，不是只删IEEE一页|
|摘要≤200词|当前长度低于上限|维持≤200，准确表述；期刊适配可取150–200交集|
|double-blind，移除身份信息，自引用第三人称|当前作者匿名，但材料需继续检查|清PDF metadata、致谢、机构、实名仓库/附件路径；保持相关自引，避免“our prior work”|
|提交后作者列表和顺序不可改|目前匿名占位不代表系统作者名单已确定|提交前全体作者确认名单、顺序、贡献，系统填写真实信息|
|不得一稿多投；评审期间不得投其他正式出版venue|尚未提交|PAKDD与IEEE期刊择一；若转换目标，先完成撤稿/拒稿流程|
|生成式AI须披露且只能语法纠错|**真实使用范围需核对，不能直接判合规**|见下一节，不能在表中失实声明“仅语法纠错”|
|至少一位作者正常注册并现场报告|作者行程/预算未知|投稿前确认可承担Wellington差旅与注册，录用后按要求安排|

Author's Kit称不合规稿件会without review拒绝。arXiv预印本本身不违反重复投稿规则，但该页附加“as long as the submitted paper does not cite them”；结合双盲应避免引用自己的对应预印本暴露身份，如对具体自引情形有疑问由作者向PC核实，不能泛化为不准引用任何他人arXiv论文。

公开实名GitHub仓库此次按授权正常整理推送；会议审稿PDF不能因此直接加入实名URL。准备匿名附件/匿名代码链接前，应去除用户名、Git元数据、日志路径和文档中的作者信息，并按会议平台实际规则处理；也不要保证公开过仓库就必然不能双盲投稿。

### 4.4 生成式AI政策是实质问题

官网原文：

> The use of generative AI models must be explicitly disclosed and explained in the submission form.
>
> Such tools do not qualify as authors and may only be employed for grammatical error correction.

此前审校涉及英文替换段落、结构与论证建议等，可能超出纯语法纠错；本报告不替作者认定所有建议已经采纳，也不认为后续人工改写就能自动消除历史使用范围的问题。

提交前由作者核对实际采纳内容和使用过程，准备真实说明；若超出官方允许范围，不能直接按“仅语法纠错”提交，需由作者向Research PC（官网 `pc@pakdd2027.org`）询问适用边界或改选允许该使用方式且如实披露的venue。此次没有代发邮件，也没有在任何投稿平台提交材料。

### 4.5 建议的12页会议版结构

以下为排版预算建议，非会议强制章节配额；最终总页数含参考文献不超过12：

|内容|建议页数|修改方向|
|---|---:|---|
|引言+相关工作|约1.5|凝练研究缺口和与DeepLTL的差别，删泛化背景|
|形式化与方法|约3|统一语义图、TD/MC、固定门控；合并重复定义|
|实验设置+结果|约4|保留内部主对照、跨环境及关键消融，先解决聚合口径|
|讨论+结论|约0.5|诚实说明小自动机、外部历史参考与复现边界|
|参考文献|约2|保留必要且真实引用，删与精简展望失去关联的条目|
|算法/机动空间|约1|只保留关键算法，避免附录复制正文|

优先删附录B未来方向、贡献重复表、无可追溯Options历史行、多个相似架构示意和重复性能解释；保留必要的目标定义与评估口径。轨迹拼图可压缩为一个有信息量的例子，不能裁图隐藏失败或改变数据含义。不要假设Research允许额外不计页补充PDF：已读取页面没有给出可据以绕过12页限制的承诺。

## 5. 其他目标建议

以下按研究内容适配和改动类型推荐，不按未经核实的分区、录用率或“容易中”排序。费用、篇幅、AI及审稿匿名政策应逐刊复核；未读取到官方细则的地方明确列出，不用往年惯例代替当年CFP。

|目标|适配与推荐程度|当前最需要的改动|官方入口与本轮核查|
|---|---|---|---|
|**Neurocomputing**|优先考察的期刊；GNN+强化学习+价值结构方向自然|增强容量匹配消融、统计与成本；聚焦HV，降低MSGNN普适贡献表述|[主页](https://www.sciencedirect.com/journal/neurocomputing)、[作者指南](https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors)；本轮403，未确认最新页数/费用/AI条款|
|**Neural Computing and Applications**|值得优先考察；神经计算方法与应用叙事相对自然|给清楚的算法增量、可信对照与稳健性；补任务复杂度和运行成本|[Scope](https://link.springer.com/journal/521/aims-and-scope)、[指南](https://link.springer.com/journal/521/submission-guidelines)；scope已读取，指南提交前逐项复核|
|**Machine Learning**|方法强化后的挑战目标，非当前稳妥替代|需要更普遍的TD/MC融合洞见、更广任务、强基线与统计，解释何时失败|[Scope](https://link.springer.com/journal/10994/aims-and-scope)、[指南](https://link.springer.com/journal/10994/submission-guidelines)；scope已读取|
|**AAMAS**（以正式可用年份CFP为准）|研究主题贴近逻辑任务与自主agent，可作为会议方向|突出任务满足/泛化、同协议LTL-RL基线与agent行为；采用该届模板和匿名规则|[IFAAMAS](https://www.ifaamas.org/)；主页可读，本轮未确认2027正式CFP/日期，不提供猜测截止日|
|**Engineering Applications of Artificial Intelligence**|若补真实工程应用后可考虑，当前优先级较低|增加有现实约束/工程效益的案例，不能只把仿真导航换称工程应用|[Scope](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/about/aims-and-scope)、[指南](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors)；本轮403，细则待核|

若希望近期尝试会议，PAKDD Research主题匹配优于Systems Journal，但应先解决AI政策及LNCS12页问题。若希望保留较完整算法/实验展开，优先考察Neurocomputing、Neural Computing and Applications，仍需修复证据口径并核对各刊官方政策。若目标是Systems Journal，建议另做真正系统级互联/多agent扩展再考虑。

## 6. 实际提交顺序

1. 先统一图5口径、门控图标注、seed/指标分母和历史产物来源；无法证明的内容删减或诚实限定，不估填实验值。
2. 确定唯一在审目标；若选PAKDD，先核清真实AI使用范围和政策适用性，再投入会议排版。
3. 新建目标稿，按对应模板、作者/匿名要求和页数重新编译；不改保存的HiLTL-final基准包。
4. 逐页人工看图字、公式、表格、引用、PDF属性；检查作者列表、ORCID或CMT信息、注册/差旅承诺、版权与利益冲突等平台必填项。
5. 上传前再次访问本指南链接，确认deadline、页数和政策未变；提交后保存确认邮件、稿件ID及提交版本SHA，评审期间不并行投第二个正式出版目标。

本报告未实际投稿，未替作者确认贡献、签署版权、选择付费OA或承诺差旅。其余整理和GitHub发布不依赖上述投稿决定。
