# TLT 长稿指导下的阶段收尾总结

阶段：2026-10-23。目标是建立清楚、可运行、可逐项核查的实现基线；本次不重跑论文性能实验，不新增或修改论文中的结果数值。

## 1. 本阶段结论

当前维护的代码主线为`src/train/train_tlt.py`，协议`tlt-upstream-curriculum-v1`。它沿用原DeepLTL的序列课程、环境/逻辑奖励、DeepSets/GRU策略、原PPO与GAE，再接入长稿的语义图、多尺度表示、进展门控双目标价值与自动机历史。

三项核心方法都有可执行实现；六个维护配置经过短时运行检查。当前不能把整个长稿称为逐字、无例外的一一复现：长稿中的固定/学习门控、MC回报/概率解释、Options、自动机规模与若干实验设置存在冲突或缺项，下文明确列出。清理没有用删除限制说明或填入未经测量的提升值来制造“完全对应”。

本次未改动`overleaf/TLT/`与`overleaf/TLT__Copy_/`两个保存版本，也没有修改原评估数据。完整验证状态、源码与保护文件哈希统一在[verification.json](verification.json)；运行方式统一在[README](../README.md)，不再另存多份现状、路线和长任务手册。

## 2. 仓库整理结果

已删除过期产品训练器、旧多公式矩阵调度和监视器、旧Options产品实现、旧GNN/层级模型与适配器、未启用Transformer分支、无当前调用者的旧评估/分析/绘图脚本和测试、文档历史压缩包、论文改稿压缩包、重复源码快照、旧试运行模型和日志、重复论文图片/模板/投稿信，以及缓存和无效CMake配置。

本轮主要清理删除9,695个文件、959,425,253字节，约0.96GB；临时维护文件的最终移除另计。未做Git历史重写，已经进入历史提交的旧源码仍可追溯；旧的忽略目录模型文件按用户授权删除，不能再据旧路径重放。

|保留内容|目的与状态|
|---|---|
|长稿、短稿全部24个文件|原位保留、逐文件SHA不变|
|原评估资源114个文件|任务及世界布局不变；不是声称本次已经全部用于评估|
|原DeepLTL论文PDF|内容SHA不变，统一为`paper/DeepLTL-ICLR2025.pdf`|
|8份历史CSV|字节保持，归并至`results/reference/finite/`及`infinite/`|
|旧产品批次与4k短测关键汇总|保留在`results/reference/`，说明旧协议、来源与原产物已清理|
|当前模型/逻辑/环境代码与测试|维护与验证的唯一实现主线|
|Rabinizer启动器及owl.jar|实际调用的Java公式转换依赖；jar虽为归档格式，但不是无用压缩包|
|定制Safety-Gymnasium运行包、模型资源、安装信息与许可证|保证当前Zone环境和原注册依赖可运行；不删除运行期可能需要的原生资源|
|当前Zone短测检查点|只留服务器`experiments/smoke/`，不进入Git，不是成熟模型|

重复的项目MD已合并：工作说明只保留README与本总结；AGENTS协作指令、第三方安装所需README及许可证属于必要元信息。没有另建一个长期历史文档仓库。

## 3. 原模型路径与兼容方式

原仓库固定参照提交为`3157200b5910cd7f5c493ffba8913994f2207eaf`，来源为https://github.com/mathiasj33/deep-ltl。用于验证的官方模型构建器、配置和提交标识保留在`tests/fixtures/deepltl/`，因此离线检查不再依赖临时下载目录。

`DeepLTLSequenceModel`保留原环境编码器、DeepSets、GRU、actor与critic参数形状。Zone基线51,446参数，FlatWorld基线27,451参数。用同seed初始化，和固定提交的官方构建器在当前公共层上的参数键、逐张量值、动作分布与价值输出一致。这是受控构建器兼容检查，不是整个上游依赖树逐字节复现。

增强模型通过零初始化的图残差接入原actor embedding，新增图模块不会在加载原序列模型后立即把策略替换为随机新actor。短头保留原critic并叠加零初始化残差；融合价值仍会因长期头加入而变化。迁移模型要求status.pth及其vocab.pkl，严格加载，不自动吞掉不匹配参数。

训练继续调用原`make_env(...,sequence=True)`和`CurriculumSampler`。课程从简单reach逐步进入reach/avoid、序列与stay，按原成功条件推进，未替换为固定产品公式混合。物理墙体负奖励、逻辑违例与完成奖励由原SequenceWrapper保留；图上下文封装不改动作、reward、done、info。

## 4. 论文主张—代码对应表

|长稿内容|当前实现|如何对应|验证/范围|
|---|---|---|---|
|LTL转换与逻辑进展|`src/ltl/automata/`、`src/envs/ldba_wrapper.py`|调用Rabinizer；真实LDBA状态和接受转移；区分epsilon与物理动作|15个环境/任务组合转换通过；不人工凑自动机状态数|
|4维节点语义|`src/model/automaton_graph.py`|接受转移目标、循环SCC指示、初始节点、归一化接受距离|特征维数、范围及有限值检查通过|
|6维边语义|同上|真实边的4类one-hot槽、目标接受指示和目标距离；guard另作词元GRU编码|边索引、guard和特征行对齐；不增加不存在的skip边|
|SE-GNN骨干|`src/model/tlt_sequence.py`的encode_graph|3层GATv2，hidden64，4heads，边条件消息、残差MLP、LayerNorm和LeakyReLU(0.2)|前向及梯度合同通过|
|多尺度图表示|同文件的representations|对每层做attention pooling，多尺度配置拼接三层；普通配置只用末层|含当前节点的图输入宽度分别256与128，检查通过|
|短期价值头|components与short MLP|环境编码、图池化、当前q；两层256隐藏单元，原critic加残差|短头/公共层梯度与实际PPO更新通过|
|长期价值头|components与long MLP|proximity、当前q嵌入、可选history LSTM；两层128隐藏单元|按长稿q/history-only设计，不隐式加入物理状态|
|动态融合|components|beta=sigmoid(3*(proximity-0.5))；total=(1-beta)*short+beta*long|接受端/不可达端边界值验证通过；使用正文明确公式|
|短头TD与长头MC|`src/torch_ac/algos/tlt_ppo.py`|短头一步bootstrap；长头完整回合真实折扣回报，辅助权重1和0.3|完整回合MC进入优化、梯度非零；未完成回合不伪造0标签|
|PPO联合更新|TLTPPO与原ppo/base|融合价值用于GAE和PPO critic；另加两个辅助loss；保留原ratio clip|六配置训练更新、有限参数及保存通过|
|自动机追踪|`src/envs/tlt_context.py`与history LSTM|只在真实env step形成q历史，不可变快照；forward/minibatch不推进q|不变性、重放、缓存与完整历史等价检查通过|
|原课程训练图|sequence_monitor|把当前采样的reach/avoid序列构造成精确monitor，不更改课程任务|不能称为任意测试公式的完整LDBA|
|测试公式图与搜索|`scripts/evaluate_tlt.py`|使用实际测试公式LDBA、真实q历史与原ExhaustiveSearch，num_loops=2|搜索异常直接报错，避免静默用-inf掩盖实现错误|
|保存、迁移与恢复|`src/train/train_tlt.py`|数据化课程状态、优化器、随机状态、自有训练源码哈希、词表与严格参数加载|旧原模型迁移、课程序列化及实际短训保存通过|

图、GNN、双头与history模块都在实际forward和优化中使用，不是只存在于文件中的占位类。完整回合MC可跨rollout甚至策略更新后才结束；当前没有额外长期回放池，但也不能称整条MC轨迹都由完全相同策略采集。

## 5. 六个维护配置

|配置|HV|多尺度池化|历史追踪|维护状态|
|---|---|---|---|---|
|baseline|否|否|否|原DeepSets/GRU策略与单critic|
|hv_only|是|否|否|基础图表示、双头与固定进展门控|
|msgnn_only|否|是|否|多层图池化与单critic|
|hv_track|是|否|是|双头与逻辑历史，当前主方法配置|
|msgnn_track|否|是|是|多尺度与历史条件策略|
|full_no_options|是|是|是|三组件组合|

这些增强配置中的基础图编码用于表示q/guard，MSGNN开关特指跨层池化。不能把hv_only解释为完全不存在任何图表示。

full_all/Options不在当前维护集合。长稿只列出名称，没有给出足以唯一确定实现的启动集、终止条件、高层策略与训练loss；旧产品实现包含自定义SMDP训练，且使用已被替代的产品奖励路径，因此本次删去，避免两套训练协议混淆。它的历史源码仍可从Git查看。**不能将六配置可运行写成七配置全部复现。** 若后续仍保留该实验行，应先明确Options方法规范，再单独实现和验证。

## 6. 训练、任务和指标口径

主环境PointLtl2-v0为连续动作；LetterEnv-v0及当前注册FlatWorld-v0为离散动作。FlatWorld源码可支持其他设置，但不能将当前-v0结果描述成连续动作实验。

原Zone默认参数是16环境、每环境rollout4096、batch2048、10epoch、lr0.0003、discount0.998、entropy0.003；原参考预算15M。新入口保留Zone这些默认值，但steps与output必须显式给定，避免误启动长训。Letter/FlatWorld折扣沿原路径分别为0.94/0.98，回合上限分别75/500，Zone1000。

长稿的1M预算、batch64、5训练seed与每seed500评估回合属于后续实验配置，本阶段没有运行该规模。当前原PPO没有额外“优势裁剪到[-1,1]”开关，不能把它与ratio clip混为一谈。正式对齐实验段时还需统一该设置，不能声称两套超参数已经完全一致。

任务清单集中在`configurations/paper_tasks.json`。主环境/FlatWorld的有限任务与长稿表一致：

|任务|公式|
|---|---|
|phi1|F blue & F green|
|phi2|F blue & F green & F yellow|
|phi3|(!blue U green) & F yellow|

Letter采用a/b/c替代blue/green/yellow，这是显式环境适配，不是声称长稿已完整定义每个环境的所有任务。两个循环模板为GF green、GF blue & GF yellow及对应Letter代换，用于有限时域诊断，不能代表完整的无限任务基准。

原eval_datasets任务/世界资源完整保留，但当前evaluate_tlt.py默认使用公式和reset seed，没有直接重放这些pkl布局；不能声称所有原世界、泛化集和零样本隔离已经重跑。若开展严格zero-shot评估，还需从实际课程语义验证held-out与seen的关系，不能只按公式字符串不同判断。

评估Zone使用确定性动作，Letter/FlatWorld默认采样。成功事件、逻辑violation、物理回报与接受访问数分列，物理撞墙不能直接标为逻辑trap。长稿有限任务可以用回合成功率；循环任务应报告有界接受访问等代理，而不是有限测试“证明”无限时域满足。

## 7. 长稿需要统一的定义与本阶段选项

1. **门控**：正文明确给距离sigmoid，实验段/算法又写可学习64维MLP。当前选择前者；没有宣称实现可学习门控。gamma=3、theta=0.5、proximity∈[0,1]时范围约[0.1824,0.8176]，不是0.95。
2. **长期值**：附录采用折扣MC，正文将其称为满足概率。当前按MC代理实现，保留负奖励，不做无依据的概率校准。q/history-only输入是长稿设计，不能因为没有物理输入就称漏实现，但它有信息限制。
3. **短头目标**：当前遵循附录的下一步short bootstrap；正文写融合的下一步V时，两种定义需统一。
4. **自动机规模**：转换器加环境合法赋值裁剪后的phi1/phi2/phi3，三个环境本次分别为4/8/5状态，而不是长稿写的5/12/18。不同转换与简化可能产生等价图；当前以真实输出为准，不补造状态或转移。
5. **skip语义**：按最短hop距离分类时，真实单条边不会自然出现跨越多个最短hop的跳跃。保留类别槽位不等于已观察到此类边，不为填满四类而改变自动机。
6. **Options与实验设置**：见第5、6节；这些尚不能称为完全对应。
7. **外部方法**：没有接入LTL2Action或GCRL-LTL官方训练实现；历史数据单列来源，不能称同一新管线公平重跑。

这些说明保留在单一总结中；本次没有改动用户要求保护的两个终稿。

## 8. 清理后实际验证

所有项目验证在SSH服务器`wty_LTL`运行，没有安装、升级或降级依赖，也没有启动新长训练。

|检查|实际覆盖与结果|
|---|---|
|test_tlt_curriculum.py|15项合同通过：原reward/feature/done/info、初始策略、不可变上下文、严格加载/迁移、MC、梯度、actor更新、缓存、墙体奖励及课程状态|
|test_paper_mapping.py|7组检查通过：三环境15个任务自动机、4/6维特征、guard对齐、有限数值、六配置范围、3层GAT、多尺度宽度及门控边界|
|verify_tlt_upstream.py|Zone与FlatWorld的官方构建器兼容检查通过，无网络前提|
|test_runtime.py|9/9短训通过：Zone baseline/hv_track各128步，FlatWorld六配置各32步，Letter hv_track32步；每组2次更新，参数有限并保存成功，总耗时约112秒|
|严格公式评估|清理后Zone检查点有限/循环入口执行记录纳入verification.json；只用于可运行性|
|保护与发布检查|两个终稿和原评估资源SHA不变；源码语法、补丁格式、相对路径和暂存安全核对在发布前完成|

合同测试为强制完成MC而设置的8步回合只存在于测试夹具中，不是实际训练入口的回合设置。九组短训也不是论文的性能实验，不能用来声称增益。

训练输出在服务器experiments/smoke，Git只保留紧凑验证结果。其他环境/消融的测试模型已自动删去，只保留两份主环境检查点供评估入口检查。旧128/4k等微型模型都不能代替成熟原策略。

## 9. 性能数据与后续交接

长稿的99.3%等成功率与+6.4个百分点等增益，在本阶段仅作为论文中的待重验主张，不重新认证、不改写成当前实测。历史CSV的来源没有和增强模型检查点完全关联，保留原始字节及provenance；已有短测与旧产品负结果也保留紧凑汇总，不能当作对历史所有实验的否定或对新实现的正向证据。

后续实验可以从这条整理后的主线开始：优先定位成熟原序列status.pth与vocab，验证冻结基线和迁移策略一致，再按同课程、同预算、预设seed和固定测试任务比较baseline与hv_track；有学习信号后再加入消融和其他环境。模型提升、稳定性、违例率、成功步数、循环表现与计算开销需要分别实测。

恢复训练只恢复模型/优化器/课程采样与随机状态，不恢复每个MuJoCo环境的中途物理轨迹。max-seconds只在更新之间停止，steps以rollout为粒度可能超过目标。没有完整回合时训练SR=null；评估中断样本不计入完整回合分母。

代码、方法说明和验证结果已经集中；后续只需维护README、本总结、单一验证记录与新实验产物，不再把每次排查过程复制成多层文档和源码归档。
