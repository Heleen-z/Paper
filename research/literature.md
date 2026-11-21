# 文献核查与复现路线建议

检索核查日期：2026-10-23。以下结论来自论文出版页、论文正文或作者公开代码。文献中的效果属于原作者实验，本课题尚未开展实测。完整结构化书目信息见 `references.json`。

## 推荐研究定位

本科主线建议为：**围绕 Office-Home 闭集跨域识别，构建可通过图像和自然语言交互、能执行主动查询与模型更新的智能体；优先试跑2024年 EvidentialADA，使用 LADA 作为可完成的保底，并以同口径实验验证其与传统采样策略的差异。** 主要工作落在交互闭环、受控工具执行、任务状态管理及可解释反馈。研究内容可以有工程创新和系统评测价值，不需要承诺发明新迁移算法。

这一定位覆盖题目的三个部分：

- 主动域适应：存在有标签源域、无标签目标池、有限人工标注预算，以及使用反馈更新的适配模型。
- 多模态：输入图像与用户文本指令，图像语义通过 CLIP 或识别模型提供；输出类别、相似样例、选样依据与文本解释。Office-Home 原数据是图像分类，不是原生对话数据。
- 交互智能体：根据用户意图、实验状态、剩余预算与工具返回结果决定下一步动作，支持追问、暂停、纠错、恢复与结果追溯。不能只给模型外面套一个聊天窗口。

## 复现方法定案

### 优先试跑：EvidentialADA（WACV 2024）

Duan 等人的 *Evidential Uncertainty Quantification: A Variance-Based Perspective* 已正式发表，作者[仓库](https://github.com/KerryDRX/EvidentialADA)含训练入口、模型、选样和不确定性模块。它在 Office-Home / VisDA 上进行真实闭集 ADA：ResNet-50证据分类器、5轮共5%人工预算，并额外使用确定样本伪标签。[论文](https://openaccess.thecvf.com/content/WACV2024/html/Duan_Evidential_Uncertainty_Quantification_A_Variance-Based_Perspective_WACV_2024_paper.html)

论文方法的重要性在于方差式不确定性可呈现类别级信息，原文并未证明其一定大幅超过熵式度量。因此，本课题应评价准确率、查询效率和交互解释价值。试跑窗口最多两周，以一对迁移任务和一轮查询验证训练、预算、日志及伪标签分离；可运行后冻结配置，再扩展多方向。未跑通则采用LADA。旧依赖指定PyTorch 1.7/CUDA 11.0，需为新显卡迁移环境。仓库未见LICENSE，公共阅读不等于获准复制再分发；成品仅给出论文和仓库链接。

### 保底方法：LADA（ICCV 2023）

论文：Sun, Lu, Ling. *Local Context-Aware Active Domain Adaptation*。采样模块 LAS 根据局部预测不一致性选样，适配模块 LAA 使用可信邻居补充训练。论文和[官方实现](https://github.com/tsun/LADA)均可查，仓库为 MIT 许可。该实现已包含 Random、Entropy、CoreSet、CLUE 等比较方法，方便统一模型、数据和训练器。

**推荐顺序：**先跑通 Random + ft、CLUE + ft、LAS + ft，验证采样差异，再跑 LAS + LAA，并给 Random/CLUE 配置相同可用训练器以检查收益来源。完整复现后再把选择器及训练器接入交互工具。LADA 是具备可复现优势的较新代表方法，不应写成“截至2026年的最新方法”。[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Local_Context-Aware_Active_Domain_Adaptation_ICCV_2023_paper.pdf)

原仓库记录 Python 3.8、PyTorch 1.8、CUDA 11.1。这些是作者当时环境，不是对 5070 Ti 的安装建议。实际应先检查显存、驱动与框架支持，再建立隔离环境，记录迁移改动、随机种子和版本。尚未执行训练，不能承诺具体训练时长或显存峰值。

### 可靠备选：CLUE（ICCV 2021）与 LAS + 简单训练器

CLUE 的不确定性加权聚类易理解、易定位错误，作者[仓库](https://github.com/virajprabhu/CLUE)为 MIT。它是强经典基线，不是仅用 Random 就能代表的传统方法。若 LAA 迁移成本超出计划，保留 LAS 采样与统一训练器，并明确论文贡献是交互系统与模块复现，不把简化实现宣称为完整 LADA。[论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.pdf)

### 2025–2026 前沿的取舍

| 工作 | 核查结果 | 对本科主线的用途 |
|---|---|---|
| EvidentialADA，WACV 2024 | 正式论文及作者代码可查，Office-Home闭集ADA，代码许可未明 | 优先试跑目标，关注类别级不确定性解释与同预算表现 |
| Category-Aware ADA，ICML 2024 | 出版论文与作者代码可查，但原实验围绕指定类别的二分类代理和影响函数 | 可启发“优先改善某类别”的交互功能，原数值不能直接并入65类top-1表 |
| GPAS + PLCS，ICIP 2025 | arXiv明确注明已接收，属于真正ADA；本次未核实可运行官方代码 | 说明确定样本、伪标签与主动查询协同发展的趋势 |
| CLIP-guided sample selection，KBS 2026 | 正式期刊出版，DOI带2025，卷期为2026；本次未核实官方代码 | 证明CLIP与ADA结合已存在，避免将其宣称为首创 |
| CEC，WACV 2025 | 正式会议论文，校准熵与邻域不确定性用于VLM主动学习 | 解释不确定性校准的重要性；其任务不自动等于ADA |
| UnDi，Information Fusion 2026 | 正式期刊论文，官方MIT代码、运行说明可查 | 若有余力，添加VLM主动学习探索；需另建Office-Home源目标协议 |
| 双提示主动CLIP适配，2026 | arXiv预印本，未核实正式接收状态 | 作为最新方法动向，不能写成已有稳定共识 |
| CCMA，2026 | arXiv预印本，存在作者实现链接 | 多模态校准与选样的相关工作，说明跨模态分歧思路已有研究 |
| BiTTA，ICML 2025 | 正式论文及代码可查 | 二元反馈降低交互负担的动机；测试时学习协议须与池式ADA分开 |

相关一手链接：[Category-Aware ADA](https://proceedings.mlr.press/v235/xiao24b.html)、[GPAS+PLCS](https://arxiv.org/abs/2505.19421)、[CLIP-guided ADA](https://doi.org/10.1016/j.knosys.2025.115165)、[CEC](https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html)、[UnDi论文](https://doi.org/10.1016/j.inffus.2026.104260)与[代码](https://github.com/Yangfan-123-cell/UnDi)、[双提示模型](https://arxiv.org/abs/2602.04340)、[CCMA](https://arxiv.org/abs/2603.23159)、[BiTTA](https://proceedings.mlr.press/v267/lee25g.html)。

## 不能混用的概念

| 概念 | 关键条件 | 本课题如何使用 |
|---|---|---|
| 主动学习 AL | 主动选取需要获得标签的样本，不要求存在源目标域差异 | Random、Entropy、CoreSet等选样比较基础 |
| 域适应 DA | 源域与目标域存在分布差异，目标域知识影响模型 | 显式写清源域、目标域、类集合与可访问数据 |
| 主动域适应 ADA | 域偏移条件下用有限预算查询目标标签并适配 | 算法验证主协议 |
| 无源主动域适应 SFADA | 适配时不能访问源数据，仅有源模型及目标数据 | 本科最小主线不额外引入该限制 |
| 测试时适配 TTA | 在部署/测试流上更新模型，常无人工标签 | 作为相关概念，不与独立测试集的池式ADA混表 |
| 提示或记忆 | 改变文本上下文、缓存或检索内容 | 可改善交互，但只有明确模型/预测规则更新并独立评测，才可讨论适配收益 |

基础来源：[DANN](https://www.jmlr.org/papers/v17/15-239.html)、[CLUE](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.pdf)、[Tent](https://arxiv.org/abs/2006.10726)。

## 公平比较与实验边界

1. **保留两类实验口径。**论文复现实验尽可能遵循原方法协议，用于判断实现是否正确；本课题系统实验使用独立目标池、验证集、测试集，用于评估交互与泛化。两者分表，不将新划分结果冒充原论文复现数值。
2. **预算定义透明。**可采用目标可查询池的1%、3%、5%作为累计预算检查点；最终5%分成五轮时，每轮增加约1%池规模。若采用65×k张的总预算，应写“每类等价预算”，不能用隐藏目标真标签强制每类配额。新增验证标签需单列并计入完整监督成本。
3. **同口径比较采样器。**同一图像编码器、预训练权重、源训练数据、目标划分、查询轮数、训练步数和种子。原ResNet LADA与CLIP迁移版本应分组展示，不能把更强预训练的收益算成采样算法改进。
4. **独立测试。**测试图像和标签均不参与选样、邻域构建、伪标签训练、温度校准、阈值选择或停止决策。若某原论文协议允许目标无标签全量传导使用，必须另行标明其传导式设置。
5. **标签来源分离。**人类完整标签、二元确认、模拟oracle标签、伪标签分别记录。二元“错误”反馈不提供正确类别，不能自动转为真实标签。模拟oracle实验不能用于证明真人标注时间下降。
6. **指标分层。**算法层报告准确率、macro-F1、随预算变化的曲线及曲线面积、多个随机种子的均值与标准差；系统层报告任务完成率、意图解析/工具调用成功率、纠错恢复率、人工动作和耗时。时间实验需要真实计时与参与者说明。
7. **结果必须允许失败。**“较新方法在相同预算下表现更好”是待验证假设。若差异不显著或有负迁移，应分析域、类别与参数条件，不能通过改测试划分或挑最好种子制造先进性。

## 交互研究的可检验问题

- 用户能否通过自然语言完成“指定域、设定预算、查看待标注样本、纠正类别、更新模型、比较历史轮次”的完整任务？
- 与等功能表单/状态机界面相比，智能体是否减少完成任务所需操作，且没有增加错误执行？
- 在含糊指令、预算不足、标签冲突、训练失败等情况下，追问、状态检查与恢复机制能否提高任务完成率？
- 在完全相同的标注样本与适配模型下，图像示例和文本解释是否帮助用户理解选样理由？这类结论需要用户任务与测量支撑。

建议把系统消融设计为：完整交互智能体、去掉澄清机制、去掉执行前状态/预算校验、等功能表单流程。避免比较一个功能完整系统与一个缺失关键功能的弱基线。

## 书目与命名核查提醒

- 本次能明确核实的“D3”为 D3GU（WACV 2024）中的 Decomposed Domain Discrimination；没有核实到独立名为“D3、2023”的主动域适应论文。报告中不使用无法定位的简称。[D3GU](https://arxiv.org/abs/2401.05465)
- CoOp按正式期刊年份写2022；LADA按正式会议年份写2023；CEC按WACV正式年份写2025。预印本年份与正式出版年份分开标注。
- CLIP-guided论文的DOI包含2025，但出版页卷期为2026年2月；UnDi卷期为2026年9月。不要根据DOI字符串推断年份。
- 所有“本研究创新点”应写成拟实现的系统设计与拟验证的问题，不能复述相关方法已有机制却不引用。

## 阅读顺序

第一阶段读 Office-Home 与 CLIP，建立数据和多模态表征概念。第二阶段读 DANN、CLUE、LADA，结合代码画出选样与适配闭环。第三阶段读 CEC、UnDi、2026 CLIP-guided ADA，确认前沿与本科工作边界。第四阶段读 BiTTA 与 Category-Aware ADA，提炼交互反馈和用户意图的研究问题。完整条目与链接见 `references.json`。
