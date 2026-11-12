# PRISM 研究重构 v2：在首次任务成功之前，学会发现可迁移的控制关系

版本日期：2026-11-09。研究代号继续使用 **DCC（Decision-Contrast Closure）**。

## 本轮完成的研究判断

PRISM 值得继续推进的科学问题是：**智能体能否通过自己设计的短实验，在首次完整任务成功之前学会可迁移的控制关系；发现这些关系的代价，能否由机制复杂度和区分实验长度来解释？**

这轮将研究对象进一步收敛为：**付费行动差学习与可执行辨别程序发现的联合过程。** 智能体需要用真实干预了解某种条件下哪个行动更好，也需要发现取得该条件的观察程序。两项学习共同使用一个受控响应模型、同一个原任务价值，并共同支付交互与计算成本。

本轮已经完成条件闭包的数学修订、有限机制下的可计算采集算法、明确的程序生长规则、近邻研究复核，以及 512 个随机种子上的有限模型学习实验。实验代码、逐种子数据、图表和复现说明随附。有限模型实验采用已给定的机制族与两读程序语法；它验证组合实验和决策采集的机制，**没有运行原生 ALFWorld、WebShop、LOCA，也没有验证开放语言程序生长的性能**。程序生长是本方案接下来承担原创性的研究假设。

研究依据包括用户附件 prism-evolve-design.md、PRISM_tmp 固定提交 a3d7e967c695bbf2eaddc57718e1f6009dd6b74c 的既有审阅，以及截至本版本日期检索的原始论文和正式出版页面。本版取代上一版中有关平均前像、在线测试发现的创新定位及“尚未执行任何 DCC 相关实验”的表述。

## 首先确定学习的对象

当前稀疏分叉训练以完整任务回报之差更新策略。两条分支都失败时，这个统计量可以完全相同，公开交互中已经出现的格式要求、权限变化、对象位置和操作后果却仍然不同。模型应当能够积累这些条件控制关系，并在新的目标和对象组合上使用。

这里包含两个时间尺度。跨任务共享的机制知识说明某类操作在某类条件下会产生什么后果；任务内状态说明当前实例已经观察到哪些条件、处于哪个物理阶段。新任务重置后者，保留前者。仅把旧轨迹重新塞入上下文，不能单独回答跨任务机制是否已经学会；实验需要未见实体、未见组合和共享参数回退对照。[1,29]

原生目标保持为预算内的完整成功：

\[
J_B(\pi)=\mathbb E_{g,\mathcal E}\Pr_\pi(Y=1,\ C_{\mathrm{total}}\le B).
\]

环境步数、模型 token、计算量与墙钟时间各自计量；不同物理单位不直接相加。效率曲线可以作为次级指标。公开的连续反馈、工具结果与局部检查进入响应学习，原生终局成功继续决定最终成绩。

移动、读取、写入、提交都属于合法行动。一次行动可以同时改变物理状态和提供观察，统一比较其完整后果即可。I/E 标签可以用于事后分析。对实际反馈的内部利用保留到下一步行动，动作序列化约束只作用于需要执行的动作载荷。所有比较方法使用同一个交互接口。

### LoRA 的位置可以明确下来

LoRA 提供权重增量的一种低秩参数化。[2] 研究机制并不要求这个参数化。回报差为零时，更换 rank 或学习率无法使该目标携带已经丢失的过程信息。

首轮科学原型直接学习有限机制后验，本轮代码已经采用这种方式。原生研究版本训练一个共享响应模型的参数与程序结构；语言底模提供相同的语义与动作先验。共享响应模型可以全参数更新。LoRA、全参底模更新、冻结底模加可训练响应网络，是在相同数据、学习目标和可报告算力下比较的参数化选项。

冻结底模这一首轮安排的用途，是把语义先验与新增机制学习分开测量。最终方案按实际性能和成本选择参数化。研究结论需要由新增机制知识与程序发现的实测贡献支撑。

## 文献复核后，哪些创新表述应当收紧

最新 agent 工作已经覆盖了多轮信用分配、反馈信息奖励、跨 episode 探索，以及决策相关世界模型学习。GiGPO、AREW、InfoPO、LaMer、AAWM、DWM、PaW 和 Early Experience 构成直接的经验背景；2026-11-02 的 TriGRPO 是本次检索中较新的树状信用分配预印本。[3–10,14,31,32] AAWM 已包含同状态替代行动数据和决策问题监督，主模型之外也报告同尺度合成模型对照。单靠真实分叉、世界模型辅助任务或省去大教师，无法明确这项研究的新增部分。[9]

此次补充检索还找到了几个更直接的理论与算法前身：

| 原始研究 | 已经覆盖的内容 | 本方案需要进一步完成的内容 |
|---|---|---|
| Online Discovery and Learning of PSRs，NIPS 2005 | 在线发现 core tests，利用预测矩阵结构约束学习 | 有限工具实验预算下的任务相关程序发现 |
| Goal-Directed Online Learning of Predictive Models，EWRL 2011 | 在线交替预测模型学习、目标导向探索与价值估计 | 未知行动差与观察程序的联合采集 |
| Reconciling Rewards with PSRs，IJCAI 2021 | 同时保持观察预测与奖励预测 | 原生目标读出与公开工具响应的一致联系 |
| Active Model Learning of Stochastic Reactive Systems，2024 | 真实随机测试、反例处理、前缀和后缀扩展 | 连续行动效果与实际目标损失驱动的查询分配 |
| Decision-Centered Abstractions，2024；2026-09-07 v4 | 保留动作 Q 差、因果估计与决策相关压缩 | 通过主动交互学习哪些条件值得辨别 |
| Minimal Separating Automata 的 L# 算法，CAV 2026 | +/−/don't-care 查询、apartness 见证和前缀传播 | 行动方向自身需要付费估计，且与辨别程序同时学习 |
| Model Discovery Agent，2026-09-27 v5 预印本 | LLM 结构提案、开放机制空间、近似 Bayes 和信息价值实验设计 | 真实条件行动差怎样提高工具程序发现效率 |

上述对应关系分别见 [22,36–41]。特别需要修订的是：2003 年 PSR 文献把部分 discovery 问题留待后续，2005 年已有直接的在线发现工作。[18,36] “在线长出预测测试”不能独立承担当前方案的原创性。CAV 2026 的分离自动机论文还明确展示了区分见证的反向传播。[41]

因此，研究主张集中为：

> **用付费配对干预产生的条件行动差，改变下一批可执行观察程序的生成分布，并检验这种耦合能否降低未见任务的机制发现成本。**

这一主张对应具体的数据、算子和对照。一般 PSR、条件策略树、Bayes 信息价值与常规世界模型训练均作为已有基础。

## 从第一性原理得到一个共同模型

将一次公开响应记为 r，其中包括观察内容、真实成本、正常终止和预算截断。有限原型可使用规范化受控随机转导器：

\[
P_\phi(s',r\mid s,a)
=T_\phi(s'\mid s,a)\,G_\phi(r\mid s',s,a).
\]

任务内状态是该机制下的信念：

\[
b_{t+1,\phi}(s')
=
\frac{\sum_s b_{t,\phi}(s)P_\phi(s',r_{t+1}\mid s,a_t)}
{\sum_{\bar s,s}b_{t,\phi}(s)P_\phi(\bar s,r_{t+1}\mid s,a_t)}.
\]

跨任务机制不确定性是对不同 \(\phi\) 的权重，任务内隐状态不确定性由各自的 \(b_\phi\) 表达。两者都由同一个概率模型定义。

一个可复用实验程序 \(e=(\rho_e,\chi_e)\) 包含执行前确定的条件续策略，以及从实际公开轨迹读取结果的函数。程序可以根据运行中新出现的反馈分支，不能事后按结果重写试验定义。联合响应为

\[
K_\phi^e(y,c,d\mid h)
=
\sum_{\tau:\operatorname{readout}_e(\tau)=(y,c,d)}
P_\phi^e(\tau\mid b_\phi(h)).
\]

状态更新、程序后果、条件策略价值均从这一分布得到。各个事件概率不由互相独立的 test head 任意给出。相同快照上的兄弟分支共享初始隐藏实例和确实共享的随机量，联合 likelihood 保留这种依赖。单分支核的边缘概率不唯一确定这种耦合；完整生成模型同时规定共享外生随机量及快照恢复时的复用语义。本轮有限实验直接使用明确的联合响应表。

有限原型直接求和；原生版本使用共享条件响应模型和粒子或集成近似。后者的近似误差与计算成本进入实验。首篇研究保持“可增长的实验程序集合”这一对象；有限最小 PSR、自动最小状态数与最优记忆压缩不作为既成性质。

### 公开响应与原生终局目标的统一闭合

公开响应 \(x=(r_{\mathrm{pub}}, c, d_{\mathrm{term}}, Y_{\mathrm{obs}})\) 提供丰富的过程监督：工具调用返回了什么标识符、哪个容器被打开、哪项属性被读取、发生了何种权限或语法错误、消耗了多少资源，以及是否触发终止。这些公开反馈直接用于训练响应模型 \(K_\phi(x_{t+1}\mid g,h_t,a_t,\beta_t)\)，不再通过手工加权积分为中间状态构造替代奖励。官方终局评分器只在原协议规定的终局时刻给出任务成败 \(Y\in\{0,1\}\)，未到达终局或因整个训练作业总预算耗尽而中断的轨迹，其原生终局标签保持缺失（`missing`），不能伪造为任务自身失败 \(Y=0\) 或手工进展分。

控制与规划使用与 `NativeTrajectory.utility` 完全同口径的墙钟时间原生终局效用：

\[
U_B(Y,C_{\mathrm{seconds}})=Y\max\!\left(0,1-\frac{C_{\mathrm{seconds}}}{B_{\mathrm{seconds}}}\right),
\]

而环境步数、语言模型 token 与侧网络计算各自通过显式 `ResourceState(before, after)` 作为独立硬约束计量与披露。由于 \(0\le C\le B\) 时 \(\mathbb E[Y(1-C/B)]=P(Y=1)(1-\mathbb E[C\mid Y=1]/B)\)，终局估值必须保留成功事件与完成耗时的联合相关性。

候选空间统一包含当前全部合法原生动作（有限集合完整进入批量评分；LOCA 开放参数空间从真实 `native_tools` 的 `function.name/parameters` 生成同预算多工具提案）与可绑定、可执行的条件程序：

\[
\mathcal E(h)=\{\text{合法原生动作}\}\cup\{\text{可执行条件程序}\}.
\]

令 \(\pi_0\) 为独立于 DCC 参数 \(\phi\) 和程序库 \(\Xi\) 的显式可调用冻结尾控制器（固定底模、公开提示、完整原生动作并集接口与采样规则）。对短程序叶端，定义由真实 \(\pi_0\) 后缀区间直接监督的固定尾策略联合终局分布：

\[
H_{\phi,m}^{\pi_0}(Y,C_{\mathrm{tail}}\mid h,g,\beta)
=
\Pr_{\phi,m}\!\bigl(Y,C_{\mathrm{tail}}\mid h,g,\beta,\operatorname{do}(\pi_0)\bigr),
\]

通过联合预测 \(P_m^{\pi_0}(Y=1\mid h,g,\beta)\) 与成功条件下的剩余耗时分布 \(P_m^{\pi_0}(C_{\mathrm{tail}}\mid Y=1,h,g,\beta)\) 计算截断期望效用，其中零残差初值 \(\phi_0\) 的尾成功与条件耗时对数几率直接由单步基线转移规律在剩余有限步长 \(K\) 上的几何展开解析导出，从第一性原理保证 \(Q_{\phi_0}^{(1)}=Q_{\phi_0}^{(2)}=Q_{\phi_0}^{(3)}\) 的贝尔曼中性与有限步边界单调递推。候选 \(\rho\in\mathcal E(h)\) 的价值通过共享符号化 `ExecutionFrame` 解释器沿真实 AST 展开（完整消费 `Call`、`Read`、带符号化返回句柄 `$ret_k` 的跨步参数绑定与 `Case` 双子树；终止状态严格吸收，仅以归一化 `live` 概率推进存活质量，不附加无变化收缩因子），并在叶端对潜在状态条件下的联合类型化字段分布做精确离散期望（消除将边缘概率均值直接送入非线性尾头的詹森偏差）：

\[
Q_m^\rho(g,h,\beta)
=
\mathbb E_{K_{\phi,m},\,\rho\rightarrow\pi_0}
\bigl[U_B(Y,C_{\mathrm{spent}}+C_{\mathrm{future}})\bigr]
=
\mathbb E_{Z\sim P_{m}}\!\left[H_{\phi,m}^{\pi_0}\bigl(b_m(\cdot\mid Z),\,\operatorname{one\_hot}(Z),\,\beta'\bigr)\right].
\]

每个集成成员 \(m\) 维护各自独立的私有状态、字段分布与资源轨迹（共享冻结语义编码器，成员间状态与资源不做隐式平均），仅在末端汇总候选均值 \(\widehat Q^\rho=\frac1M\sum_{m=1}^M Q_m^\rho\) 并选取 \(\rho^*=\arg\max_\rho\widehat Q^\rho\)（不引入悲观标准差扣减、固定安全裕度或粗粒度字段自环剔除）。训练选 \(q\)、分叉后真实模型差 \(d_{\mathrm{model}}\)（准确评估 \(u\rightarrow\pi_0\) 与 \(v\rightarrow\pi_0\) 的更新前／后差值）、在线过滤 `observe`、尾头监督与部署 `act()` 共享同一个只读 `DecisionContext` 及固定尾标识嵌入通道。整个 episode 在新环境重置起点固定不可变模型视图 `ModelView`（\(\phi_0\) 或 \(\phi_T\)，其中 \(\phi_0\) 独立持久化保存并在重加载时同步刷新视图缓存）并清空信念、执行帧与历史事件（评测态明确写入 `member_index=None`）。

冻结底模为训练、过滤与规划（含冻结尾策略 \(\pi_0\) 执行区间）提供统一共享的语言嵌入分段池化语义表示 `extract_frozen_semantic_context(g, h)`、规范化动作嵌入 `extract_frozen_action_embeddings(a)`（对运行期不透明返回 ID 与前瞻符号句柄采用同一规范化符号规约）以及单次提示预填充并通过 `DynamicCache.crop` 复用的完整动作加终止符条件对数先验 `score_frozen_action_logprobs(g, h, a)`（对离散动作累计完整序列及 EOS 对数概率，对开放查询或结构化参数工具分解为精确离散操作前缀对数概率加开放参数条件单 token 均值，再经候选集归一化与熵中心化得到先验优势 \(\alpha_0(a\mid g,h)\)，严格按物理前向 token 计费）。尾头 \(H_{\phi,m}^{\pi_0}\) 在尾段直督、\(\pi_0\) 续行边缘似然与前瞻叶端三处入口完全共享纯状态与资源基线，不混入根候选动作先验；动作先验优势仅在分叉及续行步转移估值与边缘似然中以统一对数几率偏移耦合。响应模型包含公开体制内已试动作幂等指示与单一内部四槽四值枚举，多步 AST 前瞻对每个成员独立维护 `(M, 64)` 私有资源与槽位分布而不做跨成员众数折叠，在 `Case` 真分支计算点事件后验 \(b_m(s'\mid Z_j=v_0)\)，在假分支保留三值补集条件分布 \(P_m(Z_j=v\mid s', Z_j\neq v_0)\)；程序生长按与编译二元分支一致的 \(\widehat\Gamma_{\mathrm{binary}}(f,z^*;u,v)\) 条件选择增益评分。分叉续行对 \((u,v)\) 在观测本块结果前使用 `DecisionContext` 预计算的绑定动作模板按集成决策分歧度 \(R_h(u,v)=\frac1M\sum_m\max\{Q_m(u),Q_m(v)\}-\max\{\bar Q(u),\bar Q(v)\}\) 与预计成对执行成本比选取。测量簇采用带 `parent_transition_id` 的分叉树表示并按簇缓存已付费轨迹与冻结特征，固定每簇每成员确定性 bootstrap 权重并对 \(\pi_0\) 续行转移与分支尾损失按集归一化执行同预算多步簇重放更新：

\[
\mathcal L_m(\phi)
=
\mathcal L_{m,\mathrm{prefix}}(\phi)
+
\mathcal L_{m,u\mid\mathrm{fork}}(\phi)
+
\mathcal L_{m,v\mid\mathrm{fork}}(\phi)
+
\mathcal L_{m,\mathrm{tail}}^{\pi_0}(\phi),
\]

配合簇级确定性种子 bootstrap 权重学习边缘条件响应与 \(\pi_0\) 联合终局分布；已付费的部分（删失）轨迹照常更新已观测转移并屏蔽缺失终局标签。

在首次完整任务成功之前，若两条完整续行均失败（\(Y_u=Y_v=0\)），真实终局行动差 \(d_{\mathrm{real}}=U_B(\tau_u)-U_B(\tau_v)=0\)。此阶段由公开响应反馈驱动跨任务机制学习 \(K_\phi\)，提案器在零真实差下退化为各方法共享的语法先验；当配对完整续行产生非零 \(d_{\mathrm{real}}\) 后，真实行动差见证才驱动观测程序的消减与拼接。模型估计差 \(d_{\mathrm{model}}\) 与实测终局差 \(d_{\mathrm{real}}\) 分列存储，不将模型预测回填为实测标签。

## 条件闭包的修订：把反馈保留到作出选择以后

上一版使用平均前像

\[
T_af(h)=\mathbb E[f(haO)\mid h,\operatorname{do}a].
\]

该式可以计算固定续行的平均值。在根据反馈改变续行的任务中，需要保留各个反馈分支。正确的基本对象是

\[
\boxed{(U_{a,o}f)(h)=P(o\mid h,\operatorname{do}a)\,f(hao)}.
\]

\(o\) 包含公开响应、成本与结束类别，\(T_a=\sum_oU_{a,o}\)。先保持条件关系，再按实际策略对结果求和。这是经典 observable operators 与 PSR 的数学基础。[17,36,37]

令 \(\mathcal F=\operatorname{span}\{f_1,\ldots,f_r\}\)，包含常数 \(f_1=1\) 和完整目标读出，并对所有 \(U_{a,o}\) 不变。记 \(s(h)=(f_1(h),\ldots,f_r(h))^\top\)，有

\[
P(o\mid h,a)s(hao)=M_{a,o}s(h),
\]

因此

\[
P(o\mid h,a)=e_1^\top M_{a,o}s(h),\qquad
s(hao)=\frac{M_{a,o}s(h)}{e_1^\top M_{a,o}s(h)}.
\]

对正概率反馈，模型预测和归一化状态更新由同一算子给出。矩阵坐标可以含负数；概率非负性约束作用于可达状态上的预测。首轮有限机制实现通过规范化生成模型保证概率有效，避免把任意矩阵直接当作概率表。

**充分性命题。** 在上述精确条件下，\(s(h)\) 决定目标族中任意有限观察适应策略树的价值。

证明只需归纳。终端目标是 \(w^\top s\)。若动作 a 后每个观察分支的续值是 \(w_o^\top s(hao)\)，当前策略树的价值是 \(\sum_o w_o^\top M_{a,o}s(h)\)，仍为线性函数。最优规划在所有候选策略树的线性值上取最大。特征空间无需额外对 max 封闭。

这个命题承担正确性基础。论文新增结果应来自在真实数据与有限搜索预算下如何获得足够有用的程序和响应。对开放文本的有限编码，还需直接检验编码是否保留目标相关条件；冻结编码器本身不构成充分性结论。

## 互补信息决定了采集单位必须是一段程序

考虑两位独立均匀变量 \(X_1,X_2\)，最终行动要正确提交 \(X_1\oplus X_2\)。初始最佳成功率为 \(1/2\)。单独读取任一位后，最佳成功率仍为 \(1/2\)；读取两位后成功率为 1。因此

\[
I(\mathrm{read}_1)=I(\mathrm{read}_2)=0,\qquad
I(\mathrm{read}_1;\mathrm{read}_2)=1/2.
\]

\(k\) 位 parity 将这一现象扩展到前 \(k-1\) 步。这也对应实际任务中的联合条件：文件格式与编码、商品属性与变体、对象身份与处理状态。单步正信息收益无法成为保留程序前缀的必要条件。

实验设计 \(\xi\) 因而定义为**完整的有界测量程序**：包括训练锚点、可条件分支的操作、公开结果读出，以及需要的同快照续跑。程序在评估前确定，真实执行后用整个结果块更新模型。候选语法保留零即时收益前缀；预算约束限制完整程序的执行和搜索。

如果某次探测 q 同时改变物理状态，部署决策比较

\[
J(q)=\mathbb E_Z\max_\rho Q_q(Z,\rho),
\]

其中 \(Q_q\) 已包括实际后继、剩余预算和续行成功。纯反馈使用价值为

\[
I(q)=\mathbb E_Z\max_\rho Q_q(Z,\rho)-\max_\rho\mathbb E_ZQ_q(Z,\rho).
\]

两项使用同一次物理干预与相同候选续策略，候选的初始输入中不含被遮蔽的 Z。\(I(q)\) 已经包含在 \(J(q)\) 中。实际执行按完整价值选择，保持双重控制的统一目标。[24,25]

## 采集算法：有限模型精确 EVSI 参照与原生主轨迹反事实配对

在拥有明确联合响应表与固定机制集合 \(K_1,\ldots,K_M\) 的有限模型中，固定权重 \(w_m\) 及候选策略价值 \(V_{mj}\in[0,1]\)，记测量程序 \(\xi\) 产生结果 \(z\) 的似然为 \(L_m(z;\xi)=P_{K_m}(Z_\xi=z)\)。采集前后使用完全相同的价值表 \(V_{mj}\)（不附加任何依赖头编号或结果奇偶位的效用乘子），预期决策改善化简为：

\[
\boxed{
\Delta(\xi)=
\sum_z\max_j\sum_mw_mL_m(z;\xi)V_{mj}
-
\max_j\sum_mw_mV_{mj}.
}
\]

该式满足精确上界 \(\Delta(\xi)\le\mathrm{EVPI}\le 1-\max_m w_m\)，确定性结果或单机制已知时严格给出 \(\Delta(\xi)=0\)。这一精确公式保留在有限模型中作为决策论参照。[23,42]

原生主方法主动收敛采集层，移除神经 EVSI 的必经打分路径：训练 episode 开始时固定抽取一个 bootstrap 成员并用同一原生终局目标控制主轨迹；从主轨迹已到达的真实公共前缀中选择测量锚点，解释执行预定义的可执行程序 \(q\) 并读取实际公开结果 \(z\)。若 \(q\) 未结束任务，则保存完整的 post-\(q\) 快照（环境状态、公开历史、控制器状态、变量绑定与随机数状态），不再回退至 pre-\(q\) 状态。从同一 post-\(q\) 初态出发，按预先确定的绑定映射 \(\Pi_u(z),\Pi_v(z)\) 分别接续固定尾策略 \(\pi_0\) 运行至原生终局或任务预算终点：事实分支 \(u\rightarrow\pi_0\) 直接作为当前主轨迹后缀，额外支付至多一条反事实后缀 \(v\rightarrow\pi_0\)。所有唯一物理转移进入同一个测量簇训练 \(K_\phi\)；当两支均观测到原生终局结果时记录实测行动差 \(d_{\mathrm{real}}=U_B(\tau_u)-U_B(\tau_v)\)，预算中断导致未观测时保持缺失。

## 拟议核心算子：类型化可执行 AST 与行动差引导的程序生长

可执行条件程序采用来自公开 schema 的类型化语法：

\[
\texttt{Call}(\text{tool},\text{args},\text{bind}),\quad
\texttt{Read}(\text{var},\text{field\_path}),\quad
\texttt{Seq}(p_1,p_2),\quad
\texttt{Case}(\text{expr},p_{\mathrm{true}},p_{\mathrm{false}}),\quad
\texttt{Stop}.
\]

程序库中的 `ExecutableProgram` 仅保存类型化 AST、形式角色槽与局部变量结构，绝不保存旧任务的具体对象 ID；具体实体绑定、调用返回值与已消费子树由每次真实执行或模型前瞻私有的执行栈帧 `ExecutionFrame` 维护。解释器沿显式 AST 栈逐节点求值（`Case` 消费所选分支子树后直接进入后续节点直至 `Stop`），明确返回 `Action`、`Complete` 或 `BindingFailure`，绑定失败时不冒充原程序继续归属。

比较对象 `Comparison` 在观测终局前规范化有序策略对 \((u,v)\)、目标语义与尾策略版本（交换 \(u,v\) 时同步反转 \(d\) 符号）。见证配对与 `Label-Shuffled` 标签置换严格共用同一个 `comparison_key`：仅在满足 \(d_{\mathrm{real}}(h_1;u,v)\cdot d_{\mathrm{real}}(h_2;u,v)<0\) 的同一比较键内构造见证，且单例比较组保持原样、绝不跨组扩充置换。`Delete` 依据从 AST 消费者图推导的真实依赖关系保护所有被后续节点引用的生产者；`Concat` 执行局部变量 \(\alpha\)-renaming 并保留写操作前后必要的重复读取调用。

每次构造、语法检查与评分的候选均计入提案计算预算。固定数据集 \(\mathcal D\) 的提案对照在相同记录顺序、相同 `grow` 调用次数与相同构造预算下比较真标签与同键组内标签置换（`Label-Shuffled`），正负分组、幅度排序、记录优先级与算子选择全部读取同一置换标签视图；闭环对照则允许 Full 与 Shuffle 在相同总预算下各自在线采集并演化程序库。

## 本轮实际运行的机制实验

### 已知机制：精确检查互补观察

使用 k 个相关公平 bit、N 个无关公平 bit，在只能读取 k 个字段后提交的条件下比较程序策略。模型已知哪些位相关，因此这部分检验规划逻辑。完整程序读取全部相关位，成功率为 1；允许停止的单步 EVSI 在 k≥2 时成功率为 0.5。均匀读取 k 个字段的精确成功率为

\[
\frac12+\frac{1}{2\binom{k+N}{k}}.
\]

k=2、N=6 时，均匀读取为 0.517857；强制继续、每步按单步 EVSI 并均匀处理平局的策略为 0.625；完整程序为 1。代码穷举全部 256 个二进制状态和 28 个二字段组合，独立核对随机子集公式。所有字段都是公平 bit，预测熵平局版本与随机子集相同。

这说明完整程序保留互补条件的作用。该部分已经知道控制关系，不承担未知关系发现的证据。完整候选数 \(\binom{k+N}{k}\) 给出朴素穷举的候选规模；已知相关位的模型可以直接构造正确程序。

### 未知机制：没有终局成功标签时学习

第二个实验将“哪两个公开字段影响操作，以及影响方向”隐藏起来。n 个字段产生 \(\binom n2\) 个两读候选程序，每个有两种极性。机制总数为 \(2\binom n2\)，先验均匀。每个新实例重新采样字段，控制关系跨实例保持。

一项测量从新锚点读取两个字段，再从同一后继快照分别尝试两个行动，共计四次环境动作。局部公开响应带噪：正确行动边缘通过率 0.8，另一行动为 0.2。兄弟分支共享同一个随机量，联合结果为“都失败”0.2、“都通过”0.2、“仅正确行动通过”0.6。学习器使用这一完整联合概率。

该诊断代码模拟短实验和局部部署评价；完整任务标签按协议固定为 0，完整长任务本身未实现。学习目标是公开局部结果的 likelihood。各方法共享已知有限机制族、候选语法、局部噪声规律、测量预算与相同随机种子机制。未知的是正确字段组合及极性。

评价指标是在新实例上选择行动后的**期望局部响应值**。未学习水平为 0.5，已知真实机制的上限为 0.8。训练中不提供完整任务成功，评价中没有把该局部指标称为原生成功率。

每种配置运行 512 个随机种子，32 个测量块，即 128 次环境动作。结果如下，± 为跨种子均值的 95% 正态区间半宽：

| 公开字段数 / 候选程序数 | 随机完整程序 | 最大预测熵 | 最大模型信息增益 | 完整程序 EVSI |
|---|---:|---:|---:|---:|
| 4 / 6 | 0.6923 ± 0.0114 | 0.5757 ± 0.0073 | **0.7894 ± 0.0044** | 0.7651 ± 0.0082 |
| 6 / 15 | 0.5573 ± 0.0090 | 0.5265 ± 0.0046 | 0.5580 ± 0.0090 | **0.6339 ± 0.0128** |
| 8 / 28 | 0.5220 ± 0.0054 | 0.5105 ± 0.0007 | 0.5105 ± 0.0007 | **0.5782 ± 0.0113** |

15 个候选程序时，EVSI 对随机采集的配对均值差为 0.07662，95% 正态区间 [0.06422, 0.08903]；对模型信息增益的差为 0.07586，区间 [0.06083, 0.09090]。6 个候选程序时，模型信息增益超过 EVSI，EVSI 的配对差为 −0.02422，区间 [−0.03313, −0.01530]。这个反向结果完整保留。

这一实验给出三个具体研究结论。公开的局部干预响应能够在完整任务标签全零时支持跨实例机制学习。完整程序允许学习单字段测试无法辨认的互补关系。逐次决策信息采集与模型信息采集的优劣依赖候选规模和预算，单一采集准则的普遍优势不作为论文命题。

**本实验尚未自动生成程序语法、学习噪声模型或运行程序生长算子。** 有限机制族和全部两读候选被共同给定。原生工具语义、前缀到达与模型计算不包含在“128 次环境动作”中。复现包列明这些条件，保留全部逐种子值，便于检查任何均值与区间。

## 理论结果怎样支撑研究命题

本版将条件闭包充分性、信息价值化简和短实验识别分别陈述。它们给出研究算法的明确基础，保持其经典来源和直接推导身份。

若跨任务共享机制属于一个有限精确模型族，存在 r 个长度至多 \(\ell\) 的可达短测试，使任意两个目标相关机制在至少一项 Bernoulli 响应均值上相差 \(\gamma\)，并能独立重复，那么每个测试采样

\[
n\ge \frac{8}{\gamma^2}\log\frac{2r}{\delta}
\]

次，Hoeffding 界与并合界保证全部均值误差至多 \(\gamma/4\) 的概率至少为 \(1-\delta\)。候选机制的测试均值可计算，按测试均值向量的 \(L_\infty\) 距离选择最近候选，即可识别目标相关机制类别。独立重复由新的随机实例或可控独立后续随机性取得；重复恢复同一个随机状态不会自动产生独立样本。短实验的环境交互量为

\[
O\!\left(r\ell\gamma^{-2}\log\frac r\delta\right).
\]

另行支付测试发现与锚点到达；计算费用独立报告。正确识别精确共享机制后，它可以用于不同长度和组合的后续任务。

这里弱化任务长度 H 依赖的条件十分具体：共享有限机制、短区分实验和可达锚点同时成立。连续参数近似模型仍可能累积 \(H\varepsilon\) 级终局误差，实验应独立改变 H 和局部精度。若最短有效实验长度本身随 H 增长，搜索费用也同步增长。

拟议生长算子最值得发展的新理论，是把**付费行动差估计、候选命中率、互补深度与程序复用**写入同一发现成本。最终主张要解释什么时候少量局部知识能够迁移到很长的任务，以及程序提案怎样降低获取这些知识的费用。

## 原生实验的研究协议

首先建立共享的正常交互接口：模型可保留必要的任务内计划与状态，执行器提取有效动作；所有合法行动在同一决策空间；观察保持对象绑定和真实时间顺序；推理预算和评估采样策略配平。这些条件供所有方法共享。原先受控宏动作环境的成绩继续作为受控实验，原生主表单独报告。

核心原生实验围绕一个问题：在首次原生成功之前，响应与程序学习是否已经获得可迁移能力。训练按原生协议记录完整成功数；在匹配累计交互与计算预算的检查点，评估未见实体、未见目标组合的公开响应预测、条件行动选择和最终成功。新的任务内状态重置；回退共享参数与移除生长程序分别衡量参数知识和程序结构的作用。

最小机制主表比较相同响应模型下的随机完整程序、模型信息增益、普通完整程序 EVSI，以及行动差引导的程序生长；后者还包括提案端标签打乱对照。L# 分离学习作为有限形式域对照，分别报告免费行动标签的参考配置与付费干预估计标签的配置，明确其查询权限差异。[41]

外部原生强基线采用可复现的 outcome RL、InfoPO、LaMer 与 AAWM/DWM 类决策世界模型。[3,5,6,9,10,14] 不同方法的额外教师、示范、跨 episode 机会、真实分叉权限和模型 token 使用独立披露并配平。无需同时堆叠全部外部机制到 DCC 中。

原生绑定按公开信息确定：

| 环境 | 程序所用的公开来源 | 应检验的可迁移关系 |
|---|---|---|
| ALFWorld | 文本观察、公开 admissible commands、实际发现的对象 | 位置、可见性、容器操作与同一物体状态的条件关系 |
| WebShop | 页面商品、变体、公开属性和可点击目标 | 同一商品变体的联合条件，以及提交对任务结束的影响 |
| LOCA | 工具 schema、参数类型和实际返回的标识符 | 返回值到后续实参的绑定、权限或格式前置、工件状态与后续工具结果 |

不从隐藏 simulator 状态注入目标位置，不中途查询隐藏 grader，不把不同商品、不同工件或不同反事实分支的属性拼接成联合成功。程序运行产生的错误、状态变化、终止与成本都进入共同响应模型。

最有价值的控制变量包括无关字段数 N、未知二元相关字段对及有限两阶段顺序依赖、最短区分长度 \(\ell\)、机制数 r 与完整任务长度 H。当前有限实验在同种子配对差与逐候选流式构造记账下给出了 N 和候选检查预算下的学习曲线。程序发现实验重点改变生成预算；原生实验负责检验这些结构是否解释真实任务中的学习效率。

## 方案的研究价值与完成边界

这一方案的中心对象现在足够明确：真实干预产生条件行动差；这些差异引导下一批可执行观察程序；真实程序响应更新同一个跨任务模型；模型对原任务目标规划。每项组件都有确定的数据来源和作用，LoRA、反思模块、额外信息奖励与独立认证流程均不承担方法成立的前提。

来自其他学科的启发也有明确落点。潜伏学习对应“完整成功之前已经形成何种知识”；双重控制对应“观察和改变世界共同评价”；统计决策论对应“实验按下游选择的改善计价”；控制与预测状态理论对应“知识通过可执行未来响应定义”。[23,24,26–28] 这些基础把研究问题组织起来。新增技术贡献集中于实际工具程序的发现过程。

本轮交付已经包含定义完整的学习对象、可运行的有限采集算法、条件闭包与互补信息分析、明确的程序生长操作及其决定性对照，以及实际执行的机制实验。开放语言生长算子和三项原生基准属于下一阶段研究实验，当前没有相关新成绩。

适合论文摘要的研究假设可以写成：

> 在共享局部机制、目标条件公开可表达且关键关系可由短实验揭示的任务族中，条件行动差驱动的程序发现能够在首次完整任务成功之前形成可迁移知识，并在有限交互与计算预算下降低未见组合任务的学习成本。

论文的核心证据应由程序发现效率、跨任务机制迁移与原生成功共同提供。当前结果已经支持继续推进组合实验与公开响应学习；下一项最决定性的结果，是在相同模型、数据和生成预算下，行动差见证是否确实更快产生有价值的新程序。


## 文献与核验范围

以下为本报告实际使用的原始论文或正式来源。对只取得官方摘要的工作，机制描述保持在摘要范围。表中标注的日期为首发或本报告引用的正式版本；页面抓取日期不作为发表日期。

| 编号 | 文献及链接 | 本报告用途 |
|---|---|---|
| [1] | [RL²: Fast Reinforcement Learning via Slow Reinforcement Learning, 2016](https://arxiv.org/abs/1611.02779) | 慢参数与快速任务内适应 |
| [2] | [LoRA: Low-Rank Adaptation of Large Language Models, 2021](https://arxiv.org/abs/2106.09685) | 参数化与资源含义 |
| [3] | [Group-in-Group Policy Optimization for LLM Agent Training, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html) | 多轮信用分配基线 |
| [4] | [On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents, ICML 2026](https://proceedings.mlr.press/v306/zou26b.html) | AREW 与信息自锁 |
| [5] | [InfoPO, ICML 2026](https://proceedings.mlr.press/v306/kong26g.html) | 信息奖励近邻 |
| [6] | [Meta-RL Induces Exploration in Language Agents, ICLR 2026](https://brbiclab.epfl.ch/projects/lamer/) | LaMer 官方论文与项目说明 |
| [7] | [MAGE, 2026-03-04](https://arxiv.org/abs/2603.03680) | 战略探索元学习 |
| [8] | [Spinning Straw into Gold, ICLR 2026](https://openreview.net/pdf?id=QNfmqMSR7r) | HSL 失败轨迹后见重标注 |
| [9] | [Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.599.pdf) | 最关键近邻；核对全文及附录 |
| [10] | [Discriminative World Models for Web Agents, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2401/) | 动作后果判别学习 |
| [11] | [Self-Evolving World Models for LLM Agent Planning, Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2221/) | WorldEvolver 非参数经验演化 |
| [12] | [FIRM-WM, 2026-09-19](https://arxiv.org/abs/2609.22816) | 官方摘要：同状态干预与无奖励规划；本轮未取得全文 |
| [13] | [METTA, 2026-10-22](https://arxiv.org/abs/2610.31656) | 官方摘要：两时间尺度记忆元适应；本轮未取得全文 |
| [14] | [Hindsight Credit Assignment over Trees for Agentic RL, 2026-11-02](https://arxiv.org/abs/2611.02706) | TriGRPO 最新树状信用分配 |
| [15] | [Goal-driven Bayesian Optimal Experimental Design for Robust Decision-Making Under Model Uncertainty, 2026-05-25](https://arxiv.org/abs/2605.26093) | 决策目标驱动实验设计 |
| [16] | [ActiveCQ: Active Estimation of Causal Quantities, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/85ff89c606e8b79743d3db2222ed7dcb-Abstract-Conference.html) | 主动因果量估计 |
| [17] | [Predictive Representations of State, NIPS 2001](https://papers.neurips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) | 预测状态与测试基 |
| [18] | [Learning Predictive State Representations, ICML 2003](https://cdn.aaai.org/ICML/2003/ICML03-093.pdf) | 测试基发现与概率学习 |
| [19] | [Surprise-based learning of state representations, 2018](https://www.sciencedirect.com/science/article/abs/pii/S2212683X18300264) | SDE 自动区分实验 |
| [20] | [Reconstruction of Epsilon-Machines in Predictive Frameworks and Decisional States, 2011 作者稿](https://nicolas.brodu.net/common/recherche/publications/decisional_states.pdf) | 效用相关状态归并 |
| [21] | [The Value Equivalence Principle for Model-Based Reinforcement Learning, NeurIPS 2020](https://arxiv.org/abs/2011.03506) | Bellman 相关模型等价 |
| [22] | [Decision-Centered Abstractions via Orthogonal Estimation of Difference-of-Q Functions, 2024；v4 2026-09-07](https://arxiv.org/abs/2406.08697) | 动作差与因果抽象直接前身 |
| [23] | [Bayesian Experimental Design: A Review, Chaloner & Verdinelli, 1995](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/readings/chaloner-verdinelli.pdf) | 决策理论实验设计 |
| [24] | [Dual control theory I, Feldbaum, 1960](https://www.mathnet.ru/eng/at12624) | 控制与信息的共同作用 |
| [25] | [Value of Information: A Framework for Human–Agent Communication, ACL 2026](https://aclanthology.org/2026.acl-long.1987.pdf) | LLM 信息价值直接近邻 |
| [26] | [Cognitive Maps in Rats and Men, Tolman, 1948](https://psychclassics.yorku.ca/Tolman/Maps/maps.htm) | 潜伏学习的研究启发 |
| [27] | [Computational Mechanics of Input-Output Processes: Structured transformations and the ε-transducer](https://arxiv.org/abs/1412.2690) | 2014 首发、2015 正式论文；受控响应定义状态 |
| [28] | [Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control, 2018](https://www.sciencedirect.com/science/article/pii/S000510981830133X) | 受控可观察函数的算子表达 |
| [29] | [ContraBAR: Contrastive Bayes-Adaptive Deep RL, ICML 2023](https://proceedings.mlr.press/v202/choshen23a.html) | 对比预测表征与 Bayes 适应 |
| [30] | [Learning Action-based Representations Using Invariance, RLC 2024](https://rlj.org/papers/rudolph2024learning) | 无奖励控制相关表示与递归不变性 |
| [31] | [Policy and World Modeling Co-Training for Language Agents, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1885/) | 同一策略的 RL 与转移辅助训练 |
| [32] | [Agent Learning via Early Experience, ICML 2026](https://proceedings.mlr.press/v306/zhang26m.html) | 早期无奖励交互与后续训练 |
| [33] | [The Value of Memory in Language Agents, 2026-10-22](https://arxiv.org/abs/2610.31303) | 按任务损失评价记忆；官方摘要核验 |
| [34] | [Can LLM Agents Infer World Models? Evidence from Agentic Automata Learning, 2026](https://arxiv.org/abs/2606.16576) | 主动隐藏自动机推断及强 oracle 边界 |
| [35] | [What Capable Agents Must Know, UAI 2026](https://proceedings.mlr.press/v337/nayebi26a.html) | 决策能力所需预测结构的理论边界 |
| [36] | [Online Discovery and Learning of Predictive State Representations, NIPS 2005](https://proceedings.neurips.cc/paper/2005/file/c3008b2c6f5370b744850a98a95b73ad-Paper.pdf) | 在线 core-test 发现的直接前身；论文集亦标 2006 |
| [37] | [Goal-Directed Online Learning of Predictive Models, EWRL 2011](https://ewrl.wordpress.com/wp-content/uploads/2011/08/ewrl2011_submission_40.pdf) | 目标导向在线预测模型与规划 |
| [38] | [Reconciling Rewards with Predictive State Representations, IJCAI 2021](https://ijcai.org/proceedings/2021/299) | 联合保持观察与奖励的 R-PSR |
| [39] | [Active Model Learning of Stochastic Reactive Systems, 2024-03-23](https://link.springer.com/article/10.1007/s10270-024-01158-0) | 真实随机测试、反例、前缀与后缀处理 |
| [40] | [Model Discovery Agent, 2026-08-10 首发，2026-09-27 v5](https://arxiv.org/html/2608.09696v5) | 开放机制发现与目标相关实验设计；预印本 |
| [41] | [An L# Based Algorithm for Active Learning of Minimal Separating Automata, CAV 2026](https://link.springer.com/chapter/10.1007/978-3-032-32526-6_13) | 2026-07-24 正式上线；分离见证传播与查询学习 |
| [42] | [The Knowledge-Gradient Policy for Correlated Normal Beliefs, 2009](https://people.orie.cornell.edu/pfrazier/pub/CorrelatedKG.pdf) | 按下游决策效用采集、关联信念的信息价值 |

### 对附件文献定位的修订

AREW 的正式页面为 ICML 2026；GiGPO 为 NeurIPS 2025；WorldEvolver 为 Findings EMNLP 2026。Step-DPO 的原始对象是数学推理的步骤偏好，迁移到原生工具交互时应标明改造。关于预印本的结论按其公开版本引用，不将尚未确认的会议身份写为既定事实。

本次收敛保留原项目的目标：通过真实经历获得跨任务的主动探路与反馈利用能力。新的研究把重点放在这种能力的学习结构、必要信息和交互复杂度上。

