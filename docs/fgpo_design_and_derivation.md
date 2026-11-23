# F-GPO r7：规范充分 POMDP 滤子、对称势能与均衡词元均值微批更新

版本日期：2026-11-23。执行协议：`fgpo-r7-token-mean-v1`。

本方案在 `r5`（`fgpo-r5-symmetric-v1`）与 `r6` 暴露出 1.5B 训练崩溃诊断的基础上，保留经实证稳定的前端表征并彻底修正 PPO 优化与 GCS 存储瓶颈：
1. **规范充分 POMDP 滤子 $\Pi(\mathcal{H}_t)$**：固定置顶任务/完整非 JSON `Crafting commands` 根前缀 KV，历史区仅保留紧凑单行 `(Action -> Observation)` 轨迹记录并剥离过期菜单、`&lt;think&gt;` 推理草稿与格式错句，使 1.5B 不再发生坏格式上下文传染、`z_hist` 退化或长上下文重复循环；
2. **TextCraft 公共规则显式化与对齐时域**：系统提示写明模拟器“不可 `get` 合成物”与“`craft` 必须严格逐字匹配单条配方”的两条公开公理，训练时域设为 `H=60`（对齐公开 1.5B 基线），官方评测在单次 `H=70` 轨迹上同时统计 30 步、60 步、70 步成功率及第 31–70 步新增解；
3. **均衡打乱的词元均值微批 PPO（Token-Mean）**：采用 `loss_normalization="token"` 与 `entropy_normalization="token"` 的均衡打乱微批切分（`ppo_minibatch_rows=256`, `ppo_epochs=1`, `lr=1e-6`, `credit_prior_weight=1.0`, `shuffle=true`），对齐已发表 1.5B 方法的单轮更新预算，消除 `r6` `step_token` 对长输出的 KL/PPO 稀释与熵权重失衡。

本文与 [数学及实现契约](fgpo_math_contract.md) 共同定义现行 `r7` 方案，对应实现位于 `src/fgpo/`。旧 `r6`、`r5`、`r3/v4` 与旧 TF-BEFP/DAG 可分别通过 git 提交追溯。

## 1. 2026 年研究进展与 2027 年研究定位

[GiGPO，NeurIPS 2025](https://arxiv.org/abs/2505.10978) 将轨迹级相对优势与重复 anchor state 上的步级相对优势结合，建立了无需学习 critic 的多轮智能体训练路线。2026 年的工作已经深入到历史条件、近邻分组、跨轨迹图信用及小样本可靠性。因此，r3 的贡献需要由明确的估计对象、误差分析、已发表方法比较和自身机制消融共同支撑。

| 工作与一手来源 | 2026 年的研究内容及方法 | r3 的对应研究边界 |
|---|---|---|
| [HGPO，ICLR 2026](https://arxiv.org/abs/2602.22817) | 按历史上下文一致性构造多级组，对各级优势自适应加权，处理同一局部状态下历史不同的问题。 | 历史条件化已有直接先例。r3 研究实际有界 actor 历史与同深度续值共享的组合，并记录历史表示对预测质量的影响。 |
| [BiPACE，2026 预印本](https://arxiv.org/html/2606.25556v1) | 使用当前 actor 的 prompt 末位隐藏态近邻聚类；Q-style PACE 在簇内按动作聚合，形成局部动作值与状态值之差。 | 论文的 prompt 包含近期历史。r3 使用冻结参考表示、同深度匹配、整轨迹排除和下一状态续值；本地观察值消融的结果只说明自身设计。 |
| [GraphGPO，ICML 2026](https://proceedings.mlr.press/v306/cheng26e.html) | 将多条轨迹合成状态转移图，用到成功目标的图距离变化分配动作信用。 | 失败轨迹前缀获得跨轨迹信用已有先例。r3 直接共享真实终局回报，避免将最短成功路径解释为策略平均续值。 |
| [RewardFlow，NeurIPS 2026](https://arxiv.org/abs/2603.18859) | 通过共享状态连接轨迹，从成功节点沿图反向传播与最短路径距离有关的势函数，以势差形成动作级信号。 | 图传播与稠密信用已有直接竞争方案。r3 的共享值始终来自同伴的实际 Monte Carlo 结果。 |
| [G2PO，2026 公开版本](https://arxiv.org/abs/2606.22995) | 合并相同观测形成全局转移图，聚合状态值，按边计算并标准化 TD 信用。 | r3 按剩余时域区分状态，并采用固定奖励尺度。会议归属以正式出版来源为准，当前所引公开页未标会议。 |
| [ECPO，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.172/) | 对规范动作的少量样本估计进行收缩，并依据动作内方差调节信用，处理偶然高回报与稀疏证据。 | 小样本信用可靠性已有专门工作。r3 只用可观测距离与支持质量控制共享，诊断来自正式日志，不增加校准器或额外训练目标。 |
| [MCGPO，2026 年 10 月预印本](https://arxiv.org/abs/2610.13755) | 用动作前状态摘要和规则标注进度匹配上下文，固定分折隔离候选动作与定向偏好的终局证据，以支持权重缩放 GraphGPO 之外的成对学习项。 | 分折隔离与支持加权也已有近邻工作。r3 的估计对象是同深度平均续值，使用公开历史表示与单一 PPO 更新，无 rubric 标注或额外成对目标。 |

这些工作共同说明，2027 年仍可推进的研究问题是：在有限采样、部分可观测性和有限上下文共同存在时，**哪些跨轨迹信息可以有效共享，共享多少，以及这种共享何时改善官方成功率**。r3 的具体候选贡献是同深度的留一轨迹续值估计、可解释的连续共享强度、与实际行为条件一致的训练实现，以及对应的误差与成本证据。

论文定位围绕这组可检验的贡献展开。历史表示、留一基线、核权重和 PPO 本身属于已有工具；算法组合的价值由完整 benchmark、与已发表近年方法的关系及实际成本共同体现。当前正式主比较只有 F-GPO 与 **GiGPO、HGPO、GraphGPO**。三种 baseline 分别代表局部 anchor 信用、历史层级信用和跨轨迹图信用，覆盖本方案必须面对的直接竞争关系。表中其余工作用于解释研究边界，当前最小实验计划不为其新增单元。

比较器采用作者发布源码的完整算法配方，并将来源固定到明确提交。ALFWorld 保留来源算法、提示和 batch 语义；GiGPO 从公开 1.5B 全参配方迁移到本研究 7B 基座，HGPO/GraphGPO 采用作者 7B 配方。统一 PyTorch 执行后端承载全部方法，论文呈现完整方法的质量与成本。机制归因由三个 F-GPO 自身消融完成。

## 2. 目标、信息边界与真实历史

### 2.1 有限时域的官方成功目标

对任务实例及其初始环境状态，独立采样 $K$ 条轨迹。轨迹 $k$ 包含 $T_k\leq H$ 个动作，官方终局成功为 $Y_k\in\{0,1\}$。ALFWorld 的指定时域为 $H=50$，AppWorld 为 $H=100$。环境结束或达到时域上限都关闭轨迹。

研究目标为

$$
J(\theta)=\mathbb E_{\pi_\theta}[Y],\qquad q_k=R Y_k,\qquad R=10,
$$

$$
r_{k,t}=q_k\mathbf 1[t=T_k-1],\qquad
\gamma=1,\qquad G^{\mathrm{MC}}_{k,t}=q_k.
$$

官方成功提供训练奖励。语法解析结果保留为诊断，实际动作交给原生环境执行；r3 信用估计中没有单独的语法扣分。参考模型只提供响应概率和表示。目标重构似然、观测惊奇及 TF-BEFP 都退出奖励定义。

这一选择使奖励目标与官方成功率直接对应。即使附加信号始终落在 $[0,1)$，它仍可能改变策略期望回报的排序：两条策略成功概率为 $.51$ 和 $.50$，附加信号均值分别为 $0$ 和 $.9$ 时，$10Y+u$ 的期望分别为 $5.1$ 和 $5.9$。终局奖励尺度大于附加奖励并不能保证期望目标等价。r3 将跨轨迹信息放入显式估计器，并在第 4 节分析其更新偏差。

固定 $R=10$ 便于统一记录与推导，优势最后除以同一个常数。不同任务之间不进行随机 RMS 缩放，因此每个任务的相对权重由采样协议和轨迹计数定义。

### 2.2 完整公开历史与实际 actor 输入

令公开历史为

$$
h_{k,t}=(g,o_{k,0},y_{k,0},o_{k,1},\ldots,y_{k,t-1},o_{k,t}),
\qquad p_{k,t}=C(h_{k,t}),
$$

其中 $y_{k,t}$ 是包含实际采样终止符的响应词元序列，$C$ 是实现中的序列化与有限窗口算子。理论分析可以条件化于完整 $h_t$，模型实际决策条件始终是 $p_t$。上下文上限默认 32,768 词元；完整公开记录超出此容量后，actor 输入保留初始锚定块与最近能够容纳的完整交互轮次。

`public.py` 只接受标准公开输入。ALFWorld 采用三段式整数帧拼接 `user_prefix_ids + pure_ids + user_suffix_ids`（包含步数计数、滤除 `"help"` 的无引号可执行命令清单以及不含具体动作词的字面标签说明 `Enclose concise reasoning in literal <think> and </think> tags, then enclose one exact command in literal <action> and </action> tags.`），保持 `observation_body_offset = len(user_prefix_ids)` 使 `TurnBlock` 与 `SegmentRecord` 的 `observation_span` 精确覆盖纯观测词元切片；TextCraft 在 `pure_ids` 后以独立整数元组拼接同款字面标签尾缀（避免 BPE 跨 JSON 边界合并），并将 `representation_ids`（含 `initial_representation_ids`）限定为持久可见背包文本 `"inventory"` 以剥离瞬态报错 `"feedback"`，单步上限对齐 `512` tokens；响应采用剥离完整闭合 `<think>...</think>` 后提取唯一闭合 `<action>...</action>` 小写单行指令并严格拒斥未闭合 `<think>` 的确定性解析；WebShop 保持原生简洁无 `<think>` 尾缀接口。AppWorld 保留初始任务、系统说明及初始 API 引导信息，后续过长 stdout 在入史时进行固定的首尾截取。空 stdout 在表示和消息中使用明确占位文本。内部状态哈希、评估器中间成功标签和隐藏数据库状态不进入模型或匹配特征。

`filtration.py` 在整数词元空间构建不可变块。给定生成预算 $M_{\mathrm{gen}}$（默认 ALFWorld 512、AppWorld 1024），提示上限为 $C_{\mathrm{prompt}}=C_{\max}-M_{\mathrm{gen}}$；滚动时优先将最近完整轮次截到 $0.75C_{\mathrm{prompt}}$ 的水位。若初始锚定块与最新完整轮次超过这一预留水位，则保留这两个必要块，并允许使用至提示上限；超过提示上限时显式报错。整个过程按完整轮次处理。

每个真实响应只归属一个训练段。为构建新上下文而重播的历史响应没有新的损失所有权。反馈仅在下一次实际决策需要时进入 actor 段；终末反馈保存在步骤和轮次记录中，`observation_span=None`，没有额外 feedback-only 评分段。三个正式自身消融都保留完整 F-GPO 的 actor 历史、生成预算和 PPO 条件，使各自的唯一变化落在匹配距离或共享规则上。

### 2.3 历史与局部观测表示

在冻结参考模型上，取实际动作前提示末位的倒数第 8 层表示并归一化，得到 $z^h_{k,t}$。在 AppWorld 连续输出空间，将当前公开局部观测单独编码并在相同层取末位归一化向量，得到 $z^o_{k,t}$；在 ALFWorld 离散观测空间，直接以整数词元序列相等 $m^{\mathrm{obs}}_{kj,t}=\mathbf 1[o^{\mathrm{id}}_{k,t}=o^{\mathrm{id}}_{j,t}]$ 判别局部观测同态，省略 `z_obs` 模型前向。前者保留本次决策真实可见的任务和历史；后者保留当前观测的局部匹配关系。相关特征均停止梯度。

在 AppWorld 上，相同的局部观测词元序列在一轮中仅提取一次表示；按长度和词元预算组成微批并在指定层提前结束无梯度前向。历史特征复用参考响应打包前向，中间层 hook 只保留动作前 prompt 末位向量。`fgpo_obs_only` 直接跳过历史向量的采集。同质结果组（$0/8$ 与 $8/8$）与无混合组轮次直接免除特征提取；优势与必要诊断计算完成后释放表示张量，PPO 与后续 checkpoint 保留参考概率和已固定优势。

余弦距离是可计算的近似匹配依据。它与真实续值差之间的关系由实验研究，当前实现没有将其解释为已知的 bisimulation 距离或概率校准结果。

## 3. 核心算法：同深度留一轨迹值共享

### 3.1 同伴集合与有限时域

对目标轨迹 $k$ 的第 $t$ 个动作前状态，同伴候选必须属于相同任务组、相同 benchmark、相同初始环境 seed，并在同一深度 $t$ 存在真实动作前状态。定义

$$
\mathcal P_{-k,t}=\{j:j\ne k,\ t<T_j\}.
$$

同组时域上限固定，因此相同深度对应相同剩余预算 $H-t$。整个目标轨迹从同伴数据中排除。已经结束的同伴在后续深度没有状态；实现不会将终态补成一个当前状态参与近邻比较。

这一选择保留了有限时域价值的时间索引。即使某个观察值在第 3 步和第 40 步完全相同，两者也不会直接混入同一状态估计。几何计算仅在每个深度的 $K\times K$ 矩阵内进行，无跨时间图拓扑和递归 Bellman 求解。

### 3.2 局部匹配核与同伴 Monte Carlo 值

在连续公开观测空间（**AppWorld** 与 **WebShop**）定义双视角距离与连续核权重：

$$
d_{kj,t}=\max\{1-(z^o_{k,t})^\top z^o_{j,t},\;
                       1-(z^h_{k,t})^\top z^h_{j,t}\},
$$

$$
w_{kj,t}=\mathbf 1[j\in\mathcal P_{-k,t}]
\left[1-\frac{d_{kj,t}}{\epsilon}\right]_+.
$$

其中 AppWorld 默认 $\epsilon=0.10$；WebShop 按研究配置取值 $\epsilon\in\{0.05, 0.10\}$（现行 `r7` 1.5B WebShop 采用 $\epsilon=0.05$）。

在离散局部观测空间（**ALFWorld** 与 **TextCraft**）令历史余弦距离 $d^h_{kj,t}=1-(z^h_{k,t})^\top z^h_{j,t}$，结合离散局部词元元组精确相等指示 $m^{\mathrm{obs}}_{kj,t}=\mathbf 1[o^{\mathrm{repr}}_{k,t}=o^{\mathrm{repr}}_{j,t}]$（ALFWorld 为当前动作前纯观测词元元组 `observation_ids`，TextCraft 为持久可见背包词元元组 `inventory`）定义：

$$
w_{kj,t}=\mathbf 1[j\in\mathcal P_{-k,t}]\,m^{\mathrm{obs}}_{kj,t}
\left[1-\frac{d^h_{kj,t}}{\epsilon}\right]_+,
\qquad \epsilon=0.10.
$$

数值实现将余弦距离截到 $[0,2]$。越接近目标状态，同伴权重越大；达到半径边界时权重连续变为零。定义支持质量和同伴续值

$$
S_{k,t}=\sum_j w_{kj,t},\qquad
\mu_{k,t}=\begin{cases}
\dfrac{\sum_j w_{kj,t}q_j}{S_{k,t}},&S_{k,t}>0,\\
0,&S_{k,t}=0.
\end{cases}
$$

每个 $q_j$ 都是同伴轨迹真实终局成功的缩放值。一次计算出的 $\mu$ 永远不会成为另一次 $\mu$ 的输入。支持质量同时反映邻居数量与距离；相同的邻居数可以有不同的 $S$。其数值作为连续收缩量使用，日志中可直接解释，含义限于当前组的可观测证据。

### 3.3 对称动作前基线、续值共享与未决同伴加权

先计算同任务留一轨迹先验与动作前/后对称共享强度（$\kappa=1, \lambda_{\max}=0.5$，移除正同伴续值门控）：

$$
\bar q_{-k}=\frac{1}{K-1}\sum_{j\ne k}q_j,\qquad
\lambda^{\mathrm{pre}}_{k,t}=\lambda_{\max}\frac{S_{k,t}}{S_{k,t}+\kappa},\qquad
\lambda^{\mathrm{post}}_{k,t}=\mathbf 1[t<T_k-1]\,\lambda_{\max}\frac{S_{k,t+1}}{S_{k,t+1}+\kappa},
$$

$$
\tilde b_{k,t}=(1-\lambda^{\mathrm{pre}}_{k,t})\bar q_{-k}+\lambda^{\mathrm{pre}}_{k,t}\mu_{k,t},\qquad
Z_{k,t}=(1-\lambda^{\mathrm{post}}_{k,t})G^{\mathrm{MC}}_{k,t}+\lambda^{\mathrm{post}}_{k,t}\bigl(r_{k,t}+\mu_{k,t+1}\bigr).
$$

记深度 $t$ 前已提前获胜完赛的留一同伴数为 $C^+_{-k,t}=\sum_{j\neq k}(1-\mathbf 1[t<T_j])Y_j$，定义同先验平滑的未决同伴权重与最终步优势：

$$
\omega_{-k,t}=\frac{(K-1-C^+_{-k,t})+\kappa}{(K-1)+\kappa},\qquad
\tilde A_{k,t}=\frac{\omega_{-k,t}}{R}\bigl(Z_{k,t}-\tilde b_{k,t}\bigr).
$$

基线 $\tilde b_{k,t}$ 与缩放 $\omega_{-k,t}$ 只读取目标动作前公开信息与同伴数据，在采样 $y_{k,t}$ 前完全确定。因 $\lambda^{\mathrm{pre}}$ 与 $\lambda^{\mathrm{post}}$ 对称使用同一 $\lambda_{\max}$，定义有界同伴势能 $\Phi_{-k,t}=\lambda^{\mathrm{pre}}_{k,t}(\mu_{k,t}-\bar q_{-k})$，则有无权标量步骤级的精确伸缩恒等式：

$$
Z_{k,t}-\tilde b_{k,t}=(1-\lambda^{\mathrm{post}}_{k,t})(q_k-\bar q_{-k})+\bigl(\Phi_{-k,t+1}-\Phi_{-k,t}\bigr),\qquad
\sum_{t=0}^{T_k-1}(\Phi_{-k,t+1}-\Phi_{-k,t})\equiv 0
$$

（末步令 $\lambda^{\mathrm{pre}}_{k,T_k}=0$ 且当同组均匀根匹配给出 $\mu_{k,0}=\bar q_{-k}$ 时 $\Phi_{-k,0}=0$；需注意实际参数更新将步级优势乘以随深度变化的 $\omega_{-k,t}$，$\sum_{t=0}^{T_k-1}\omega_{-k,t}(\Phi_{-k,t+1}-\Phi_{-k,t})=\sum_{t=1}^{T_k-1}(\omega_{-k,t-1}-\omega_{-k,t})\Phi_{-k,t}$，再乘以微批响应词元计数 $1/N_{\mathrm{tok},m}$、PPO 裁剪指示与各步 score 向量 $\psi_{k,t}$，因此实际参数梯度不会自动伸缩抵消为零）。

在单赢家（$M=1$）组中，唯一胜轨的同伴均为败轨（$\bar q_{-w}=\mu_{w,t}=0$ 且 $\omega_{-w,t}=1$），故其每步优势满足 $\tilde A_{w,t}=1-\lambda^{\mathrm{post}}_{w,t}\in[0.5,1]>0$；而在 $M\ge 2$ 的组中，若更快完赛的获胜同伴提前离场或后续状态分叉使动作前高支持基线 $\tilde b_{w,t}$ 高于后态目标 $Z_{w,t}$，绕路进入低支持区域的胜轨步可以合法地得到负优势 $\tilde A_{w,t}<0$（`r7` 在线 WebShop $M=6$ 组 `R25` 中已实测观察到该现象）。在高胜率组中全部 $M$ 条胜轨完赛后（$C^+_{-\ell,t}=M,\mu_{\ell,t}=0$），仍在漫游的孤立败轨步优势为 $\tilde A_{\ell,t}=-\omega_{-\ell,t}(1-\lambda^{\mathrm{pre}}_{\ell,t})\frac{M}{K-1}$：当 $S_{\ell,t}=0,\kappa=1$ 时精确等于 $-\frac{K}{K-1}\hat{p}(1-\hat{p})=-\frac{M(8-M)}{56}\in[-0.286,-0.125]$（在 `7/8` 组把晚期步扣罚从 `-1.0` 削减 `8x`，在 `M=1` 且 $S_{\ell,t}=0$ 时仅从 `-1/7` 削减 `12.5%` 至 `-1/8`），若存活败轨间发生局部匹配（$S_{\ell,t}>0$）则再乘以收缩因子 $(1-\lambda^{\mathrm{pre}}_{\ell,t})$。同质组直接零步返回；混合组信用量在参数更新前计算并冻结。

实现按 `[depth, trajectory, feature]` 组织表示，用批量 Gram 矩阵一次形成所有同深度距离，再按 `[depth, target, peer]` 归约。时间复杂度为 $O(HK^2D)$，权重存储为 $O(HK^2)$。表示使用 CPU float32，信用加权与回报归约使用 CPU float64，避免大量 Python 子图循环与小规模 GPU 同步。

## 4. 数学含义、偏差与明确成立的性质

### 4.1 基线的 score 恒等式

在固定任务初态、冻结的行为策略 $\pi_{\mathrm{old}}$ 和参考参数下，假设同伴轨迹使用独立随机流生成。记整个目标轨迹之外的数据为 $D_{-k}$，动作响应的 score 为

$$
\psi_{k,t}=\left.\nabla_\theta\log\pi_\theta(y_{k,t}\mid p_{k,t})\right|_{\theta=\theta_{\mathrm{old}}}.
$$

在固定任务初态、冻结的行为策略 $\pi_{\mathrm{old}}$ 和参考参数下，由于 $\omega_{-k,t}$ 与 $\tilde b_{-k,t}$ 均在当前动作前确定，有精确零期望恒等式 $\mathbb E[\omega_{-k,t}\tilde b_{k,t}\psi_{k,t}\mid h_{k,t},D_{-k}]=0$。进一步，令 $V^{\pi_{\mathrm{old}}}(h_{t+1})$ 为相同有限时域、相同行为策略下的真实后续回报均值，在独立与转移条件满足时，相对 $\omega_t$ 加权的 Monte Carlo 策略梯度存在如下共享偏差分解：

$$
\begin{aligned}
\Delta_t
&=\mathbb E[\omega_t(Z_t-\tilde b_t)\psi_t]-\mathbb E[\omega_t G^{\mathrm{MC}}_t\psi_t]\\
&=\mathbb E\!\left[\omega_t\lambda^{\mathrm{post}}_t
\bigl(\mu_{-k,t+1}-V^{\pi_{\mathrm{old}}}(h_{t+1})\bigr)\psi_t\right].
\end{aligned}
$$

推导从 $Z-G^{\mathrm{MC}}=\lambda^{\mathrm{post}}(r+\mu_{t+1}-G^{\mathrm{MC}})$ 出发。对当前转移及同伴数据条件化后，$G^{\mathrm{MC}}-r$ 的条件均值为真实续值，基线项由上式消去，从而把共享估计误差显式归于 $\mu_{t+1}-V^{\pi_{\mathrm{old}}}(h_{t+1})$。需明确指出，上述等式比较的对象是 $\omega_t$ 加权后的 Monte Carlo 梯度；相对于无权原始官方成功目标梯度 $\mathbb E[G^{\mathrm{MC}}_t \psi_t]$，只要存在提前获胜完赛同伴（$C^+_{-k,t}>0$），还包含未决重加权项 $\mathbb E[(\omega_t-1)G^{\mathrm{MC}}_t \psi_t]$，且多微批、裁剪并经词元均值与 AdamW 预条件化的完整更新仍属于正则化替代目标而非精确无偏梯度。

若在所讨论分布上有 $|\mu-V|\leq\delta$、$\lambda\leq\lambda_{\max}$，并且 $\mathbb E\|\psi_t\|\leq L_t$，则归一化后

$$
\|\Delta_t/R\|\leq \lambda_{\max}\delta L_t/R,\qquad
\left\|\sum_{t=0}^{H-1}\Delta_t/R\right\|
\leq\frac{\lambda_{\max}\delta}{R}\sum_{t=0}^{H-1}L_t.
$$

这是带显式条件的误差分析；支持量消融使用权重上界 1。余弦半径没有给出已知的 $\delta$；有限上下文、表示混叠和有限同伴采样均影响实际误差。支持收缩提供可控的共享幅度，正式实验进一步测量其预测误差与任务收益。r3 的论证以这些条件为边界，不使用“精确无偏续值”或“全局 DAG 正确性”作为前提。

### 4.3 值域、同质组及失败前缀

因为所有核权重非负、$\kappa>0$ 且 $0\leq\lambda\leq1$，有

$$
0\leq b_{k,t},\mu_{k,t},Z_{k,t}\leq R,\qquad -1\leq A_{k,t}\leq1.
$$

若某组所有 $Y_j$ 相等（全失败 $0/8$ 或全成功 $8/8$），该组作为同质组满足先验、基线和目标全部相等，任务优势严格为零，直接免除特征提取返回零优势。熵正则提供有限的分布扩散，参考正则约束偏离；二者在混合组批次中发挥作用，整体算法在无混合组轮次直接执行零步旁路而不靠纯正则更新参数。`all_zero_groups`、`all_one_groups` 与 `homogeneous_round_bypass` 会直接记录这一状态。

一个失败轨迹中的动作可以因下一状态的成功同伴获得正优势，具体条件是 $\lambda^{\mathrm{post}}_{k,t}\mu_{k,t+1} + (1 - \lambda^{\mathrm{post}}_{k,t})q_k > \tilde b_{k,t}$。在 `fgpo` 移除正同伴门控后，在单赢家（$M=1$）组中，任何获胜轨迹步骤只要其下一状态在失败同伴处获得正支持（$S_{w,t+1}>0,\mu_{w,t+1}=0$），其正优势均会依支持量收缩为 $1-\lambda^{\mathrm{post}}_{w,t}\in[0.5,1)$；该收缩仅由同伴支持量 $S_{w,t+1}$ 驱动而缺少动作质量标签，因此对无效死胡同步与在失败同伴中高频出现的必要成功前缀会施加相同的幅值收缩。`positive_failed_steps` 与 `mean_pending_weight` 用于解释失败轨迹正信用位置与完赛未决权重变化。

## 5. PPO 更新与原子混合组权重

每个步骤的 $A_{k,t}$ 应用于该响应的全部实际采样词元。人工补入的 ChatML 闭合词元、环境观测、滚动重播的旧响应都没有新的损失所有权。令第 $i$ 个响应词元的概率比为 $\rho_i=\exp(\ell_i^\theta-\ell_i^{\mathrm{old}})$。

实现使用 clip $0.2$、dual clip $3$ 的 PPO token 损失，叠加系数 $0.001$ 的条件熵奖励和系数 $0.01$ 的截断参考 k3 正则。参考项为

$$
d_i=\ell_i^{\mathrm{ref}}-\ell_i^\theta,\qquad
k_i=\min\{10,\exp(d_i)-d_i-1\}.
$$

这些是保存的行为样本上的正则化 surrogate。裁剪、抽样参考项和近似续值共同决定实际更新；训练并不宣称每次参数变化都等于官方成功目标的精确梯度。默认全参数 bf16/SDPA、无 dropout，AdamW 学习率 $10^{-6}$、weight decay $0.01$，全局梯度范数裁剪 $1$。

F-GPO 及其自身消融在 `r7` 采用 `loss_normalization="token"` 与 `entropy_normalization="token"` 的均衡打乱微批更新（`ppo_epochs=1`, `ppo_minibatch_rows=256`, `shuffle=true`, `mixed_token_mean`）。一轮首先筛选满足 $1\le\sum_{k\in g}Y_k\le K-1$ 的混合结果组 $\mathcal G_{\mathrm{mix}}$；若 $M=|\mathcal G_{\mathrm{mix}}|=0$，整轮跳过参考前向、PPO 与 AdamW（执行 `0` 次优化器步）；若 $M\ge 1$，将混合组全体响应步按 `shuffle=true` 打乱并均衡划分为 $B=\lceil N_{\mathrm{rows}}/256\rceil$ 个微批 $\mathcal B_m$，记微批响应词元数为 $N_{\mathrm{tok},m}=|\mathcal B_m|$，对策略、KL 与熵统一按词元均值化：

$$
\mathcal L_m^{\mathrm{r7}}=\frac{1}{N_{\mathrm{tok},m}}
\sum_{i\in\mathcal B_m}\bigl(\ell^{\mathrm{PPO}}_i+\beta k_i-\alpha H_i\bigr).
$$

统一词元均值权重彻底消除了 `r6` 逐阶归一化对长思考输出的 `ppo_kl` 稀释以及与 `1/N_tok` 熵正则的量纲失衡，同时使每轮最大更新步数收敛至与已发表 1.5B 基线（`256` 行、`1` epoch、`1e-6` 学习率）一致的稳定区间。参数在每个微批更新后变化。三个已发表 baseline 保留作者 virtual-rank / microbatch 内的 token mean 及其聚合权重、优势归一化和 minibatch 设置。

对 baseline 的一次 optimizer update，令 $W$ 为来源配方的 virtual world size，$A$ 为固定 gradient accumulation，$T_{r,b}$ 为 rank $r$ 上 microbatch $b$ 的真实响应词元数，则对应目标为

$$
\mathcal L_m^{\mathrm{published}}
=\sum_{r=1}^{W}\sum_{b\in m_r}
\frac{1}{W A T_{r,b}}
\sum_{i\in\mathcal R_{r,b}}\ell_i.
$$

最后不足额 minibatch 保持同一 $A$。同一响应的复制记录若落在同一个 optimizer update，其系数可相加后复用一次模型计算；它的目标仍由每次出现所属的 $T_{r,b}$ 定义。源码 batch copy、balance 和优势计算的先后关系同样保留，具体来源设置见第 7.2 节。虚拟 rank 在本后端定义目标权重及分组，实际 GPU 放置另由运行配置决定。

## 6. 系统实现与性能优化的成立条件

### 6.1 因果打包与参考计算

完整段包含按时间串接的动作和观测。在相同词元、位置编码、因果 mask 及参数下，响应前缀位置的隐藏态可以从整段前向获得；后面的词元被因果注意力屏蔽。这允许一个段内的多个响应共享主干计算，同时保持各自真实前缀。滚动窗口重新开始一个段，旧前缀依赖的 KV 不能直接作为新窗口的条件继续使用。

冻结参考权重仅在当轮存在混合结果组时集中安装，完成混合组轨迹的响应概率与所需特征计算后恢复 actor；全同结果轮直接旁路该阶段。同段所有拥有损失的响应合并成有界 head chunks，按每条响应的 `span.start-1 : span.stop-1` 因果位置评分；参考前向与 `response_head` 在段粒度复用 `aggregate_model_calls` 并携带真实 `round`/`group_id` 归属；目标 token 一次上传，每段的分数一次下载。每块词表 logits 在下一块计算前释放。历史 hook 只截取所需 prompt 向量；ALFWorld 省略局部观测模型前向而直接使用纯观测词元元组匹配，AppWorld 局部观测采用去重、长度排序和词元预算微批。此处的复用范围以相同参数版本和实际段为单位。

信用阶段形成同深度匹配权重后及时释放临时特征。完成优势及聚合诊断后调用 `ReferenceRound.release_features()`，去除后续更新和 checkpoint 中无用途的表示引用，保留冻结参考的响应条件概率；行为段 `SegmentRecord` 一次冻结共享，并在轮末日志落盘后立即释放已完成轮的 `groups`/轨迹与局部引用，避免两轮样本在主机内存重叠常驻。

### 6.2 持久环境进程与提前派发

每条轨迹通过独占 lease 使用一个官方环境实例。持久进程池复用 Python 进程和已加载依赖；新 lease 重新建立该任务的隔离状态，旧任务数据库与可变环境对象不会在轨迹间共享。池的容量按运行配置约束，进程启动、复用与环境调用记入成本日志。

GPU 仍以单主线程处理模型批次。每个槽位独立维护 `torch.Generator`；采样保持各槽位自己的随机流，将 token 与 log-prob 合并为一次 CPU 读取。某条响应遇到实际终止符或生成长度上限后，立即将解析、原生环境执行和公开观测投影派给 CPU 工作线程，此时其余槽位仍可继续 GPU 解码。

下一交互轮保留 turn barrier：本轮所有活跃槽位的反馈完成后，才构造下一轮模型输入。F-GPO 在轮内重叠环境调用与解码。已发表 baseline 同样采用组内批量模型采样与并行环境调用，完整异长提示通过 `prefill_batch` 进入同一批次；其提示、记忆和动作解析仍按作者配方构造。训练温度为 1，评估温度为 .4，`top_p=1`；保存的行为 log-prob 来自实际采样温度。独立 generator 保留各槽位随机流所有权，批形状与浮点舍入变化可能影响最终离散结果，实际执行配置随结果保存。

### 6.3 几何增长 KV 与真实内存

`cache.py` 的 arena 在容量充足时原位写入 KV，容量不足时按几何增长扩容。它维护每行有效长度、物理跨度、attention mask 与逻辑位置。当主机已知 `right_padded` 且所有物理列全部有效时，cached 单 token 解码传入 `attention_mask=None` 避免冗余掩码与 GQA 头展开，同时完整保留有界填充/ragged 及无缓存显式 `position_ids` 语义；依赖特定后端/版本的异长 GQA 重构（如 varlen/paged 或验证后的 masked GQA）保留为后续工作。行退出与 `select_cache`/`split_cache` 为继续行保留既有扩容余量、为停放行仅分配占用长度并按 `valid_spans` 裁去尾部全空列；`_sample_turn` 直接移交停放分片，由 `_advance_cohort` 仅合并真实续用的 `"append"` 行。`append_cache` 对长增量重放按词元预算分段并保持物理列不越界。已紧凑行在 `compact` 中跳过冗余 `index_select`。变长批次仍然具有 padding；`crop` 改变逻辑长度并保留底层容量，不能据此声称显存已经释放。

这项设计减少逐 token 全前缀拼接分配与退出重合并拷贝，同时保留 padding、位置、末词元单次追加与独立 generator 语义。根前缀可以共享一次 prefill 结果，分叉后各槽位持有可独立更新的缓存。窗口丢弃历史时重新 prefill 新窗口。行选择的 indices 按设备上传一次供各层复用；已结束响应的 padding 槽位跳过词表投影，原分布保留在独占缓存对象中。连续解码已处于 eval 状态时，跳过重复模块遍历。数学条件一致与浮点逐位一致分别陈述，实际吞吐由正式日志报告。

以 [Qwen2.5-7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json) 的 28 层、4 个 KV 头、头维度 128 和 bf16 为例，8 个槽位的纯 KV 张量在容量 4096 时为 $28\times2\times8\times4\times4096\times128\times2$ 字节，即 1.75 GiB；容量 32768 时为 14 GiB。这是指定张量的容量计算，整机峰值还包括权重、梯度、激活、输出头、缓存扩容临时量和分配器保留空间。

### 6.4 输出头分块与一次主干反向

每个被当前 minibatch 使用的段保留其全部必要前缀，并将主干前向截到最后一个被选响应的末位。从 $f_\theta(S)$ gather 当前响应位置形成 $H_{\mathcal R}$；显式 hidden adjoint 缓冲按真实响应词元数分配。输出头按默认 64 个响应词元分块，每块以 detached leaf 计算损失并立即反向，累积输出头参数梯度及响应隐藏态梯度，随后释放该块词表图。全部块完成后，对 $H_{\mathcal R}$ 执行一次 backward，autograd 经 gather 将梯度送回正确的主干位置：

$$
\nabla_\theta\mathcal L
=\nabla_\theta\mathcal L\big|_{H_{\mathcal R}\ \mathrm{fixed}}
 +\left(\frac{\partial H_{\mathcal R}}{\partial\theta}\right)^\top
   \frac{\partial\mathcal L}{\partial H_{\mathcal R}}.
$$

在相同参数和损失定义下，这实现一阶链式法则；若输出头与输入 embedding 绑权，两部分参数梯度自然相加。每段保留一个活跃词表块和响应位置的显式梯度缓冲，主干图只遍历一次。

词表头的 `_LogProbabilityEntropy` 保存 logits 与行级统计，在反向中重建概率并合并上游的 logprob、entropy 系数。其一阶导数为 $\partial\log p_y/\partial z_v=\mathbf1[v=y]-p_v$ 和 $\partial H/\partial z_v=-p_v(\log p_v+H)$。完整词表参与概率和熵计算，跨块保留的 softmax 损失图被消除。主干使用非重入梯度检查点。`activation_offload` 默认关闭；实际内存峰值通过正式 workload 的阶段日志记录。

### 6.5 有界 CPU AdamW 与阶段计时

AdamW 的 fp32 主权重、一阶和二阶矩保存在 CPU。默认 64 MiB 传输桶，并为每个设备和 dtype 分配两个有限的锁页传输槽。GPU 梯度按桶传回，事件完成后 CPU 才读取；更新后的参数写入权重镜像再传回，缓冲区在传输结束后方可复用。GPU 全局范数先按设备聚合，避免每个参数触发一次 CPU 标量同步。参数更新保持完整参数集合，传输空间随桶预算约束。

模型主干和词表头热路径移除显式逐调用同步，其日志耗时标为 `host_dispatch`；解码、参考及 PPO 段内部按 role、真实轮次与组归属复用 `aggregate_model_calls` 聚合模型调用、真实 token、位置、calls 和 padding 计算计数，PPO 裁剪计数、有序 FP64 差值和与最大值在输出设备按段归集并在段/cursor 持久化边界读回，减少逐块同步、JSON 写入与事件常驻。环境事件即时保存。正式阶段边界进行 CUDA 同步，记录 `synchronized_wall` 耗时与每张卡的 allocated、reserved 和真实 peak bytes。嵌套阶段峰值标明所属外层阶段；重叠环境时间与 GPU 时间各自解释，端到端成本使用 ledger 的 wall/device-seconds，包含启动恢复、checkpoint 发布与最终回收发布边界。吞吐、实际 token、环境步数、GPU 小时和峰值显存均由已有正式运行账本产生。

用户性能提案中的异步派发、持久环境、减少同步、批量表征、几何缓存、有界传输与分块输出头思路已按上述条件纳入。所有模块均以 r3 数学对象为准；旧三遍估计器的删除属于算法变更。理论容量计算与预计收益可以指导配置，具体加速倍数和 benchmark 增幅由服务器结果填写。

### 6.6 持久化与主机内存生命周期

固定参考的安装/恢复复用 optimizer 已有 FP32 master：已更新参数按原 Adam 写回路径先在 CPU 舍入为 BF16，再分桶写回 actor；未更新参数从不可变初始参考恢复。模型导出沿官方 Transformers/safetensors 分片路径执行，默认分片预算 1024 MiB，避免提前把整 actor 复制到 CPU。CPU Adam 的 master/moments 精度与更新关系保持相同。公共 runner 在加载模型前按实际权重 header 预留完整 CPU 工作集，同一用户的合作 cell 使用主机文件锁共同登记预算，包含梯度恢复和环境余量。

完整恢复快照在初始、每 8 轮（Colab 为 60 轮）、训练结束与托管中断的安全边界保存；采样/reference 进度只在 actor/Adam 与已保存版本相同时更新。GCS 后端使用版本化 `fgpo-gcs-manifest-v2` 与增量 `fgpo-gcs-journal-v2`（同时兼容读取 `v1`），复用不变静态产物并以几何分级合并的有界增量段持久化 `costs.jsonl`、`training.jsonl` 与 `evaluation-*.jsonl`；所有读路径（含 `current`/`inspect`/`status`/`pull`/`restore_cell`/`prepare_resume`/`load`/`restore_final` 与 `resources` 估算）均通过 `open_snapshot`/`_reader_pins` 先发布并复核 `readers/<lease>.json` 读租约后再读取与下载对象并在异常和正常退出统一清理（活跃读租约存在时，周期性 `sync-logs` 延迟清理旧 LOGS 对象，全量 GC 保留其引用闭包）。快照始终包含同版本参数、moments、pending gradients、轨迹、优势、cursor 和 RNG；传输统计按 `actor`、`optimizer`、`gradients`、`reference`、`teacher`、`state`、`artifacts` 与 `journals` 区分新上传与复用字节，另单列 `metadata_uploaded_bytes`。启动恢复区间（`runner.cold_asset_restore`、`runner.cell_startup`）锚定入 `wall_seconds` / `allocated_device_seconds`，完成态先完成恢复态回收与计量 `publish_final_logs`（记录 `checkpoint.final_publish_logs`），再冻结重写终态 `cell_summary.json` 并执行明确排除在自身成本外的终端导出 `export_final_terminal_state` 同步至 `FINAL`/`LOGS`（中断完成态可在 `--resume` 收敛）。强制终止会重做最近保存点之后的工作；恢复时截回训练统计，物理成本日志按等长前缀合并保留实际发生的完整事件并将未提交半写尾与恢复放弃尾转为一次计入的单调 `discarded_incomplete_tail_bytes` 及 `usage_complete=false`。模型版本、日志字节游标、参数文件和 CURRENT 的发布顺序共同定义一致恢复点。

默认持久产物为有效配置/来源、任务清单、组级 claim 诊断、成本、逐任务官方评估和最终模型。历史原始轨迹和逐 token reference 数据仅在显式 `retain_training_artifacts=true` 时另存，训练 group 为完整轨迹的单一持久化来源；当前恢复输入仍随快照保存。AppWorld 先完成官方评分并把结果复制进轨迹，关闭环境后删除当前实例创建的 episode 输出目录。最终模型与完成摘要持久化后再回收恢复状态及默认不保留的原始文件。上述实现约定不增加实验或改变研究 claim；容量公式与代码审计范围见 [优化方案](fgpo_optimization_proposal.md)。

## 7. 正式研究协议与实验解释

### 7.1 按 claim 保留最小实验

当前训练以完整 F-GPO 的官方表现和实际成本为主，TextCraft 与 WebShop 的执行顺序由[当前排程](fgpo_experiment_schedule_current.md)维护。已启动的消融保存并暂停，未启动的消融移出执行队列。第 7.4 节保留各机制对应的诊断定义。

### 7.2 已发表 baseline 的来源、配方与迁移

正式 baseline 为 [GiGPO / NeurIPS 2025](https://arxiv.org/abs/2505.10978)、[HGPO / ICLR 2026](https://openreview.net/forum?id=T8Dev99qnz) 和 [GraphGPO / ICML 2026](https://proceedings.mlr.press/v306/cheng26e.html)。作者实现统一固定到 [verl-agent 提交 `20bd331bdbc9026a5668e11362178e10ab7400c8`](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)。[baseline_specs.py](../src/fgpo/baseline_specs.py) 集中记录论文版本、上游具体文件、训练设置和 paper/code 差异；[published_baselines.py](../src/fgpo/published_baselines.py) 实现信用规则，[published_batching.py](../src/fgpo/published_batching.py) 保留来源 batch 和损失权重语义。

| 方法 | 来源代码的执行规则 | 已记录的论文与代码差异 |
|---|---|---|
| GiGPO | 轨迹级优势加公开 anchor-state 组内优势；保留 γ=.95、sample std、逐行语法惩罚及发布的奖励处理顺序 | 发布代码的 episode 统计按重复的 step rows 计算，论文式 3 使用轨迹级统计；当前 invalid-action penalty 在折扣回报之后扣在该行，episode score 同样逐行扣除 |
| HGPO | 按不同长度的观测后缀分组，使用 population std、长度权重和 `base_group=False` | 源码分组只读 observation suffix，actor 记忆仍含动作；只聚合非零优势并重归一化，后缀长度 $k$ 使用 $(k+1)^\alpha$，与论文式 7 的索引及全层求和写法分别记录 |
| GraphGPO | 以公开状态构图，到成功目标求最短路径，发布实现使用 $10\gamma^{d(\mathrm{next})}$，γ=.10，再结合轨迹级优势 | 论文式 4 另含转移代价指数；发布 episode 标准化沿用 step-row 统计。ALFWorld 提示含公开动作/反馈历史摘要，图键还包含排序后的可用命令 |

这些差异决定当前实现选择 **released-source recipe**。作者代码的历史、折扣、优势统计、生成预算、训练长度及 loss reduction 保持各自定义；把所有 baseline 替换为 F-GPO 的奖励和归一化会改变比较对象。

来源配方的 `adjust_batch` 通过 `np.random.choice(..., replace=False)` 选择已有记录并在末尾追加，使行数适配规定的 divisor；`balance_batch` 使用 equal-size Karmarkar–Karp 按完整 prompt 加响应的有效长度划分，每个 partition 内按索引排序。GiGPO 在 copy 和 balance 后计算优势，HGPO/GraphGPO 先计算优势，再 copy 和 balance。每个 virtual rank 独立形成连续 minibatch，同一序号的 rank minibatch 合成一次参数更新。复制只影响已采样记录的统计及 loss 权重，实际轨迹与环境采样仍按原始数据计数。

| 来源配方 | Virtual world size | 每 rank microbatch | Global row minibatch | 固定 accumulation | Copy divisor |
|---|---:|---:|---:|---:|---:|
| GiGPO 公开 1.5B 全参配方，迁移到本研究 7B | 2 | 32 | 256 | 4 | 64 |
| HGPO 7B 配方 | 4 | 8 | 256 | 8 | 32 |
| GraphGPO 7B 配方 | 4 | 4 | 64 | 4 | 64 |

ALFWorld 接入作者发布的公开观测、原生动作与提示配方。HGPO/GraphGPO 的来源就是 7B，GiGPO 公开全参脚本使用 1.5B，本研究明确标注其 **7B backbone 迁移**并保留上述来源 batch 配方。所有参考和 PPO 概率都在最终实际采样前缀上重算。

执行后端统一为本仓库的 PyTorch/SDPA、组内批量采样、持久官方环境与 CPU-master AdamW。复现范围是算法及发布配方在该后端的实现，计算系统与原作者分布式系统分别注明。主比较因此支持完整方法级的质量—成本判断。

### 7.3 实际预算与完整官方评测

| 项目 | F-GPO 及自身消融 | GiGPO | HGPO | GraphGPO |
|---|---|---|---|---|
| 训练 rounds | 60 | 150 | 160 | 150 |
| 每轮任务组 × 每组轨迹 | 16 × 8 | 16 × 8 | 16 × 8 | 16 × 8 |
| 每种子完整轨迹量 | 7680 | 19200 | 20480 | 19200 |
| 信用折扣 γ | 1 | .95 | .95 | .10 |
| ALFWorld 每步生成上限 | 512 | 512 | 512 | 512 |
| ALFWorld 提示 / 历史规则 | 32768 总上下文内的初始锚定与完整轮次滚动（三块整数帧） | 2048 prompt，最近两轮记忆，超限 error | 4096 prompt，最近两轮记忆，作者 left truncation | 2048 prompt，最近两轮记忆及公开状态摘要，超限 error |
| PPO 批次划分 | `r7` 1.5B 主线：均衡打乱 256 行混合组微批；历史 7B `V4`：混合结果组 $K=8$ 原子整组 | 256 row | 256 row | 64 row（7B 配方） |
| Loss reduction | `r7`：`mixed_token_mean`（策略/KL/熵统一批内词元均值）；历史 `V4`：固定分母 $Z_m=8$ | 作者 rank / microbatch token mean | 作者 rank / microbatch token mean | 作者 rank / microbatch token mean |

上表主基座对照按公开 7B 协议组织（全参数 bf16/SDPA，种子 0、1、2，AdamW `lr=1e-6`、weight decay `0.01`、`betas=(0.9,0.999)`、`epsilon=1e-8`、全局梯度范数上限 `1`、`ppo_epochs=1`、`clip=0.2`、`dual_clip=3`、`entropy=0.001`、`kl=0.01`、训练温度 `1`/评估 `0.4`/`top_p=1`）。F-GPO 信用参数默认 $R=10,\epsilon=0.10,\kappa=1,\lambda_{\max}=0.5$：现行 `r7`（`fgpo-r7-token-mean-v1`）采用对称势能与未决同伴权重 $\omega_{-k,t}$（无正同伴延续门控，仅 `fgpo_exact_first_hit` 保留门控）；历史 7B `V4` 启用了正同伴门控与原子组轨迹分母 $Z_m=8$。baseline 的奖励尺度和语法处理以来源配方为准。

上表描述已发布的家务基准配方。TextCraft 与 WebShop 的实际预算、官方划分和追加对照见[当前排程](fgpo_experiment_schedule_current.md)。正式终评按各自既定任务计算，部分结果保留原分母与完成状态。

主表同时列最终质量、实际轨迹量、环境步数、响应词元、端到端时间与设备小时。各方法按已声明的配方终点报告，训练窗口差异随结果呈现；训练组成功率曲线解释学习进展，最终测试分数单独列出。当前计划的效率结论以质量与实际成本的联合结果为依据，不增加中途测试集挑选、额外预算扫描或独立系统基准实验。

### 7.4 只复用必要的 claim 诊断

| Claim | 直接实验 | 复用的诊断及用途 |
|---|---|---|
| 完整方法的任务表现和实际成本 | `comparison` 的四种完整方法 | 最终官方指标、真实样本与 token 数、wall/device-seconds、阶段峰值、语法/合法动作率与逐批梯度范数，形成质量—成本报告 |
| 历史条件化改善本方法的匹配 | 完整 F-GPO 与 `fgpo_obs_only` | 匹配覆盖、支持质量和留一 outcome Brier，解释距离改变后的预测表现 |
| 下一状态共享提供任务收益 | 完整 F-GPO 与 `fgpo_no_transfer` | 共享比例、正同伴门控步数、回报移动量及训练进展，确认实际使用的共享量与任务变化 |
| 连续支持收缩有作用 | 完整 F-GPO 与 `fgpo_no_support_weight` | $S$、有效同伴数与 $\lambda$，解释共享强度改变后的任务及预测结果 |

这些量全部从正式训练或官方评测的既有记录提取，不触发新 rollout。同质组、失败轨迹正优势计数、批次裁剪前范数与首批次对数概率一致性误差作为已有训练与信用输出的解释字段保留，正文图表按上述 claim 选取。主留一诊断定义为

$$
\operatorname{Brier}_{\mathrm{LOO}}
=\frac{1}{\sum_k T_k}\sum_{k,t<T_k}
\left(b_{k,t}/R-Y_k\right)^2.
$$

另记录任务先验 Brier 和有匹配状态上的未收缩 $\mu/R$ Brier。目标轨迹结果在预测时被留出，随后用于计算预测损失。若真实条件成功概率为 $p$，条件平方误差可分为 $(\widehat p-p)^2+p(1-p)$，因此该指标包含 Bernoulli outcome 噪声，其用途是解释留出结果预测；真实 Q 误差和理论中的 $\delta$ 是不同对象。Brier 不回流为训练权重。

诊断按真实步骤统计，长轨迹贡献更多预测样本；跨组聚合按相应样本数加权，并保留每组原始统计。每个 claim 由对应的正式结果和这些既有统计支撑，诊断自身不扩展训练矩阵。

## 8. 运行身份与交付状态

协议 `fgpo-r7-token-mean-v1` 的运行按 comparison 与 ablation 分目录保存。同一 study 可保存各自的单元和汇总，并引用已有的完整 F-GPO 结果。当前启动入口由 `configs/fgpo-r7-1p5b-*.json` 与 `scripts/{colab_run_r7_1p5b_queue,run_fgpo_host_r7_1p5b_queue}.sh` 指向各主机与 Colab 配置。

恢复由 `effective_training_identity/require_training_identity` 约束：方法及发布配方来源、模型、任务、种子、历史、采样、信用、PPO 和 Adam 的有效语义保持一致。GPU 放置、KV 容量、offload、head/表示微批、staging、保存周期/分片、原始文件保留与主机预留等资源项可以调整；保存的 CUDA RNG 流数量保持一致。完成采集组、参考概率、优势与更新 cursor 在明确操作边界恢复。

指定服务器上的研究记录和正式日志保存实际命令、检查点与官方结果。
