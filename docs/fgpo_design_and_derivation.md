# F-GPO r3：有限时域、历史条件化的跨轨迹续值共享

版本日期：2026-11-16。执行协议：`fgpo-r3-published-v3`。核心算法标识：`r3_history_conditioned_peer_value_sharing`。

本方案研究一个具体问题：在只有公开交互记录和终局成功信号的长时程任务中，怎样利用同任务的独立同伴轨迹改善中间动作的价值估计，并控制近似状态匹配造成的误差。r3 将研究对象收敛为**有限时域下、按实际可见历史匹配的同伴 Monte Carlo 续值共享**。终局成功定义训练目标，距离与同伴支持度决定共享量，实际行为前缀贯穿采样、参考评分和 PPO 更新。

本文与 [数学及实现契约](fgpo_math_contract.md) 共同定义现行方案，对应实现位于 `src/fgpo/`。旧 TF-BEFP、三遍图回传及随机 RMS 设计可以通过基准提交 `977bb86ac31a4b1f95ea26b7b53334be5cadf57e` 追溯。当前交付包含静态实现与源代码审阅；正式服务器训练、性能测量及完整 benchmark 结果待执行。本文以公式和源码说明估计器及优化路径，数值结果由规定的正式实验填写。

## 1. 2026 年研究进展与 2027 年研究定位

[GiGPO，NeurIPS 2025](https://arxiv.org/abs/2505.10978) 将轨迹级相对优势与重复 anchor state 上的步级相对优势结合，建立了无需学习 critic 的多轮智能体训练路线。2026 年的工作已经深入到历史条件、近邻分组、跨轨迹图信用及小样本可靠性。因此，r3 的贡献需要由明确的估计对象、误差分析、已发表方法比较和自身机制消融共同支撑。

| 工作与一手来源 | 2026 年的研究内容及方法 | r3 的对应研究边界 |
|---|---|---|
| [HGPO，ICLR 2026](https://arxiv.org/abs/2602.22817) | 按历史上下文一致性构造多级组，对各级优势自适应加权，处理同一局部状态下历史不同的问题。 | 历史条件化已有直接先例。r3 研究实际有界 actor 历史与同深度续值共享的组合，并记录历史表示对预测质量的影响。 |
| [BiPACE，2026 预印本](https://arxiv.org/html/2606.25556v1) | 使用当前 actor 的 prompt 末位隐藏态近邻聚类；Q-style PACE 在簇内按动作聚合，形成局部动作值与状态值之差。 | 论文的 prompt 包含近期历史。r3 使用冻结参考表示、同深度匹配、整轨迹排除和下一状态续值；本地观察值消融的结果只说明自身设计。 |
| [GraphGPO，ICML 2026](https://proceedings.mlr.press/v306/cheng26e.html) | 将多条轨迹合成状态转移图，用到成功目标的图距离变化分配动作信用。 | 失败轨迹前缀获得跨轨迹信用已有先例。r3 直接共享真实终局回报，避免将最短成功路径解释为策略平均续值。 |
| [RewardFlow，NeurIPS 2026](https://arxiv.org/abs/2603.18859) | 通过共享状态连接轨迹，从成功节点沿图反向传播与最短路径距离有关的势函数，以势差形成动作级信号。 | 图传播与稠密信用已有直接竞争方案。r3 的共享值始终来自同伴的实际 Monte Carlo 结果。 |
| [G2PO，2026 公开版本](https://arxiv.org/abs/2606.22995) | 合并相同观测形成全局转移图，聚合状态值，按边计算并标准化 TD 信用；包含 AppWorld 实验。 | r3 按剩余时域区分状态，并采用固定奖励尺度。会议归属以正式出版来源为准，当前所引公开页未标会议。 |
| [ECPO，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.172/) | 对规范动作的少量样本估计进行收缩，并依据动作内方差调节信用，处理偶然高回报与稀疏证据。 | 小样本信用可靠性已有专门工作。r3 只用可观测距离与支持质量控制共享，诊断来自正式日志，不增加校准器或额外训练目标。 |
| [MCGPO，2026 年 10 月预印本](https://arxiv.org/abs/2610.13755) | 用动作前状态摘要和规则标注进度匹配上下文，固定分折隔离候选动作与定向偏好的终局证据，以支持权重缩放 GraphGPO 之外的成对学习项。 | 分折隔离与支持加权也已有近邻工作。r3 的估计对象是同深度平均续值，使用公开历史表示与单一 PPO 更新，无 rubric 标注或额外成对目标。 |

这些工作共同说明，2027 年仍可推进的研究问题是：在有限采样、部分可观测性和有限上下文共同存在时，**哪些跨轨迹信息可以有效共享，共享多少，以及这种共享何时改善官方成功率**。r3 的具体候选贡献是同深度的留一轨迹续值估计、可解释的连续共享强度、与实际行为条件一致的训练实现，以及对应的误差与成本证据。

论文定位围绕这组可检验的贡献展开。历史表示、留一基线、核权重和 PPO 本身属于已有工具；算法组合的价值由完整 benchmark、与已发表近年方法的关系及实际成本共同体现。当前正式主比较只有 F-GPO 与 **GiGPO、HGPO、GraphGPO**。三种 baseline 分别代表局部 anchor 信用、历史层级信用和跨轨迹图信用，覆盖本方案必须面对的直接竞争关系。表中其余工作用于解释研究边界，当前最小实验计划不为其新增单元。

比较器采用作者发布源码的完整算法配方，并将来源固定到明确提交。ALFWorld 保留来源算法、提示和 batch 语义；GiGPO 从公开 1.5B 全参配方迁移到本研究 7B 基座，HGPO/GraphGPO 采用作者 7B 配方。AppWorld 另标注为 benchmark 迁移。统一 PyTorch 执行后端承载全部方法，论文呈现完整方法的质量与成本。机制归因由三个 F-GPO 自身消融完成。

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

`public.py` 只接受标准公开输入。ALFWorld 使用环境公开观测和可用命令；AppWorld 保留初始任务、系统说明及初始 API 引导信息，后续过长 stdout 在入史时进行固定的首尾截取。空 stdout 在表示和消息中使用明确占位文本。内部状态哈希、评估器中间成功标签和隐藏数据库状态不进入模型或匹配特征。

`filtration.py` 在整数词元空间构建不可变块。给定生成预算 $M_{\mathrm{gen}}$，提示上限为 $C_{\mathrm{prompt}}=C_{\max}-M_{\mathrm{gen}}$；滚动时优先将最近完整轮次截到 $0.75C_{\mathrm{prompt}}$ 的水位。若初始锚定块与最新完整轮次超过这一预留水位，则保留这两个必要块，并允许使用至提示上限；超过提示上限时显式报错。整个过程按完整轮次处理。

每个真实响应只归属一个训练段。为构建新上下文而重播的历史响应没有新的损失所有权。反馈仅在下一次实际决策需要时进入 actor 段；终末反馈保存在步骤和轮次记录中，`observation_span=None`，没有额外 feedback-only 评分段。三个正式自身消融都保留完整 F-GPO 的 actor 历史、生成预算和 PPO 条件，使各自的唯一变化落在匹配距离或共享规则上。

### 2.3 历史与局部观测表示

在冻结参考模型上，取实际动作前提示末位的倒数第 8 层表示并归一化，得到 $z^h_{k,t}$。将当前公开局部观测单独编码，在相同层取末位表示并归一化，得到 $z^o_{k,t}$。前者包含本次决策真实可见的任务和历史；后者保留当前观测的局部相似性。两种向量均停止梯度。

相同的局部观测词元序列在一轮中仅提取一次表示；按长度和词元预算组成微批。历史特征复用参考响应打包前向，中间层 hook 只保留动作前 prompt 末位向量。独立观测表示在指定层提前结束无梯度前向。`fgpo_obs_only` 直接跳过历史向量的采集。优势与必要诊断计算完成后释放表示张量，PPO 与后续 checkpoint 保留参考概率和已固定优势。

余弦距离是可计算的近似匹配依据。它与真实续值差之间的关系由实验研究，当前实现没有将其解释为已知的 bisimulation 距离或概率校准结果。

## 3. 核心算法：同深度留一轨迹值共享

### 3.1 同伴集合与有限时域

对目标轨迹 $k$ 的第 $t$ 个动作前状态，同伴候选必须属于相同任务组、相同 benchmark、相同初始环境 seed，并在同一深度 $t$ 存在真实动作前状态。定义

$$
\mathcal P_{-k,t}=\{j:j\ne k,\ t<T_j\}.
$$

同组时域上限固定，因此相同深度对应相同剩余预算 $H-t$。整个目标轨迹从同伴数据中排除。已经结束的同伴在后续深度没有状态；实现不会将终态补成一个当前状态参与近邻比较。

这一选择保留了有限时域价值的时间索引。即使某个观察值在第 3 步和第 40 步完全相同，两者也不会直接混入同一状态估计。几何计算仅在每个深度的 $K\times K$ 矩阵内进行，无跨时间图拓扑和递归 Bellman 求解。

### 3.2 连续核与同伴 Monte Carlo 值

定义双视角距离与核权重

$$
d_{kj,t}=\max\{1-(z^o_{k,t})^\top z^o_{j,t},\;
                       1-(z^h_{k,t})^\top z^h_{j,t}\},
$$

$$
w_{kj,t}=\mathbf 1[j\in\mathcal P_{-k,t}]
\left[1-\frac{d_{kj,t}}{\epsilon}\right]_+,
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

### 3.3 动作前基线

先计算同任务留一轨迹先验，再构造动作前基线：

$$
\bar q_{-k}=\frac{1}{K-1}\sum_{j\ne k}q_j,\qquad
b_{k,t}=\frac{\sum_j w_{kj,t}q_j+\kappa\bar q_{-k}}
                   {S_{k,t}+\kappa},\qquad \kappa=1.
$$

无局部支持时，基线自然等于任务级留一均值。已终止同伴仍可贡献任务级结果先验，其作用不会变成某个后续深度的虚构状态。基线只读取目标动作前信息和独立同伴数据；目标当前动作、当前动作后的观测及自身最终成败都不参与基线计算。

### 3.4 下一状态共享与最终优势

非终末动作的共享权重取下一状态的支持质量：

$$
\lambda_{k,t}=\lambda_{\max}\frac{S_{k,t+1}}{S_{k,t+1}+\kappa},
\qquad \lambda_{\max}=1.
$$

终末动作规定 $\lambda_{k,t}=0$。下一状态表示来自当前已经发生的动作与公开反馈；$\lambda$ 不读取目标轨迹在这之后的动作或终局结果，也不使用预测误差调权。定义

$$
Z_{k,t}=(1-\lambda_{k,t})G^{\mathrm{MC}}_{k,t}
       +\lambda_{k,t}\bigl(r_{k,t}+\mu_{k,t+1}\bigr),\qquad
A_{k,t}=\frac{Z_{k,t}-b_{k,t}}{R}.
$$

默认公式在没有下一状态支持时恰好使用自身 Monte Carlo 回报；随着支持增加，使用更多同伴续值。末步完整保留官方终局回报。所有信用量在参数更新前计算并冻结。

实现按 `[depth, trajectory, feature]` 组织表示，用批量 Gram 矩阵一次形成所有同深度距离，再按 `[depth, target, peer]` 归约。时间复杂度为 $O(HK^2D)$，权重存储为 $O(HK^2)$。表示使用 CPU float32，信用加权与回报归约使用 CPU float64，避免大量 Python 子图循环与小规模 GPU 同步。

## 4. 数学含义、偏差与明确成立的性质

### 4.1 基线的 score 恒等式

在固定任务初态、冻结的行为策略 $\pi_{\mathrm{old}}$ 和参考参数下，假设同伴轨迹使用独立随机流生成。记整个目标轨迹之外的数据为 $D_{-k}$，动作响应的 score 为

$$
\psi_{k,t}=\left.\nabla_\theta\log\pi_\theta(y_{k,t}\mid p_{k,t})\right|_{\theta=\theta_{\mathrm{old}}}.
$$

由于 $b_{k,t}$ 在当前动作前已经确定，

$$
\mathbb E[b_{k,t}\psi_{k,t}\mid h_{k,t},D_{-k}]
=b_{k,t}\sum_y\pi_{\mathrm{old}}(y\mid p_{k,t})
\nabla\log\pi_{\mathrm{old}}(y\mid p_{k,t})=0.
$$

该结论针对行为策略采样点的基线 score 项；参数更新后的 PPO、多轮优化和近似续值目标需要分别分析。它也没有要求样本内平均优势为零。整个响应的 score 等于其实际采样词元 score 之和，因而与本文的轨迹损失求和口径一致。

### 4.2 共享目标的偏差恒等式

令 $V^{\pi_{\mathrm{old}}}(h_{t+1})$ 为相同有限时域、相同行为策略下的真实后续回报均值。分析条件包含：同伴与目标后续随机性在固定初态下独立；$\lambda_t$ 仅依赖当前已实现转移和同伴数据；环境转移遵循本次 rollout 的条件分布。把终末后继值定义为零，则

$$
\begin{aligned}
\Delta_t
&=\mathbb E[(Z_t-b_t)\psi_t]-\mathbb E[G^{\mathrm{MC}}_t\psi_t]\\
&=\mathbb E\!\left[\lambda_t
\bigl(\mu_{-k,t+1}-V^{\pi_{\mathrm{old}}}(h_{t+1})\bigr)\psi_t\right].
\end{aligned}
$$

推导从 $Z-G^{\mathrm{MC}}=\lambda(r+\mu-G^{\mathrm{MC}})$ 出发。对当前转移及同伴数据条件化后，$G^{\mathrm{MC}}-r$ 的条件均值为真实续值，基线项由上一节消去。由此可将偏差直接归于共享估计与真实后续值之差。

若在所讨论分布上有 $|\mu-V|\leq\delta$、$\lambda\leq\lambda_{\max}$，并且 $\mathbb E\|\psi_t\|\leq L_t$，则归一化后

$$
\|\Delta_t/R\|\leq \lambda_{\max}\delta L_t/R,\qquad
\left\|\sum_{t=0}^{H-1}\Delta_t/R\right\|
\leq\frac{\lambda_{\max}\delta}{R}\sum_{t=0}^{H-1}L_t.
$$

这是带显式条件的误差分析；支持量消融使用权重上界 1。余弦半径没有给出已知的 $\delta$；有限上下文、表示混叠和有限同伴采样均影响实际误差。支持收缩提供可控的共享幅度，正式实验进一步测量其预测误差与任务收益。r3 的论证以这些条件为边界，不使用“精确无偏续值”或“全局 DAG 正确性”作为前提。

### 4.3 值域、全零组及失败前缀

因为所有核权重非负、$\kappa>0$ 且 $0\leq\lambda\leq1$，有

$$
0\leq b_{k,t},\mu_{k,t},Z_{k,t}\leq R,\qquad -1\leq A_{k,t}\leq1.
$$

若某组所有 $Y_j=0$，该组的先验、局部值、共享目标和任务优势全部为零。熵正则提供有限的分布扩散，参考正则约束偏离；二者属于实际训练辅助，整体算法没有全失败组的成功发现保证。`all_zero_groups` 会直接记录这一状态。

一个失败轨迹中的动作可以因下一状态的成功同伴获得正优势，具体条件是 $\lambda_{k,t}\mu_{k,t+1}>b_{k,t}$。这种条件将“得到正信用”的含义绑定到可计算估计。它也说明了良好前缀仍可能有负优势：设 $K=8$，目标最终失败，另七条轨迹成功；当前及下一深度完全匹配且均非终态。此时 $S_t=S_{t+1}=7$、$b_t=R$、$\mu_{t+1}=R$、$\lambda_t=7/8$，于是 $A_t=-1/8$。即使这一步是共同的必要前缀，目标自身失败回报仍保留了 $1/8$ 的权重。

研究用官方任务表现和留出终局预测检验共享机制的作用。`positive_failed_steps` 直接描述算法对失败轨迹分配正优势的次数；关于有效动作信用的解释依照上述共享条件，并与任务结果一同呈现。

## 5. PPO 更新与轨迹权重

每个步骤的 $A_{k,t}$ 应用于该响应的全部实际采样词元。人工补入的 ChatML 闭合词元、环境观测、滚动重播的旧响应都没有新的损失所有权。令第 $i$ 个响应词元的概率比为 $\rho_i=\exp(\ell_i^\theta-\ell_i^{\mathrm{old}})$。

实现使用 clip $0.2$、dual clip $3$ 的 PPO token 损失，叠加系数 $0.001$ 的条件熵奖励和系数 $0.01$ 的截断参考 k3 正则。参考项为

$$
d_i=\ell_i^{\mathrm{ref}}-\ell_i^\theta,\qquad
k_i=\min\{10,\exp(d_i)-d_i-1\}.
$$

这些是保存的行为样本上的正则化 surrogate。裁剪、抽样参考项和近似续值共同决定实际更新；训练并不宣称每次参数变化都等于官方成功目标的精确梯度。默认全参数 bf16/SDPA、无 dropout，AdamW 学习率 $10^{-6}$、weight decay $0.01$，全局梯度范数裁剪 $1$。

F-GPO 及其自身消融采用 `loss_normalization="trajectory"`。一轮包含 $N_{\mathrm{traj}}$ 条真实轨迹，事先按响应行数分成 $M$ 个 minibatch，则每个 minibatch 使用相同分母

$$
Z_{\mathrm{mb}}=N_{\mathrm{traj}}/M,\qquad
\mathcal L_m=\frac{1}{Z_{\mathrm{mb}}}
\sum_{i\in\mathcal B_m}\bigl(\ell^{\mathrm{PPO}}_i-\alpha H_i+\beta k_i\bigr).
$$

在固定参数下对这些 minibatch 损失取平均，恰好等于所有响应词元损失之和除以轨迹数。每条轨迹内部使用词元求和；长轨迹具有更多决策项。末个不足额 minibatch 保持同一分母。参数在每次 optimizer step 后变化，后续相关段按当前参数重新计算。三个已发表 baseline 保留作者 virtual-rank / microbatch 内的 token mean 及其聚合权重、优势归一化和 minibatch 设置；本节的轨迹归一化是 F-GPO 的方法定义。

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

冻结参考权重在每轮参考阶段集中安装，完成响应概率与历史特征计算后恢复 actor。同段所有拥有损失的响应合并成有界 head chunks，按每条响应的 `span.start-1 : span.stop-1` 因果位置评分；目标 token 一次上传，每段的分数一次下载。每块词表 logits 在下一块计算前释放。历史 hook 只截取所需 prompt 向量，局部观测采用去重、长度排序和词元预算微批。此处的复用范围以相同参数版本和实际段为单位。

信用阶段形成观测和历史 Gram 矩阵后及时释放临时特征，最大距离写回已有小矩阵。完成优势及聚合诊断后调用 `ReferenceRound.release_features()`，去除后续更新和 checkpoint 中无用途的表示引用，保留冻结参考的响应条件概率。

### 6.2 持久环境进程与提前派发

每条轨迹通过独占 lease 使用一个官方环境实例。持久进程池复用 Python 进程和已加载依赖；新 lease 重新建立该任务的隔离状态，旧任务数据库与可变环境对象不会在轨迹间共享。池的容量按运行配置约束，进程启动、复用与环境调用记入成本日志。

GPU 仍以单主线程处理模型批次。每个槽位独立维护 `torch.Generator`；采样保持各槽位自己的随机流，将 token 与 log-prob 合并为一次 CPU 读取。某条响应遇到实际终止符或生成长度上限后，立即将解析、原生环境执行和公开观测投影派给 CPU 工作线程，此时其余槽位仍可继续 GPU 解码。

下一交互轮保留 turn barrier：本轮所有活跃槽位的反馈完成后，才构造下一轮模型输入。F-GPO 在轮内重叠环境调用与解码。已发表 baseline 同样采用组内批量模型采样与并行环境调用，完整异长提示通过 `prefill_batch` 进入同一批次；其提示、记忆和动作解析仍按作者配方构造。训练温度为 1，评估温度为 .4，`top_p=1`；保存的行为 log-prob 来自实际采样温度。独立 generator 保留各槽位随机流所有权，批形状与浮点舍入变化可能影响最终离散结果，实际执行配置随结果保存。

### 6.3 几何增长 KV 与真实内存

`cache.py` 的 arena 在容量充足时原位写入 KV，容量不足时按几何增长扩容。它维护每行有效长度、物理跨度、attention mask 与逻辑位置。变长批次仍然具有 padding；压紧会真实移动有效列，只有已经紧凑的布局才能跳过这项操作。`crop` 改变逻辑长度并保留底层容量，不能据此声称显存已经释放。

这项设计减少逐 token 全前缀拼接分配，同时保留 padding 和位置语义。根前缀可以共享一次 prefill 结果，分叉后各槽位持有可独立更新的缓存。窗口丢弃历史时重新 prefill 新窗口。行选择的 indices 按设备上传一次供各层复用；已结束响应的 padding 槽位跳过词表投影，原分布保留在独占缓存对象中。连续解码已处于 eval 状态时，跳过重复模块遍历。数学条件一致与浮点逐位一致分别陈述，实际吞吐由正式日志报告。

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

模型主干和词表头热路径移除显式逐调用同步，其日志耗时标为 `host_dispatch`；解码操作内部按 role 聚合模型调用、真实 token 和 padding 计算计数，减少逐 token 的 JSON 文件写入与事件常驻。环境事件即时保存。正式阶段边界进行 CUDA 同步，记录 `synchronized_wall` 耗时与每张卡的 allocated、reserved 和真实 peak bytes。嵌套阶段峰值标明所属外层阶段；重叠环境时间与 GPU 时间各自解释，端到端成本使用 ledger 的 wall/device-seconds。吞吐、实际 token、环境步数、GPU 小时和峰值显存均由已有正式运行账本产生。

用户性能提案中的异步派发、持久环境、减少同步、批量表征、几何缓存、有界传输与分块输出头思路已按上述条件纳入。所有模块均以 r3 数学对象为准；旧三遍估计器的删除属于算法变更。理论容量计算与预计收益可以指导配置，具体加速倍数和 benchmark 增幅由服务器结果填写。

## 7. 正式研究协议与实验解释

### 7.1 按 claim 保留最小实验

`experiment_suites` 把已发表方法比较与自身消融分开定义。主比较为四种方法 × 两个 benchmark × 三个训练种子，共 **24 个训练单元**；三个自身消融只在 ALFWorld 上分别运行三个种子，新增 **9 个单元**。消融中的完整 F-GPO 直接复用主比较的三个 ALFWorld 单元，合计 **33 个训练单元**。

| Suite 与方法 | Benchmark / 种子 | 新增单元 | 必须回答的 claim 及保留理由 |
|---|---|---:|---|
| `comparison / fgpo` | ALFWorld、AppWorld / 0、1、2 | 6 | 完整方法的官方任务表现及实际成本；提供所有主比较和自身消融的目标结果 |
| `comparison / gigpo` | ALFWorld、AppWorld / 0、1、2 | 6 | 对照出发点 GiGPO 的已发表相对状态信用方法，检验方案相对原研究路线的进展 |
| `comparison / hgpo` | ALFWorld、AppWorld / 0、1、2 | 6 | 对照直接使用历史层级信用的已发表方法，建立历史条件化方向的竞争关系 |
| `comparison / graphgpo` | ALFWorld、AppWorld / 0、1、2 | 6 | 对照已发表跨轨迹成功路径图信用方法，检验本方案续值共享的完整方法表现 |
| `ablation / fgpo_obs_only` | ALFWorld / 0、1、2 | 3 | 只去除历史距离项，检验本方法历史匹配对留出预测与任务表现的贡献 |
| `ablation / fgpo_no_transfer` | ALFWorld / 0、1、2 | 3 | 设全部 $\lambda=0$，保留原匹配与动作前基线，检验下一状态续值共享 |
| `ablation / fgpo_no_support_weight` | ALFWorld / 0、1、2 | 3 | 非终末下一状态有支持时令 $\lambda=1$，其余核与基线保持一致，检验连续共享收缩 |
| `ablation / fgpo` 参考行 | 复用主比较的 ALFWorld 三种子 | 0 | 为三个机制消融提供同一完整方法结果，不再训练 |

ALFWorld 承担直接来源 benchmark 与机制消融，AppWorld 承担完整方法向原生代码 agent 的迁移证据；机制消融的直接结论限定在 ALFWorld。三个训练种子用于公开训练变动，完整官方划分保证终点指标有一致口径。短历史实验、超参数扫描、模型规模扫描、系统模块消融和独立诊断 rollout 均不在计划中。已有代码兼容路径与相关文献不会自动增加实验单元。

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

ALFWorld 接入作者发布的公开观测、原生动作与提示配方。HGPO/GraphGPO 的来源就是 7B，GiGPO 公开全参脚本使用 1.5B，本研究明确标注其 **7B backbone 迁移**并保留上述来源 batch 配方。三个已发表工作均未报告 AppWorld 实验，因此 AppWorld 的结果另记为**已发表算法的 benchmark 迁移**：用原生 Python action 与公开 stdout，保留初始任务/API bootstrap 及最近记忆，接入各自信用规则。AppWorld 使用 1024-token 生成上限和完整消息窗口；ALFWorld 仍保留各作者的提示容量与截断规则。所有参考和 PPO 概率都在最终实际采样前缀上重算。

执行后端统一为本仓库的 PyTorch/SDPA、组内批量采样、持久官方环境与 CPU-master AdamW。复现范围是算法及发布配方在该后端的实现，计算系统与原作者分布式系统分别注明。主比较因此支持完整方法级的质量—成本判断，机制归因由第 7.1 节的三个自身消融承担。

### 7.3 实际预算与完整官方评测

| 项目 | F-GPO 及自身消融 | GiGPO | HGPO | GraphGPO |
|---|---|---|---|---|
| 训练 rounds | 60 | 150 | 160 | 150 |
| 每轮任务组 × 每组轨迹 | 16 × 8 | 16 × 8 | 16 × 8 | 16 × 8 |
| 每种子完整轨迹量 | 7680 | 19200 | 20480 | 19200 |
| 信用折扣 γ | 1 | .95 | .95 | .10 |
| ALFWorld 每步生成上限 | 256 | 512 | 512 | 512 |
| AppWorld 每步生成上限 | 1024 | 1024 | 1024 | 1024 |
| ALFWorld 提示 / 历史规则 | 32768 总上下文内的初始锚定与完整轮次滚动 | 2048 prompt，最近两轮记忆，超限 error | 4096 prompt，最近两轮记忆，作者 left truncation | 2048 prompt，最近两轮记忆及公开状态摘要，超限 error |
| PPO row minibatch | 256 | 256 | 256 | 64（7B 配方） |
| Loss reduction | 固定轨迹分母 | 作者 rank / microbatch token mean | 作者 rank / microbatch token mean | 作者 rank / microbatch token mean |

共同基座为 Qwen2.5-7B-Instruct，全参数 bf16/SDPA；种子为 0、1、2；AdamW lr=`1e-6`、weight decay=`.01`、betas=`(.9,.999)`、epsilon=`1e-8`、全局梯度范数上限 1。每轮一次 PPO epoch，clip=.2、dual clip=3、entropy=.001、参考正则=.01。训练采样温度为 1，评估为 .4，`top_p=1`。F-GPO 的信用参数为 $R=10,\epsilon=.10,\kappa=1,\lambda_{\max}=1$；baseline 的奖励尺度和语法处理以来源配方为准。

时域上限为 ALFWorld 50 步、AppWorld 100 步。训练任务取官方训练划分，每个最终 checkpoint 完整评测 ALFWorld `valid_unseen` 的 134 个任务，或 AppWorld `test_normal` 的 168 个任务与 `test_challenge` 的 417 个任务。ALFWorld 报告 SR，AppWorld 报告官方 TGC 和基于完整 scenario 的 SGC，并附完成任务数。三个种子的逐项结果、均值与标准差均公开；中断产生的部分评测保持部分状态。

主表同时列最终质量、实际轨迹量、环境步数、响应词元、端到端时间与设备小时。各方法按已声明的配方终点报告，训练窗口差异随结果呈现；训练组成功率曲线解释学习进展，最终测试分数单独列出。当前计划的效率结论以质量与实际成本的联合结果为依据，不增加中途测试集挑选、额外预算扫描或独立系统基准实验。

### 7.4 只复用必要的 claim 诊断

| Claim | 直接实验 | 复用的诊断及用途 |
|---|---|---|
| 完整方法的任务表现和实际成本 | `comparison` 的四种完整方法 | 最终官方 SR/TGC/SGC、真实样本与 token 数、wall/device-seconds、阶段峰值，形成质量—成本报告 |
| 历史条件化改善本方法的匹配 | 完整 F-GPO 与 `fgpo_obs_only` | 匹配覆盖、支持质量和留一 outcome Brier，解释距离改变后的预测表现 |
| 下一状态共享提供任务收益 | 完整 F-GPO 与 `fgpo_no_transfer` | 共享比例、回报移动量及训练进展，确认实际使用的共享量与任务变化 |
| 连续支持收缩有作用 | 完整 F-GPO 与 `fgpo_no_support_weight` | $S$、有效同伴数与 $\lambda$，解释共享强度改变后的任务及预测结果 |

这些量全部从正式训练或官方评测的既有记录提取，不触发新 rollout。全零组和失败轨迹正优势计数作为已有信用输出的解释字段保留，正文图表按上述 claim 选取。主留一诊断定义为

$$
\operatorname{Brier}_{\mathrm{LOO}}
=\frac{1}{\sum_k T_k}\sum_{k,t<T_k}
\left(b_{k,t}/R-Y_k\right)^2.
$$

另记录任务先验 Brier 和有匹配状态上的未收缩 $\mu/R$ Brier。目标轨迹结果在预测时被留出，随后用于计算预测损失。若真实条件成功概率为 $p$，条件平方误差可分为 $(\widehat p-p)^2+p(1-p)$，因此该指标包含 Bernoulli outcome 噪声，其用途是解释留出结果预测；真实 Q 误差和理论中的 $\delta$ 是不同对象。Brier 不回流为训练权重。

诊断按真实步骤统计，长轨迹贡献更多预测样本；跨组聚合按相应样本数加权，并保留每组原始统计。每个 claim 由对应的正式结果和这些既有统计支撑，诊断自身不扩展训练矩阵。

## 8. 运行身份与交付状态

`fgpo-r3-published-v3` 使用新的 study 根目录，内部按 suite 分成 `comparison/` 与 `ablation/`，各自保存单元和 `study_summary.json`。先完成主比较，再以同一 `--output-dir` 执行消融；消融 summary 引用 `../comparison/alfworld/fgpo/seed-X/cell_summary.json`，复用完整方法结果。

在指定服务器的独立 checkout 中，将 `PYTHONPATH` 指向 `src`，使用配置指定的既有 Python 与真实分配卡号：

```bash
python -m fgpo.runner --config configs/fgpo-r3-host-a.json --output-dir outputs/fgpo-r3-published-study --suite comparison --gpu <physical_gpu_ids>
python -m fgpo.runner --config configs/fgpo-r3-host-a.json --output-dir outputs/fgpo-r3-published-study --suite ablation --gpu <physical_gpu_ids>
```

Host B 使用 `configs/fgpo-r3-host-b.json`。`comparison` 是默认 suite；`--methods/--benchmarks/--seeds` 只选择该 suite 的计划子集，用于按完整 cell 分配服务器。新 suite 可以在已有 study 根目录中创建；`--resume` 只恢复当前所选 suite 的目录。旧协议及其 pending update 留在原 study，不迁入新的训练状态。

恢复由 `effective_training_identity/require_training_identity` 约束：方法及发布配方来源、模型、任务、种子、历史、采样、信用、PPO 和 Adam 的有效语义保持一致。GPU 放置、KV 容量、offload、head/表示微批和 staging 等资源项可以调整；保存的 CUDA RNG 流数量保持一致。完成采集组、参考概率、优势与更新 cursor 在明确操作边界恢复。

当前交付是完整方案和对应源代码的静态修订与交叉审阅，尚未执行服务器训练、显存测量或正式 benchmark。仓库 [AGENTS.md](../AGENTS.md) 要求所有项目运行在指定服务器完成；当前会话没有 `SSH_Document.md`，已知 `mgt01` 入口在可用连接环境中无法解析。文中没有已执行的成功率、显存节省比例或加速倍数。正式连接恢复后直接执行上述 33 个必要训练单元及其官方评测，从相同日志形成任务表现、共享机制和实际成本的研究结论。

指定服务器上的 `.research/research.sqlite` 继续作为实际研究记录来源；当前会话没有访问该数据库，未生成或改写运行记录。
