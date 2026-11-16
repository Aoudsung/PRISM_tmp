# F-GPO r3 数学与实现契约

本文件与 [研究方案及推导](fgpo_design_and_derivation.md) 定义 `fgpo-r3-success-v2` 的实际实现，版本日期为 2026-11-16。核心实现是 `src/fgpo/credit.py` 的 `CrossTrajectoryCredit`，对外算法名为 `r3_history_conditioned_peer_value_sharing`。旧方案通过基准提交 `977bb86ac31a4b1f95ea26b7b53334be5cadf57e` 追溯；r3 使用新 study，不接续旧奖励和旧优势的 checkpoint。

本文规定数据、公式、默认值、更新和测量口径。当前仅完成静态编写与源代码审阅，所有项目执行及正式评测均在指定服务器进行，尚无 r3 的实测成功率或加速倍数。

## 1. 目标与输入对象

对同任务组的 $K\geq2$ 条独立轨迹，记轨迹长度为 $T_k\leq H$，官方终局成功为 $Y_k\in\{0,1\}$。定义 $R=10$、$q_k=RY_k$、$\gamma=1$：

$$
r_{k,t}=q_k\mathbf1[t=T_k-1],\qquad
G^{\mathrm{MC}}_{k,t}=\sum_{u=t}^{T_k-1}r_{k,u}=q_k.
$$

`credit.py` 从 `official_success` 重建回报，不使用记录中的任意环境中间 reward、参考 reward 或语法标签替代官方终局目标。`ReferenceStep.reward` 同样只保存该终局奖励，便于记录。主方法没有 TF-BEFP、目标似然奖励、手工语法惩罚及随机 RMS。

每组轨迹的 `group_id/task_id/benchmark/initial_seed/goal` 相同，`trajectory_id` 唯一。步骤 `row.index` 连续为 $0,\ldots,T_k-1$，`row.trajectory_id` 与所属轨迹一致，`step_id` 在整个收集轮中唯一。合法终末标志由

$$
\operatorname{terminal}_{k,t}
=\mathbf1[t=T_k-1]
=\mathbf1[\texttt{row.done}\ \lor\ t+1=H]
$$

确定。`row.done` 保存环境原始结束标志，因时域截断而结束的末步可以为 false。中途停止的残缺轨迹不能作为完整训练轨迹进入该估计器。上述条件属于普通输入契约。

默认 $H_{\mathrm{ALF}}=50$、$H_{\mathrm{App}}=100$。只存在真实动作前状态 $t<T_k$；终态没有可供后续深度匹配的状态。空轨迹没有步骤优势；若整个合法组没有步骤，返回空字典和有限的组级元信息。PPO 更新要求存在真实采样响应。

## 2. 实际行为条件与词元所有权

数学公开历史记为 $h_t$，actor 真实输入是 $p_t=C(h_t)$。$C$ 固定保留初始任务与公开 bootstrap，随后保留有限容量内的最近完整动作—反馈轮次。每步生成上限默认为 ALFWorld 256、AppWorld 1024，模型上下文默认 32768。提示容量为 $C_{\mathrm{prompt}}=C_{\max}-M_{\mathrm{gen}}$。

通常滚动水位为 $0.75C_{\mathrm{prompt}}$；实际水位取该值与“初始锚定块长度 + 最新完整轮次长度”的较大值，并满足提示上限。历史窗口由完整块构成，初始块与最新完整轮次无法容纳时显式报错。`fgpo_no_full_filtration` 的候选历史仅含最近两轮，仍遵循相同容量规则。

环境文本采用 `split_special_tokens=True` 编码，帧结构由静态 ChatML 整数 token 构建。实际采样的 `im_end/endoftext` 属于响应；人工补入的闭合 token 不属于响应损失。一个实际响应只有一个 `response_span`，滚动复制的历史响应不重新拥有损失。

若响应起点为 $a$、长度为 $m$，在输入 `S[:-1]` 的因果段前向中使用隐藏态 `[a-1:a-1+m]` 预测其实际目标 token。历史特征读取 `prompt_length-1`。参考与 PPO 都读取该响应的实际段和实际前缀，短历史消融也完全遵循这一条件。

只有 actor 下一步实际消费的反馈进入段。终末反馈仍保存于步骤及 `TurnBlock`，其 `observation_span=None`，没有独立的 feedback-only 评分段。参考阶段读取至 `behavior_closed_length`；段中未来位置由因果 attention mask 屏蔽。

## 3. 表示、同伴权重与支持质量

冻结参考模型在实际动作前 prompt 末位的倒数第 8 层表示给出 $z^h_{k,t}$；当前局部公开观测的独立编码末位表示给出 $z^o_{k,t}$。二者 L2 归一化并停止梯度。第零步使用初始观测，后续步骤使用上一动作返回的公开观测；空文本用 `[Empty stdout]` 的表示 token。

定义有效位置 $v_{k,t}=\mathbf1[t<T_k]$。同伴必须来自目标轨迹之外，并位于相同深度；因此拥有同样的剩余时域 $H-t$。默认距离与权重为

$$
d_{kj,t}=\max\left\{
\operatorname{clip}(1-(z^o_{k,t})^\top z^o_{j,t},0,2),\;
\operatorname{clip}(1-(z^h_{k,t})^\top z^h_{j,t},0,2)
\right\},
$$

$$
w_{kj,t}=v_{k,t}v_{j,t}\mathbf1[j\ne k]
\left[1-\frac{d_{kj,t}}{\epsilon}\right]_+,\qquad
S_{k,t}=\sum_jw_{kj,t}.
$$

恰好在半径边界的权重为零。当前目标轨迹的其他步骤也不会成为同伴；距离矩阵只在各深度内部构建。已结束同伴在后续深度的 $v_{j,t}$ 为零。

令

$$
U_{k,t}=\sum_jw_{kj,t}q_j,\qquad
\mu_{k,t}=\begin{cases}U_{k,t}/S_{k,t},&S_{k,t}>0,\\0,&S_{k,t}=0.\end{cases}
$$

每个被共享的值来源都是真实 Monte Carlo 结果 $q_j$；算法没有递归值传播。$S$ 是有限同伴的核质量，不是校准置信概率。实现以 CPU float32 批量矩阵乘法计算距离，以 CPU float64 计算权重归约、回报、基线和优势。

## 4. 动作前基线、共享目标与优势

先验与动作前基线为

$$
\bar q_{-k}=\frac{\sum_{j\ne k}q_j}{K-1},\qquad
b_{k,t}=\frac{U_{k,t}+\kappa\bar q_{-k}}{S_{k,t}+\kappa}.
$$

基线使用目标动作前特征及同伴数据。没有当前动作、动作后的观测或目标自身终局结果进入 $b$。所有同伴的完整终局结果都可用于任务级先验；已经结束的同伴仍然不会获得当前深度的状态权重。$S=0$ 时 $b=\bar q_{-k}$。

默认共享强度和目标为

$$
\lambda_{k,t}=
\begin{cases}
\lambda_{\max}S_{k,t+1}/(S_{k,t+1}+\kappa),&t<T_k-1,\\
0,&t=T_k-1,
\end{cases}
$$

$$
Z_{k,t}=(1-\lambda_{k,t})q_k+
\lambda_{k,t}(r_{k,t}+\mu_{k,t+1}),\qquad
A_{k,t}=(Z_{k,t}-b_{k,t})/R.
$$

下一状态的特征与支持在当前真实转移后即可确定，$\lambda$ 不读取目标在这之后的成败。终末位置强制 $\lambda=0$。没有支持时 $\mu=0$ 是上述定义的一部分，目标仍因 $\lambda=0$ 使用自身真实回报。

在合法输入上，$0\leq b,\mu,Z\leq R$ 且 $A\in[-1,1]$。全失败组满足 $b=\mu=Z=A=0$。默认公式下，失败非终末动作获得正优势的充要条件是 $\lambda\mu_{t+1}>b_t$；“有成功同伴”本身只决定其中一部分量。

### 配置及方法 ID

| 配置项 | 默认值 | 实现约束及含义 |
|---|---|---|
| `reward_scale` | `10.0` | 有限正数；正式协议为 $R=10$，所有回报与优势共用该固定尺度 |
| `credit_radius` | `0.10` | 有限正数，核半径 $\epsilon$ |
| `credit_prior_weight` | `1.0` | 有限正数，$\kappa$ 同时控制动作前先验和默认共享收缩 |
| `credit_transfer_max` | `1.0` | $[0,1]$ 内，默认公式的 $\lambda_{\max}$ |
| `gamma` | `1.0` | r3 信用公式固定无折扣；不得通过旧折扣回报替换 |

| 方法 | 与 `fgpo` 的唯一区别 |
|---|---|
| `fgpo_no_full_filtration` | actor 实际历史保留初始锚定块与最近两轮；参考、PPO 和历史特征采用同一真实输入 |
| `fgpo_no_transfer` | 所有共享权重 $\lambda=0$，原核与基线保留 |
| `fgpo_obs_only` | $d$ 只保留观测余弦距离，actor 的历史窗口不变 |
| `fgpo_no_support_weight` | 非终末且下一状态 $S>0$ 时 $\lambda=1$，否则为零；该消融不使用 `credit_transfer_max` |

`fgpo_no_befp` 和 `fgpo_no_dag_loo` 显式拒绝，不映射到任意新消融。外部共享比较器为 `gigpo_shared`、`bipace_q_shared`、`native_qwen_shared`，从各自估计器入口调用。原作者配置 ID 单列为迁移配方，不能静默变成共享配方。

## 5. 理论声明的适用范围

设行为策略固定为 $\pi_{\mathrm{old}}$，同任务初态条件下同伴采样与目标轨迹独立，$\psi_t=\nabla_\theta\log\pi_\theta(y_t\mid p_t)|_{\theta_{\mathrm{old}}}$。动作前基线满足

$$
\mathbb E[b_t\psi_t\mid h_t,D_{-k}]=0.
$$

该恒等式只处理基线项，既不要求样本平均优势为零，也不把共享目标变为精确无偏回报。若 $\lambda_t$ 仅依赖当前转移及同伴数据，且目标后续的条件回报均值为 $V^{\pi_{\mathrm{old}}}(h_{t+1})$，则

$$
\mathbb E[(Z_t-b_t)\psi_t]-\mathbb E[G_t^{\mathrm{MC}}\psi_t]
=\mathbb E\!\left[\lambda_t
  \bigl(\mu_{t+1}-V^{\pi_{\mathrm{old}}}(h_{t+1})\bigr)\psi_t\right].
$$

在 $|\mu-V|\leq\delta$、$\lambda\leq\lambda_{\max}$、$\mathbb E\|\psi_t\|\leq L_t$ 条件下，归一化偏差上界为 $\lambda_{\max}\delta L_t/R$，全时域上界为 $\lambda_{\max}\delta\sum_tL_t/R$。这个 $\delta$ 是分析条件，源码中的余弦半径和 Brier 日志均不提供其认证值。`no_support_weight` 的对应权重上界为 1。

良好前缀可以获得负优势。八轨迹组中目标失败、七个同伴成功，且当前和下一状态都完全匹配时，默认参数给出 $b=R,\mu_{t+1}=R,\lambda=7/8,A=-1/8$。全失败组的任务优势为零。熵正则仅提供有限探索辅助，r3 没有冷启动成功保证。

## 6. PPO 及轨迹归一化

固定行为 log-prob、参考 log-prob 和步骤优势；每个实际响应 token $i$ 的优势为所属步骤 $A_i$。设 $\rho_i=\exp(\ell_i^\theta-\ell_i^{\mathrm{old}})$，clip $\eta=.2$，dual clip $c=3$：

$$
u_i=\max\{-A_i\rho_i,-A_i\operatorname{clip}(\rho_i,1-\eta,1+\eta)\},
\quad
\ell_i^{\mathrm{PPO}}=
\begin{cases}\min(u_i,-cA_i),&A_i<0,\\u_i,&A_i\geq0.\end{cases}
$$

对当前完整词表 logits $z_i$，实现计算

$$
\ell_i^\theta=z_{i,y_i}-\operatorname{LSE}(z_i),\qquad
H_i=\operatorname{LSE}(z_i)-\sum_v\operatorname{softmax}(z_i)_v z_{i,v},
$$

$$
d_i=\ell_i^{\mathrm{ref}}-\ell_i^\theta,\qquad
k_i=\operatorname{clip}(\exp(d_i)-d_i-1,0,10).
$$

为防止无意义的指数溢出，概率比的指数输入上限为 $\log(\max(c,1+\eta))+1$，该范围外双裁剪损失已是常数；$d_i$ 在计算指数前截到 $[-11,\log22]$，原 $k_i$ 在这个范围外已饱和到 10。这些数值处理保持所定义的截断目标。

每轮先固定响应行 minibatch 数 $M$。共享协议定义 $Z_{\mathrm{mb}}=N_{\mathrm{traj}}/M$，并使用

$$
\mathcal L_m=\frac{1}{Z_{\mathrm{mb}}}\sum_{i\in\mathcal B_m}
(\ell_i^{\mathrm{PPO}}-.001H_i+.01k_i).
$$

这是所有响应 token 损失的求和，轨迹内部不按长度平均。末批维持同一分母；完整轮的轨迹数和 minibatch 数在恢复后也保持相同。固定参数下 $\frac1M\sum_m\mathcal L_m$ 等于轨迹归一化的整体 surrogate。`metrics.loss` 累加实际 minibatch 损失，比较时要保留该口径与 optimizer step 数。

每个 minibatch 完成后执行一次 AdamW，默认学习率 `1e-6`、weight decay `.01`、全局梯度裁剪 `1`。参考权重固定；行为权重随训练轮更新。PPO 裁剪、k3 采样正则、条件熵及有偏续值估计构成实际正则化 surrogate，其性质不能直接等同于每次更新都计算精确成功率梯度。

## 7. 系统执行契约

因果段只在相同参数版本下共享主干前向。某段在不同 minibatch 的参数更新之后再次被使用时，必须重算其主干激活。参考阶段按轮集中安装冻结权重，完成响应评分后恢复 actor；相同公开观察值按词元序列去重、按长度和 token 预算微批提取表示。

输出头每次只保留默认 `head_chunk_tokens=64` 的图。隐藏态块 detach 为叶子后执行该块 backward，将其梯度用位置索引加回完整隐藏态梯度缓冲；全部块完成后对主干输出调用一次 backward。输出头参数梯度与主干参数梯度按一阶链式法则累积。bf16 的实际数值由实现精度决定，批处理变更不声明逐位一致。

采样槽位独立拥有 generator。每个 token 步仍使用独立随机流，结果合并为一次 CPU 读取；达到响应结束条件立即派发 CPU 环境调用。下一交互轮保留 turn barrier。持久环境池复用进程、独占 lease 隔离轨迹状态，新的任务重新初始化原生环境对象。已结束轨迹不继续采样。

arena cache 按容量几何增长，保留 padding mask、实际有效跨度与逻辑 position。`compact` 对需要压紧的布局移动数据；`crop` 只缩短逻辑视图并保留容量。丢弃有效历史后重新 prefill 新窗口。增量缓存、整段评分和参数更新分别遵守对应输入条件。

CPU AdamW 保存 fp32 master/m/v。传输桶默认 64 MiB，每个设备与 dtype 使用两个有限槽；D2H 完成后才能读取 CPU 梯度，H2D 完成后才能复用其 host weight 缓冲。模型更新前同步所有未完成参数写回。主干梯度检查点继续使用；`activation_offload=false` 为默认值，显式开启时采用非锁页 CPU 激活保存。

热路径的模型调用时间记录为 `host_dispatch`。阶段边界同步 CUDA，记录 `synchronized_wall` 与实际 allocated/reserved/peak bytes；嵌套峰值归属外层阶段。环境与 GPU 重叠时间不重复相加为端到端成本。

## 8. 正式日志诊断

对非空组，`advantages(..., diagnostics=output)` 将诊断写入 `output[group_id]`，同时返回 `{step_id: float_advantage}`。`engine.py` 将这些值写入正式 `training.jsonl` 的 `diagnostics.credit_groups`，并保存原始 reference/advantages 供解释正式结果。

| 日志字段 | 精确定义 |
|---|---|
| `trajectory_count / step_count / successes` | 组内轨迹数、真实动作数、官方成功轨迹数 |
| `all_zero_group` | 所有轨迹官方成功均为零 |
| `matched_steps / matched_step_fraction` | 当前状态 $S_t>0$ 的真实步骤数及其占比 |
| `transferred_steps / transfer_step_fraction` | 当前转移 $\lambda_t>0$ 的真实步骤数及其占比 |
| `mean_support_mass` | 所有真实状态的 $S_t$ 均值 |
| `mean_effective_peers` | $(\sum_jw_j)^2/\sum_jw_j^2$ 的真实状态均值；无支持时为零 |
| `mean_transfer_weight` | 所有真实转移的 $\lambda_t$ 均值，包括末步的零 |
| `loo_outcome_brier` | 所有真实步骤上的 $(b_t/R-Y_k)^2$ 均值 |
| `loo_task_prior_brier` | 所有真实步骤上的 $(\bar q_{-k}/R-Y_k)^2$ 均值 |
| `matched_loo_outcome_brier` | 仅 $S_t>0$ 位置上的 $(\mu_t/R-Y_k)^2$ 均值；没有匹配时为 null |
| `mean_abs_return_shift` | 所有真实步骤上的 $\lvert Z_t/R-Y_k\rvert$ 均值 |
| `positive_failed_steps` | $Y_k=0$ 且 $A_{k,t}>0$ 的真实步骤数 |
| `mean_advantage / advantage_min / advantage_max` | 所有真实步骤优势的均值、最小值和最大值 |

组诊断还记录算法、方法、时域及实际信用参数。Brier 使用整条目标轨迹之外的数据进行预测，再用目标终局结果计算损失；它包含 Bernoulli outcome 噪声，衡量留出结果预测，不能作为真实 Q 误差、置信界或校准分数。这些诊断不参与共享权重或奖励。跨组聚合采用相应真实样本数加权。

## 9. 研究身份、基线与源码映射

主配置为 `configs/fgpo-r3-host-a.json` 和 `configs/fgpo-r3-host-b.json`。默认主矩阵包含本方法、四个自身消融、`gigpo_shared`、`bipace_q_shared` 和 `native_qwen_shared`；两种可训练共享比较器与 r3 遵循同一 collector、context、输出预算、60 轮、16 组、8 轨迹、优化器与轨迹损失分母。GiGPO/BiPACE 的相对优势定义由各自估计器保留；BiPACE 指纹来自当前 actor 的实际历史 prompt。作者配方作为可选单列运行，保留其已注明迁移超参数。

种子默认为 0、1、2；完整评测为 ALFWorld `valid_unseen` 的 134 任务，以及 AppWorld `test_normal` 的 168 任务与 `test_challenge` 的 417 任务。官方 success/TGC/SGC 与任务完成数一并记录。性能结果报告真实阶段耗时、输出 token、环境步数、GPU 小时和实际显存峰值。

`effective_training_identity/require_training_identity` 约束恢复时相同的协议、方法、种子、模型、任务、上下文、生成、信用、PPO 和 Adam 有效参数。GPU 放置、缓存容量、offload、传输桶及 head/representation 微批属于可调整的资源项，row minibatch 大小仍属于训练身份。CUDA RNG 流数量保持一致；SELF 的 dropout 还要求相同的逻辑层放置与随机流分配。旧协议待执行优势和优化状态禁止复用。中断恢复以已保存完整组和更新 cursor 为单位，不假设恢复任意时刻的环境进程内部状态。

| 文件 | 负责的契约 |
|---|---|
| `public.py / filtration.py / records.py` | 公开输入、实际有界历史、词元所有权和步骤记录 |
| `rollout.py / envs.py` | 独立采样、提前环境派发、轮间同步和持久隔离 lease |
| `policy.py / cache.py` | 因果前向、表征批处理、采样、缓存和阶段实测 |
| `reference.py` | 固定参考响应 log-prob、动作前表示、真实终局奖励 |
| `credit.py` | 同深度排除、核支持、基线、MC 续值共享和诊断 |
| `optimization.py` | 轨迹归一化 surrogate、分块输出头反向、有界 AdamW |
| `protocol.py / engine.py` | 方法身份、共享比较、生命周期、记录和恢复 |

正式研究只围绕方案中的成功率、匹配与共享估计、系统成本 claim 开展，诊断直接复用正式日志。没有新增校准、认证、分阶段放行或与研究主线无关的运行机制。
