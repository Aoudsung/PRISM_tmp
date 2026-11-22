# F-GPO r3 数学与实现契约

本文件与 [研究方案及推导](fgpo_design_and_derivation.md) 定义 `fgpo-r3-published-v4` 的实际实现，版本日期为 2026-11-17。第 1—6 节规定 F-GPO 及其自身消融的数学对象；第 9 节单列已发表基线的来源、配方和比较范围。核心实现是 `src/fgpo/credit.py` 的 `CrossTrajectoryCredit`，对外算法名为 `r3_history_conditioned_peer_value_sharing`。旧方案通过基准提交 `977bb86ac31a4b1f95ea26b7b53334be5cadf57e` 追溯；本协议使用新 study，不接续旧协议（含 `fgpo-r3-published-v3`）的奖励、优势或待完成更新。

本文规定数据、公式、默认值、更新和测量口径。当前仅完成静态编写与源代码审阅，所有项目执行及正式评测均在指定服务器进行，尚无 v4 的实测成功率或加速倍数。

## 1. 目标与输入对象

对同任务组的 $K=8$ 条独立轨迹，记轨迹长度为 $T_k\leq H$，官方终局成功为 $Y_k\in\{0,1\}$。定义 $R=10$、$q_k=RY_k$、$\gamma=1$：

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

默认 $H_{\mathrm{ALF}}=50$、$H_{\mathrm{App}}=100$。只存在真实动作前状态 $t<T_k$；终态没有可供后续深度匹配的状态。空轨迹没有步骤优势；若整个合法组没有步骤，或全组结果同质（$\sum_k Y_k\in\{0,K\}$），返回零优势与有限的组级诊断信息，跳过参考前向与特征计算。PPO 更新要求存在真实采样响应。

## 2. 实际行为条件与词元所有权

数学公开历史记为 $h_t$，actor 真实输入是 $p_t=C(h_t)$。$C$ 固定保留初始任务与公开 bootstrap，随后保留有限容量内的最近完整动作—反馈轮次。每步生成上限默认为 ALFWorld 512、AppWorld 1024，模型上下文默认 32768。提示容量为 $C_{\mathrm{prompt}}=C_{\max}-M_{\mathrm{gen}}$。

ALFWorld 的用户消息由静态任务/步数头部 `user_prefix_ids`、纯净环境观测 `pure_ids` 与无引号可行动作提示尾部 `user_suffix_ids`（滤除 `"help"`）三块整数切片拼接，`observation_body_offset = len(user_prefix_ids)` 使所有分块与窗口偏移继续精确指向 `pure_ids`。响应解码先移除完整闭合 `<think>...</think>`，再检验并提取唯一闭合 `<action>...</action>` 中的小写单行指令，未闭合标签、多重动作标签或空动作直接标记为 `invalid`。

通常滚动水位为 $0.75C_{\mathrm{prompt}}$；实际水位取该值与“初始锚定块长度 + 最新完整轮次长度”的较大值，并满足提示上限。历史窗口由完整块构成，初始块与最新完整轮次无法容纳时显式报错。三个正式自身消融均保留这一 actor 历史条件。

环境文本采用 `split_special_tokens=True` 编码，帧结构由静态 ChatML 整数 token 构建。实际采样的 `im_end/endoftext` 属于响应；人工补入的闭合 token 不属于响应损失。一个实际响应只有一个 `response_span`，滚动复制的历史响应不重新拥有损失。

若响应起点为 $a$、长度为 $m$，在输入 `S[:-1]` 的因果段前向中使用隐藏态 `[a-1:a-1+m]` 预测其实际目标 token。历史特征读取 `prompt_length-1`。参考与 PPO 都读取该响应的实际段和实际前缀；PPO 可将段的末端截到当前 minibatch 最后一个响应目标之后，完整前缀和位置编号保持相同。

只有 actor 下一步实际消费的反馈进入段。终末反馈仍保存于步骤及 `TurnBlock`，其 `observation_span=None`，没有独立的 feedback-only 评分段。参考阶段读取至 `behavior_closed_length`；段中未来位置由因果 attention mask 屏蔽。

## 3. 表示、同伴权重与支持质量

冻结参考模型在实际动作前 prompt 末位的倒数第 8 层表示给出 $z^h_{k,t}$。在 AppWorld 上，当前局部公开观测的独立编码末位表示给出 $z^o_{k,t}$，二者 L2 归一化并停止梯度；在 ALFWorld 离散观测空间，直接比较动作前纯观测整数词元元组是否完全相等 $m^{\mathrm{obs}}_{kj,t}=\mathbf 1[o^{\mathrm{id}}_{k,t}=o^{\mathrm{id}}_{j,t}]$，不执行额外的 `z_obs` 模型前向。第零步使用初始观测，后续步骤使用上一动作返回的公开观测；空文本用 `[Empty stdout]` 的表示 token。

定义有效位置 $v_{k,t}=\mathbf1[t<T_k]$。同伴必须来自目标轨迹之外，并位于相同深度；因此拥有同样的剩余时域 $H-t$。在 AppWorld 上，双视角距离与三角核权重为

$$
d_{kj,t}=\max\left\{
\operatorname{clip}(1-(z^o_{k,t})^\top z^o_{j,t},0,2),\;
\operatorname{clip}(1-(z^h_{k,t})^\top z^h_{j,t},0,2)
\right\},\qquad
w_{kj,t}=v_{k,t}v_{j,t}\mathbf1[j\ne k]
\left[1-\frac{d_{kj,t}}{\epsilon}\right]_+.
$$

在 ALFWorld 上，令历史余弦距离 $d^h_{kj,t}=\operatorname{clip}(1-(z^h_{k,t})^\top z^h_{j,t},0,2)$，权重定义为

$$
w_{kj,t}=v_{k,t}v_{j,t}\mathbf1[j\ne k]\,m^{\mathrm{obs}}_{kj,t}
\left[1-\frac{d^h_{kj,t}}{\epsilon}\right]_+,\qquad \epsilon=0.10,
$$

并记支持质量 $S_{k,t}=\sum_j w_{kj,t}$。恰好在半径边界的权重为零。当前目标轨迹的其他步骤也不会成为同伴；距离矩阵只在各深度内部构建。已结束同伴在后续深度的 $v_{j,t}$ 为零。

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

记下一深度存活同伴中的成功轨迹计数为 $N^+_{-k,t+1}=\sum_{j\in\mathcal P_{-k,t+1}}Y_j$。默认 $\lambda_{\max}=0.5$，共享强度和目标为

$$
\lambda_{k,t}=
\begin{cases}
\mathbf 1[N^+_{-k,t+1}\ge 1]\,\lambda_{\max}S_{k,t+1}/(S_{k,t+1}+\kappa),&t<T_k-1,\\
0,&t=T_k-1,
\end{cases}
$$

$$
Z_{k,t}=(1-\lambda_{k,t})q_k+
\lambda_{k,t}(r_{k,t}+\mu_{k,t+1}),\qquad
A_{k,t}=(Z_{k,t}-b_{k,t})/R.
$$

下一状态的特征、支持与存活正同伴指示都在当前真实转移后确定，$\lambda$ 不读取目标在这之后的成败。当唯一胜者或领先轨迹进入后续深度且所有剩余同伴均失败（$N^+_{-k,t+1}=0$）时，$\lambda_{k,t}=0$，从而完整保留自身终局 Monte Carlo 回报 $q_k$。终末位置同样强制 $\lambda=0$。没有支持时 $\mu=0$ 是上述定义的一部分，目标仍因 $\lambda=0$ 使用自身真实回报。

在合法输入上，$0\leq b,\mu,Z\leq R$ 且 $A\in[-1,1]$。全失败组满足 $b=\mu=Z=A=0$。默认公式下，失败非终末动作获得正优势的充要条件是 $\lambda\mu_{t+1}>b_t$；“有成功同伴”本身只决定其中一部分量。

### 配置及方法 ID

| 配置项 | 默认值 | 实现约束及含义 |
|---|---|---|
| `reward_scale` | `10.0` | 有限正数；正式协议为 $R=10$，所有回报与优势共用该固定尺度 |
| `credit_radius` | `0.10` | 有限正数，核半径 $\epsilon$ |
| `credit_prior_weight` | `1.0` | 有限正数，$\kappa$ 同时控制动作前先验和默认共享收缩 |
| `credit_transfer_max` | `0.5` | $[0,1]$ 内，默认公式的 $\lambda_{\max}$ |
| `gamma` | `1.0` | r3 信用公式固定无折扣；不得通过旧折扣回报替换 |

| 方法 | 与 `fgpo` 的唯一区别 |
|---|---|
| `fgpo_no_transfer` | 所有共享权重 $\lambda=0$，原核与基线保留 |
| `fgpo_obs_only` | ALFWorld 仅用精确纯观测匹配 $m^{\mathrm{obs}}$、AppWorld 仅保留观测余弦距离，actor 的历史窗口不变 |
| `fgpo_no_support_weight` | 非终末且下一状态满足 $N^+_{-k,t+1}\ge 1 \land S>0$ 时 $\lambda=1$，否则为零；该消融不使用 `credit_transfer_max` |

正式消融只使用表中三个方法，全部在 ALFWorld 上执行；完整 F-GPO 复用主比较中的对应种子。短两轮历史变体 `fgpo_no_full_filtration` 不进入正式矩阵。旧 BEFP、DAG-LOO 与自定义 shared 变体不映射为已发表基线；主比较使用第 9 节的 GiGPO、HGPO 和 GraphGPO。

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

良好前缀可以获得负优势。八轨迹组中目标失败、七个同伴成功，且当前和下一状态都完全匹配时，默认参数（$\lambda_{\max}=0.5$）给出 $b=R,\mu_{t+1}=R,N^+_{-k,t+1}=7\ge 1,\lambda=0.5\times 7/8=7/16,A=-9/16$。同质组（全失败或全成功）的任务优势全部严格为零。熵正则仅提供有限探索辅助，r3 没有冷启动成功保证。

## 6. F-GPO 原子混合组 PPO 及轨迹归一化

固定行为 log-prob、参考 log-prob 和步骤优势；每个实际响应 token $i$ 的优势为所属步骤 $A_i$。F-GPO 及三个自身消融仅筛选混合结果组 $\mathcal G_{\mathrm{mix}}=\{g:1\le\sum_{k\in g}Y_k\le K-1\}$（$M=|\mathcal G_{\mathrm{mix}}|$）构成原子整组批次：当 $M=0$ 时整轮旁路参考前向、PPO 与 AdamW（`0` 次优化器更新）；当 $M\ge 1$ 时按每个完整 $K=8$ 轨迹混合组执行一次独立参数更新，固定轨迹分母为 $Z_m=K=8.0$。设 $\rho_i=\exp(\ell_i^\theta-\ell_i^{\mathrm{old}})$，clip $\eta=.2$，dual clip $c=3$：

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

F-GPO 及三个自身消融的每个原子混合组批次 $m\in\mathcal G_{\mathrm{mix}}$ 使用固定分母 $Z_m=K=8.0$：

$$
\mathcal L_m=\frac{1}{Z_m}\sum_{i\in\mathcal B_m}
(\ell_i^{\mathrm{PPO}}-.001H_i+.01k_i).
$$

这是组内所有响应 token 损失的求和，轨迹内部不按长度平均；完整轮在恢复后保持相同的混合组序列、`minibatch_count`、分母 $Z_m=8$ 与 `atomic_mixed_group_trajectory` 权重身份。`metrics.loss` 累加实际混合组批次损失，并在 `cursor.metrics` 记录逐批裁剪前梯度范数 `batch_gradient_norms`、裁剪缩放因子 `batch_clip_scales`、PPO 裁剪词元占比 `ppo_clip_fraction` 以及首批次更新前同权重对数概率一致性误差 `first_batch_mean_abs_logprob_diff` / `first_batch_max_abs_logprob_diff`，比较时同时核对有效混合组数与 optimizer step 数。

每个 minibatch 完成后执行一次 AdamW，默认学习率 `1e-6`、weight decay `.01`、全局梯度裁剪 `1`。参考权重固定；行为权重随训练轮更新。PPO 裁剪、k3 采样正则、条件熵及有偏续值估计构成实际正则化 surrogate，其性质不能直接等同于每次更新都计算精确成功率梯度。本节的轨迹分母只用于 F-GPO 系列。已发表基线保留发布源码的 batch 复制与顺序、virtual rank/microbatch 内部 token-mean 及其外层平均权重；GiGPO 在优势估计之前复制，HGPO/GraphGPO 在优势估计之后复制。分组与每个响应的损失系数均须遵循这一执行顺序，具体配方和批处理规则由 `baseline_specs.py / published_batching.py` 定义。

对已发表方法，令来源 virtual rank 数为 $W$、每 rank 的 actor microbatch 行数为 $b$、全局 PPO minibatch 行数为 $B$，固定梯度累积数为 $A=B/(Wb)$。第 $m$ 个 virtual microbatch 的真实响应 token 数为 $N_m$，则该批每个 token 的损失系数为

$$
c_i=\frac{1}{W A N_m}.
$$

最后一个不满 PPO minibatch 仍使用来源固定的 $A$。`plan_published_batch` 保留复制、按序列长度平衡及各 rank 内的行顺序，将同一个 optimizer step 内重复响应行的正系数相加；`ppo_update(..., published_plan=...)` 只执行该行一次前向，将合并系数同时乘 PPO、熵、KL 及对应统计。合并不跨越参数更新边界。`EngineState.published_batch_plan` 保存计划，`credit_row_ids` 保留 GiGPO 估计所需的重复次数，cursor 记录 `published_virtual_microbatch_mean` 权重身份。batch 复制仅引用已采样响应；实际采样预算按原始轨迹与环境调用计数。

## 7. 系统执行契约

因果段只在相同参数版本下共享主干前向。PPO 对当前段仅前向至所选响应的最大 `response_span.stop`，送入主干的 token 为该前缀去掉最后一个目标 token；所有前文和绝对位置保留。某段在参数更新后的另一个 minibatch 再次使用时，重新计算其主干激活。

设主干输出为 $H=f_\theta(x)$，一次可微位置 gather 得到实际响应隐藏态 $Q=H[I]$。每个默认最多 64 个 token 的词表块使用 $Q_c$ 的 detached 叶子计算并立即反向，收集 $\partial L/\partial Q_c$；全部块结束后执行一次 `Q.backward(dL/dQ)`，由 gather 将梯度散射回主干。循环期间仅保持响应位置梯度缓冲，主干反向所需的完整激活仍由训练 checkpoint 策略管理。跨设备复制参与 autograd，绑权 embedding/head 的两条路径累积到同一参数。

词表投影输出 FP32 logits。`_LogProbabilityEntropy` 保存 logits、行级 log-partition、熵和目标索引，并使用解析一阶 adjoint。对 $p=\operatorname{softmax}(z)$、$\ell=\log p_y$、熵 $\mathcal H$，

$$
\frac{\partial\ell}{\partial z}=\operatorname{onehot}(y)-p,\qquad
\frac{\partial\mathcal H}{\partial z}=-p(\log p+\mathcal H),
$$

$$
\frac{\partial L}{\partial z}
=g_\ell\bigl(\operatorname{onehot}(y)-p\bigr)-g_Hp(\log p+\mathcal H).
$$

其中 $g_H=\partial L/\partial\mathcal H$；F-GPO 使用 $g_H=-\mathrm{entropy\_coef}/Z_{\mathrm{mb}}$，已发表基线使用其 virtual rank/microbatch 规则对应的逐响应系数。$g_\ell$ 由 PPO、dual clip、k3 截断及相同损失系数的小图求得。反向重建概率，原位复用仅涉及新建临时张量，保存的 logits 保持不变。`once_differentiable` 将接口限定为一阶训练。bf16 主干、FP32 词表统计与运算顺序共同决定实际数值，正式记录中的数值容差与实测结果不声明逐位一致。

参考阶段仅在当轮存在混合结果组（$M_{\mathrm{mix}}\ge 1$）时集中安装冻结权重，完成混合组轨迹的响应概率与所需特征计算后恢复 actor；当 $M_{\mathrm{mix}}=0$ 时整轮旁路参考前向与特征提取。同段实际响应合并为一次目标上传和一次分数回传，词表投影仍按 chunk 执行，每块立即释放 logits。历史层 hook 只保留动作前所需位置；`fgpo_obs_only` 跳过历史表示。ALFWorld 直接在 `credit.py` 比较离散纯观测词元元组从而免除 `z_obs` 模型前向；AppWorld 的公开观测按 token 序列去重、按长度和 token 预算微批编码，纯表示前向可在指定中间层结束。信用及组诊断完成后调用 `ReferenceRound.release_features()` 释放 `z_hist/z_obs`，保留 PPO 所需的冻结 log-prob、已计算优势和诊断。尚待信用计算的断点保留其所需表示。

固定参考通过 `WeightReference(fixed=True)` 声明不可变。actor 参数仅由 `OffloadedAdamW` 写入；已初始化参数从其 FP32 master 在 CPU 舍入为 actor dtype 后分桶复制回设备，与 Adam step 的 CPU 权重桶舍入路径相同。没有 Adam 状态的参数从初始参考恢复。安装前验证参数对象归属及源张量形状/dtype，异常退出也恢复已经安装的参数；普通 reference 切换不建立整 actor CPU 备份。EMA reference 保留独立接口，不能借用初始参考恢复约定。

采样槽位独立拥有 generator，token 与对应行为 log-prob 合并读取；F-GPO 采样支持 `active-rows-v3` 停放完成槽解码（`cached_query_key_positions` 按实际非填充查询/键范围累计）。主机已知 `right_padded` 且物理列全有效时，cached 单 token 解码传入 `attention_mask=None` 避免冗余掩码与 GQA 头展开，同时完整保留有界填充/ragged 及无缓存显式 `position_ids` 语义。F-GPO 在响应结束后立即派发该槽的 CPU 环境调用，下一交互轮保留 turn barrier。已发表方法的完整原生 prompt 使用批量 prefill；完全相同的 prompt 先单次 prefill 再复制缓存，其他 prompt 显式右侧 padding，并按每行最后一个真实 token 取分布。结束槽及时移除，停放分片仅按占用长度保留并在轮转直接筛选续用行；增量 head 只投影本次有新增 token 的行，未变化行保留已有分布。批处理保持各槽原有 generator、逻辑位置和 prompt token。

arena cache 按容量几何增长并保留 padding mask、实际有效跨度与逻辑 position；`select_cache`/`split_cache` 为继续行保留扩容余量并裁去尾部全空列，`append_cache` 按词元预算分段处理长增量重放并约束物理填充不越过上下文上限。`compact` 仅对含内部空洞的行移动数据；`crop` 只缩短逻辑视图并保留容量。丢弃有效历史后重新 prefill 新窗口。持久环境池复用进程，独占 lease 隔离轨迹，新的任务重新初始化原生环境对象。缓存、增量 logits 和环境句柄均有明确的消费与释放边界。

CPU AdamW 保存 FP32 master/m/v，先完成全局范数的设备归约与有限值检查。传输桶默认 64 MiB，每个设备与 dtype 最多使用两个有限槽；每槽的前次 H2D/参数写回、下次 D2H 和 `downloaded` event 按同一 CUDA stream 排序。CPU 在该 event 完成后读取梯度或覆盖 host weight，因此不再额外提前阻塞等待前次 H2D。梯度桶转 FP32 后原位缩放、完成当前桶更新后释放临时量；step 返回前同步全部写回。主干梯度 checkpoint 保持启用；`activation_offload=false` 为默认值，显式开启时采用非锁页 CPU 激活保存。

热路径模型调用在解码、参考与 PPO 段粒度按操作、role、真实轮次和组归属聚合日志，保留 calls、实际 token、padded compute token、投影位置数和 host dispatch 总量；PPO 裁剪计数、有序 FP64 差值和与最大值在输出设备按段归集并在段持久化边界读回；行为段 `SegmentRecord` 一次冻结共享，已完成轮的轨迹与特征引用在轮末边界释放。环境转移、同步阶段、启动恢复与断点/最终发布边界保留各自事件。`host_dispatch` 表示主机提交时间。阶段边界同步 CUDA 后记录 `synchronized_wall` 与 allocated/reserved/peak bytes，嵌套峰值归属外层阶段。端到端成本采用实际运行墙钟，环境与 GPU 重叠时间不重复相加。

## 8. 正式日志诊断

对非空组，`advantages(..., diagnostics=output)` 将诊断写入 `output[group_id]`，同时返回 `{step_id: float_advantage}`。`engine.py` 将这些诊断与整轮执行健康指标写入正式 `training.jsonl`（其中 `diagnostics.credit_groups` 保存逐组记录）；当前更新与恢复状态保留冻结响应分数、优势及实际轨迹，用于计算这些量的参考表示在信用完成后释放。历史原始轨迹与逐 token 分数只在显式 `retain_training_artifacts=true` 时另存；默认日志中的支持、门控、Brier、信用改变、批次范数、首批概率一致性与成本字段完整保留，覆盖本方案 claim。

| 日志字段 | 精确定义 |
|---|---|
| `trajectory_count / step_count / successes` | 组内轨迹数、真实动作数、官方成功轨迹数 |
| `all_zero_group / all_one_group` | 所有轨迹官方成功均为零 / 均为一 |
| `homogeneous_group / homogeneous_bypass` | 组结果同质指示及同质直出零优势旁路标志 |
| `local_observation_matching` | 局部观测匹配模式（ALFWorld 为 `exact_token_id`，AppWorld 为 `dual_view_cosine`） |
| `matched_steps / matched_step_fraction` | 当前状态 $S_t>0$ 的真实步骤数及其占比 |
| `transferred_steps / transfer_step_fraction` | 当前转移 $\lambda_t>0$ 的真实步骤数及其占比 |
| `positive_peer_gated_steps` | 下一深度存在局部支持（$S_{t+1}>0$）但因存活成功同伴为零（$N^+_{-k,t+1}=0$）而拦截共享的非终末步骤数 |
| `mean_support_mass` | 所有真实状态的 $S_t$ 均值 |
| `mean_effective_peers` | $(\sum_jw_j)^2/\sum_jw_j^2$ 的真实状态均值；无支持时为零 |
| `mean_transfer_weight` | 所有真实转移的 $\lambda_t$ 均值，包括末步的零 |
| `loo_outcome_brier` | 所有真实步骤上的 $(b_t/R-Y_k)^2$ 均值 |
| `loo_task_prior_brier` | 所有真实步骤上的 $(\bar q_{-k}/R-Y_k)^2$ 均值 |
| `matched_loo_outcome_brier` | 仅 $S_t>0$ 位置上的 $(\mu_t/R-Y_k)^2$ 均值；没有匹配时为 null |
| `mean_abs_return_shift` | 所有真实步骤上的 $\lvert Z_t/R-Y_k\rvert$ 均值 |
| `positive_failed_steps` | $Y_k=0$ 且 $A_{k,t}>0$ 的真实步骤数 |
| `mean_advantage / advantage_min / advantage_max` | 所有真实步骤优势的均值、最小值和最大值 |

轮级 `training.jsonl` 另行在 `diagnostics` 写入 `total_groups`、`mixed_groups`、`homogeneous_groups`、`all_one_groups`、`homogeneous_round_bypass`、`optimizer_steps_in_round`、`reference_scored_trajectories`、`total_steps`、`syntax_valid_steps`、`syntax_valid_fraction`、`admissible_valid_steps`、`admissible_valid_fraction` 与 `unterminated_response_steps`，并在 `metrics` 写入逐批裁剪前范数 `batch_gradient_norms`、裁剪缩放比 `batch_clip_scales`、PPO 裁剪占比 `ppo_clip_fraction` 与首批次一致性指标 `first_batch_mean_abs_logprob_diff` / `first_batch_max_abs_logprob_diff`。组诊断还记录算法、方法、时域及实际信用参数。Brier 使用整条目标轨迹之外的数据进行预测，再用目标终局结果计算损失；它包含 Bernoulli outcome 噪声，衡量留出结果预测，不能作为真实 Q 误差、置信界或校准分数。这些诊断不参与共享权重或奖励。跨组聚合采用相应真实样本数加权。

这些字段只用于正式主比较和自身消融的 claim：历史匹配是否改变同伴支持与留出结果预测、续值共享是否实际改变失败轨迹的信用、支持收缩是否限制低支持更新。诊断直接读取已完成的训练日志；不增加反馈探针、额外 rollout、反事实分支或独立诊断实验。正优势计数描述更新发生的位置，对这些动作质量的判断结合官方结果及对应自身消融。

## 9. 研究身份、基线与源码映射

运行身份由协议 fgpo-r3-published-v4 与其实际配置共同决定。主比较与自身消融分开保存，当前执行范围见[实验排程](fgpo_experiment_schedule_current.md)。

| 正式方法 ID | 来源与本研究采用的配方 | 原 ALFWorld 配方的主要设置 |
|---|---|---|
| `fgpo` | 本文件第 1—6 节 | 60 轮，16 任务组/轮，8 轨迹/组，响应 512，混合结果组 $K=8$ 原子整组轨迹归一化，$\lambda_{\max}=0.5$，$\gamma=1$ |
| `gigpo` | [GiGPO NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)；官方 1.5B full-finetuning 配方迁移到本研究 7B | 150 轮，最近两轮记忆，prompt 2048，响应 512，row minibatch 256，$\gamma=.95$ |
| `hgpo` | [HGPO ICLR 2026](https://openreview.net/forum?id=T8Dev99qnz)及作者发布代码 | 160 轮，最近两轮记忆，prompt 4096，响应 512，row minibatch 256，$\gamma=.95$ |
| `graphgpo` | [GraphGPO ICML 2026](https://proceedings.mlr.press/v306/cheng26e.html)及作者发布代码 | 150 轮，最近两轮记忆与公开状态摘要，prompt 2048，响应 512，7B row minibatch 64，$\gamma=.10$ |

三种已发表方法采用 [`verl-agent` 固定来源快照](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)的 `released-source-recipe`，来源文件、论文/代码差异和解析后的参数集中在 `src/fgpo/baseline_specs.py`。保留来源中的分组、回报、归一化、无效动作惩罚、prompt 与截断方式、采样温度，以及 batch 复制、平衡顺序和 virtual rank/microbatch token-mean 权重。GiGPO 的步骤行 episode 统计、HGPO 的非零层级项重归一化与 population std、GraphGPO 的下一节点距离指数均按该代码来源实现，并在运行身份中披露。这些基线使用自己的原生 collector 和优势入口。

本研究使用的来源虚拟批参数 $(W,b,A)$ 分别为 GiGPO $(2,32,4)$、HGPO $(4,8,8)$、GraphGPO $(4,4,4)$。这些值定义数学上的响应权重，和实际执行模型的物理 GPU 数分别记录。GiGPO 固定来源中可用的 full-finetuning 脚本使用 Qwen2.5-1.5B，本研究将其原配方迁移到统一的 Qwen2.5-7B，并保留该来源的两 rank 与 32 行 microbatch 规则。HGPO 和 GraphGPO 使用其 7B 配方。

ALFWorld 保留发布算法与来源配方并披露上述 backbone 迁移。四方法运行于本仓库的 SDPA、缓存、offloaded AdamW 与日志路径，原作者的 vLLM/Ray/FSDP 运行栈不构成本次逐位复现承诺。各方法保留自己的训练轮数、prompt、输出和更新配方，官方测试只评价最终模型。共同预算区间内的曲线仅来自每轮已有行为采样的训练成功率日志，不增加中途测试集评价；实际轨迹、token、环境步、optimizer step、GPU 小时和峰值显存共同说明成本。

主比较衡量正式官方效果与实际成本。三个自身消融的目标依次为历史条件匹配、下一状态续值共享和支持收缩。诊断复用正式日志。已启动的消融保存并暂停，未启动的消融移出队列。

官方终评按各实验预先指定的完整任务集逐种子记录指标、分母和任务完成数。

`effective_training_identity/require_training_identity` 约束恢复时相同的协议、方法、种子、模型、任务、上下文、生成、信用、PPO 和 Adam 有效参数，已发表方法同时保留来源与解析配方。GPU 放置、缓存容量、offload、传输桶及 head/representation 微批属于资源项，row minibatch（或 F-GPO 原子混合组语义）属于训练身份。CUDA RNG 流数量保持一致。旧 `fgpo-r3-published-v3`、`fgpo-r3-success-v2` 及更早协议的待执行优势和优化状态禁止复用；中断恢复以已保存完整组、信用结果和更新 cursor 为单位。

完整恢复快照默认在初始任务顺序确定后、每 8 轮、训练结束和托管中断的安全边界保存。采样/reference 进度保存只复用同一 optimizer model version 的 actor 与 Adam；有未保存更新时等待下一个完整保存点；同质轮（$M_{\mathrm{mix}}=0$）在 `reference -> update` 边界设 `durable=False` 避免冗余持久化。`CURRENT` 发布前同步参数、trainer/RNG、训练与评估日志、实际成本游标；所有 GCS 读路径（`current`/`inspect`/`status`/`pull`/`restore_cell`/`prepare_resume`/`load`/`restore_final` 及 `resources`）均通过 `open_snapshot`/`_reader_pins` 发布的 `readers/<lease>.json` 租约与全量 GC 闭包保留防止读取期间误删，并在指针切换竞争或校验异常时清理租约而不消费过期对象或回落未加锁读取。硬中断恢复时，训练统计截回 `EngineState.training_log_offset`；成本统计接纳游标之后完整的实际事件，重算成本另外记录。最后一条未完成的成本记录可由本地截断或远端恢复放弃（包含相同 `committed_bytes` 前缀上来自 `LOGS` 的 `uncommitted_tail_bytes_at_publish` 合并），其缺失必须一次计入并保留单调 `discarded_incomplete_tail_bytes`、标记 `usage_complete=false`（区别于活跃写入期间的瞬时 `uncommitted_tail_bytes_at_publish`）。完整评测只复用同一最终 model version 的结果。

`checkpoint_interval_rounds`、`checkpoint_interval_groups`、`checkpoint_shard_mib`、`retain_training_artifacts` 与主机内存预留属于资源/持久化选项，不改变训练身份。默认产物保留配置、来源、任务、claim 诊断、成本、官方评估和最终模型；AppWorld 在复制评估结果并关闭该 episode 后回收其独占输出。启动冷恢复与进引擎前开销通过 `CostLedger.credit_startup` 锚定进 `wall_seconds` / `allocated_device_seconds`。终态依次完成 `reclaim_completed_checkpoint`、计量 `publish_final_logs`（写入 `checkpoint.final_publish_logs`）、冻结重写 `finalization_stage="finalized"` 的 `cell_summary.json`，随后执行明确不计入自身成本的终端导出 `export_final_terminal_state` 同步至 `FINAL`/`LOGS` 并写入 `.finalization_complete.json`；任何初步摘要或导出阶段的中断在 `--resume` 时由 `_resume_pending_finalization` 继续完成。恢复所需的当前组、reference、优势、梯度与 cursor 始终随完整快照一起保存。

| 文件 | 负责的契约 |
|---|---|
| `public.py / filtration.py / records.py` | 公开输入（含 ALFWorld 三块整数帧与 `<think>/<action>` 解析）、实际有界历史、词元所有权和步骤记录 |
| `rollout.py / envs.py` | 独立采样（含 `active-rows-v3` 停放完成槽解码）、提前环境派发、轮间同步和持久隔离 lease |
| `policy.py / cache.py` | 因果前向、表征批处理、采样、缓存和阶段实测 |
| `reference.py` | 混合组段级分块参考评分、AppWorld 去重局部观测表示、信用后释放表示 |
| `credit.py` | 同结果组旁路、同深度排除、ALFWorld 离散/AppWorld 余弦匹配、正同伴门控续值共享和诊断 |
| `optimization.py` | 原子混合组批次与分母、解析一阶 head adjoint、响应梯度、逐批范数与首批一致性统计和双槽 AdamW |
| `baseline_specs.py / published_baselines.py / published_batching.py / baselines.py` | 已发表来源配方、信用算法、batch 复制与更新权重、原生 prompt/动作与 benchmark 迁移 |
| `protocol.py / engine.py / runner.py` | published-v4 方法身份、正式矩阵、同质轮零步旁路、GCS/本地恢复与生命周期 |

任务结果、阶段成本和运行资源以服务器正式记录为准。
