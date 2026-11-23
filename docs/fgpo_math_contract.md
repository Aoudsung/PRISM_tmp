# F-GPO r8 数学与实现契约

本文件与 [研究方案及推导](fgpo_design_and_derivation.md) 定义 `fgpo-r8-scale-invariant-v1` 的实际实现（算法标识 `r8_symmetric_empirical_bayes_peer_sharing`），版本日期为 2026-11-23。第 1—6 节规定 F-GPO 及其自身消融的数学对象；第 9 节单列已发表基线的来源、配方和比较范围。核心实现位于 `src/fgpo/{public,filtration,reference,credit,optimization,textcraft,checkpoint_gcs}.py`。旧 `r7`、`r6`、`r5`、`r3/v4` 与旧 TF-BEFP/DAG 可通过 git 提交追溯；本协议使用新 study 目录，不接续旧协议奖励或未完成更新。

本文规定数据、公式、默认值、更新和测量口径。所有项目执行、数值验证及正式评测均在指定服务器与 Colab 实例进行。

## 1. 目标与输入对象

对同任务组的 $K=8$ 条独立轨迹，记轨迹长度为 $T_k\leq H$，官方终局成功为 $Y_k\in\{0,1\}$。定义 $R=10$、$q_k=RY_k$、$\gamma=1$：

$$
r_{k,t}=q_k\mathbf1[t=T_k-1],\qquad
G^{\mathrm{MC}}_{k,t}=\sum_{u=t}^{T_k-1}r_{k,u}=q_k.
$$

`credit.py` 从 `official_success` 重建回报，不使用记录中的任意环境中间 reward、参考 reward 或语法标签替代官方终局目标。主方法没有 TF-BEFP、目标似然奖励、手工语法惩罚及随机 RMS。

每组轨迹的 `group_id/task_id/benchmark/initial_seed/goal` 相同，`trajectory_id` 唯一。步骤 `row.index` 连续为 $0,\ldots,T_k-1$，合法终末标志由

$$
\operatorname{terminal}_{k,t}
=\mathbf1[t=T_k-1]
=\mathbf1[\texttt{row.done}\ \lor\ t+1=H]
$$

确定。默认训练时域 $H_{\mathrm{ALF}}=50$、$H_{\mathrm{WebShop}}=15$、$H_{\mathrm{TextCraft}}=60$（对齐公开 1.5B GiGPO/BiPACE 预算），TextCraft 正式评测采用 `EVAL_HORIZONS["textcraft"] = 70` 并从单次 79 题评测轨迹同步统计 30 步、60 步、70 步成功率与第 31–70 步新解决题目。全组结果同质（$\sum_k Y_k\in\{0,K\}$）时返回零优势并跳过参考前向、特征计算与优化器更新。

## 2. 保留标签的规范充分 POMDP 滤子与词元所有权

数学公开历史记为 $h_t=(g,o_0,u_0,\ldots,u_{t-1},o_t)$，actor 与冻结参考在动作前严格共享同一规范充分投影 $\Pi(h_t)$：

1. **共享根前缀 `canonical_root_prefix`**：包含 ChatML system 指令（TextCraft 直接复用原样 `TEXTCRAFT_SYSTEM_PROMPT`，ALFWorld 保留 `First reason concisely enclosed within literal <think> and </think> tags, then output exactly one admissible command verbatim without quotes enclosed within literal <action> and </action> tags.` 显式字面标签指令，确保冷启动 XML 格式收敛与已发表基线零冲突）、任务目标 `Task` / ALFWorld 初始场景概览 / TextCraft 完整去 JSON 转义的 `Crafting commands + Goal`，在整个 episode 期间只 prefill 一次并在 KV 根缓存复用。
2. **保留标签的紧凑规范历史 `canonical_history_entry_ids`**：对已执行步骤 $\tau\in\{1,\ldots,t\}$ 在离散基准保持 `[Step \tau] <action>u_{\tau-1}</action> -> ...`（其中 WebShop 通过 `strip_webshop_repeated_header` 剥离重复 `WebShop [SEP]`、`Instruction:` 与 `Back to Search` 导航头后再保留最多 320 字符页态与可点选项；AppWorld 保留 `[Step \tau] Code: ... -> Observation: ...`）。合法语法解析失败记为 `(invalid format: missing <think>...</think><action>...</action>)`，剥离过期菜单与历史 `<think>` 推理独白同时保留 `<action>` 输出范式，防止 1.5B 坏格式传染与语法退化。
3. **当前决策尾缀 `canonical_step_tail`**：在历史上限 $C_{\mathrm{prompt}}=C_{\max}-M_{\mathrm{gen}}$ 内保留最新完整历史条目后，拼接当前步计数 `Current step: t+1/H`、当前观测 `pure_ids`（TextCraft 为非 JSON `Latest feedback + Inventory`）、当前合法指令列表及显式字面标签指令（`Enclose concise reasoning in literal <think> and </think> tags, then enclose one exact ... in literal <action> and </action> tags.`）。TextCraft 系统提示明确陈述两条公开模拟器公理（`get` 仅适用于不作为任何 `craft` 产物的叶原料；`craft` 必须与所列配方名称与精确整数量逐字一致，需多份时重复执行）。

每步仅在当前步新建独立 `SegmentRecord` 承载 `Prompt_t + y_t`，实际采样词元 $y_t$ 拥有唯一一次损失归属；参考与 PPO 阶段均采用长度排序的零掩码右填充无缓存因果微批（`attention_mask=None`）提取 `z_hist` 与隐藏态。

## 3. 尺度不变表征、精确离散公开状态与有界跨深度同伴核

冻结参考模型在比例语义层 $\ell^*(L)=\lfloor 5L/7\rfloor$（对于 28 层的 Qwen2.5-1.5B/7B 即第 20 层，等价 `-8`）取实际动作前 prompt 末位归一化表示作为 $z^h_{k,t}$。在连续公开输出空间（**AppWorld**），将当前公开局部观测单独编码并在相同 $\ell^*(L)$ 层取末位归一化向量，得到 $z^o_{k,t}$，二者 L2 归一化并停止梯度；在三大离散动作基准（`EXACT_OBSERVATION_BENCHMARKS = {"alfworld", "webshop", "textcraft"}`），直接比较非混叠离散公开状态 `representation_ids` 整数词元元组是否精确相等 $m^{\mathrm{obs}}_{(k,t),(j,t')}=\mathbf 1[o^{\mathrm{repr}}_{k,t}=o^{\mathrm{repr}}_{j,t'}]$，省去 `z_obs` 模型前向：

- **TextCraft**：持久可见背包文本 `inventory`（剥离瞬态 `feedback`）；
- **WebShop**：`strip_webshop_repeated_header(text, goal=goal)` 剥离重复模板头后的当前页面本体拼接有序 `||` 可点动作菜单 `available_actions`；
- **ALFWorld**：有序可交互命令集 `available_actions` 拼接由纯公开环境反馈追踪的持久状态字典 `public_object_effects`（`clean:<obj>=clean`、`temp:<obj>=hot/cool`、`power:<obj>=on` 及其关闭擦除、`sliced:<obj>=sliced`）。第零步使用初始观测，后续步骤使用上一动作返回的公开观测；空文本用 `[Empty stdout]` 的表示 token。

定义有效位置 $v_{k,t}=\mathbf1[t<T_k]$，标准化时间差 $\tau_{t,t'}=\frac{|t-t'|}{H}$，当 $\epsilon_{\mathrm{time}}>0$ 时时间比 $r^{\mathrm{time}}_{t,t'}=\tau_{t,t'}/\epsilon_{\mathrm{time}}$（$\epsilon_{\mathrm{time}}=0$ 退化为严格同深度 $t'=t$）。同伴 $j\ne k$ 只需在时间窗 $r^{\mathrm{time}}_{t,t'}<1$ 内存在有效步 $t'$，每个同伴轨迹 $j\ne k$ 对目标步 $(k,t)$ **至多贡献一票**（取窗口内最大核权重）：

- 在 **AppWorld**（连续状态空间，非 `fgpo_obs_only` 时 $d_{\mathrm{rep}}=\max(d^o, d^h)$，仅观测消融仅用 $d^o$）：
$$
w_{k,t,j}=\max_{t':\, r^{\mathrm{time}}_{t,t'}<1} v_{k,t}v_{j,t'}\mathbf1[j\ne k]\left[1-\max\!\left(\frac{d_{\mathrm{rep}}((k,t),(j,t'))}{\epsilon_{\mathrm{rep}}},\;r^{\mathrm{time}}_{t,t'}\right)\right]_+,\qquad \epsilon_{\mathrm{rep}}=0.08,\;\epsilon_{\mathrm{time}}=0.10.
$$
- 在三大离散基准 **ALFWorld / WebShop / TextCraft**（经 $m^{\mathrm{obs}}_{(k,t),(j,t')}$ 精确门控，`fgpo_obs_only` 时仅以时间衰减加权）：
$$
w_{k,t,j}=\max_{t':\, r^{\mathrm{time}}_{t,t'}<1} v_{k,t}v_{j,t'}\mathbf1[j\ne k]\,m^{\mathrm{obs}}_{(k,t),(j,t')}\left[1-\max\!\left(\frac{d^h((k,t),(j,t'))}{\epsilon_{\mathrm{rep}}},\;r^{\mathrm{time}}_{t,t'}\right)\right]_+,\qquad \epsilon_{\mathrm{rep}}=0.08,\;\epsilon_{\mathrm{time}}=0.10.
$$

支持质量 $S_{k,t}=\sum_{j\ne k} w_{k,t,j}$。因为每个同伴 $j\ne k$ 最多贡献 1 票，故 $S_{k,t}\le K-1=7$ 且不会因时间窗放宽产生多重计数；当轨迹长度错位几个步长（例如 `1.5B` 探索步数略多但达到相同关键状态）时，跨深度窗允许邻近深度互为可信支持并缓解尾部长败孤立问题。

令

$$
U_{k,t}=\sum_jw_{kj,t}q_j,\qquad
\mu_{k,t}=\begin{cases}U_{k,t}/S_{k,t},&S_{k,t}>0,\\0,&S_{k,t}=0.\end{cases}
$$

每个被共享的值来源都是真实 Monte Carlo 结果 $q_j$；算法没有递归值传播。$S$ 是有限同伴的核质量，不是校准置信概率。实现以 CPU float32 批量矩阵乘法计算距离，以 CPU float64 计算权重归约、回报、基线和优势。

## 4. 对称经验贝叶斯（Empirical Bayes）同伴势能

`r8` 采用对称先验权重与完整对称收缩（默认 $\kappa=1.0,\ \lambda_{\max}=1.0,\ \omega_{k,t}\equiv 1.0$，统一移除 `has_positive_peer` 与未决权重折扣 $\omega_{-k,t}$）：

$$
\bar q_{-k}=\frac{\sum_{j\ne k}q_j}{K-1},\qquad
\lambda^{\mathrm{pre}}_{k,t}=\lambda_{\max}\frac{S_{k,t}}{S_{k,t}+\kappa},\qquad
\lambda^{\mathrm{post}}_{k,t}=\mathbf 1[t<T_k-1]\,\lambda_{\max}\frac{S_{k,t+1}}{S_{k,t+1}+\kappa},
$$

$$
\tilde b_{k,t}=(1-\lambda^{\mathrm{pre}}_{k,t})\bar q_{-k}+\lambda^{\mathrm{pre}}_{k,t}\mu_{k,t},\qquad
Z_{k,t}=(1-\lambda^{\mathrm{post}}_{k,t})q_k+\lambda^{\mathrm{post}}_{k,t}(r_{k,t}+\mu_{k,t+1}),\qquad
\tilde A_{k,t}=\frac{1}{R}\bigl(Z_{k,t}-\tilde b_{k,t}\bigr).
$$

在默认 $\lambda_{\max}=1.0$ 下，动作前基线精确为经验贝叶斯（Empirical Bayes）后验预测均数 $\tilde b_{k,t}=\frac{S_{k,t}\mu_{k,t}+\kappa\bar q_{-k}}{S_{k,t}+\kappa}$，非终末目标为 $Z_{k,t}=\frac{S_{k,t+1}(r_{k,t}+\mu_{k,t+1})+\kappa q_k}{S_{k,t+1}+\kappa}$；同时全组保留均匀权重 $\omega_{k,t}\equiv 1$（不再随已完赛胜轨数改变深度权重，也不对单赢家尾步引入 $\frac{K-1}{K}$ 扭曲）。历史配对对照 `fgpo_exact_first_hit` 在对比时单独保持其原 `exact_first_hit_v4` 估计语义（$\epsilon=0.10$, 正同伴门控）。定义有界同伴势能 $\Phi_{-k,t}=\lambda^{\mathrm{pre}}_{k,t}(\mu_{k,t}-\bar q_{-k})$，因为 $\omega\equiv 1$，步骤差精确满足完整势能分解与全轨精确零和伸缩恒等式：

$$
Z_{k,t}-\tilde b_{k,t}=(1-\lambda^{\mathrm{post}}_{k,t})(q_k-\bar q_{-k})+\bigl(\Phi_{-k,t+1}-\Phi_{-k,t}\bigr),\qquad
\sum_{t=0}^{T_k-1}(\Phi_{-k,t+1}-\Phi_{-k,t})\equiv 0
$$

（末步 $\lambda^{\mathrm{pre}}_{k,T_k}=0$；当均匀根匹配给出 $\mu_{k,0}=\bar q_{-k}$ 时 $\Phi_{-k,0}=0$；组内跨轨迹后态预算近似守恒 $\sum_{k}Z_{k,t}\approx\sum_k q_k$）。同质组直接返回零优势。

在合法输入上，$0\leq \tilde b,\mu,Z\leq R$、$\omega\equiv 1$ 且 $\tilde A\in[-1,1]$。全失败组满足 $\tilde b=\mu=Z=\tilde A=0$。

### 配置及方法 ID

| 配置项 | 默认值 | 实现约束及含义 |
|---|---|---|
| `reward_scale` | `10.0` | 有限正数；正式协议为 $R=10$，所有回报与优势共用该固定尺度 |
| `credit_radius` | `0.08` | 有限正数，表示空间核半径 $\epsilon_{\mathrm{rep}}$ |
| `credit_time_radius` | `0.10` | 有限非负数，跨深度核半径 $\epsilon_{\mathrm{time}}$（`0.0` 严格退化为同深度） |
| `credit_prior_weight` | `1.0` | 有限正数，经验贝叶斯先验强度 $\kappa$ |
| `credit_transfer_max` | `1.0` | $[0,1]$ 内，对称收缩上限 $\lambda_{\max}$ |
| `gamma` | `1.0` | 信用公式固定无折扣；不得通过旧折扣回报替换 |

| 方法 | 与 `fgpo` 的唯一区别 |
|---|---|
| `fgpo_no_transfer` | 所有后续共享权重 $\lambda^{\mathrm{post}}=0$，跨深度核与对称 EB 动作前基线保留 |
| `fgpo_obs_only` | 三大离散基准仅用精确公开状态与跨深度时间权重 $m^{\mathrm{obs}}(1-r^{\mathrm{time}})_+$、AppWorld 仅用观测余弦距离，actor 历史窗口不变 |
| `fgpo_no_support_weight` | 当 $S>0$ 时 $\lambda^{\mathrm{pre}}=1$、非终末且下一状态 $S_{t+1}>0$ 时 $\lambda^{\mathrm{post}}=1$，否则为零；该消融不使用 `credit_transfer_max` 收缩分母 |

正式消融只使用表中三个方法；主比较使用第 9 节的 GiGPO、HGPO 和 GraphGPO。

## 5. 理论声明的适用范围

在固定初态、冻结策略 $\pi_{\mathrm{old}}$ 与独立同伴条件下，因 $\tilde b_{k,t}$ 仅依赖动作前历史与同伴数据，满足精确无偏基线正交条件 $\mathbb E[\tilde b_{k,t}\psi_{k,t}\mid h_{k,t},D_{-k}]=0$。若 $\lambda^{\mathrm{post}}_t$ 仅依赖当前转移及同伴数据，且目标后续条件回报均值为 $V^{\pi_{\mathrm{old}}}(h_{t+1})$，则相对于无权官方成功目标梯度的共享偏差精确为（不再含 $\omega_t-1$ 扭曲项）：

$$
\mathbb E[(Z_t-\tilde b_t)\psi_t]-\mathbb E[G_t^{\mathrm{MC}}\psi_t]
=\mathbb E\!\left[\lambda^{\mathrm{post}}_t
  \bigl(\mu_{t+1}-V^{\pi_{\mathrm{old}}}(h_{t+1})\bigr)\psi_t\right].
$$

在 $|\mu-V|\leq\delta$、$\lambda^{\mathrm{post}}\leq\lambda_{\max}$、$\mathbb E\|\psi_t\|\leq L_t$ 条件下，单步与全时域归一化误差上界分别为 $\lambda_{\max}\delta L_t/R$ 与 $\lambda_{\max}\delta\sum_t L_t/R$。八轨迹组中目标失败、七个同伴均成功且状态充分匹配（$\mu_{k,t}=\mu_{k,t+1}=R, S\gg \kappa$）时，经验贝叶斯基线 $\tilde b\to R$、惩罚仅聚焦于从高成功状态滑向失败轨迹转移的局部，而孤立单赢家（$M=1$）组晚期失败步保持准确负向基线 $\tilde A_{\ell,t}=-1/7$。

## 6. 混合组均衡词元均值、宽度感知学习率与零掩码右填充 SDPA PPO

固定行为 log-prob、参考 log-prob 和步骤优势；每个实际响应 token $i$ 的优势为所属步骤 $A_{k,t}$。F-GPO 及三个自身消融仅筛选混合结果组 $\mathcal G_{\mathrm{mix}}=\{g:1\le\sum_{k\in g}Y_k\le K-1\}$（$M=|\mathcal G_{\mathrm{mix}}|$）构成动作步样本池：当 $M=0$ 时整轮旁路参考前向、PPO 与 AdamW（`0` 次优化器更新）；当 $M\ge 1$ 时按 `ppo_epochs=1` 将全部混合组决策步打乱并均衡切分为 $\lceil N_{\mathrm{rows}}/256\rceil$ 个步级微批 $\mathcal B_m$（`ppo_minibatch_rows=256`, `loss_normalization="token"`, `entropy_normalization="token"`, `mixed_token_mean`, `shuffle=true`）。宽度感知学习率 $\eta(d_{\mathrm{model}})=10^{-6}\min(1, 3584/d_{\mathrm{model}})$ 在 `1.5B/7B` 统一为 `1e-6` 并在更宽基座按主干宽度反比例自动缩减。记 $N_{\mathrm{tok},m}=\sum_{(k,t)\in\mathcal B_m}|y_{k,t}|$，$\rho_i=\exp(\ell_i^\theta-\ell_i^{\mathrm{old}})$，clip $\eta=.2$，dual clip $c=3$：

$$
\mathcal L_m^{\mathrm{r8}}=\frac{1}{N_{\mathrm{tok},m}}\sum_{i\in\mathcal B_m}\bigl(\ell_i^{\mathrm{PPO}}+.01k_i-.001H_i\bigr).
$$

在参考（`reference.packed_segment` / `reference.observation_embedding`）与 PPO（`ppo.current_segment`）的无缓存（`use_cache=False, past=0`）变长右填充微批前向中，有效前缀区间 `[0, L_r)` 的每个查询位置 $q < L_r$ 在严格因果注意力下仅关注键位置 $k \le q < L_r$，完全不依赖右侧填充 `[L_r, W)` 且采用统一零起点线性位置编码 `position_ids = arange(W)`。因此代码直接传入 `attention_mask=None`（在 `CostLedger` 中记录 `"attention_mask_omitted": True`），使 Transformers 原生启用 FlashAttention-2 / 零掩码 SDPA+GQA 融合核并在不改变有效词元任何数学计算的前提下彻底免除 $O(BW^2)$ 4D 掩码构造与 KV 头内存展开；词表头分块与 `_LogProbabilityEntropy` 一阶伴随公式保持不变。

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

其中 $g_H=\partial L/\partial\mathcal H$；F-GPO 在 `entropy_normalization="token"` 下使用 $g_H=-\mathrm{entropy\_coef}/N_{\mathrm{tok},m}$，已发表基线使用其 virtual rank/microbatch 规则对应的逐响应系数。$g_\ell$ 由 PPO、dual clip、k3 截断及相同损失系数的小图求得。反向重建概率，原位复用仅涉及新建临时张量，保存的 logits 保持不变。`once_differentiable` 将接口限定为一阶训练。bf16 主干、FP32 词表统计与运算顺序共同决定实际数值，正式记录中的数值容差与实测结果不声明逐位一致。

参考阶段仅在当轮存在混合结果组（$M_{\mathrm{mix}}\ge 1$）时集中安装冻结权重，完成混合组轨迹的响应概率与所需特征计算后恢复 actor；当 $M_{\mathrm{mix}}=0$ 时整轮旁路参考前向与特征提取。同段实际响应合并为一次目标上传和一次分数回传，词表投影仍按 chunk 执行，每块立即释放 logits。历史层 hook 只保留动作前所需位置（$\ell^*(L)=\lfloor 5L/7\rfloor$）；`fgpo_obs_only` 跳过历史表示。ALFWorld、WebShop 与 TextCraft 均直接在 `credit.py` 比较精确非混叠离散 `representation_ids` 从而免除 `z_obs` 模型前向；AppWorld 的公开观测按 token 序列去重、按长度和 token 预算零掩码（`attention_mask=None`）微批编码，纯表示前向在指定比例层提前结束。信用及组诊断完成后调用 `ReferenceRound.release_features()` 释放 `z_hist/z_obs`，保留 PPO 所需的冻结 log-prob、已计算优势和诊断。尚待信用计算的断点保留其所需表示。

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
| `local_observation_matching` | 局部观测匹配模式（ALFWorld / WebShop / TextCraft 为 `exact_token_id`，AppWorld 为 `dual_view_cosine`） |
| `matched_steps / matched_step_fraction` | 当前状态 $S_t>0$ 的真实步骤数及其占比 |
| `transferred_steps / transfer_step_fraction` | 当前转移 $\lambda_t>0$ 的真实步骤数及其占比 |
| `positive_peer_gated_steps` | 下一深度存在局部支持（$S_{t+1}>0$）但因存活成功同伴为零（$N^+_{-k,t+1}=0$）而拦截共享的非终末步骤数（`r8` 主法零门控为 `0`，仅 `exact_first_hit` 启用） |
| `mean_support_mass` | 所有真实状态的 $S_t$ 均值 |
| `mean_effective_peers` | $(\sum_jw_j)^2/\sum_jw_j^2$ 的真实状态均值；无支持时为零 |
| `mean_transfer_weight` | 所有真实转移的 $\lambda^{\mathrm{post}}_t$ 均值，包括末步的零 |
| `mean_pending_weight` | 所有真实步骤上的未决权重均值（`r8` 恒为 `1.0`） |
| `loo_outcome_brier` | 所有真实步骤上的 $(\tilde b_t/R-Y_k)^2$ 均值 |
| `loo_task_prior_brier` | 所有真实步骤上的 $(\bar q_{-k}/R-Y_k)^2$ 均值 |
| `matched_loo_outcome_brier` | 仅 $S_t>0$ 位置上的 $(\mu_t/R-Y_k)^2$ 均值；没有匹配时为 null |
| `mean_abs_return_shift` | 所有真实步骤上的 $\lvert Z_t/R-Y_k\rvert$ 均值 |
| `positive_failed_steps` | $Y_k=0$ 且 $\tilde A_{k,t}>0$ 的真实步骤数 |
| `mean_advantage / advantage_min / advantage_max` | 所有真实步骤优势的均值、最小值和最大值 |

轮级 `training.jsonl` 另行在 `diagnostics` 写入 `total_groups`、`mixed_groups`、`homogeneous_groups`、`all_one_groups`、`homogeneous_round_bypass`、`optimizer_steps_in_round`、`reference_scored_trajectories`、`total_steps`、`syntax_valid_steps`、`syntax_valid_fraction`、`admissible_valid_steps`、`admissible_valid_fraction` 与 `unterminated_response_steps`，并在 `metrics` 写入逐批裁剪前范数 `batch_gradient_norms`、裁剪缩放比 `batch_clip_scales`、PPO 裁剪占比 `ppo_clip_fraction` 与首批次一致性指标 `first_batch_mean_abs_logprob_diff` / `first_batch_max_abs_logprob_diff`。组诊断还记录算法、方法、时域及实际信用参数（含 `credit_time_radius`）。Brier 使用整条目标轨迹之外的数据进行预测，再用目标终局结果计算损失；它包含 Bernoulli outcome 噪声，衡量留出结果预测，不能作为真实 Q 误差、置信界或校准分数。这些诊断不参与共享权重或奖励。跨组聚合采用相应真实样本数加权。

这些字段只用于正式主比较和自身消融的 claim：历史匹配是否改变同伴支持与留出结果预测、续值共享是否实际改变失败轨迹的信用、支持收缩是否限制低支持更新。诊断直接读取已完成的训练日志；不增加反馈探针、额外 rollout、反事实分支或独立诊断实验。正优势计数描述更新发生的位置，对这些动作质量的判断结合官方结果及对应自身消融。

## 9. 研究身份、基线与源码映射

运行身份由协议 `fgpo-r8-scale-invariant-v1` 与其实际配置共同决定。主比较与自身消融分开保存，统一跨尺度配置见 `configs/fgpo-r8-{1p5b,7b}-*.json`。

| 正式方法 ID | 来源与本研究采用的配方 | 原 ALFWorld 配方的主要设置 |
|---|---|---|
| `fgpo` | 本文件第 1—6 节 | `r8` 统一尺度不变配方（`1.5B`/`7B` 超参完全一致）：60 轮（TextCraft 50 轮 / WebShop 150 轮），16 任务组/轮，8 轨迹/组，响应 512，混合结果组均衡打乱 256 行微批与零掩码批内统一词元均值归一，跨深度核 $\epsilon_{\mathrm{rep}}=0.08, \epsilon_{\mathrm{time}}=0.10$，$\kappa=1.0, \lambda_{\max}=1.0, \omega\equiv 1.0, \gamma=1$ |
| `gigpo` | [GiGPO NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)；官方 1.5B full-finetuning 配方迁移到本研究 7B | 150 轮，最近两轮记忆，prompt 2048，响应 512，row minibatch 256，$\gamma=.95$ |
| `hgpo` | [HGPO ICLR 2026](https://openreview.net/forum?id=T8Dev99qnz)及作者发布代码 | 160 轮，最近两轮记忆，prompt 4096，响应 512，row minibatch 256，$\gamma=.95$ |
| `graphgpo` | [GraphGPO ICML 2026](https://proceedings.mlr.press/v306/cheng26e.html)及作者发布代码 | 150 轮，最近两轮记忆与公开状态摘要，prompt 2048，响应 512，7B row minibatch 64，$\gamma=.10$ |

三种已发表方法采用 [`verl-agent` 固定来源快照](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)的 `released-source-recipe`，来源文件、论文/代码差异和解析后的参数集中在 `src/fgpo/baseline_specs.py`。保留来源中的分组、回报、归一化、无效动作惩罚、prompt 与截断方式、采样温度，以及 batch 复制、平衡顺序和 virtual rank/microbatch token-mean 权重；`r8` 严禁变动 `TEXTCRAFT_SYSTEM_PROMPT` 原文，确保已发表基线在所有基准上的输入与配方零漂移。GiGPO 的步骤行 episode 统计、HGPO 的非零层级项重归一化与 population std、GraphGPO 的下一节点距离指数均按该代码来源实现，并在运行身份中披露。这些基线使用自己的原生 collector 和优势入口。

本研究使用的来源虚拟批参数 $(W,b,A)$ 分别为 GiGPO $(2,32,4)$、HGPO $(4,8,8)$、GraphGPO $(4,4,4)$。这些值定义数学上的响应权重，和实际执行模型的物理 GPU 数分别记录。GiGPO 固定来源中可用的 full-finetuning 脚本使用 Qwen2.5-1.5B，本研究将其原配方迁移到统一的 Qwen2.5-7B，并保留该来源的两 rank 与 32 行 microbatch 规则。HGPO 和 GraphGPO 使用其 7B 配方。

ALFWorld 保留发布算法与来源配方并披露上述 backbone 迁移。四方法运行于本仓库的 SDPA、缓存、offloaded AdamW 与日志路径，原作者的 vLLM/Ray/FSDP 运行栈不构成本次逐位复现承诺。各方法保留自己的训练轮数、prompt、输出和更新配方，官方测试只评价最终模型。共同预算区间内的曲线仅来自每轮已有行为采样的训练成功率日志，不增加中途测试集评价；实际轨迹、token、环境步、optimizer step、GPU 小时和峰值显存共同说明成本。

主比较衡量正式官方效果与实际成本。三个自身消融的目标依次为历史条件匹配、下一状态续值共享和支持收缩。诊断复用正式日志。已启动的消融保存并暂停，未启动的消融移出队列。

官方终评按各实验预先指定的完整任务集逐种子记录指标、分母和任务完成数。

`effective_training_identity/require_training_identity` 约束恢复时相同的协议、方法、种子、模型、任务、上下文、生成、信用、PPO 和 Adam 有效参数，已发表方法同时保留来源与解析配方。GPU 放置、缓存容量、offload、传输桶及 head/representation 微批属于资源项，row minibatch（或 F-GPO 原子混合组语义）属于训练身份。CUDA RNG 流数量保持一致。旧 `fgpo-r7-token-mean-v1`、`fgpo-r3-published-v3`、`fgpo-r3-success-v2` 及更早协议的待执行优势和优化状态禁止复用；中断恢复以已保存完整组、信用结果和更新 cursor 为单位。

完整恢复快照默认在初始任务顺序确定后、每 8 轮、训练结束和托管中断的安全边界保存。采样/reference 进度保存只复用同一 optimizer model version 的 actor 与 Adam；有未保存更新时等待下一个完整保存点；同质轮（$M_{\mathrm{mix}}=0$）在 `reference -> update` 边界设 `durable=False` 避免冗余持久化。`CURRENT` 发布前同步参数、trainer/RNG、训练与评估日志、实际成本游标；所有 GCS 读路径（`current`/`inspect`/`status`/`pull`/`restore_cell`/`prepare_resume`/`load`/`restore_final` 及 `resources`）均通过 `open_snapshot`/`_reader_pins` 发布的 `readers/<lease>.json` 租约与全量 GC 闭包保留防止读取期间误删，并在指针切换竞争或校验异常时清理租约而不消费过期对象或回落未加锁读取。硬中断恢复时，训练统计截回 `EngineState.training_log_offset`；成本统计接纳游标之后完整的实际事件，重算成本另外记录。最后一条未完成的成本记录可由本地截断或远端恢复放弃（包含相同 `committed_bytes` 前缀上来自 `LOGS` 的 `uncommitted_tail_bytes_at_publish` 合并），其缺失必须一次计入并保留单调 `discarded_incomplete_tail_bytes`、标记 `usage_complete=false`（区别于活跃写入期间的瞬时 `uncommitted_tail_bytes_at_publish`）。完整评测只复用同一最终 model version 的结果。

`checkpoint_interval_rounds`、`checkpoint_interval_groups`、`checkpoint_shard_mib`、`retain_training_artifacts` 与主机内存预留属于资源/持久化选项，不改变训练身份。默认产物保留配置、来源、任务、claim 诊断、成本、官方评估和最终模型；AppWorld 在复制评估结果并关闭该 episode 后回收其独占输出。启动冷恢复与进引擎前开销通过 `CostLedger.credit_startup` 锚定进 `wall_seconds` / `allocated_device_seconds`。终态依次完成 `reclaim_completed_checkpoint`、计量 `publish_final_logs`（写入 `checkpoint.final_publish_logs`）、冻结重写 `finalization_stage="finalized"` 的 `cell_summary.json`，随后执行明确不计入自身成本的终端导出 `export_final_terminal_state` 同步至 `FINAL`/`LOGS` 并写入 `.finalization_complete.json`；任何初步摘要或导出阶段的中断在 `--resume` 时由 `_resume_pending_finalization` 继续完成。恢复所需的当前组、reference、优势、梯度与 cursor 始终随完整快照一起保存。

| 文件 | 负责的契约 |
|---|---|
| `public.py / filtration.py / records.py` | 保留 `<action>` 标签的规范滤子、WebShop 去重头部、ALFWorld `public_object_effects`、三大离散基准精确状态映射与步骤记录 |
| `rollout.py / envs.py` | 独立采样（含 `active-rows-v3` 停放完成槽解码）、ALFWorld 公开对象效应追踪、提前环境派发、轮间同步和持久隔离 lease |
| `policy.py / cache.py` | 比例表征层 $\lfloor 5L/7\rfloor$、无缓存右填充 `attention_mask=None` 零掩码原生因果 SDPA/GQA、表征批处理、采样、缓存和阶段实测 |
| `reference.py` | 混合组零掩码分块参考评分、比例层历史抽取、AppWorld 去重局部观测表示、信用后释放表示 |
| `credit.py` | 同结果组旁路、有界跨深度核（$\epsilon_{\mathrm{rep}}=0.08,\epsilon_{\mathrm{time}}=0.10$）、三大离散基准精确状态匹配、对称 EB 同伴势能（$\lambda_{\max}=1,\kappa=1,\omega\equiv 1$；仅 `exact_first_hit` 保留正同伴门控）及诊断 |
| `optimization.py` | 均衡打乱零掩码混合组微批与统一词元均值分母、解析一阶 head adjoint、响应梯度、逐批范数与首批一致性统计和双槽 AdamW |
| `baseline_specs.py / published_baselines.py / published_batching.py / baselines.py` | 已发表来源配方、信用算法、batch 复制与更新权重、原生 prompt/动作（保持零漂移）与 benchmark 迁移 |
| `protocol.py / engine.py / runner.py` | `fgpo-r8-scale-invariant-v1` 方法身份、宽度感知学习率、正式矩阵、同质轮零步旁路、GCS/本地恢复与生命周期 |

任务结果、阶段成本和运行资源以服务器正式记录为准。
