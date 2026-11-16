# F-GPO r3：性能方案的采纳与实现

本文件记录性能优化建议在 `fgpo-r3-published-v3` 中的实现、数学边界与正式实验口径。算法定义见 [设计与推导](fgpo_design_and_derivation.md)，精确更新关系见 [数学契约](fgpo_math_contract.md)。代码从 `977bb86` 重构；本次修改继续保持 F-GPO 的终局成功目标、同深度整轨迹留一匹配、有限支持续值共享和轨迹归一化 PPO。

当前完成的是静态实现与源代码审阅。容量公式属于解析估算，吞吐、峰值显存、更新耗时和官方成功率由指定服务器的正式任务运行记录。当前仓库未提供服务器连接说明，本次未执行本地项目代码，也没有服务器性能或 benchmark 结果。

## 目标与测量口径

优化围绕三个实际阶段展开：有效采样、固定参考评分和模型更新。先消除无用投影、重复前向、长期保留的张量与细粒度日志开销，再将可用容量用于完整上下文和有效批量。显存分配量本身不作为效果指标；报告实际任务结果以及取得结果所用的资源。

正式记录包含完整轨迹与官方成功数、环境步、生成和 teacher-forced token、optimizer step、实际墙钟和 GPU 小时，以及各阶段 CUDA allocated/reserved/peak bytes。ALFWorld 最终模型使用官方 success rate，AppWorld 最终模型使用官方 TGC/SGC。已发表方法保留各自来源配方的训练轮数、prompt、输出和更新预算，终点结果同时标明这些预算。曲线只使用每轮已有行为采样日志中的训练成功率，在共同覆盖的累计轨迹、token 或墙钟区间解释；官方测试仅在最终模型执行，不增加中途评测。

`CausalPolicy.phase` 在阶段边界同步设备，记录 `synchronized_wall` 和峰值。`head/body` 的 `time_basis=host_dispatch` 记录主机提交耗时。嵌套显存范围标记为 `enclosing_phase`；端到端时间使用实际运行墙钟，环境与 GPU 的重叠阶段不重复求和。PPO 前向、反向、AdamW、采样组与参考计算均有对应阶段日志。

## 已实现优化与保持的契约

| 实现位置 | 当前实现 | 保持的数学或资源条件 |
|---|---|---|
| `optimization.py` 因果主干 | 当前段仅前向至本 minibatch 最后一个响应的 `span.stop` | 完整前缀、原始位置和目标 token 不变；参数更新后重新前向 |
| `optimization.py` 隐藏态梯度 | 一次可微 gather 收集响应位置，各词表块立即反向，最后一次 gather/主干反向 | 仅在循环期间保留响应位置的梯度；主干训练图与激活 checkpoint 保留 |
| `optimization.py` 输出统计 | FP32 log-prob/熵共用 log-partition，使用联合解析一阶 adjoint | PPO、dual clip、k3、熵及损失分母均由原标量目标决定 |
| `reference.py / policy.py` 参考评分 | 同段所有响应合并上传目标、分块投影、统一回传分数；块间释放 logits | 各响应实际前缀、目标顺序与冻结参考权重相同 |
| `reference.py / policy.py` 特征 | hook 只保留动作前位置；观测去重并按长度/token 预算微批；纯表示前向在中间层退出 | 保留指定层和归一化；`obs_only` 跳过其未使用的历史特征 |
| `reference.py / engine.py` 生命周期 | 信用与诊断完成后释放 `z_hist/z_obs`，保留响应分数和已算优势 | 待信用计算的断点仍需特征；进入更新阶段后不再重算信用 |
| `credit.py` 同伴运算 | `[H,K,D]` 批量相似度与 `[H,K,K]` 同深度权重；及时释放特征临时量，原位合并距离 | CPU FP32 距离、FP64 归约与整轨迹排除定义保持相同 |
| `optimization.py` AdamW | 按设备和 dtype 使用最多两个有界传输槽，桶内 FP32 更新、原位缩放与参数写回 | FP32 master/m/v、全局梯度裁剪、每参数步数与断点状态保持相同 |
| `baselines.py / policy.py` 原生采样 | 相同完整 prompt 单次 prefill 后复制缓存；不同 prompt 带显式 mask 批量 prefill | 各槽完整 prompt、独立 generator 与逻辑位置保留 |
| `policy.py` 增量投影 | 只投影本次追加 token 的行，完成槽保留已有分布或及时移除 | 不对 padding 行计算无用词表分布；实际行为 log-prob 对应真实采样温度 |
| `cache.py / policy.py` KV | arena 几何增长，按有效前缀取 view，必要时 compact，crop 保留容量 | padding mask、物理跨度与逻辑 RoPE 位置分别维护；移除历史后重新 prefill |
| `rollout.py / envs.py` 环境 | 持久进程池和独占 lease；F-GPO 每个响应完成后提交 native step | 真实反馈先于下一动作，下一轮保留 barrier，每集重置原生状态 |
| `records.py / rollout.py / baselines.py` 日志 | 按一次采样操作和 role 聚合模型调用，完成或异常时落盘 | calls、token、投影位置及 host dispatch 总量保留；环境和同步阶段独立记录 |

主干 activation checkpoint 继续启用。`activation_offload=false` 为默认值；显式开启时使用非锁页 CPU 保存激活。GPU 放置、缓存容量、head chunk、参考微批和 staging 桶按实际资源设置，这些资源选项不改变动作序列所有权、优势公式或更新分母。

## 词表反向与响应梯度

设实际因果主干输出为 $H=f_\theta(x)$，所有本批响应位置为 $I$，一次可微 gather 得到 $Q=H[I]$。词表损失按响应 token 分块：

\[
L=\sum_c L_c(g_W(Q_c)).
\]

实现将每个 $Q_c$ detach 成叶子，即刻执行该块 backward，直接累积输出头参数梯度，并将叶子梯度写入对应的响应梯度缓冲。完成全部块后执行一次 `Q.backward(dL/dQ)`，由 gather 的反向把梯度散射到主干原始位置。绑权 embedding/head 的贡献累积到同一参数；head 与主干位于不同设备时，hidden 的复制保留 autograd 关系。

设本段实际输入长度为 $S$、本批响应 token 总数为 $N$、隐藏维度为 $D$。循环期间显式隐藏态梯度缓冲由 $S\times D$ 改为 $N\times D$；主干反向时 scatter 的输出和训练所需激活仍计入实际峰值。每个段的当前前向只包含本批最后一个目标之前的完整因果输入。跨参数版本复用激活会改变更新，因此每个发生参数更新的 minibatch 仍重新前向。

输出头返回 FP32 logits $z$。令 $a=\operatorname{LSE}(z)$、$p=\exp(z-a)$，响应 log-prob 与熵为

\[
\ell=z_y-a,\qquad \mathcal H=a-\sum_v p_vz_v.
\]

`_LogProbabilityEntropy` 前向保存 logits 和行级统计，反向一次重建概率。对归一化损失的上游系数 $g_\ell=\partial L/\partial\ell$、$g_H=\partial L/\partial\mathcal H$，

\[
\frac{\partial L}{\partial z}
=g_\ell\bigl(\operatorname{onehot}(y)-p\bigr)
-g_Hp(\log p+\mathcal H).
\]

PPO/k3 小图负责裁剪分支和 $g_\ell$；F-GPO 熵项提供 $g_H=-\mathrm{entropy\_coef}/Z$，已发表基线采用其 virtual rank/microbatch 的逐响应损失系数。反向只原位修改新建临时量，保存的 logits 不变；熵系数为零时，临时 log-prob 张量直接原位指数化。该接口使用 `once_differentiable`，明确支持一阶训练。它去掉长期保留的通用 softmax 损失图，不改变定义中的完整词表归一化。

默认 `head_chunk_tokens=64`。target IDs、旧策略分数、参考分数和优势一次传到输出设备；每块完成后释放词表图。chunk 改变及解析反向会改变浮点运算顺序，数值容差需随服务器正式真实记录核对并记录；本次未生成实测梯度误差或加速结论，也不为此增加单独的合成实验矩阵。

## 参考评分与特征生命周期

参考权重在整轮评分期间集中安装，随后恢复 actor。`prepare_round` 对每个实际行为段合并所有响应目标和位置，只上传一次目标并回传一次分数。`sampled_logprobs` 在块内 gather 对应 hidden，投影后立即释放 logits 与选择结果；相邻响应共享块容量，同时保留各自响应边界和分数顺序。

历史表示在模型中间层 hook 内仅选择 `prompt_length-1`，避免保存该层所有 token 的输出。观测表示按整数 token 去重、按长度与 token 预算分微批；仅需要中间层表示时提前结束后续层计算。观测和历史特征的读取对象、层号、归一化与 stop-gradient 保持数学契约中的定义。`fgpo_obs_only` 直接跳过历史表示。

信用计算结束并生成正式组诊断后，`ReferenceRound.release_features()` 清除 `z_hist/z_obs`。此后 PPO 只需要冻结响应分数和已经确定的优势，checkpoint 也不必持续携带这些表示。若断点处于信用计算之前，其输入表示仍然保留。引用、释放和恢复均以完整操作边界为单位，不引入额外采样或重复估值。

## CPU AdamW 的内存与传输

FP32 master、`exp_avg` 和 `exp_avg_sq` 常驻普通 CPU 内存。每个 staging slot 有主机梯度桶、主机权重桶、设备梯度桶和设备权重桶。默认每桶 64 MiB，两个 slot 对每个设备、每种 dtype 合计最多约 256 MiB pinned 主机存储与 256 MiB GPU staging；CPU FP32 算术临时量另计。最后一个桶按实际长度使用，参数增大时 staging 上限保持不变。

每个设备先在设备上归约 FP32 参数梯度范数，以 FP64 标量汇总平方和，再传回一个标量。全局范数检查完成后才更新 master、moments、step 或模型权重。巨大 embedding/head 与其他参数一样按桶切片，保留精度规则并避免一次建立完整参数的 GPU FP32 副本。

同一 slot 的 CUDA stream 顺序为：上一桶 H2D 和参数写回、下一桶 D2H、`downloaded` event。CPU 在 event 完成后读取梯度或覆盖 host weight；这个顺序已经覆盖上一桶 H2D 的完成条件。去掉前置重复 host 等待后，CPU 可以继续更新另一已就绪槽。step 末尾仍同步全部 slot stream，随后才能开始下一次 forward、保存权重或 checkpoint。

CPU 梯度桶转换到 FP32 后原位缩放，桶更新结束立即释放转换临时量。CPU 参数直更新路径显式复制梯度，保持原始 `p.grad`。每参数 Adam 步数、bias correction、weight decay 与 FP32 master/m/v 的 checkpoint 格式保持不变。已发表方法可使用相同运行组件，但学习配方按各自来源解析；共享实现本身不要求共享信用定义、损失分母或训练轮数。

## 采样、KV 与运行日志

每条轨迹使用独立 generator。原生 baseline collector 将同一决策轮的完整 prompt 批量 prefill，完全相同的 prompt 先复用一次根计算再复制缓存。不同长度 prompt 显式右侧 padding，采用有效 mask 和每行最后一个真实位置获得下一 token 分布；后续追加保持逻辑 RoPE 位置。结束槽及时释放 KV 行。增量 head 只投影有新增 token 的行，未变化行保留已有分布。

Arena 只向注意力提供实际已用前缀的 K/V view，预留容量不增加逻辑长度。主机维护每行有效跨度，压紧不依赖逐行 GPU 形状回传。缓存的选择、复制与消费具有明确所有权；窗口丢弃旧历史后重新 prefill 保留的完整轮次，以保持新的行为条件。

以 [Qwen2.5-7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)中的 28 层、4 个 KV heads、head dimension 128 和 bf16 估算，单个缓存批的原始存储为

\[
M_{KV}=28\times2\times B\times4\times S\times128\times2\ \mathrm{bytes}.
\]

$B=8,S=4096$ 时约 1.75 GiB，$S=32768$ 时约 14 GiB。模型约 $7.615\times10^9$ 参数对应 bf16 权重约 14.18 GiB。这些估算尚未包含梯度、激活、词表中间量、共享根副本、arena 扩容过程和 allocator reserved；部署容量由实际阶段峰值决定。正式模型逻辑上下文上限仍为 32768，已发表 ALFWorld prompt 上限按各自来源配方设置。

持久环境池为各槽独立租用进程，每集重新初始化原生环境与随机种子，AppWorld 使用独立实验目录。F-GPO 的一个响应完成后立即提交 CPU 环境执行，其他槽继续解码；所有真实反馈收齐后进入下一决策轮。异常路径先收齐已提交操作，再关闭 episode 和 KV。

`aggregate_model_calls` 在操作内按 role 聚合反复模型调用。聚合保留实际工作量、calls、padded compute token、词表投影位置和 host dispatch 总量，并在正常结束、异常及需要独立事件的边界刷新。环境转移和同步阶段继续单独记录，供官方效果与成本 claim 使用。由此减少逐 token JSONL 写入和 Python 记录分配，同时保持本次论文所需的计量字段。

## 正式比较、必要性与执行范围

主配置为 `configs/fgpo-r3-host-a.json` 与 `configs/fgpo-r3-host-b.json`。正式主比较固定为 F-GPO、[GiGPO（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)、[HGPO（ICLR 2026）](https://openreview.net/forum?id=T8Dev99qnz)和 [GraphGPO（ICML 2026）](https://proceedings.mlr.press/v306/cheng26e.html)。四方法分别在 ALFWorld/AppWorld 使用种子 0、1、2，共 24 个训练单元。三项自身消融只在 ALFWorld 使用三个种子，新增 9 个单元；F-GPO 完整方法直接复用主比较，总计 33 个单元。

| 正式内容 | 必须回答的 claim | 新增运行与必要性 |
|---|---|---|
| 四方法主比较 | 相对于已发表的状态分组、层级历史分组和图路径信用方法，F-GPO 的官方任务效果与成本如何 | 24 个单元；直接覆盖起点工作及两个 2026 机制相关工作，两个 benchmark 覆盖文本行动与原生代码交互 |
| `fgpo_obs_only` | 历史条件匹配是否贡献于方法效果 | ALFWorld 3 个单元；保持 actor 历史与其他机制相同，仅移除距离中的历史项 |
| `fgpo_no_transfer` | 同伴续值共享是否贡献于方法效果 | ALFWorld 3 个单元；保留动作前基线与核匹配，仅令共享权重为零 |
| `fgpo_no_support_weight` | 支持收缩是否改善共享更新 | ALFWorld 3 个单元；有支持时共享权重固定为 1，其余定义保持相同 |
| 机制与系统日志分析 | 支持、信用改变和实际资源成本是否符合相应机制解释 | 0 个新增单元；直接复用上述正式运行的轨迹、组诊断、阶段与成本日志 |

ALFWorld 消融回答三项核心机制贡献，AppWorld 主比较覆盖迁移效果；同一消融不再展开第二个 benchmark 的重复矩阵。主比较表只包含四个完整方法，三个自身消融在机制表呈现。短两轮历史、shared 变体、冻结 native 模型、BiPACE、SELF 和 Skill-DisCo 不进入本次正式矩阵，不安排参数 sweep、系统配置网格、额外反馈 rollout 或独立诊断实验。

三种已发表基线以 [`verl-agent` 固定快照](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)为来源，`src/fgpo/baseline_specs.py` 保存出处、解析参数和论文/代码差异。ALFWorld 保留发布算法及来源配方：GiGPO/HGPO/GraphGPO 分别训练 150/160/150 轮，prompt 上限 2048/4096/2048，响应上限均为 512，保留各自折扣、minibatch、归一化及无效动作处理。GiGPO 的固定来源 full-finetuning 脚本使用 Qwen2.5-1.5B，本研究明确将其配方迁移到统一 7B；HGPO 和 GraphGPO 使用来源 7B 配方。`published_batching.py` 保留 `adjust_batch` 的复制、batch 平衡顺序及 virtual rank/microbatch 内部 token-mean 后再平均的权重；GiGPO 在优势估计之前复制，HGPO/GraphGPO 在优势估计之后复制。F-GPO 为 60 轮、响应上限 256，采用本方案的轨迹归一化。所有完整终点按实际训练成本解释。

每个 published virtual microbatch 按真实响应 token 数 $N_m$ 归一化，再乘固定累积数 $A$ 与 virtual rank 数 $W$ 的倒数，得到每 token 系数 $1/(WAN_m)$；末批保留固定 $A$。GiGPO/HGPO/GraphGPO 的 $(W,\text{microbatch rows},A)$ 分别为 $(2,32,4)$、$(4,8,8)$、$(4,4,4)$。`EngineState.published_batch_plan` 保存批计划，`credit_row_ids` 保留 GiGPO 优势统计中的重复次数；`ppo_update(..., published_plan=...)` 在一个 optimizer step 内合并同一响应的正系数，只前向一次，再同时加权 PPO、熵、KL 及相应统计。系数不跨更新边界合并。cursor 记录权重身份，训练 batch 的复制不增加真实 rollout；环境采样量按原始轨迹计数。这些数学参数纳入训练身份，物理 GPU 与 head chunk 等资源参数分别保存。

AppWorld 明确标为本研究的 benchmark 迁移：原生 Python 动作、固定任务/API bootstrap、近期完整消息、响应上限 1024，以及官方时域和 TGC/SGC。这些论文没有提供此处的 AppWorld 结果。运行时使用本仓库 SDPA、缓存与 CPU AdamW 路径，和原作者 vLLM/Ray/FSDP 栈存在实现差异；论文/源码配方中的算法差异与 AppWorld 适配随运行身份记录。

机制诊断只读取正式日志中的支持量、有效同伴数、transfer weight、优势变化、失败轨迹正优势位置和留一 outcome Brier。Brier 衡量含结果噪声的留出预测误差，其解释结合真实成功率与相应自身消融。系统日志描述实际工作量和运行成本；本方案没有独立系统开关矩阵，因而不从这些日志虚构每项优化的单独加速倍数。任何整体效率结论都注明方法、采样量、硬件和运行配方。

最终模型的完整评测包括 ALFWorld `valid_unseen` 134 个任务，以及 AppWorld `test_normal` 168 个任务和 `test_challenge` 417 个任务。两个 AppWorld split 属于同一训练单元的评价，训练单元不重复计数。每个种子报告任务完成数、官方指标与实际成本；训练中的效果曲线直接读取训练日志。

## 配置、恢复与当前交付状态

主配置继承已有模型、Python 和官方数据资产路径，源码目录使用独立的 `fgpo-r3`。GPU 由正式运行命令显式分配；旧资源快照不作为当前可用资源。CPU 核心数、设备内存与互联条件、软件版本、批量与并发配置随服务器正式运行记录。

`checkpoint_interval_groups=16` 默认在完整采样轮保存，托管停止仍在操作边界保存；保存频率可按服务器停机需求设置。未变化的模型/Adam 参数文件继续使用硬链接。恢复锁定 `fgpo-r3-published-v3`、方法、来源配方、模型、任务、种子、历史、输出、奖励、信用、PPO 和 Adam 语义。缓存容量、head chunk、参考微批和 staging 桶属于资源设置；响应行 minibatch、采样温度和更新分母属于训练身份。旧 `fgpo-r3-success-v2` 及更早协议的待执行更新不续入本 study。

本次没有运行服务器正式实验，方案中的内存数值仅为公式估算，未将性能建议中的速度倍数或 benchmark 提升写成实测结果。接下来的执行范围就是上述 33 个正式训练单元及其官方评价，数值、成本和机制证据均附着于这些真实任务记录。

实现接口依据：[Transformers v4.57.1 Cache 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)、[PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html)、[PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)。服务器正式记录应保留实际安装版本。
