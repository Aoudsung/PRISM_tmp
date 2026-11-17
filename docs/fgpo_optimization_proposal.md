# F-GPO r3：性能方案的采纳与实现

本文件记录性能优化建议在 `fgpo-r3-published-v4` 中的实现、数学边界与正式实验口径。算法定义见 [设计与推导](fgpo_design_and_derivation.md)，精确更新关系见 [数学契约](fgpo_math_contract.md)。代码从 `977bb86` 重构并经 v4 修订；本次修改继续保持 F-GPO 的终局成功目标、同深度整轨迹留一匹配、正同伴门控有限支持续值共享和混合组原子轨迹归一化 PPO。

当前完成的是静态实现与源代码审阅。容量公式属于解析估算，吞吐、峰值显存、更新耗时和官方成功率由指定服务器的正式任务运行记录。当前仓库未提供服务器连接说明，本次未执行本地项目代码，也没有服务器性能或 benchmark 结果。

## 目标与测量口径

优化围绕三个实际阶段展开：有效采样、固定参考评分和模型更新。先消除无用投影、重复前向、长期保留的张量与细粒度日志开销，再将可用容量用于完整上下文和有效批量。显存分配量本身不作为效果指标；报告实际任务结果以及取得结果所用的资源。

正式记录包含完整轨迹的数量与官方成功数、环境步、生成和 teacher-forced token、optimizer step、实际墙钟和 GPU 小时，以及各阶段 CUDA allocated/reserved/peak bytes。当前轮的完整轨迹服务于更新和恢复，历史原始记录按显式保留选项存储。ALFWorld 最终模型使用官方 success rate，AppWorld 最终模型使用官方 TGC/SGC。已发表方法保留各自来源配方的训练轮数、prompt、输出和更新预算，终点结果同时标明这些预算。曲线只使用每轮已有行为采样日志中的训练成功率，在共同覆盖的累计轨迹、token 或墙钟区间解释；官方测试仅在最终模型执行，不增加中途评测。

`CausalPolicy.phase` 在阶段边界同步设备，记录 `synchronized_wall` 和峰值。`head/body` 的 `time_basis=host_dispatch` 记录主机提交耗时。嵌套显存范围标记为 `enclosing_phase`；端到端时间使用实际运行墙钟，环境与 GPU 的重叠阶段不重复求和。PPO 前向、反向、AdamW、采样组与参考计算均有对应阶段日志。

## 已实现优化与保持的契约

| 实现位置 | 当前实现 | 保持的数学或资源条件 |
|---|---|---|
| `engine.py / optimization.py` 混合组与零步旁路 | 筛选混合结果组 $1 \le \sum Y_k \le K-1$ 以整组 $K=8$ 原子批与分母 $Z_m=8$ 更新；$M_{\mathrm{mix}}=0$ 整轮跳过参考/PPO/AdamW；记录逐批裁剪前范数与首批 `\|lp - old_lp\|` | 同结果组优势严格恒零；保持 $K=8$ 对称正负组内对比、单 epoch、clip 与正则语义不变 |
| `optimization.py` 因果主干 | 当前段仅前向至本 minibatch 最后一个响应的 `span.stop` | 完整前缀、原始位置和目标 token 不变；参数更新后重新前向 |
| `optimization.py` 隐藏态梯度 | 一次可微 gather 收集响应位置，各词表块立即反向，最后一次 gather/主干反向 | 仅在循环期间保留响应位置的梯度；主干训练图与激活 checkpoint 保留 |
| `optimization.py` 输出统计与设备归集 | FP32 log-prob/熵共用 log-partition 与联合解析 adjoint；PPO 裁剪计数、ordered FP64 差值和与最大值在输出设备按段归集，段边界/cursor 持久化前一次读回；参考与 PPO 段复用 `aggregate_model_calls` 并携带真实轮次与组归属 | PPO、dual clip、k3、熵、损失分母、原子段游标语义及调用/位置/token/时间口径保持不变 |
| `reference.py / engine.py / filtration.py / rollout.py` 生命周期 | 信用与诊断后释放 `z_hist/z_obs`；行为段 `SegmentRecord` 一次冻结共享；轮末写入日志后释放已完成轮的 `groups`/轨迹/参考局部变量；保留当前轮恢复数据直到轮完成 | 待信用计算断点保留特征；完成轮前中断恢复仍持有当前组与优势 |
| `cache.py / policy.py / rollout.py` KV 与无填充解码 | 主机已知 `right_padded` 且全部物理列有效时，cached 单 token 解码传入 `attention_mask=None` 避免冗余掩码与 KV 展开；继续行在 `split_cache`/`select_cache` 保留容量、停放行仅按占用长度分配并裁去尾部全空列；`_sample_turn` 直接移交停放分片避免冗余 merge+split 拷贝；`append_cache` 按预算分块长增量并限制物理 padding；异长/ragged GQA 后端改造（依赖目标 PyTorch/Transformers/CUDA 版本的 varlen/paged 或 masked GQA 支持）保留为后续工作 | 掩码、RoPE、完整前缀、EOS/cap 末词元单次追加、独立 slot generator 与调用顺序不变 |
| `reference.py / policy.py` 参考评分 | 仅对混合组轨迹按同段合并上传目标、分块投影、统一回传分数；块间释放 logits | 各响应实际前缀、目标顺序与冻结参考权重相同 |
| `reference.py / credit.py` 观测与历史特征 | hook 只保留动作前位置；ALFWorld 实施离散纯观测词元元组精确匹配，省去 `z_obs` 前向；AppWorld 维持去重局部观测前向 | 保留指定层与归一化；同结果组直出零优势并免除特征构造；`obs_only` 跳过历史特征 |
| `reference.py / engine.py` 生命周期 | 信用与诊断完成后释放 `z_hist/z_obs`，保留响应分数和已算优势 | 待信用计算的断点仍需特征；进入更新阶段后不再重算信用 |
| `credit.py` 同伴运算 | `[H,K,D]` 批量相似度与 `[H,K,K]` 同深度权重；正同伴门控 $\mathbf 1[N^+_{-k,t+1}\ge 1]$ 与 $\lambda_{\max}=0.5$；及时释放特征临时量 | CPU FP32 距离、FP64 归约与整轨迹排除定义保持相同 |
| `optimization.py` AdamW | 按设备和 dtype 使用最多两个有界传输槽，桶内 FP32 更新、原位缩放与参数写回 | FP32 master/m/v、全局梯度裁剪、每参数步数与断点状态保持相同 |
| `baselines.py / policy.py` 原生采样 | 相同完整 prompt 单次 prefill 后复制缓存；不同 prompt 带显式 mask 批量 prefill | 各槽完整 prompt、独立 generator 与逻辑位置保留 |
| `policy.py` 增量投影 | 只投影本次追加 token 的行，完成槽保留已有分布或及时移除 | 不对 padding 行计算无用词表分布；实际行为 log-prob 对应真实采样温度 |
| `cache.py / policy.py` KV | arena 几何增长，按有效前缀取 view，必要时 compact，crop 保留容量 | padding mask、物理跨度与逻辑 RoPE 位置分别维护；移除历史后重新 prefill |
| `rollout.py / envs.py` 环境 | 持久进程池和独占 lease；F-GPO 每个响应完成后提交 native step | 真实反馈先于下一动作，下一轮保留 barrier，每集重置原生状态 |
| `records.py / rollout.py / baselines.py` 日志 | 按一次采样操作和 role 聚合模型调用，完成或异常时落盘 | calls、token、投影位置及 host dispatch 总量保留；环境和同步阶段独立记录 |
| `engine.py / checkpoint.py` 恢复快照 | 初始、每 8 轮、结束及托管停止保存一致快照；同版本复用硬链接 | 新 actor/Adam 不与旧 trainer 混搭；硬回退同时回退轮统计，实际成本保留 |
| `policy.py / optimization.py` CPU 副本 | 官方逐片模型导出；固定参考从已有 master 分桶恢复；梯度按 storage 写出 | FP32 master/moments 与 CPU BF16 舍入路径保持相同 |
| `engine.py / baselines.py / envs.py` 原始产物 | 默认只留 claim 日志；组内轨迹单一持久化；episode 结束后回收独占目录 | 官方指标先复制到结果记录，当前恢复数据始终存在于完整快照 |
| `resources.py / runner.py` 主机并发 | 加载前按 header 计算 CPU 预算，同一用户通过文件锁共同预留 | 全参数 Adam、参考、恢复梯度、序列化和环境余量提前计入 |

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

源码默认 `head_chunk_tokens=64`；当前 Host B 与 Colab 配置采用已部署的 `256`，Host A 保留 `64`。target IDs、旧策略分数、参考分数和优势一次传到输出设备；每块完成后释放词表图。全部真实目标位置与完整词表、loss 分母、熵/KL、原子混合组批和 optimizer step 数保持相同。分块改变会影响矩阵形状与 BF16 梯度累计边界的舍入，峰值和耗时随正式运行日志记录；本次未生成实测梯度误差或单项加速结论，也不为此增加单独的合成实验矩阵。

## 参考评分与特征生命周期

参考权重仅在存在混合结果组（$M_{\mathrm{mix}} \ge 1$）时集中安装，随后恢复 actor；$M_{\mathrm{mix}}=0$ 时整轮跳过参考评分、PPO 与 AdamW。`prepare_round` 仅对混合组轨迹按实际行为段合并所有响应目标和位置，只上传一次目标并回传一次分数。`sampled_logprobs` 在块内 gather 对应 hidden，投影后立即释放 logits 与选择结果；相邻响应共享块容量，同时保留各自响应边界和分数顺序。

历史表示在模型中间层 hook 内仅选择 `prompt_length-1`，避免保存该层所有 token 的输出。AppWorld 的局部观测表示按整数 token 去重、按长度与 token 预算分微批并在中间层退出；ALFWorld 在信用核中改为纯观测整数元组精确匹配，从而彻底省略局部观测前向。`fgpo_obs_only` 直接跳过历史表示。

信用计算结束并生成正式组诊断后，`ReferenceRound.release_features()` 清除 `z_hist/z_obs`。此后 PPO 只需要冻结响应分数和已经确定的优势，checkpoint 也不必持续携带这些表示。若断点处于信用计算之前，其输入表示仍然保留。引用、释放和恢复均以完整操作边界为单位，不引入额外采样或重复估值。

## CPU AdamW 的内存与传输

FP32 master、`exp_avg` 和 `exp_avg_sq` 常驻普通 CPU 内存。每个 staging slot 有主机梯度桶、主机权重桶、设备梯度桶和设备权重桶。默认每桶 64 MiB，两个 slot 对每个设备、每种 dtype 合计最多约 256 MiB pinned 主机存储与 256 MiB GPU staging；CPU FP32 算术临时量另计。最后一个桶按实际长度使用，参数增大时 staging 上限保持不变。

每个设备先在设备上归约 FP32 参数梯度范数，以 FP64 标量汇总平方和，再传回一个标量。全局范数检查完成后才更新 master、moments、step 或模型权重。巨大 embedding/head 与其他参数一样按桶切片，保留精度规则并避免一次建立完整参数的 GPU FP32 副本。

同一 slot 的 CUDA stream 顺序为：上一桶 H2D 和参数写回、下一桶 D2H、`downloaded` event。CPU 在 event 完成后读取梯度或覆盖 host weight；这个顺序已经覆盖上一桶 H2D 的完成条件。去掉前置重复 host 等待后，CPU 可以继续更新另一已就绪槽。step 末尾仍同步全部 slot stream，随后才能开始下一次 forward、保存权重或 checkpoint。

CPU 梯度桶转换到 FP32 后原位缩放，桶更新结束立即释放转换临时量。CPU 参数直更新路径显式复制梯度，保持原始 `p.grad`。每参数 Adam 步数、bias correction、weight decay 与 FP32 master/m/v 的 checkpoint 格式保持不变。已发表方法可使用相同运行组件，但学习配方按各自来源解析；共享实现本身不要求共享信用定义、损失分母或训练轮数。

固定参考切换通过 `activate(model, optimizer=...)` 借用已存在的 FP32 master 作为 actor 恢复源，不保存第二份完整 BF16 actor。未初始化 Adam 状态的参数还没有更新，恢复源为不可变初始 reference。每个恢复块先在 CPU 转为 actor dtype，再复制到设备，与 `step()` 的 CPU host-weight 舍入位置一致；异常路径恢复已经安装的参数。此路径只适用于 `fixed=True` 的初始参考，不用于 EMA。

`save_model()` 将设备上的 state_dict 别名交给标准 `save_pretrained`，由 Transformers 按 `checkpoint_shard_mib=1024` 先分片，再由 safetensors 处理当前分片。config、generation_config、tokenizer、绑权处理与索引格式继续使用官方实现。单个 tensor 可以独占超出预算的分片，旧版 safetensors 还会同时保留分片 bytes 和当前 tensor 的 CPU 副本。Qwen2.5-7B 的 embedding/head 单个 BF16 tensor 约 1.09 GB，所以不能把 1024 MiB 分片写成严格 RSS 上限。中断时 `gradient_state()` 借用设备上的 detached 梯度，由同步 `torch.save` 逐 storage 序列化；恢复仍按 `map_location="cpu"` 读取梯度，再装回对应设备。

设参数量为 $P$，Adam master/m/v 的常驻量为 $12P$，固定 BF16 reference 为 $2P$。当前模型的这两项合计约 106.62 GB；原先 reference 切换或保存时另外产生的约 15.23 GB 整 actor 副本已移除。恢复未完成 minibatch 时，`torch.load` 仍可能短时持有完整 CPU 梯度字典，主机预算显式覆盖该部分。

公共 runner 在模型导入及 cell 初始化前读取 safetensors header 计算 $P$ 和最大单 tensor，预留

\[
M_{\mathrm{cell}}=14P+\max\{2\max(S,L),G+L\}+4BD+4B+W.
\]

其中 $S$ 是模型分片预算，$L$ 是最大 BF16 tensor，$G$ 是待恢复 gradients.pt 的文件大小（新训练为零），$B$ 是传输桶大小，$D$ 是计算 GPU 数，$W$ 是 `host_runtime_reserve_gib`，默认 16 GiB。该式保留序列化/恢复与环境余量，不改变 optimizer 状态精度。通过 `/tmp/fgpo-host-memory-<uid>` 的文件锁串行登记活动 cell；进程 PID、启动时间和存活状态用于回收退出后残留的预留。多个有效配置共用其中最小的 `host_memory_fraction × MemTotal` 上限，默认 fraction 为 0.85；同时检查 `MemAvailable` 与活动 cell 尚未实际分配的预算。容量不足在采样前报出具体预算，完成或异常退出后释放自己的登记。预算针对同一原生 Linux 主机、同一用户通过公共 runner 启动的合作进程；未参与预留的新外部负载仍需由主机调度控制。环境和轨迹的实际工作集由正式日志确定，必要时增大 $W$。

## 采样、KV 与运行日志

每条轨迹使用独立 generator。原生 baseline collector 将同一决策轮的完整 prompt 批量 prefill，完全相同的 prompt 先复用一次根计算再复制缓存。不同长度 prompt 显式右侧 padding，采用有效 mask 和每行最后一个真实位置获得下一 token 分布；后续追加保持逻辑 RoPE 位置。结束槽及时释放 KV 行。增量 head 只投影有新增 token 的行，未变化行保留已有分布。

Arena 只向注意力提供实际已用前缀的 K/V view，预留容量不增加逻辑长度。主机维护每行有效跨度，压紧不依赖逐行 GPU 形状回传。缓存的选择、复制与消费具有明确所有权；窗口丢弃旧历史后重新 prefill 保留的完整轮次，以保持新的行为条件。

当前共同实现 `common-efficiency-v2` 由 `cache.py`、`policy.py`、`rollout.py`、`resources.py` 四文件整体组成。窗口滚动后的存活兄弟在既有 turn barrier 按层消费式合并完整 K/V，及时释放旧层存储，并保持 slot ID 与原 generator 的对应关系。单响应有界 workspace 保留同次 multinomial 产生的设备整数 token，直接供后续 decode 使用；CPU token/log-prob 接口、EOS 与预算派发顺序不变。输入、位置、mask 与选择索引复用受原有 packed CPU readback 和各设备 current stream 顺序约束，不增加逐 token synchronize、随机操作或精度转换。合并搬运记录在 `rollout.merge_cohorts` phase。

恢复时主机预算一次读取 `checkpoint/CURRENT`，以对应 `snapshots/<snapshot>/model` 的 safetensors header 与同一 snapshot 的 gradients 文件建立 inventory；新 seed 继续读取配置中的基础模型路径。预算公式、模型路径身份、加载及 checkpoint 格式保持相同。公共 runner 同时保留主机预算登记和逐 cell 文件锁；共享 suite 的 `study_summary.json` 在锁内重新读取并合并，支持互不重叠 cell 并行写入。

以 [Qwen2.5-7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)中的 28 层、4 个 KV heads、head dimension 128 和 bf16 估算，单个缓存批的原始存储为

\[
M_{KV}=28\times2\times B\times4\times S\times128\times2\ \mathrm{bytes}.
\]

$B=8,S=4096$ 时约 1.75 GiB，$S=32768$ 时约 14 GiB。模型约 $7.615\times10^9$ 参数对应 bf16 权重约 14.18 GiB。这些估算尚未包含梯度、激活、词表中间量、共享根副本、arena 扩容过程和 allocator reserved；部署容量由实际阶段峰值决定。正式模型逻辑上下文上限仍为 32768，已发表 ALFWorld prompt 上限按各自来源配方设置。

持久环境池为各槽独立租用进程，每集重新初始化原生环境与随机种子，AppWorld 使用独立实验目录。F-GPO 的一个响应完成后立即提交 CPU 环境执行，其他槽继续解码；所有真实反馈收齐后进入下一决策轮。异常路径先收齐已提交操作，再关闭 episode 和 KV。

AppWorld 的输出根从实际 `path_store.experiment_outputs` 获取，每次 reset 独占创建并登记一个 UUID episode 目录。collector 先调用官方 evaluator 并复制结果，再关闭 world/数据库，最后删除该实例拥有的 episode 目录；构造中断、reset 和 worker release 走同一清理路径。共享基础数据、其他 episode 与既有历史运行不做扫描删除。默认完整矩阵将执行 206,700 个 AppWorld episodes，其正常完成的输出随 episode 回收，持续产物保留在 cell 的官方结果与 claim 日志中。

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

三种已发表基线以 [`verl-agent` 固定快照](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)为来源，`src/fgpo/baseline_specs.py` 保存出处、解析参数和论文/代码差异。ALFWorld 保留发布算法及来源配方：GiGPO/HGPO/GraphGPO 分别训练 150/160/150 轮，prompt 上限 2048/4096/2048，响应上限均为 512，保留各自折扣、minibatch、归一化及无效动作处理。GiGPO 的固定来源 full-finetuning 脚本使用 Qwen2.5-1.5B，本研究明确将其配方迁移到统一 7B；HGPO 和 GraphGPO 使用来源 7B 配方。`published_batching.py` 保留 `adjust_batch` 的复制、batch 平衡顺序及 virtual rank/microbatch 内部 token-mean 后再平均的权重；GiGPO 在优势估计之前复制，HGPO/GraphGPO 在优势估计之后复制。F-GPO 为 60 轮、响应上限 ALFWorld 512 与 AppWorld 1024，采用混合结果组原子整组与固定轨迹归一化 $Z_m=8$。所有完整终点按实际训练成本解释。

每个 published virtual microbatch 按真实响应 token 数 $N_m$ 归一化，再乘固定累积数 $A$ 与 virtual rank 数 $W$ 的倒数，得到每 token 系数 $1/(WAN_m)$；末批保留固定 $A$。GiGPO/HGPO/GraphGPO 的 $(W,\text{microbatch rows},A)$ 分别为 $(2,32,4)$、$(4,8,8)$、$(4,4,4)$。`EngineState.published_batch_plan` 保存批计划，`credit_row_ids` 保留 GiGPO 优势统计中的重复次数；`ppo_update(..., published_plan=...)` 在一个 optimizer step 内合并同一响应的正系数，只前向一次，再同时加权 PPO、熵、KL 及相应统计。系数不跨更新边界合并。cursor 记录权重身份，训练 batch 的复制不增加真实 rollout；环境采样量按原始轨迹计数。这些数学参数纳入训练身份，物理 GPU 与 head chunk 等资源参数分别保存。

AppWorld 明确标为本研究的 benchmark 迁移：原生 Python 动作、固定任务/API bootstrap、近期完整消息、响应上限 1024，以及官方时域和 TGC/SGC。这些论文没有提供此处的 AppWorld 结果。运行时使用本仓库 SDPA、缓存与 CPU AdamW 路径，和原作者 vLLM/Ray/FSDP 栈存在实现差异；论文/源码配方中的算法差异与 AppWorld 适配随运行身份记录。

机制诊断只读取正式日志中的支持量、有效同伴数、transfer weight、优势变化、失败轨迹正优势位置和留一 outcome Brier。Brier 衡量含结果噪声的留出预测误差，其解释结合真实成功率与相应自身消融。系统日志描述实际工作量和运行成本；本方案没有独立系统开关矩阵，因而不从这些日志虚构每项优化的单独加速倍数。任何整体效率结论都注明方法、采样量、硬件和运行配方。

最终模型的完整评测包括 ALFWorld `valid_unseen` 134 个任务，以及 AppWorld `test_normal` 168 个任务和 `test_challenge` 417 个任务。两个 AppWorld split 属于同一训练单元的评价，训练单元不重复计数。每个种子报告任务完成数、官方指标与实际成本；训练中的效果曲线直接读取训练日志。

## 配置、恢复与当前交付状态

主配置为 `configs/fgpo-r3-host-a.json` 与 `configs/fgpo-r3-host-b.json`。正式主比较固定为 F-GPO、GiGPO、HGPO、GraphGPO，ALFWorld 与 AppWorld 三种子。配置继承已有模型、Python 和官方数据资产路径，源码目录使用独立的 `fgpo-r3`；协议为 `fgpo-r3-published-v4`。GPU 由正式运行命令显式分配；旧资源快照不作为当前可用资源。CPU 核心数、设备内存与互联条件、软件版本、批量与并发配置随服务器正式运行记录。

当前 v4 执行选择 comparison suite 的 F-GPO：Host B 执行 AppWorld 三种子（新输出根目录），Colab 执行 ALFWorld 三种子（新输出根目录 `fgpo-r3-published-v4-alf`）；baseline 与自身消融源码及规划保留。Host B 保持每 8 轮保存完整快照；Colab 使用 `checkpoint_interval_rounds=60`，两端同样保留首次采样前、训练结束及托管停止的保存语义，其中同质轮（$M_{\mathrm{mix}}=0$）省略冗余 `reference -> update` 检查点写盘。Colab 的 1024 MiB 导出分片、关闭 raw 训练产物与主机内存预算字段沿用 `retained_device_tokens_active_rows_v3` 停放槽解码及既有环境配置。

完整恢复快照由 `checkpoint_interval_rounds=8` 控制：首次采样前、每 8 轮、训练结束以及 SIGINT/SIGTERM 的安全边界保存。`checkpoint_interval_groups=16` 仅在当前 actor/Adam model version 已保存时更新采样进度，reference 完成边界遵守同一条件（零混合组轮 `M_mix=0` 设 `durable=False`）。优雅停止保存已采样组、参考概率、优势、pending gradients、PPO cursor 与 RNG；强制终止恢复最近已发布快照，最多重做一个保存间隔的工作。训练结果按 `training_log_offset` 截断到恢复点，物理成本接纳全部完整实际事件，避免重算工作从成本统计中消失。最后一条半写成本事件被截去并标记 `usage_complete=false`。最终评估只复用同一最终模型版本，训练摘要同时核对完成状态、轮数和 optimizer steps。

`checkpoint.save` 在 CURRENT 原子切换前同步新写文件、所需日志和目录；已持久化的 actor/Adam/reference 硬链接只需同步新目录项。写入失败保留先前有效 CURRENT；一次成功发布后回收旧快照，同一 snapshots 下的自有 UUID 孤儿也由保存路径回收。完整官方工作结束、最终模型与 cell_summary 持久化之后才删除恢复状态。`retain_training_artifacts=false` 默认不生成历史 group、reference-round 与单条轨迹文件；显式 true 时训练 group 为原始轨迹的单一持久化来源。完成回收按 summary 中的策略清理本 cell 这些原始文件，旧完成 summary 缺少保留策略时维持其历史产物。

以 [Qwen2.5-7B 官方权重索引](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/model.safetensors.index.json) 的 BF16 数据量 15,231,233,024 字节计算，$P=7,615,616,512$。忽略 trainer/log/file header、且没有额外中断保存，参数级累计写入量由 $(14R+4)P$ 降为 $(14\lceil R/8\rceil+4)P$：

| 正式训练长度 | 原先累计写入 | 默认 8 轮周期 | 更新后模型/Adam 全量写入次数 |
|---|---:|---:|---:|
| 60 轮 F-GPO / 自身消融 | 6.43 TB | 0.88 TB | 8 |
| 150 轮 GiGPO / GraphGPO | 16.02 TB | 2.06 TB | 19 |
| 160 轮 HGPO | 17.09 TB | 2.16 TB | 20 |

累计写入减少约 86–87%。原子替换仍同时持有新旧 actor 与 Adam，固定 reference 共享硬链接，参数级峰值为 $30P\approx228.47$ GB；磁盘还需容纳 trainer state、日志和部署模型。此处保留精确 FP32 Adam 状态与有效旧恢复点，容量规划不得使用累计写入下降比例推导峰值下降。

恢复锁定 `fgpo-r3-published-v4`、方法、来源配方、模型、任务、种子、历史、输出、奖励、信用、PPO、混合组原子批处理及 Adam 语义。保存周期、分片大小、raw 保留、主机预留、缓存容量、head chunk、参考微批和 staging 桶属于资源设置；`credit_transfer_max`、采样温度、生成上限与原子组更新分母属于训练身份。旧 `fgpo-r3-published-v3`、`fgpo-r3-success-v2` 及更早协议的待执行更新不续入本 study。

本次资源修订的正确性/完整性审计覆盖完整保存与恢复、同版本复用、PPO 梯度游标、训练统计回退与物理成本保留、最终结果复用、完成后文件回收、AppWorld 评分先于删除、参考权重恢复的 dtype 与所有权、分片导出的官方加载格式，以及并发 CPU 预算的登记/释放。审计采用源码逐路径审阅与独立交叉审阅，没有执行本地项目代码、测试或工具链；当前会话缺少 AGENTS.md 要求的 SSH_Document.md，服务器执行结果仍为空。

本次没有运行服务器正式实验，方案中的内存数值仅为公式估算，未将性能建议中的速度倍数或 benchmark 提升写成实测结果。接下来的执行范围就是上述 33 个正式训练单元及其官方评价，数值、成本和机制证据均附着于这些真实任务记录。

实现接口依据：[Transformers v4.57.1 Cache 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)、[Transformers save_pretrained](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py)、[safetensors v0.5.3 torch 保存](https://github.com/safetensors/safetensors/blob/v0.5.3/bindings/python/py_src/safetensors/torch.py)、[PyTorch v2.8 序列化](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/serialization.py)、[PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html)、[PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)、[AppWorld 输出与并行约定](https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md)。服务器正式记录应保留实际安装版本。

## Colab 显式 GCS 恢复存储（2026-11-17）

当前 Colab 资源配置改用 `checkpoint_backend=gcs`；Host B 保留原 local backend。GCS study/cell/seed 独立的 CURRENT 为训练恢复权威，全部对象不可变并记录 generation/size，上传 manifest 后用旧 generation 条件发布 CURRENT，再更新本地小指针并按 CURRENT/FINAL/LOGS 的引用集合回收对象。失败直接终止存储操作，已提交的完整 CURRENT 保留。此资源实现保留 fgpo-adamw-v1、原 model_path、EngineState、梯度、参考、sampler 和所有 RNG 内容，科学定义与正式实验条件保持原值。

现行 GCS 清单与增量日志协议为 `fgpo-gcs-manifest-v2` 与 `fgpo-gcs-journal-v2`（同时兼容读取既有 `fgpo-gcs-manifest-v1` / `fgpo-gcs-logs-v1` 检查点）。静态文件（`config.json`、`official_tasks.json` 与完成态摘要）在未变化时直接复用已有不可变对象；增长中的 `costs.jsonl`、`training.jsonl` 与 `evaluation-<split>.jsonl` 只读取完整换行提交字节，以上传增量段和大小分级合并（保持每条日志在 $O(\log F)$ 个不可变段内，避免平方级重传与清单膨胀）。`costs.jsonl` 跨重启单调保留最长完整流，通过统一 `_select_costs_journal` 在相等提交长度上合并 `CURRENT`/`FINAL`/`LOGS` 的 `uncommitted_tail_bytes_at_publish`（区分瞬时活跃写入与恢复放弃后一次计入的单调 `discarded_incomplete_tail_bytes`，仅后者置 `usage_complete=false`）；`training.jsonl` 在硬回退或恢复时重新以 CURRENT 游标与新 `lineage_id` 锚定避免混入崩溃前未提交尾。所有保留读路径（`current`/`inspect`/`status`/`pull`/`restore_cell`/`prepare_resume`/`load`/`restore_final`/`resources`）通过 `open_snapshot`/`_reader_pins` 先发布并复核指针稳定的线性化读租约，整个读取与异常清理全程覆盖该租约并要求调用者显式传入同一快照内的 LOGS（`logs=None` 表示已确认缺失）；存在活跃读租约时，`publish_logs` 延迟清理旧 LOGS 对象，全量 GC（`save`/`import_local_snapshot`/`reclaim_completed`/`publish_final_artifacts`）保留所有活跃读租约引用闭包直至读取释放。传输统计 `by_category` 分别计入 `actor`、`optimizer`、`gradients`、`reference`、`teacher`、`state`、`artifacts` 与 `journals` 的实际新上传字节与复用字节，并将 manifest/pointer 单独列为 `metadata_uploaded_bytes`。`CostLedger.credit_startup` 将 `runner.cold_asset_restore` 与 `runner.cell_startup` 非重叠区间锚定进 `wall_seconds` 与 `allocated_device_seconds`；完成阶段先回收并记录 `checkpoint.reclaim_completed_resume_state`，再执行并计量 `publish_final_logs` 记录 `checkpoint.final_publish_logs`，据此重写 `finalization_stage="finalized"` 的 `cell_summary.json`，最后执行一次明确不自计入成本的终端导出 `export_final_terminal_state` 将终态摘要与含终态事件的 `costs.jsonl` 同步至 `FINAL`/`LOGS` 并写入 `.finalization_complete.json`（中断恢复由 `_resume_pending_finalization` 继续完成）。`scripts/colab_ops.sh` 通过指定远端运行时执行 `status`/`pull` 并拒绝无用的 `--bucket`/`--prefix` 覆盖，`stop` 强制要求 `--cell`、`--config` 与离机 `--dest-dir`，等待进程与检查点完整退出、校验 GCS/本地状态并将恢复态拉取至离机目录后再销毁实例。

主协调收尾约定：可选远端状态探测通过不创建本地目录的快照读取完成，仅在选中远端状态后创建恢复目标；`restore_artifacts` 要求显式 `logs` 参数且调用者在整个成员读取期间持有选定快照。终端发布先提交包含最终摘要与日志的 `LOGS`，最后提交带 `terminal_export_completed` 的 `FINAL`，完成态静态文件以该 FINAL 为权威。

成本尾部使用写入 session 与已提交字节偏移构成观察身份。恢复消费该观察后，将身份、字节数和累计弃尾计数同时保留在成本所有者的 `.cost-tail-state.json`、ledger state 与远端 journal 元数据；GCS 发布器只读取该本地账本。合并等长前缀时删除已消费的待处理观察，避免 `restore → sync → restore` 重计；后续写入 session 的新尾部具有不同身份。旧无身份的 v2 观察按其单调前缀导入一次，不新增哈希计算。

Colab 的 Adam/reference/gradients/state 直接流式 torch.save 到有界上传器，加载时通过固定 generation 的 range/seek/readinto 读取。当前 actor 的官方分片目录单独在 `<cell>/actor-cache/model` 暂存与恢复；无本地 optimizer.pt 大文件。actor/Adam 同版本复用、同 seed reference 固定、每次 gradients/state 更新。完成官方评价且模型、摘要、数据与成本/claim 日志持久化后，训练 CURRENT 与大态才回收。每个新 seed 从既有 GCS 原始 Qwen 冷资产建立 actor/reference，前一 seed 最终模型只作为远端完成产物保留。原生 CausalPolicy 的 tokenizer/config 来源路径保持原 model_path，小文件在恢复时补齐。

GCS 预算从远端 manifest 的对象长度及真实 actor cache header 读取梯度量 G、参数量 P 与最大 BF16 tensor L，沿用原 `14P + max(2max(S,L),G+L) + 4BD + 4B + W` 并另计两块传输缓冲；序列化临时项同时覆盖一个 FP32 tensor 的 `2L`。8 MiB 缓冲为传输边界，单 tensor CPU staging 与恢复 gradients 仍显式计入主机内存。range reader 的 readinto 直接填充 PyTorch 分配的目标 storage，不建立第二份 optimizer 字典。现有官方部署与后续真实保存/导入/恢复提供执行证据；该源码交付只进行了静态阅读与编辑。

日志成员使用明确白名单，包含 config、official_tasks、训练诊断、完整成本、训练摘要及各官方 split 的 journal/records/summary。训练恢复按保存的 training_log_offset 回退；成本 journal 接纳已实际发生的完整尾事件，半写尾沿用原处理。独立 LOGS 指针保存后续正式工作日志，CURRENT 仍决定训练游标。token 由进程外刷新并通过受限临时文件按请求读取，传输层不接收 ADC 或 refresh token，session URL 不进入记录。完整接口、当前活进程的操作方迁移边界和实际命令见 `docs/fgpo_colab_a100_guide.md`。
