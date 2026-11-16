# F-GPO r3：性能方案的采纳与实现

本文件记录用户提供的性能优化建议在 r3 中的具体落地方式。算法定义见 [设计与推导](fgpo_design_and_derivation.md)，精确更新关系见 [数学契约](fgpo_math_contract.md)。代码从 `977bb86` 重构，采用新协议 `fgpo-r3-success-v2`。这里的容量计算是解析估算；吞吐、峰值显存、更新耗时和成功率均以指定服务器的实际日志为准。

## 优化目标与测量口径

主目标是提高同一硬件上完成有效采样、参考评分和模型更新的效率，并让同一官方任务成功目标得到更有效的信用分配。报告应同时包含：每轮官方成功数与完整轨迹数、实际环境步数、生成与 teacher-forced token 数、训练更新数、各阶段同步墙钟时间、CUDA allocated/reserved 峰值，以及官方 ALFWorld SR、AppWorld TGC/SGC。显存占用率本身不充当性能目标；释放无用张量所获得的容量应用于实际批量和上下文。

`CausalPolicy.phase` 在阶段边界同步设备，并记录各设备峰值。`head`、`body` 内部日志的 `time_basis=host_dispatch` 表示主机提交耗时，不能相加解释为 GPU kernel 时间。嵌套阶段的显存范围标为 `enclosing_phase`，避免把外层累计峰值误读成某个子阶段的独立峰值。PPO 主干前向、反向、AdamW、采样组和参考计算分别记录阶段。跨阶段耗时有包含关系，统计总时间时采用不重叠的阶段。

## 采纳情况

| 建议 | r3 的具体实现 | 数学或资源边界 |
|---|---|---|
| 移除逐 token 的显式 CUDA 同步 | `policy.py` 的 `body/head` 只提交工作；采样将 token 与 log probability 合并回传 | 每个 slot 保留独立 generator 与逐行 multinomial；不承诺不同 kernel/batch 的 bitwise 相同 |
| 预分配并复用 KV | `cache.py` 的 Arena 缓冲几何增长，crop 后保留容量，注意力只读取已占用前缀 | padding mask、逻辑 RoPE 位置和物理列分开维护；采用当前 Transformers Cache/Layer 接口 |
| 取消不必要的缓存压紧 | 已连续且长度合适时直接复用；存在 padding holes 时按主机维护的有效区间压紧 | 不把所有 compact 调用改成空操作；被截掉历史条件化的 KV 也不会冒充新窗口 KV |
| 改进激活保存 | 训练主干保留 checkpoint；默认关闭 CPU activation offload，可显式开启 | 合适的 GPU 留存由实际峰值决定；较小显存部署可以选择 offload |
| 减少词表头重算 | `optimization.py` 对每个 head chunk 立即 backward，累计 hidden 梯度，随后一次主干 backward | 单纯删除 checkpoint 并把所有 chunk loss 留到最后仍会保留全部计算图；r3 改为显式链式法则分解 |
| 聚合梯度范数 | 每个设备先完成 FP32 范数与 FP64 标量平方和归约，再回传一个标量 | 检查完整范数后才修改 master/moments/weights |
| pinned memory 与批量优化器传输 | 每设备、每 dtype 最多两个复用 staging slot；默认每个输入/输出桶 64 MiB | 不将全部 CPU Adam 状态锁页；CUDA events 保护 DMA 期间的主机缓冲 |
| 避免大参数完整 GPU 临时副本 | 巨大 embedding/head 也按桶切片；CPU FP32 更新后在主机转目标 dtype，再分块写回 | 保留 FP32 master、一阶与二阶矩；参数内原位写回 |
| 并行环境与进程复用 | cell 级环境进程池，episode 释放后保留解释器；每个响应完成后立即提交 native step | 同一槽一次只有一个 RPC；下一决策仍等待该步真实反馈；每集 reset 原生状态，AppWorld 使用独立实验目录 |
| 向量化图匹配 | 新信用分配用 `[H,K,D]` 表示和 `[H,K,K]` 同深度相似度 | 复杂度为 O(H K² D)；已删除三遍递归图回传，不能将旧循环的局部倍数作为新算法总加速 |
| 批量观测表示 | 局部观测去重，按长度和 token budget 分微批；通过中间层 hook 提前结束 | 不将所有长观测一次 padding；不修改模型 ModuleList；每批及时释放输出 |
| 删除无用辅助模型工作 | 取消 TF-BEFP、指令重构、反馈 surprise 及 feedback-only scoring segments | 这是明确的 r3 算法重构；官方成功奖励仍保留，不能描述为旧算法无语义变化的加速 |

## 流式词表头反向的精确分解

设某个实际 packed actor segment 的主干输出为 `H=f_θ(x)`，词表投影为 `g_W`，相应损失由响应 token 分块相加：

\[
L=\sum_c L_c(g_W(H[I_c])).
\]

每块使用 `h_c = H[I_c].detach().requires_grad_(True)` 计算损失，立即反向累积词表参数梯度，并把 `h_c.grad` 加到对应 `dL/dH` 位置。完成全部块后调用一次 `H.backward(dL/dH)`。这种实现只让一个词表 chunk 的计算图处于活动状态，同时复用一个完整主干图。若 embedding 与 head 绑权，head 与主干路径的梯度累加到同一个参数。

主干的 activation checkpoint 保持启用；head 的立即反向与主干重算是两个不同层面的内存选择。r3 默认 `head_chunk_tokens=64`，并把 target IDs、旧策略分数、参考分数和优势一次传到输出设备，避免每块重复上传小数组。服务器对照应采用同一批真实保存轨迹、相同权重与目标，比较 loss、梯度和一次更新的数值容差；正常浮点求和顺序变化允许非 bitwise 差异。

## CPU AdamW 的内存与传输

FP32 master、`exp_avg` 和 `exp_avg_sq` 常驻普通 CPU 内存。每个 staging slot 各有一个主机梯度桶、主机权重桶、设备梯度桶和设备权重桶。两个 slot 在默认 64 MiB 桶下，每个设备、每种 dtype 合计最多约 256 MiB pinned 主机存储和 256 MiB GPU staging；实际最后一个桶可能更小。CPU FP32 算术临时量另计。模型大小增长时，staging 上限保持不变。

D2H 完成事件先于 CPU 读梯度；H2D 完成事件先于主机权重桶复用。参数切片位于同一设备 stream，下一 forward、保存模型或保存断点之前保证写回完成。共享系统方法使用完全相同的 AdamW 路径和精度，以免把系统实现差异混入信用分配的比较。

## KV 与上下文

Qwen2.5-7B 的缓存几何按模型官方配置中的 28 层、4 个 KV heads、head dimension 128、bf16 估算。单个缓存批的原始存储是

\[
M_{KV}=28\times 2\times B\times 4\times S\times128\times2\ \mathrm{bytes}.
\]

当 B=8，S=4096 时约 1.75 GiB；S=32768 时约 14 GiB。这还没有计入共享根、两轮历史后缀、扩容时的新旧缓冲、容量余量、压紧临时量和 CUDA allocator reserved。将 7.615×10⁹ 参数按 bf16 计算，权重约 15.23 GB，即 14.18 GiB。附加梯度、激活、词表中间量和优化器 staging 后才能讨论总峰值。

Arena 以逻辑已用长度提供 K/V view，避免注意力扫描预留容量。物理 padding 仍通过 mask 排除；主机同时记录每行有效长度与有效物理区间，压紧不依赖逐行 `torch.where` 的设备形状回传。选择、复制和消费缓存明确所有权。窗口移除旧历史后，对保留的完整轮重新计算其因果表示；不能直接把旧历史条件化的 K/V 裁掉前缀后复用。

官方配置上限仍是 32768 tokens。新窗口保留固定任务/bootstrap 和最近完整轮；75% rollover 目标只是预留空间策略，当完整最新轮需要更多空间时允许使用到实际 prompt limit。终局反馈保存在真实 step record，不再为已结束的 episode 构造额外评分窗口。

## 环境流水

每个 cell 维护固定上限的环境解释器池，默认 8 个，与同任务轨迹数一致。开始一组时为各槽独立租用进程；ALFWorld 每集建立并 seed 官方 game，AppWorld 每集重置原生 world、随机种子和唯一实验目录。释放 episode 关闭 world；cell 结束关闭解释器。

一个槽生成 EOS 或达到响应上限时即可将该响应交给 CPU worker 解析并执行 native step，剩余槽继续 GPU 解码。每步记录仍按稳定 slot 顺序写入 ledger。所有实际反馈收齐后才构造下一轮批量 prompt。异常路径先等已提交 RPC 结束，再释放 episode 与 KV，避免同一 JSON 通道并发 step/close。

## 配置、比较和实验结论

新入口使用 `configs/fgpo-r3-host-a.json` 或 `configs/fgpo-r3-host-b.json`。这些文件继承已知模型、Python 和官方数据资产路径，并将源码目录改为独立的 `fgpo-r3`。具体 GPU 由命令显式分配；旧脚本中的资源快照不作为现在的可用资源依据。

`fgpo`、自身消融、`gigpo_shared`、`bipace_q_shared` 和 `native_qwen_shared` 共享 rollout、上下文上限、256/1024 响应上限与官方时域。可训练主方法与共享基线均为 60 rounds ×16 groups ×8 trajectories，AdamW lr=1e-6、weight decay=.01，PPO 按固定轨迹质量归一化。作者配置的迁移实验使用独立方法 ID，结果单列。

`checkpoint_interval_groups=16` 默认在完整采样轮保存，托管停止仍在操作边界保存；可以根据服务器停机需求调整保存频率。已有断点实现对未变化的模型/Adam 参数文件使用硬链接。续训锁定实际训练语义和任务划分，允许改变缓存、head chunk、pinned staging 等资源设置；v1 状态不能续入 r3。

正式实验直接服务于三个问题：同一环境采样量下的官方成功率是否提高、同一硬件下的成功率—墙钟曲线是否改善、失败轨迹中获得正优势的前缀是否有可用同伴支撑。后者用已有训练日志分析支撑量、有效同伴数、transfer weight 与留一 outcome Brier；Brier 包含结果噪声，不能充当真 Q 误差或保证。系统效应通过相同真实轨迹或相同正式配置的阶段日志比较，不能用旧版与 r3 的算法差异解释纯系统加速。

截至本次代码交付，仅完成静态实现与交叉审查。指定服务器的连接说明未随仓库提供，当前会话没有执行本地项目代码，也没有产生服务器吞吐、显存峰值或 benchmark 结果。原建议中的 2.44×、11.3×、560× 等数字均未作为实测结论写入本方案。

实现接口依据：[Transformers v4.57.1 Cache 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)、[Qwen2.5-7B 官方模型配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)、[PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html)、[PyTorch CUDA 语义](https://docs.pytorch.org/docs/stable/notes/cuda.html)。服务器实测须记录实际安装版本。
