# F-GPO r3

本分支从 `977bb86` 重构方案与实现。F-GPO 在同任务独立轨迹中，按**相同步深度、相近实际可见历史**共享真实终局成功回报，研究有限时域 agent 的跨轨迹信用分配。方法采用 γ=1、固定回报尺度、整轨迹留一基线和支持质量控制的下一状态续值共享。

当前协议为 **`fgpo-r3-published-v3`**。正式研究包含 24 个已发表方法比较单元与 9 个自身消融单元，总计 33 个训练单元；每个单元附完整官方评测。方案与源码已完成静态修订，服务器训练、显存测量和 benchmark 结果尚未执行。

## 方案、实现与来源

[设计与推导](docs/fgpo_design_and_derivation.md) 说明研究定位、数学对象、实现和最小实验；[数学契约](docs/fgpo_math_contract.md) 定义精确的前缀、留一关系及 PPO 更新；[性能实现说明](docs/fgpo_optimization_proposal.md) 对照性能建议说明具体取舍。

| 路径 | 职责 |
|---|---|
| `src/fgpo/filtration.py`、`records.py` | 整数 token 历史、完整轮窗口、唯一响应归属和真实成本记录 |
| `src/fgpo/credit.py`、`reference.py` | 同深度留一续值、支持收缩、真实 actor-prefix 参考概率与表示 |
| `src/fgpo/baseline_specs.py`、`published_baselines.py`、`published_batching.py` | 已发表 baseline 的固定来源、有效配方、信用规则与作者 batch/损失语义 |
| `src/fgpo/cache.py`、`policy.py` | 几何 KV、有效位置、完整提示批处理、采样与有界表示微批 |
| `src/fgpo/optimization.py` | 响应词元分块、解析词表反向、一次主干反向和 CPU-master AdamW |
| `src/fgpo/envs.py`、`rollout.py`、`baselines.py` | 持久官方环境、组内批量采样及各方法原生观测/动作接口 |
| `src/fgpo/protocol.py`、`engine.py`、`runner.py` | 方法与配方身份、两个实验 suite、恢复和完整官方评测 |

参考模型固定为 cell 初始 Qwen 权重。历史特征取实际 actor prompt 末位的倒数第八层；局部观测特征取当前公开反馈的独立表示。F-GPO 使用最多 32768 tokens 的有界历史，超限时保留初始任务/bootstrap 与最近完整轮次。采样、参考评分和 PPO 使用同一真实前缀；终末反馈保存在轨迹中，参考阶段只计算训练需要的响应概率和动作前特征。

动作前基线的 score 恒等式适用于行为策略采样点；续值共享的偏差由匹配误差和共享权重决定。正式训练使用带 clipping、entropy 和参考正则的 PPO。论文论证围绕这一明确的估计对象、条件性误差分析和官方任务表现展开。

## 必要的正式实验

配置中的 `experiment_suites` 明确区分 `comparison` 与 `ablation`。主表只有完整 F-GPO 和三个已发表 baseline；机制归因使用 F-GPO 自身消融。

| Suite | 方法与范围 | 新训练单元 | 必要性与对应 claim |
|---|---|---:|---|
| `comparison` | `fgpo`、`gigpo`、`hgpo`、`graphgpo` × ALFWorld/AppWorld × seeds 0/1/2 | 24 | 检验完整方法相对已发表状态信用、历史层级信用及图信用方法的最终表现和实际训练成本 |
| `ablation` | `fgpo_obs_only` × ALFWorld × seeds 0/1/2 | 3 | 检验历史匹配项是否改善本方法的留出预测和任务表现；actor 的可见历史保持一致 |
| `ablation` | `fgpo_no_transfer` × ALFWorld × seeds 0/1/2 | 3 | 检验下一状态续值共享的增益；保留原匹配与留一基线 |
| `ablation` | `fgpo_no_support_weight` × ALFWorld × seeds 0/1/2 | 3 | 检验连续支持收缩的作用；有支持时设 λ=1，其余核和基线保持一致 |

消融表直接引用 `comparison` 中 ALFWorld 的三个 F-GPO 单元。短历史实验、额外超参数扫描、模型规模扫描、系统模块消融和独立诊断 rollout 均不进入当前计划。相关工作只承担文献定位作用；它们不会自动扩展运行矩阵。

### 已发表 baseline 的复现范围

| 方法 | 正式出处 | 作者代码配方的主要设置 |
|---|---|---|
| [GiGPO](https://arxiv.org/abs/2505.10978) | NeurIPS 2025 | 150 rounds，γ=.95，global row minibatch 256；发布的 1.5B 全参配方迁移到本研究 7B 基座 |
| [HGPO](https://openreview.net/forum?id=T8Dev99qnz) | ICLR 2026 | 160 rounds，γ=.95，row minibatch 256；不同长度的观测后缀分组与长度加权 |
| [GraphGPO](https://proceedings.mlr.press/v306/cheng26e.html) | ICML 2026 | 150 rounds，γ=.10，7B 配方 row minibatch 64；成功目标最短路径信用与轨迹级优势 |

三者来源统一固定为作者仓库 [verl-agent，提交 `20bd331bdbc9026a5668e11362178e10ab7400c8`](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)。[baseline_specs.py](src/fgpo/baseline_specs.py) 保存论文版本、逐文件来源、参数及论文/代码差异。例如，GiGPO 发布代码的 episode 标准化按重复的 step rows 统计；HGPO 的代码按观测后缀分组、排除零优势后重归一化；GraphGPO 发布代码使用 `10 * gamma**d(next)`。本实现按这些已标明的发布源码配方执行。

ALFWorld 使用作者发布的算法、提示与 batch 配方；HGPO/GraphGPO 来源即 7B 配方，GiGPO 的公开全参训练脚本为 1.5B，本研究明确记录其 **7B backbone 迁移**。AppWorld 另标注为 **已发表方法的 benchmark 迁移**：保留方法的信用规则，以原生 Python 动作、初始任务/API bootstrap 和公开交互记录接入。上述三篇论文未报告 AppWorld 实验。

所有方法共用本仓库的 PyTorch/SDPA、批量采样、CPU-master AdamW 和官方评分后端，保留各自配方中的历史、折扣、生成预算、优势归一化、训练长度和 loss reduction。baseline 的 reduction 保留作者 virtual-rank / microbatch 内的 token mean 及聚合权重，batch copy 与 balance 的处理顺序也属于复现定义。主比较报告完整方法级表现；三个自身消融承担机制归因。最终分数同时附实际轨迹量、环境步、响应词元和设备小时。

发布源码的 batch 准备由 `published_batching.py` 显式重现：GiGPO 在复制、平衡已有记录后计算优势，HGPO/GraphGPO 在优势计算后复制和平衡。每个 microbatch 的 token 系数为 `1 / (virtual_world_size × fixed_accumulation × actual_response_tokens)`，最后不足额 minibatch 仍保留固定 accumulation。复制只改变已采样记录的统计与 loss 权重，真实轨迹、环境交互和采样成本均按原始数据计数。

### 训练量与官方评测

F-GPO 及其消融为 60 rounds ×16 groups ×8 trajectories，每种子 7680 条完整轨迹；GiGPO/GraphGPO 为每种子 19200 条，HGPO 为 20480 条。共同基座是 Qwen2.5-7B-Instruct。训练采样温度为 1，评估为 .4，`top_p=1`。F-GPO 的每步生成上限为 ALFWorld 256、AppWorld 1024；三个 baseline 的 ALFWorld 原配方为 512，AppWorld 迁移为 1024。时域上限分别为 50 和 100。

每个最终模型评测 ALFWorld `valid_unseen` 全部 134 个任务，或 AppWorld `test_normal` 全部 168、`test_challenge` 全部 417 个任务。ALFWorld 报告 SR；AppWorld 按官方完整任务与 scenario 成员关系报告 TGC/SGC。论文同时呈现三个种子的结果、均值和标准差。训练日志中的成功率曲线用于说明训练进展，最终官方测试分数单列。

诊断复用上述训练样本与成本日志：历史匹配对应留一 outcome Brier 和匹配覆盖；共享与收缩对应支持质量、λ、共享比例及回报移动量；效率对应端到端耗时、token/环境步吞吐和显存峰值。Brier 包含终局成败噪声，作为留出结果预测指标；这些统计完成后释放表示张量，后续更新与 checkpoint 只保留所需概率和优势。

## 指定服务器上的运行入口

所有项目执行按 [AGENTS.md](AGENTS.md) 在指定服务器完成。本地只编辑与静态阅读源码。配置为 [Host A](configs/fgpo-r3-host-a.json) 和 [Host B](configs/fgpo-r3-host-b.json)，分别使用 `/apps/users/adminuser/fgpo-r3` 与 `/home/chenxiangwei/workspace/fgpo-r3` 下的独立 checkout，并沿用各自配置中已有的 Python、模型和官方数据路径。

在指定服务器上，将 `PYTHONPATH` 设为该 checkout 的 `src`，使用配置中的 Python。先运行主比较，再用同一 study 根目录运行消融；`<physical_gpu_ids>` 填当次实际分配的物理卡号：

```bash
python -m fgpo.runner --config configs/fgpo-r3-host-a.json --output-dir outputs/fgpo-r3-published-study --suite comparison --gpu <physical_gpu_ids>
python -m fgpo.runner --config configs/fgpo-r3-host-a.json --output-dir outputs/fgpo-r3-published-study --suite ablation --gpu <physical_gpu_ids>
```

`comparison` 为默认 suite。主结果位于 `comparison/`，消融结果位于 `ablation/`，各自保存 `study_summary.json`。消融 summary 引用 `../comparison/alfworld/fgpo/seed-X/cell_summary.json`，复用已有完整 F-GPO 结果。

`--methods`、`--benchmarks`、`--seeds` 只能选择当前 suite 的计划子集，用于分配完整 cell；主比较不接受自身消融 ID。`--resume` 恢复所选 suite 目录中的状态；已有 study 根目录可以新建另一 suite。Host B 使用对应配置。每个进程显式分配不重叠 GPU；`--model-gpu` 从其已分配卡中选择模型计算子集。

`scripts/launch_fgpo_formal.py` 与 `scripts/launch_fgpo_host_b.py` 是公共 runner 的薄入口。新的 v3 study 使用独立输出目录，旧奖励、旧比较协议和旧 pending update 保留在原 study 中。

## 显存、计算与更新实现

参考阶段按真实 actor 段一次计算主干；同段多个响应合并为有界 head chunks，每段统一下载概率。层 hook 只保留动作前 prompt 末位表示，`obs_only` 跳过历史特征，局部观测先去重再按长度和 token budget 分微批。信用计算完成后释放特征引用，减少更新阶段的 CPU 常驻内存和 checkpoint 负担。

采样使用逐层几何 KV 缓冲、真实有效 mask 和逻辑 RoPE 位置。缓存行选择按设备上传一次 indices；已结束响应的 padding 槽位跳过词表投影。不同完整提示可以批量 prefill；所有正式方法都采用组内批量模型采样。环境进程复用解释器，每个 episode 重新建立独立 world；F-GPO 的完成响应可提前派发环境执行。

PPO 只计算当前 minibatch 所拥有响应所需的因果前缀。输出头按响应词元分块立即反向，使用 logprob/entropy 的一阶解析导数，显式 adjoint 缓冲随响应词元数分配，最后通过 gather 对主干执行一次反向。AdamW 的 FP32 master/moments 在 CPU，梯度与权重通过有界 staging 桶传输。解码日志按操作和 role 合并精确计数，减少逐 token 的 JSON 写入；环境和同步阶段事件保留。

资源默认值包括 `head_chunk_tokens=64`、`optimizer_bucket_mib=64`、`cache_initial_capacity=1024`、`representation_token_budget=8192`、`representation_batch_size=8`、`environment_pool_size=8`、`working_memory_gib=6`、`activation_offload=false`。实际峰值由正式运行记录；几何容量、逻辑长度和 allocator reserved bytes 分别报告。

模型导出使用 `checkpoint_shard_mib=1024` 的官方 safetensors 分片路径，不预先复制整个 actor 到 CPU。固定参考评分结束后，从 Adam 已有的 FP32 master 在 CPU 分桶舍入为 BF16 并恢复 actor；尚未更新的参数从不可变初始参考恢复。优化器公式、moments 精度和训练配方保持原样。单个大 tensor 可以超过分片预算，因此该值不等同于进程 RSS 上限。

公共 runner 在加载模型、创建 cell 输出和采样前，以同一 Linux 用户的主机级文件锁登记 CPU 内存预算。预算覆盖全参数 FP32 Adam、固定参考、序列化/恢复临时量、传输桶以及 `host_runtime_reserve_gib=16` 的环境与轨迹余量；所有合作 cell 共用 `host_memory_fraction=0.85` 的主机上限，并检查当前 `MemAvailable`。容量不足时直接报告所需与已预留容量。该预算按模型 header 计算，随进程退出释放；未使用该 runner 的其他程序不参与预留。资源余量可按正式任务日志调整。

## 交付与记录状态

每个 cell 默认保存有效配置与 baseline 来源、官方任务清单、训练 claim 诊断、成本日志、逐任务官方评测及最终模型。当前恢复点仍包含其所需完整轨迹、概率、优势和更新 cursor。`retain_training_artifacts=false` 关闭历史 group、reference-round 和单条轨迹的重复归档；设为 true 时保留训练 group 与每轮参考数据，baseline 的组内轨迹只保存在 group 中。AppWorld 在官方结果复制进轨迹后关闭数据库并删除本 episode 独占的输出目录，共享基础数据保留。`costs.jsonl` 区分 `host_dispatch` 和阶段 `synchronized_wall`；端到端效率使用 `wall_seconds`、`allocated_device_seconds`，有重叠的事件耗时分别解释。

续训身份包括协议、有效方法配方、任务、采样、信用、PPO 和 Adam 语义。默认 `checkpoint_interval_rounds=8`：首组采样前、每 8 轮、训练结束及托管停止时保存完整一致快照。组/reference 边界只在 actor/Adam 版本已保存时复用参数文件保存进度。SIGINT/SIGTERM 在安全边界保存待更新梯度和 cursor；强制终止从最近完整快照恢复，最多重做一个 8 轮间隔内的工作。训练统计回到快照的字节游标，成本日志保留已实际发生的完整事件；损坏的最后半行被移除并将成本完整性标为 false。新快照文件和日志同步后发布 CURRENT，随后回收旧快照。

完整训练与官方评测完成、最终模型和完成摘要持久化后，回收恢复状态；按记录在完成摘要中的保留策略清理本 cell 的历史原始文件。默认 8 轮周期下，Qwen2.5-7B 的主要 checkpoint 累计写入估算从每 cell 约 6.43/16.02/17.09 TB 降至 0.88/2.06/2.16 TB，分别对应 60/150/160 轮。原子替换仍短暂保留新旧恢复点，参数级磁盘峰值约 228.47 GB；需另外容纳当前 trainer state 与日志。详细容量公式与审计范围见 [优化方案](docs/fgpo_optimization_proposal.md)。

**当前结果是可审阅的方案与静态实现。** 当前会话没有仓库要求的 `SSH_Document.md`，已知 `mgt01` 入口在可用连接环境中无法解析，正式服务器运行尚未开始。本分支没有已执行 benchmark、显存节省比例或加速倍数；这些数值由上述必要实验的真实日志填写。

研究记录沿用指定服务器 `.research/research.sqlite` 的单写机制。当前会话没有该数据库的可用连接，未生成或改写实验运行记录。
