# F-GPO r7 (`fgpo-r7-token-mean-v1`)

协作模型的执行平台、session ID、PI 续接命令与退役记录见 [协作会话登记](docs/COLLABORATOR_SESSIONS.md)。

本分支在 `V4`（`fgpo-r3-published-v4`）的基础上，经 `r5`（对称势能、未决同伴缩放 $\kappa$、按词元均值化熵）、`r6`（规范充分 POMDP 滤子 $\Pi(\mathcal{H}_t)$、TextCraft 两条公开公理与 `H=60/70` 双评估）演化至 **`r7`（`fgpo-r7-token-mean-v1`）**。F-GPO 在同任务独立轨迹中，按**相同步深度、相近实际可见历史**共享真实终局成功回报，使用 $\gamma=1$、固定回报尺度 $R=10$、整轨迹留一基线、正同伴对称有限支持续值共享与均衡打乱的词元均值微批 PPO（`loss_normalization="token"`, `entropy_normalization="token"`, `ppo_epochs=1`, `ppo_minibatch_rows=256`, `lr=1e-6`, `credit_prior_weight=1.0`），并配合 8 线程并发 GCS 分片 `compose` 检查点（Colab `checkpoint_interval_rounds=20`，零本地检查点下行流量）。

当前主推协议为 **`fgpo-r7-token-mean-v1`**。核心配置与运行入口见 `configs/fgpo-r7-1p5b-*.json` 及 `scripts/{colab_run_r7_1p5b_queue,run_fgpo_host_r7_1p5b_queue,colab_gcs_keepalive,compare_r7_vs_r6_r5_v4}.{sh,py}`。

## 方案、实现与来源

[设计与推导](docs/fgpo_design_and_derivation.md) 说明研究定位、数学对象、实现和最小实验；[数学契约](docs/fgpo_math_contract.md) 定义精确的前缀、留一关系及 PPO 更新；[性能实现说明](docs/fgpo_optimization_proposal.md) 对照性能建议说明具体取舍。

| 路径 | 职责 |
|---|---|
| `src/fgpo/filtration.py`、`public.py`、`records.py` | 规范充分 POMDP 滤子、整数 token 历史、三块公开观测偏移、唯一响应归属与真实成本记录 |
| `src/fgpo/credit.py`、`reference.py`、`textcraft.py` | 对称动作前/后势能、$\kappa$ 平滑未决同伴权重、TextCraft 公开规则与 `H=60/70` 双时域评估、混合组筛选后参考评分 |
| `src/fgpo/baseline_specs.py`、`published_baselines.py`、`published_batching.py` | 已发表 baseline 的固定来源、有效配方、信用规则与作者 batch/损失语义 |
| `src/fgpo/cache.py`、`policy.py` | 几何 KV、停放完成槽解码、有效位置、完整提示批处理、采样与有界表示微批 |
| `src/fgpo/optimization.py` | 均衡打乱 `mixed_token_mean`（及统一 `step_token`）微批、解析词表反向、一次主干反向与 CPU-master AdamW |
| `src/fgpo/envs.py`、`rollout.py`、`baselines.py` | 持久官方环境、组内批量采样、ALFWorld `<think>/<action>` 提取及各方法原生观测/动作接口 |
| `src/fgpo/checkpoint_gcs.py` | 8 并发 GCS 分片上传与 `compose` 合并、大盘直存 `.gcs-staging` 落盘暂存及增量日志流同步 |
| `src/fgpo/protocol.py`、`engine.py`、`runner.py` | `r7` 协议与配方身份、同质轮零步旁路、恢复和完整官方评测 |

参考模型固定为 cell 初始 Qwen 权重。历史特征取实际 actor prompt 末位的倒数第八层；局部观测特征取当前公开反馈的独立表示。F-GPO 使用有界历史，超限时保留初始任务/bootstrap 与最近完整轮次；采样、参考评分和 PPO 使用同一真实前缀。

动作前基线的 score 恒等式适用于行为策略采样点；续值共享的偏差由匹配误差和共享权重决定。正式训练使用带 clipping、token-mean 条件熵与截断 k3 参考正则的 PPO。

## 正式实验

TextCraft 在 Host A 和 Host B 评测完整 79 道验证题，报告总体、配方深度分组成功率和实际成本。Host B 在 WebShop 使用同种子的 128 道官方题，报告满分成功率、平均 Score 和实际成本。已取得的家务基准官方结果继续用于论文主表与训练日志诊断。[TextCraft 排程](docs/fgpo_v4_textcraft_execution_plan.md) · [WebShop 排程](docs/fgpo_webshop_host_b_schedule.md)

### 已发表 baseline 的复现范围

| 方法 | 正式出处 | 作者代码配方的主要设置 |
|---|---|---|
| [GiGPO](https://arxiv.org/abs/2505.10978) | NeurIPS 2025 | 150 rounds，γ=.95，global row minibatch 256；发布的 1.5B 全参配方迁移到本研究 7B 基座 |
| [HGPO](https://openreview.net/forum?id=T8Dev99qnz) | ICLR 2026 | 160 rounds，γ=.95，row minibatch 256；不同长度的观测后缀分组与长度加权 |
| [GraphGPO](https://proceedings.mlr.press/v306/cheng26e.html) | ICML 2026 | 150 rounds，γ=.10，7B 配方 row minibatch 64；成功目标最短路径信用与轨迹级优势 |

三者来源统一固定为作者仓库 [verl-agent，提交 `20bd331bdbc9026a5668e11362178e10ab7400c8`](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)。[baseline_specs.py](src/fgpo/baseline_specs.py) 保存论文版本、逐文件来源、参数及论文/代码差异。例如，GiGPO 发布代码的 episode 标准化按重复的 step rows 统计；HGPO 的代码按观测后缀分组、排除零优势后重归一化；GraphGPO 发布代码使用 `10 * gamma**d(next)`。本实现按这些已标明的发布源码配方执行。

ALFWorld 使用作者发布的算法、提示与 batch 配方；HGPO/GraphGPO 来源即 7B 配方，GiGPO 的公开全参训练脚本为 1.5B，本研究明确记录其 **7B backbone 迁移**。

所有方法共用本仓库的 PyTorch/SDPA、批量采样、CPU-master AdamW 和官方评分后端，保留各自配方中的历史、折扣、生成预算、优势归一化、训练长度和 loss reduction。baseline 的 reduction 保留作者 virtual-rank / microbatch 内的 token mean 及聚合权重，batch copy 与 balance 的处理顺序也属于复现定义。主比较报告完整方法及实际成本。最终分数同时附实际轨迹量、环境步、响应词元和设备小时。

发布源码的 batch 准备由 `published_batching.py` 显式重现：GiGPO 在复制、平衡已有记录后计算优势，HGPO/GraphGPO 在优势计算后复制和平衡。每个 microbatch 的 token 系数为 `1 / (virtual_world_size × fixed_accumulation × actual_response_tokens)`，最后不足额 minibatch 仍保留固定 accumulation。复制只改变已采样记录的统计与 loss 权重，真实轨迹、环境交互和采样成本均按原始数据计数。

### 训练量与官方评测

F-GPO 每轮采样 16 组、每组 8 条轨迹。家务基准已有 60 轮结果，官方评测包括完整 134 道未见任务。WebShop 使用配对的 60 轮与 150 轮检查点。TextCraft 的训练预算为 50 轮，每步最多 128 个生成 token。各基准的默认、联合参数（joint）和 radius 配置见 [F-GPO 分基准超参数](docs/fgpo_hyperparameters.md)。

ALFWorld 评测摘要同时在 `task_types` 中按官方任务前缀输出 `Pick / Look / Clean / Heat / Cool / Pick2 / All` 七列百分比成功率，`All` 与原总体 SR 相同。`aggregate_cells` 给出各列实际种子的 `mean`、`sample_std`（ddof=1）、`seeds` 与 `seed_count`；未齐三个种子时如实标记，单种子的标准差为 null。正式 study 摘要包含 `alfworld_table1` 并写出 `study_table1.md`。已运行进程的既有官方评测可用同一 runner 的 `--aggregate-only` 从落盘记录生成 `reporting_summary.json`、`study_reporting_summary.json` 与 `study_reporting_table1.md`；该入口直接汇总，不加载模型、重跑训练或评测，也不改终态检查点与成本。

诊断复用上述训练样本与成本日志：历史匹配对应留一 outcome Brier 和匹配覆盖；共享与收缩对应支持质量、λ、共享比例及回报移动量；效率对应端到端耗时、token/环境步吞吐和显存峰值。Brier 包含终局成败噪声，作为留出结果预测指标；这些统计完成后释放表示张量，后续更新与 checkpoint 只保留所需概率和优势。

## 指定服务器上的运行入口

所有训练和评估在指定服务器执行，沿用各配置中已有的 Python、模型和官方数据路径。现行命令和顺序由[当前实验排程](docs/fgpo_experiment_schedule_current.md)指向两台主机的具体安排。已保存的运行可通过通用 runner 的 resume 参数从同协议快照继续；实验入口保留 comparison 和 ablation 的独立目录，以及对已有正式结果的汇总能力。

## 显存、计算与更新实现

参考阶段仅在存在混合结果组时激活，按真实 actor 段一次计算混合组轨迹的主干；同段多个响应合并为有界 head chunks，段级复用 `aggregate_model_calls` 并携带真实轮次与组归属，每段统一下载概率。层 hook 只保留动作前 prompt 末位表示，`obs_only` 跳过历史特征；WebShop 局部观测先去重再按长度和 token budget 分微批，ALFWorld 直接使用纯观测整数元组匹配从而省略 `z_obs` 前向。信用计算完成后释放特征引用，行为段 `SegmentRecord` 一次冻结共享并在轮末日志落盘后释放已完成轮的轨迹引用，减少更新阶段的 CPU 常驻内存和 checkpoint 负担。

采样使用逐层几何 KV 缓冲、`active-rows-v3` 停放完成槽解码、真实有效 mask 和逻辑 RoPE 位置；当主机已知 `right_padded` 且全部物理列有效时，cached 单 token 解码传入 `attention_mask=None` 避免冗余掩码与 GQA 展开，继续行保留扩容余量并裁去退出后尾部空列，`_sample_turn` 直接向轮转步移交停放分片避免无用合批拷贝，并限制增量重放中的物理 padding 与长窗口查询块峰值（异长/ragged GQA 后端改造保留为后续工作）。缓存行选择按设备上传一次 indices；已结束响应的 padding 槽位跳过词表投影。不同完整提示可以批量 prefill；所有正式方法都采用组内批量模型采样。环境进程复用解释器，每个 episode 重新建立独立 world；F-GPO 的完成响应可提前派发环境执行。

PPO 在全同结果轮（`M_mix=0`）直接零步旁路；存在混合结果组时在 `r7` 将全部混合组响应步打乱并均衡切分为 `ppo_minibatch_rows=256`、`ppo_epochs=1` 的词元均值微批（`mixed_token_mean`，策略、k3 KL 与条件熵统一按 `1/N_tok_m` 归一化），只计算当前组内段所拥有响应所需的因果前缀。输出头按响应词元分块立即反向，使用 logprob/entropy 的一阶解析导数，显式 adjoint 缓冲随响应词元数分配，最后通过 gather 对主干执行一次反向，裁剪计数、有序 FP64 差值和与最大值在输出设备按段归集并于段边界读回，同时记录逐批裁剪前范数与首批次同权重概率一致性误差。AdamW 的 FP32 master/moments 在 CPU，梯度与权重通过有界 staging 桶传输；Colab GCS 断点支持 8 并发分段与服务端 `compose` 合并。

资源默认值包括 `head_chunk_tokens=64`、`optimizer_bucket_mib=64`、`cache_initial_capacity=1024`、`representation_token_budget=8192`、`representation_batch_size=8`、`environment_pool_size=8`、`working_memory_gib=6`、`activation_offload=false`。实际峰值由正式运行记录；几何容量、逻辑长度和 allocator reserved bytes 分别报告。

模型导出使用 `checkpoint_shard_mib=1024` 的官方 safetensors 分片路径，不预先复制整个 actor 到 CPU。固定参考评分结束后，从 Adam 已有的 FP32 master 在 CPU 分桶舍入为 BF16 并恢复 actor；尚未更新的参数从不可变初始参考恢复。优化器公式、moments 精度和训练配方保持原样。单个大 tensor 可以超过分片预算，因此该值不等同于进程 RSS 上限。

公共 runner 在加载模型、创建 cell 输出和采样前，以同一 Linux 用户的主机级文件锁登记 CPU 内存预算。预算覆盖全参数 FP32 Adam、固定参考、序列化/恢复临时量、传输桶以及 `host_runtime_reserve_gib=16` 的环境与轨迹余量；所有合作 cell 共用 `host_memory_fraction=0.85` 的主机上限，并检查当前 `MemAvailable`。容量不足时直接报告所需与已预留容量。该预算按模型 header 计算，随进程退出释放；未使用该 runner 的其他程序不参与预留。资源余量可按正式任务日志调整。

## 交付与记录状态

每个 cell 默认保存有效配置与 baseline 来源、官方任务清单、训练 claim 诊断、成本日志、逐任务官方评测及最终模型。当前恢复点仍包含其所需完整轨迹、概率、优势和更新 cursor。`retain_training_artifacts=false` 关闭历史 group、reference-round 和单条轨迹的重复归档；设为 true 时保留训练 group 与每轮参考数据，baseline 的组内轨迹只保存在 group 中。`costs.jsonl` 区分 `host_dispatch` 和阶段 `synchronized_wall`；端到端效率使用 `wall_seconds`、`allocated_device_seconds`（由 `CostLedger.credit_startup` 锚定进启动冷资产与进引擎前区间，不与相位叠加混淆）。

续训身份包括协议、有效方法配方、任务、采样、信用、PPO 和 Adam 语义。默认 `checkpoint_interval_rounds=8`：首组采样前、每 8 轮、训练结束及托管停止时保存完整一致快照。组/reference 边界只在 actor/Adam 版本已保存时复用参数文件保存进度。SIGINT/SIGTERM 在安全边界保存待更新梯度和 cursor；强制终止从最近完整快照恢复，最多重做一个 8 轮间隔内的工作。训练统计回到快照的字节游标，成本日志保留已实际发生的完整事件；损坏或恢复放弃（包括相同已提交前缀上来自 `LOGS` 的未提交尾）的半写尾一次计入单调 `discarded_incomplete_tail_bytes` 并将 `usage_complete` 标为 false（区别于活跃发布中的瞬时 `uncommitted_tail_bytes_at_publish`）。GCS 所有保留读路径通过 `open_snapshot`/`readers/<lease>.json` 线性化读租约保护整个读取与异常清理周期，传输统计区分 `actor`/`optimizer`/`gradients`/`reference`/`teacher`/`state`/`artifacts`/`journals` 的实际新传与复用及 `metadata_uploaded_bytes`。新快照文件和日志同步后发布 CURRENT，随后回收旧快照。

完整训练与官方评测完成、最终模型和恢复态回收完成后，执行并计量 `publish_final_logs`（写入 `checkpoint.final_publish_logs`），随后重写终态一致 `cell_summary.json`（`finalization_stage="finalized"`）并由明确不自计入成本的 `export_final_terminal_state` 同步至 `FINAL`/`LOGS`、写入 `.finalization_complete.json`（中断可在 `--resume` 接续收敛）；按记录在完成摘要中的保留策略清理本 cell 的历史原始文件。默认 8 轮周期下，Qwen2.5-7B 的主要 checkpoint 累计写入估算从每 cell 约 6.43/16.02/17.09 TB 降至 0.88/2.06/2.16 TB，分别对应 60/150/160 轮。原子替换仍短暂保留新旧恢复点，参数级磁盘峰值约 228.47 GB；需另外容纳当前 trainer state 与日志。详细容量公式与审计范围见 [优化方案](docs/fgpo_optimization_proposal.md)。

**当前结果是可审阅的方案与静态实现。** 当前会话没有仓库要求的 `SSH_Document.md`，已知 `mgt01` 入口在可用连接环境中无法解析，正式服务器运行尚未开始。本分支没有已执行 benchmark、显存节省比例或加速倍数；这些数值由上述必要实验的真实日志填写。

研究记录沿用指定服务器 `.research/research.sqlite` 的单写机制。当前会话没有该数据库的可用连接，未生成或改写实验运行记录。
