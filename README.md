# F-GPO r3

本分支从 `977bb86` 重构研究方案与实现。当前方法将同任务独立轨迹的真实终局成功回报，沿**相同步深度、相近实际历史状态**共享，以改善长时域 agent 的信用分配。目标是官方有限时域内的成功概率；采用 γ=1、固定回报尺度、整轨迹留一基线和有支撑权重的单步值共享。

方案、代码与实验采用协议 `fgpo-r3-success-v2`。主设计已经移除 TF-BEFP 内奖、目标指令重构和三遍递归图估值；新建 study 才能运行 r3。旧结果与检查点继续属于原协议，不转换为 r3 结果。

## 设计与实现

[设计与推导](docs/fgpo_design_and_derivation.md) 包含研究问题、文献位置、算法、误差条件和正式实验；[数学契约](docs/fgpo_math_contract.md) 定义精确状态、目标、留一关系与 PPO 更新；[性能实现说明](docs/fgpo_optimization_proposal.md) 逐项说明用户性能建议的采纳与修正。

| 路径 | 职责 |
|---|---|
| `src/fgpo/filtration.py`、`records.py` | 实际整数 token 历史、完整轮窗口、唯一响应归属和官方轨迹记录 |
| `src/fgpo/credit.py` | 同深度的整轨迹留一同伴值、支撑权重、固定尺度优势与内生诊断 |
| `src/fgpo/reference.py` | 固定参考模型的真实 actor-prefix 概率和历史/局部观测表示 |
| `src/fgpo/cache.py`、`policy.py` | 几何增长的 KV 缓存、掩码与逻辑位置、批量采样和有界表示微批 |
| `src/fgpo/optimization.py` | 分块立即反向的词表头、主干单次反向、有界传输的 CPU-master AdamW |
| `src/fgpo/envs.py`、`rollout.py` | 官方环境进程复用、独立 episode、提前派发动作和真实反馈归集 |
| `src/fgpo/protocol.py`、`engine.py`、`runner.py` | 方法身份、共享实验条件、续训语义和完整官方评测 |

参考模型固定在每个 cell 的初始 Qwen 权重。历史特征来自实际 actor prompt 的倒数第八层；局部观测特征来自公开反馈，独立做有界表示计算。上下文最多 32768 tokens，超限后保留任务/bootstrap 和最近完整轮。因此本方法明确使用**有界可见历史**。终局反馈保存在 step record，无额外 feedback-only 模型评分。

留一基线对目标动作的零均值结论适用于行为策略处的未裁剪 score estimator。同伴值替代是依赖历史匹配质量的近似；正式训练使用带 clipping、entropy 和 reference KL 的 PPO。方案没有把这些条件提升为严格无偏、单调改进或成功率保证。

## 正式比较

主配置运行下列八个方法，ALFWorld 与 AppWorld 均使用 seeds 0/1/2。

| 方法 ID | 作用 |
|---|---|
| `fgpo` | r3 完整方法 |
| `fgpo_no_full_filtration` | 两轮历史消融；保留此历史 ID，含义以 r3 为准 |
| `fgpo_no_transfer` | 关闭下一状态同伴值替代，保留同样的留一基线 |
| `fgpo_obs_only` | 相似度只使用当前局部观测 |
| `fgpo_no_support_weight` | 有同伴时直接采用同伴值的消融 |
| `gigpo_shared` | 在公共采样、历史、奖励和优化器下使用 GiGPO 信用分配 |
| `bipace_q_shared` | 在相同公共条件下使用 BiPACE-Q 信用分配，特征保留真实 prompt 历史 |
| `native_qwen_shared` | 相同推理接口的冻结 Qwen |

可训练主方法与共享基线均采用 60 rounds ×16 groups ×8 trajectories，lr=1e-6、weight decay=.01、PPO minibatch=256 rows、one epoch，按固定轨迹质量归一化。每个任务组共享任务与初始环境 seed，各轨迹使用独立采样 generator。输出上限 ALFWorld 256、AppWorld 1024；时域为 50/100。完整评测包含 ALFWorld valid_unseen 的 134 个任务，以及 AppWorld test_normal 168、test_challenge 417 个任务。ALFWorld 报告 SR；AppWorld 根据完整官方任务与 scenario 成员关系报告 TGC/SGC。

`gigpo`、`bipace_q`、`self`、`skill_disco`、`native_qwen` 仍是可显式选择的作者配置迁移实验，结果单列。GiGPO/BiPACE 的训练窗口为 150 iterations、γ=.95；SELF 使用 LoRA、EMA 与反馈更新，60 rounds 是本项目设定的迁移窗口；Skill-DisCo 执行一次离线归纳与验证流程。共享基线的作用是隔离信用分配差异，作者迁移实验用于补充完整方法比较。

## 指定服务器上的运行入口

本地仅编辑和静态审查；所有项目执行均按 [AGENTS.md](AGENTS.md) 在指定服务器完成。新的配置是：

- [Host A](configs/fgpo-r3-host-a.json)：源码目录 `/apps/users/adminuser/fgpo-r3`；继承已有模型、Python 和官方数据路径，保留已知排除 GPU 列表。
- [Host B](configs/fgpo-r3-host-b.json)：源码目录 `/home/chenxiangwei/workspace/fgpo-r3`；继承该服务器已有资产路径。

在对应服务器的独立 r3 checkout 中，使用配置指定的既有 Python，并把 `PYTHONPATH` 设为该 checkout 的 `src`。入口如下；`<physical_gpu_ids>` 由当次实际资源分配填写：

```bash
python -m fgpo.runner --config configs/fgpo-r3-host-a.json --output-dir outputs/fgpo-r3-study --gpu <physical_gpu_ids>
```

Host B 换用 `configs/fgpo-r3-host-b.json`。不指定 method、benchmark、seed 选择器时运行完整配置矩阵；可用 `--methods`、`--benchmarks`、`--seeds` 将完整单元分派给不同进程。每个进程显式拥有不重叠的物理 GPU，runner 通过 `CUDA_VISIBLE_DEVICES` 映射成进程内逻辑编号。`--model-gpu` 可以从已分配卡中选定实际模型计算子集。不要采用旧 launcher 内保存的历史资源快照安排新工作。

`cache.py` 使用已核对的 Transformers 4.57.1 Cache/Layer 接口；模型使用 bf16 与 SDPA、PyTorch autograd、Accelerate device map，SELF 另需 PEFT。实际服务器须记录已装版本及其接口结果。本次修改没有在本地安装或运行依赖。

## 内存与吞吐实现

KV 使用按层、按设备的几何缓冲，保留真实有效掩码及逻辑 RoPE 位置。普通连续缓存直接复用；存在 padding holes 时执行压紧。完整历史窗口变化会从固定根重放，避免复用受已淘汰历史影响的 K/V。每个 slot 的 generator 独立，采样结果合并一次回传；响应提前结束的槽立即进入 CPU 环境执行。

训练主干保留 activation checkpoint，默认 `activation_offload=false`；词表头每块立即 backward，仅累计 hidden 梯度，再进行一次主干 backward。CPU-master AdamW 使用 FP32 状态与有界 pinned staging，不创建整个参数的 GPU 写回镜像。环境池保留解释器并重新初始化每集官方 world。局部观测先去重，再按长度与 token budget 分微批。

主要资源参数为 `head_chunk_tokens=64`、`optimizer_bucket_mib=64`、`cache_initial_capacity=1024`、`representation_token_budget=8192`、`representation_batch_size=8`、`environment_pool_size=8`、`working_memory_gib=6`。6 GiB 是设备规划预留，实际峰值取决于上下文、KV 扩容、批量和模型更新阶段；Arena 预留容量与峰值显存要和吞吐一起评估。

## 记录、续训与交付状态

每个 cell 保存完整有效配置、官方任务清单、真实轨迹、参考概率/优势、训练与成本日志、官方逐任务评测和最终模型。`costs.jsonl` 区分 `host_dispatch` 与阶段 `synchronized_wall`；总效率使用 ledger 的 `wall_seconds` / `allocated_device_seconds`，不能将有重叠的所有 `seconds` 相加解释为训练墙钟。阶段日志包含每张卡的 peak allocated/reserved。

已有 r3 study 需要显式 `--resume`。续训比较实际生效的任务、奖励、历史、采样、PPO 和 Adam 配置；可调整 head/表示微批、KV 容量、offload 与 staging 等资源参数。旧协议及已改变的训练语义会被拒绝。断点保存待执行组、优势、部分累计梯度、optimizer/master/moments、固定参考与 RNG；未变化的参数文件通过硬链接复用。默认每 16 个组保存，托管停止仍在实际操作边界保存。

完整训练和全部官方评测结束后，原有完成流程保留 final model/adapters、复现元数据和原始结果，回收该 cell 已无后续用途的完整训练状态。最终部署模型支持评测复现，继续未完成训练使用完整 resume checkpoint。

**本次交付状态：代码与研究文档完成静态重构及交叉审查；服务器运行尚未执行。** 当前会话未获得仓库所要求的 `SSH_Document.md`，已知 `mgt01` 别名在可用连接环境中无法解析。因此本分支尚没有实测的显存、加速倍数或 benchmark 提升。后续正式运行直接验证设计中的信用分配与系统效率 claim，结果写入真实日志后再更新研究结论。

`scripts/launch_fgpo_formal.py` 与 `scripts/launch_fgpo_host_b.py` 已改为公共 runner 的薄入口，默认选择对应 r3 配置，均要求显式 `--output-dir` 和 `--gpu`，不会自动接续旧任务。`configs/*official-20261116.json` 与旧 Colab 指南保留作 v1 历史记录。当前源码拒绝以 v1 配置运行；旧 TF-BEFP/DAG 消融 ID 已退役。

## 研究记录工具

研究状态沿用 `AGENTS.md` 中的指定服务器 `.research/research.sqlite` 单写机制；本次会话没有可用的服务器数据库副本或连接入口，未生成、改写或伪造研究运行记录。实验完成后的实际记录与状态页由指定服务器上的现有工具维护。
