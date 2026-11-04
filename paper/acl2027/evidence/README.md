# 实验结果与论文证据索引

本目录整理截至 2026-11-04 已保存的三模型实验结果，供论文结果段、claim、表格和绘图直接取数。统一入口是仓库中的 `outputs/prism_evolve_experiments/prism_evolve_full_results.json`；它与三份模型结果逐对象一致，27B 数据也与大显存服务器上的原始副本一致。

**写作从 [claim_evidence.md](claim_evidence.md) 开始。** 其中按论文问题组织了完整端点、对比数值、英文候选句和图表落点。需要扩展表格或作图时读取 [metrics.json](data/metrics.json)；常用差值、预算对比和恢复计数已经计算在 [comparisons.json](data/comparisons.json) 中。每条记录带原始 JSON pointer，可以直接追溯到统一数据源。

## 已整理的内容

| 数据分组 | 数量 | 用途 |
|---|---:|---|
| 主结果 | 84 | 3 模型 × 2 基准 × 14 方法，保留均值与标准差 |
| 种子记录 | 252 | 42/43/44 三种子的最终 dev/test 指标和成本账本 |
| 阶段结果 | 336 | 每种方法的 Stage 0–3 开发集演化 |
| 训练日志 | 756 | 每种子、每阶段的 loss、return、累计步数与 token 计数 |
| 预算配置 | 126 | AppWorld dev，3 模型 × 14 方法 × 3 档预算 |
| 分叉步结果 | 9 | 3 模型 × 根分叉、前两步均匀分叉、前三步均匀分叉 |
| 成本账本 | 84 | 环境步、token、计时、梯度步数和快照操作 |
| 方法对比 | 36 | Frozen、最强 peer、最强其他方法、Unweighted、Unstratified、RLOO |
| 轨迹结果计数 | 84 | 无记录失败的成功、恢复成功、最终失败；合计 32,508 次 seed-task 评估 |
| 理论数值核验 | 1 套 | 精确枚举、Autograd 与 10,000 次 Monte Carlo |

这里的 32,508 次是任务在不同方法和种子下的评估次数。固定任务集为 AppWorld test_normal 的 168 个任务、PlanBench-XL test 的 90 个查询；三个模型的 train/dev/test 划分分别使用 AppWorld 90/57/168 和 PlanBench-XL 180/57/90。

## 文件与来源

[来源清单](source_manifest.json) 给出服务器路径、归档成员、重复副本关系和版本记录。两个原始快照已下载并纳入 Git：

| 快照 | 内容 |
|---|---|
| [mgt01 原始结果与源码](sources/prism-evolve-mgt01-evidence-20261104.tar.gz) | 完整汇总、三模型 JSON、编码元数据、报告、历史单模型 JSON、当前 runner 与策略/环境源码 |
| [27B 服务器原始结果与源码](sources/prism-evolve-ubuntu27b-evidence-20261104.tar.gz) | 27B 原始 JSON、当地汇总和报告、编码元数据、当前源码 |

模型权重、约 11.7 MB/模型的 embedding 缓存和原始基准数据继续保留在服务器。轻量 [27B 编码元数据](sources/qwen38_27b_embed_cache.meta.json) 已单独收集。

`results/prism_evolve/prism_evolve_full_results.json` 是旧的单模型版本，已经作为历史材料保存在 mgt01 快照中。新的取数统一使用 `outputs/` 下的完整三模型文件。完整文件根节点的 `part2_appworld`、`part3_diagnostics`、`part4_planbench_xl` 是 Qwen7 的别名；导出只遍历 `backbones`，避免重复计数。

## 取数约定

正文数值保持原始精度计算，展示时通常取四位小数；效用差的 `pts` 定义为 `100 × (SPAR − 对照)`，成功率差用百分点，成本下降百分比用 `100 × (对照成本 − SPAR成本) / 对照成本`。`±` 为三个种子之间的样本标准差。

`final_test` 用于测试集主结果；阶段变化、预算扫描和 `Δ_use` 用于开发诊断。反馈遮蔽按当前 runner 的固定 `dev_specs[:24]` 执行，遮蔽首次信息动作的返回内容；没有信息动作的样本贡献零差值。分叉步实验的变化对象是分叉位置 `T`。

恢复计数来自每个种子的 `num_tasks`、`success_rate`、`had_failure_count` 和 `post_failure_recovery_rate`。导出同时保留原报告的种子恢复率均值与按失败任务数汇总的 `pooled_recovery_rate`；写作使用其中哪一种，表题便标明对应口径。预算原文件保存三种子均值；分叉步文件保存聚合均值/标准差，完整种子摘要留在各主实验记录中。

成本账本的 `cost_ledger_mean` 是每种子训练总量在三个种子间的平均值。`total_tokens` 和 `model_inference_sec` 包含适配器计数与缓存编码摊销，`wall_clock_sec` 单独保留。原文件中的 `delta_evo_ci95_*_mean` 保留原名，表示种子内场景分组区间端点的平均；本索引的候选句采用均值、标准差和明确差值。

实验设置段统一说明当前执行对象即可：三种冻结模型表征与结构特征共同输入 512 维低秩策略，在基准数据构造的固定宏动作适配环境中训练和评测，比较方法采用当前仓库的统一适配实现。模型规模相关特征也参与策略构造。两端收集时源码与本地一致，原结果未记录运行时源码 revision；原始结果和对应源码快照均已保留。

## 更新方式

在 mgt01 项目目录执行：

```bash
.venv-prism-evolve/bin/python scripts/collect_prism_evolve_evidence.py \
  --input outputs/prism_evolve_experiments/prism_evolve_full_results.json \
  --output-dir paper/acl2027/evidence/data
```

首次导出在隔离输出目录执行，实际命令与返回码见 [commands.txt](verification/commands.txt)，条数、指针回读及轨迹计数核对见 [verification.log](verification/verification.log)。本次仅整理已保存结果，没有新增训练或模型调用。
