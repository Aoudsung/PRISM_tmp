# 可直接用于论文的实验素材

以下数值来自完整三模型结果，按当前论文 RQ1–RQ5 组织。英文句可作为结果段草稿使用，实验设置统一采用 [README 的取数约定](README.md#取数约定)。所有差值均由未四舍五入的原值计算，表格仅负责展示。下文 Qwen7、Llama8、Qwen27 分别指 Qwen2.5-7B、Llama-3.1-8B、Qwen3.8-27B。

## C1 · 六组测试设置中的整体效用

SPAR 在三模型、两基准的六组设置中都取得了 14 种方法中的最高平均测试效用。把机制消融也计入候选对照后，相对各组最强其他方法的优势为 **1.77–7.39 pts**。若单独讨论 GiGPO、Step-DPO、InfoPO、AREW、WorldEvolver、PoS/EnvProbe 六种命名 peer，优势为 **1.77–10.50 pts**。

| 模型 / 基准 | SPAR | 最强其他方法 | 对照效用 | 增益 pts |
|---|---:|---|---:|---:|
| Qwen7 / AppWorld | 0.6130 | Unweighted | 0.5842 | +2.87 |
| Qwen7 / PlanBench-XL | 0.6243 | Unweighted | 0.5932 | +3.11 |
| Llama8 / AppWorld | 0.6279 | Step-DPO | 0.5871 | +4.08 |
| Llama8 / PlanBench-XL | 0.6677 | Unweighted | 0.5939 | +7.39 |
| Qwen27 / AppWorld | 0.6948 | Step-DPO | 0.6770 | +1.77 |
| Qwen27 / PlanBench-XL | 0.6847 | Step-DPO | 0.6643 | +2.04 |

> Across the six model–benchmark settings, SPAR achieved the highest mean test utility among all 14 evaluated methods, improving over the strongest alternative in each setting by 1.77–7.39 points.

这句适合放在 RQ1 开头，随后引用正文主表。来源：`comparisons.json → method_comparisons[reference_role=strongest_other_method]`；均值和标准差见 `metrics.json → main_results`。九方法正文表之外的五种机制对照也已收集，扩充附录时可直接取数。

## C2 · 门控加权同时改善效用、成本和反馈利用

同分叉机制下，门控加权相对 Unweighted 在六组测试设置中提高效用 **2.87–10.00 pts**，平均宏动作成本下降 **7.36%–10.40%**，开发反馈利用增益提高 **11.21–19.71 pts**。这三条结果可以组成 RQ2 的主要叙述。

| 模型 / 基准 | 测试效用增益 pts | 成本下降 % | 开发 Δ_use 增益 pts |
|---|---:|---:|---:|
| Qwen7 / AppWorld | +2.87 | 8.02 | +11.21 |
| Qwen7 / PlanBench-XL | +3.11 | 7.36 | +13.21 |
| Llama8 / AppWorld | +10.00 | 10.40 | +16.82 |
| Llama8 / PlanBench-XL | +7.39 | 9.28 | +17.19 |
| Qwen27 / AppWorld | +4.64 | 9.14 | +19.71 |
| Qwen27 / PlanBench-XL | +4.61 | 8.30 | +16.19 |

> Relative to unweighted branching, aligned gate weighting increased test utility by 2.87–10.00 points while reducing mean interaction cost by 7.36%–10.40%. The corresponding masked-feedback gain increased by 11.21–19.71 points across the six settings.

来源：`comparisons.json → method_comparisons[reference_role=unweighted]`，含双方完整端点。现有 Fig.4(a) 可扩充 Qwen27；Fig.5(a) 的 AppWorld 三模型标注采用 +11.21/+16.82/+19.71 pts，六组总体范围也已在本表列出。

## C3 · 分层分支覆盖与普通采样对照

相对 Unstratified-2Branch，SPAR 的测试效用提高 **21.04–26.85 pts**；相对 Outcome-RLOO，提高 **27.49–31.31 pts**。这组结果能把“两分支覆盖”的实验比较补入 RQ2，并与精确核验段连接起来。

| 模型 / 基准 | Unstratified-2Branch | Outcome-RLOO | SPAR |
|---|---:|---:|---:|
| Qwen7 / AppWorld | 0.3444 | 0.3056 | 0.6130 |
| Qwen7 / PlanBench-XL | 0.3926 | 0.3212 | 0.6243 |
| Llama8 / AppWorld | 0.3808 | 0.3188 | 0.6279 |
| Llama8 / PlanBench-XL | 0.3998 | 0.3546 | 0.6677 |
| Qwen27 / AppWorld | 0.4712 | 0.4038 | 0.6948 |
| Qwen27 / PlanBench-XL | 0.4743 | 0.4099 | 0.6847 |

> Stratified branching improved mean test utility over the unstratified two-branch control by 21.04–26.85 points across the six settings. Gains over outcome-RLOO ranged from 27.49 to 31.31 points.

来源：`comparisons.json → method_comparisons[reference_role=unstratified/outcome_rloo]`。Outcome-PG、Outcome-Value、Info-RL-Neighbor 的完整记录也在 `metrics.json → main_results`，可以按同样口径扩展机制对照表。

## C4 · 精确枚举与 Monte Carlo 的一致结果

两阶段例子的精确目标为 `58/125 = 0.4640`，解析梯度、分层枚举、RLOO 枚举和 Autograd 均给出 `[0.0384, 0.0448, 0]`。该例子的方差恒等式两侧均为 `0.0164736`。现有论文表之外，10,000 次 Monte Carlo 已提供一组可以直接用于 RQ3 的数值复核。

| 量 | 精确值 | Monte Carlo |
|---|---:|---:|
| 分层总梯度方差 | 0.0395168 | 0.0395689 |
| RLOO 总梯度方差 | 0.0711904 | 0.0713668 |
| 总方差下降 | 44.49% | 44.56% |
| 分层方差 × 成本 | 0.0197584 | 0.0197844 |
| RLOO 方差 × 成本 | 0.0341714 | 0.0342668 |
| 方差 × 成本下降 | 42.18% | 42.26% |
| p=0.02 时零信息分支概率 | 96.04% | 96.12% |

> In the exact two-stage example, stratification reduced total gradient variance by 44.49% and the variance–cost product by 42.18%. A 10,000-episode Monte Carlo check yielded corresponding reductions of 44.56% and 42.26%.

来源：`metrics.json → exact_verification.values`，原始指针 `/part1_verification`。两服务器的这组统计值相同，仅计时字段不同，写作中作为同一套固定设置核验使用。

## C5 · 三模型完整阶段演化

六组模型×基准的 Stage 0–3 记录已经全部整理。三个训练阶段后，平均开发效用提高 **32.44–45.33 pts**。Qwen27 的两条完整曲线可以补足当前 Fig.2 仅展示部分模型的情况。

| 模型 / 基准 | 开发效用 Stage 0 → 3 | Δ_use Stage 0 → 3 |
|---|---|---|
| Qwen7 / AppWorld | 0.2723 → 0.5967 | 0.2042 → 0.5033 |
| Qwen7 / PlanBench-XL | 0.2073 → 0.6189 | 0.1167 → 0.3724 |
| Llama8 / AppWorld | 0.2877 → 0.6130 | 0.2333 → 0.5579 |
| Llama8 / PlanBench-XL | 0.2232 → 0.6765 | 0.1167 → 0.4306 |
| Qwen27 / AppWorld | 0.3482 → 0.6906 | 0.2819 → 0.6708 |
| Qwen27 / PlanBench-XL | 0.2554 → 0.6964 | 0.1458 → 0.5365 |

> Over three training stages, mean development utility increased by 32.44–45.33 points across the six settings, accompanied by higher masked-feedback utilization in every setting.

来源：`comparisons.json → stage_changes`；四阶段均值/标准差见 `metrics.json → stage_results`。`main.tex` 现有 RQ4 注释里的 Llama AppWorld `0.2139→0.5453` 更新写作时采用本表的 `0.2333→0.5579`。

## C6 · 九组预算比较

AppWorld dev 的全部九组模型×预算设置已整理。表内为 **SPAR 效用（相对 Unweighted 的增益 pts）**，后续正文可以同时写清优势大小如何随模型与预算变化。

| 模型 | 0.6 B₀ | 1.0 B₀ | 1.4 B₀ |
|---|---:|---:|---:|
| Qwen7 | 0.4136 (+2.53) | 0.5967 (+0.73) | 0.6752 (+0.02) |
| Llama8 | 0.4290 (+6.12) | 0.6130 (+7.56) | 0.6918 (+8.24) |
| Qwen27 | 0.4891 (+6.92) | 0.6906 (+6.49) | 0.7837 (+6.71) |

> Under the tight 0.6B₀ budget, SPAR improved over unweighted branching by 2.53, 6.12, and 6.92 points for Qwen7, Llama8, and Qwen27, respectively. The full budget sweep shows that the size of this advantage varies across backbones and budgets.

来源：`comparisons.json → budget_vs_unweighted`；14 种方法在所有预算下的端点见 `metrics.json → budgets`。RQ5 和 Fig.4(b) 可使用完整三模型矩阵；当前图中的 +6.1 pts 对应 Llama8 的紧预算设置。

## C7 · 根分叉与后续分叉的完整对照

根分叉在三个模型上均取得最高平均测试效用。将分叉位置均匀分布到前两步或前三步，对应测试效用下降 **6.44–23.72 pts**。Qwen27 的这部分数据此前尚未进入 Fig.4(c)。

| 模型 | T=0 | T∈{0,1} | T∈{0,1,2} |
|---|---:|---:|---:|
| Qwen7 | 0.6130 | 0.4788 | 0.3758 |
| Llama8 | 0.6279 | 0.5052 | 0.4127 |
| Qwen27 | 0.6948 | 0.6304 | 0.4908 |

> Root splitting achieved the highest mean test utility for all three backbones. Spreading the split uniformly over the first two or three steps lowered utility by 6.44–23.72 points in the evaluated schedules.

来源：`metrics.json → split_steps` 与 `comparisons.json → split_vs_root`。同一组记录还保留开发 Δ_use 和根门控概率，适合与效用并列展示。

## C8 · 用现有摘要恢复可追溯的轨迹结果

已从每个种子的任务数、成功率、发生失败的任务数和恢复率中恢复三类完整计数。这些计数可以直接用于后续重绘 Fig.5(b)，并补充恢复行为的结果段。

| SPAR 设置 | seed-task 评估数 | 无记录失败的成功 | 恢复成功 | 最终失败 | 汇总恢复率 |
|---|---:|---:|---:|---:|---:|
| Qwen7 / AppWorld | 504 | 409 | 37 | 58 | 37/95 = 38.95% |
| Qwen7 / PlanBench-XL | 270 | 208 | 39 | 23 | 39/62 = 62.90% |
| Llama8 / AppWorld | 504 | 433 | 22 | 49 | 22/71 = 30.99% |
| Llama8 / PlanBench-XL | 270 | 222 | 43 | 5 | 43/48 = 89.58% |
| Qwen27 / AppWorld | 504 | 492 | 10 | 2 | 10/12 = 83.33% |
| Qwen27 / PlanBench-XL | 270 | 264 | 0 | 6 | 0/6 = 0.00% |

> On PlanBench-XL with Llama8, SPAR recovered successfully in 43 of the 48 evaluations that encountered an intermediate failure, giving a pooled recovery rate of 89.58% across three seeds.

这里的汇总恢复率以“发生过中间失败的 seed-task 评估”为分母，和“全部评估中的恢复成功占比”分别展示。Qwen27 / PlanBench-XL 的结果则可以表述为 264/270 次评估在未记录中间失败的情况下成功。所有方法的三类计数、种子分解及原报告恢复率均值见 `comparisons.json → trajectory_outcomes`。当前 Fig.5(b) 的四类标签后续可改用这些有原始计数来源的三类结果。

## C9 · 资源账本与训练过程素材

每组 SPAR 的每种子平均记录为 15 次梯度更新、180 次快照保存和 180 次恢复。除原表展示的 token 总量外，`metrics.json → cost_ledgers` 保留全部方法的环境步、计时和快照字段；`seed_results` 保留种子分解，`training_progress` 保留 756 条阶段训练日志。

这部分适合支撑实验配置和资源开销附录，正文可简述训练安排，具体成本表采用 README 中的账本口径。模型编码元数据保存 GPU、文本数量、token 数量和预编码耗时，可与后续策略训练计时分栏展示。

## 后续写作落点

主结果段使用 C1 建立六组设置的全貌，机制段连续使用 C2、C3 说明加权和分层采样的实验差异，精确核验段使用 C4 将理论表与 Monte Carlo 连接起来。反馈利用段采用 C5 和 C8，预算与分叉位置段使用 C6、C7，C9 放进实验配置或附录。这样可以把现有结果组织成连贯论证，同时让每句数值都有现成来源。
