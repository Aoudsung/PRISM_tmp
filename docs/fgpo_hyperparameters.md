# F-GPO 分基准超参数

实验的方法名统一为 **F-GPO**（`method=fgpo`，协议 `fgpo-r3-published-v4`）。**默认、联合参数（joint）、radius** 是同一方法的配置标签。7B 表中的加粗项表示相对该基准默认配置的改动。1.5B 表列出原 joint 和追加的正式适配对照，后文记录选参依据。

## Qwen2.5-7B-Instruct 已运行配置

| Benchmark | 超参配置 | 训练 seed | 学习率 | 匹配半径 ε | 先验权重 κ | 续值共享上限 λmax | 熵系数 | 参考 KL 系数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALFWorld | 默认 | 0、1 | 1e-6 | 0.10 | 1.0 | 0.5 | 0.001 | 0.01 |
| WebShop | 默认 | 0、1、2 | 1e-6 | 0.10 | 1.0 | 0.5 | 0.001 | 0.01 |
| WebShop | 联合参数 | 1 | 1e-6 | 0.10 | 1.0 | **0.25** | **0.002** | 0.01 |
| WebShop | radius | 1 | 1e-6 | **0.05** | **0.5** | **0.125** | **0.002** | 0.01 |
| TextCraft | 默认 | 0、1（部分训练） | 1e-6 | 0.10 | 1.0 | 0.5 | 0.001 | 0.01 |
| TextCraft | 联合参数 | 0 | **2e-6** | 0.10 | **0.5** | 0.5 | 0.001 | 0.01 |
| TextCraft | radius | 1 | 1e-6 | **0.05** | 1.0 | 0.5 | 0.001 | 0.01 |

学习率、ε、κ、λmax、熵系数与参考 KL 系数依次对应 `learning_rate`、`credit_radius`、`credit_prior_weight`、`credit_transfer_max`、`entropy_coef` 和 `kl_coef`。其中 ε 决定匹配支持范围，κ 决定同伴估计向同题平均值收缩的程度，λmax 限制续值共享权重。配置文件省略 ε 或 κ 时，F-GPO 分别采用 0.10 和 1.0。[协议与配置解析](../src/fgpo/protocol.py) · [信用参数的默认值与用法](../src/fgpo/credit.py)

## Qwen2.5-7B-Instruct 训练起点与预算

| Benchmark | 训练起点与轮次 | 每条轨迹最多交互步数 | 每步最多生成 token | 上下文上限（token） | 终评口径 |
| --- | --- | ---: | ---: | ---: | --- |
| ALFWorld | 默认配置从基础模型训练 60 轮 | 50 | 512 | 32,768 | 完整 valid_unseen 134 题。seed 0 另有 valid_seen 128 回合评测 |
| WebShop | 默认配置从基础模型训练 60 轮，再续至 150 轮。seed 1 的三组均从同一默认配置的第 60 轮完整检查点续训，表中各自的超参用于第 61–150 轮 | 15 | 512 | 32,768 | 每种子固定 128 道测试题，报告满分 SR 与平均 Score |
| TextCraft | 各配置从同一基础模型独立启动，预算为 50 轮。默认配置 seed 0、1 分别在完成 7、3 个完整轮次后暂停 | 30 | 128 | 8,192 | 第 50 轮终态评完整 val 79 题，报告总体和分配方深度 SR |

WebShop 的 radius 相对联合参数同时改变 ε、κ 和 λmax，熵系数沿用 0.002。TextCraft 的 radius 与同 seed 默认配置只有 ε 不同。TextCraft 联合参数实际使用每步 128 token、熵系数 0.001、KL 系数 0.01。曾排程的 192 token 配置已撤出，未纳入实跑表。[WebShop 续训协议](../src/fgpo/webshop_continuation.py) · [TextCraft 执行记录](fgpo_v4_textcraft_execution_plan.md)

## Qwen2.5-1.5B-Instruct 正式运行配置

三项均从同一份 Qwen2.5-1.5B-Instruct 原始指令模型开始，actor 和固定参考均使用该尺寸的权重。表内参数从第 1 轮起生效。各基准沿用上表对应的交互步数、token 上限和主终评集。

| Benchmark | 超参配置 | 训练 seed 顺序 | 学习率 | 匹配半径 ε | 先验权重 κ | 续值共享上限 λmax | 熵系数 | 参考 KL 系数 | 每种子轮数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WebShop | [joint](../configs/fgpo-1p5b-webshop-joint.json) | 1 | 1e-6 | 0.10 | 1.0 | 0.25 | 0.002 | 0.01 | 150 |
| WebShop | [适配版 joint](../configs/fgpo-1p5b-webshop-joint-adapted.json) | 1，原 seed 1 终评后 | 1e-6 | **0.05** | **0.5** | 0.25 | 0.002 | 0.01 | 150 |
| ALFWorld | [joint](../configs/fgpo-1p5b-alfworld-joint.json) | 0 | 1e-6 | 0.10 | 1.0 | 0.25 | 0.002 | 0.01 | 60 |
| TextCraft | [joint](../configs/fgpo-1p5b-textcraft-joint.json) | 1，已保存 | 2e-6 | 0.10 | 0.5 | 0.5 | 0.001 | 0.01 | 原定 50，已完成 15 |
| TextCraft | [joint，单个公开格式示例](../configs/fgpo-1p5b-textcraft-joint-format.json) | 1，同基座从头接替 | 2e-6 | 0.10 | 0.5 | 0.5 | 0.001 | 0.01 | 50 |

WebShop 沿用 7B joint 的参数值，1.5B 全程使用 joint。7B 的 seed 1 是先用默认配置训练 60 轮，再用 joint 续训 90 轮，比较终态时需保留这项训练履历差异。ALFWorld 首次采用 joint，沿用 WebShop 的参数组合。TextCraft 采用 7B TextCraft joint 的参数值，学习率设为 2e-6、κ 为 0.5，其余信用和熵参数采用默认值。[各卡位和评测安排](fgpo_experiment_schedule_current.md#15b-实验计划)

## 1.5B 超参适配实测

选参使用官方训练题，正式 SR 由预定的最终检查点评估。下表取完整训练轮次，每轮 16 个任务组、每组 8 条轨迹。混合组同时含有成功和失败轨迹，每个混合组触发一次 AdamW 更新。

冻结模型小试取 `T=0.7` 作为降温候选，这个值来自 Qwen 发布的生成配置。小试使用 F-GPO 的全词表采样，`top_p=1`。正式 V4 的当前策略、行为策略和参考策略以单位温度对响应评分。正式训练如采用其他温度，这三处及熵计算均须按同一训练分布实现。[Qwen 原始生成配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json) · [正式协议](../src/fgpo/protocol.py)

| 1.5B 原 joint | 完整训练窗口（R0 起计） | 满分成功 / 轨迹 | 全败 / 混合 / 全胜组 | 窗口内 AdamW 更新 |
| --- | --- | ---: | ---: | ---: |
| ALFWorld seed 0 | R0–R5 | 2/768（0.26%） | 94 / 2 / 0 | 2 |
| TextCraft seed 1 | R0–R11 | 0/1,536（0%） | 192 / 0 / 0 | 0 |
| WebShop seed 1 | R0–R9 | 104/1,280（8.13%） | 119 / 40 / 1 | 40 |
| WebShop seed 1 | R134–R143 | 995/1,280（77.73%） | 25 / 33 / 102 | 33 |

### ALFWorld

前六轮的两次成功分别出现在 Look（1/56）和 Pick（1/176）。Clean、Cool、Heat、Pick2 共 536 条轨迹全部失败。同 seed 的 7B V4 在相同的前六轮成功 273/768，形成 53 个混合组。7B 用默认熵系数 0.001 和 λmax=0.5，1.5B 用 0.002 和 0.25，两者每步上限均为 512 token。该比较包含模型尺寸和超参两方面差异。[7B V4 同 seed 原始记录](../outputs/fgpo-colab-956ceab-ag-20261116/gdrive_staging/comparison/alfworld/fgpo/seed-0/training.jsonl)

对 1.5B 首轮保留的 128 条轨迹逐步检查，38 条首步命中环境的可行动作，42 条在整条轨迹中至少命中过一次，127 条走满 50 步。生成长度中位数为 40 token。把原始响应在 256 token 处截断并重新解析，只节省 4,374/306,078 个生成 token（1.43%），唯一成功轨迹中已有两步的解析结果会改变。这是固定历史的截断回放，支持继续使用 512 上限。4,505 个解析失败的响应中，有 387 个仍在单独一行或 `action:` 行完整写出了当前菜单中的命令。实际样本也出现了拿到正确物品后重复输入无效命令的情况。选参同时关注合法动作、整条任务的完成情况和混合组。[六类任务与长度回放](../outputs/fgpo-1p5b-alfworld-adaptation-20261121.json)

两次实际更新的裁剪前梯度范数分别约为 873 和 2,225，均触及范数上限 1。PPO 概率比裁剪仅涉及 25/37,495 个已评分的响应 token。其余 94 个全败组没有梯度或优化器步。当前采用学习率 1e-6、PPO clip 0.2、λmax=0.25，优先通过配对采样寻找能增加混合组的条件。

在正式 R0 的 16 个训练任务、每题原 8 个槽位上，重放完全相同的首步提示、基础模型与槽位随机种子。原温度 1.0 逐 token 复现了全部 128 条原响应。降温后 15 个原本合法的槽位变为不合法，仅 2 个反向改善。升温后分别为 13 个和 5 个。三档中 `T=1.0` 保留最多合法首步。这个面板覆盖 R0 的五种任务类型，Look 仅计入前六轮的训练统计。[逐槽与分类型温度结果](../outputs/fgpo-1p5b-alfworld-first-actions-t12-20261121.json)

| 训练采样温度（每步上限 512） | 可提取首步命令 | 首步命中当时的可行动作 | 达到 token 上限 |
| ---: | ---: | ---: | ---: |
| 0.7 | 26/128 | 25/128 | 0/128 |
| **1.0** | **40/128** | **38/128** | 0/128 |
| 1.2 | 34/128 | 30/128 | 4/128 |

用原种子配对跑完一道 Clean 和一道 Pick 训练题，每题 8 条轨迹、每条至多 50 步。`T=1.0` 与 `T=0.7` 的终局成功均为 0/16，混合组均为 0/2。合法动作分别为 **216/800** 和 **201/800**，可提取动作分别为 382/800 和 250/800。降温减少了生成 token（41,197 → 21,840）和采样耗时（662 → 303 秒），代价是较少的合法动作。结合覆盖更多任务的首步配对，正式训练沿用 `T=1.0、top_p=1、512 token`。两道题的完整轨迹仅用于判断冷启动行为，官方终评采用完整 valid_unseen 134 题。[同题完整轨迹对照](../outputs/fgpo-1p5b-adaptation-20261121/paired-sampling/numeric/summary.json)

单独在原系统提示末尾加一条固定 `inventory` 格式示例，`T=1.0`、长度 512 与任务相同。128 个配对首步中，可提取命令从 40 增至 117，命中原生菜单从 38 增至 98。其中 `inventory` 从 0 增至 60，`look` 从 6 增至 19。其他原生合法命令从 **32 降至 19**（新增 12、失去 25）。这解释了为什么要同时报告查背包以外的动作和真实终局成功。[独立输入对照的逐任务结果](../outputs/fgpo-1p5b-alfworld-first-actions-example-20261121.json)

![配对首步上温度与格式示例的影响](../outputs/fgpo-1p5b-alfworld-first-actions-20261121.png)

在相同两道题的完整轨迹上，示例使合法动作从 216/800 增至 411/800。新增条件共执行 163 次 `inventory` 和 80 次 `look`，其中 116 次是连续重复 `inventory`，另有 168 次其他合法命令。两种输入的终局仍均为 **0/16 成功、0/2 混合组**。目前正式训练使用原始提示。[完整轨迹的单示例对照](../outputs/fgpo-1p5b-adaptation-20261121/paired-sampling/alfworld-format-example/summary.json)

### TextCraft

前 12 轮的 192 个任务组全败，F-GPO 依定义跳过这些组的更新。此时改变学习率、熵系数、KL 系数或匹配半径，都不会改变下一条轨迹的采样策略。第 12 轮留存的 3,840 个环境步中，3,838 次收到空命令拒绝，另一次收到 `...` 的拒绝，一次 `inventory` 成功返回空背包。原生模拟器没有提供统一的动作有效性布尔值。历史日志中缺失该字段时写入的 0% 不能用于判定命令是否执行。看板将这项指标显示为缺失，适配实验根据模拟器实际返回的取物、合成及查背包反馈统计命令被接受的次数。[动作解析和环境调用](../src/fgpo/textcraft.py) · [配对实验实现](../experiments/fgpo_1p5b_adaptation/paired_sampling.py)

在两道预先选定的正式训练题 `jungle_fence`（深度 2）和 `hopper_minecart`（深度 4）上，冻结同一 1.5B 基座，每题每个条件使用同样的 8 个槽位种子并运行完整 30 步。原配置 `T=1 / 128 token` 和候选 `T=0.7 / 256 token` 均为 **0/16 成功、0/2 混合组、0/480 步可提取命令**。触顶且未生成结束符的步数从 51/480 降到 11/480。该对照比较的是整个采样组合，终局结果没有支持替换正式配方。额外格式示例使用原温度和长度，在相同题目上独立比较。

只增加固定的 `inventory` 格式示例后，480 步中可提取命令从 0 增至 165，环境返回成功取物、合成或背包查询反馈的命令从 0 增至 22。共有 15 次 `inventory`，5/16 条轨迹出现合计 7 次背包变化。终局仍是 **0/16，混合组 0/2**。示例改善了命令表达，当前证据支持单列输入对照。正式 F-GPO 的奖励仍来自原生任务完成。[格式示例实测](../outputs/fgpo-1p5b-adaptation-20261121/paired-sampling/format-example/summary.json)

在需要接替的正式 seed 1 上，再取原 R0 固定顺序的前 8 个任务组，各用原有 8 个槽位种子。原始模型在这一轮尚未更新。旧输入已记录 **0/64**，同一基座加示例得到 **1/64 成功、1/8 混合组**，另 63 对均失败。成功轨迹用 9 步完成 `dark_oak_pressure_plate`，产生 3 次背包变化。这个结果支持该输入能够触发 F-GPO 的首次更新。正式替换只改公开输入，沿用原解析器和终局奖励，官方效果由新单元的 val 79 题终评衡量。[seed 1 同任务确认](../outputs/fgpo-1p5b-adaptation-20261121/paired-sampling/textcraft-seed1-confirmation/summary.json)

正式训练首轮，原 seed 1 的输入为 0/128 满分、0 次更新。加入示例后，同一组的 `dark_oak_pressure_plate` 成功，首轮共 **1/128 满分、1/16 混合组、1 次 AdamW 更新**，裁剪前梯度范数为 563.68，随后开始第二轮。这个结果验证了模型已获得一次真实更新。终评使用完整 79 道验证题。[正式适配配置](../configs/fgpo-1p5b-textcraft-joint-format.json)

### WebShop

对原 joint 的 R136、R144 两轮正式训练轨迹重放信用估计。两轮共有 256 条训练轨迹，原生训练 SR / Score 分别为 78.13% / 89.79 和 76.56% / 82.83。下表只纳入 8 个混合组的 436 个动作步，在同一动作步上预测留一轨迹的二值终局，按动作步数汇总 Brier，越低越好。λmax 固定为 0.25。

| 匹配半径 ε | 先验权重 κ | 留一 Brier | 有同伴支持的动作步 |
| ---: | ---: | ---: | ---: |
| 0.10 | 1.0 | 0.23827 | 376/436 |
| 0.05 | 1.0 | 0.22298 | 359/436 |
| **0.05** | **0.5** | **0.22103** | 359/436 |

窄半径在两轮均降低了 Brier。最佳组合相对原配方的 0.01724 改善中，约 95% 来自两个混合组。基线重放与原日志 Brier 相差 0.00040。正式配对采用 **ε=0.05、κ=0.5、λmax=0.25、学习率 1e-6、熵系数 0.002、参考 KL 系数 0.01**。同 seed 1 从同一指令基座独立训练 150 轮，在相同 128 道测试任务上计算两臂满分 SR、平均 Score、原来失败后成功及原来成功后失败的题数。[逐组回放](../outputs/fgpo-1p5b-webshop-credit-20261121.json) · [适配配置](../configs/fgpo-1p5b-webshop-joint-adapted.json)

原 joint seed 1 已完成 150 轮和 856 次更新。在完整 128 题官方测试上，**满分 SR 为 68.75%（88/128），平均 Score 为 78.04**。适配版 seed 1 已在同卡从同一份基础权重独立启动，沿用同一训练任务顺序和测试清单。[原 joint 官方终评](../outputs/fgpo-1p5b-webshop-original-seed1-evaluation-20261121.json)

## WebShop 与 TextCraft 的文献对照方案

2026-11-22 起，正式冻结成绩采用模型在本项目训练时实际使用的输入、历史与解析。WebShop 的环境版本、三组任务和 SR/Score 定义继续对齐，TextCraft 将步数上限扩到 70，保留训练时的每步 128 token。以下第三方提示设置用于说明文献协议及已归档的接口迁移诊断。当前执行范围与逐项数据来源见[原生补评排程](fgpo_experiment_schedule_current.md#7b-冻结模型补评)。

同一 7B joint 终态在面板 1 前 8 题的配对结果为：按原训练接口 7/8 满分，Score 94.6429；按 GraphGPO 作者接口 0/8 满分，Score 55.0794。两边分别生成 920 token / 53 个环境步与 15,763 token / 58 个环境步。作者接口有 6 次部分分购买、2 次走满 15 步，10 个达到 512 token 的响应均未通过动作格式判定。第 1 题的 60 cm 镜子在选项里可见，作者接口下选对商品后直接结账，得 5/6；原接口该题得满分。6 个相同初始状态上的确定性首步回放，在原模板下均直接生成带标签的搜索命令，换作者模板后均先生成思考标签。这一回放未执行环境动作。[逐题、逐步与同输入回放证据](../outputs/fgpo-author-eval-542-20261121/webshop-native-interface-diagnosis-20261122.json)

看板截图中的面板 0 前 48 题也已在原训练接口下实际重新推理。权重、任务、worker seed 与模型采样种子逐条一致。满分从 5/48 增至 38/48，原有 5 题均保留，新增 33 题；平均 Score 从 55.24 增至 87.56。每步平均输出从 273.52 降至 17.10 token。原接口在面板 1 全部 128 题的成绩为 109/128、Score 94.90，作为该面板的归档记录复用。[48 题配对记录](../outputs/fgpo-author-eval-542-20261121/webshop-native-interface-diagnosis-20261122.json)

7B joint 的三个原生面板已完成。面板 0 为 101/128、Score 90.85，面板 1 为 109/128、Score 94.90，面板 2 为 109/128、Score 92.96。合计 **319/384 满分，SR 83.07%，平均 Score 92.90**。这三组读数来自同一份训练权重和三组测试抽样。[冻结评测汇总](../outputs/fgpo-author-eval-542-20261121/native-training/webshop.summary.json)

7B 的冻结补评只取联合参数版本的 F-GPO。WebShop 用 joint seed 1 的第 150 轮权重（前 60 轮默认配方，后 90 轮 joint），384 行由面板 1 的 128 条真实原生终评和面板 0、2 的 256 条新增推理组成。TextCraft 用从基座按 joint 训练的 seed 0 终态与项目训练的 GraphGPO seed 0 各评 79 条。1.5B 的 414 行由两份 WebShop 各 128 条原生终评和两份 TextCraft 各 79 条新增推理组成。[7B 补评看板](http://127.0.0.1:8765/?tab=frozen-eval) · [1.5B 任务清单与依赖](fgpo_experiment_schedule_current.md#训练归档后自动接续的-414-条冻结补评)

### 历史设计与文献口径

以下是切换前的评测设计与成本估算，供解释已归档的接口迁移结果。WebShop 以 GraphGPO 表 1 为主要文献参照，TextCraft 以 BiPACE 表 3 为主要文献参照。文献结果注明各自输入、训练轨迹、时域及检查点选取口径。[GraphGPO 附件，第 7、17 页](../papers/2026-agent-rl/core/2026_ICML_GraphGPO.pdf) · [BiPACE 附件，第 6、14、16 页](../papers/2026-agent-rl/core/BiPACE.pdf)

| 对照 | 可复用结果 | 影响结论的主要差异 |
| --- | --- | --- |
| WebShop，GraphGPO 表 1 | 作者在 `text_rich` 下运行 GraphGPO 并重测 GRPO、GiGPO。7B GraphGPO 的 SR / Score 为 80.31 / 86.94，1.5B 为 78.65 / 89.29。PPO、RLOO 与提示方法保留其转引来源。 | 现有 7B F-GPO 与作者主配方均为 150 轮、16×8、15 步、每步 512 token、训练/验证温度 1.0/0.4。作者使用最近两步历史，prompt 上限 5,120；F-GPO 的总上下文上限为 32,768，菜单过滤不可执行的搜索按钮。表 1 写的是大部分结果按三个测试种子汇总，论文未给出每个表格单元的精确测试抽样和选点。 |
| TextCraft，BiPACE 表 3 | BiPACE 作者实跑 Qwen、GRPO、GiGPO、HGPO、BiPACE-Q。7B 后四者的总体峰值 SR 分别为 76.5、87.0、87.5、91.1，1.5B 为 43.8、58.3、59.4、65.1。 | BiPACE-Q 的公开配方每轮 16×12，最多 70 步，每步 512 token。7B 训练 50 轮、1.5B 训练 100 轮，表内报告三训练种子的窗口内验证峰值。F-GPO 两个尺寸均为 16×8、50 轮、30 步、每步 128 token，在预定终态评全 79 题。 |
| KnowGPO | 采用作者在受控比较中固定初始权重、训练数据、评测工具和采样条件的做法。 | 作者使用同一份 Qwen3-8B SFT 检查点比较搜索策略，任务为 Xbench、BrowseComp-ZH 和 GAIA。两项目标基准的结果栏标为「原文未报告」。[KnowGPO 附件，§4.1 与附录 E](../papers/2026-agent-rl/core/2026_EMNLP_KnowGPO.pdf) |

**WebShop 的最小补评。** 当前 7B 默认配置三个训练种子的末轮 SR 为 82.03±1.35%，Score 为 89.60±2.10。1.5B 原 joint seed 1 为 68.75% / 78.04，适配版按相同清单独立报告。若需要贴近 GraphGPO 表 1 的测试口径，先选定一份 F-GPO 末轮权重，在作者 `text_rich` 公共提示、两步历史、5,120 prompt token 和原生动作解析下，用三份预先固定的 128 条清单补评，共 384 条推理轨迹。每条记录 worker seed、session index、模型采样种子、满分和原生得分。三个测试抽样度量同一模型的推理与题目波动，原有的三次独立训练另列。[三种子原始汇总](../outputs/webshop150-paired-20261119/webshop-paired-report/summary.md) · [GraphGPO 作者配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/recipe/GraphGPO/run_qwen2.5_7b_webshop_train.sh) · [逐种子任务抽样](../src/fgpo/webshop.py)

需要逐题直接对照时，可加测 GiGPO 作者已发布的 7B WebShop 权重。两份权重使用完全相同的输入包装、模型采样种子和清单，三批合计 768 条推理轨迹，可以报告满分得失及逐题 Score 差。这份公开权重属于 GiGPO 原作。GraphGPO 表 1 的 GiGPO 行属于后续作者的 `text_rich` 重跑。BiPACE 表 2 的 WebShop 7B 使用另一组预算，16×12×150 为 28,800 条，对应 F-GPO 的 19,200 条。其公开环境默认使用 `text`，1.5B 的论文组宽 12 与脚本组宽 8 也须保留各自来源。[GiGPO 作者权重](https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop/tree/1d57ea6286ea35609acce9d5ee44a0d884c4aed8) · [BiPACE 表 2](https://arxiv.org/html/2606.25556v1#S4.T2) · [BiPACE 环境入口](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/agent_system/environments/env_manager.py#L705-L715) · [BiPACE 1.5B WebShop 脚本](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/examples/bipace/webshop/qwen1.5b_2gpu/train_bipace.sh)

**TextCraft 的最小补评。** 从现有终评逐题账本读取，7B F-GPO radius seed 1 的 17 次失败和本项目移植版 GraphGPO seed 0 的 24 次失败全部走满 30 步。两者分别成功 62/79、55/79，训练种子不同。优先在 F-GPO joint seed 0 完成后，冻结它和已归档的 GraphGPO seed 0，用同一修复版模拟器、79 个目标、固定的初始配方与采样种子重评。采用 BiPACE-Q 的公开提示（含作者示例）、四步历史、3,500 prompt token、每步 512 token、温度 0.4、70 步和作者的动作解析。每个模型只做 79 条推理，同一条轨迹同时统计第 30 步和第 70 步前的累计成功，以及深度 2/3/4 的成功数。这样能测出放宽交互步数实际救回多少题。79 题终态结果继续单列；BiPACE-Q 作者的公开脚本每 10 轮从 79 个验证目标中轮转评 64 条，再按窗口取峰值，GRPO/HGPO 发布脚本的验证批量也有差异。[BiPACE-Q 7B 脚本](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/examples/bipace/textcraft/qwen7b_4xh100/train_bipace.sh) · [作者目标抽样](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/agent_system/environments/env_package/textcraft/envs.py) · [作者动作解析](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/agent_system/environments/env_package/textcraft/projection.py) · [本地任务与部署记录](fgpo_v4_textcraft_execution_plan.md)

若论文需要在 **BiPACE-Q 的训练预算和选择规则** 下直接比较，优先复用作者逐轮日志或终态权重。相同期次、任务、种子和计分口径的日志可直接重算；统一全 79 题评测需要该检查点的完整逐题记录或权重。已查公开仓库尚未定位到表 3 对应的 TextCraft 权重或逐轮原始日志。下一档成本是仅从同一基座新训 F-GPO，优先 7B，按 BiPACE-Q 的 16×12×50、70 步、512 token、公共模板及同一验证轮转和选点规则，预先确定训练种子，并列固定末轮与验证峰值。一个种子为 9,600 条训练轨迹，三个为 28,800 条；1.5B 作者窗口的一次训练为 19,200 条。旧的 6,400 条短预算轨迹属于原实验。F-GPO 的二值终局奖励、组级更新与 BiPACE-Q 的非法动作惩罚和优化配方仍各自注明；若主张仅涉及信用估计器，另需一个在相同输入、训练奖励及更新规则下只替换估计器的对照。[BiPACE 附录 I、N、O](https://arxiv.org/html/2606.25556v1#A9) · [BiPACE 发布材料](https://github.com/TianxiangZhao/BiPACE/tree/f19f6fe1d140dd342d81b8e797f09c37c620b42f/examples/bipace) · [F-GPO 训练配置](../src/fgpo/protocol.py)

若作者提供可追溯的逐轮记录，比较样本效率时可以按累计轨迹数选点。7B WebShop 的 F-GPO 第 150 轮与 BiPACE 第 100 轮均为 19,200 条；TextCraft 的 F-GPO 第 45 轮与 BiPACE 第 30 轮均为 5,760 条。只有双方留存了对应权重或同口径评测记录，才报告该点。已检查的已完成 7B TextCraft radius 和 GraphGPO 目录均只保留终态权重。各点仍须列出交互上限、实际环境步和生成 token。逐题差值与误差区间只用于有共同任务和采样清单的本地比较，训练种子数与重复推理次数分别报告。

实现上优先复用已导出的终态 actor，用独立评测入口读取任务与种子清单，调用所选作者的提示、历史、解析器和原生环境，逐题写出结果。这样可让同一个 actor 接受不同公开接口的评测，并保留原正式成绩。若另设仅比较信用估计器的受控训练，GraphGPO WebShop 作者配方的 actor 学习率为 1e-6，BiPACE-Q TextCraft 两个尺寸均为 5e-7，二者 KL 系数均为 0.01 且都启用 0.1 的无效动作惩罚。该受控实验需要给两臂相同的奖励和更新条件，并与原 V4 的纯终局奖励分别标识。[GraphGPO WebShop 脚本](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/recipe/GraphGPO/run_qwen2.5_7b_webshop_train.sh#L35-L59) · [BiPACE-Q TextCraft 1.5B 脚本](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/examples/bipace/textcraft/qwen1.5b_4xh100/train_bipace.sh#L82-L90)

## 正式训练共同设置

两个尺寸均使用 Qwen2.5-Instruct，按各自权重全参数训练。每轮采样 16 个任务组，每组 8 条轨迹，共 128 条。以官方二值终局成功训练，奖励尺度为 10，折扣 γ=1，语法惩罚为 0。每个成败混合组作为一个 PPO 更新批次，按轨迹归一化，分母为 8。

| 参数 | 有效值 |
| --- | --- |
| AdamW 的 β₁、β₂ / ε / 权重衰减 | (0.9, 0.999) / 1e-8 / 0.01 |
| PPO epochs / 概率比裁剪 / dual clip | 1 / 0.2 / 3.0 |
| 梯度范数上限 | 1.0 |
| 训练采样温度 / 终评采样温度 | 1.0 / 0.4 |
| 训练 top-p / 终评 top-p | 1.0 / 1.0 |

参数取自各运行单元的 `config.json` 与协议默认值。可从下列入口复核基准配置和覆盖项。[ALFWorld 运行配置](../outputs/fgpo-colab-956ceab-ag-20261116/gdrive_staging/comparison/alfworld/fgpo/seed-0/config.json) · [WebShop 150 轮配置](../configs/fgpo-webshop150-host-b.json) · [WebShop 覆盖项](../src/fgpo/protocol.py) · [TextCraft 默认配置](../configs/fgpo-v4-host-b-textcraft-20261120.json) · [TextCraft 半径配置入口](../scripts/run_fgpo_host_b_textcraft_radius.sh) · [联合配置与运行记录](../controls.md)
