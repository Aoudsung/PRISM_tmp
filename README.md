# PRISM-Evolve / DCC v2

当前活动实现为 **`prism-dcc-v2-unified-r5`**，活动基准为 ALFWorld、WebShop 与 LOCA。设计依据是 [DCC v2 设计文档](docs/prism-dcc-v2-design.md)，实验口径见 [原生实验协议](docs/native-experiment-protocol.md)，当前研究状态见服务器生成的只读 [RESEARCH_STATE](docs/RESEARCH_STATE.md)。旧 PRISM-Evolve 定义保留在 [历史设计](docs/prism-evolve-design.md)，不与 DCC v2 混为同一方法。

## 当前状态与证据

- `35354b9` 保存了 r5 核心实现及 r11/r12 试点证据；`cf80b60` 开始三端目录清理与资产索引。
- 正式 DCC 运行仍未启动。本次整理不启动训练、评测、模型服务或自动续跑。
- 当前 pilot 显示退化，不能称为 SOTA 结果：ALFWorld 为 Frozen Base **2/12**、PRISM **1/12**；WebShop **1/12 → 0/12**；LOCA 两者均 **0/2**。
- [诊断数据包](results/dcc-v2-r5-pilot-diagnostics-20261111/README.md) 保留原始摘要、训练日志、程序库与配对轨迹摘录。它是小样本开发诊断，不是完整正式测试。README 中的根因解释应结合原始证据审阅，不能把梯度更新等同于性能提升，也不能仅凭零成功断言模型规模或方法不可能有效。
- `tmp_pro_audit_bundle.md` 是本地临时审计产物，保持不变且不纳入 Git；它不是新的实验结果来源。

## 活动目录

| 路径 | 用途 |
|---|---|
| `src/prism_evolve/` | 当前实现、原生环境桥接、基线、调度与统计；旧公共接口仍可能被调用，不按文件日期删除 |
| `configs/prism-native-host-a.json`、`prism-native-host-b.json` | 两台服务器的环境与资源路径 |
| `configs/prism-native-qwen7b-dcc-v2-20261110-r10.json` | 正式计划及已登记训练资产；不表示已获准启动或已跑完 |
| `configs/prism-native-qwen7b-dcc-v2-pilot-20261110-r10p.json` | 当前 pilot 计划配置；r12 专用脚本内的短诊断范围另行声明 |
| `scripts/run_prism_native_experiments.py` | 活动调度入口；`--prepare` / `--pipeline` 必须显式指定 `--config` |
| `scripts/collect_prism_native_experiments.py` | 活动结果收集入口 |
| `scripts/run_dcc_v2_r12_sota_pilot.py` | r12 pilot，文件名中的 `sota` 不是性能结论 |
| `scripts/monitor_prism_native.py`、`relay_prism_host_b.py` | 按实际 run 身份监控与协调；不恢复已删除的旧目录 |
| `scripts/run_formal_experiment.py` | 保留的旧 native-r3 R/W/P 兼容入口，不是 DCC v2 正式入口；历史配置仅在服务器 archive/Git 历史中 |
| `results/dcc-v2-r5-pilot-diagnostics-20261111/` | 当前可版本化的精简证据包 |
| `dcc_research_program/` | 有限合成模型研究；其成功不代替原生基准证据 |
| `paper/acl2027/` | 论文与图表；本次不修改、不提交其中原有未提交改动 |

## 双机资产位置与清理边界

- **Host A / mgt01**：`/apps/users/adminuser/prism-exact-reproduction`。
  - 保留 `.research/research.sqlite` 单写主库、`.git-remote-r1.git`、`.venv-prism-native`、外部模型与环境。
  - `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-dcc-v2-pilot-20261111-r12p-sota-gpu4/` 保留完整原始轨迹、成本与摘要；pilot adapter 副本已清理，不能承诺从该目录重载原 checkpoint。
  - r11 的精简消融证据保存在上述 Git 诊断数据包；旧大体积运行目录已按用户授权清理。
- **Host B**：`/home/chenxiangwei/workspace/long-horizon-decision`。
  - 统一资产入口：`outputs/reusable_baselines/qwen25_7b_s42/manifest.json`。
  - `training_cells/` 索引 **15 个训练阶段终点**（3 环境 × GiGPO、Step-DPO、InfoPO、AREW、T3）。部分为零更新或预算截断的终点，不能称为全部有效学习或可直接续训；没有声称恢复了优化器状态。
  - `completed_eval_cells/alfworld_test_134/` 索引 **8 个无门控 baseline 的 134 题完整历史评测**（5 个训练基线、WorldEvolver、EnvProbe-Simple、EnvProbe-Judge），并保留真实轨迹、成本与 WorldEvolver 题后记忆。
  - `partial_eval_cells/webshop_test/` 保留 **9 个方法的部分评测**；不标记为完成、不与新实验拼接。
  - `historical_eval_cells/` 保留旧门控 Frozen Base 的 134 题记录。其 first3 初始 checkpoint 已在清理旧目录时删除，不能作为当前无门控 Frozen Base 策略复用。
  - 统一入口使用符号链接，真实文件仍位于原 `main11-r5/r6/r7` 与 `dcc-v2-r7` 的 `executions/` 下。**这些目录不是待删垃圾**：manifest、schedule、checkpoint 与记忆绝对路径依赖它们。活动配置保留原位路径以满足 `_train_cell` 的来源校验。
  - `completed_evaluation_imports` 仅登记历史资产，`reuse_completed_paired_evaluations=false`；本次不改变评价顺序、GPU、方法身份或自动放宽复用条件。
- **历史配置与脚本**：两台服务器的 `archive/configs-and-scripts-20261112/` 保存已归档的 17 个本地旧配置及 5 个一次性脚本；本地不存归档副本。旧版本也可从 Git 历史（如 `35354b9`）读取。
- **本地**：只保留源码、配置、精简证据、论文改动与 `.external/pos-official` 静态参考；旧 `.external/pbxl-repo`、大体积实验副本与重复审阅包已删除。服务器外部依赖与模型权重未纳入本次删除。

低分、失败与中断本身不是“无用数据”；本轮按用户授权清理已淘汰的大产物，同时保留当前诊断证据与明确的可复用基线资产。被清理的原始数据不能再从路径读取，Git 只保存已提交文件，不是全部实验文件的备份。

## 方法与基准口径

`action_contract_version=prism-native-text-v1`。DCC 的冻结底模先验、响应模型、AST 程序与控制行为以 DCC 设计及实际配置为准；原生 RL baseline 不使用 PRISM 门控。AppWorld、PlanBench-XL 与 ClinDiag 不在活动清单，相关公共源码暂保留，不表示旧大体积结果仍在。

| 基准 | 正式划分 | 正式动作上限 | 原生成功判定 |
|---|---|---:|---|
| ALFWorld | train / valid_seen / valid_unseen 全部 134 测试题 | 50 | TextWorld `won` |
| WebShop | 固定 seed=42 打乱后 train `[1500,6910)` / dev `[500,1500)` / test `[0,500)` | 15 | 原生终止且购物分数为 1 |
| LOCA | 15 模板，train 300 / dev 75；官方五种子 test 75 | 100 | 原生 `claim_done` 终止且评分不小于 1 |

共同任务效用为 `U_B = Y * max(0, 1 - C_seconds / B_seconds)`。r12 pilot 使用 ALFWorld 25 步、WebShop 15 步、LOCA 6 步的短诊断帽，不能冒充上表的完整正式实验。

WebShop 保持 `use_small=True`、`human_goals=False`；LOCA 的 8K 是环境规模标签，不是模型上下文上限。固定来源为 PoS `6818cfa6434dbc91fcf67f0b3eb12a91db47433f`、GiGPO/verl-agent `20bd331bdbc9026a5668e11362178e10ab7400c8`；PoS 内置 LOCA 来自 `hkust-nlp/LOCA-bench@8b6fac49d9edd92922593e703b74ea255357c3ec`。项目统一回报、划分与迁移实现单独披露，不冒充完整官方论文复现。

## 执行与研究记录工具

本地仅文本编辑、版本管理和文件传输，禁止本地运行项目、测试、格式化或检查工具链。所有执行在指定远程环境中进行；连接见私有 `SSH_Document.md`，不提交凭据。

- Host A native Python：`.venv-prism-native/bin/python`。
- Host B native Python：`/home/chenxiangwei/env/prism-native-v1/bin/python`。
- 研究记录只在 mgt01 更新：`PYTHONPATH=src .venv/bin/python -m prism_knowledge --db "$PWD/.research/research.sqlite" status`；任务上下文用 `context --task R5.1`。
- 新进度用 `apply` 更新既有任务并提供 `expected_revision`；本地 `.research/` 仅为服务器一致性快照。`docs/RESEARCH_STATE.md` 自动生成，不手改。
- 历史自动续跑、每小时检查与旧队列描述不构成当前清理任务的新实验授权。重新启动须另行明确安排范围与新 run 身份。
