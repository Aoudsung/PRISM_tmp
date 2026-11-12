# PRISM DCC v2 (`prism-dcc-v2-unified-r5`) 当前版本问题诊断实验数据包

本目录收录了当前 `r1` 分支版本（`DCC_CONTRACT_VERSION = "prism-dcc-v2-unified-r5"`，及紧邻的前序消融版本 `r4`）在远程服务器 **Host A (`mgt01`)** 与 **Host B** 上真实运行产生的原始实验汇总、训练日志、测量簇记录、程序库状态、分阶段消融对比及逐题配对轨迹，用于客观反映当前版本存在的算法、实现与基准匹配问题。

---

## 1. 目录结构与数据来源

| 路径 | 来源主机与原始运行目录 | 内容说明 |
|---|---|---|
| `r12p-sota-gpu4/r12_sota_pilot_summary.json` | Host A (`mgt01`, GPU 4) `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-dcc-v2-pilot-20261111-r12p-sota-gpu4/` | 当前 `r5` 版本在 `WebShop`、`ALFWorld`、`LOCA` 三基准上的完整试点总表（含 Stage 1 核验与 Stage 2 `frozen_base` vs `prism_evolve` 配对评测） |
| `r12p-sota-gpu4/stage1_r5_checks/stage1_summary.json` | 同上 | $\phi_0$ 跨深度贝尔曼中性核验、`DynamicCache.crop` 物理 KV 复用一致性与 WebShop 搜索先验分解实测值 |
| `r12p-sota-gpu4/{webshop,alfworld,loca}/` | 同上 | 各基准原始 `training_summary.json`、`training.jsonl`、`training_costs.jsonl`、`dcc_measurement_records.json`、`dcc_proposal_runs.json`、`dcc_program_library.json`、`dcc_mechanism_ablations.json` 及 `eval_{frozen_base,prism_evolve}/eval_summary.json` |
| `r12p-sota-gpu4/paired_eval_trajectories.json` | 由同目录原始 `eval_*/traces/task-*.json` 提取（去除百万级重复 `prompt_token_ids` 数组） | 全部评测题目的逐步配对动作、模式门控标记、采样对数概率和、Token 消耗、耗时及环境观测摘要 |
| `r11p-v4-gpu7/r11_staged_summary.json` | Host A (`mgt01`, GPU 7) `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-dcc-v2-pilot-20261111-r11p-v4-gpu7/` | 7 组机制消融对照（`frozen_base`、`dcc_k0_planner`、`dcc_reset_parameters`、`dcc_no_program_growth`、`dcc_fixed_d_shuffled`、`prism_evolve_lex_1upd`、`prism_evolve`），含逐步 `prior_q_mean` vs `selected_q_mean` 覆盖明细（`action_change_examples`）与检查点等价性检验 |
| `host_b_qwen7b_baseline_reference.json` | Host B `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-main11-s42-20261109-r5..r7` 及 `formal_prism_runs/20261111-165115-r12-prism-only-gpu1` | 同一 `Qwen2.5-7B-Instruct` 基座下各无门控原生 RL/偏好基线（`t3`、`gigpo`、`step_dpo`、`infopo`、`arew`）与 `prism_evolve` 的训练终态成功率对照，用于剥离 7B 模型能力边界与 DCC v2 自身退化 |

---

## 2. 核心实验现象总览（`r12p-sota-gpu4`，`Qwen2.5-7B-Instruct`，`seed=42`）

| 基准 | 方法 | 评测成功率 | 平均效用 $U_B$ | 连续重复动作步数 (`consecutive_repeat_steps`) | 先验覆盖步数 / 总步数 (`prior_action_overrides`) | 程序归因步数 (`program_attributed_steps`) | 单题平均耗时 (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| **ALFWorld** (12 题) | `frozen_base` | **16.67%** (2/12) | **0.1661** | 42 | 0 / 0 | 0 | 22.77 |
| **ALFWorld** (12 题) | `prism_evolve` (`r5`) | **8.33%** (1/12) | **0.0823** | **171** (248 步中占 **69.0%**) | **24 / 248** | 20 | 47.32 |
| **WebShop** (12 题) | `frozen_base` | **8.33%** (1/12) | **0.0823** | 15 | 0 / 0 | 0 | 11.39 |
| **WebShop** (12 题) | `prism_evolve` (`r5`) | **0.00%** (0/12) | **0.0000** | 17 | **18 / 140** | 11 | 20.87 |
| **LOCA** (2 题) | `frozen_base` | **0.00%** (0/2) | **0.0000** | 5 | 0 / 0 | 0 | 35.44 |
| **LOCA** (2 题) | `prism_evolve` (`r5`) | **0.00%** (0/2) | **0.0000** | **8** (10 步中占 **80.0%**) | **9 / 10** (**90.0%**) | **0** | 101.89 |

---

## 3. 实验数据直接暴露的五类具体问题

### 问题一：DCC 控制器发生系统性负迁移，覆盖正确先验并引发 4.07 倍死循环爆炸
- **数据证据**：
  - `r12p-sota-gpu4/alfworld/alfworld_summary.json` 与 `paired_eval_trajectories.json`（`alfworld` 第 2 题 `look_at_obj_in_light-Bowl-None-DeskLamp-316`）：`frozen_base` 仅用 **3 步**即完成任务（`go to desk 1` $\to$ `take bowl 1 from desk 1` $\to$ `use desklamp 1`，$U_B = 0.9979$）；而 `prism_evolve` 在第 1 步覆盖先验改选 starter 程序 `examine desk 1` 后，连续执行 **21 次 `examine desk 1`** 直至耗尽 `budget:model_tokens`（124,570 tokens，$U_B = 0$）。
  - `r11p-v4-gpu7/r11_staged_summary.json` 的 `action_change_examples` 记录了同一题目的精确 Q 值覆盖现场：
    - 第 2 步：`prior_action = "take bowl 1 from desk 1"`（`prior_q_mean = 0.04637`）被 `executed_action = "examine desk 1"`（`selected_q_mean = 0.06701`，`starter` 程序）覆盖；
    - 第 3 步：`prior_action = "use desklamp 1"`（`prior_q_mean = 0.06136`）再次被 `"examine desk 1"`（`selected_q_mean = 0.06257`）覆盖。
  - 在 `stage1_existing_data_replay.alias_single_snapshot_replay` 中，**甚至在未经任何参数更新的初始参数 $\phi_0$ 下**（`dcc_k0_planner` / `dcc_reset_parameters`），规划器已直接选出 `"examine desk 1"`（`selected_q_mean = 0.06747`）。
- **代码与机制根因**：
  1. `native_policy.py` 中 `expected_utility` 模式将覆盖阈值设为 `adv_margin = 1e-7` 且关闭了历史自环过滤（`use_legacy_filter = False`）。在 `evaluation_greedy = False`（$T=0.7$）设置下，`frozen_base` 依赖随机采样跳出局部死胡同，而 DCC 对全体候选动作确定性取 $\arg\max_c [V_\phi(c) + 0.25\,\alpha_0(c)]$，直接抹杀了基础模型的采样探索并偏向 token 数更少、序列对数概率和更高的短命令（如 `examine desk 1`）。
  2. 一旦覆盖动作写入对话上下文，后续步中冻结 LLM 对该重复动作的条件先验 $\log \pi_0(a\mid h)$ 飙升至接近 $0$（`paired_eval_trajectories.json` 中第 2、4 步 `sampling_logprob_sum` 分别为 `-0.0495`、`-0.0030`），压倒 `ACTION_REPEAT_PENALTY = 0.05` 并形成“控制器覆盖 $\leftrightarrow$ 上下文自举复制”死循环。

### 问题二：终态稀疏二值回报下闭合信号缺失，Witness 发现与程序库生长完全停滞
- **数据证据**：
  - `r12p-sota-gpu4/{webshop,alfworld,loca}/training_prism_evolve/training_summary.json`、`dcc_proposal_runs.json`、`dcc_program_library.json`：
    - **WebShop**：16 条训练分支成功 0 次（`nonzero_d_real_blocks = 0`，`zero_success_throughout_training = true`）；
    - **ALFWorld**：16 条训练分支仅成功 1 次（`nonzero_d_real_blocks = 1`）；
    - **LOCA**：2 条训练分支成功 0 次（`nonzero_d_real_blocks = 0`）；
    - 三个基准全部出现：`witness_pairs_found = 0`、`unique_witness_pairs = 0`、`proposed_counts = {"witness_full": 0, "witness_delete": 0, "witness_concat": 0, "prior_fallback": 0}`、`library_size = 4`（始终停留在 4 个手工 starter 程序，**新生长程序数为 0**），且 8 个集成成员后验权重最终保持完全均匀 `final_mechanism_weights = [0.125, ..., 0.125]`。
  - `r11p-v4-gpu7/r11_staged_summary.json` 进一步证明：由于零程序生长，更新后的 `prism_evolve`、`dcc_no_program_growth`、`dcc_fixed_d_shuffled` 三个方法退化为**逐比特完全等价的别名**（`phi_T_aliases_strictly_identical = true`，`selected_q_mean = 0.06357990950345993` 完全相同）。
  - 同时，即便 `d_real = 0`，`K_phi` 仍在 8 个训练块中执行了 32 步非零梯度更新（WebShop `parameter_delta_l2 = 10.17`，ALFWorld `8.96`），在缺乏真实效用差信号时破坏了 $\phi_0$ 的贝尔曼中性并引入未校准估值噪声。

### 问题三：开放参数动作先验分解偏差与绑定器缺陷（WebShop 与 LOCA）
- **数据证据**：
  - **WebShop 搜索先验失衡**（`r12p-sota-gpu4/stage1_r5_checks/stage1_summary.json`）：`extract_role_bindings` 提取的 `goal_query` 为 17 词拼接串 `"home office furniture sets dining room living room color blue gray shape round size ft in"`（含残缺单位词 `ft in`），其冻结先验对数概率为 **`-6.99159`**，而离散按钮 `"click[search]"` 为 **`-0.01816`**（中心化优势 `-6.96691` vs `+0.00652`），导致开放查询候选在先验项上天然处于巨大劣势。评测中 `frozen_base` 第 11 题成功（1/12），`prism_evolve` 发生 18 次先验覆盖后降至 0/12。
  - **LOCA 合成默认参数塌缩**（`r12p-sota-gpu4/loca/eval_prism_evolve/eval_summary.json`）：面对开放 JSON 工具参数空间，`enumerate_admissible_actions` 构造的短默认参数调用 `{"arguments":{"course_id":1.0,"question_id":1.0,"quiz_id":1.0},"name":"canvas_canvas_update_quiz_question"}` 在 2 道测试题（`seed_20001`、`seed_20002`）的共 10 步中被 DCC 覆盖选中 **9 次**（`program_attributed_steps = 0`），连续 5 步重复向无关测验写入哑参数 `1.0` 直至耗尽 Token 预算。

### 问题四：公开字段与四槽离散状态编码过于粗糙，且存在字段提取缺陷
- **数据证据**：
  - 在 `r11p-v4-gpu7/r11_staged_summary.json` 的 ALFWorld 覆盖记录中，反复执行 `examine desk 1` 前后，`public_fields` 始终保持 `{"open_receptacle_state": "unknown", "inventory_object": "none", "object_treatment_state": "raw", "target_object_visible": "yes"}` 不变，四槽状态 `encode_typed_field_slots` 无法区分“未检查容器”与“已检查容器但未拾取目标物品”。
  - 在 `r12p-sota-gpu4/loca/training_prism_evolve/dcc_measurement_records.json` 与 `r11_staged_summary.json` 中，LOCA 的 `last_tool_family` 字段值为 `'{"arguments":{"account_id":1.0},'`（`native_dcc.py` 直接截取了 JSON 原文前 32 个字符 `act_text[:32]`，未解析工具 `"name"` 字段）。

### 问题五：7B 基座能力与基准任务难度的客观边界对照
- **数据证据**（`host_b_qwen7b_baseline_reference.json`）：
  - **WebShop 与 ALFWorld 并非 7B 模型不可解**：在相同 `Qwen2.5-7B-Instruct` 基座与 `prism-native-text-v1` 接口下，无 DCC 门控的原生 RL 基线在 **WebShop** 上达到 `14.29%`（`t3`，15/105）、`13.75%`（`gigpo`，11/80）、`10.34%`（`step_dpo`，21/203），在 **ALFWorld** 上 `t3` 达到 `10.00%`（15/150），而同期 Host B 上 `prism_evolve`（DCC v2 `r7`）在 ALFWorld 351 条轨迹中仅 `1.71%`（6/351）。因此这两个基准上的低分主要源于上述 DCC v2 控制覆盖与零闭合问题，而非 7B 参数规模完全失效。
  - **LOCA 与 AppWorld/PlanBench-XL 存在显著的 7B 规模与预算瓶颈**：在 **LOCA** 上，单步工具目录与观察使提示长度达 20K–27K tokens（`131,072` token 预算仅够执行 5 步），Host B 全部 7B 方法（`gigpo`、`infopo`、`step_dpo`、`arew`、`t3` 共 143 条训练轨迹）及 Host A 的 `frozen_base`、`prism_evolve` 终态成功率均为 **0.0%**；Host B 上 `appworld` 与 `planbench_xl`（`formal_r12`）的 `R / W / P` 三臂评测成功率亦全部为 **0.0%**。

*(注：`scripts/run_dcc_v2_r12_sota_pilot.py` 的 `_summarize_train_jsonl` 读取键名为 `dcc_side_loss`，而 `training.jsonl` 实际记录键名为 `loss` 与 `final_replay_loss`，因此汇总表中的 `first_3_losses` 为空列表，真实逐步损失完整保存在各基准 `training_prism_evolve/training.jsonl` 中。)*
