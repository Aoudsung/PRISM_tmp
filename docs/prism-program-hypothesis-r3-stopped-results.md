partial — 已按用户中途停止指令终止 r3，并整理全部已落盘结果；未补跑。

# PRISM program-hypothesis-r3 已完成实验与停止报告

## 实际停止状态

工作分支为 `program-hypothesis-r3`，起点为 `629b7f0`。R2 源码、30 条原始训练轨迹与失败结论保持原样。本轮没有追加采集、单元测试、独立审计或停止后的模型调用。

已精确认领并停止 mgt01 上的本项目主进程 **1191383**。该次 nohup 启动继承了忽略 SIGINT 的信号配置，随后仅对这个已确认 PID 执行 STOP → TERM → CONT，启动壳记录的实际退出状态为 **143**；主进程及其已确认的 LOCA 直属 worker **1894468** 均已退出。停止后仅执行既有记录的汇总。汇总落盘时间为 **2026-11-12 13:49:44 UTC**。

精确停止点是 **`source_repair-loca-base-027-007-R`**，目标为 `loca/8k/CanvasArrangeExamS2LEnv/seed_10009`。该项已保存 **4 个真实原生动作及回执**，入口和局部锚点两个快照仍在输出目录。下一次 `shared.actor.tool` 调用 **`e7ea45c564954e24b18e3fdaac86fc7c`** 已开始，尚未闭合；其最终输出 token、完成状态和完整调用费用保持未知。该项不计入完成数，也不赋予终局成功。

正式计划为 108 个 B/R/P 试验，已有 **93 个完整 TrialRecord**，另外 **1 个在途项被停止、14 个尚未开始**。93 个完整记录包含 ALFWorld 36 个、WebShop 36 个、LOCA 21 个。

## 实现与协议入口

新增 `src/prism_evolve/program_hypothesis_r3.py`，实现完整 RepairContext、去重文档工作视图、公开短目录、按实际引用闭合来源、ProgramProposal、受限 IR 验证、效果分类和独立研究租约。新增 `src/prism_evolve/program_hypothesis_runtime_r3.py`，实现共享的工具名先行/选中原生 schema 参数生成、当前公开状态绑定、SemanticBatch、Frontier、带 fuel 的 Loop、If3 的 T/F/U 分支、真实原生执行、fallback、快照和异常恢复/重试记录。新增统一运行器、r3 配置及协议说明。

全部 30 集提案完成后才进入统一 B/R/P。B 使用共享接口的冻结基座，R 使用同一完整源经验直接重规划，P 在无有效程序时使用同经验冻结 fallback。源题事后修复、native dev 留出目标与 native test 正式新目标分别登记。运行中没有手工修复模型提案、放宽来源/类型检查或补充目标答案。

## 提案结果

30 条既有完整训练轨迹均已处理，共发生 **39 次真实冻结 Qwen2.5-7B 提案调用**，每集均未超过两次，总数低于 60 次研究预算。**通过 0，拒绝 30，可执行程序 0**。这些调用属于独立研究预算；账本一直记录 `online_rho_satisfied=false`。

| 拒绝类别 | 集数 |
|---|---:|
| 输出结构与受限 IR 不匹配，产生 `string indices must be integers` | 16 |
| 证据 span 超出原文 | 5 |
| 生成截断或未完整结束 | 3 |
| JSON 语法错误 | 2 |
| If3 缺少合法运行时谓词及 T/F/U 分支 | 1 |
| 不存在的文档 ID | 1 |
| 缺少所引用的 `content` 字段 | 1 |
| Call/Frontier 数据叶使用来源字面值而非当前变量 | 1 |
| **合计** | **30** |

提案已知输入为 **289,248 token**，已知输出为 **36,261 token**，已闭合提案调用的实测总墙钟为 **1,470.51 秒**。原始模型输出、prompt、token ID、拒绝原因及调用身份全部保留。

## B/R/P 完成结果

下表仅包含完整落盘的 93 个试验。Token 为输入与输出的真实计费和；墙钟采用已落盘 `trial_outer` 实测区间，包含该项 reset、快照、模型、环境和局部执行工作。`native_terminal` 包含环境自身的步数上限出口；`model_tokens` 表示触及共同模型预算。非成功或未知终局的 Y 按协议为零，没有表面进展奖励。

### 源题事后修复：81 个完成项

| 基准 | 臂 | 完成 | ΣY | Token | 墙钟/秒 | 环境步 | 终止 |
|---|---|---:|---:|---:|---:|---:|---|
| ALFWorld | B | 10 | 0 | 1,436,693 | 499.43 | 500 | native_terminal ×10 |
| ALFWorld | R | 10 | 0 | 2,590,056 | 571.84 | 254 | model_tokens ×10 |
| ALFWorld | P | 10 | 0 | 2,590,056 | 572.32 | 254 | model_tokens ×10 |
| WebShop | B | 10 | 0 | 731,134 | 236.12 | 135 | native_terminal ×10 |
| WebShop | R | 10 | 1 | 1,171,805 | 265.67 | 89 | native_terminal ×10 |
| WebShop | P | 10 | 1 | 1,171,805 | 260.92 | 89 | native_terminal ×10 |
| LOCA | B | 7 | 0 | 1,765,311 | 1,132.24 | 98 | model_tokens ×7 |
| LOCA | R | 7 | 0 | 1,759,167 | 1,015.58 | 54 | model_tokens ×7 |
| LOCA | P | 7 | 0 | 1,756,336 | 1,012.76 | 54 | model_tokens ×7 |

### 开发留出目标：6 个完成项

配置中的 `development_leave_one_out` 实际采用 **native dev 中、身份不属于 30 个训练源题的新目标**，经验也不包含该目标。这里登记实际执行口径。

| 基准 | 臂 | 完成 | ΣY | Token | 墙钟/秒 | 环境步 | 终止 |
|---|---|---:|---:|---:|---:|---:|---|
| ALFWorld | B | 1 | 0 | 142,350 | 49.29 | 50 | native_terminal |
| ALFWorld | R | 1 | 0 | 258,984 | 56.76 | 24 | model_tokens |
| ALFWorld | P | 1 | 0 | 258,984 | 56.73 | 24 | model_tokens |
| WebShop | B | 1 | 0 | 107,639 | 26.03 | 15 | native_terminal |
| WebShop | R | 1 | 0 | 68,116 | 14.48 | 3 | native_terminal |
| WebShop | P | 1 | 0 | 68,116 | 14.60 | 3 | native_terminal |
| LOCA | B/R/P 各臂 | 0 | — | — | — | — | 未开始 |

### 正式新目标：6 个完成项

| 基准 | 臂 | 完成 | ΣY | Token | 墙钟/秒 | 环境步 | 终止 |
|---|---|---:|---:|---:|---:|---:|---|
| ALFWorld | B | 1 | 0 | 138,267 | 49.28 | 50 | native_terminal |
| ALFWorld | R | 1 | 0 | 259,359 | 57.18 | 26 | model_tokens |
| ALFWorld | P | 1 | 0 | 259,359 | 57.15 | 26 | model_tokens |
| WebShop | B | 1 | 0 | 71,482 | 20.31 | 15 | native_terminal |
| WebShop | R | 1 | 0 | 166,064 | 36.57 | 7 | native_terminal |
| WebShop | P | 1 | 0 | 166,064 | 38.27 | 7 | native_terminal |
| LOCA | B/R/P 各臂 | 0 | — | — | — | — | 未开始 |

ALFWorld 开发目标为 `valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790/game.tw-pddl`，正式目标为 `valid_unseen/look_at_obj_in_light-AlarmClock-None-DeskLamp-308/trial_T20190908_222917_366542/game.tw-pddl`。WebShop 开发目标为 `webshop/1000`，正式目标为 `webshop/0`。LOCA 的开发与正式新目标均尚未执行，身份分别为 `loca/8k/ABTestingS2LEnv/seed_20001` 与 `loca/8k/ABTestingS2LEnv/seed_123`。

### 完成项总账与程序执行比例

| 臂 | 完成 | ΣY | Token | 墙钟/秒 | 环境步 | 真实程序动作 | P fallback 动作 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 31 | 0 | 4,392,876 | 2,012.71 | 863 | 0 | — |
| R | 31 | 1 | 6,273,551 | 2,018.07 | 457 | 0 | — |
| P | 31 | 1 | 6,270,720 | 2,012.76 | 457 | 0 | 457 |
| **合计** | **93** | **2** | **16,937,147** | **6,043.54** | **1,777** | **0** | **457** |

P 的 **31/31 个完整试验均为 `NO_REUSABLE_PROGRAM` 后的同经验 fallback**。真实程序试验比例为 **0/31**；原生动作层面为 **0/457 个程序动作、457/457 个 fallback 动作**。本轮没有产生可交付的真实程序绑定执行轨迹；已交付的当前状态真实执行均来自 B/R 和 P 的 fallback。

## 共同接口、直接重规划与程序机制结论

31 组已完成 B/R/P 的共同初始策略输入检查全部相等，即 **31/31**。持久记录比较了实际 reset 后公共目标、观察、公开工具索引、种子、同一个冻结 actor 入口和采样配置；没有增加首动作 probe。加入经验后的 R/P 则按注册的同经验协议运行。

共享接口的历史比较使用原有 R2 冻结基座源题记录；LOCA 下表仅取当前已完成的相同 7 个源 episode。历史比较与同轮程序机制对照分别报告。

| 基准/同源题数 | R2 B：ΣY / Token / 墙钟秒 | R3 共享接口 B：ΣY / Token / 墙钟秒 |
|---|---|---|
| ALFWorld / 10 | 0 / 2,550,616 / 576.78 | 0 / 1,436,693 / 499.43 |
| WebShop / 10 | 0 / 609,694 / 199.51 | 0 / 731,134 / 236.12 |
| LOCA / 7 | 0 / 1,238,277 / 902.44 | 0 / 1,765,311 / 1,132.24 |

共享接口在 ALFWorld 的这些历史同源题上减少了 token 和墙钟；WebShop 与已完成 LOCA 子集的成本增加。三个基准均未出现 B 的成功数提升。

同经验直接重规划在 **WebShop 源题 `webshop/1516`** 获得一次原生成功：`source_repair-webshop-base-018-008-R` 与对应 P 各执行 4 步、54,890 token，原生 `WebShop.WebAgentTextEnv.get_reward` 得分均为 1.0，B 同题失败。P 的该成功全部来自与 R 相同的 fallback，归为同经验重规划收益。开发与正式新目标已完成项的 B/R/P 成功数均为零。

程序机制本轮的观测结果为 **0/30 有效程序、0 次真实程序执行、P−R 成功数增量为 0**。R/P 实测墙钟和少量 token 的差异保留在真实账中，不归为程序收益。源题一次重规划成功、接口成本变化与程序机制结果分别保留。

## 全部已发生费用与在途状态

物理账本包含 **3,621 个 call_started、3,620 个 call_finished**。其中已闭合提案调用 39 次，`shared.actor.tool` 已闭合 1,800 次，`shared.actor.arguments` 已闭合 1,781 次。原生动作已记录 1,781 步，包含完整试验的 1,777 步与在途 R 的 4 步。

| 已落盘费用分项 | 实测值 |
|---|---:|
| 已知输入 token | 17,321,000 |
| 已知输出 token | 66,041 |
| 已知模型 token 合计 | 17,387,041 |
| 完整试验 token | 16,937,147 |
| 提案已知 token | 325,509 |
| 在途试验已闭合部分 token | 124,385 |
| 已闭合物理调用总墙钟 | 5,100.70 秒 |
| 可用 prefill GPU 时间 | 2,861.37 秒 |
| 可用 decode GPU 时间 | 1,826.61 秒 |
| 原生动作环境时间 | 534.23 秒 |
| reset 外层分项，含在途项 | 1,659.21 秒 |
| 快照导出分项 | 72.40 秒 |
| 快照序列化分项 | 1.06 秒 |
| 公共工作视图投影 | 8.20 秒 |
| schema contract 实际哈希/准备 | 1.88 秒 |
| 原生终局评价 | 0.75 秒 |
| 模型 setup，两次实际装载合计 | 72.43 秒 |
| 环境 setup | 3.74 秒 |
| 完整试验 trial_outer 合计 | 6,043.54 秒 |

内层模型、环境、reset、快照与 trial_outer 存在包含关系，**不能直接相加**。已知 token 总数保留为已知部分，另外保留两处未知状态。

第一处为测量路径故障恢复时中断的提案调用 `8d1f39318c6645ec8ee15e50686c75e5`：输入 3,513 token 已知，输出 usage 未知。原来挂在 PEFT 外层的 forward hook 未观察到实际 generate；这之前 **11 次调用的原始零相位分项保持原样并标记不可用**，没有回填。经批准把 hook 改到 `get_base_model()` 的实际 generate 路径后继续未完成工作；已完成 episode 没有重新提案。

第二处为此次用户停止时未闭合的 `e7ea45c564954e24b18e3fdaac86fc7c`，所属 `source_repair-loca-base-027-007-R`。其在途输出、最终 usage 和完整调用墙钟未知，不补成零。该项 trial 租约 `348e6111897243089103e177a7c2d29c` 与当前工具名调用租约 `42955bb5983449a5816b9d2984d127d3` 的原始 RESERVED 记录保留；分别申请的上界为 100 步 / 262,144 token / 1,800 秒，以及 17,768 token / 1,692.97 秒。预留上界没有计成实际消费，未开始的后续工作也没有计费。

## 未完成清单与恢复点

未完成的 **15 个项目全部属于 LOCA**。`loca-base-027` 对应源题三臂尚未完成，其中 R 为已保存 4 步后的在途停止项，P/B 未开始；`loca-base-028` 三臂未开始；`loca-base-029` 三臂未开始；开发留出新目标三臂未开始；正式新目标三臂未开始。精确 ID、目标身份、顺序、状态及快照路径见 `stopped_summary.json` 的 `missing` 数组。

在途恢复记录位于 `trial_attempts/source_repair-loca-base-027-007-R.json`、`trial_events/source_repair-loca-base-027-007-R.jsonl`、`snapshots/source_repair-loca-base-027-007-R/0.json` 和 `1.json`。源轨迹、完整 RepairContext、39 次提案调用身份、30 份拒绝结果以及已发生费用均保持原样。没有自动恢复或继续执行。

## 产物目录、命令与退出状态

服务器输出根目录为：

`/apps/users/adminuser/prism-exact-reproduction/outputs/prism_gate0/prism-program-hypothesis-20261112-r3/`

本次停止汇总为该目录内的 **`stopped_summary.json`**。93 份完整结果在 `trial_records/`，在途实际动作在 `trial_events/`，原始 7B 生成在 `raw_generations.jsonl`，物理成本在 `physical_events.jsonl`，外层成本在 `outer_costs.jsonl`，租约在 `leases.jsonl`。`repair_contexts/`、`program_proposals/`、`source_traces/`、`snapshots/` 和 `trial_plans/` 均保留。目录中的 `r3_report.json` 是首次测量修复中断时的早期快照；此次主动终止没有生成运行器的 `r3_final_report.json`，本报告以实际完整记录及 `stopped_summary.json` 为准。

实际正式入口，主机均为 mgt01，目录均为 `/apps/users/adminuser/prism-exact-reproduction`：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv-prism-native/bin/python -u scripts/run_program_hypothesis_r3.py \
  --config configs/prism-program-hypothesis-20261112-r3.json
# 首次测量路径故障恢复：SIGINT 后实际退出 1，原始结果完整保留。

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv-prism-native/bin/python -u scripts/run_program_hypothesis_r3.py \
  --config configs/prism-program-hypothesis-20261112-r3.json --resume
# 本次用户要求停止：实际退出 143；未补跑。
```

命令、日志和退出状态分别保存在 `outputs/prism_gate0/r3-operation/command.txt`、`run.log`、`exit-status.txt`、`resume-command.txt`、`resume.log` 和 `resume-exit-status.txt`。停止后仅用服务器 Python 标准库读取既有 JSON/JSONL 生成停止汇总，命令退出 0；没有加载模型或调用环境。

Changed files：`src/prism_evolve/program_hypothesis_r3.py`、`src/prism_evolve/program_hypothesis_runtime_r3.py`、`scripts/run_program_hypothesis_r3.py`、`configs/prism-program-hypothesis-20261112-r3.json`、`docs/prism-program-hypothesis-r3.md`。它们均为独立新增文件；未触碰原有论文等其他未提交修改。`git diff --cached --name-only` 输出为空，无暂存文件。测试新增/更新为空；本地未执行项目代码。

下一步为主代理验收本报告与现有产物；当前保持停止。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "已实现独立r3五个新增文件，并按最新停止指令终止确认PID1191383、实际退出143；仅汇总93个完整试验，未补跑、未修改r2。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "报告提供39次真实提案、30次拒绝、93/108完成试验的分基准分scope分臂Y/token/墙钟/终止、31/31初始输入等价、0次程序执行、15项未完成清单、原始账路径及两处未知usage。"
    }
  ],
  "changedFiles": [
    "src/prism_evolve/program_hypothesis_r3.py",
    "src/prism_evolve/program_hypothesis_runtime_r3.py",
    "scripts/run_program_hypothesis_r3.py",
    "configs/prism-program-hypothesis-20261112-r3.json",
    "docs/prism-program-hypothesis-r3.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "mgt01: CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv-prism-native/bin/python -u scripts/run_program_hypothesis_r3.py --config configs/prism-program-hypothesis-20261112-r3.json",
      "result": "failed",
      "summary": "测量路径故障恢复时SIGINT终止，实际退出1；此前调用、拒绝及未知usage原样保留。"
    },
    {
      "command": "mgt01: CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=src .venv-prism-native/bin/python -u scripts/run_program_hypothesis_r3.py --config configs/prism-program-hypothesis-20261112-r3.json --resume",
      "result": "failed",
      "summary": "只续未完成工作；完成全部30集提案及93个试验，按用户停止要求退出143。"
    },
    {
      "command": "mgt01: kill -STOP 1191383; kill -TERM 1191383; kill -CONT 1191383",
      "result": "passed",
      "summary": "已确认本项目PID及直属worker退出，resume-exit-status.txt为143；未继续调用或试验。"
    },
    {
      "command": "mgt01: Python标准库读取既有trial_records/program_proposals/physical_events/outer_costs/leases/trial_plans，生成stopped_summary.json",
      "result": "passed",
      "summary": "仅整理已落盘结果：93个完整试验，39次提案调用，30/30拒绝，15个未完成项。"
    },
    {
      "command": "git diff --cached --name-only",
      "result": "passed",
      "summary": "输出为空，无暂存文件。"
    }
  ],
  "validationOutput": [
    "93个完整试验：B/R/P各31，成功数0/1/1，完整试验token16937147，墙钟6043.54秒。",
    "31/31组共同初始输入等价；P的31/31试验全部fallback，457/457个原生动作均非程序动作。",
    "物理调用started3621、finished3620；已知总token17387041。",
    "未运行单元测试、独立审计或本地项目代码。"
  ],
  "residualRisks": [
    "当前用户主动停止，15个试验未完成，包含1个保存4步后的在途R试验。",
    "当前1个未闭合模型调用及历史1个中断调用输出usage未知，11次历史prefill/decode分项不可用；均未回填。",
    "30/30提案被拒，真实程序执行为0，本轮程序绑定与控制执行没有获得真实候选运行证据。",
    "主代理验收及要求的review gate尚未完成。"
  ],
  "noStagedFiles": true,
  "diffSummary": "五个独立新增文件实现r3公共工作视图、受限假设IR、共享接口、真实运行器、成本及恢复入口；r2与其失败结论保持原样。",
  "reviewFindings": [
    "本轮没有单独启动评审；现有证据明确记录提案全拒绝、程序执行0及主动停止的部分完成状态。"
  ],
  "manualNotes": "最新指令为停止并整理已有实验。本报告按该指令交付，保持停止，不自动恢复。"
}
```
