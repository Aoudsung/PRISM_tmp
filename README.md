# PRISM-Evolve: Online Cross-Task Policy Evolution via Stratified Dual-Mode Counterfactual Branching

本项目实现 `PRISM-Evolve` 长时程智能体在线跨任务演化框架（设计规范见 [`docs/prism-evolve-design.md`](docs/prism-evolve-design.md)），包含单策略双模式架构（信息模式 `I` vs 执行模式 `E`）、同前缀反事实双分支分层采样、定理 5.1 目标保持部署门控概率加权策略更新、六维统一成本账本（`CostLedger`）、遮蔽反馈利用诊断（`MaskedFeedbackDiagnostic`），以及在 **AppWorld** 与 **PlanBench-XL (`retail`)** 上跨三大开源基座模型（`Qwen/Qwen2.5-7B-Instruct`、`meta-llama/Llama-3.1-8B-Instruct`、`Qwen/Qwen3.8-27B`）与 2025–2026 顶会基线的完整评测套件。

## 目录结构

- [`docs/prism-evolve-design.md`](docs/prism-evolve-design.md)：`PRISM-Evolve` 正式数学与算法设计规范。
- [`src/prism_evolve/`](src/prism_evolve)：六大核心算法模块与基准环境适配器：
  - [`event_log.py`](src/prism_evolve/event_log.py)：模块 1 — 公共事件日志与六元组观测规范化（`PublicEvent`, `PublicHistory`）。
  - [`policy.py`](src/prism_evolve/policy.py)：模块 2 — 单策略双模式接口、两阶段精确解析策略与多基座双模式 LoRA 策略（`ContextualLoRADualModePolicy`）。
  - [`collector.py`](src/prism_evolve/collector.py)：模块 3 — 同前缀反事实 `I/E` 双分支采集器（`BranchCollector`）。
  - [`utility.py`](src/prism_evolve/utility.py)：模块 4 — 统一预算效用 $U_B(\tau)$ 与六维成本账本（`CostLedger`）。
  - [`updater.py`](src/prism_evolve/updater.py)：模块 5 — 定理 5.1 目标保持门控概率加权更新器（`StratifiedUpdater`）与两阶段精确枚举核验。
  - [`evaluator.py`](src/prism_evolve/evaluator.py)：模块 6 — 跨任务演化增益评估器（`CrossTaskEvaluator`）与遮蔽反馈诊断器（`MaskedFeedbackDiagnostic`）。
  - [`peer_baselines.py`](src/prism_evolve/peer_baselines.py)：2025–2026 顶会对比方法实现（`GiGPO`, `Step-DPO`, `InfoPO`, `AREW/T3`, `WorldEvolver`, `PoS/EnvProbe`）。
  - [`environments/`](src/prism_evolve/environments)：`AppWorld`（457 APIs, 315 tasks）与 `PlanBench-XL`（`retail`, 1,665 tools, 327 queries）快照兼容环境适配器。
- [`scripts/run_prism_evolve_experiments.py`](scripts/run_prism_evolve_experiments.py)：端到端定理核验、多基座训练评测、消融分析与报告生成主入口。
- [`outputs/prism_evolve_experiments/`](outputs/prism_evolve_experiments) & [`results/prism_evolve/`](results/prism_evolve)：三大基座模型完整实验数据（JSON）与评测报告（Markdown / LaTeX）。
- [`paper/acl2027/`](paper/acl2027)：ACL 论文 LaTeX 源码工程。

## 远程服务器实验执行

所有代码运行与 GPU 特征提取均在指定远程服务器执行（连接信息见 `SSH_Document.md`）：

- **`mgt01` (`10.8.128.25:54678`)**：运行 `Qwen/Qwen2.5-7B-Instruct` 与 `meta-llama/Llama-3.1-8B-Instruct` 实验及多基座汇总。
- **大显存服务器 (`10.8.128.241:22`, `NVIDIA RTX 6000D` 85.6 GB)**：使用 GPU 1 或 2 运行 `Qwen/Qwen3.8-27B` 实验。

```bash
# 运行指定基座的完整实验并汇总报告（在远程服务器执行）
python scripts/run_prism_evolve_experiments.py --backbones Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct
```
