# Host B WebShop 正式排程

2026-11-19（洛杉矶）。Host B 完成 F-GPO 三种子与两项自身消融。每个单元从同一版 Qwen2.5-7B-Instruct 基础权重起跑，使用已修复的 WebShop 动作菜单，训练 60 轮，每轮 16 组、每组 8 条轨迹。终态权重完成 128 题官方评测，逐种子报告满分 SR、平均 Score 及成本，再计算三种子的均值和样本标准差。

| Host B 物理卡 | 执行顺序 |
| --- | --- |
| GPU 0 | F-GPO 0 → F-GPO 2 → Obs-Only 0 → Obs-Only 2 → No-Transfer 2 |
| GPU 2 | F-GPO 1 → Obs-Only 1 → No-Transfer 1 → No-Transfer 0 |

两卡各运行一个进程，沿用 80 GiB 显存上限。队列接在现有 F-GPO seed 0、1 后面，以上一个单元的训练和终评完成为接续条件。消融复用同种子的 F-GPO 结果，分别量化历史匹配和续值共享对 SR 与 Score 的影响。合计 9 个单元、69,120 条训练轨迹、1,152 条终评轨迹。[逐单元队列](../configs/fgpo-v4-webshop-host-b-queue.tsv)

## WebShop 基线覆盖

原主表的 Qwen2.5、ReAct、Reflexion、PPO、RLOO、GRPO、GiGPO 和 GraphGPO 都已覆盖 7B Score 与 SR。旧扩展计划中登记的四个方法也有公开 7B 成绩，按作者口径列入文献对照。本排程新增基线训练单元数为零。[GraphGPO 表 1](https://arxiv.org/html/2605.26684v1#S5.T1) · [旧扩展计划](../outputs/exp-plan-20261118/claim_and_baseline_plan.md)

| 公开对照 | Score | SR (%) | 作者设置与出处 |
| --- | ---: | ---: | --- |
| HGPO | 88.93 ± 0.84 | 76.43 ± 1.47 | 15 步，K=2。[附录表 7](https://arxiv.org/pdf/2602.22817#page=21) |
| G2PO | 89.8 ± 1.7 | 78.3 ± 0.6 | 100 轮。[表 1](https://arxiv.org/html/2606.22995#S5.T1) |
| ECPO | 85.5 ± 0.5 | 74.7 ± 3.0 | 150 轮。[表 1](https://aclanthology.org/2026.emnlp-main.172.pdf#page=7) |
| RewardFlow | 85.3 ± 0.9 | 74.7 ± 3.0 | 100 轮，波动来自固定检查点的三个推理种子。[表 1](https://arxiv.org/pdf/2603.18859#page=8) · [附录 E.1](https://arxiv.org/pdf/2603.18859#page=30) |

RewardFlow 150 轮 7B 是旧 Tier 2B 的独立迁移项，当前排程采用上表的作者 100 轮口径。文献结果保留各自的训练预算及统计定义。[RewardFlow 附录 E.11](https://arxiv.org/pdf/2603.18859#page=34)

运行复用 Host B 已就绪的 WebShop 原生环境、索引、Java 和基础模型。新增资产先在本机下载，再上传到 Host B 离线使用。
