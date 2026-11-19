# F-GPO 实验排程

2026-11-19（洛杉矶）。[监控页](http://127.0.0.1:8765) 读取[当前逐种子表](../outputs/exp-plan-20261118/compute_schedule_current.csv)。表中“计划”表示已确定的执行顺序，是否已提交由服务器上的队列和实际进程决定。[11 月 18 日排程](fgpo_v4_experiment_execution_plan.md) 留作历史。

| 责任会话 | 卡位 | 执行顺序 |
| --- | --- | --- |
| [WebShop](thread://01a1ef8f-3bfa-7a20-912a-a094231a97d3?hostId=local) | Host B GPU 0 | F-GPO 0、2 → Obs-Only 0、2 → No-Transfer 2。 |
| 同上 | Host B GPU 2 | F-GPO 1 → Obs-Only 1 → No-Transfer 1、0。 |
| [TextCraft](thread://01a1ef71-a755-7b81-8549-482a331be888?hostId=local) | Host A L40 3+4 | F-GPO → Obs-Only → No-Support-Weight，各按种子 0、1、2。 |
| 同上 | Host A L40 1+7 | GraphGPO → G2PO → No-Transfer，各按种子 0、1、2。 |
| 现有 Colab 安排 | Colab | 保留 alfworld seed 0、1 已归档结果 → 待正式 GPU 会话接续 F-GPO seed 2 → No-Transfer seed 0–2。 |

Host B 的 9 个 WebShop 单元每种子 60 轮、每轮 16 组、每组 8 条，随后用 128 局正式终评报告满分成功率和平均 Score。现行训练与后续消融共用修复后的动作菜单和同版基础权重。每组消融在同种子完整方法终评完成后启动。现行比较方法已有公开 7B Score 和 SR，按作者各自协议引用。[Host B 执行排程](fgpo_webshop_host_b_schedule.md)

Host A 已通过 TextCraft 原生 worker 与 Ray 入口的真实验证回放。18 个正式单元每种子 50 轮，终态模型评完整 79 道验证题。GraphGPO 和 G2PO 填补 BiPACE 报告的缺项。正式启动和进度以实际作业与日志为准。[Host A 执行排程](fgpo_v4_textcraft_execution_plan.md)

两台主机的 AppWorld 已停止，检查点、结果与性能记录保留。旧 Host A WebShop seed 0 成绩及 seed 1 检查点独立保留。新增数据、模型、代码及依赖在本机下载后上传，两台主机离线配置。[WebShop 保存回执](../outputs/hosta-appworld-direct-rollout-20261118/webshop-release-for-textcraft-20261119.md) · [当前执行约束](../controls.md)
