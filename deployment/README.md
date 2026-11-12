# 当前 DCC v2 双机部署与保留资产

本轮仅整理文件、保留资产和修复归档后的入口引用；**不启动或恢复实验**。历史 AppWorld / PlanBench-XL 短训练计划不再是活动部署说明，旧说明保存在服务器 `archive/configs-and-scripts-20261112/pre-cleanup-local-docs/`。

## 主机与环境

| 主机 | 项目根 | native Python |
|---|---|---|
| Host A (`mgt01`) | `/apps/users/adminuser/prism-exact-reproduction` | `.venv-prism-native/bin/python` |
| Host B | `/home/chenxiangwei/workspace/long-horizon-decision` | `/home/chenxiangwei/env/prism-native-v1/bin/python` |

Host A 原有运行库设置需要时沿用：

```bash
export LD_PRELOAD=/apps/users/adminuser/miniconda3/envs/chronomop/lib/libstdc++.so.6
```

服务器模型、外部环境、LOCA QEMU 基础依赖及 native Python 环境未纳入清理。服务器已有代码可能含研究工具与早期模块，不以本地精简目录执行 `rsync --delete`。本地仅源码文本操作与文件传输。

## 活动配置与入口

仅四个活动配置：

- `configs/prism-native-host-a.json`
- `configs/prism-native-host-b.json`
- `configs/prism-native-qwen7b-dcc-v2-20261110-r10.json`
- `configs/prism-native-qwen7b-dcc-v2-pilot-20261110-r10p.json`

当前调度入口是 `scripts/run_prism_native_experiments.py`；创建计划或启动 pipeline 必须显式指定 `--config`，不再使用已归档的 `prism-native-v1.json` 缺省值。收集入口为 `scripts/collect_prism_native_experiments.py`。`run_dcc_v2_r12_sota_pilot.py` 是独立短 pilot，其实际范围以脚本及运行 manifest 为准。

`run_formal_experiment.py` 是旧 native-r3 R/W/P 兼容入口，不能用于 DCC v2 正式计划。历史配置与一次性脚本只在两台服务器的 `archive/configs-and-scripts-20261112/` 或旧 Git commit 中查阅，本地不保存 archive。

## 资产与复用

Host B `outputs/reusable_baselines/qwen25_7b_s42/manifest.json` 是统一索引：15 个训练终点、8 组无门控 ALFWorld 全量历史评测、9 组 WebShop 部分评测、1 组旧门控 Frozen Base 历史记录。训练终点存在不意味着非零更新、性能改善或当前实现的完整兼容性已验证。

索引目录使用符号链接；原 `outputs/prism_evolve_pos_benchmarks/` 中留下的 `main11-r5/r6/r7` 与 `dcc-v2-r7` 源单元必须原位保留。`training_phase.json`、manifest/schedule、WorldEvolver 记忆与 checkpoint 中存在绝对路径，不为美观移动或改写历史记录。旧门控 Frozen Base 的原初始 checkpoint 已删除，只保留评测证据，不再登记为当前策略复用。

当前正式配置登记全部 15 个训练来源与 8 个 ALFWorld 历史评测来源。`reuse_completed_paired_evaluations=false` 保持；未来复用必须继续满足原动作接口、任务、预算、方法顺序与 GPU 等校验，不自动启用或放宽。

Host A 保留 r12 原始轨迹、成本和摘要；原 pilot adapters 已清理。小型 r11/r12 证据包保存在 `results/dcc-v2-r5-pilot-diagnostics-20261111/` 与 Git `35354b9`。它不是可完整恢复旧训练的备份。

## 同步与记录

- 显式同步本次变更的源码、配置与文档，避免覆盖论文改动、模型、外部依赖或研究主库。
- `.research/research.sqlite` 只在 mgt01 写入；完成后生成一致性 backup、JSONL export、状态页及 `snapshot.json`，完整下载后替换本地只读副本。
- 旧 relay/finish 进程已停止，旧协调队列不恢复。新的正式实验需明确的范围、资源、新目录与独立身份。
- 不在 SSH 命令或日志输出密码、密钥、完整私有请求。认证辅助脚本与 `SSH_Document.md` 均不提交。
