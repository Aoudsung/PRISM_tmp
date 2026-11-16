# F-GPO (`long-horizon-decision`) Colab A100 连接与操作手册

> 历史文档：下文属于 `fgpo-official-v1` 的部署记录，未作为 r3 执行入口更新。r3 的方案、配置和指定服务器操作以 [README](../README.md)、[现行设计](fgpo_design_and_derivation.md) 与 [AGENTS.md](../AGENTS.md) 为准；本文不构成新增服务器的执行授权。

本文档说明如何在 **Google Colab High-RAM A100（80 GB）** 上连接环境、核查资源、部署并运行当前项目 [`src/fgpo`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo) 的 **`fgpo`** 正式任务（AppWorld / ALFWorld），以及如何与 **Host B** 分工协同、回收结果并及时止损停机。

> [!IMPORTANT]
> 本手册专用于当前项目的 **F-GPO（7B 全参数训练）**，不再沿用旧 PACT/LoRA 手册中的双开并发、每 90 秒全量 `tar -czf` 备份、以及 `run_contract` / `verify_before_launch` 校验脚本（遵循 [`controls.md`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/controls.md#L11) 与 [`AGENTS.md`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/AGENTS.md)）。

---

## 1. 与旧 PACT 方案的核心差异（必读）

| 维度 | 旧 PACT (`r032-pact-rebuild`) | 当前 F-GPO (`src/fgpo`) | 对 Colab 操作的直接约束 |
| :--- | :--- | :--- | :--- |
| **训练参数量** | LoRA（单进程 ~18–23 GB VRAM） | **7B 全参数训练**（[`policy.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/policy.py#L25-L98)） | **单张 80 GB A100 只能跑 1 个 cell**，严禁同卡并发双任务 |
| **CPU 内存** | 轻量 | `OffloadedAdamW` + `WeightReference` 在 CPU 驻留 `master`/`exp_avg`/`exp_avg_sq`（FP32）与参考权重（bf16），合计约 `14 B/param`（100 GB+） | 开卡**必须拿到 `--high-mem`**，严禁降级普通内存实例 |
| **断点体积** | LoRA 检查点几十 MB | 全参数快照含 `model/`、`optimizer.pt`、`gradients.pt`、`reference.pt`、`state.pt`；`save()` 更新参数时新旧快照短暂并存 | **禁止每 90 秒全量 `tar -czf` 上传 GCS**；开卡需先确认 `/content` 磁盘余量 |
| **续跑校验** | 依赖 `run_contract.py` SHA 校验 | [`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py#L37-L99) 原生 `require_identity` + `--resume` | 固定 `/content/...` 路径即可直接 `--resume`；1 卡断点只能对 1 卡续跑 |
| **GPU 排除列表** | 无 | 服务器配置含 `excluded_gpu_ids: [0, 2]`（排除 `mgt01` 故障卡） | Colab 必须设 **`"excluded_gpu_ids": []`**，否则单卡 `--gpu 0` 会被 `isolate_gpus` 拒绝 |

---

## 2. Host B 与 Colab A100 的分工原则

当前首要目标是先完整跑通 **F-GPO 主方法**（`--methods fgpo`，2 个 benchmark × 3 个 seed = **6 个完全独立的单卡 cell**）：

- **Host B**（RTX 6000D × 3，461 GB CPU 内存，17 TB 硬盘，无回收超时）：
  - 优先承接长耗时、大评测集、需要大容量持久盘的 **`appworld`**（`horizon=100`，`test_normal=168` + `test_challenge=417`）及部分 `alfworld` cell。
- **Colab High-RAM A100**（单卡 80 GB，`--gpu 0`，临时盘与会话存在超时/回收风险）：
  - 优先承接单轮更快、评测集更轻的 **`alfworld`**（`horizon=50`，`valid_unseen=134`），争取在单次连续会话内跑完至 `final_checkpoint`（自动执行 [`reclaim_completed_resume_state`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/checkpoint.py#L202-L239) 清理大型优化器状态后，只需回传最终模型与评测日志）。
- **两条硬约束**：
  1. **显式指定方法**：[`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py#L191-L193) 外层遍历 `benchmark`、中层遍历 `method`，所有机器均需显式传入 `--methods fgpo`，不要挂带消融或基线。
  2. **独立 `--output-dir`**：同机并行或跨机拆跑时，每个独立 worker 必须使用互不重叠的 `--output-dir`，避免并发读写 `study_summary.json` 或触发 `FileExistsError`。
  3. **1 卡对 1 卡约束**：[`checkpoint.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/checkpoint.py#L40-L42) 的 `restore_rng` 严格校对 CUDA RNG 流数量，Colab（1 卡）只适合自建新 cell 或接续原本就是 **1 卡** 的 cell，不能直接接手 2 卡断点。

---

## 3. 本地连接环境与 CLI 机制

### 3.1 本地已就绪的工具与认证
- **Colab CLI**：`/Users/cxw/.local/bin/colab`（指向 `/Users/cxw/.local/share/uv/tools/google-colab-cli/bin/colab`，版本 `0.7.5`）。
  - 该入口脚本内置了两项自动处理：
    1. 若未显式设置 `HTTPS_PROXY`/`https_proxy`，自动探测 `127.0.0.1:7897` 并注入 `HTTP_PROXY` / `HTTPS_PROXY`；
    2. 若未显式传 `--auth` 且不存在 `~/.config/colab-cli/token.json`，而存在 `~/.config/gcloud/application_default_credentials.json`，自动注入 `--auth adc`。
  - > [!WARNING]
    > **不要直接执行 `colab update --install` 或 `uv tool install -U google-colab-cli`**，否则会覆盖该自定义入口脚本；若以后升级了工具，调用时需手动保证代理环境变量生效并显式加上 `colab --auth adc ...`。
- **GCP / GCS CLI**：`/Users/cxw/google-cloud-sdk/bin/gcloud`（支持 `gcloud storage`），认证文件位于 `/Users/cxw/.config/gcloud/application_default_credentials.json`。
  - **凭据安全**：严禁将 ADC JSON、OAuth token 或 `SSH_Document.md` 内容复制到脚本仓库、日志或输出中。
  - **GCS 隔离**：若复用现有 GCS 桶，建议统一放到独立前缀 `gs://<BUCKET>/fgpo-20261116/` 下，避免与历史 `r032-pact-rebuild` 数据混放。

### 3.2 常用连接与管理命令

先确保 `PATH` 包含 `~/.local/bin` 与 `~/google-cloud-sdk/bin`（若本地通过代理访问外网，按需导出代理）：

```bash
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
```

#### 1) 查看算力余额与当前会话
```bash
colab sessions
colab usage
```

#### 2) 申请 High-RAM A100 实例（1 个会话只跑 1 个 cell）
```bash
# 必须带 --high-mem；若提示不可用不要盲目用普通内存跑全参数 7B
colab new -s fgpo-alf-s0 --gpu A100 --high-mem
```

#### 3) 连接与执行命令（SSH / Exec）
- **方式 A：交互式登录 Colab（默认落入 `/content`）**
  ```bash
  colab ssh -s fgpo-alf-s0
  ```
- **方式 B：标准 `ssh` 非交互执行命令或管道传输**
  ```bash
  ssh -o ProxyCommand="colab ssh --proxy-mode -s fgpo-alf-s0" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      root@colab-runtime "nvidia-smi"
  ```
- **方式 C：配置 `~/.ssh/config` 简化日常 SSH / SCP**（可选）：
  ```sshconfig
  Host fgpo-*
      User root
      ProxyCommand /Users/cxw/.local/bin/colab ssh --proxy-mode -s %n
      StrictHostKeyChecking no
      UserKnownHostsFile /dev/null
  ```
  配置后即可直接 `ssh fgpo-alf-s0` 或 `scp local_file fgpo-alf-s0:/tmp/`。
- **方式 D：通过 Jupyter Kernel 执行 Python（不占 SSH 通道，同时刷新空闲计时器）**
  ```bash
  colab exec -s fgpo-alf-s0 -f /tmp/check_or_keepalive.py
  ```

> [!NOTE]
> **SSH 单连接限制**：Colab 每个会话**同一时刻仅允许 1 个活跃 SSH 连接**，并发发起第二个 SSH 会返回 `HTTP 429`。日常查看进度优先通过 GCS 或 `colab exec`，不要开多个长连 SSH 终端占用通道。

#### 4) 文件上传 / 下载与会话清理
```bash
# 上传 / 下载单文件
colab upload -s fgpo-alf-s0 ./local_file /content/remote_file
colab download -s fgpo-alf-s0 /content/remote_file ./local_file

# 停止释放 A100 计费
colab stop -s fgpo-alf-s0
colab sessions   # 核对已无残留会话
```

---

## 4. Colab A100 运行 F-GPO 标准操作流程

### Step 1：开卡即做硬件、CPU 内存与磁盘门禁检查
拿到 A100 会话后，第一时间通过一次 SSH 检查是否满足全参数 F-GPO 资源下限；**任一项不满足立即 `colab stop` 止损**：

```bash
ssh -o ProxyCommand="colab ssh --proxy-mode -s fgpo-alf-s0" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@colab-runtime '
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
  echo "--- CPU RAM (GB) ---"
  free -g
  echo "--- /content Disk ---"
  df -BG /content
'
```

核对标准：
1. GPU 为 `A100-SXM4-80GB`（可用显存 `>= 75 GiB`）；
2. 系统内存确认为 High-RAM（足以容纳约 100 GB+ 的 `OffloadedAdamW` / `WeightReference` CPU 状态 + 环境 worker）；
3. `/content` 剩余磁盘足以同时容纳 **15 GB 基座权重 + 虚拟环境/数据 + 单 cell 全参数快照更新峰值**。

### Step 2：0 GPU 浪费资产准备（GCS 预置冷资产 + 轻量代码包）
F-GPO 配置要求 `local_files_only: true`，且 ALFWorld 需要独立环境（`alfworld_python_executable`）。不要在 A100 计费状态下从 PyPI / HuggingFace 慢速重装：

1. **冷资产（一次性准备并存 GCS）**：预先从服务器或无 GPU 的普通 Colab 会话将以下目录打包存入 `gs://<BUCKET>/fgpo-20261116/assets/`（建议使用 `zstd` 压缩，解压远快于 `gzip`）：
   - 主训练 venv（含 `torch`、`transformers` 等）
   - ALFWorld venv（对应 `alfworld_python_executable`）
   - `Qwen2.5-7B-Instruct` 本地权重目录
   - `alfworld`（`json_2.1.1`）与 `appworld` 官方数据目录
2. **源码同步包（本地 Mac 每次改代码后只传几十 KB）**：
   macOS 打包务必加 `COPYFILE_DISABLE=1`，避免生成 `._*` 扩展属性文件：
   ```bash
   cd /Users/cxw/Documents/科研方案实现/long-horizon-decision
   COPYFILE_DISABLE=1 tar \
     --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='._*' \
     -czf /tmp/fgpo_src.tar.gz src pyproject.toml configs
   ```
   上传至 GCS（或仅在更新代码时通过 `colab upload` 传到 `/tmp/` 再覆盖 `/content/fgpo-reproduction`）：
   ```bash
   gcloud storage cp /tmp/fgpo_src.tar.gz gs://<BUCKET>/fgpo-20261116/code/fgpo_src.tar.gz
   ```
3. **VM 端 GCS 认证**：将本地 `/Users/cxw/.config/gcloud/application_default_credentials.json` 上传到远端 `/root/.gcs_adc.json`（`chmod 600 /root/.gcs_adc.json`），让 VM 内脚本自行按需刷新 OAuth access token，**不在 SSH 命令字符串中明文拼接 token**。

### Step 3：Colab 专用配置规范（`configs/fgpo-colab-a100.json`）
在 Colab 上运行需将路径全部固定为 `/content/...`，并将 `excluded_gpu_ids` 清空。配置结构对齐 [`configs/fgpo-host-b-official-20261116.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-host-b-official-20261116.json)，关键字段如下：

```json
{
  "protocol": "fgpo-official-v1",
  "install_root": "/content/fgpo-reproduction",
  "python_executable": "/content/venvs/fgpo-main/bin/python",
  "alfworld_python_executable": "/content/venvs/alfworld/bin/python",
  "model_path": "/content/models/Qwen2.5-7B-Instruct",
  "alfworld_data_root": "/content/data/alfworld",
  "appworld_data_root": "/content/data/appworld-root",
  "local_files_only": true,
  "excluded_gpu_ids": []
}
```
> 其余超参（`horizons`、`groups: 16`、`trajectories: 8`、`max_context_tokens: 32768`、`activation_offload: true`、`method_configs.fgpo` 等）须与 [`configs/fgpo-host-b-official-20261116.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-host-b-official-20261116.json#L10-L34) 保持完全一致。

### Step 4：启动单个 F-GPO Cell（或断点 `--resume`）
登录 Colab 后从 GCS 内网拉取冷资产与 `fgpo_src.tar.gz` 到 `/content/fgpo-reproduction`，确认目录就绪后后台拉起单个 cell（每卡一个独立 `--output-dir`）：

```bash
mkdir -p /content/logs /content/outputs
cd /content/fgpo-reproduction

PYTHONPATH=/content/fgpo-reproduction/src \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
nohup /content/venvs/fgpo-main/bin/python -u -m fgpo.runner \
  --config /content/fgpo-reproduction/configs/fgpo-colab-a100.json \
  --output-dir /content/outputs/fgpo-alf-s0 \
  --methods fgpo \
  --benchmarks alfworld \
  --seeds 0 \
  --gpu 0 \
  > /content/logs/fgpo-alf-s0.log 2>&1 < /dev/null &
```

- **确认已正常进入采样/训练**：等待约 30–60 秒后观察 `/content/logs/fgpo-alf-s0.log` 与 `nvidia-smi`，确认无报错退出且显存已按预期加载。
- **跨 VM 断点恢复**：先将 GCS 中保存的该 cell 目录恢复到**完全相同的路径** `/content/outputs/fgpo-alf-s0/alfworld/fgpo/seed-0`，确保 `model_path` 仍是 `/content/models/Qwen2.5-7B-Instruct`，然后在上述命令中追加 `--resume` 即可从 `checkpoint/CURRENT` 精确续跑。

### Step 5：保活与轻量化状态监控

1. **Kernel 保活（避免 15 分钟空闲回收）**
   - Colab 按 Jupyter Kernel 活跃度判定空闲；需要持续每 ~45 秒执行一次 `colab exec -s <session> -f keepalive.py`。
   - **可靠性建议**：尽量把保活轮询跑在**长期在线的 Linux 主机**（如 Host B）后台，避免个人笔记本合盖休眠、切网或本地代理抖动导致保活中断；若只能在 Mac 跑，务必用 `caffeinate -i` 包裹循环，并对单次 `colab sessions` / `colab exec` 网络异常做重试容错，不要一出错就退出。

2. **轻量监控（只同步小文件，不碰百 GB 断点）**
   - 在 Colab VM 后台运行一个轻量循环，每 60 秒将以下几 KB–MB 级文件同步到 `gs://<BUCKET>/fgpo-20261116/live/<cell_id>/`：
     - 训练与评测日志：`<cell>/training.jsonl`、`<cell>/costs.jsonl`、`<cell>/evaluation-*.jsonl`、`<cell>/cell_summary.json`
     - 运行文本日志末尾与轻量 `live_status.json`（`pgrep` 进程存活、`nvidia-smi` 显存/利用率、`df -BG /content` 余量、UTC 时间戳）
   - 本地随时用 `gcloud storage cp` / `cat` 查看即可掌握当前 `round`、`successes`、`trajectories` 与磁盘余量，无需反复开 SSH 抢通道。

### Step 6：优雅停止、成果回传与停机止费

- **情况 A：Cell 完整跑完**
  1. [`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py#L157-L165) 会在发布 `cell_summary.json` 后自动执行 `reclaim_completed_resume_state`，删除体积最大的 `checkpoint/`（优化器与参考权重），仅保留 `final_checkpoint/`（~15 GB 权重 + 状态）及官方评估文件。
  2. 将完整的 `<cell>` 目录（`cell_summary.json`、`config.json`、`official_tasks.json`、`training.jsonl`、`costs.jsonl`、`evaluation-*.jsonl`、`final_checkpoint/`）同步到 GCS 与本地/Host B 汇总。
  3. 确认回传完成后立即停机：
     ```bash
     colab stop -s fgpo-alf-s0
     colab sessions
     ```

- **情况 B：未跑完需要主动释放或转回 Host B 续跑**
  1. 先向 `fgpo.runner` 主进程发送 `SIGTERM`（不要直接 `colab stop` 强杀）：
     ```bash
     kill -TERM <fgpo_runner_pid>
     ```
     [`engine.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/engine.py#L112-L128) 原生捕获 `SIGINT`/`SIGTERM`，会在安全边界写入完整断点并更新 `checkpoint/CURRENT` 后以状态码 `130` 退出。
  2. 确认进程已退出且 `checkpoint/CURRENT` 指向有效快照后，将整个 `<cell>` 目录（至少包含 `config.json`、`official_tasks.json`、`costs.jsonl`、`training_summary.json`、`evaluation-*` 与 `checkpoint/CURRENT` 指向的 snapshot）传到 GCS 或目标机，再执行 `colab stop -s <session>`。
  3. 若将该单卡 cell 迁回 Host B 续跑，需参考 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) 将 `config.json` 与 `state.pt` 中的机器路径与 `journal_path` 重映射到 Host B 路径后再加 `--resume` 启动。

---

## 5. 文件传输、代码更新与配置更新最佳实践

当前项目 [`src/`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src) 约 `240 KB`、[`configs/`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs) 约 `20 KB`，打包后不足 `100 KB`；而冷资产（基座模型/环境）约数十 GB、完整断点可达百 GB。按体量与目标端分流操作，才能兼顾速度与零失误：

### 5.1 分场景最优传输矩阵

| 场景 | 体量 | 推荐通道与工具 | 关键避坑点 |
| :--- | :--- | :--- | :--- |
| **本地 Mac → Host B（代码/配置）** | `< 100 KB` | **`rsync -avz --delete`**（走 SSH） | 只同步白名单或严格 `--exclude`；自动清理本地已删除的旧 `.py`，且默认写临时文件后原子替换 |
| **本地 Mac → Colab（代码/配置）** | `< 100 KB` | `COPYFILE_DISABLE=1 tar` 打包后**单通道上传**：建新 VM 时只传 GCS；已开会话且空闲时 `ssh` 管道或 `colab upload` | **不要同时既传 GCS 又 `colab upload` 传两次**；解压前先清空旧 `src/fgpo`；macOS 必须加 `COPYFILE_DISABLE=1` |
| **冷资产 → Colab（venv / Qwen / 数据）** | `~20–30 GB` | **GCS 桶一次性冷存 → Colab 机房内网拉取** | 严禁从本地 Mac 用 `colab upload` 传大文件，也不在 A100 计费期从外网现下 |
| **Colab 日常监控（日志/进度）** | `几 KB–MB` | **Colab 定时推 GCS + 本地直读 GCS** | 不占用 Colab 单并发 SSH 通道（防 `HTTP 429`） |
| **大成果 / 断点 ↔ Host B / 本地** | `15 GB – 100+ GB` | **`Host B ↔ GCS ↔ Colab` 直连**；Host B 与 Host A 间用 **`rsync -avP --partial`** | 本地 Mac **不做**几十 GB 大文件的二传手；Colab 尽量跑完触发 `reclaim_completed_resume_state` 瘦身后只传 ~15 GB `final_checkpoint` |

> [!TIP]
> **为什么更新小文件到 GCS 时 Python 直调 REST API 比 `gcloud storage cp` 更快？**
> 本机实测 `gcloud` CLI 自身启动耗时约 10 秒左右；如果仅上传不足 100 KB 的源码包或轮询下载 KB 级日志，用本地 Python（如 `requests` + ADC refresh token）直调 GCS HTTP 接口通常 `1–2 秒` 即可完成；传大体积冷资产目录时再用 `gcloud storage cp -r` 更省事。

### 5.2 更新代码的可靠三步法（禁止“热覆盖运行中进程”）

Python 进程启动时已把模块加载进内存，**往正在跑的 `fgpo.runner` 目录直接覆盖 `.py` 不会热更新主循环，反而可能让环境子进程读到新文件导致新旧逻辑混用**。正确做法：

1. **先停再更**：若会话里正在运行训练，先向 `fgpo.runner` 发 `kill -TERM <pid>` 等其保存完断点安全退出。
2. **白名单镜像打包与干净覆盖（Colab）**：
   ```bash
   cd /Users/cxw/Documents/科研方案实现/long-horizon-decision
   COPYFILE_DISABLE=1 tar \
     --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='._*' \
     -czf /tmp/fgpo_src.tar.gz src pyproject.toml configs

   # 若会话已存在且无其他 SSH 连接：先清旧 src 再解压，保证删改文件不残留
   cat /tmp/fgpo_src.tar.gz | ssh -o ProxyCommand="colab ssh --proxy-mode -s fgpo-alf-s0" \
     -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@colab-runtime \
     "rm -rf /content/fgpo-reproduction/src/fgpo && tar -xzf - -C /content/fgpo-reproduction"
   ```
   > 若有多个 Colab 会话或打算后续重建 VM，把 `/tmp/fgpo_src.tar.gz` 推一份到 GCS `code/fgpo_src.tar.gz` 即可。
3. **同步到 Host B（推荐 `rsync` 一条命令完成）**：
   ```bash
   rsync -avz --delete \
     --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='._*' \
     /Users/cxw/Documents/科研方案实现/long-horizon-decision/src/ \
     <host_b_user>@<host_b_host>:/home/chenxiangwei/workspace/fgpo-reproduction-20261116/src/

   rsync -avz --delete \
     --exclude='.DS_Store' --exclude='._*' \
     /Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/ \
     <host_b_user>@<host_b_host>:/home/chenxiangwei/workspace/fgpo-reproduction-20261116/configs/
   ```
   > 仅同步 `src/`、`configs/`（及按需的 `scripts/`、`pyproject.toml`），绝不将本地 `.git`、`.research*`、`outputs/`、`SSH_Document.md` 或 `*apiKey*.csv` 同步到远端。

### 5.3 更新配置与设置的隔离原则

1. **按机器拆独立配置文件，永不互相覆盖**
   - Host A：[`configs/fgpo-qwen7b-official-20261116.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-qwen7b-official-20261116.json)
   - Host B：[`configs/fgpo-host-b-official-20261116.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-host-b-official-20261116.json)
   - Colab A100：`configs/fgpo-colab-a100.json`
   - 三份文件名彼此独立，全量同步 `configs/` 时零冲突。
2. **区分“改科学设置”与“改机器资源设置”**
   - **改科学参数**（如 `rounds`、`learning_rate`、`output_tokens`、`groups` 等）：**必须新建 `--output-dir` 重跑**。旧 `checkpoint/snapshots/<id>/state.pt` 内部保存了原 `trainer['config']`，不能把新超参混进半截旧 cell。
   - **改机器路径 / GPU 编号**（保持单卡对单卡与同一套科学参数）：使用 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) 的方式更新路径与 `journal_path` 后，再配合 `--resume` 续跑。
3. **环境变量固化在启动命令中**
   - 将 `PYTHONPATH`、`OMP_NUM_THREADS`、`TOKENIZERS_PARALLELISM=false`、`CUDA_VISIBLE_DEVICES` 写进对应机器的启动脚本或包装脚本，避免手动 SSH 时漏配导致线程打架或加载错旧包。

### 5.4 断点与大文件跨机迁移的原子顺序

若必须把 Colab 未完成的 cell 断点传到 GCS 或迁移到 Host B，严格按以下顺序防止传半截损坏：
1. 确认 `fgpo.runner` 已退出，读取 `checkpoint/CURRENT` 获知当前唯一有效 snapshot UUID（只传这一个 snapshot 目录，不传无关临时文件）。
2. 先传 cell 根文件（`config.json`、`official_tasks.json`、`costs.jsonl`、`training_summary.json`、`evaluation-*`）及 `checkpoint/snapshots/<uuid>/` 内的全部文件。
3. **最后一步再写入/更新目标端的 `checkpoint/CURRENT`**。只有全部文件落齐后 `CURRENT` 才生效，任何一步断线重传都不会被误判为完整断点。

---

## 6. 常见故障速查表

| 现象 / 报错 | 原因 | 处理方式 |
| :--- | :--- | :--- |
| `HTTP 429 Already-active SSH session` | 同一会话同时有超过 1 个 SSH 连接 | 关闭多余终端或等待前一条 SSH 命令结束；看进度改用 GCS 或 `colab exec` |
| `The physical allocation includes a known faulty GPU (0 or 2)` | 直接复用了 `mgt01` 配置的 `excluded_gpu_ids: [0, 2]` | 将 Colab 配置中 `excluded_gpu_ids` 改为 `[]` |
| `Existing cell identity differs at model_path` | 恢复运行时的 `model_path` 与保存时不一致 | 确保每次 Colab 都固定使用 `/content/models/Qwen2.5-7B-Instruct`，或迁机时先做资源路径重映射 |
| `Exact resume requires the saved number of CUDA RNG streams` | 跨不同 GPU 数量恢复（如 2 卡断点拿到 1 卡 Colab） | 保持 1 卡对 1 卡续跑，不能更改卡数 |
| `Existing cell requires explicit --resume` | 目标 `--output-dir` 下已有目录但启动命令未加 `--resume` | 确认是续跑则加上 `--resume`，新实验则换一个新的 `--output-dir` |
| 保存断点时 `No space left on device` | Colab 临时盘装不下 `save()` 期间双全参数快照过渡峰值 | 清理非必要缓存或直接迁移未完成断点到 Host B（17 TB 盘）续跑 |
| `colab` 命令报认证失败或无法连接 | 代理 `127.0.0.1:7897` 未开，或 `colab` 被升级覆盖掉了包装逻辑 | 确认代理端口可用，或显式传 `HTTPS_PROXY=... colab --auth adc ...` |
