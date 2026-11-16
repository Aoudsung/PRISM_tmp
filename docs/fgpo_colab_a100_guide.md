# F-GPO r3 (`long-horizon-decision`) Colab GPU（G4 / A100）连接与操作手册

> 使用边界：本文档与配套脚本 [`scripts/colab_*.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_launch_fgpo.sh) 为 **`fgpo-r3-published-v3`** 准备的静态工具与操作说明。主实验方案与指定服务器操作以 [README](../README.md)、[现行设计](fgpo_design_and_derivation.md) 与 [AGENTS.md](../AGENTS.md) 为准；实际启用 Colab 需由用户显式运行对应命令，本文不构成未授权自启动服务器的依据。

本文档说明如何在 **Google Colab `G4`（RTX PRO 6000 96 GB / 176 GB RAM）或 High-RAM A100（80 GB）** 上通过预置冷资产、硬件门槛预检、轻量日志同步及原子断点处理，运行当前项目 [`src/fgpo`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo) 的 **`fgpo`** 正式任务（AppWorld / ALFWorld），以及如何与 **Host B** 并发协同而不撞目录锁。

> [!IMPORTANT]
> 本方案针对旧 Colab 方案中的四类隐患做了彻底重构：
> 1. **取消多份冗余检查点与每 90 秒全量 `tar -czf`**：训练中仅同步 KB–MB 级 JSONL 与 `live_status.json`；
> 2. **冷资产与计费 GPU 解耦**：已通过 0-GPU CPU 会话 [`colab_stage_assets.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_stage_assets.sh) 将环境/权重/数据打包为 `.tar.zst` 存入 `gs://r032-pact-rebuild-f1be230a/fgpo-r3/`；
> 3. **开卡即做显存 / CPU 内存 / 磁盘硬预检 + 禁盲传百 GB 未完成断点**：不满足立即 `colab stop` 止损；
> 4. **修复并发目录锁**：[`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py) 与 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) 已原生支持多卡并发写同一 `--output-dir` 与 r3 跨机迁移。

---

## 1. 与旧 PACT 方案的核心差异（必读）

| 维度 | 旧 PACT (`r032-pact-rebuild`) | 当前 F-GPO r3 (`src/fgpo`) | 对 Colab 操作的直接约束 |
| :--- | :--- | :--- | :--- |
| **训练参数量** | LoRA（单进程 ~18–23 GB VRAM） | **7B 全参数训练**（[`policy.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/policy.py#L25-L98)，`activation_offload: false`） | **单张 G4 / A100 只能跑 1 个 cell**（显存要求 `>= 75000 MiB`），严禁同卡并发双任务 |
| **CPU 内存** | 轻量 | `OffloadedAdamW` + `WeightReference` 在 CPU 驻留 `master`/`exp_avg`/`exp_avg_sq`（FP32）与参考权重（bf16），合计约 `14 B/param`（100 GB+） | 必须满足 `MemTotal >= 130 GiB`：优先用 **`--gpu G4`**（默认 176 GiB，勿加 `--high-mem`）；用 A100 须加 `--high-mem` 严禁降级 |
| **断点体积** | LoRA 检查点几十 MB，每轮存 7 份 `.pt` | 全参数快照含 `model/`、`optimizer.pt`、`gradients.pt`、`reference.pt`、`state.pt`；`save()` 更新参数时新旧快照短暂并存（峰值 ~212+ GB） | 启动前校验 `/content >= 225 GiB`（`< 260 GiB` 时告警提示）；**禁止训练期打包上传百 GB `checkpoint/`** |
| **续跑校验** | 依赖 `run_contract.py` SHA 校验 | [`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py) 原生 `require_training_identity` + `--resume` | 固定 `/content/...` 路径即可 `--resume`；1 卡断点只能对 1 卡续跑 |
| **GPU 排除列表** | 无 | Host A 配置含 `excluded_gpu_ids: [0, 2]` | Colab 使用 [`fgpo-r3-colab-a100.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-colab-a100.json)（`excluded_gpu_ids: []`） |

---

## 2. Host B 与 Colab 的分工原则

当前首要目标是先完整跑通 **F-GPO 主方法**（`--suite comparison --methods fgpo`，2 个 benchmark × 3 个 seed = **6 个完全独立的单卡 cell**）：

- **Host B**（RTX 6000D × 3，461 GB CPU 内存，17 TB 硬盘，无回收超时）：
  - 优先承接长耗时、大评测集、需要大容量持久盘的 **`appworld`**（`horizon=100`，`test_normal=168` + `test_challenge=417`）及部分 `alfworld` cell。
  - [`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py#L290-L330) 已支持同一 `--output-dir` 下多张卡并行跑互不重叠的 `seed` / `benchmark`（单元级文件锁防重复启动、`study_summary.json` 加锁合并聚合）。
- **Colab `G4` / High-RAM A100**（单卡 96 GB / 80 GB，`--gpu 0`，容器盘 `236 GiB` 且有超时/回收风险）：
  - 若启用低磁盘保存策略或短程执行，优先承接单轮更快、评测集更轻的 **`alfworld`**（`horizon=50`，`valid_unseen=134`），争取在单次连续会话内跑完至 `final_checkpoint`（自动执行 [`reclaim_completed_resume_state`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/checkpoint.py#L202-L239) 清理大型优化器状态后，只需回传 ~15 GB 最终模型与评测日志）。
- **两条硬约束**：
  1. **显式指定方法**：[`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py#L267-L315) 按 `benchmark → method → seed` 遍历，所有机器均需显式传入 `--suite comparison --methods fgpo`，避免先跑完 `alfworld` 全部对比/消融才开始 `appworld`。
  2. **1 卡对 1 卡约束**：[`checkpoint.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/checkpoint.py#L40-L42) 的 `restore_rng` 严格校对 CUDA RNG 流数量，Colab（1 卡）只适合自建新 cell 或接续原本就是 **1 卡** 的 cell，不能直接接手 2 卡断点。

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
  - **GCS 持久路径**：当前已配置并预置冷资产至 **`gs://r032-pact-rebuild-f1be230a/fgpo-r3/`**（与历史 `r032-pact-rebuild` 对象隔离，可通过环境变量 `FGPO_GCS_BUCKET` / `FGPO_GCS_PREFIX` 或 `--bucket` / `--prefix` 覆盖）。

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

#### 2) 申请 Colab GPU 实例（1 个会话只跑 1 个 cell）
实测当前账号机型状态：
- **`G4`（`NVIDIA RTX PRO 6000 Blackwell Server Edition`）**：实测可用，默认自带 **`97,887 MiB` 显存、`176 GiB` 内存、`236 GiB` `/content` 盘**。注意 `G4` **不要传 `--high-mem`**（后端会对 `--gpu G4 --high-mem` 返回 HTTP 400）。
- **`A100`**：`--high-mem` 当前常遇 `503 Service Unavailable`，而不带 `--high-mem` 的标准 `A100` 实测仅 `40 GB VRAM / 83 GB RAM`，放不下全参数 F-GPO。

```bash
# 推荐：G4 (RTX PRO 6000, 96GB VRAM / 176GB RAM) —— 不传 --high-mem
colab new -s fgpo-alfworld-s0 --gpu G4

# 仅在确认有 High-RAM A100 配额时使用（若 503 勿降级标准 A100）
colab new -s fgpo-alfworld-s0 --gpu A100 --high-mem
```

#### 3) 连接与执行命令（SSH / Exec）
- **方式 A：交互式登录 Colab（默认落入 `/content`）**
  ```bash
  colab ssh -s fgpo-alfworld-s0
  ```
- **方式 B：标准 `ssh` 非交互执行命令或管道传输**
  ```bash
  ssh -o ProxyCommand="colab ssh --proxy-mode -s fgpo-alfworld-s0" \
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
  配置后即可直接 `ssh fgpo-alfworld-s0` 或 `scp local_file fgpo-alfworld-s0:/tmp/`。
- **方式 D：通过 Jupyter Kernel 执行 Python（不占 SSH 通道，同时刷新空闲计时器）**
  ```bash
  colab exec -s fgpo-alfworld-s0 -f /tmp/check_or_keepalive.py
  ```

> [!NOTE]
> **SSH 单连接限制**：Colab 每个会话**同一时刻仅允许 1 个活跃 SSH 连接**，并发发起第二个 SSH 会返回 `HTTP 429`。日常查看进度优先通过 GCS 或 `colab exec`，不要开多个长连 SSH 终端占用通道。

#### 4) 文件上传 / 下载与会话清理
```bash
# 上传 / 下载单文件
colab upload -s fgpo-alfworld-s0 ./local_file /content/remote_file
colab download -s fgpo-alfworld-s0 /content/remote_file ./local_file

# 停止释放 GPU 计费
colab stop -s fgpo-alfworld-s0
colab sessions   # 核对已无残留会话
```

---

## 4. Colab 运行 F-GPO r3 一键工具与标准操作流程

已针对旧方案的所有膨胀与竞态隐患提供可直接运行的脚本（位于 [`scripts/`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts)）与专用配置 [`configs/fgpo-r3-colab-a100.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-colab-a100.json)：

| 文件 | 职责 | 解决的问题 |
| :--- | :--- | :--- |
| [`colab_stage_assets.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_stage_assets.sh) | 用 **0-GPU CPU 会话**预装双 venv、模型权重与数据并打为 `.tar.zst` 存入 GCS | 彻底杜绝开出计费 GPU 后干等 pip/HF 下载 |
| [`colab_launch_fgpo.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_launch_fgpo.sh) | 开 `G4`/High-RAM A100 → 硬件/内存/盘硬门槛预检 → 内网流式解压 → 启动单 cell + 60s 轻监控 + 容错保活 | 不达标秒停不烧卡；不每 90s 打包百 GB 断点；结束只传瘦身后成果 |
| [`colab_ops.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_ops.sh) | 统一提供 `sync-code` / `status` / `pull` / `stop` 日常操作 | 禁热覆盖运行中进程；`status` 不抢 SSH；`stop` 先发 `SIGTERM` |
| [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) | 按目标配置文件重映射已停止断点路径、GPU 与 `journal_path` | 去除旧 `v1` Host B 硬编码，兼容 `fgpo-r3` 跨机 1 卡迁移 |

### Step 1：0 GPU 计费预置冷资产（已完成，仅后续更新依赖时需重跑）
> 当前 `gs://r032-pact-rebuild-f1be230a/fgpo-r3/assets/` 已预置全套 5 个 `.tar.zst`（`fgpo_venv_main`、`fgpo_venv_alfworld`、`alfworld_data`、`appworld_data`、`qwen25_7b_instruct`）及 `code/fgpo_r3_src.tar.gz`，无需重复执行。后续如新增依赖可用 `--force` 刷新：
```bash
# 仅在需要更新 venv 或数据资产时执行（0 GPU 费用，支持断点跳过已上传项）
./scripts/colab_stage_assets.sh --appworld-bundle-obj appworld_020_bundle.tar.gz
```

### Step 2：启动单个 F-GPO Cell（含硬件/盘门槛预检与轻监控）
[`colab_launch_fgpo.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_launch_fgpo.sh) 默认申请 **`--gpu G4`**（RTX PRO 6000，96 GB VRAM / 176 GB RAM），并先校验 `VRAM >= 75000 MiB`、`MemTotal >= 130 GiB`、`/content` 剩余空间 `>= 225 GiB`；任一项不达标即**立刻 `colab stop` 退出**，不会拉资产或空烧算力单元：

> [!WARNING]
> **实测 `/content` 磁盘硬限制**：Colab `G4` 容器总盘为 **`236 GiB`**，解压全部冷资产（`25 GiB`）后剩余 **`212 GiB`**；而 [`checkpoint.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/checkpoint.py#L71-L115) 轮末保存全参数断点期间新旧两份快照（各约 99 GB）并存加 `reference.pt`（约 14 GB）峰值约 **`212+ GiB`**（另加每轮累积的 `groups/*.pt`）。因此，在未对 Colab 开启低磁盘存储模式（如跳过中间轮次全量 `optimizer.pt` 双份保留）前，长轮次全参数 `fgpo` 训练在 Colab 上仍有在保存阶段爆盘风险，建议优先放 Host B（17 TB）或先明确裁剪保存策略后再启动长训练。

```bash
# 在 Colab G4 (PRO 6000) 上启动 comparison suite 的 alfworld / fgpo / seed 0
./scripts/colab_launch_fgpo.sh --gpu G4 --benchmark alfworld --seed 0 --session fgpo-alfworld-s0

# 若需接续 GCS 中已有的断点（保持相同 /content 路径）
./scripts/colab_launch_fgpo.sh --gpu G4 --benchmark alfworld --seed 0 --session fgpo-alfworld-s0 --resume
```
> 若你打算把 `colab exec` 保活放到长期在线的 Linux 主机上跑，启动时加 `--no-keepalive` 即可避免依赖个人 Mac 不休眠。

### Step 3：查看进度、拉取结果与优雅停机
```bash
# 1) 零 SSH 冲突查看进度（读 GCS live_status.json 与 training.jsonl 末尾）
./scripts/colab_ops.sh status --cell comparison/alfworld/fgpo/seed-0

# 2) 仅在必要时用单次 SSH 实时看显存/进程/磁盘
./scripts/colab_ops.sh status --session fgpo-alfworld-s0 --ssh

# 3) 拉取完成后的日志与评测结果到仓库 outputs/colab_synced/（加 --with-final-model 可一并拉 ~15 GB final_checkpoint）
./scripts/colab_ops.sh pull comparison/alfworld/fgpo/seed-0

# 4) 中途需要主动停机：先发 SIGTERM 等 engine.py 安全写完 checkpoint/CURRENT，再停实例
./scripts/colab_ops.sh stop fgpo-alfworld-s0 --cell comparison/alfworld/fgpo/seed-0
```
> ⚠️ 若需要将未完成单卡 cell 从 Colab 迁回 Host B（或反之），须先停止训练、传入完整的 `checkpoint/CURRENT` 指向快照、再在目标机运行 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py)（`--resource-config configs/fgpo-r3-host-b.json --physical-gpu <id>`）后加 `--resume` 续跑。

---

## 5. 文件传输、代码更新与配置更新最佳实践

当前项目 [`src/`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src) 约 `240 KB`、[`configs/`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs) 约 `20 KB`，打包后不足 `100 KB`；而冷资产（基座模型/环境）约数十 GB、完整断点可达百 GB。按体量与目标端分流操作，才能兼顾速度与零失误：

### 5.1 分场景最优传输矩阵

| 场景 | 体量 | 推荐通道与工具 | 关键避坑点 |
| :--- | :--- | :--- | :--- |
| **本地 Mac → Host B（代码/配置）** | `< 100 KB` | **`rsync -avz --delete`**（走 SSH） | 只同步白名单（`src/`、`configs/`、`scripts/`）；自动清理本地已删除的旧 `.py`，且原子写替换 |
| **本地 Mac → Colab（代码/配置）** | `< 100 KB` | `./scripts/colab_ops.sh sync-code [session]` | `COPYFILE_DISABLE=1` 防 `._*`；走 Python 直调 GCS REST（1–2s，避开 `gcloud` ~10s 冷启动）；解压前先清空旧 `src/fgpo` |
| **冷资产 → Colab（venv / Qwen / 数据）** | `~20–30 GB` | `./scripts/colab_stage_assets.sh`（**0-GPU CPU**）→ **Colab 内网流式 `zstd` 解压** | 严禁从本地 Mac 用 `colab upload` 传大文件，也不在 GPU 计费期从外网现下 |
| **Colab 日常监控（日志/进度）** | `几 KB–MB` | **Colab 60s 轻同步 → `./scripts/colab_ops.sh status`** | 不占用 Colab 单并发 SSH 通道（防 `HTTP 429`）；**不打包上传百 GB `checkpoint/`** |
| **大成果 / 断点 ↔ Host B / 本地** | `15 GB – 100+ GB` | **`Host B ↔ GCS ↔ Colab` 直连**；Host B ↔ Host A 用 **`rsync -avP --partial`** | 本地 Mac **不做**几十 GB 大文件的二传手；Colab 尽量跑完让 `reclaim_completed_resume_state` 瘦身后只传 ~15 GB `final_checkpoint` |

### 5.2 更新代码的可靠三步法（禁止“热覆盖运行中进程”）

Python 进程启动时已把模块加载进内存，**往正在跑的 `fgpo.runner` 目录直接覆盖 `.py` 不会热更新主循环，反而可能让环境子进程读到新文件导致新旧逻辑混用**。[`colab_ops.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_ops.sh) 的 `sync-code` 默认拦截运行中覆盖：

1. **先停再更**：若会话里正在运行训练，先执行 `./scripts/colab_ops.sh stop <session>`（自动发 `SIGTERM` 等安全保存退出）。
2. **更新 GCS 代码包及空闲会话**：
   ```bash
   # 仅更新 GCS 最新代码包（1–2 秒）
   ./scripts/colab_ops.sh sync-code

   # 或同步到尚未启动训练的空闲 Colab 会话（先删旧 src/fgpo 再解压）
   ./scripts/colab_ops.sh sync-code fgpo-alfworld-s0
   ```
3. **同步到 Host B（推荐 `rsync` 一条命令完成）**：
   ```bash
   rsync -avz --delete \
     --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' --exclude='._*' \
     /Users/cxw/Documents/科研方案实现/long-horizon-decision/src/ \
     <host_b_user>@<host_b_host>:/home/chenxiangwei/workspace/fgpo-r3/src/

   rsync -avz --delete \
     --exclude='.DS_Store' --exclude='._*' \
     /Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/ \
     <host_b_user>@<host_b_host>:/home/chenxiangwei/workspace/fgpo-r3/configs/
   ```
   > 仅同步 `src/`、`configs/`、`scripts/`、`pyproject.toml`，绝不将本地 `.git`、`.research*`、`outputs/`、`SSH_Document.md` 或 `*apiKey*.csv` 同步到远端。

### 5.3 更新配置与设置的隔离原则

1. **按机器拆独立配置文件，永不互相覆盖**
   - Host A：[`configs/fgpo-r3-host-a.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-host-a.json)
   - Host B：[`configs/fgpo-r3-host-b.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-host-b.json)
   - Colab（`G4` / High-RAM A100）：[`configs/fgpo-r3-colab-a100.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-colab-a100.json)
   - 三份文件名彼此独立，全量同步 `configs/` 时零冲突。
2. **区分“改科学设置”与“改机器资源设置”**
   - **改科学参数**（如 `rounds`、`learning_rate`、`output_tokens`、`groups` 等）：**必须新建 `--output-dir` 重跑**，旧 `state.pt` 内的 `trainer['config']` 由 `require_training_identity` 严格保护。
   - **改机器路径 / GPU 编号**（保持卡数一致）：保持科学参数不变，先运行 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) 重映射路径与 `journal_path`，再加 `--resume` 续跑。
3. **环境变量固化在启动命令中**
   - `PYTHONPATH`、`OMP_NUM_THREADS`、`TOKENIZERS_PARALLELISM=false`、`CUDA_VISIBLE_DEVICES` 均固化在启动脚本或 [`runner.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/src/fgpo/runner.py) 内，避免手动 SSH 时漏配。

### 5.4 断点与大文件跨机迁移的原子顺序

若必须把未完成的 cell 断点跨机迁移，严格按以下顺序防止传半截损坏（[`colab_ops.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_ops.sh) 的 `stop --backup-checkpoint` 与 [`colab_launch_fgpo.sh`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/colab_launch_fgpo.sh) 的 `--resume` 均已按此实现）：
1. 确认 `fgpo.runner` 已退出，读取 `checkpoint/CURRENT` 获知当前唯一有效 snapshot UUID（只传这一个 snapshot 目录）。
2. 先传 cell 根文件（`config.json`、`official_tasks.json`、`costs.jsonl`、`training_summary.json`、`evaluation-*`）及 `checkpoint/snapshots/<uuid>/` 内的全部文件。
3. **最后一步再写入/更新目标端的 `checkpoint/CURRENT`**。

---

## 6. 常见故障速查表

| 现象 / 报错 | 原因 | 处理方式 |
| :--- | :--- | :--- |
| `HTTP 429 Already-active SSH session` | 同一会话同时有超过 1 个 SSH 连接 | 关闭多余终端或等待前一条 SSH 命令结束；日常查看改用 `./scripts/colab_ops.sh status` |
| `Pre-flight check failed` 并自动 `colab stop` | 分配的 Colab VM 显存 `< 75 GiB`、内存 `< 130 GiB` 或 `/content` 盘 `< 225 GiB`（以及 `--gpu G4 --high-mem` 触发 400） | 脚本已自动止损停机；`G4` 请勿加 `--high-mem`，或将该 cell 改放到 Host B（17 TB 盘）运行 |
| `The physical allocation includes a GPU excluded by this server configuration` | 直接复用了 `mgt01` 配置的 `excluded_gpu_ids: [0, 2]` | 使用 [`configs/fgpo-r3-colab-a100.json`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/configs/fgpo-r3-colab-a100.json)（`excluded_gpu_ids: []`） |
| `Existing cell identity differs at model_path` | 恢复运行时的 `model_path` 与保存时不一致 | 保持 `/content/models/Qwen2.5-7B-Instruct` 固定路径，或先用 [`relocate_fgpo_cell.py`](file:///Users/cxw/Documents/科研方案实现/long-horizon-decision/scripts/relocate_fgpo_cell.py) 重映射 |
| `Exact resume requires the saved number of CUDA RNG streams` | 跨不同 GPU 数量恢复（如 2 卡断点拿到 1 卡 Colab） | 保持 1 卡对 1 卡续跑，不能更改卡数 |
| `Existing cell requires explicit --resume` / `Cell is already locked` | 目标 cell 已存在且未传 `--resume`，或有另一进程正在写同一 cell | 确认是续跑则加 `--resume`；新任务需分配不同 `seed`/`benchmark` 或换新 `--output-dir` |
| `colab` 命令报认证失败或无法连接 | 代理 `127.0.0.1:7897` 未开，或 `colab` 被升级覆盖掉了包装逻辑 | 确认代理端口可用，或显式传 `HTTPS_PROXY=... colab --auth adc ...` |
