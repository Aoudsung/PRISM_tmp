# Colab F-GPO r3 GCS 恢复与顺序运行

Colab 使用显式 `checkpoint_backend=gcs`。`configs/fgpo-r3-colab-a100.json`（协议 `fgpo-r3-published-v4`）的 study 前缀为 `gs://r032-pact-rebuild-f1be230a/fgpo-r3/checkpoints/fgpo-r3-published-v4-alf`，其后按 `<suite>/alfworld/<method>/seed-0` 等 cell 隔离；`outputs/fgpo-r3-956ceab-alf` 为既有 v3 独立归档目录。Host B 配置继续使用默认 local backend。本文涉及的命令均由原 Colab 执行者在既有实例执行；源码交付没有启动实例、改变当前进程或执行迁移。

每个 cell 的 GCS `CURRENT` 为训练恢复权威，指向不可变 manifest（当前为 `fgpo-gcs-manifest-v2` / `fgpo-gcs-journal-v2`，并兼容读取既有 `fgpo-gcs-manifest-v1` 检查点）。manifest 保存 actor 官方 `save_pretrained` 文件及 optimizer、reference、gradients、trainer/RNG、不可变静态任务/配置（未变直接复用）与增量分段的日志对象名、提交字节偏移、精确 generation 与长度；`costs.jsonl` 同时区分瞬时未换行尾 `uncommitted_tail_bytes_at_publish` 与恢复放弃后的单调 `discarded_incomplete_tail_bytes`，并在相同提交长度合并 `CURRENT`/`FINAL`/`LOGS` 尾证据。actor/Adam 相同版本复用，初始 reference 在同 seed 内复用，gradients 与 state 每次保存真实当前内容；增长中的 `costs.jsonl`、`training.jsonl` 与 `evaluation-*.jsonl` 仅上传新提交换行段并通过几何分级合并保持有界段数。全部成员上传成功后发布 manifest，再用旧 CURRENT generation 条件替换指针，随后更新本地小指针并回收失效对象。所有读路径（`current`、`inspect`、`status`、`pull`、`restore_cell`、`prepare_resume`、`load`、`restore_final` 及 `resources` 预算估算）均在 `open_snapshot` / `_reader_pins` 内先发布并复核 `readers/<lease>.json` 租约与指针不变后再读取 manifest 与成员，且所有校验异常同样进入租约释放清理；活跃读租约存在时，周期性 `sync-logs` 延迟清理旧 LOGS 对象，全量 GC 保留所有活跃读租约引用闭包直至读取释放。传输失败直接终止该操作；上一完整 CURRENT 保留。指针提交成功但后续本地更新或 GC 失败时，新 CURRENT 已完整有效。

最终提交顺序为 `LOGS` 后 `FINAL`：只有匹配的终态日志已发布，FINAL 才公开完成标记；完成态恢复的静态摘要以 FINAL 为权威。新任务的可选远端探测不创建本地 cell 目录。`.cost-tail-state.json` 与实际成本日志一起保留写入 session、已消费弃尾观察及累计字节数；远端 journal 携带对应证据，使同一弃尾在恢复和再次同步后仍只计一次。

Adam/reference/gradients/state 直接 `torch.save` 到 8 MiB 分块上传器；恢复使用 generation 固定的 `seek/read/readinto` 对象。Adam 与 reference 没有本地全文件副本。actor 单独暂存于 `<cell>/actor-cache/model`，保持分片 safetensors、索引、config、generation_config 和 tokenizer；恢复时只物化此目录。原 `model_path=/content/models/Qwen2.5-7B-Instruct` 身份保持原值；CausalPolicy 所需的小 tokenizer/config 文件在该路径保留。每个新 seed 必须从原始 Qwen 冷资产加载 actor 并建立固定 reference。配置显式授权在首次远端保存后释放该原始目录的 safetensors 权重；下一 seed 的 launcher 重新流式解压原始资产。完成态模型保留于 GCS FINAL，训练态回收后删除当前本地 actor cache。

凭据由原执行者在进程外刷新短期 access token，写入本机同用户、权限 0600 的受限临时文件。传输层每次 HTTP 请求重新读取 `FGPO_GCS_ACCESS_TOKEN_FILE`。该环境变量仅包含文件路径，token、ADC、refresh token 和上传 session URL 均不进入配置、状态、manifest 或日志。launcher 不提供 token 刷新机制，操作方须在运行期间持续原子更新该文件。传输层使用 Python 标准库，不需要安装 GCS SDK；冷资产解压沿用已有 tar/zstd。

下面示例采用原始 cell 路径和实际运行配置 `fgpo-r3-colab-956ceab-alf.json`。操作方将模板中的四项 `checkpoint_backend`、`checkpoint_gcs_uri`、`checkpoint_gcs_chunk_mib`、`checkpoint_gcs_release_base_weights` 合入这份现用配置，保留原 GPU 及全部训练设置。执行前由操作方确认实际远端主机、目录及已部署源码。导入不反序列化 optimizer/reference/gradients：直接上传原文件；只读取 state.pt 绑定训练身份及 journal 原路径。默认保留本地 CURRENT，适用于原代码仍持有旧快照的迁移准备。

```bash
cd /content/fgpo-r3
export PYTHONPATH=/content/fgpo-r3/src
export FGPO_GCS_ACCESS_TOKEN_FILE=/run/fgpo-gcs/access-token
PY=/content/venvs/fgpo-main/bin/python
CELL=/content/outputs/fgpo-r3-956ceab-alf/comparison/alfworld/fgpo/seed-0
CONFIG=/content/fgpo-r3/configs/fgpo-r3-colab-956ceab-alf.json

"$PY" scripts/fgpo_gcs_checkpoint.py import-local --config "$CONFIG" \
  --cell-dir "$CELL" --benchmark alfworld --seed 0
"$PY" scripts/fgpo_gcs_checkpoint.py status --config "$CONFIG" \
  --cell-dir "$CELL" --benchmark alfworld --seed 0
```

当前存活进程的过渡由 Root 和原执行者处理。既有 model50 可先导入；必须等待经授权的旧进程在真实组边界保存最新 optimizer125+ 并正常退出，再导入更新后的完整本地 CURRENT。tmpfs actor 桥接、信号、wrapper 暂停和本地大型恢复文件回收均属于该操作方的独立迁移指令，本 CLI 不执行这些动作。更新快照提交远端 CURRENT 后，才可回收已持久化的本地旧大态。`--activate-local-pointer` 仅适用于旧进程已退出且已授权切换本地指针时；该选项本身保留旧 snapshots 文件，操作方负责其显式回收。

```bash
# 在最新完整 snapshot 已导入、旧进程已退出后恢复数据及当前 actor；
# 原有本机完整成本尾事件继续保留。
"$PY" scripts/fgpo_gcs_checkpoint.py restore --config "$CONFIG" \
  --cell-dir "$CELL" --benchmark alfworld --seed 0
"$PY" -u -m fgpo.runner --config "$CONFIG" --output-dir /content/outputs/fgpo-r3-956ceab-alf \
  --suite comparison --methods fgpo --benchmarks alfworld --seeds 0 --gpu 0 --resume
```

runner 在构造 CostLedger 和 CPU 模型运行时之前恢复 cell 数据。`state.pt` 继续保存原 EngineState（round/phase/groups/group_cursor/task_order/reference_data/advantages/update_cursor/training_log_offset/evaluation_completed）、sampler，以及 Python/NumPy/Torch/CUDA RNG。恢复后按旧训练 byte offset 截断统计日志；物理成本保留完整尾事件、按原有规则丢弃半写尾并记录 usage_complete。LOGS 指针提供周期性日志持久化，始终保留成本的最长完整来源；同快照的其他日志可用于继续官方评价。未上传的最后一个日志间隔随实例永久丢失时无法由远端重建，交付不将其计为已持久化成本。

```bash
# 可由既有操作循环每 45 秒调用（需传入 --suite 与 --method），失败直接报告，不重试。
"$PY" scripts/fgpo_gcs_checkpoint.py sync-logs --config "$CONFIG" \
  --cell-dir "$CELL" --suite comparison --method fgpo --benchmark alfworld --seed 0

# 在远端指定运行时上执行 manifest-v2/v1 status 或带读锁重建增量日志与最终模型：
"$PY" scripts/fgpo_gcs_checkpoint.py status --config "$CONFIG" \
  --cell-dir "$CELL" --suite comparison --method fgpo --benchmark alfworld --seed 0
"$PY" scripts/fgpo_gcs_checkpoint.py pull --config "$CONFIG" \
  --cell-dir "$CELL" --suite comparison --method fgpo --benchmark alfworld --seed 0 \
  --dest-dir /tmp/fgpo_pull_stage --with-final-model

# 本地控制端运维包装（不在本地执行项目 Python）：支持显式 --transport ssh|exec（默认 ssh，无 429 重试或隐式回退）。
# exec 传输通过项目内单次直连 Jupyter KernelClient 适配器执行，禁用本地 CLI history/colab.log 记录与核内 store_history，通过 stdin_hook 隔离输入、用每调用随机 nonce 校验完整 stdout/stderr/RC 三帧与严格 base64，不在失败时自动重试连接；
# status/pull 通过 --session 与 --config 在指定远端实例重建并回传结果，拒绝无效的 --bucket/--prefix 覆盖，并自动随同保留已生成的 reporting_summary.json / study_reporting_summary.json / study_reporting_table1.md；
# refresh-token 使用目标目录内唯一 mktemp 文件原子替换远端短期 access token；
# aggregate 在远端以 --aggregate-only 轻量生成原生 Table 1 衍生报告（不初始化 GPU/Engine，不改动训练/终态权威），可选拉取至本地目录；
# stop 强制要求 --cell、--config 与离机 --dest-dir，写入停止标志后基于真实可执行 argv（并排除 stop 自身祖先链）等待 runner、detached supervisor 与检查点完整退出，校验 GCS/本地可恢复状态并将恢复状态及既有衍生 Table 1 报告拉取至离机目录后再销毁实例：
./scripts/colab_ops.sh refresh-token -s fgpo-r3-956ceab-alf \
  --remote-token-file /run/fgpo-gcs/access-token
./scripts/colab_ops.sh status -s fgpo-r3-956ceab-alf --cell comparison/alfworld/fgpo/seed-0 \
  --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json
./scripts/colab_ops.sh pull -s fgpo-r3-956ceab-alf comparison/alfworld/fgpo/seed-0 ./outputs/colab_synced \
  --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json
./scripts/colab_ops.sh aggregate -s fgpo-r3-956ceab-alf --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json \
  --seeds 0,1,2 --dest-dir ./outputs/colab_synced
./scripts/colab_ops.sh stop fgpo-r3-956ceab-alf --cell comparison/alfworld/fgpo/seed-0 \
  --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json --dest-dir ./outputs/colab_stopped

# 从控制端使用已存在并已部署环境的固定会话，通过单写者锁 (.fgpo-colab-supervisor.lock) 下的远端 setsid 脱离式 supervisor 顺序完成三 seed（将进程归属 .fgpo-colab-supervisor.pid 与首 cell 前置/启动就绪标志 .fgpo-colab-supervisor.ready 分离，首 cell 就绪后才返回成功并断开 SSH/exec）。
# 若原前台 SSH launcher 退出留下孤儿 runner PID（且子进程继承着旧 `.fgpo-colab-sequence.lock` FD 9），可通过 `--adopt-pid PID` 绑定该 PID 的 `/proc/<pid>/stat` 启动时间戳、完整命令行及精确 cell/config/seed 后接管定期 `sync-logs` 与后续 seed，不删锁、不强制解锁、不向训练进程发信号；
# 若与旧版前台长连接 SSH 会话共存时，status/pull/refresh-token 须显式传 --transport exec，严禁并发第二条 proxy SSH：
./scripts/colab_launch_fgpo.sh --session fgpo-r3-956ceab-alf --seeds 0,1,2 \
  --config /content/fgpo-r3/configs/fgpo-r3-colab-a100.json \
  --output-dir /content/outputs/fgpo-r3-published-v4-alf \
  --remote-token-file /run/fgpo-gcs/access-token
```

launcher 通过短连接在单写者锁 `$OUTPUT/.fgpo-colab-supervisor.lock`（FD 8）持续持有下完成校验、写 `$OUTPUT/.fgpo-colab-supervisor.sh` 并以 `start_new_session=True` 移交锁描述符拉起脱离式顺序监督进程，在等待 `$OUTPUT/.fgpo-colab-supervisor.ready` 确认配置解析、GCS/本地续跑前提检查、首 cell 冷资产恢复（或接管校验）与首 runner 拉起就绪后才返回退出；所有子进程（`fgpo.runner`、`fgpo_gcs_checkpoint.py` 等）在 `exec` 前显式关闭锁 FD `8/9`，不再向训练子进程泄漏锁。使用 `--adopt-pid PID` 时，launcher 核验旧 `.fgpo-colab-sequence.lock` 除被接管 PID 外无其他持有者、绑定其 `/proc/<pid>/stat` 启动时刻与完整命令行（防止 PID 复用误判）、定期轮询非僵尸状态并触发 `sync-logs`；孤儿进程或自启进程退出后须同时核验本地 `cell_summary.json`（`status="complete"` 且 `finalization_stage="finalized"`）、`.finalization_complete.json`（`terminal_export_completed=true`）以及 GCS `status` DTO（`status="complete"`、`terminal_export_completed=true`、`final_generation` 非空）并接回 `.fgpo-colab-sequence.lock`，随后自动对当前全部已完成 finalized 种子调用 `python -m fgpo.runner ... --aggregate-only` 刷新 `reporting_summary.json`、`comparison/study_reporting_summary.json` 与 `comparison/study_reporting_table1.md`，方才推进下一 seed。监督进程继续在启动时把冷资产恢复收据写入 `FGPO_STARTUP_RECEIPT_FILE` 供 `CostLedger.credit_startup` 将 `runner.cold_asset_restore` 与 `runner.cell_startup` 非重叠启动区间锚定进 `wall_seconds` / `allocated_device_seconds`（不与相位累加混淆）。每个 seed 结束后，`runner` 先回收恢复态并记录 `checkpoint.reclaim_completed_resume_state`，再执行并计量 `engine.publish_final_logs()` 写入 `checkpoint.final_publish_logs`，随后将含该终态事件的 ledger 冻结重写为 `finalization_stage="finalized"` 的 `cell_summary.json`，最后执行一次明确排除在自身成本计量之外的终端导出 `engine.export_final_terminal_state()` 将终态摘要与完整 `costs.jsonl` 同步至 `FINAL`/`LOGS` 并写入 `.finalization_complete.json`（直接运行 `fgpo.runner` 与 launcher 达成等价终态；任何在初步摘要、最终日志或终端导出阶段的中断在 `--resume` 时继续完成而不重跑训练/评价）。每个 seed 的训练失败、日志持久化失败或未产生 finalized cell_summary 都会结束序列。`colab_monitor_ui.py` 默认通过显式 `--transport exec` 调用 `colab_common.sh` 单次直连适配器只读轮询与自动按活跃 seed 跟踪主日志，不触发 `HTTP 429`、CLI 自动重试或隐式回退。launcher 接口使用预先部署的既有实例：申请实例、代码部署、环境冷资产准备和外部 token 刷新由原执行者完成。新 seed 的原始 Qwen 资产使用 `restore-asset --uri gs://r032-pact-rebuild-f1be230a/fgpo-r3/assets/qwen25_7b_instruct.tar.zst --destination /content/models` 流式恢复；若配置切换为 `checkpoint_backend=local`（`checkpoint_interval_rounds=60`），launcher 会跳过 GCS status/sync-logs 并沿用本地断点语义。

CPU 预算读取远端 CURRENT manifest 的对象长度，并读取实际 actor cache 的 safetensors header；gradients.pt 的真实长度进入恢复预算。常驻 Adam/reference 继续为 `14P`，保留原 actor 分片临时量、恢复 CPU gradients 字典和最大单 tensor，另加两块传输缓冲。PyTorch file-like 序列化可能临时处理一个 FP32 tensor，预算覆盖 `2L`（L 为最大 BF16 tensor）。实际硬件、软件版本、主机预算、保存/恢复耗时与恢复后的真实参数更新由后续正式操作日志记录；没有本地执行结果。

最终官方 ALFWorld valid_unseen 134 项任务、训练摘要、评价 journal/records/summary、模型与 provenance 都持久化后，runner 才完成恢复态回收、最终日志计量发布、终态 `cell_summary.json` 冻结与 `FINAL`/`LOGS` 终端导出。FINAL 保留官方模型、config/tokenizer、trainer/RNG provenance 和 claim/cost 日志；GC 根据 CURRENT/FINAL/LOGS 及活跃 `readers/` 租约的引用闭包回收，跨 UUID 复用的 reference 或 actor/Adam 会保留。`restore-final` 可物化最终部署模型；其输出明确 `training_resume_supported=false`。当前 Colab v4 配置运行 FGPO，60 rounds、16×8、H50/out512、混合结果组原子更新、$\lambda_{\max}=0.5$ 和三个既定 seed。
