# PRISM-Evolve 原生环境实验协议 native-v1

本协议记录固定官方源码的原生迁移实现与共同实验口径。用户最终确定的后续基准为家居文本交互环境（`alfworld`）、电商购物交互环境（`webshop`）与长上下文代理基准（`loca`，LOCA-Bench），后续不再使用 ClinDiag。PRISM §3–§5 的成功时间效用和分层估计器保持，所有强化学习方法的任务回报统一使用 U。正式运行与短诊断保持停止，Colab 暂缓；本轮只在服务器完成纯用户态环境配置、桥接调用与必要接口验证，不启动新实验。

当前三环境均采用完全自包含的本地官方实现与数据，无需外部商业 API 密钥：`alfworld` 使用官方 TextWorld 游戏环境；`webshop` 直接接入固定官方仓库 `gigpo-official/agent_system/environments/env_package/webshop/webshop`（`verl-agent` commit `20bd331bdbc9026a5668e11362178e10ab7400c8`）中的 `WebAgentTextEnv` 与 Pyserini Lucene 检索索引；`loca` 已完成单轨原生工具接口、15 个模板的训练参考以及基于纯用户态 QEMU 虚拟机（`savevm`／`loadvm` + `qcow2` + `virtio-serial`）的完整同状态进程与磁盘快照桥接。

PoS 的任务、预算和模型依据为[论文附录 C–D](https://arxiv.org/html/2610.01415v1)及[固定源码 `6818cfa`](https://github.com/luoyu100/PoS/tree/6818cfa6434dbc91fcf67f0b3eb12a91db47433f)；WebShop 的环境与划分依据为 [GiGPO / verl-agent 官方实现](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/webshop)。本项目新增的训练划分、I/E 域适配、基座及解码差异在下文分别登记。

## 必做实验矩阵与主张范围

主比较保留 Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct 两个基座、上述三个环境（`alfworld`、`webshop`、`loca`）以及种子 42/43/44。十一项主方法为 Frozen Base、SPAR、GiGPO、Step-DPO、InfoPO、AREW、T3、WorldEvolver、PoS、EnvProbe-Simple 与 EnvProbe-Judge，其必要机制完整保留。两项 SPAR 自身消融 Unweighted、Unstratified 仅在 Qwen2.5-7B 上进行。

| 必做内容 | 范围 | 正式测试单元 |
|---|---|---:|
| 完整主比较 | 11 方法 × 2 模型 × 3 环境 × 3 种子 | 198 |
| SPAR 自身消融 | 2 变体 × 1 模型 × 3 环境 × 3 种子 | 18 |
| 合计 | 保留各环境完整登记测试集 | 216 |

每个模型／方法／种子配置分别测试 `alfworld` 的 134 题、`webshop` 的 500 题和 LOCA-Bench 的 75 例；全族正式测试登记共 51,048 回合。LOCA 使用用户选择的 PoS 默认 8K、75 例设置，论文主表的七种长度共 525 例不进入本轮。AppWorld、PlanBench-XL 和 ClinDiag 均不在当前活动清单，既有代码、数据及结果保留。Outcome-PG、Outcome-Value、Outcome-RLOO 与 27B 配置继续退出本轮清单。本轮不开展额外预算、分叉步或模型规模扫描。

主表仅支持这两个基座与三个环境中的预算内任务表现比较；机制结论限定于 7B 上登记的两项消融。反馈诊断检验 SPAR 初始参数与训练终点在固定初始策略锚点分布下的反馈利用变化。正式结果产生前不预写领先或提升结论；未测的同行反馈排名、广泛规模规律及预算/分叉步鲁棒性均不作为已证实主张。

## 实验对象与结果身份

r7 的 PoS 一致性返回适配只将模型直接给出的 JSON 问题列表封装到 `issues` 键，保留原问题内容与原模型文本，不生成分数或增加调用；其他角色保持原始返回，沿用官方有界 JSON 修复预算。r6 已完成的有效首题输出按来源执行 cohort 分组继承，仍匹配任务、种子、原方法位置及 GPU；不同来源参考策略保留各自完成覆盖。复用 WorldEvolver 的连续题目前缀时，必须具备该题真实题后记忆 checkpoint，并恢复完成题目顺序后继续新题；缺少记忆时拒绝复用，不以空记忆冒充连续执行。各版本及修复／中断成本分别披露。

r6 修复 LOCA InfoPO 反事实评分提示与采样提示的序列化不一致：无门控生成保存其公开工具 schema，遮蔽反馈评分使用同一个 `_text_action_messages` 构造器和同一原生全词表支持，只更换规定反馈。r5 该单元在 879,611 计算 token 后因重建提示超出上下文提前结束、0 参数更新，保留为受影响结果，不作为完整训练产物复用。其余完成的训练阶段按相同模型、种子、预算、数据和动作契约继承；正在执行且无优化器状态的阶段从原初始参数重新完成预算，成本与版本分别披露。主比较范围与每小时监测规则保持。

用户最新明确启动其余九项 baseline。r5 主比较使用 Qwen2.5-7B、seed42、三个环境、原登记十一主方法，共33单元／7,799测试回合；不扩展其他种子、模型或消融。保留原完整训练与测试预算，全局最多三个并发单元。r3已完成的初始／终点训练由 `completed_training_imports` 继承，r4完成的参考策略测试由 `completed_evaluation_imports` 继承；新增九项方法按原定义训练或执行控制器。`allow_expanded_method_evaluation_reuse` 显式登记从双方法扩展至十一方法：继承记录按来源配置验证原方法位置，保留原时间和位置，另记当前块计划位置，不冒充十一方法同时间交错证据；新方法测试保持同题同种子同GPU，墙钟效用披露阶段和共享负载。全部方法齐备后统一登记比较族统计。r4 后续 memory 默认 `./memory.json` 相对宿主目录而触发隔离检查，r5 将原生 memory 相对路径按公开 agent_workspace 解析，绝对路径仍接受同一沙箱检查。

r4 是 r3 已完成训练的续评测版本。terminal 配置明确将缺省 `allowed_dir` 设为当前 `agent_workspace`，修复上游别名默认值 `"."` 覆盖路径的问题，沙箱原有公开目录校验保持。`completed_training_imports` 为六个已完成初始／终点训练阶段的绝对路径；加载时校对方法、模型、种子、任务登记、训练预算、LoRA、上下文和动作契约，保存来源身份及原成本归属。`reuse_completed_paired_evaluations` 仅复用该任务所有登记策略均已落盘的完整策略对，逐项匹配任务、actor seed、方法次序和 GPU，保留原始评分与实际时间；缺少任何一方时整对执行。复用不依赖成功或失败结果。新记录带 `evaluation_reuse`，新配对块标明 `reused_methods`，报告按版本披露继续执行与既有结果来源，不重复计费、不重复训练。

用户最新授权后续真实故障按 `rules.md` 修订代码或方案并自动续跑，替代本轮逐次确认启动的约束；并发上限三、独立版本身份及原始成本记录保持。r2 已验证 WebShop 分支恢复和 LOCA 首次模型调用，但 LOCA 工具返回后的工作区已同步并冻结 QEMU，终态评分再次请求 VM 同步造成通信超时。r3 只删除这次冻结后的重复同步，受信任评分继续读取冻结前已同步的工作区。r2 没有可恢复的中途优化器状态，r3 从相同初始参数重新完成登记训练；此前部分轨迹和成本保留独立身份，不拼接为一次训练。

2026-11-09 首轮 `prism-qwen7b-first3-s42-20261109` 仅登记 Host B Qwen2.5-7B、seed42，三个环境各一个 PRISM 训练单元及同接口 Frozen Base 初始策略，共六个完整测试单元、1,418 个测试回合；最多同时三个单环境 cohort。此次运行因 WebShop 分支恢复状态不一致、LOCA 首次动作前上下文超限而停止，原目录不续跑、不覆盖。修复版本使用独立 `-r2` 配置及运行身份，用户已明确批准启动，原有预算和划分保持。

LOCA 的固定公开工具目录在模型提示中只序列化一次，保留原始工具描述、名称与参数；门控看到完整 I/E 工具目录，条件动作看到对应模式目录。历史中重复的工具定义及由其展开的 JSON Schema 从模型提示副本移除，历史原文、原生观察、动作支持与 JSON 解码约束保持。WebShop 重置及前缀恢复时，删除相同登记 session ID 的旧会话，再由官方环境按同一目标索引建立新会话，避免上一个分支的 `done`、`reward` 等字段残留；目标顺序和评分不变。

旧 `outputs/prism_evolve_experiments/`、`results/prism_evolve/` 和论文证据包保留为冻结表征、512维策略、固定宏动作环境的历史受控实验；此前 AppWorld／PlanBench 的 `prism-native-official-20261108` 身份也予保留。三环境配置使用新的 `prism-native-pos-benchmarks-20261108` 实验及协调身份，输出根为 `outputs/prism_evolve_pos_benchmarks/`，运行必须选择新目录；旧轨迹、checkpoint 和结果不跨版本混池。

PRISM 策略采用真实自回归语言模型与同一组 LoRA 参数。I/E 概率由同一模型的两个合法模式标签归一化得到，条件原生动作按其合法 token 集合生成。训练记录实际 prompt/response token、assistant-only mask、行为概率和原生反馈。模型结构大小不进入初始化、奖励或动作评分；不存在任务 ID 奖励扰动、长度衰减或额外 SNR 梯度收缩。

PRISM、Frozen Base 与自身消融在三环境采用 `prism-native-text-v1`：选定 I/E 后直接输出当前环境的原生命令或原生工具 JSON；Frozen Base 保持同接口及同初始参数 θ0。该路径绕过 `native_program.py` 的 PRISM JSON AST，采样与重评分使用同一公开动作支持和合法 token 分布。只有一种模式可用时精确使用该模式；两类均非空时才形成 I/E 分层分支。各方法实际 actor 契约写入 manifest 与训练记录。

GiGPO、InfoPO、AREW、T3、Step-DPO 使用不受 PRISM 门控的原生动作 actor；PoS、WorldEvolver、EnvProbe 使用 `official-controller-native-v1` 和相同环境。全部方法接收相同的公开原生接口与各自交互产生的反馈，保留原方法的推理流程。普通 PG/Value/RLOO 仅保留历史受控实现，不在当前完整矩阵。

PRISM 的公开目录和绑定仅依据可见历史重建；反馈遮蔽同时屏蔽对应派生来源。InfoPO 的自由原生 actor 使用共同完整词表，对真实下一动作 token 在可见和遮蔽反馈下重评分；不把支持差异或自动绑定名变动处理为零信息奖励。

家居环境采用用户确认的方案 A：`look`、`inventory`、`examine`、`go to`、`open` 属于 I；拿取、放置、加工、使用、关闭等属于 E。I 可以改变位置或容器状态，这一领域扩展按信息获取意图分类，不能宣称无副作用。WebShop 采用原生文本交互动作：`search[...]` 及商品浏览、翻页、详情／特征／评论查看、属性选择等非终局 `click[...]` 属于 I，`click[buy now]` 终局购买提交属于 E。LOCA 按公开工具能力确定信息获取与业务修改／提交类别，不能依据隐藏答案或调用后的结果分类。EnvProbe 使用单独声明的核验候选，不能把整个 PRISM I 集自动视为只读探针。

环境直接执行策略实际生成的一个原生动作，原始输出、解析动作和原生反馈分别留存。非法或截断输出、原生失败与基础设施异常分别记录；未执行真实信息操作时，不因采样到 I 标签而计为有效信息覆盖。训练快照必须保存并恢复信息动作造成的真实状态变化、公共前缀和随机状态；恢复不回滚已付成本。普通 baselines 不增加 PRISM gate。

本轮训练日志分别保存原生任务成功、任务效用、每步接口错误、目标函数值、optimizer 调用、梯度范数、非零梯度和实际参数变化。一次 optimizer 调用不代表参数已经变化；整段预算内零有效参数更新按实际学习信号缺失记录，并保留有效动作、原生失败、预算终止及有效监督不足的具体原因。Frozen Base、PoS、EnvProbe 等冻结推理方法的参数训练标为不适用，按完整决策流程与原生任务表现记录；WorldEvolver 的真实记忆更新独立保存。真实零值和缺失字段按各自含义保留，禁止人工补奖励、伪造成功或将缺失诊断解释为已学习。上述记录用于正式运行中的故障定位与结果解释，不新增自动重试或逐阶段放行。

已停止的各轮训练轨迹包含同题分支与组内采样，其成功数、错误数和 loss 保持原身份。全部旧 checkpoint 与运行目录保留；下一轮使用全新协调 session 和空目录，不拼接旧轨迹或恢复旧更新状态。

主评测沿用训练目标的 temperature=1 单轨采样分布，模式与动作都按该分布执行；每题固定并记录独立随机流，避免训练消耗量改变测试随机序列。各方法 actor 使用同一解码约定，冻结辅助评审保持确定性。PoS 论文使用 temperature=0 及其论文基座，本项目保留既定 7B／8B 基座和共同采样目标，结果身份为原生环境迁移比较。Greedy 可通过独立运行配置评估，单独注明为另一解码协议，不将其称为分层估计器严格保持的部署目标。

三环境按各自原生信号计分：家居环境报告整体及任务类别的 micro 成功率；WebShop 报告官方 `WebAgentTextEnv` 的精确购买成功率（`reward == 1.0`）与平均任务得分（`mean_native_score`，即官方细粒度属性／选项／价格匹配得分 × 100 的 `mean_task_score`）；LOCA 报告 `claim_done` 原生评分与各任务模板成功率。项目预算内成功 Y、成功时间效用 U 和原生分数分别保留。所有环境均在本地受控运行，评价侧真值留在私有评价侧。LOCA 的任意代码与文件工具在清理凭据后的隔离 QEMU 虚拟机工作区运行，评价侧真值不向虚拟机挂载。

## 保留的旧环境接口

以下 AppWorld 和 PlanBench-XL 约定用于保留的既有实现与历史身份，当前三环境配置不调度它们。

AppWorld 使用官方可执行 Python/API 环境及 `AppWorld.evaluate()`。策略只接触任务说明、监督者公开信息、公开 API 文档和真实调用返回。策略输入与训练公共轨迹不包含 ground-truth 文件、正确路径、测试断言或答案预览。全部方法在 t0 同样获得 `show_app_descriptions`、`show_api_descriptions`、`show_api_doc` 三个官方 GET 入口和 `supervisor.complete_task` POST 的实际公开签名；其 answer 默认为 None。其他 API 仅在真实 `show_api_doc`/`search_api_docs` 详单返回并经 emit/print 进入模型可见观察后加入目录，只有名称和简介的目录不扩充候选。完整内部 API 表仅用于原生请求守卫，不向生成器提供业务候选。

`schema_for_history` 与 `public_action_schema` 从可见历史重建 PRISM 公共程序上下文；隐藏唯一来源时其派生目录和绑定随之移除。自由原生 AppWorld baseline 保留官方 stdout 与跨步 shell 状态，只有真实打印的返回进入模型观察。两种动作路径均保留原生错误、部分副作用与真实计费；需要恢复时按同初态、同种子重放。

公共提示与动作 schema 明确 `apis` 已由官方环境注入，应直接调用 `apis.<app>.<api>(...)`，无需且不得导入 `apis` 或其 app 属性；单次导入或调用报错按该次操作解释，后续动作依据真实错误和公开文档选择。沿用官方交互约定，模型每轮生成一个小代码块，发现接口时先打印并读取真实文档返回，再按文档中的精确 API 名、参数与 JSON 返回类型构造后续调用；赋值只保留变量，观察结果需要打印，分页结果按文档遍历。沙箱与标准库限制、任务中的文件指向 `file_system` app 等说明同样公开；不加入任务解法或示例答案。

程序合法性与效果规则统一由 `native_program` 提供，桥接层不另造 AST 效果规则。公共系统提示仅描述环境与会话语义，代码或程序表示的格式及模式约束由各自的动作提示承载。PlanBench 的工具目录只从官方检索真正返回并进入公共历史的工具签名形成，保留此前仍公开可见的工具；隐藏 registry 不参与解码候选。参数引用与变量信息按公开历史处理，原生参数依赖、步骤、预算与评分保持。

PlanBench-XL 直接调用官方检索、工具执行和最终答案 handlers，保留工具发现权限、参数依赖、来源追踪、噪声候选和原生答案判断。策略自行生成检索内容、工具名、参数及答案，单步不包含整条 GT 工具链。主条件为官方 retail default：blocker 关闭、候选 noisy siblings、候选上限 30、环境上限 100 轮。原文 blocker 条件通过独立的同题条件配置登记，成绩与主条件分开报告。

## 划分与信息权限

家居环境使用原生 train；开发诊断来自 `eval_in_distribution`／valid_seen；正式测试完整保留 PoS 的 `eval_out_of_distribution`／valid_unseen 134 题。seen 用于开发，不与 unseen 正式结果混合。WebShop 直接使用固定官方 `gigpo-official/agent_system/environments/env_package/webshop/webshop` 中的 `WebAgentTextEnv`（`observation_mode="text"`、`use_small=True`、`human_goals=False`、固定打乱种子 `seed=42`，共 6,910 个官方购物目标），本项目保留现有三分法：`test` 为索引 `0..499`（500 题），`dev` 为索引 `500..1499`（1,000 题），`train` 为索引 `1500..6909`（5,410 题）。固定 GiGPO 的 `webshop/envs.py` 实际使用 `[500,N)` 训练（三分法只留在注释中），其示例训练脚本设置 `env.seed=0`，并由底层按该种子打乱目标；本项目明确保留 `seed=42` 和上述划分，不声称与该 GiGPO 运行设置具有相同训练集或完全相同的 500 个测试目标。ClinDiag 已从后续活动基准清单中移除。

LOCA 使用 8K 的同一批 15 个任务模板。test 固定 PoS 的种子 42、123、456、789、2024，共 75 例；train 固定 10001–10020，共 300 例；dev 固定 20001–20005，共 75 例。三个种子集合不相交，这项训练设置衡量同模板下的新实例泛化，不称为未见模板泛化。实际任务 ID、模板、实例种子和数据来源写入 manifest；不能依成绩改变划分。此前 AppWorld 0.2.0 与 PlanBench 自定义 180/57/90 划分随原数据保留。

LOCA 的 8K 是官方任务环境规模标签，不是将每次模型输入强制截断为 8K。PoS 默认将超出 8,000 字符的结构化工具结果物化为工作区内可读取的工件，并保留少量预览；这一观察呈现机制与工具接口一起保留并记录。模型上下文上限仍按本项目配置执行。

LOCA 完整分支恢复采用纯用户态隔离 QEMU 虚拟机（`src/prism_evolve/native_loca_vm.py` 中的 `LOCAVMSession`）：每个并发环境使用独立的可写 qcow2 增量镜像和虚拟机内工作区，将全部原生 MCP 工具进程及其数据库置于同一虚拟机内，通过 QMP `savevm`／`loadvm` 保存和恢复 CPU、内存、设备、运行中 MCP 进程及可写 qcow2 磁盘；宿主侧通过 `virtio-serial` 与 `--mcp-stdio-proxy` 桥接调用，并另行保存运行器的公共前缀、预算账本与随机状态。环境快照还深拷贝保存并恢复 PoS 宿主适配器的 `artifact_contracts`、`_workspace_file_index`、`_last_tool_success_counter` 与 `score`，使后续产物验证、文件变化观测与工具执行计数不残留其他分支状态；恢复前拒绝缺少这些字段的旧快照，不以当前分支值补齐。虚拟机只运行环境和工具，不需要 GPU，基础镜像预存 `boot_ready` 快照以实现亚秒级就绪恢复。官方说明见 [QEMU 虚拟机快照](https://www.qemu.org/docs/master/system/images.html#vm-snapshots)。

真值和凭据留在受信任的宿主评价侧，不向虚拟机提供宿主真值目录挂载；通过受控 `virtio-serial` 通道导入公开初态并导出供评价的任务产物。虚拟机快照不替代宿主侧计费、进程间连接重建和评价端状态管理，快照开销记入物理账本。两站均已在 `<external_root>/native-text-data-20261108/qemu/` 与 `loca-vm/` 完成纯用户态 QEMU、内核、initramfs 与 `loca-base.qcow2` 配置并通过桥接接入 `NativeLOCAEnv.save_snapshot`／`restore_snapshot`。

无测试反馈预编码、测试辅助标签训练或测试成绩 checkpoint 选择。各训练方法统一采用预算结束时的最终 checkpoint；三阶段训练和阶段快照保留，不反复开展完整 dev 选模。测试前冻结策略；PoS 每题重置信念。WorldEvolver 按原法保留跨题在线记忆，使用预登记的任务顺序，报告为 prequential 结果；该记忆不会回流训练。冻结记忆结果如另行生成，使用不同变体名称。

反馈利用量按 §6.2 在主体测试完成后测量，仅调度 Qwen2.5-7B 上 SPAR 的初始与最终两个 checkpoint。每环境预登记 24 个 dev 任务，按 scenario 分组、固定种子 20261105 进行与成绩无关的分组轮转选择，并保存原任务位置及实际清单。由初始策略产生锚点公共前缀，在第一个非终止信息动作后捕获物理状态；两个 checkpoint 共用该锚点、已付成本和随机流，各自只改变反馈可见性进行 visible/masked 续跑。完全没有可用信息锚点的任务记录为 `no_anchor`，报告覆盖率和有效样本数，不补题或伪造零值。结论限定于该固定初始策略锚点分布，最多 432 个反馈配对；不扩展到同行方法的反馈能力排名。

每环境×种子的两个 checkpoint 合用独立诊断总账本，上限为 9,600 次物理环境操作、25,165,824 个模型计算 token、172,800 秒。每条部署续跑仍遵守原每题上限；快照重放与实际调用均真实计账。额度提前结束明确保留 partial 和未估计结果，不扩大预算或填补缺项。反馈诊断与主测试完整性分别记录，不作为运行前置或阶段放行条件。

WorldEvolver 默认以空记忆开始，每个测试 split 独立初始化，并按固定顺序在 split 内 prequential 更新。`memory_initialization=train_prequential` 可显式选择训练任务预热的迁移变体；相应记忆、调用和成本单独记录，不混同于默认原法。

## 方法迁移与必要组件

GiGPO、InfoPO、AREW/T3 使用不同的官方 verl fork。新实现复用固定版本的纯数学核，保存函数源码文件、普通版本及必要兼容修补，并在共同 HF actor 上实现 token PPO 更新。身份为“同基座、原生环境迁移实现”；官方 trainer 运行与迁移实现分别记录。环境、分组、完整损失、token mask 和反馈构造共同决定该身份。

| 方法 | 本协议保留的机制 |
|---|---|
| GiGPO | 同题同初态8条独立轨迹；公开完整历史作为保守 anchor；固定官方默认 `official_step_weighted`，可显式选择论文的 `paper_trajectory_equal` 宏观归一化。 |
| Step-DPO | 训练参考仅用于首错定位和结果验证；冻结 reference 从同一公开正确前缀采样修正续写，成功后提取首步，与错误步骤构造 DPO；无有效 pair 时按事实记录。 |
| InfoPO | 每题5条轨迹；共同完整词表对实际下一动作的可见/遮蔽反馈评分，固定默认 `official_token_nonzero`、当前动作长度分配与官方门控缩放。论文逐 turn 归一化与该官方代码口径分开说明。 |
| AREW | 每题3条轨迹，采用论文和官方工具任务支持的 AS-only 通道；正标签来自原生状态进展、成功或真实新公开业务证据，硬错误为负，其余中性。默认固定官方 `minority_fixed`；可选 `paper_step_mean` 的独立未裁剪逐步骤似然间隔。当前环境没有原论文 BT 候选/真值契约，不将本变体称作 AS+BT。 |
| T3 | 每题5条轨迹，调用官方 τ² `_update_proxy_state`；以实际新增工具及稳定的公开业务证据判断进展，过滤回执身份，保留官方 hard/soft 容忍和仅训练截断。 |
| WorldEvolver | 官方 episodic/semantic memory、世界模型预测和 selective foresight；默认空记忆、split 内 prequential 更新，保留单次预算迁移口径及全部辅助调用。 |
| PoS | 官方 ReAct、完整 BeliefManager、consistency、progress、trapping 和 recovery；保留官方有界 JSON 修复及替换决策，全部调用计量。 |
| EnvProbe-Simple / Judge | 官方四维评分或 Judge 和选择流程；从公开信息生成只读业务核验动作，查询实际状态并计入预算；执行候选来自同次 actor 输出。 |

上述定义固定为本轮原生任务迁移协议。各方法保存固定官方源码与迁移约定；不将共同HF actor上的迁移训练称为上游分布式trainer或原论文数据集复现。真实终局低分不自动变成某个错误步骤，标签、有效监督与辅助调用成本逐项保留。基础设施异常直接传播；模型自然无效输出按本步原生反馈记录，不猜修。

Step-DPO 的标注与验证只使用登记 train 的参考；候选本身只从无参考的公开前缀由冻结 reference 采样，dev/test 禁止参考读取。LOCA 训练实例及从官方生成数据和评价条件构造目标状态参考均属于项目扩展，官方没有提供这 15 个模板的逐步金标准轨迹。独立 R/W/P 诊断使用 train 参考构造 W 示范，P 从 W 出发；示范及其成本属于该诊断身份。PRISM 和 RL 基线任务回报统一使用 U，train 进度仅可作为记录，不加入回报；测试 actor 不读取参考。

## 公共预算与真实成本

原生交互单位沿用各基准：家居环境为一个文本动作，上限 50 步；WebShop 为一次 `search[...]` 或 `click[...]` 文本动作，上限 15 步（沿用官方 `WebShop` 与 `GiGPO` 默认上限）；LOCA 为一次真实 MCP 工具调用，调用总数和 policy 轮数都不超过 100。表格同时标明交互单位，跨环境步数不直接混合。保留接口中的 AppWorld 代码块和 PlanBench 单动作继续按原定义计量。

训练、开发和测试分别记账。各训练方法每个模型×环境×种子共享同一资源上限：12,000 次物理环境操作、8,000,000 个模型计算 token、14,400 秒，任一到限停止。计算 token 分别保存输入、生成输出和 teacher-forced forward 数，并以三者之和执行上限。最多三个阶段，阶段进度依据实际共同预算消费，不以相同 SGD 次数代替资源配平。不同组大小保留原法，不足以完成的新组停止并保留 partial group。

每题动作上限分别为 50（`alfworld`）／15（`webshop`）／100（`loca`），模型计算 token 和墙钟上限仍为 262,144 与 1,800 秒。模型单次 actor 输出上限分别为 512（`alfworld`、`webshop`）／4,096（`loca`）；PoS 在三个活动基准上的 belief 上限均为 4,096，consistency／progress Sentinel 均为 512；LOCA 的两个 Sentinel 上限遵循固定官方 `configs/loca/pos.yaml`，不再使用 4,096。所有 actor、belief、Sentinel、judge、世界模型预测、记忆归纳、信息遮蔽评分和偏好样本生成共用计量层。错误、重试、快照恢复和前缀重放均记录；恢复环境不回滚账本。训练更新耗时与调用成本独立保存。

成功时间面积使用真实任务终局结果与提交前的实际任务耗时。跨任务模型加载单列一次性开销；单题内的控制与信念维护计入其耗时。仅环境物理重放属于训练反事实实现开销，不计入部署轨迹成本，仍计入训练总量。CPU 与 GPU 使用 OS monotonic clock，保留的 AppWorld 冻结时间逻辑不影响这一计量。

统一使用预算终点 checkpoint；`dev_budget` 保留为配置项。WorldEvolver `sf_confidence_pct=50` 是 `100*exp(mean_logprob)` 的绝对百分数阈值，对应 0.5 概率，属于预登记适配；论文校准阈值和 best-of-5 与当前共享预算下的 Success@1 分别报告。

## 消融与统计

本轮仅做 SPAR 自身的两项消融。Unweighted 只将两分支条件更新的 p/(1-p) 改为 0.5/0.5，根门控项系数保持 1。Unstratified 在同前缀、同恢复权限下自然采样两条完整策略续跑，使用两样本 RLOO。其余损失、前缀权重和预算定义固定，不另行调度普通 PG、Value 或 RLOO 实验。

主检验以 SPAR 与每个 baseline 的同题结果配对，按 benchmark×backbone 分层。相同题目在种子 42/43/44 下先形成任务均值，按原生 scenario 整组 bootstrap 和 sign-flip；成功率另按每个种子报告精确 McNemar。统一 Holm 校正整个登记比较族。报告种子均值和样本标准差，置信区间不平均端点，不将重复 seed-task 当作新任务。配对缺失、模板归属不一致或未完成单元存在时，先报告完整性状态；完整比较族齐备后生成正式显著性结果。

反馈变化是独立登记的次要检验族，含三个环境中的初末参数差值。每题先对有锚点的种子求平均，再按 scenario 整组 bootstrap 和 sign-flip，并对这三个检验统一 Holm 校正；种子均值和样本标准差另列。所有预登记任务均保留覆盖状态，统计目标明确条件于可用的初始策略锚点，不将 no_anchor 当作零差值。主结果完整性和反馈完整性分别报告，两者都完成才标记整项实验完成。

## Host A / Host B 联合调度

本轮先在 Host A、Host B 完成正式实验，Colab 保持备选状态。既有 Colab 配置、脚本、源码与环境归档保留，当前不部署、不连接 MCP、不申请 GPU，也不进入本轮 workload。是否后续使用 Colab，由双机正式运行表现和用户后续安排决定。

Host A 为八张 NVIDIA L40（每卡约 46–49 GB 显存）、144 个逻辑 CPU，GPU 3/4/2 分别承担 Llama-3.1-8B 的种子 42/43/44。Host B 为三张 RTX 6000D（每卡约 85.6 GB 显存）、152 个逻辑 CPU，GPU 0/1/2 分别承担 Qwen2.5-7B 的种子 42/43/44。每个种子的三个环境合为一个 resident cohort，依次处理。

默认 `configs/prism-native-v1.json`、`configs/prism-native-host-a.json`、`configs/prism-native-host-b.json` 保留六个 cohort workload（环境均为 `["alfworld", "webshop", "loca"]`）；实验与协调身份均为 `prism-native-pos-benchmarks-20261108`。配置只作准备，不自动提交队列。家居环境与 WebShop 分别在独立 Python 环境中运行，LOCA 使用独立运行依赖与纯用户态 QEMU 虚拟机快照后端。

A 每个三环境 cohort 为十一主方法，共 33 单元，三个种子合为 99 单元；B 每组为十一主方法加两项 SPAR 消融、三个环境，共 39 单元，三个种子合为 117 单元。全局登记 216 单元、51,048 正式测试回合与最多 432 个自身反馈配对。训练预算、LoRA、解码和统计参数保持。每个 cohort 的全部方法在其实际同节点同卡完成训练及逐题配对评测，完整非矩形比较族合并后统一 Holm 校正；两项消融仅统计 7B。

训练和评测均共享调度，A/B 保留每卡一个本项目模型 worker、各站配置中的 allocator 比例和原空余显存准入方式。额度按物理 GPU UUID 在本机 POSIX 文件锁下管理；真实占用持续采集，实际超额报错，失主额度保留诊断，不自动回收或降级。CPU 每 lane 默认 4 线程，其他用户作业保留。Host A 的 GPU0 因不可纠正 ECC 故障退出资源池。

AppWorld 的冻结日历继续作用于模拟器和原生 API，模型库从时钟模块扫描中排除。模型期限与资源计量使用操作系统单调时钟；单卡权重直接流式装载到选定显卡。LoRA 所需已保存激活按位移至 CPU，参数保留 GPU，保留 1 MiB 激活卸载阈值。模型权重、数据与官方源码沿用既有固定版本，精确 token 教师强制采用逐动作反向累积。

正式评测在同模型、同 GPU 的逐题配对块内轮换各方法，按预登记随机流平衡顺序，保存真实 UTC 边界和 GPU 负载。墙钟效用按实际共享负载解释。每个独立训练单元恢复相同初始 LoRA 与 critic，先完成登记训练及主测试，再做该 cohort 的反馈诊断。

每个源 scheduler 完成自身科学契约、源码和固定模型能力登记后，立即领取符合条件的已登记 pending cohort。所有 pending→launch 经 Host A 持久事务原子领取，本机双机监控用 `scripts/monitor_prism_native.py --servers-only --runtime-file ...` 通过 SSH 转发压缩控制消息；本轮无需 Google 或 Colab MCP。源 scheduler 优先领取原指定 GPU 的任务；空闲兼容节点可领取尚未开始的完整 cohort，保存原分配和实际执行身份。A 的固定 Qwen 副本位于 `/apps/users/adminuser/models/Qwen2.5-7B-Instruct`。运行中的 cohort 不拆分，失败不重新入队，没有超时重领、重复执行或伪造完成收据。

每次领取保存独立 `executions/<execution_id>/`、实际 host/site/GPU UUID、资源配置、原 manifest 与 bundle 编号、转交依据及授权。双机全部任务与反馈完成后，将 B 完整新运行目录传到 A，汇总两个源目录的全部 execution shard 与同 session 的 `completed-registry.json`，拒绝重复、缺失及身份不符。统计族与公式沿用原协议，全局完成只包含本轮 A/B workload。

本轮先完成代码修复并保持所有实验停止。现有记录按原版本保留；后续如明确安排运行，使用独立目录与实际源码、配置和方法身份。R/W/P 入口仅在配置、源码快照及阶段身份一致时可复用已完成阶段，未完成阶段、旧配置和旧日志均保留。
