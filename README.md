# PRISM-Evolve

PRISM-Evolve 在原生长时程环境中，以同前缀 I/E 分支采样及目标保持概率加权更新训练真实 LoRA 策略。数学定义见[设计文档](docs/prism-evolve-design.md)，实验与动作表示见[原生协议](docs/native-experiment-protocol.md)，环境及部署说明见[部署文档](deployment/README.md)。

当前修复后的 r3 正式实验按用户自动修复续跑授权执行，Colab 暂缓。后续活动基准为家居文本交互环境 ALFWorld（`alfworld`）、电商购物交互环境 WebShop（`webshop`）和 LOCA-Bench（`loca`）；ALFWorld 与 LOCA 的测试设置以固定 PoS 源码为主要参照，WebShop 使用固定 GiGPO／verl-agent 的原生环境及本项目保留的划分与种子；PRISM 与训练基线的共同任务回报使用设计文档的成功时间效用 U。旧结果保持原身份，本轮运行单独记录。

当前已编写三环境桥接、LOCA 的 15 个模板训练参考和基于纯用户态 QEMU 的进程／磁盘快照。LOCA 快照另保存并恢复宿主适配器的产物契约、工作区文件索引、工具成功计数基线与分数，避免分支间状态残留；缺少这些字段的旧快照不能用于恢复，须重新捕获。原批次异常现场保留；WebShop 恢复与 LOCA 首次模型输入已在 r2 通过，r3 修复终态评分时暂停 QEMU 后再次同步造成的超时。

## 动作表示与方法身份

活动配置使用 `action_contract_version=prism-native-text-v1`。Frozen Base、PRISM、Unweighted、Unstratified 在 I/E 模式下直接生成原生命令或原生工具 JSON；Frozen Base 是同接口、同约束、同初始参数 θ0 的冻结对照。GiGPO、InfoPO、AREW、T3、Step-DPO 保留不受 PRISM 门控的原生动作 actor；PoS、WorldEvolver、EnvProbe 使用 `official-controller-native-v1`。每项实际身份写入 `manifest.actor_contracts`。

三环境走原生文本／工具调用接口，跳过 `native_program.py` 的 PRISM JSON AST。家居环境的 I 包括观察、移动和开容器，属于用户确认的可改变状态的信息动作扩展；WebShop 的 I 包括搜索、导航和查看详情，E 包括购买选项及购买提交；LOCA 按公开工具能力划分 I/E，信息工具仍可能产生原生缓存、游标等状态变化。InfoPO 在共同完整词表上对可见／遮蔽反馈评分，EnvProbe 使用独立核验候选。官方终态评分、预算内成功和项目效用分别记录。原有 AppWorld、PlanBench-XL 的程序接口、数据和结果保留。

## 双机配置与运行状态

r7（`configs/prism-native-qwen7b-main11-20261109-r7.json`）修订 PoS 一致性 Sentinel 的返回容器：原模型返回裸问题列表时，消费者按对象读取导致属性错误；现在仅将该列表无损封装为 `{"issues": [...]}`，原始／规范化返回分别留日志，内容、调用预算和其他角色返回保持。完成的训练阶段继续继承，已完成的有效首题结果按各来源 cohort 分组复用；WorldEvolver 同时载入真实的题后记忆 checkpoint 与完成题目顺序，保证在线记忆连续性。原在训未完成阶段保留成本并按原预算重跑，范围与每小时检查保持。

r6（`configs/prism-native-qwen7b-main11-20261109-r6.json`）修复 LOCA InfoPO 的遮蔽反馈评分提示：原采样已使用单次工具目录，遮蔽评分却按旧序列化恢复重复目录，导致 879,611 计算 tokens 时提前以 `context_window` 结束且未更新参数。现在评分与采样使用同一公开提示序列化，仅替换规定的反馈内容，动作 token、全词表支持及 InfoPO 优势公式不变。r6 复用 r5 已完成的有效训练阶段，排除这项受影响的 LOCA InfoPO；原在训未完成阶段因无优化器续存状态，从同初始参数和原完整预算重跑，所有旧成本及现场保留。十一方法、seed42、三环境、33 单元／7,799 回合及并发最多三保持。

用户最新要求实验每小时检查一次，当前按台北时间每小时第 9 分钟触发同对话提醒；全部实验及比较汇总完成后关闭提醒。实际故障仍按 `rules.md` 自动修订续跑，全局最多三个实验并发。r5 的 `runtime_estimates.json` 继续保存实测训练速率、参考测试逐题耗时及九项 baseline 的主动运行耗时估计，供进度判断使用；尚未实测的方法使用明确标注的先验，估计不包含队列等待，不用于性能结论。

当前主比较扩展为 `prism-qwen7b-main11-s42-20261109-r5`（配置 `configs/prism-native-qwen7b-main11-20261109-r5.json`）：Qwen7B、seed42、三个环境、十一主方法，共 33 个单元、7,799 个完整测试回合。新增九项 baseline 为 GiGPO、Step-DPO、InfoPO、AREW、T3、WorldEvolver、PoS、EnvProbe-Simple、EnvProbe-Judge；各方法在同环境 cohort 中按既有完整预算排队，三张卡各最多一个 worker，全局最多三个并发单元。r3 的 PRISM／Frozen 训练产物及 r4 完成的所有参考策略对保留并复用，不重复训练或调用。原双方法的时间次序与本次十一方法的次序分别记录，复用记录保留原时间及原方法位置，不声称所有方法曾在同一时间块交错执行。LOCA memory 的缺省相对存储路径按当前公开工作区解析，沙箱边界保持；无门控原生 LOCA actor 复用单次工具目录提示，保留全部 I/E 工具并继续自由原生输出。

当前续评测版本为 `prism-qwen7b-first3-s42-20261109-r4`（配置 `configs/prism-native-qwen7b-first3-20261109-r4.json`）。r3 三环境已用尽登记训练 token 预算，终点更新分别为 alfworld 0、WebShop 67、LOCA 0。LOCA 第 11 个测试目标因上游 terminal 的 `allowed_dir="."` 默认别名覆盖 `agent_workspace` 而初始化失败；r4 在上游构造配置前将缺省别名明确设为当前公开工作区，不放宽目录隔离。r4 复用六个明确身份的初始／终点策略产物，不重新训练；已完成的全部同题策略对按原种子、次序和 GPU 身份保留原评分、耗时及成本，并逐条记录 `evaluation_reuse`。后续目标继续全量评测，保持总计 1,418 回合、并发最多 3；旧现场独立保留。

用户已授权后续实际问题按 `rules.md` 修订代码或方案并自动续跑，继续保持最多三个并发实验。r2 因 LOCA 终态评分在冻结 QEMU 后再次同步而止损；WebShop 停止前七组中四组有真实参数更新。r3 删除已冻结状态下的重复同步，使用工具返回后、冻结前同步的工作区副本评分。r2 未保存可恢复的中途优化器状态，因此 r3 三个训练单元均从原初始参数开始，r2 成本及部分轨迹单独保留，不合并为连续训练。

当前运行身份为 `prism-qwen7b-first3-s42-20261109-r3`，配置为 `configs/prism-native-qwen7b-first3-20261109-r3.json`，Host B 输出目录为 `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-first3-s42-20261109-r3/`。预算、模型、种子、完整测试范围及最大并发 3 与首轮相同；成功完成后生成同名 `-comparison/` 配对汇总目录。原首轮身份与中断记录继续保留。

当前首轮使用 `configs/prism-native-qwen7b-first3-20261109.json`：Host B 的 Qwen2.5-7B-Instruct、seed 42，在三个环境各完成一个 PRISM 训练单元，以及 Frozen Base 初始策略与训练终点的完整同题评测，共 6 个评测单元、1,418 个测试回合。三个单环境 cohort 分别在 GPU 0/1/2 执行，`max_workers=3`、每卡最多一个本项目 worker；训练预算及测试划分保持。此轮关闭额外反馈诊断，完成后由用户决定扩展。独立输出为 `outputs/prism_evolve_pos_benchmarks/prism-qwen7b-first3-s42-20261109/`，实际执行状态见 `pipeline_state.json`、`scheduler_state.json` 与各 execution 的原始记录。Host A 通过 `scripts/relay_prism_host_b.py` 仅承担协调，不加载模型；协调记录允许 Host A 作为计划外的无模型协调来源登记。

三份活动配置为 `configs/prism-native-v1.json`、`configs/prism-native-host-a.json`、`configs/prism-native-host-b.json`，预留的新实验及协调编号为 `prism-native-pos-benchmarks-20261108`，输出根为 `outputs/prism_evolve_pos_benchmarks/`。这些配置不自动启动运行；后续运行采用新目录，保留旧现场。

该首轮已停止，6 个评测单元均未完整完成。WebShop 训练在分支恢复时出现原生状态不一致；固定上游的 `reset` 会复用含终止／奖励字段的旧会话，修复为按原登记目标重建该会话。LOCA 首题在首次动作前触及 32K 上下文限制，训练参数更新为 0；修复将固定工具目录在模型输入中仅序列化一次，保留原工具描述与参数，完整原始反馈及解码约束仍保存。ALFWorld 停止前首组为零训练信号。原始版本、轨迹、成本及中断记录全部保留；修复后的 `configs/prism-native-qwen7b-first3-20261109-r2.json` 采用独立新身份，训练预算、32K 上下文、完整测试集、模型和并发上限保持，用户已明确批准并启动独立 r2 批次。

| 基准 | 训练／开发／测试 | 每题动作上限 | 原生成功判定 |
|---|---|---:|---|
| ALFWorld | 原生 train／valid_seen／valid_unseen 全部 134 题 | 50 | 官方 TextWorld `won` |
| WebShop | 固定 `seed=42` 打乱后，train `[1500,6910)`／dev `[500,1500)`／test `[0,500)` | 15 | 原生终止且购物分数等于 1 |
| LOCA | 同一 15 模板，新种子 train 300／dev 75；官方五种子 test 75 | 100 | 原生 `claim_done` 终止且评分不小于 1 |

WebShop 保持 `use_small=True`、`human_goals=False` 和现有三分法，不随 GiGPO 的训练脚本改动。固定 GiGPO 执行代码使用 `[500,N)` 训练、示例脚本 `env.seed=0`；因此本项目不声称与其训练划分或前 500 个测试目标身份完全相同。LOCA 的 8K 是任务环境规模标签，不是模型上下文截断上限；训练／开发只衡量同模板新实例泛化。PoS 在三个活动基准上的 belief 上限均为 4,096，consistency／progress Sentinel 均为 512；actor 上限分别为 512／512／4,096。详细划分、模型和官方论文差异见[原生协议](docs/native-experiment-protocol.md)。

既定正式计划保留 7B/8B、三环境、三个种子的完整主比较，以及 7B 的两项自身消融，共 216 单元、51,048 测试回合。Host A 的 Llama8 与 Host B 的 Qwen7 各有三个三环境 cohort；GPU 分别为 3/4/2 和 0/1/2。后续正式或短诊断范围由主代理另行安排，预留完整计划不自动成为新运行请求。

旧宏动作结果、原生 v6、固定短诊断、checkpoint 与停止现场分别保留。v6 的外部监控协调故障与短诊断结果已登记；已有零信号、预算截断和未形成偏好对按实际记录解释。主代理依据真实产物验收，进程存活和 optimizer 调用不作为学习改善的替代证据。

## 依赖与研究记录

Host A/B native actor 环境保留 `xgrammar==0.2.8`、`apache-tvm-ffi==0.1.15.post2` 及原 torch／transformers。ALFWorld、WebShop 与 LOCA 使用独立环境依赖；当前三个基准的环境执行与评分无需外部商业 API 密钥。LOCA 工具在独立 QEMU 工作区中执行，真值保留在受信任宿主评价侧。ClinDiag、AppWorld 和 PlanBench-XL 不在活动清单，既有实现、数据和结果保留。数据准备、依赖接入、源码修复与实际验证状态分别记录，配置文件本身不构成已验证声明。

固定来源：PoS [`6818cfa`](https://github.com/luoyu100/PoS/tree/6818cfa6434dbc91fcf67f0b3eb12a91db47433f)，GiGPO／verl-agent [`20bd331`](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)；PoS 内置 LOCA 来自 `hkust-nlp/LOCA-bench@8b6fac49d9edd92922593e703b74ea255357c3ec`。项目统一回报 U、训练设置和动作包装均单独披露，不将迁移实现冒充完整官方实验复现。

mgt01 的 `.research/research.sqlite` 为单写研究主库；本地副本只读。服务器入口为 `PYTHONPATH=src .venv/bin/python -m prism_knowledge --db <absolute-db-path> status` 与 `context --task R5.1`。通过 `apply` 更新必要记录，实际交付后按普通文件方式同步。保留源文件快照、配置与原始运行记录，不调用额外摘要或完整性比对流程。工作布置和回报按 `rules.md` 执行，AGENTS及旧结果保持。
