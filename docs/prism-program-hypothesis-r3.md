# PRISM program-hypothesis-r3

本协议独立于 `program-gate0-r2`。R2 的 30 条完整冻结基座训练轨迹作为唯一经验输入，原件及 FAIL_INPUT_COVERAGE 结论保持原样。R3 不使用正窗、错误文本正则或多次 Loop 见证门，不追加采集。实现位于 `program_hypothesis_r3.py`、`program_hypothesis_runtime_r3.py` 和 `run_program_hypothesis_r3.py`。

每个完整 episode 落盘一个 RepairContext。工作视图按原始公共文档内容去重，完整保留动作顺序、动作前后公共观察和目录引用；目录以固定完整短索引呈现，模型可按工具名展开原生 schema。只在提案实际选定 document ID、UTF-8 span 或 JSON pointer 后闭合证据，不预枚举片段组合。源题事后信息只用于显式标记的 source repair；跨实例的绑定根仅包括当前目标的 goal、observation、last_return、history 和运行时 values。私有终局评估、诊断和 receipts 保留在原始记录中，均不进入模型公共视图。

每集最多两次真实冻结 Qwen2.5-7B 调用。首调用可以直接生成主提案或明确拒绝，也可定位证据、锚点与所需 schema，再由末次调用生成 AST。30 集至多 60 次调用属于独立研究预算，所有账本明确记录 `online_rho_satisfied=false`。AST 支持当前状态 Bind、集合级 SemanticBatch、有限 Frontier、带 fuel 的 Loop、含 Unknown 分支的 If3、真实 Call、Return 和 Yield。每个节点标记效果类型与 HYPOTHESIS；所有原生效果保留。Call 的数据叶只能引用当前运行变量，工具名可以引用此前未执行的公开 API。

B、R、P 共用 `PublicRuntime.actor`。它先真实生成通用工具名，再仅按选中工具的原生 schema 约束参数。R 直接使用同一完整经验重规划，P 执行该经验产生的程序，退出后使用带相同经验的冻结 fallback。B 无经验。所有臂使用相同目标、种子、环境步数、模型 token 和墙时上限。每组持久保存并比较空经验时共同初始策略的真实 reset 输入、工具索引、采样配置与冻结模型身份；不增加首动作 probe。

正式运行先完整生成 30 个主提案，然后统一执行所有 30 个源题的 B/R/P。源题锚点之前的必要因果动作由真实环境恢复，首次 reset/return 分歧即停止读取来源后继。模型与原生动作分歧后的结果全部由真实环境裁决。开发目标每 benchmark 取 native dev 中按 ID 排序首个、且不属于 30 个源题的实例；目标身份完全留出源经验。正式新目标每 benchmark 取 native test 中对应的首个新实例。两类身份、原生 split 与源题修复分别登记，均按 episode ID 选首个有效提案，不按终局结果挑程序。配置固定三种轮转臂顺序。

TrialRecord 保存真实当前状态绑定、动作及回执、集合语义调用、fallback、快照、异常恢复与一次重试、局部退出、完整原生终局和全成本。没有已知原生终局成功时 Y 和效用均为零。每项实际工作才申请对应可兑现上界；候选产出后独立申请 trial，不在编译前占用依赖 trial 预留。真实已发生费用始终保留。物理账本记录输入/输出 token、prefill/decode GPU 时间和总调用墙时；外层单列 reset、快照导出/序列化/恢复、公共投影、schema contract 实际哈希、结果导出及 setup。内外层嵌套区间不重复求和。

服务器正式入口为 `.venv-prism-native/bin/python -u scripts/run_program_hypothesis_r3.py --config configs/prism-program-hypothesis-20261112-r3.json`。本轮不运行单元测试、阶段审计或额外被动采集。`r3_report.json` 分别给出源题修复、跨实例开发绑定、正式新目标终局迁移，以及 B/R/P 的真实原生 Y、效用、token、墙时和环境步数。程序收益统一以 P-R 报告；B 相对 R2 历史源题基座的共享接口收益另列。所有提案失败和执行失败保留。
