# Gate 0–2 共享公开程序 IR

本接口对应 2026-11-12 Gate 合同 §2、§3、§5。版本为 `prism-public-program-ir-v1`，实现入口为 `native_program_ir.py`、`native_program_execution.py`、`native_trace_compile.py`。当前实现与服务器实验的完成状态分别记录，本文不声明 Gate 已通过。

主代理批准的最小约定如下。`Frontier` 仅枚举当前公开证据中来源闭合的有限 Ref 集合，结果是当前快照的 `RefSet`；`complete_collection=false` 是强制字段，枚举不表示隐藏全集完成。`Loop` 对该有限集合逐项绑定 `item_value`，每轮以及每个节点执行前检查预算；预算耗尽、来源失效、无法判定或无法重新绑定当前原生动作时走 `UnknownYield`，禁止无界循环。`If3` 恰有 `T/F/U` 三条分支；缺公开材料时使用 U。初始受控谓词为 `exists`、同类型 `equals`、文本 `contains`、数值 `number_le`；仅对所引用公开对象作判断，`exists` 不从部分列表推出全集不存在。未登记的语义算子拒绝准入。

每个 Source 有稳定 ID、来源事件序号、公开文档 ID、解码后 UTF-8 字节区间、值及类型。区间必须与原文一致。参数具有类型和来源，节点具有 `reads`、`defines`、来源列表、效果类别、历史效果、RNG 命名空间与显式缓存引用。类型、来源、支配关系及版本身份由确定性代码检查。公开目标使用事件序号 -1，第 i 个动作之前可见的来源序号不大于 i。调用后才能把该次返回交给程序。

活性以分支并集、循环最小不动点计算；死 `PURE_TOTAL` 节点可删，未消费的 `AUX_CALL` 不发请求。辅助请求不写 actor 历史，使用独立 RNG 域，缓存消费者显式引用。`NATIVE_CALL` 默认未知效果并触及真实历史，静态活性不能删除它。原生删除产生新的 UNTESTED 版本，清空原执行/成功计数，重新检查来源与绑定。改变语义 batch 字段需要新的 semantic schema 身份和执行证据。

循环折叠要求至少两个不同绑定的真实消费片段，保持连续原生效果区间，并验证替换绑定后的动作结构相同。重复错误或相同动作重复本身不构成循环见证。新分支作为 `GRAMMAR_HYPOTHESIS` 对应的 `hypothesis=true` 节点保留，`unexecuted_nodes` 必须完整列出。节点来源类别使用 `LOCAL_WITNESS` 或 `INTERFACE_REPAIR_HYPOTHESIS`，未执行内容不继承实测成功。

严格源回放使用懒动作流。来源读取器只提供当前待比较动作及延迟交付观察的函数；首次动作分歧后不交付该观察，也不读取后继来源事件。状态使用合同的六个枚举，回放的环境执行和成功计数均为零。公开来源的独立审查仍是合格 AST 计数的必要条件。
