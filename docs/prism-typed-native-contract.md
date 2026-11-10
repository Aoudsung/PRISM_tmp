# PRISM 公开原生程序契约

`prism-typed-native-v1` 用于 PRISM、Frozen θ₀ 和两个自身消融。效用、模式概率、分层估计器和消融公式沿用设计；普通基线使用自由原生 actor。程序语法和效应约束由 `native_program.program_schema(context, mode)` 同时提供给生成、重评分及执行入口。

公开上下文使用 `public-native-context-v1`。AppWorld 的 `apis` 是完整 `app.api` 名称到已公开详细签名的映射，签名包含 `method` 和参数列表；参数项提供 `name`、`required`。只有模型实际收到的详细文档可扩展映射。初始的三个文档 GET 入口和 `supervisor.complete_task` POST 签名由环境层依官方公开定义提供给所有方法。GET/HEAD 视为读类，POST/PUT/PATCH/DELETE 视为执行类。这里遵循接口声明的 HTTP 类别。

AppWorld 一个动作是 `before/action/after` 块。前后两个数组允许任意数量的辅助语句；中间 `action` 必须是所选模式的调用，或者 then/else 都为同模式完整块的条件分支。I 中所有调用均为公开读类；E 的辅助语句可以读取和计算，中间动作必须是公开写类或提交。循环可以执行零次，所以循环仅出现在辅助语句中。此布局使每条正常完成路径具有必要类别的调用；原生参数、认证或业务错误仍会中止执行并留下真实反馈。

模型生成每个 API 名、参数表达式、绑定名称和控制条件。普通 `call` 为 `{"call":{"api":"app.api","args":{...}}}`；绑定调用同时提供 `bind:"v_name"` 和 `body`。`let` 提供 `name/value/body`。局部引用使用 de Bruijn 索引：`{"local":0}` 指向最近绑定，1 指向外层绑定，语法只允许当前词法深度内的索引。展示名字由模型生成，局部引用不受同名遮蔽影响。跨动作 `{"var":"v_name"}` 只能选择当前公开上下文的绑定名，取本动作开始时的公共变量快照。

普通绑定的 body 是语句数组；必需 action 中的纯 let，其 body 是完整模式块。E 中若用读取调用来绑定后续写操作的参数，该绑定的 body 同样是完整 E 块。必需的同模式调用已经满足效应条件，其绑定 body 可以是普通语句数组。`if` 的两个分支保持各自词法作用域。`for_each` 提供 `bind/in/limit/body`，循环项成为 body 最近的 local；`repeat` 提供 `bind/count/body`，local 为零起始迭代序号。limit 和 count 由模型提供有限非负整数，`break_if` 仅在循环体出现。原生接口调用、执行时间和 token 预算继续约束实际循环开销。

纯表达式支持标量、`literal` JSON 容器、公开 `var`、词法 `local`、字段或索引 `get`、`slice`，以及 `PURE_OPERATIONS` 中公开的固定操作。操作覆盖算术、比较、布尔、字符串、集合、排序、聚合、JSON 和日期时间。API 调用目标始终是目录中的完整常量名称；没有动态 callee、任意 imports、eval/exec/getattr、用户函数或类、OS 访问和无界循环。算子参数数量受同一语法约束，业务值、数据索引、转换和原生参数错误沿原生异常路径返回。

以下 I 程序调用并公开应用目录，全部节点都由模型生成：

```json
{
  "before": [],
  "action": {
    "call": {"api": "api_docs.show_app_descriptions", "args": {}},
    "bind": "v_apps",
    "body": [{"emit": {"local": 0}}]
  },
  "after": []
}
```

E 使用同样结构，中间调用换为已公开的业务写接口或提交。编译器不会主动选择提交时机、填写参数或补充任务路线。

`compile_program` 按共享合法程序语法确定性产生一个官方 Python 代码块。`emit` 转换为 `print`；固定标准库 imports 只实现已公开的纯操作。官方执行顺序、部分状态变化、异常和实际调用次数保持。返回 action 保存 `typed_program`、`typed_program_version`、普通 `program_id`、`generation_token_budget`、`declared_variables` 和同值的 `definitely_assigned_variables`；模型原始 token 与转换代码分别记录。原文程序用于概率评分，转换代码不再次当成生成 token。

编译块最后调用环境提供的 `_prism_complete(program_id, names)`，由真实原生 stdout 发布 `typed-public-v1` 完成回执，附带普通动作序号。names 只来自静态正常完成分析：顺序语句取并集，条件分支取交集，可能为空的循环不新增名称。回执只公开完成事实和名字；值仍由模型显式 emit。程序异常时自然无法到达末尾回执，原生部分状态变化仍保留。目录来源保存实际公开观测原文、动作序号和程序编号；可见历史移除某条反馈时，该反馈独有的 API 和绑定来源同步移除。更早仍可见的来源继续有效。

PlanBench 使用该契约的单动作形式。I 为 `retrieve_tools` 及官方公开 retrieval schema；E 为一个已检索工具的 `tool_call`，或 `final_answer`。工具的 `arguments_schema` 和标识来自已公开检索结果。一次程序只对应一次原生操作，保持基准步数和 handler 语义。

约束解码器使用同一 schema 和完整原始字符串构造合法 token 集；teacher-forcing 和冻结参考复用相同集合归一化。EOS 仅在完整合法值结束后允许。预算结束时保留真实截断，不补闭合、不重采样。环境层只接收明确编译出的程序，并用真实原生结果计分。

词法 schema 按 `generation_token_budget` 展开，这是本动作既有生成预算提供的保守深度上界；没有独立固定层数预算。该字段不进入模型提示。采样和评分保存并复用相同 schema 与预算，环境按相同上下文确定性编译。缓存使用 schema/context 的原值及普通契约编号。源文件快照、配置和原始日志直接保存，不生成文件、模型、程序或目录指纹，也不设置摘要比对放行步骤。执行入口不会追加采样器没有实现的作用域过滤。

本次只进行静态实现与接口对齐，没有运行单元测试、模型或独立实验。
