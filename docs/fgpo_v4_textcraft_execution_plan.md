# Host A TextCraft 正式实验排程

2026-11-19（洛杉矶）。Host A 用于 TextCraft。BiPACE 作者提交 f19f6fe 的原生环境已在 Host A 离线部署，含 860 个配方与固定的 318 个训练目标和 79 个验证目标。源码及新增依赖先在本机下载，再上传离线安装。原生验证目标的动作、库存和终局判分已通过运行验收。[环境部署记录](../outputs/textcraft-bipace-offline-20261119/host-a.md) · [作者发布环境](https://github.com/TianxiangZhao/BiPACE/tree/f19f6fe1d140dd342d81b8e797f09c37c620b42f/agent_system)

## 正式单元

统一从同一份 Qwen2.5-7B-Instruct 开始。每个方法跑种子 0、1、2，每种子训练 50 轮，每轮 16 个原始任务组，每组 8 条独立轨迹。每条最多 30 次交互，每次最多 128 个生成 token，总上下文 8,192。组内共享相同目标、配方顺序与起始库存，抽样流独立。各方法同种子使用同一训练目标次序。50 轮取自 BiPACE TextCraft 的 50 次训练迭代窗口，30 步和 128 token 是本研究实际实施的较短预算。每单元在预定终态评完整 79 道题，报告整体及分配方深度的成功率、环境步、生成 token 和实际 GPU 小时。[运行配置](../configs/fgpo-r3-host-a-textcraft-20261119.json) · [BiPACE 训练配方](https://github.com/TianxiangZhao/BiPACE/blob/f19f6fe1d140dd342d81b8e797f09c37c620b42f/examples/bipace/textcraft/qwen7b_4xh100/train_gigpo.sh)

| 排程 | 先跑的单元 | 后续依次接续 |
| --- | --- | --- |
| Host A GPU 3+4 | F-GPO 0、1、2 | Obs-Only 0、1、2 → No-Support-Weight 0、1、2 |
| Host A GPU 1+7 | GraphGPO 0、1、2 | G2PO 0、1、2 → No-Transfer 0、1、2 |

F-GPO、GraphGPO、G2PO 共九个主比较单元，检验预定终态成功率和实际成本（C1、C5）。GraphGPO 与 G2PO 在 BiPACE 的 TextCraft 结果中缺失，迁移其现有发布算法和更新配方，两个图共享同一公开观测作为状态键。另九个单元分别检验历史条件匹配（C2）、下一状态续值共享（C3）、支持权重（C4），直接复用三个完整 F-GPO 单元作对照。合计 115,200 条训练轨迹、1,422 道次终评。BiPACE 表 3 已报告 GRPO、GiGPO、HGPO 和 BiPACE-Q 的作者协议下峰值验证成功率，作为有明确来源的文献结果引用。[机制定义](fgpo_math_contract.md) · [BiPACE 表 3](https://arxiv.org/html/2606.25556v1#S4.SS3)

## 模拟器与结果

训练和评测调用同一个原生 TextCraft worker。动作后的文本反馈和作者公开的库存一起进入观测；完整策略使用实际看到的历史。模拟器收到模型真实声明的命令。缺失或歧义的动作消耗一步原生无效命令。成功以模拟器的 `won` 为准。F-GPO 保留原奖励、混合组与轨迹归一化更新，图方法保留各自已发布的估计量和更新语义。计算和方法诊断从正式日志提取。

Host A 用同一部署版本回放了真实验证任务 `minecraft:jungle_boat`，修复了动作结果在进程间传输时的序列化错误。回放确认三个主比较方法的首轮输入 token 一致，原生环境在第六步判定合成成功并返回 10 分。缺少动作标签的回复实际提交空命令，在第 30 步自然终止并判定失败。两类终局记录均通过六种条件共用的终评接口。验证汇总要求完整 79 道题。[接口回放记录](../outputs/textcraft-bipace-offline-20261119/interface-replay.json)

在两组 L40 上按各自队列从种子 0 起跑，单元从自己的最近原子检查点恢复。Host A 的 WebShop seed 0 结果和 seed 1 检查点另行保留。进程及队列使用独立会话，训练、采样、终评和最终结果均落盘。输出仍在 Host A，保存容量按现行文件系统占用排程。[WebShop 交接记录](../outputs/hosta-appworld-direct-rollout-20261118/webshop-release-for-textcraft-20261119.md)
