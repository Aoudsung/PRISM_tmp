# Third-party implementation notices

The published-comparator adapters in `src/fgpo/published_baselines.py`, batch
mathematics in `src/fgpo/published_batching.py`, and ALFWorld prompt support in
`src/fgpo/baselines.py` include adaptations of
[verl-agent](https://github.com/langfengQ/verl-agent), snapshot
`20bd331bdbc9026a5668e11362178e10ab7400c8`.

Copyright 2024 Bytedance Ltd. and/or its affiliates; Copyright 2025 Nanyang
Technological University (NTU), Singapore, and the verl-agent (GiGPO) team.
The original project distributes these sources under
the Apache License, Version 2.0. Its license text is retained in
`THIRD_PARTY_LICENSE_APACHE_2_0.txt`.

The adapted components are `gigpo/core_gigpo.py`, `recipe/hgpo/core_hgpo.py`,
`recipe/GraphGPO/core_graph.py`, the public action/feedback history construction
in `recipe/GraphGPO/alfworld_envs.py` and
`recipe/GraphGPO/graphgpo_env_manager.py`, the ALFWorld prompt and action
projection, and `SimpleMemory`. Batch layout adapts `adjust_batch` in
`agent_system/multi_turn_rollout/utils.py` and the equal-size Karmarkar--Karp
partitioning in `verl/utils/seqlen_balancing.py`; virtual-rank weights preserve
the reduction in `verl/workers/actor/dp_actor.py`. Exact source URLs and
publication provenance are recorded in `src/fgpo/baseline_specs.py`.

Local changes replace verl's batch objects with immutable trajectory records,
use scalar float32 credit calculations, compute unit-edge shortest distances
with reverse breadth-first search, preserve source sample-copy multiplicity,
balance order and microbatch weights through a saved virtual-rank plan, and
merge repeated row coefficients only within the same optimizer update. Native
prompts use the study's batched Qwen execution backend. The study uses its own model placement,
bounded memory implementation, optimizer storage, checkpointing, and cost
ledger. GiGPO's released full-finetuning ALFWorld script targets Qwen2.5-1.5B;
the common study backbone is explicitly transferred to Qwen2.5-7B. HGPO and
GraphGPO use their released 7B recipes. AppWorld uses an explicitly described
native-Python interface transfer. Official training-task catalogs and the study's
seeded task order supply the common environment schedule.
These components implement the released recipe's algorithm semantics; the
study does not claim a bit-for-bit reproduction of the authors' complete
training stack.
