"""Published comparisons, own credit ablations and recoverable formal work."""
from __future__ import annotations
from dataclasses import dataclass,field
import json
from pathlib import Path
import random
import signal

from .policy import CausalPolicy,clock
from .records import CostLedger
from .optimization import OffloadedAdamW,UpdateCursor,WeightReference,ppo_update
from .reference import prepare_round
from .credit import advantages
from .baselines import collect_native_group,collect_native_trajectory,baseline_advantages
from .baseline_specs import PUBLISHED_SOURCES, published_recipe
from .published_batching import plan_published_batch
from .public import TERMINATORS
from . import checkpoint
from .protocol import (PROTOCOL, OWN_METHODS, METHODS, METHOD_ROLES, require_protocol,
                       resolve_method_config, effective_training_identity, require_training_identity)


def _mixed_groups(groups):
    return [g for g in groups if 1 <= sum(1 for t in g.trajectories if t.official_success) <= len(g.trajectories) - 1]


class TrainingInterrupted(Exception):
    """Managed interruption after saving a complete operation boundary."""


@dataclass
class EngineState:
    round: int = 0
    phase: str = "collect"
    group_cursor: int = 0
    groups: list = field(default_factory=list)
    task_order: list = field(default_factory=list)
    reference_data: object = None
    advantages: dict = field(default_factory=dict)
    credit_diagnostics: dict = field(default_factory=dict)
    update_cursor: object = None
    published_batch_plan: object = None
    training_log_offset: int = 0
    evaluation_completed: dict = field(default_factory=dict)
    completed: bool = False


class Engine:
    def __init__(self,config,output_dir,resume_from=None,*,ledger=None,startup_receipt=None):
        self.config = resolve_method_config(config)
        method = self.config["method"]
        placement = {key:self.config[key] for key in (
            "allocated_physical_gpu_ids","allocated_logical_gpu_ids","rng_physical_gpu_ids","rng_logical_gpu_ids",
            "compute_physical_gpu_ids","compute_logical_gpu_ids","model_placement_explicit") if key in self.config}
        self.config.update(placement)
        require_protocol(self.config)
        if method not in METHODS:
            raise ValueError(f"Unknown formal study method {method!r}")
        self.method,self.benchmark,self.seed = method,self.config["benchmark"],int(self.config["seed"])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.config.setdefault("checkpoint_interval_rounds",8)
        self.config.setdefault("retain_training_artifacts",False)
        if int(self.config["checkpoint_interval_rounds"]) < 1:
            raise ValueError("checkpoint_interval_rounds must be positive")
        self.config["artifact_dir"] = (str(self.output_dir/"trajectories")
            if self.config["retain_training_artifacts"] else None)
        self.gpu_ids = tuple(int(i) for i in self.config["gpu_ids"])
        self.compute_gpu_ids = tuple(int(i) for i in self.config.get("compute_logical_gpu_ids",self.gpu_ids))
        if (not self.compute_gpu_ids or len(set(self.compute_gpu_ids))!=len(self.compute_gpu_ids) or
            not set(self.compute_gpu_ids).issubset(self.gpu_ids)):
            raise ValueError("Model computation must use an explicit subset of the allocated logical devices")
        self.ledger = ledger or CostLedger(self.output_dir/"costs.jsonl",gpu_ids=self.gpu_ids)
        if startup_receipt is not None and not self.ledger._startup_credited:
            self.ledger.credit_startup(
                pre_engine_seconds=startup_receipt.get("pre_engine_seconds", 0.),
                startup_details=startup_receipt.get("details"),
                cold_asset_receipt=startup_receipt.get("cold_asset_restore"),
            )
        checkpoint.seed_everything(self.seed,self.gpu_ids)
        self.rng = random.Random(self.seed)
        self.state,self.stop_requested = EngineState(),False
        self.checkpoint_path = self.output_dir/"checkpoint"
        from .checkpoint_gcs import configured,GCSCheckpoint
        self.storage = GCSCheckpoint(self.config,self.checkpoint_path) if configured(self.config) else None
        self._saved_checkpoint_path,self._saved_model_version = None,None
        self._signals = {}
        self._storage_finalized = False
        self.policy,self.optimizer,self.reference,self.teacher = None,None,None,None
        if resume_from is not None:
            self.checkpoint_path = Path(resume_from)
            self._restore_checkpoint(checkpoint.load(resume_from,storage=self.storage))
        else:
            self.policy = self._make_policy()
            self.optimizer,self.reference,self.teacher = self._make_learning_state()

    def _make_policy(self,path=None):
        start = clock()
        policy = CausalPolicy(self.config["model_path"],gpu_ids=self.compute_gpu_ids,ledger=self.ledger,
            trainable=True,checkpoint=path,
            free_memory_gib=self.config.get("gpu_free_memory_gib"),
            activation_offload=self.config.get("activation_offload",False),benchmark=self.benchmark,
            max_context_tokens=int(self.config.get("max_context_tokens",32768)),
            max_new_tokens=int(self.config.get("output_tokens",512 if self.benchmark=="alfworld" else 1024)),
            cache_initial_capacity=int(self.config.get("cache_initial_capacity",1024)),
            representation_token_budget=int(self.config.get("representation_token_budget",8192)),
            representation_batch_size=int(self.config.get("representation_batch_size",8)),
            working_memory_gib=float(self.config.get("working_memory_gib",6.)),
            checkpoint_shard_mib=float(self.config.get("checkpoint_shard_mib",1024.)))
        policy.sampling_temperature = float(self.config["temperature"])
        policy.metadata.update({"method":self.method,"method_role":METHOD_ROLES[self.method],
            "max_context_tokens":policy.max_context_tokens,
            "allocated_logical_gpu_ids":list(self.gpu_ids),
            "allocated_physical_gpu_ids":self.config.get("allocated_physical_gpu_ids",self.config.get("physical_gpu_ids")),
            "rng_logical_gpu_ids":list(self.gpu_ids),
            "rng_physical_gpu_ids":self.config.get("rng_physical_gpu_ids",self.config.get("physical_gpu_ids")),
            "compute_logical_gpu_ids":list(self.compute_gpu_ids),
            "compute_physical_gpu_ids":self.config.get("compute_physical_gpu_ids",self.config.get("physical_gpu_ids")),
            "model_placement_explicit":self.config.get("model_placement_explicit",False),
            "output_tokens":policy.max_new_tokens,"protocol":PROTOCOL,
            "effective_training_identity":effective_training_identity(self.config),
            "recipe":({"lr":float(self.config.get("learning_rate",1e-6)),
                "weight_decay":float(self.config.get("weight_decay",.01)),
                "groups":int(self.config.get("groups",16)),"trajectories":int(self.config.get("trajectories",8)),
                "rounds":int(self.config.get("rounds",60)),"row_order":"atomic_mixed_group",
                "loss_normalization":self.config.get("loss_normalization","trajectory"),
                "chunk_tokens":int(self.config.get("head_chunk_tokens",64)),
                "gamma":1.,"intrinsic_reward":False}
                if self.method in OWN_METHODS else published_recipe(self.method,self.benchmark,self.config["model_path"]))})
        if self.method not in OWN_METHODS:
            policy.metadata["published_source"] = PUBLISHED_SOURCES[self.method]
            policy.metadata["execution_transfer"] = {
                "runtime":"shared_PyTorch_full_finetune_CPU_FP32_AdamW",
                "loss_weighting":"released_virtual_rank_microbatch_means_fixed_accumulation",
                "upstream_execution":"verl_FSDP_microbatch_token_means_and_vLLM_rollout",
                "scope":"released_credit_prompt_reward_recipe; numerical_execution_port"}
            policy.metadata["prompt_transfer"] = {"history_length":self.config["history"],"alfworld":"official_SimpleMemory_template",
                "alfworld_prompt_capacity":self.config["author_prompt_length"],
                "prompt_truncation":self.config["prompt_truncation"],
                "appworld":"pinned_native_bootstrap_plus_recent_observation_executed_code_pairs",
                "appworld_prompt_capacity":policy.max_context_tokens-policy.max_new_tokens}
        self.ledger.record("initialization","model.load",seconds=clock()-start,gpu_ids=self.compute_gpu_ids,
                           details=policy.metadata)
        return policy

    def _make_learning_state(self):
        optimizer = OffloadedAdamW(self.policy.model,
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
            betas=tuple(self.config.get("optimizer_betas",(.9,.999))),
            eps=float(self.config.get("optimizer_eps",1e-8)),
            bucket_mib=int(self.config.get("optimizer_bucket_mib",64)),
            pin_memory=bool(self.config.get("optimizer_pin_memory",True)))
        return optimizer,WeightReference(self.policy.model,fixed=True),None

    def _install_signals(self):
        def request(_signum,_frame):
            self.stop_requested = True
        for name in (signal.SIGINT,signal.SIGTERM):
            self._signals[name] = signal.getsignal(name)
            signal.signal(name,request)

    def _restore_signals(self):
        for name,value in self._signals.items():
            signal.signal(name,value)
        self._signals.clear()

    def _boundary(self,cursor=None,*,durable=False,reuse_only=False):
        if cursor is not None:
            self.state.update_cursor = cursor
        reusable = (self._saved_checkpoint_path == self.checkpoint_path.resolve()
                    and self._saved_model_version == self.optimizer.steps)
        if self.stop_requested or (durable and (not reuse_only or reusable)):
            self.save_checkpoint(self.checkpoint_path)
        if self.stop_requested:
            raise TrainingInterrupted()

    def train(self,task_rows,env_factory):
        if self._storage_finalized:
            raise ValueError("Completed-cell deployment state cannot implicitly restart training")
        self.policy.sampling_temperature = float(self.config["temperature"])
        if not self.state.task_order:
            rows = [dict(row) for row in task_rows]
            if any(str(row.get("split","train")) not in ("train","training") for row in rows):
                raise ValueError("Only official training rows can enter study updates")
            self.rng.shuffle(rows)
            self.state.task_order = rows
        tasks = self.state.task_order
        self._install_signals()
        try:
            rounds = int(self.config["rounds"])
            if rounds < 1:
                raise ValueError("Trainable methods require a positive configured round count")
            if self._saved_model_version is None:
                # Save the initial actor, empty Adam state, fixed reference and
                # task/RNG state before any environment work can be discarded.
                self._boundary(durable=True)
            while self.state.round<rounds:
                current_round = self.state.round
                with self.ledger.context(group_id=None,round_index=current_round):
                    if self.state.phase=="collect":
                        count = int(self.config["groups"])
                        if count < 1:
                            raise ValueError("A training round requires at least one task group")
                        while self.state.group_cursor<count:
                            g = self.state.group_cursor
                            task = tasks[(current_round*count+g)%len(tasks)]
                            initial_seed = self.seed*1000003+current_round*count+g
                            k = int(self.config["trajectories"])
                            if self.method in OWN_METHODS and k < 2:
                                raise ValueError("r3 peer credit requires at least two independent trajectories")
                            seeds = [self.seed*1000000007+current_round*1000003+g*101+slot for slot in range(k)]
                            group_id = f"{self.method}:{self.seed}:{current_round}:{g}"
                            with self.ledger.context(group_id=group_id,round_index=current_round):
                                if self.method in OWN_METHODS:
                                    from .rollout import collect_group
                                    with self.policy.phase("rollout.group"):
                                        group = collect_group(self.policy,env_factory,task_id=task["task_id"],benchmark=self.benchmark,
                                            initial_seed=initial_seed,slot_seeds=seeds,group_id=group_id,
                                            ledger=self.ledger)
                                else:
                                    with self.policy.phase("rollout.group"):
                                        group = collect_native_group(self.policy,env_factory,task_id=task["task_id"],initial_seed=initial_seed,
                                            slot_seeds=seeds,group_id=group_id,method=self.method,config=self.config)
                            self.state.groups.append(group)
                            if self.config["retain_training_artifacts"]:
                                import torch
                                directory = self.output_dir/"groups"
                                directory.mkdir(exist_ok=True)
                                torch.save(group,directory/f"round-{current_round}-group-{g}.pt")
                            del group
                            self.state.group_cursor += 1
                            self._boundary(durable=self.state.group_cursor%int(self.config.get("checkpoint_interval_groups",16))==0,
                                           reuse_only=True)
                        self.state.phase = "reference"
                    trajectories = [t for group in self.state.groups for t in group.trajectories]
                    mixed_groups = _mixed_groups(self.state.groups) if self.method in OWN_METHODS else None
                    if self.state.phase=="reference":
                        if self.method not in OWN_METHODS:
                            with self.policy.phase("baseline.advantages"):
                                if self.method == "gigpo":
                                    self.state.published_batch_plan = plan_published_batch(trajectories,self.method,config=self.config)
                                    self.state.advantages = baseline_advantages(self.policy,trajectories,self.method,config=self.config,
                                        credit_row_ids=self.state.published_batch_plan["credit_row_ids"])
                                else:
                                    self.state.advantages = baseline_advantages(self.policy,trajectories,self.method,config=self.config)
                                    self.state.published_batch_plan = plan_published_batch(trajectories,self.method,config=self.config)
                            with self.policy.phase("reference.single_round_activation"),self.reference.activate(self.policy.model,optimizer=self.optimizer):
                                self.policy._reference_active = True
                                try:
                                    self.state.reference_data = prepare_round(self.policy,trajectories,method=self.method,
                                        config=self.config,round_index=current_round)
                                finally:
                                    self.policy._reference_active = False
                            self.state.reference_data.release_features()
                            self.state.phase = "update"
                            self._boundary(durable=True,reuse_only=True)
                        else:
                            mixed_trajectories = [t for g in mixed_groups for t in g.trajectories]
                            if mixed_trajectories:
                                with self.policy.phase("reference.single_round_activation"),self.reference.activate(self.policy.model,optimizer=self.optimizer):
                                    self.policy._reference_active = True
                                    try:
                                        self.state.reference_data = prepare_round(self.policy,mixed_trajectories,method=self.method,
                                            config=self.config,round_index=current_round)
                                    finally:
                                        self.policy._reference_active = False
                            else:
                                self.state.reference_data = None
                                self.ledger.record("controller","reference.homogeneous_round_bypass",seconds=0.,
                                                   details={"groups":len(self.state.groups),"mixed_groups":0})
                            del mixed_trajectories
                            start = clock()
                            self.state.credit_diagnostics = {}
                            self.state.advantages = advantages(trajectories,self.state.reference_data,method=self.method,
                                config=self.config,diagnostics=self.state.credit_diagnostics)
                            self.ledger.record("controller","credit.peer_values",seconds=clock()-start,
                                details={"groups":len(self.state.credit_diagnostics),"mixed_groups":len(mixed_groups),
                                         "homogeneous_groups":len(self.state.groups)-len(mixed_groups),
                                         "recursive_bootstrapping":False})
                            if self.state.reference_data is not None:
                                self.state.reference_data.release_features()
                            self.state.phase = "update"
                            self._boundary(durable=bool(mixed_groups),reuse_only=True)
                    if self.state.phase=="update":
                        if self.method in OWN_METHODS:
                            if not mixed_groups:
                                denom = float(self.config["trajectories"])
                                self.state.update_cursor = UpdateCursor(finished=True,denominator=denom,
                                    metrics={"loss":0.,"entropy":0.,"reference_kl":0.,"steps":0,"gradient_norm":None,
                                             "batch_gradient_norms":[],"batch_clip_scales":[],
                                             "evaluated_response_tokens":0,"ppo_clipped_tokens":0,"ppo_clip_fraction":0.,
                                             "first_batch_tokens":0,"first_batch_abs_logprob_diff_sum":0.,
                                             "first_batch_mean_abs_logprob_diff":None,
                                             "first_batch_max_abs_logprob_diff":None,
                                             "loss_normalization":self.config["loss_normalization"],
                                             "loss_weighting":"atomic_mixed_group_trajectory","minibatch_count":0,
                                             "mixed_groups":0,"group_trajectory_denominator":denom,
                                             "homogeneous_round_bypass":True})
                                self.ledger.record("controller","ppo.homogeneous_round_bypass",seconds=0.,
                                                   details={"groups":len(self.state.groups),"mixed_groups":0,"optimizer_steps_executed":0})
                            else:
                                mixed_trajectories = [t for g in mixed_groups for t in g.trajectories]
                                try:
                                    self.state.update_cursor = ppo_update(self.policy,mixed_trajectories,self.state.reference_data,
                                        self.state.advantages,self.optimizer,minibatch_rows=None,
                                        cursor=self.state.update_cursor,checkpoint_callback=self._boundary,
                                        constant_denominator=True,
                                        chunk_tokens=int(self.config.get("head_chunk_tokens",64)),
                                        loss_normalization=self.config["loss_normalization"],
                                        clip_ratio=float(self.config.get("ppo_clip_ratio",.2)),
                                        dual_clip=float(self.config.get("ppo_dual_clip",3.)),
                                        entropy_coef=float(self.config.get("entropy_coef",.001)),
                                        kl_coef=float(self.config.get("kl_coef",.01)),
                                        max_grad_norm=float(self.config.get("max_grad_norm",1.)),
                                        trajectory_groups=mixed_groups,round_index=current_round)
                                finally:
                                    del mixed_trajectories
                        else:
                            self.state.update_cursor = ppo_update(self.policy,trajectories,self.state.reference_data,
                                self.state.advantages,self.optimizer,minibatch_rows=int(self.config["ppo_minibatch_rows"]),
                                cursor=self.state.update_cursor,checkpoint_callback=self._boundary,
                                constant_denominator=False,
                                chunk_tokens=int(self.config.get("head_chunk_tokens",64)),
                                loss_normalization=self.config["loss_normalization"],
                                clip_ratio=float(self.config.get("ppo_clip_ratio",.2)),
                                dual_clip=float(self.config.get("ppo_dual_clip",3.)),
                                entropy_coef=float(self.config.get("entropy_coef",.001)),
                                kl_coef=float(self.config.get("kl_coef",.01)),
                                max_grad_norm=float(self.config.get("max_grad_norm",1.)),
                                published_plan=self.state.published_batch_plan,round_index=current_round)
                    self._write_round(trajectories)
                    del trajectories, mixed_groups
                    self.state.round += 1
                    self.state.phase,self.state.group_cursor = "collect",0
                    self.state.groups,self.state.reference_data,self.state.advantages,self.state.update_cursor = [],None,{},None
                    self.state.credit_diagnostics = {}
                    self.state.published_batch_plan = None
                    self.ledger.begin_group(None)
                self._boundary(durable=self.state.round%int(self.config["checkpoint_interval_rounds"])==0)
            self.state.completed,self.state.phase = True,"complete"
            self._boundary(durable=True)
            return self._result("complete")
        except TrainingInterrupted:
            return self._result("interrupted")
        finally:
            self._restore_signals()

    def _write_round(self,trajectories):
        if self.config["retain_training_artifacts"] and self.state.reference_data is not None:
            import torch
            torch.save({"reference":self.state.reference_data,"advantages":self.state.advantages,
                        "round":self.state.round,"method":self.method,"published_batch_plan":self.state.published_batch_plan},
                       self.output_dir/f"reference-round-{self.state.round}.pt")
        payload = {"method":self.method,"round":self.state.round,
            "successes":sum(t.official_success for t in trajectories),"trajectories":len(trajectories),
            "method_role":METHOD_ROLES[self.method],
            "protocol":PROTOCOL,"optimizer_steps":self.optimizer.steps,"costs":self.ledger.snapshot(),
            "metrics":self.state.update_cursor.metrics}
        if self.state.published_batch_plan is not None:
            payload["published_batch"] = self.state.published_batch_plan["diagnostics"]
        if self.method in OWN_METHODS:
            mixed = _mixed_groups(self.state.groups)
            all_zero = sum(not any(t.official_success for t in g.trajectories) for g in self.state.groups)
            all_one = sum(all(t.official_success for t in g.trajectories) for g in self.state.groups)
            total_groups = len(self.state.groups)
            steps = [row for t in trajectories for row in t.steps]
            total_steps = len(steps)
            syntax_valid = sum(bool(row.syntax_valid) for row in steps)
            admissible_valid = sum(bool(row.environment_info.get("valid",False)) for row in steps)
            unterminated = sum((not row.response_ids) or (row.response_ids[-1] not in TERMINATORS) for row in steps)
            scored = (0 if self.state.reference_data is None
                      else int(self.state.reference_data.diagnostics.get("scored_trajectories",sum(len(g.trajectories) for g in mixed))))
            payload["diagnostics"] = {
                "total_groups":total_groups,
                "mixed_groups":len(mixed),
                "homogeneous_groups":total_groups-len(mixed),
                "all_zero_groups":all_zero,
                "all_one_groups":all_one,
                "homogeneous_round_bypass":not mixed,
                "optimizer_steps_in_round":int(self.state.update_cursor.metrics.get("steps",0)),
                "reference_scored_trajectories":scored,
                "total_steps":total_steps,
                "syntax_valid_steps":syntax_valid,
                "syntax_valid_fraction":(syntax_valid/total_steps) if total_steps else 0.,
                "admissible_valid_steps":admissible_valid,
                "admissible_valid_fraction":(admissible_valid/total_steps) if total_steps else 0.,
                "unterminated_response_steps":unterminated,
                "positive_peer_gated_steps":sum(int(item.get("positive_peer_gated_steps") or 0) for item in self.state.credit_diagnostics.values()),
                "nonzero_advantages":sum(abs(v)>0 for v in self.state.advantages.values()),
                "positive_failed_steps":sum(self.state.advantages[row.step_id]>0 for t in trajectories if not t.official_success for row in t.steps),
                "intrinsic_reward_enabled":False,"credit_groups":self.state.credit_diagnostics,
                "credit_diagnostic_scope":"held_out_outcomes_include_Bernoulli_noise_not_true_Q",
                "claim_scope":"same_depth_history_support_and_nonrecursive_continuation_sharing",
                "positive_failed_steps_interpretation":"credit_assignment_count_not_verified_helpful_actions",
                "additional_diagnostic_rollouts":0}
        with (self.output_dir/"training.jsonl").open("ab") as handle:
            handle.write((json.dumps(payload,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8"))
            self.state.training_log_offset = handle.tell()

    def evaluate(self,task_rows,env_factory,*,split):
        path = self.output_dir/f"evaluation-{split}.jsonl"
        completed = {}
        committed_bytes = 0
        if path.exists():
            with path.open("rb") as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    for key,value in (("protocol",PROTOCOL),("method",self.method),("benchmark",self.benchmark),("seed",self.seed),("split",split)):
                        if row[key]!=value:
                            raise ValueError("Persisted official evaluation belongs to another study identity")
                    if row.get("model_version",self.optimizer.steps) != self.optimizer.steps:
                        raise ValueError("Official evaluation journal belongs to a different actor update")
                    completed[row["task_id"]] = row
                    committed_bytes = handle.tell()
            if path.stat().st_size!=committed_bytes:
                with path.open("r+b") as handle:
                    handle.truncate(committed_bytes)
                self.ledger.record("artifact","evaluation.uncommitted_row",details={"split":split,"committed_bytes":committed_bytes})
        if not set(self.state.evaluation_completed.get(split,())).issubset(completed):
            raise ValueError("Checkpoint names completed official results absent from its evaluation journal")
        self.state.evaluation_completed[split] = list(completed)
        records = []
        self.policy.model.eval()
        self.policy.sampling_temperature = float(self.config["eval_temperature"])
        evaluation_config = dict(self.config,temperature=self.config["eval_temperature"],
                                 top_p=self.config["eval_top_p"])
        status = "complete"
        self._install_signals()
        try:
            for index,task in enumerate(task_rows):
                if task["task_id"] in completed:
                    records.append(completed[task["task_id"]])
                    continue
                group_id = f"evaluation:{split}:{index}"
                group = trajectory = costs = None
                with self.ledger.context(group_id=group_id,round_index=None):
                    if self.method in OWN_METHODS:
                        from .rollout import collect_group
                        with self.policy.phase("evaluation.rollout"):
                            group = collect_group(self.policy,env_factory,task_id=task["task_id"],benchmark=self.benchmark,
                                initial_seed=self.seed,slot_seeds=[self.seed*1000003+index],group_id=group_id,
                                ledger=self.ledger)
                        trajectory = group.trajectories[0]
                    else:
                        self.ledger.begin_group(group_id)
                        with self.policy.phase("evaluation.rollout"):
                            trajectory = collect_native_trajectory(self.policy,env_factory,task_id=task["task_id"],initial_seed=self.seed,
                                sampling_seed=self.seed*1000003+index,group_id=group_id,slot=0,method=self.method,
                                config=evaluation_config)
                    costs = group.costs if self.method in OWN_METHODS else trajectory.costs
                    row = {"task_id":task["task_id"],"trajectory_id":trajectory.trajectory_id,
                        "official_evaluation":dict(trajectory.official_metrics),"steps":len(trajectory.steps),
                        "input_tokens":sum(c.input_tokens or 0 for c in costs),"output_tokens":sum(c.output_tokens or 0 for c in costs),
                        "protocol":PROTOCOL,"method":self.method,"benchmark":self.benchmark,"seed":self.seed,"split":split,
                        "model_version":self.optimizer.steps}
                del group, trajectory, costs
                with path.open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(row,ensure_ascii=False)+"\n")
                records.append(row)
                completed[task["task_id"]] = row
                self.state.evaluation_completed[split] = list(completed)
                if self.stop_requested:
                    self.save_checkpoint(self.checkpoint_path)
                    status = "interrupted"
                    break
        finally:
            self.ledger.begin_group(None)
            self.policy.sampling_temperature = float(self.config["temperature"])
            self._restore_signals()
        return {"status":status,"protocol":PROTOCOL,"records":records,"method":self.method,"benchmark":self.benchmark,"seed":self.seed,"split":split}

    def _result(self,status):
        return {"status":status,"protocol":PROTOCOL,"method":self.method,"benchmark":self.benchmark,"seed":self.seed,
            "method_role":METHOD_ROLES[self.method],
            "rounds_completed":self.state.round,"phase":self.state.phase,
            "optimizer_steps":self.optimizer.steps if self.optimizer else 0,
            "costs_scope":"initialization_and_training_before_official_evaluation",
            "costs":self.ledger.snapshot()}

    def save_checkpoint(self,path):
        if self._storage_finalized:
            raise ValueError("Completed final models do not contain incomplete-training resume state")
        # A published trainer cursor can only name durable diagnostic/result
        # rows. Physical cost events are synced inside checkpoint.save after
        # it records the actual parameter-serialization work.
        for journal in (self.output_dir/"training.jsonl", *(
                self.output_dir/f"evaluation-{split}.jsonl"
                for split in self.config["evaluation_splits"][self.benchmark])):
            if journal.exists():
                checkpoint._sync_file(journal)
        version = self.optimizer.steps if self.optimizer else 0
        save_start = clock()
        checkpoint.save(path,policy=self.policy,optimizer=self.optimizer,reference=self.reference,
            teacher=self.teacher,trainer_state={"config":self.config,"state":self.state,"sampler_rng":self.rng.getstate()},
            ledger_state=self.ledger.state_dict(),rng_state=checkpoint.capture_rng(self.gpu_ids),
            model_version=version,storage=self.storage)
        pub_details = {"model_version":version,"backend":"gcs" if self.storage is not None else "local"}
        if self.storage is not None and self.storage.last_transfer_stats is not None:
            pub_details["transfer"] = dict(self.storage.last_transfer_stats)
        self.ledger.record("checkpoint","checkpoint.publish",seconds=clock()-save_start,details=pub_details)
        self._saved_checkpoint_path,self._saved_model_version = Path(path).resolve(),version
        return Path(path)

    def prepare_completed_checkpoint(self,training,evaluation):
        """Retain deployment/reproduction artifacts; leave full resume state intact."""
        expected = set(self.config["evaluation_splits"][self.benchmark])
        if (training.get("status")!="complete" or set(evaluation)!=expected or
            any(not item["dataset_complete"] for item in evaluation.values())):
            raise ValueError("Incomplete official work retains its full exact-resume checkpoint")
        if not self.state.completed:
            raise ValueError("Training has pending work; its full state remains retained")
        full_root = self.output_dir/"checkpoint"
        self.save_checkpoint(full_root)
        provenance = {"config":self.config,"model":self.policy.metadata,
            "training":training,"evaluation":evaluation,"official_tasks":"official_tasks.json",
            "cost_journal":str(self.ledger.path),"costs":self.ledger.snapshot(),
            "retain_training_artifacts":bool(self.config["retain_training_artifacts"]),
            "results_retained":"training claim diagnostics, physical costs and official evaluation results",
            "raw_artifact_policy":("retain explicitly requested group/reference and single-evaluation traces"
                if self.config["retain_training_artifacts"] else
                "reclaim this cell's groups, trajectories and reference-round files after durable completion")}
        start = clock()
        artifact = checkpoint.prepare_completed(full_root,self.output_dir/"final_checkpoint",provenance=provenance,storage=self.storage)
        self.ledger.record("checkpoint","checkpoint.completed_artifacts",seconds=clock()-start,
                           details=artifact)
        return artifact

    def reclaim_completed_checkpoint(self,summary_path):
        """Called after preliminary complete cell_summary.json is durable."""
        start = clock()
        reclaimed = checkpoint.reclaim_completed_resume_state(
            summary_path,storage=self.storage,keep_journal_state=(self.storage is not None))
        details = {"reclaimed":reclaimed,"cell":str(self.output_dir)}
        if self.storage is not None and self.storage.last_transfer_stats is not None:
            details["transfer"] = dict(self.storage.last_transfer_stats)
        self.ledger.record("checkpoint","checkpoint.reclaim_completed_resume_state",seconds=clock()-start,
                           details=details)
        self.optimizer,self.reference,self.teacher = None,None,None
        self._storage_finalized = True
        self.checkpoint_path = self.output_dir/"final_checkpoint"
        return reclaimed

    def publish_final_logs(self):
        """Complete and measure final log publication before freezing cell_summary costs."""
        pub_start = clock()
        if self.storage is not None:
            pub_receipt = self.storage.publish_logs()
        else:
            if self.ledger.path is not None and self.ledger.path.is_file():
                checkpoint._sync_file(self.ledger.path)
            pub_receipt = {"backend": "local"}
        self.ledger.record("checkpoint","checkpoint.final_publish_logs",seconds=clock()-pub_start,
                           details=pub_receipt)
        return pub_receipt

    def export_final_terminal_state(self,summary_path):
        """Explicitly excluded terminal mirror carrying finalized summary and costs.jsonl."""
        checkpoint._sync_file(summary_path)
        if self.ledger.path is not None and self.ledger.path.is_file():
            checkpoint._sync_file(self.ledger.path)
        if self.storage is None:
            return None
        return self.storage.publish_final_artifacts(summary_path)

    def load_checkpoint(self,path):
        self.optimizer,self.reference,self.teacher = None,None,None
        if self.policy is not None:
            del self.policy.model
        self.policy = None
        import torch
        for gpu in self.gpu_ids:
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        self.checkpoint_path = Path(path)
        if self.storage is not None:
            from .checkpoint_gcs import GCSCheckpoint
            self.storage = GCSCheckpoint(self.config,self.checkpoint_path)
        self._restore_checkpoint(checkpoint.load(path,storage=self.storage))
        return self._result("loaded")

    def _restore_training_log(self):
        """Roll back round statistics while retaining all actual physical costs."""
        path = self.output_dir/"training.jsonl"
        offset = vars(self.state).get("training_log_offset")
        if offset is None:
            # Earlier v3 snapshots did not store a byte cursor. Their round
            # counter advances after _write_round, so rows below it committed.
            offset = 0
            if path.exists():
                with path.open("rb") as handle:
                    for line in handle:
                        if not line.endswith(b"\n") or json.loads(line)["round"] >= self.state.round:
                            break
                        offset = handle.tell()
        self.state.training_log_offset = int(offset)
        size = path.stat().st_size if path.exists() else 0
        if size < offset:
            raise ValueError("Training diagnostic journal is shorter than its checkpoint cursor")
        if size > offset:
            with path.open("r+b") as handle:
                handle.truncate(offset)
            checkpoint._sync_file(path)
            self.ledger.record("artifact","training.rollback",details={
                "committed_rounds":self.state.round,"committed_bytes":offset,
                "discarded_training_bytes":size-offset,"physical_costs_retained":True})

    def _require_evaluation_after_completed_training(self):
        if self.state.completed:
            return
        for split in self.config["evaluation_splits"][self.benchmark]:
            for path in (self.output_dir/f"evaluation-{split}.jsonl",
                         self.output_dir/split/"evaluation_summary.json",
                         self.output_dir/split/"evaluation_records.json"):
                if path.exists() and path.stat().st_size:
                    raise ValueError(f"Incomplete training snapshot at round {self.state.round} "
                        f"cannot reuse official evaluation output {path}; resume this cell's "
                        "completed-training snapshot before continuing its evaluation")

    def _restore_checkpoint(self,saved):
        config = saved["trainer"]["config"]
        require_training_identity(config,self.config)
        self.state = saved["trainer"]["state"]
        self._require_evaluation_after_completed_training()
        self.policy = self._make_policy(saved["model_path"])
        if saved["metadata"]["has_optimizer"]:
            self.optimizer = OffloadedAdamW(self.policy.model,
                bucket_mib=float(self.config.get("optimizer_bucket_mib",64)),
                pin_memory=bool(self.config.get("optimizer_pin_memory",True)))
            self.optimizer.load_state_dict(saved["optimizer"])
            self.optimizer.bucket_bytes = int(float(self.config.get("optimizer_bucket_mib",64))*1024*1024)
            self.optimizer.pin_memory = bool(self.config.get("optimizer_pin_memory",True))
            self.optimizer.restore_gradients(saved["gradients"])
        self.reference = WeightReference(state=saved["reference"],fixed=True) if saved["metadata"]["has_reference"] else None
        self.teacher = WeightReference(state=saved["teacher"]) if saved["metadata"]["has_teacher"] else None
        self.rng.setstate(saved["trainer"]["sampler_rng"])
        self.ledger.load_state_dict(saved["ledger"])
        self._restore_training_log()
        self._saved_checkpoint_path = self.checkpoint_path.resolve()
        self._saved_model_version = int(saved["metadata"]["model_version"])
        checkpoint.restore_rng(saved["rng"],self.gpu_ids)

    def close(self):
        self._restore_signals()
        self.optimizer,self.reference,self.teacher = None,None,None
        if hasattr(self.policy,"model"):
            del self.policy.model
        import torch
        for device in self.gpu_ids:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
