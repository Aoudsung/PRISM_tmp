"""CPU-master AdamW and packed-segment token PPO with resumable update cursors."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class OffloadedAdamW:
    def __init__(self, model, *, lr=1e-6, weight_decay=.01, betas=(.9,.999), eps=1e-8):
        self.params = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
        self.lr, self.weight_decay, self.betas, self.eps = float(lr),float(weight_decay),tuple(betas),float(eps)
        self.state, self.steps = {},0

    def zero_grad(self):
        for _,p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, max_norm=1.):
        squared = sum(float(torch.linalg.vector_norm(p.grad.detach(),dtype=torch.float32))**2
                      for _,p in self.params if p.grad is not None)
        norm = math.sqrt(squared)
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        scale = min(1.,max_norm/(norm+1e-6)) if max_norm is not None else 1.
        b1,b2 = self.betas
        self.steps += 1
        for name,p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach().to(device="cpu",dtype=torch.float32).mul_(scale)
            if name not in self.state:
                self.state[name] = {"master":p.detach().to(device="cpu",dtype=torch.float32).clone(),
                                    "exp_avg":torch.zeros_like(grad),"exp_avg_sq":torch.zeros_like(grad),"step":0}
            s = self.state[name]
            s["step"] += 1
            s["exp_avg"].mul_(b1).add_(grad,alpha=1-b1)
            s["exp_avg_sq"].mul_(b2).addcmul_(grad,grad,value=1-b2)
            denominator = s["exp_avg_sq"].sqrt().div_(math.sqrt(1-b2**s["step"])).add_(self.eps)
            s["master"].mul_(1-self.lr*self.weight_decay).addcdiv_(s["exp_avg"],denominator,
                                                               value=-self.lr/(1-b1**s["step"]))
            p.copy_(s["master"].to(p))
        return norm

    def state_dict(self):
        return {"format":"fgpo-adamw-v1","parameter_names":[n for n,_ in self.params],
                "initialized_parameter_names":list(self.state),"state":self.state,"steps":self.steps,
                "lr":self.lr,"weight_decay":self.weight_decay,"betas":self.betas,"eps":self.eps}

    def load_state_dict(self, state):
        if state["format"] != "fgpo-adamw-v1" or state["parameter_names"] != [n for n,_ in self.params]:
            raise ValueError("Optimizer state belongs to a different exact parameter space")
        if set(state["initialized_parameter_names"]) != set(state["state"]):
            raise ValueError("Checkpoint is missing saved Adam master weights or moments")
        self.state = state["state"]
        self.steps,self.lr,self.weight_decay = state["steps"],state["lr"],state["weight_decay"]
        self.betas,self.eps = tuple(state["betas"]),state["eps"]

    def gradient_state(self):
        return {name:p.grad.detach().cpu().clone() for name,p in self.params if p.grad is not None}

    def restore_gradients(self, values):
        self.zero_grad()
        params = dict(self.params)
        for name,value in values.items():
            params[name].grad = value.to(params[name])


class WeightReference:
    def __init__(self, model=None, *, trainable_only=False, state=None):
        self.weights = (state if state is not None else {name:p.detach().cpu().clone()
                        for name,p in model.named_parameters() if not trainable_only or p.requires_grad})

    @contextmanager
    def activate(self, model):
        current,previous = {},model.training
        try:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in self.weights:
                        current[name] = p.detach().cpu().clone()
                        p.copy_(self.weights[name].to(p))
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name in current:
                        p.copy_(current[name].to(p))
            model.train(previous)

    @torch.no_grad()
    def update_ema(self, model, decay=.99):
        for name,p in model.named_parameters():
            if name in self.weights:
                self.weights[name].mul_(decay).add_(p.detach().cpu(),alpha=1-decay)


@dataclass
class UpdateCursor:
    minibatch: int = 0
    segment: int = 0
    optimizer_ready: bool = False
    finished: bool = False
    denominator: float = 0.
    metrics: dict[str,Any] = field(default_factory=lambda:{"loss":0.,"entropy":0.,"reference_kl":0.,"steps":0})


def ppo_update(policy, trajectories, reference_data, advantages, optimizer, *,
               minibatch_rows=256, chunk_tokens=32, cursor=None, checkpoint_callback=None,
               constant_denominator=True):
    cursor = cursor or UpdateCursor()
    rows = [step for trajectory in trajectories for step in trajectory.steps]
    if not rows:
        raise ValueError("PPO requires real sampled response rows")
    minibatches = [rows[i:i+minibatch_rows] for i in range(0,len(rows),minibatch_rows)]
    if not cursor.denominator:
        cursor.denominator = sum(len(row.response_ids) for row in rows)/len(minibatches)
    segments = {s.segment_id:s for t in trajectories for s in t.segments}
    policy.model.train()
    try:
        while cursor.minibatch < len(minibatches):
            batch = minibatches[cursor.minibatch]
            grouped = {}
            for row in batch:
                grouped.setdefault(row.segment_id,[]).append(row)
            items = list(grouped.items())
            denominator = cursor.denominator if constant_denominator else sum(len(row.response_ids) for row in batch)
            if cursor.segment == 0 and not cursor.optimizer_ready:
                optimizer.zero_grad()
            while cursor.segment < len(items):
                segment_id,selected = items[cursor.segment]
                segment = segments[segment_id]
                stop = segment.behavior_closed_length or len(segment.token_ids)
                hidden,_ = policy.packed_segment_hidden(segment.token_ids[:stop],hidden_layer=None,role="ppo.current_segment")
                total_loss = None
                for row in selected:
                    span = row.response_span
                    states = hidden[0,span.start-1:span.start-1+span.length]
                    old = tuple(row.old_token_logprobs)
                    ref = reference_data.steps[row.step_id].token_logprobs
                    advantage = float(advantages[row.step_id])
                    for start in range(0,span.length,chunk_tokens):
                        end = min(start+chunk_tokens,span.length)
                        def objective(h,targets=row.response_ids[start:end],old_lp=old[start:end],
                                      ref_lp=ref[start:end],a=advantage,z=denominator):
                            logp = F.log_softmax(policy.head(h,role="ppo.head"),dim=-1)
                            ids = torch.tensor(targets,dtype=torch.long,device=logp.device)
                            lp = logp.gather(-1,ids[:,None]).squeeze(-1)
                            ratio = (lp-lp.new_tensor(old_lp)).exp()
                            dual = torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))
                            if a < 0:
                                dual = torch.minimum(dual,dual.new_tensor(-3*a))
                            entropy = -(logp.exp()*logp).sum(-1)
                            d = lp.new_tensor(ref_lp)-lp
                            kl = (d.exp()-d-1).clamp(-10,10)
                            return (dual-.001*entropy+.01*kl).sum()/z,entropy.sum().detach()/z,kl.sum().detach()/z
                        with policy.activation_saving():
                            loss,entropy,kl = checkpoint(objective,states[start:end],use_reentrant=False)
                        total_loss = loss if total_loss is None else total_loss+loss
                        cursor.metrics["loss"] += float(loss.detach())
                        cursor.metrics["entropy"] += float(entropy)
                        cursor.metrics["reference_kl"] += float(kl)
                with policy.phase("ppo.backward"):
                    total_loss.backward()
                cursor.segment += 1
                if checkpoint_callback is not None:
                    checkpoint_callback(cursor)
            cursor.optimizer_ready = True
            if checkpoint_callback is not None:
                checkpoint_callback(cursor)
            with policy.phase("ppo.adamw"):
                cursor.metrics["gradient_norm"] = optimizer.step(max_norm=1.)
            optimizer.zero_grad()
            cursor.metrics["steps"] += 1
            cursor.minibatch += 1
            cursor.segment,cursor.optimizer_ready = 0,False
            if checkpoint_callback is not None:
                checkpoint_callback(cursor)
        cursor.finished = True
        return cursor
    finally:
        policy.model.eval()
