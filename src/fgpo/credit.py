"""Filtration first-visit geometry, three reverse passes and nested leave-one-out RMS."""
from __future__ import annotations
import math
import torch


class FiltrationCredit:
    def __init__(self, trajectories, reference_data, *, no_dag=False):
        self.trajectories = tuple(trajectories)
        self.reference = reference_data
        self.no_dag = no_dag
        self.gamma,self.epsilon,self.floor = .99,.10,.05
        self.rows = [row for trajectory in trajectories for row in trajectory.steps]
        self.offsets,self.lengths = [],[]
        offset = 0
        for trajectory in trajectories:
            self.offsets.append(offset)
            self.lengths.append(len(trajectory.steps))
            offset += len(trajectory.steps)
        obs = torch.stack([reference_data.steps[row.step_id].z_obs for row in self.rows]).float()
        hist = torch.stack([reference_data.steps[row.step_id].z_hist for row in self.rows]).float()
        local = 1-obs@obs.T
        self.distance = local if no_dag else torch.maximum(local,1-hist@hist.T)
        self.match = self.distance <= self.epsilon
        self.rewards = [[reference_data.steps[row.step_id].reward for row in t.steps] for t in trajectories]
        self.first = {}
        for i,length in enumerate(self.lengths):
            for t in range(length):
                node = self.offsets[i]+t
                for j,other_length in enumerate(self.lengths):
                    candidates = torch.where(self.match[node,self.offsets[j]:self.offsets[j]+other_length])[0]
                    self.first[i,t,j] = int(candidates[0]) if len(candidates) else None
        self.novel = {}
        for i,length in enumerate(self.lengths):
            for t in range(length):
                self.novel[i,t] = (t+1 < length and not bool(self.match[self.offsets[i]+t+1,
                                                               self.offsets[i]:self.offsets[i]+t+1].any()))
        self.tables,self.residuals = {},{}

    def match_mean(self, i,t,peers,table):
        matches = [(j,self.first[i,t,j]) for j in peers if self.first[i,t,j] is not None]
        return (sum(table[j][s] for j,s in matches)/len(matches) if matches else 0.),len(matches)

    def values(self, peers):
        peers = tuple(sorted(peers))
        if peers in self.tables:
            return self.tables[peers]
        previous = {}
        for i in peers:
            value = [0.]*(self.lengths[i]+1)
            for t in reversed(range(self.lengths[i])):
                value[t] = self.rewards[i][t]+self.gamma*value[t+1]
            previous[i] = value
        for _ in range(3):
            current = {}
            for i in peers:
                value = [0.]*(self.lengths[i]+1)
                others = tuple(j for j in peers if j != i)
                for t in reversed(range(self.lengths[i])):
                    if self.novel[i,t] and not self.no_dag:
                        average,count = self.match_mean(i,t+1,others,previous)
                    else:
                        average,count = 0.,0
                    alpha = count/(count+1)
                    value[t] = self.rewards[i][t]+self.gamma*(alpha*average+(1-alpha)*value[t+1])
                current[i] = value
            previous = current
        self.tables[peers] = previous
        return previous

    def delta(self, target,peers):
        peers = tuple(sorted(peers))
        key = (target,peers)
        if key in self.residuals:
            return self.residuals[key]
        table = self.values(peers)
        length = self.lengths[target]
        g = [0.]*(length+1)
        result = [0.]*length
        for t in reversed(range(length)):
            if self.novel[target,t] and not self.no_dag:
                average,count = self.match_mean(target,t+1,peers,table)
            else:
                average,count = 0.,0
            alpha = count/(count+1)
            g[t] = self.rewards[target][t]+self.gamma*(alpha*average+(1-alpha)*g[t+1])
            current_mean,current_count = self.match_mean(target,t,peers,table)
            beta = current_count/(current_count+1)
            prior = sum(table[j][min(t,self.lengths[j])] for j in peers)/len(peers) if peers else 0.
            baseline = beta*current_mean+(1-beta)*prior
            result[t] = g[t]-baseline
        self.residuals[key] = result
        return result

    def advantages(self):
        result = {}
        all_ids = set(range(len(self.trajectories)))
        for k,trajectory in enumerate(self.trajectories):
            peers = tuple(sorted(all_ids-{k}))
            inner = []
            for j in peers:
                # Recompute three-pass tables on Q=P\{j}; merely deleting a
                # row from the P table would retain j's influence in its peers.
                q = tuple(i for i in peers if i != j)
                inner.extend(self.delta(j,q))
            rms = math.sqrt((sum(v*v for v in inner)/len(inner) if inner else 0.)+self.floor**2)
            for row,value in zip(trajectory.steps,self.delta(k,peers)):
                isolated = value if row.syntax_valid else min(value,0.)-.1
                result[row.step_id] = isolated/rms
        return result


def advantages(trajectories, reference_data, *, method="fgpo"):
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id,[]).append(trajectory)
    result = {}
    for group in groups.values():
        result.update(FiltrationCredit(group,reference_data,no_dag=method=="fgpo_no_dag_loo").advantages())
    return result
