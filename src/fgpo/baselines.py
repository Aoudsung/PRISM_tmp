"""Author-recipe external comparators, isolated from the F-GPO reward and credit.

Qwen transfer: retain task/bootstrap; native windows are complete-message
windows. Skill proposals partition whole source steps/operations at actual model
capacity and preserve global indices. No environment answers enter the actor.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from contextlib import redirect_stdout
import builtins
import io
import json
import math
import multiprocessing
from pathlib import Path
import re
import sys
import traceback
import uuid

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .records import StepRecord, SegmentRecord, TrajectoryRecord, GroupRecord, TokenSpan, TurnBlock
from .policy import clock
from .optimization import UpdateCursor


AUTHOR_RECIPES = {
    "native_qwen":{"trainable":False,"output_alf":512,"output_app":1024},
    "gigpo":{"lr":1e-6,"weight_decay":.01,"minibatch":256,"rounds":150,"history":2,"gamma":.95},
    "bipace_q":{"lr":5e-7,"weight_decay":.01,"minibatch":128,"rounds":150,"history":2,"gamma":.95},
    "self":{"lr":3e-4,"weight_decay":0.,"batch":16,"lora_rank":96,"lora_alpha":128,
            "dropout":.05,"ema":.99,"rounds":60,"round_window_identity":"study-Qwen-transfer"},
    "skill_disco":{"trainable":False,"induction_alf":200,"induction_app":90,"refinements":3,"operation_batch":32},
}


def _pure(raw):
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    return str(value.get("native_observation",raw)) if isinstance(value,dict) else raw


def _available(schema):
    return list(schema.get("available_actions",()))


def parse_action(text,benchmark):
    if benchmark == "alfworld":
        match = re.fullmatch(r"(?:<think>.*?</think>\s*)?<action>(.*?)</action>",text.strip(),re.DOTALL)
        command = match.group(1).strip() if match else text.strip()
        return {"kind":"native_text","text":command}
    fences = list(re.finditer(r"```",text))
    if not fences:
        return {"kind":"code","code":text.strip()}
    if len(fences)==2:
        opening = re.match(r"```([^\r\n`]*)\r?\n",text[fences[0].start():])
        if opening and opening.group(1).strip().lower() in ("","python"):
            return {"kind":"code","code":text[fences[0].start()+opening.end():fences[1].start()]}
    return {"kind":"code","code":text.strip(),"parser_invalid":True}


def _chat_ids(policy,messages):
    value = policy.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    return tuple(int(x) for x in value)


def native_prompt(policy,pin,history,current,*,method,config,memory=None,schema=None,goal=""):
    system,initial = pin
    if method in ("gigpo","bipace_q"):
        # Official SimpleMemory recipe; AppWorld uses its declared native
        # bootstrap/context adaptation rather than losing initial API documents.
        recent = list(memory or ())[-2:]
        current_observation = _pure(current["content"])
        if policy.benchmark=="alfworld":
            # ALFWorld prompts retain the Apache-2.0 verl-agent/BiPACE
            # SimpleMemory template (NTU Singapore and verl-agent, 2025).
            commands = "\n ".join(repr(a) for a in _available(schema) if a!="help")
            if recent:
                context = "\n".join(f"[Observation {len(memory)-len(recent)+i+1}: '{o}', Action {len(memory)-len(recent)+i+1}: '{a}']"
                                    for i,(o,a) in enumerate(recent))
                text = (f"You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {goal}\n"
                    f"Prior to this step, you have already taken {len(memory)} step(s). Below are the most recent {len(recent)} "
                    f"observations and the corresponding actions you took: {context}\n"
                    f"You are now at step {len(memory)+1} and your current observation is: {current_observation}\n")
            else:
                text = f"You are an expert agent operating in the ALFRED Embodied Environment.\nYour current observation is: {current_observation}\n"
            text += (f"Your admissible actions of the current situation are: [{commands}].\nNow it's your turn to take an action.\n"
                "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\n"
                "Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.")
            messages = [system,{"role":"user","content":text}]
        else:
            messages = [system,initial,{"role":"user","content":json.dumps({"task":goal,
                "recent_observation_action_pairs":recent,"current_observation":current_observation},ensure_ascii=False)}]
        maximum = int(config.get("author_prompt_length",2048)) if policy.benchmark=="alfworld" else policy.max_context_tokens-policy.max_new_tokens
        ids = _chat_ids(policy,messages)
        if len(ids)>maximum:
            raise ValueError("Author PPO prompt exceeds its declared complete-message context")
        return ids
    maximum = policy.max_context_tokens-policy.max_new_tokens
    retained = list(history)
    while True:
        messages = [system,initial]+[m for pair in retained for m in pair]
        ids = _chat_ids(policy,messages)
        if len(ids)<=maximum:
            return ids
        if not retained:
            raise ValueError("Pinned native task/bootstrap and current complete observation exceed Qwen capacity")
        retained.pop(0)


def sample_ids(policy,prompt,generator,limit,*,role):
    state = policy.prefill_root(prompt,role=role+".prefill")
    ids,logp = [],[]
    for i in range(limit):
        tokens,scores = policy.draw_tokens(state,[0],[generator])
        ids.extend(tokens);logp.extend(scores)
        if tokens[0] in policy.eos_token_ids or i+1==limit:
            break
        state = policy.append_cache(state,[tokens],role=role+".decode")
    return tuple(ids),tuple(logp)


def collect_native_trajectory(policy,env_factory,*,task_id,initial_seed,sampling_seed,group_id,
                              slot,method,config,library=None):
    trajectory_id = uuid.uuid4().hex
    start_cost = len(policy.ledger.records)
    env = env_factory(task_id,initial_seed,trajectory_id)
    try:
        started = clock()
        context = env.reset(task_id,initial_seed)
        policy.record("environment","native.reset",seconds=clock()-started,details={"task_id":task_id})
        schema = dict(context.get("action_schema",env.action_schema))
        goal,initial_raw = str(context["instruction"]),str(context["observation"])
        system = {"role":"system","content":str(schema["native_system_prompt"])}
        if library is not None:
            system["content"] += "\nProcedural skills:\n"+json.dumps([s.public() for s in library.skills],ensure_ascii=False)+ \
                '\nReturn one primitive native action or <skill>{"name":"name","arguments":{...}}</skill>.'
        pin = (system,{"role":"user","content":json.dumps({"task":goal,"observation":initial_raw},ensure_ascii=False)})
        pinned = _chat_ids(policy,list(pin))
        history,memory,steps,segments = [],[],[],[]
        observation,observation_ids = initial_raw,policy.encode(_pure(initial_raw))
        generator = policy.new_generator(sampling_seed)
        horizon = 50 if policy.benchmark=="alfworld" else 100
        physical = 0
        for index in range(horizon):
            current = {"role":"user","content":observation}
            prompt = native_prompt(policy,pin,history,current,method=method,config=config,memory=memory,
                                   schema=env.action_schema,goal=goal)
            if index==0:
                pinned = prompt
            cost_start = len(policy.ledger.records)
            response,old = sample_ids(policy,prompt,generator,policy.max_new_tokens,role=method+".actor")
            text = policy.tokenizer.decode(response,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            request = re.fullmatch(r"<skill>([\s\S]+?)</skill>",text.strip()) if library is not None else None
            if request:
                action = {"kind":"skill",**json.loads(request.group(1))}
                execution = library.execute(env,action,_pure(observation),_available(env.action_schema),horizon-physical)
                physical += execution["primitive_count"]
                returned = json.dumps({"result":execution["result"],"error":execution["error"]},ensure_ascii=False)
                native_done = bool(execution["done"])
                actual_reward = execution.get("environment_reward",0.)
                valid = not bool(execution["error"])
            else:
                action = parse_action(text,policy.benchmark)
                valid = (action["text"] in _available(env.action_schema)) if policy.benchmark=="alfworld" else not action.get("parser_invalid",False)
                started = clock()
                result = env.step_native(action)
                policy.record("environment","native.step",seconds=clock()-started,env_steps=1,
                              details={"task_id":task_id,"action":action,"metadata":dict(result.metadata)})
                physical += 1
                returned,native_done = result.observation,result.done
                actual_reward = float(result.reward)
            step_id,segment_id = uuid.uuid4().hex,uuid.uuid4().hex
            pure_ids = policy.encode(_pure(returned))
            closure = (198,) if response and response[-1]==151645 else (151645,198)
            user_header = (151644,)+policy.encode("user\n")
            # External actor prefixes are independently templated author prompts.
            # Each sampled row owns one exact response-only segment.
            behavior_tokens = prompt+response
            tokens = behavior_tokens
            rsp = TokenSpan(segment_id,len(prompt),len(response),step_id)
            obs = TokenSpan(segment_id,len(tokens)+len(closure)+len(user_header),len(pure_ids),step_id)
            user = user_header+pure_ids+(151645,198,151644)+policy.encode("assistant\n")
            turn = TurnBlock(step_id,response,closure,user,response+closure+user,
                             len(response)+len(closure)+len(user_header),pure_ids)
            tokens = behavior_tokens+closure+user
            segment = SegmentRecord(segment_id,tokens,(rsp,),(obs,),len(behavior_tokens),len(behavior_tokens))
            row = StepRecord(step_id,trajectory_id,index,segment_id,len(prompt),rsp,obs,response,old,pure_ids,
                pure_ids or policy.encode("[Empty stdout]"),_pure(returned),returned,action,actual_reward,
                bool(native_done or physical>=horizon or index+1==horizon),valid,
                turn,segment_id,len(prompt),rsp,obs,tuple(policy.ledger.records[cost_start:]))
            steps.append(row);segments.append(segment)
            history.append(({"role":"assistant","content":text},{"role":"user","content":returned}))
            memory.append((_pure(observation),action.get("text",action.get("code",text))))
            observation = returned
            if native_done or physical>=horizon:
                break
        started = clock()
        metrics = dict(env.finalize_evaluation())
        policy.record("environment","native.official_scoring",seconds=clock()-started,details={"task_id":task_id})
        success = bool(metrics["success"])
        trajectory = TrajectoryRecord(trajectory_id,group_id,slot,task_id,policy.benchmark,initial_seed,sampling_seed,goal,
            policy.encode(goal),policy.encode(_pure(initial_raw)),pinned,tuple(steps),tuple(segments),tuple(segments),
            success,metrics,tuple(policy.ledger.records[start_cost:]),2 if method in ("gigpo","bipace_q") else None)
        if config.get("artifact_dir"):
            directory = Path(config["artifact_dir"])
            directory.mkdir(parents=True,exist_ok=True)
            torch.save(trajectory,directory/(trajectory_id+".pt"))
        return trajectory
    finally:
        env.close()


def collect_native_group(policy,env_factory,*,task_id,initial_seed,slot_seeds,group_id,method,config,library=None):
    policy.ledger.begin_group(group_id)
    start = len(policy.ledger.records)
    trajectories = tuple(collect_native_trajectory(policy,env_factory,task_id=task_id,initial_seed=initial_seed,
        sampling_seed=seed,group_id=group_id,slot=slot,method=method,config=config,library=library)
        for slot,seed in enumerate(slot_seeds))
    return GroupRecord(group_id,task_id,policy.benchmark,initial_seed,trajectories,tuple(policy.ledger.records[start:]))


def baseline_advantages(policy,trajectories,method):
    groups,episode,discounted,anchors,keys = {},{},{},{},{}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_id,[]).extend(trajectory.steps)
        running = 0.
        previous = trajectory.initial_observation_ids
        for row in trajectory.steps:
            anchors[row.step_id] = tuple(previous)
            previous = row.observation_ids
            raw = policy.tokenizer.decode(row.response_ids,skip_special_tokens=True)
            match = re.search(r"<action>([\s\S]+?)</action>",raw,re.IGNORECASE)
            keys[row.step_id] = ("CODE::"+row.action["code"] if policy.benchmark=="appworld" else
                "ACT::"+match.group(1).strip() if match else "FIRST8::"+str(row.response_ids[:8]))
        for row in reversed(trajectory.steps):
            reward = 10.*float(trajectory.official_success) if row.index==len(trajectory.steps)-1 else 0.
            running = reward+.95*running
            episode[row.step_id] = 10.*float(trajectory.official_success)-.1*(not row.syntax_valid)
            discounted[row.step_id] = running-.1*(not row.syntax_valid)
    embeddings = {}
    if method=="bipace_q":
        rows = [row for t in trajectories for row in t.steps]
        for start in range(0,len(rows),8):
            selected = rows[start:start+8]
            with torch.no_grad():
                vectors = policy.observation_embeddings([anchors[r.step_id] or policy.encode("[Empty stdout]") for r in selected],
                                                       role="bipace.current_actor_anchor")
            embeddings.update({r.step_id:z for r,z in zip(selected,vectors)})
    output = {}
    for rows in groups.values():
        values = torch.tensor([episode[r.step_id] for r in rows],dtype=torch.float32)
        normalized = (values-values.mean())/(values.std(unbiased=True)+1e-6) if len(rows)>1 else torch.zeros_like(values)
        output.update({r.step_id:float(a) for r,a in zip(rows,normalized)})
        clusters,centroids = [],[]
        if method=="gigpo":
            by_anchor = {}
            for row in rows:
                by_anchor.setdefault(anchors[row.step_id],[]).append(row)
            clusters = list(by_anchor.values())
        else:
            for row in rows:
                z = embeddings[row.step_id]
                nearest = int((torch.stack(centroids)@z).argmax()) if centroids else -1
                if nearest>=0 and 1-float(centroids[nearest]@z)<=.10:
                    clusters[nearest].append(row)
                    m = len(clusters[nearest])
                    centroids[nearest] = F.normalize(centroids[nearest]+(z-centroids[nearest])/m,dim=0)
                else:
                    clusters.append([row]);centroids.append(z.clone())
        for cluster in clusters:
            returns = torch.tensor([discounted[r.step_id] for r in cluster],dtype=torch.float32)
            if method=="gigpo":
                a = (returns-returns.mean())/(returns.std(unbiased=True)+1e-6) if len(cluster)>1 else torch.zeros_like(returns)
                for row,value in zip(cluster,a):
                    output[row.step_id] += float(value)
            elif len(cluster)>1:
                action_groups = {}
                for row in cluster:
                    action_groups.setdefault(keys[row.step_id],[]).append(row)
                if len(action_groups)>1:
                    for members in action_groups.values():
                        q = torch.tensor([discounted[r.step_id] for r in members],dtype=torch.float32).mean()
                        value = float((q-returns.mean())/(returns.std(unbiased=False)+1e-6))
                        for row in members:
                            output[row.step_id] += value
    return output


def _response_hidden(policy,prompt,target,role):
    if len(prompt)+len(target)>policy.max_context_tokens:
        raise ValueError("Complete author SELF target exceeds its declared native context")
    hidden,_,_ = policy.body(tuple(prompt)+tuple(target[:-1]),role=role)
    return hidden[0,-len(target):]


def _feedback_prompt(policy,trajectory,row,feedback):
    segment = next(s for s in trajectory.segments if s.segment_id==row.segment_id)
    prompt = segment.token_ids[:row.prompt_length]
    assistant = (151644,)+policy.encode("assistant\n")
    frame = (151644,)+policy.encode("user\n")+policy.encode(feedback)+(151645,198)+assistant
    return _bounded_conditioning(policy,trajectory,prompt[:-len(assistant)],frame)


def _bounded_conditioning(policy,trajectory,history,tail):
    assistant = (151644,)+policy.encode("assistant\n")
    pinned = trajectory.pinned_ids[:-len(assistant)]
    remaining = tuple(history[len(pinned):])
    boundaries = [i for i,token in enumerate(remaining) if token==151644]+[len(remaining)]
    frames = [remaining[a:b] for a,b in zip(boundaries,boundaries[1:])]
    limit = policy.max_context_tokens-policy.max_new_tokens
    while len(pinned)+sum(len(x) for x in frames)+len(tail)>limit:
        if not frames:
            raise ValueError("Pinned SELF conditioning and one complete privileged feedback exceed Qwen capacity")
        frames.pop(0)
    result = pinned+tuple(token for frame in frames for token in frame)+tuple(tail)
    policy.record("conditioning","self.teacher.window",details={"trajectory_id":trajectory.trajectory_id,
        "source_history_tokens":len(history),"retained_frames":len(frames),
        "conditioning_ids":result,"output_reserve":policy.max_new_tokens})
    return result


def _feedback_nll_backward(policy,trajectory,row,history,tail,feedback,*,chunk_tokens=32):
    if not feedback:
        return 0.
    assistant = (151644,)+policy.encode("assistant\n")
    pinned = trajectory.pinned_ids[:-len(assistant)]
    remaining = tuple(history[len(pinned):])
    boundaries = [i for i,token in enumerate(remaining) if token==151644]+[len(remaining)]
    frames = [(a,b) for a,b in zip(boundaries,boundaries[1:])]
    capacity = policy.max_context_tokens-chunk_tokens
    if len(pinned)+len(tail)>capacity:
        raise ValueError("Irreducible SELF bootstrap and actual action conditioning exceed Qwen capacity")
    def context(start):
        while frames and len(pinned)+sum(b-a for a,b in frames)+len(tail)+start>capacity:
            frames.pop(0)
        available = capacity-len(pinned)-sum(b-a for a,b in frames)-len(tail)
        suffix_start = max(0,start-available)
        prefix = pinned+tuple(token for a,b in frames for token in remaining[a:b])+tuple(tail)+tuple(feedback[suffix_start:start])
        policy.record("conditioning","self.feedback.window",details={"step_id":row.step_id,
            "retained_conditioning_frame_ranges":tuple(frames),"target_prefix_range":(suffix_start,start),
            "fixed_position_reserve":chunk_tokens,"full_target_length":len(feedback)})
        return prefix
    prefix = context(0)
    value = 0.
    if len(prefix)+len(feedback)<=policy.max_context_tokens:
        hidden = _response_hidden(policy,prefix,feedback,"self.feedback_forward")
        loss = None
        for start in range(0,len(feedback),chunk_tokens):
            stop = min(start+chunk_tokens,len(feedback))
            def objective(h,ids=feedback[start:stop],den=len(feedback)*16):
                logits = policy.head(h,role="self.feedback_head")
                return F.cross_entropy(logits,torch.tensor(ids,device=logits.device),reduction="sum")/den
            with policy.activation_saving():
                nll = checkpoint(objective,hidden[start:stop],use_reentrant=False)
            loss = nll if loss is None else loss+nll
            value += float(nll.detach())
        with policy.phase("self.feedback_backward"):
            (.2*loss).backward()
        return value
    for start in range(0,len(feedback),chunk_tokens):
        stop = min(start+chunk_tokens,len(feedback))
        prefix = context(start)
        consumed = tuple(feedback[start:stop])
        hidden = _response_hidden(policy,prefix,consumed,"self.feedback_continuation")
        def objective(h,ids=consumed,den=len(feedback)*16):
            logits = policy.head(h,role="self.feedback_head")
            return F.cross_entropy(logits,torch.tensor(ids,device=logits.device),reduction="sum")/den
        with policy.activation_saving():
            nll = checkpoint(objective,hidden,use_reentrant=False)
        with policy.phase("self.feedback_backward"):
            (.2*nll).backward()
        value += float(nll.detach())
    return value


@dataclass
class SelfCursor:
    update: int = 0
    row: int = 0
    minibatch: list[tuple[int,int]] = field(default_factory=list)
    metrics: dict = field(default_factory=lambda:{"steps":0,"kl":0.,"feedback_nll":0.})
    finished: bool = False


def self_update(policy,trajectories,optimizer,teacher,rng,*,cursor=None,on_boundary=None,chunk_tokens=32):
    cursor = cursor or SelfCursor()
    turns = sum(len(t.steps) for t in trajectories)
    updates = math.ceil(turns/16)
    if any(p.requires_grad for p in policy.model.get_output_embeddings().parameters()):
        raise ValueError("Author all-linear SELF recipe keeps its vocabulary projection frozen")
    while cursor.update<updates:
        if not cursor.minibatch:
            cursor.minibatch = []
            for _ in range(16):
                t = rng.randrange(len(trajectories))
                cursor.minibatch.append((t,rng.randrange(len(trajectories[t].steps))))
            optimizer.zero_grad()
        while cursor.row<16:
            t,i = cursor.minibatch[cursor.row]
            trajectory,row = trajectories[t],trajectories[t].steps[i]
            segment = next(s for s in trajectory.segments if s.segment_id==row.segment_id)
            prompt,target = segment.token_ids[:row.prompt_length],row.response_ids
            teacher_prompt = _feedback_prompt(policy,trajectory,row,
                "Hindsight consequence of the next action:\n"+row.raw_observation)
            with torch.no_grad(),teacher.activate(policy.model),policy.phase("self.teacher"):
                teacher_hidden = _response_hidden(policy,teacher_prompt,target,"self.teacher_forward").detach().cpu()
            policy.model.train()
            student = _response_hidden(policy,prompt,target,"self.student_action")
            loss = None
            for start in range(0,len(target),chunk_tokens):
                stop = min(start+chunk_tokens,len(target))
                def objective(h,th=teacher_hidden[start:stop],den=len(target)*16):
                    with torch.no_grad():
                        tea = F.log_softmax(policy.head(th,role="self.teacher_head"),dim=-1)
                    stu = F.log_softmax(policy.head(h,role="self.student_head"),dim=-1)
                    return (stu.exp()*(stu-tea.to(stu))).sum()/den
                with policy.activation_saving():
                    kl = checkpoint(objective,student[start:stop],use_reentrant=False)
                loss = kl if loss is None else loss+kl
                cursor.metrics["kl"] += float(kl.detach())
            with policy.phase("self.action_backward"):
                loss.backward()
            feedback = policy.encode(row.raw_observation)
            closure = (198,) if target[-1]==151645 else (151645,198)
            assistant = (151644,)+policy.encode("assistant\n")
            feedback_tail = assistant+target+closure+(151644,)+policy.encode("user\nPredict the exact environmental feedback.")+(151645,198,151644)+policy.encode("assistant\n")
            cursor.metrics["feedback_nll"] += _feedback_nll_backward(policy,trajectory,row,
                prompt[:-len(assistant)],feedback_tail,feedback,chunk_tokens=chunk_tokens)
            cursor.row += 1
            policy.model.eval()
            if on_boundary:
                on_boundary(cursor)
        with policy.phase("self.adamw"):
            optimizer.step(max_norm=None)
        with policy.phase("self.ema"):
            teacher.update_ema(policy.model,.99)
        optimizer.zero_grad()
        cursor.update += 1
        cursor.row,cursor.minibatch = 0,[]
        cursor.metrics["steps"] += 1
        if on_boundary:
            on_boundary(cursor)
    cursor.finished = True
    return cursor


@dataclass
class Skill:
    name: str
    description: str
    signature: str
    parameters: list
    preconditions: list
    postconditions: list
    side_effects: list
    canonical_action_sequence: list
    examples: list
    coverage: list
    implementation: str = ""
    revision: int = -1
    calls: list = field(default_factory=list)

    def public(self):
        return {key:getattr(self,key) for key in ("name","description","signature","parameters",
                                                "preconditions","postconditions","side_effects")}


def _skill_process(pipe,source,name,arguments,observation,available):
    class Environment:
        def __init__(self):
            self.observation,self.available_actions = observation,available
        def step(self,action):
            pipe.send({"kind":"step","action":action})
            reply = pipe.recv()
            if reply["kind"]=="error":
                raise RuntimeError(reply["error"])
            self.observation,self.available_actions = reply["observation"],reply["available"]
            return self.observation,self.available_actions
    denied = {"os","sys","subprocess","socket","pathlib","shutil","ctypes","multiprocessing",
              "importlib","resource","signal","pickle","marshal","shelve","dbm","sqlite3","http",
              "urllib","ftplib","smtplib","telnetlib","pty","fcntl","posix","nt"}
    def import_stdlib(module,globals=None,locals=None,fromlist=(),level=0):
        root = module.split(".",1)[0]
        if level or root not in sys.stdlib_module_names or root in denied:
            raise ImportError("Independent skills use stdlib computation and native env.step")
        return builtins.__import__(module,globals,locals,fromlist,level)
    scope_builtins = {k:v for k,v in vars(builtins).items() if k not in
                      {"open","exec","eval","compile","input","breakpoint","__import__"}}
    scope_builtins["__import__"] = import_stdlib
    scope = {"__builtins__":scope_builtins,"__name__":"skill_disco","env":Environment()}
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exec(compile(source,"<skill_disco>","exec"),scope)
            result = scope[name](**arguments)
        if not isinstance(result,dict):
            raise TypeError("A procedural skill returns a structured dictionary")
        pipe.send({"kind":"result","result":result,"stdout":output.getvalue()})
    except BaseException as error:
        pipe.send({"kind":"error","error":f"{type(error).__name__}: {error}",
                   "traceback":traceback.format_exc(),"stdout":output.getvalue()})
    finally:
        pipe.close()


@dataclass
class SkillState:
    phase: str = "collect"
    task_cursor: int = 0
    trace_cursor: int = 0
    piece_cursor: int = 0
    extraction_cursor: int = 0
    symbols: dict = field(default_factory=dict)
    batch_cursor: int = 0
    candidate_cursor: int = 0
    verification_round: int = 0
    successes: list = field(default_factory=list)
    operations: list = field(default_factory=list)
    local_groups: list = field(default_factory=list)
    consolidated: list = field(default_factory=list)
    skills: list = field(default_factory=list)
    journal: dict = field(default_factory=dict)
    proposal_calls: int = 0
    finished: bool = False


class SkillLibrary:
    def __init__(self,policy,config,*,state=None,on_boundary=None):
        self.policy,self.config = policy,config
        self.state = state or SkillState()
        self.on_boundary = on_boundary
        self.record_training_calls = True

    @property
    def skills(self):
        return self.state.skills

    def boundary(self):
        if self.on_boundary:
            self.on_boundary(self.state)

    def _messages(self,instruction,payload):
        return [{"role":"system","content":instruction+"\nReturn one JSON value."},
                {"role":"user","content":json.dumps(payload,ensure_ascii=False)}]

    def _fits(self,instruction,payload):
        return len(_chat_ids(self.policy,self._messages(instruction,payload)))+8192 <= self.policy.max_context_tokens

    def batches(self,items,instruction,base,*,maximum=32):
        batches,current = [],[]
        for item in items:
            trial = current+[item]
            if len(trial)>maximum or not self._fits(instruction,{**base,"items":trial}):
                if not current:
                    raise ValueError("An indivisible real Skill-DisCo source object exceeds the Qwen context")
                batches.append(current)
                current = [item]
                if not self._fits(instruction,{**base,"items":current}):
                    raise ValueError("An indivisible real Skill-DisCo source object exceeds the Qwen context")
            else:
                current = trial
        if current:
            batches.append(current)
        return batches

    def ask(self,key,instruction,payload):
        if key in self.state.journal:
            return self.state.journal[key]
        messages = self._messages(instruction,payload)
        prompt = _chat_ids(self.policy,messages)
        if len(prompt)+8192>self.policy.max_context_tokens:
            raise ValueError("Complete Skill-DisCo proposal exceeds its explicit Qwen context adaptation")
        generator = self.policy.new_generator(int(self.config["seed"])*1000003+self.state.proposal_calls+900001)
        target,logp = sample_ids(self.policy,prompt,generator,8192,role="skill_disco."+key.split(":",1)[0])
        text = self.policy.tokenizer.decode(target,skip_special_tokens=True).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$","",text)
        value = json.loads(text)
        self.state.proposal_calls += 1
        self.state.journal[key] = value
        self.policy.record("proposal","skill_disco.parsed",details={"key":key,"prompt_ids":prompt,
                           "response_ids":target,"old_logprobs":logp,"value":value})
        self.boundary()
        return value

    def stats(self,skill):
        calls = [c for c in skill.calls if c["revision"]==skill.revision]
        n = len(calls)
        errors = sum(bool(c["error"]) for c in calls)
        successes = sum(bool(c["result"].get("success",False)) for c in calls)
        rate = successes/n if n else None
        return {"name":skill.name,"revision":skill.revision,"invocations":n,"unhandled_exceptions":errors,
                "self_reported_success_rate":rate,"failed":bool(errors or (n>=3 and rate<.30))}

    def execute(self,env,action,observation,available,remaining_steps):
        selected = next((s for s in self.skills if s.name==action["name"]),None)
        if selected is None:
            error = {"result":{},"error":"NameError: no such induced procedural skill", "primitive_count":0,"done":False}
            self.policy.record("controller","skill_disco.unknown_callable",details={"action":action,**error})
            return error
        context = multiprocessing.get_context("spawn")
        parent,child = context.Pipe()
        process = context.Process(target=_skill_process,args=(child,selected.implementation,selected.name,
                                                             action["arguments"],observation,available))
        started = clock()
        process.start();child.close()
        native,result,error,done,trace = [],{},"",False,""
        native_calls,environment_reward,stop_reason = 0,0.,""
        try:
            while True:
                try:
                    message = parent.recv()
                except EOFError:
                    error = "Skill executor exited without a return value"
                    break
                if message["kind"]=="step":
                    if native_calls>=remaining_steps or done:
                        stop_reason,done = "official_horizon_or_terminal",True
                        break
                    native_started = clock()
                    try:
                        parsed = (dict(message["action"]) if isinstance(message["action"],dict)
                                  else parse_action(message["action"],self.policy.benchmark))
                    except Exception as exception:
                        parent.send({"kind":"error","error":f"{type(exception).__name__}: {exception}"})
                        continue
                    native_calls += 1
                    try:
                        step = env.step_native(parsed)
                    except Exception as exception:
                        error_text = f"{type(exception).__name__}: {exception}"
                        self.policy.record("environment","skill_disco.native_error",seconds=clock()-native_started,env_steps=1,
                                           details={"action":parsed,"error":error_text})
                        native.append({"action":message["action"],"error":error_text})
                        parent.send({"kind":"error","error":error_text})
                        continue
                    self.policy.record("environment","skill_disco.native_step",seconds=clock()-native_started,env_steps=1,
                                       details={"action":parsed,"metadata":step.metadata})
                    native.append({"action":message["action"],"observation":step.observation})
                    environment_reward += float(step.reward)
                    done = bool(step.done)
                    parent.send({"kind":"observation","observation":_pure(step.observation),
                                 "available":_available(env.action_schema)})
                elif message["kind"]=="result":
                    result = message["result"]
                    break
                else:
                    error,trace = message["error"],message.get("traceback","")
                    break
        finally:
            if process.is_alive():
                process.terminate()
            process.join();parent.close()
            self.policy.record("controller","skill_disco.python",seconds=clock()-started,
                               details={"skill":selected.name,"revision":selected.revision,"error":error})
        receipt = {"revision":selected.revision,"arguments":action["arguments"],"result":result,"error":error,
                   "traceback":trace,"native_steps":native,"primitive_count":native_calls,"done":done,
                   "environment_reward":environment_reward,"stop_reason":stop_reason}
        if self.record_training_calls:
            selected.calls.append(receipt)
        return receipt

    def _synthesize(self,skill,key,feedback):
        instruction = ("Synthesize the complete typed Python procedural skill using global env.step(action), returning "
            "(observation,available_actions). Initialize local state from env.observation/env.available_actions. "
            "Use exact native command/Python API syntax; branch on latest real feedback. Use stdlib computation and env.step. "
            "Return {implementation:str}. The function returns success,observation,available_actions,process_trace. "
            "Incorporate every supplied example and refinement feedback; previous implementation is a partial compilation artifact.")
        base = {"contract":skill.public(),"feedback_statistics":feedback.get("statistics",{})}
        material = [{"kind":"source_operation","value":example} for example in skill.examples]
        material.extend({"kind":"actual_invocation","value":{k:v for k,v in call.items() if k!="environment_reward"}}
                        for call in feedback.get("calls",()))
        implementation = skill.implementation
        offset = 0
        while offset<len(material):
            payload = {**base,"previous_implementation":implementation}
            batch = self.batches(material[offset:],instruction,payload)[0]
            value = self.ask(f"synthesize:{key}:{offset}",instruction,{**payload,"items":batch})
            implementation = value["implementation"]
            offset += len(batch)
        skill.implementation = implementation
        skill.revision += 1

    def train(self,tasks,env_factory):
        s = self.state
        count = 200 if self.policy.benchmark=="alfworld" else 90
        if len(tasks)<count:
            raise ValueError("The complete author induction set must be supplied")
        induction = list(tasks[:count])
        if s.phase=="collect":
            while s.task_cursor<count:
                i = s.task_cursor
                self.policy.ledger.begin_group(f"skill-induction-{i}")
                trajectory = collect_native_trajectory(self.policy,env_factory,task_id=induction[i]["task_id"],
                    initial_seed=int(self.config["seed"])+i,sampling_seed=int(self.config["seed"])*1000003+i,
                    group_id=f"skill-induction-{i}",slot=0,method="skill_disco",config=self.config)
                if trajectory.official_success:
                    s.successes.append(trajectory)
                s.task_cursor += 1
                self.boundary()
            s.phase = "discover"
        if s.phase=="discover":
            while s.trace_cursor<len(s.successes):
                trajectory = s.successes[s.trace_cursor]
                trace = []
                previous = self.policy.tokenizer.decode(trajectory.initial_observation_ids)
                for row in trajectory.steps:
                    trace.append({"index":row.index,"observation":previous,"response":self.policy.tokenizer.decode(row.response_ids),
                                  "action":dict(row.action),"feedback":row.raw_observation})
                    previous = row.raw_observation
                instruction = ("Normalize this ordered portion of a real successful trace into Python-like control flow. "
                    "Preserve every observed action/feedback and global source index, including recovery. Carry prior symbols/interfaces "
                    "for continuity; this portion is not a complete trajectory. Return {program:str,symbols:dict}.")
                base = {"task":trajectory.goal,"bootstrap":self.policy.tokenizer.decode(trajectory.initial_observation_ids)}
                while s.piece_cursor<len(trace):
                    i = s.piece_cursor
                    piece = self.batches(trace[i:],instruction,{**base,"prior_symbols":s.symbols},maximum=100)[0]
                    program = self.ask(f"normalize:{s.trace_cursor}:{i}",instruction,{**base,"items":piece,"prior_symbols":s.symbols})
                    symbols = program["symbols"]
                    extract_instruction = ("Extract coherent multi-primitive subgoal operations from this normalized trace portion. "
                        "Use authoritative observed action/feedback and global step indices. Return "
                        "{operations:[{description,code_snippet,start_step,end_step,parameters,succeeded}]}. "
                        "Emit an operation only in the piece containing its start_step. "
                        "succeeded uses final observed subgoal feedback. Preserve code/control flow.")
                    extract_base = {**base,"program":program["program"],"symbols":symbols}
                    extract_pieces = self.batches(piece,extract_instruction,extract_base,maximum=100)
                    while s.extraction_cursor<len(extract_pieces):
                        j = s.extraction_cursor
                        extracted = self.ask(f"extract:{s.trace_cursor}:{i}:{j}",extract_instruction,
                                             {**extract_base,"items":extract_pieces[j]})
                        for operation in extracted["operations"]:
                            a,b = int(operation["start_step"]),int(operation["end_step"])
                            if 0<=a<b<len(trace) and operation["succeeded"]:
                                s.operations.append({**operation,"operation_id":len(s.operations),"trajectory_id":trajectory.trajectory_id,
                                                     "real_trace":trace[a:b+1]})
                        s.extraction_cursor += 1
                        self.boundary()
                    s.symbols = symbols
                    s.piece_cursor += len(piece)
                    s.extraction_cursor = 0
                    self.boundary()
                s.trace_cursor += 1
                s.piece_cursor,s.symbols = 0,{}
                self.boundary()
            s.phase = "group"
        if s.phase=="group":
            instruction = ("Group operations by shared reusable control-flow/action-observation structure and subgoal. "
                "Return {groups:[{description,operation_ids:[int]}]}. Use only source IDs; prefer compact non-overlapping groups.")
            batches = self.batches(s.operations,instruction,{},maximum=32)
            while s.batch_cursor<len(batches):
                value = self.ask(f"group:{s.batch_cursor}",instruction,{"items":batches[s.batch_cursor]})
                s.local_groups.extend(value["groups"])
                s.batch_cursor += 1
                self.boundary()
            s.phase,s.batch_cursor = "consolidate",0
        if s.phase=="consolidate":
            instruction = ("Incrementally consolidate semantically equivalent operation groups into a minimal reusable skill set. "
                           "Keep every source operation ID; return {groups:[{description,operation_ids:[int]}]}.")
            batches = self.batches(s.local_groups,instruction,{},maximum=32)
            while s.batch_cursor<len(batches):
                value = self.ask(f"consolidate:{s.batch_cursor}",instruction,
                                {"items":batches[s.batch_cursor],"existing_groups":s.consolidated})
                s.consolidated = value["groups"]
                s.batch_cursor += 1
                self.boundary()
            s.phase = "compile"
        if s.phase=="compile":
            by_id = {op["operation_id"]:op for op in s.operations}
            while s.candidate_cursor<len(s.consolidated):
                i = s.candidate_cursor
                group = s.consolidated[i]
                source_ids = dict.fromkeys(int(j) for j in group["operation_ids"])
                examples = [by_id[j] for j in source_ids if j in by_id]
                support = sorted({op["trajectory_id"] for op in examples})
                if len(support)>=2:
                    instruction = ("Define/update a complete reusable typed skill contract from all these successful operation examples. "
                        "Return {name,description,signature,parameters,preconditions,postconditions,side_effects,canonical_action_sequence}. "
                        "env is global; all parameters are domain values. Preserve compatible control-flow variants.")
                    contract = {}
                    offset = 0
                    while offset<len(examples):
                        payload = {"group":group,"prior_contract":contract}
                        batch = self.batches(examples[offset:],instruction,payload)[0]
                        contract = self.ask(f"contract:{i}:{offset}",instruction,{**payload,"items":batch})
                        offset += len(batch)
                    skill = Skill(**contract,examples=examples,coverage=support)
                    self._synthesize(skill,f"initial:{i}",{})
                    s.skills.append(skill)
                s.candidate_cursor += 1
                self.boundary()
            s.phase,s.task_cursor = "verify",0
            if not s.skills:
                s.finished,s.phase = True,"finished"
        if s.phase=="verify":
            while s.verification_round<4:
                while s.task_cursor<count:
                    i = s.task_cursor
                    self.policy.ledger.begin_group(f"skill-verification-{s.verification_round}-{i}")
                    collect_native_trajectory(self.policy,env_factory,task_id=induction[i]["task_id"],
                        initial_seed=int(self.config["seed"])+i,
                        sampling_seed=int(self.config["seed"])*1000003+100000+s.verification_round*count+i,
                        group_id=f"skill-verification-{s.verification_round}-{i}",slot=0,method="skill_disco",
                        config=self.config,library=self)
                    s.task_cursor += 1
                    self.boundary()
                failed = [skill for skill in s.skills if self.stats(skill)["failed"]]
                if not failed or s.verification_round==3:
                    s.skills = [skill for skill in s.skills if not self.stats(skill)["failed"]]
                    s.finished,s.phase = True,"finished"
                    break
                for skill in failed:
                    calls = [call for call in skill.calls if call["revision"]==skill.revision]
                    self._synthesize(skill,f"refine:{s.verification_round}:{skill.name}",
                                     {"statistics":self.stats(skill),"calls":calls})
                s.verification_round += 1
                s.task_cursor = 0
                self.boundary()
        self.boundary()
        return s
