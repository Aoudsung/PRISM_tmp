# The datasets will be uploaded to huggingface 

def make_prefix(dp, dataset="", belief_cot=True):
    # `dp`, `belief_cot` are deprecated
    if dataset == 'CircuitDecoding':
        prefix = CircuitDecoding
    elif dataset == 'MovieRec':
        # PE and MR described in the paper share the same prompt, but differ in the final-turn instruction.
        prefix = MovieRec_pref_est
    else:
        assert NotImplementedError(f"dataset {dataset} not implemented")
    return prefix


CircuitDecoding = """**Welcome to the Circuit Deduction Challenge!**

**The Setup:**
* There are {num_circuits} circuits, labeled {circuit_labels}.
* Each circuit accepts {num_inputs} binary inputs (0 or 1) and produces a single binary output (0 or 1).
* Each circuit is drawn from a fixed candidate list of {num_candidates} possible logical structures, each associated with an index:

{candidate_list_str}

**Your Goal:** Identify which circuits from the candidate list correspond exactly to circuits {circuit_labels}.

**How to Play:**
You can interact with me for several turns to determine the true underlying circuits:
1. At each turn, query one circuit with any binary input of your choice.
2. Use the specified format for your query. For example, to query circuit A with inputs x0=1, x1=0, x2=1, ask: `<interact>A(1, 0, 1)</interact>`.
3. You must make only one query at each turn. I will return the binary output for that circuit on the given input.
4. Ask **strategic queries** that maximize information gain. Your goal is to minimize the number of turns by leveraging the feedback at each step to **narrow down the candidate possibilities**.

**Final Submission:**
Once you are confident, submit your final answer by providing the indices of the identified circuits from the candidate list inside <answer> and </answer>.  
For example, if A corresponds to candidate 13 and B corresponds to 6, your answer must be: <answer>13, 6</answer>.

Please start with your first query.
"""


MovieRec_pref_est = """You are a movie recommendation agent. 
Your goal is to infer the hidden user preference vector (w1,...,w{len_attributes}) through interaction.

**Setup:**
- You are given {len_seen} movies with scores on {len_attributes} attributes (indexed 1...{len_attributes}):
{seen_movie_sample}
- User satisfaction = w1*attr1 + ... + w{len_attributes}*attr{len_attributes}, 
  where each wi ∈ [0,1]. The user always answers consistently.

**Interaction Rules (per round):**
1. Reflect on all past feedback and reason about how it changes your estimate of the preference vector.  
   - Think about which attributes gained or lost importance.  
   - Adjust your estimate strategically.  
2. Output both your updated guess and a new pairwise query in the exact format:

<interact>
Guess: w1,w2,...,w{len_attributes}
Question: Would you prefer $option_1 over $option_2?
</interact>

- Guess must be comma-separated numbers in [0,1].  
- $option_1 and $option_2 must be movie names only.  

The user replies with one of: "Yes" (prefer $option_1), "No" (prefer $option_2), or "Equal".

**Final Stage:**
- Once you are confident about the user preference after several turns, output your final preference vector as:

<answer>w1,w2,...,w{len_attributes}</answer>

Please Start with your first <interact> block.
"""
