import re
from copy import deepcopy

import numpy as np


class MovieRecSpace:
    def __init__(self, controller, max_turns, global_step=0, trunc_strength=None, epsilon_win=None):
        # `global_step` is deprecated
        # `epsilon_win` controls unsupervised early truncation (Appendix D3 of the paper).
        # `trunc_strength` is only meaningful for the supervised variant.

        self.controller = controller
        self.user_pref = controller['user_preference_function']
        self.seen_movies = controller['seen_movies_names'].tolist()
        self.seen_movies_scores = controller['seen_movies_attribute_scores']
        self.unseen_movies = controller['unseen_movies_names']
        self.unseen_movies_scores = controller['unseen_movies_attribute_scores']
        self.seen_scores = controller['sort_seen']

        self.current_pref = None
        self.all_diff = []
        self.all_queries = []
        self.cut = 0.
        self.must_stop = 0.
        self.max_counts = max_turns - 1
        self.query_counts = 0
        self.repeat_counts = 0
        self.stalling_counts = 0
        self.consecutive_drop_counts = 0
        self.tolerate_num = trunc_strength
        self.epsilon_win = epsilon_win
        self.all_adj = []

        # for asmp1 verification, deprecated now
        self.asp1_record_vec = []
        self.asp1_record_mij = []
        self.asp1_record_yij = []

        self.answer_instruct = """Final Turn: In this turn, output your most recent estimate of the preference vector. The response must be formatted in <answer> and </answer>.
        """
        # unseen moview generalization
        # self.answer_instruct = answer_instruct.format(
        #     unseen_movie_sample="\n".join([f"{k}: {v}" for k, v in zip(controller['unseen_movies_names'], controller['unseen_movies_attribute_scores'])]),
        # )
    
    def compute_feedback(self, guess: str, validate: bool = False):
        
        if not validate and self.must_stop == 1.:
            print("The traj must stop.")
            return None
        
        self.query_counts += 1

        try:
            feedback = self._compute_feedback(guess)
        except Exception as e:
            self.cut = 1.
            print(e)
            feedback = f"The query you made is invalid."
            self.asp1_record_yij.append(None)
            self.asp1_record_mij.append(None)
        
        try:
            cur_pref = self._extract_guess(guess)
        except Exception as e:
            print(e)
            cur_pref = None
        self.asp1_record_vec.append(cur_pref)

        if self.current_pref is not None:
            # supervised manner
            # diff = self.cal_sim(cur_pref, self.user_pref) - self.cal_sim(self.current_pref, self.user_pref) if cur_pref is not None else -1
            # self.all_diff.append(diff)
            # if diff <= 0:
            #     self.stalling_counts += 1
            #     print(f"{self.stalling_counts} -- stalling.")
            #     self.cut = 1.
            # self.consecutive_drop_counts = self.max_consecutive_negatives(self.all_diff)
            # if self.consecutive_drop_counts >= 2:
            #     print(f"Consecutive drop. {self.consecutive_drop_counts}")
            
            # unsupervised manner (See Appendix D3 of the paper)
            diff = np.linalg.norm(cur_pref - self.current_pref) if cur_pref is not None else 0
            self.all_diff.append(diff)

        if cur_pref is not None:
            self.current_pref = cur_pref

        if self.query_counts >= self.max_counts:
            # self.must_stop = 1.
            return feedback + '\n' + self.answer_instruct

        ## Early-truncation supervised
        # if self.detect_truncation() and not validate:
        ## Early-truncation unsupervised (See Appendix D3 of the paper)
        if self.detect_truncation_unsup() and not validate:
            self.must_stop = 1.
            return feedback + '\n' + self.answer_instruct 
        return feedback + '\n' + 'First, think about how the latest feedback should adjust your 8-dimensional preference vector. Then update your estimate of the vector and issue the next query inside <interact>...</interact>.'

    def _compute_feedback(self, guess):
        query_pattern = r'Would you prefer (.+?) over (.+?)\?'
        
        prefer_match = re.search(query_pattern, guess)
        assert prefer_match and len(prefer_match.groups()) == 2, f"Failed in extracting two objects."
        tgt1, tgt2 = prefer_match.groups()
        assert tgt1 in self.seen_movies and tgt2 in self.seen_movies, f"Failed in extracting two objects."
        scr1, scr2 = self.seen_scores[self.seen_movies.index(tgt1)], self.seen_scores[self.seen_movies.index(tgt2)]
        feedback = "Yes." if scr1 > scr2 else "Equal." if scr1 == scr2 else "No."
        self.asp1_record_yij.append(1 if scr1 > scr2 else 0 if scr1 == scr2 else -1)
        self.asp1_record_mij.append(
            self.seen_movies_scores[self.seen_movies.index(tgt1)] - self.seen_movies_scores[self.seen_movies.index(tgt2)]
        )
        
        if (tgt1, tgt2) in self.all_queries:
            self.repeat_counts += 1
            print(f"{self.repeat_counts} -- repeating query type.")
            self.cut = 1.
        self.all_queries.append((tgt1, tgt2))
        
        return feedback

    def _extract_guess(self, guess):
        vec_pattern = r'([\d\.]+)'
        numbers = re.findall(vec_pattern, guess)
        assert len(numbers) == len(self.user_pref), f"Expected {len(self.user_pref)} dim vector. Output {len(numbers)}"
        assert all(float(k) >= 0 for k in numbers), f"All guess must be positive. Output {numbers}"

        return np.array(numbers, dtype=float)

    def detect_truncation(self):
        # truncation condition without ground-truth
        return self.consecutive_drop_counts >= self.tolerate_num

    def detect_truncation_unsup(self):
        # truncation condition without ground-truth (See Appendix D3 and D4 of the paper)
        # In unsupervised manner, we use 4 as the default window size
        if len(self.all_diff) > 4:
            self.all_diff.pop(0)

        if len(self.all_diff) == 4:
            avg_delta = sum(self.all_diff) / 4
            if avg_delta < self.epsilon_win:
                self.all_adj.append(avg_delta)
                return True
        return False

    @staticmethod
    def cal_sim(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    @staticmethod
    def max_consecutive_negatives(nums):
        max_count = 0
        current_count = 0
        for num in nums:
            if num <= 0:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


class CircuitDecodingSpace:
    def __init__(self, controller, global_step=0, trunc_strength=None):
        # `global_step` is deprecated
        self.controller = controller
        self.circuits = controller['circuits']
        self.num_inputs = controller['num_inputs']
        self.cut = 0.
        self.must_stop = 0.
        self.query_counts = 0
        self.repeat_counts = 0
        self.stalling_counts = 0
        self.candidate_spaces = {k: deepcopy(controller['candidates']) for k in range(len(self.circuits))}
        self.all_prob = self.get_all_prob()
        self.max_counts = 14
        self.all_queries = []
        self.tolerate_num = trunc_strength


    def compute_feedback(self, guess: str, validate: bool = False):
        
        if not validate and self.must_stop == 1.:
            print("The traj must stop.")
            return None
        
        self.query_counts += 1
        if guess in self.all_queries:
            self.repeat_counts += 1
            print(f"{self.repeat_counts} -- repeating query type.")
            self.cut = 1.
        self.all_queries.append(guess)

        try:
            feedback = self._compute_feedback(guess, self.circuits, self.num_inputs)
        except Exception:
            print("invalid type.")
            self.cut = 1.
            feedback = f"The query you made is invalid."

        if self.get_all_prob() == self.all_prob:
            self.stalling_counts += 1
            print(f"{self.stalling_counts} -- stalling.")
            self.cut = 1.
        self.all_prob = self.get_all_prob()

        if self.query_counts >= self.max_counts:
            feedback = feedback + '\n' + f"Maximum valid query count exceeded. Now present your answer inside <answer> and </answer> using the format specified in the instruction."

        ## Early-truncation
        if self.detect_truncation() and not validate:
            return feedback + '\n' + f"Maximum valid query count exceeded. Now present your answer inside <answer> and </answer> using the format specified in the instruction."

        return feedback

    def detect_truncation(self):
        if self.stalling_counts >= self.tolerate_num:
            self.must_stop = 1.
            return True
        return False
    
    def get_all_prob(self):
        return sum(len(itm) for k,itm in self.candidate_spaces.items())
    
    def _compute_feedback(self, query: str, circuit_formulas: list[str], num_inputs: int):
        
        match = re.match(r"([A-Z])\((\s*\d\s*,\s*\d\s*,\s*\d\s*)\)", query.strip())
        if not match:
            raise ValueError(f"Invalid query format: {query}.")

        circuit_label = match.group(1)
        inputs = tuple(map(int, match.group(2).split(",")))
        if len(inputs) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(inputs)}")

        circuit_index = ord(circuit_label) - ord("A")
        if circuit_index >= len(circuit_formulas):
            raise ValueError(f"Circuit {circuit_label} not found (only {len(circuit_formulas)} circuits)")

        feedback = self.get_results(circuit_formulas[circuit_index], inputs) 
        
        self.candidate_spaces[circuit_index] = [itm for itm in self.candidate_spaces[circuit_index] if self.get_results(itm,inputs) == feedback]
        
        return feedback

    @staticmethod
    def get_results(expr, inputs):
        env = {f"x{i}": bool(inputs[i]) for i in range(len(inputs))}
        val = eval(expr, {"__builtins__": None}, env)
        return str(int(val))


answer_instruct = """Final Turn:  
Now you have reached the last turn. Instead of asking a new question, use your most recent preference guess to score the following unseen movies and recommend the best one.  

{unseen_movie_sample}

**Here is an example of how to proceed**:  

Preference vector (guess): 0.2,0.7,0.5  
Example Unseen movies:  
Movie_A: [0.6,1.0,0.8]  
Movie_B: [1.2,0.3,0.4]   
Movie_C: [0.5,0.8,0.9]  

Scoring:  
Movie_A = 0.2*0.6 + 0.7*1.0 + 0.5*0.8 = 1.22
Movie_B = 0.2*1.2 + 0.7*0.3 + 0.5*0.4 = 0.65  
Movie_C = 0.2*0.5 + 0.7*0.8 + 0.5*0.9 = 1.11  
Best = Movie_A  

<answer>Movie_A</answer>

**Your goal**:
Now do the same with your own latest preference vector and the given unseen movies.  
After scoring, return the final answer enclosed within <answer> and </answer>. The answer must be exactly one of the unseen movie names.

"""
