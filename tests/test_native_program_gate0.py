"""Focused server-only checks for the Gate 0 source and physical-ledger contracts.

All fixtures are explicitly synthetic; no fixture is a real 7B output or part
of the frozen Gate 0 experiment denominator.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from prism_evolve.native_core import BudgetLimits, BudgetMeter, ExplorationLedger, Resources
from prism_evolve.native_program_ir import (
    Effect, Node, ProgramCandidate, Source, liveness, native_deletion_version,
    strict_source_replay, validate,
)
from prism_evolve.native_program_execution import execute
from prism_evolve.native_trace_artifacts import canonical_bytes, public_view
from prism_evolve.native_trace_compile import build_material, construct_candidate


class NativeProgramGate0Tests(unittest.TestCase):
    def test_physical_failure_is_closed_with_unknown_usage_and_one_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = BudgetMeter(BudgetLimits(10, 1000, 60), event_path=Path(directory) / "cost.jsonl", charged_round_id="r1")
            child = BudgetMeter(BudgetLimits(2, 500, 10), parent=parent)
            call = child.start_model_call("compiler.annotate_trace", {"model": "synthetic"}, "p1")
            call.generation_started = True
            call.input_tokens, call.output_tokens = 12, None
            call.status = "ERROR"
            child.finish_model_call(call)
            rows = [json.loads(line) for line in parent.event_path.read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows], ["call_started", "call_finished", "model_call"])
            self.assertIsNone(rows[1]["output_tokens"])
            self.assertFalse(rows[1]["usage_complete"])
            self.assertEqual(rows[2]["status"], "ERROR")
            self.assertEqual(rows[2]["known_input_tokens"], 12)
            self.assertEqual(parent.total_model_tokens, child.total_model_tokens)
            self.assertFalse(parent.usage_complete)
            self.assertEqual(parent.cost_events[0]["cost_event_id"], child.cost_events[0]["cost_event_id"])
            with self.assertRaises(ValueError):
                child.finish_model_call(call)

    def test_native_policy_complete_finally_closes_generation_exception(self):
        import torch
        from prism_evolve.native_policy import NativeCausalPolicy
        policy = NativeCausalPolicy.__new__(NativeCausalPolicy)
        torch.nn.Module.__init__(policy)
        policy.meter = BudgetMeter(BudgetLimits(5, 1000, 60))
        policy.model_metadata = {"model": "synthetic_no_model_loaded"}
        def fail(*args, call_record, **kwargs):
            call_record.generation_started = True
            call_record.input_tokens = 12
            call_record.output_tokens = None
            raise RuntimeError("synthetic generation error")
        policy._complete_impl = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic generation error"):
            policy.complete([{"role": "user", "content": "fixture"}])
        self.assertEqual(len(policy.meter.model_calls), 1)
        call = policy.meter.model_calls[0]
        self.assertIsNotNone(call.finished_at)
        self.assertEqual(call.status, "ERROR")
        self.assertIsNone(call.output_tokens)
        self.assertFalse(call.usage_complete)
        self.assertTrue(policy.meter.exploration_stopped)

    def test_finished_call_uses_the_actual_parent_deadline(self):
        from unittest.mock import patch
        parent = BudgetMeter(BudgetLimits(10, 1000, 2))
        child = BudgetMeter(BudgetLimits(5, 500, 10), parent=parent)
        parent.started = child.started = 100.0
        with patch("prism_evolve.native_core.monotonic_seconds", side_effect=[100.1, 102.1]):
            call = child.start_model_call("synthetic", {}, "p")
            call.status, call.usage_complete = "OK", True
            child.finish_model_call(call)
        self.assertEqual(call.lease_status, "LEASE_OVERRUN")
        self.assertTrue(parent.exploration_stopped)

    def test_return_bound_parameter_is_local_and_repair_is_explicit(self):
        before = {"native_observation": "{\"id\":1}", "classic_text_action_state": {"available_actions": []}}
        view = {"public_contract": "synthetic", "goal": "inspect course", "events": [
            {"event_id": "e00000", "obs_before": before, "obs_after": before,
             "action": {"kind": "native_text", "text": '{"name":"list","arguments":{}}'},
             "response": '{"name":"list","arguments":{}}', "schema": {}},
            {"event_id": "e00001", "obs_before": before,
             "obs_after": {**before, "native_observation": "Course not found: 999"},
             "action": {"kind": "native_text", "text": '{"name":"get","arguments":{"course_id":999}}'},
             "response": '{"name":"get","arguments":{"course_id":999}}', "schema": {}}]}
        material = build_material(view, 2)
        # Add an independently typed exact public span; the repaired action is unexecuted.
        document = "e00001.before.native"
        material.sources["typed-id"] = Source("typed-id", 1, "public_return", "Number", 1, document, 6, 7)
        annotation = {"parameters": [{"name": "course_id", "type": "Number", "source_id": "typed-id"}],
            "bindings": [{"event_index": 1, "source_id": "typed-id", "parameter": "course_id",
                          "action_substring": "999", "argument_path": "/arguments/course_id"}], "structures": []}
        candidate, _ = construct_candidate(material, annotation, {"predicates": []}, source_class="INTERFACE_REPAIR_HYPOTHESIS")
        self.assertEqual(candidate.entry_parameters, {})
        self.assertIn("call_1", candidate.unexecuted_nodes)
        self.assertEqual(candidate.native_success_count, 0)
        self.assertTrue(any(node.op == "Bind" for node in candidate.ast))

    def test_parent_child_environment_cost_identity_is_stable(self):
        parent = BudgetMeter(BudgetLimits(10, 100, 60), charged_round_id="round")
        child = BudgetMeter(BudgetLimits(5, 50, 30), parent=parent)
        child.record_env_step(0.2)
        child.record_api_calls(1, successful=0)
        self.assertEqual([event["cost_event_id"] for event in child.cost_events],
                         [event["cost_event_id"] for event in parent.cost_events])
        self.assertTrue(all(event["charged_round_id"] == "round" for event in parent.cost_events))

    def test_public_projection_ignores_private_diagnostics(self):
        observation = {"native_observation": "公開 observation", "feedback_type": "native_text_transition",
                       "classic_text_action_state": {"available_actions": ["look"], "state_changed": False}}
        trace = {"task_id": "fixture", "initial_observation": json.dumps(observation), "goal": "goal",
                 "events": [{"action": {"kind": "native_text", "text": "look"}, "response": "look",
                             "observation": json.dumps(observation), "sampling": {"native_action_schema": {}},
                             "native_step": {"state_changed": False}}]}
        altered = deepcopy(trace)
        altered["metadata"] = {"facts": "secret", "native_success": True}
        altered["events"][0]["native_step"] = {"state_changed": True, "evaluator": 99}
        self.assertEqual(canonical_bytes(public_view(trace)), canonical_bytes(public_view(altered)))

    def test_liveness_keeps_unknown_native_effects_and_eliminates_dead_calls(self):
        nodes = (Node("dead", "Bind", Effect.PURE_TOTAL, defines=("unused",)),
                 Node("aux", "SemanticBatch", Effect.AUX_CALL, defines=("also_unused",), rng_scope="auxiliary"),
                 Node("open", "Call", Effect.NATIVE_CALL, touches_history=True, args={"template": "open box"}),
                 Node("take", "Call", Effect.NATIVE_CALL, touches_history=True, args={"template": "take item"}))
        kept, live, report = liveness(nodes)
        self.assertEqual([node.node_id for node in kept], ["open", "take"])
        self.assertIn("$history", live)
        self.assertEqual({row["decision"] for row in report}, {"KEEP", "DELETE_PURE_TOTAL", "SKIP_UNDEMANDED_AUX"})

    def test_future_source_is_rejected_and_span_bytes_are_checked(self):
        source = Source("ref", 1, "public_return", "Text", "值", "doc", 0, 3)
        node = Node("bind", "Bind", Effect.PURE_TOTAL, defines=("x",), sources=("ref",), args={"type": "Text"})
        candidate = ProgramCandidate("", (node,), {}, {}, (), (), "LOCAL_WITNESS").seal()
        with self.assertRaisesRegex(ValueError, "Future source"):
            validate(candidate, {"ref": source}, {"doc": "值"})
        with self.assertRaises(ValueError):
            replace(source, end_utf8=2).verify({"doc": "值"})

    def test_native_deletion_has_new_version_and_no_inherited_successes(self):
        node = Node("native", "Call", Effect.NATIVE_CALL, touches_history=True)
        candidate = ProgramCandidate("", (node,), {}, {}, (), (), "LOCAL_WITNESS",
                                     native_execution_count=4, native_success_count=2).seal()
        deleted = native_deletion_version(candidate, "native")
        self.assertNotEqual(candidate.version, deleted.version)
        self.assertEqual(deleted.parent_version, candidate.version)
        self.assertEqual((deleted.native_execution_count, deleted.native_success_count), (0, 0))
        self.assertEqual(deleted.status, "UNTESTED")

    def test_strict_replay_stops_before_delivering_divergent_return(self):
        read, delivered = [], []
        def source(index):
            read.append(index)
            return {"action": ["a", "b", "c"][index],
                    "deliver_observation": lambda: delivered.append(index)}
        result = strict_source_replay(iter(["a", "different", "c"]), source, source_length=3, max_steps=5)
        self.assertEqual(result.status, "DIVERGED_AT_EVENT")
        self.assertEqual(result.first_divergence_event, 1)
        self.assertEqual(read, [0, 1])
        self.assertEqual(delivered, [0])
        self.assertEqual((result.native_execution_count, result.native_success_count), (0, 0))

    def test_finite_loop_obeys_budget_and_unknown_yields(self):
        sources = {name: Source(name, 0, "ActionRef", "Text", name, name, 0, 1) for name in ("a", "b")}
        frontier = Node("frontier", "Frontier", Effect.PURE_TOTAL, defines=("refs",), sources=("a", "b"),
                        args={"complete_collection": False})
        call = Node("call", "Call", Effect.NATIVE_CALL, reads=("item",), sources=("a",), touches_history=True,
                    args={"template": "${item}", "source_event": 0})
        loop = Node("loop", "Loop", Effect.CONTROL, reads=("refs",), body=(call,),
                    args={"item_value": "item", "finite_frontier": True})
        candidate = ProgramCandidate("", (frontier, loop), {}, {}, (), (), "LOCAL_WITNESS").seal()
        actions = []
        result = execute(candidate, sources, resolve_source=lambda source, prefix: source.value,
                         call_native=lambda action, node, prefix: actions.append(action) or "return",
                         action_is_bound=lambda action, prefix: True,
                         semantic_batch=lambda *args: self.fail("No semantic call is live"),
                         budget_available=lambda: len(actions) < 1)
        self.assertEqual(actions, ["a"])
        self.assertEqual(result.exit, "UnknownYield")
        self.assertEqual(result.unknown_reason, "BudgetUnknownYield")

    def test_joint_startup_lease_requires_credit_and_preserves_trial_reservation(self):
        ledger = ExplorationLedger()
        ledger.open_round("round", Resources(1800, 262144, 50))
        compile_lease, trial = Resources(60, 18432, 0), Resources(60, 9216, 3)
        self.assertIsNone(ledger.reserve_compile_and_trial("case", "round", compile_lease, trial))
        for index in range(10):
            ledger.credit_paid_base(str(index), Resources(60, 20000, 50), usage_complete=True)
        leases = ledger.reserve_compile_and_trial("case", "round", compile_lease, trial)
        self.assertIsNotNone(leases)
        ledger.settle(leases[0], Resources(20, 9000, 0), usage_complete=True)
        self.assertEqual(ledger.leases[leases[1]]["status"], "RESERVED")
        self.assertEqual(ledger._reserved().env_steps, 3)
        with self.assertRaises(ValueError):
            ledger.credit_paid_base("0", Resources(60, 20000, 50), usage_complete=True)

    def test_overrun_is_not_clipped_and_blocks_later_exploration(self):
        ledger = ExplorationLedger()
        ledger.open_round("round", Resources(1800, 262144, 50))
        ledger.credit_paid_base("paid", Resources(1000, 200000, 500), usage_complete=True)
        leases = ledger.reserve_compile_and_trial("case", "round", Resources(60, 18432, 0), Resources(60, 9216, 3))
        ledger.settle(leases[0], Resources(80, 20000, 0), usage_complete=False)
        self.assertEqual(ledger.spent.wall_seconds, 80)
        self.assertEqual(ledger.leases[leases[0]]["status"], "LEASE_OVERRUN")
        self.assertTrue(ledger.stopped)
        with self.assertRaises(ValueError):
            ledger.reserve_compile_and_trial("later", "round", Resources(1, 1, 0), Resources(1, 1, 1))

    def test_distinct_binding_loop_fold_is_deterministic(self):
        before = {"native_observation": "a b", "classic_text_action_state": {"available_actions": ["go a", "go b"]}}
        view = {"public_contract": "synthetic", "goal": "a b", "events": [
            {"event_id": f"e{index:05d}", "obs_before": before, "obs_after": before,
             "action": {"kind": "native_text", "text": action}, "response": action, "schema": {}}
            for index, action in enumerate(("go a", "go b"))]}
        material = build_material(view, 2)
        a = next(key for key, source in material.sources.items() if source.document_id == "goal" and source.value == "a")
        b = next(key for key, source in material.sources.items() if source.document_id == "goal" and source.value == "b")
        annotation = {"parameters": [{"name": "item", "type": "Ref", "source_id": a}],
            "bindings": [{"event_index": index, "source_id": ref, "parameter": "item", "action_substring": value}
                         for index, ref, value in ((0, a, "a"), (1, b, "b"))],
            "structures": [{"op": "Loop", "event_index": 0, "source_ids": [a, b],
                            "predicate_id": None, "body_events": [[0], [1]]}]}
        candidate, _ = construct_candidate(material, annotation, {"predicates": []}, source_class="LOCAL_WITNESS")
        self.assertEqual([node.op for node in candidate.ast], ["Frontier", "Loop"])
        self.assertEqual(candidate.ast[1].body[0].args["template"], "go ${loop_item_0}")
        self.assertEqual(candidate.native_execution_count, 0)

    def test_original_failed_call_cannot_be_recorded_as_executable_witness(self):
        before = {"native_observation": "course", "classic_text_action_state": {"available_actions": ["go course"]}}
        view = {"public_contract": "synthetic", "goal": "course", "events": [
            {"event_id": "e00000", "obs_before": before,
             "obs_after": {**before, "native_observation": "Course not found"},
             "action": {"kind": "native_text", "text": "go course"}, "response": "go course", "schema": {}}]}
        material = build_material(view, 1)
        ref = next(key for key, source in material.sources.items() if source.document_id == "goal")
        annotation = {"parameters": [{"name": "course", "type": "Ref", "source_id": ref}],
            "bindings": [{"event_index": 0, "source_id": ref, "parameter": "course", "action_substring": "course"}],
            "structures": []}
        with self.assertRaisesRegex(ValueError, "original failed call"):
            construct_candidate(material, annotation, {"predicates": []}, source_class="LOCAL_WITNESS")

    def test_input_structure_requires_two_distinct_consumed_bindings(self):
        from check_prism_gate0_input_structure import evidence_for_window
        actions = ["go desk 1", "examine desk 1", "go bed 1", "examine bed 1"]
        public = {"native_observation": "desk 1 bed 1", "classic_text_action_state": {"available_actions": actions}}
        view = {"events": [{"action": {"kind": "native_text", "text": action}, "response": action,
                            "obs_before": public, "obs_after": public} for action in actions]}
        self.assertFalse(evidence_for_window(view, 2)["loop_witnesses"])
        self.assertTrue(evidence_for_window(view, 4)["loop_witnesses"])
        repeated = deepcopy(view)
        for index in (2, 3):
            repeated["events"][index]["action"]["text"] = actions[index - 2]
        self.assertFalse(evidence_for_window(repeated, 4)["loop_witnesses"])

    def test_empty_semantic_demand_skips_request(self):
        aux = Node("aux", "SemanticBatch", Effect.AUX_CALL, defines=("unused",), rng_scope="auxiliary",
                   args={"fields": {"unused": {"type": "Truth"}}})
        candidate = ProgramCandidate("", (aux,), {}, {}, (), (), "LOCAL_WITNESS").seal()
        result = execute(candidate, {}, resolve_source=lambda *args: None, call_native=lambda *args: None,
                         action_is_bound=lambda *args: True,
                         semantic_batch=lambda *args: self.fail("Dead semantic request was issued"),
                         budget_available=lambda: True)
        self.assertEqual(result.semantic_batches, 0)


if __name__ == "__main__":
    unittest.main()
