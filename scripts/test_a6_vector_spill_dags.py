import unittest
from types import SimpleNamespace

from plot_a6_vector_spill_dags import dataflow_edges, memory_raw, operation_layout, with_reference_membars


def node(seq, kind, start, end):
    return {"seq": seq, "class": kind, "memory_ranges": [dict(
        resolved=True, base_object_id="ub", byte_start=start, byte_end=end)]}


class MemoryDagTests(unittest.TestCase):
    def test_display_omits_storage_reuse_edges_without_changing_audit(self):
        kinds = ("register", "predicate", "state", "ub_raw", "ub_war", "ub_waw")
        edges = [dict(src=i, dst=i+1, kind=kind) for i, kind in enumerate(kinds)]
        displayed = dataflow_edges(edges)
        self.assertEqual([e["kind"] for e in displayed], list(kinds[:4]))
        self.assertEqual([e["kind"] for e in edges], list(kinds))

    def test_only_latest_writer_of_each_byte_is_used(self):
        nodes = [node(0, "STORE", 0, 256), node(1, "STORE", 0, 256), node(2, "LOAD", 0, 256)]
        self.assertEqual(memory_raw(nodes), [dict(src=1, dst=2, kind="ub_raw")])

    def test_partial_write_keeps_both_producers(self):
        nodes = [node(0, "STORE", 0, 256), node(1, "STORE", 0, 4), node(2, "LOAD", 0, 256)]
        self.assertEqual({e["src"] for e in memory_raw(nodes)}, {0, 1})

    def test_adjacent_ranges_are_independent(self):
        self.assertEqual(memory_raw([node(0, "STORE", 0, 256), node(1, "LOAD", 256, 512)]), [])

    def test_other_buffer_is_independent(self):
        nodes = [node(0, "STORE", 0, 256), node(1, "LOAD", 0, 256)]
        nodes[1]["memory_ranges"][0]["base_object_id"] = "other"
        self.assertEqual(memory_raw(nodes), [])

    def test_historical_layout_uses_operation_lanes(self):
        nodes = [dict(seq=0, op="VLDS", category="load"), dict(seq=1, op="VMUL", category="alu")]
        shown, lanes, mapping = operation_layout(dict(variant="local", case="history-adam_apply_one-u8"), nodes, [], {n["seq"]: n for n in nodes})
        self.assertEqual([n["seq"] for n in shown], [0, 1])
        self.assertEqual([lane["key"] for lane in lanes], ["load", "VMUL"])
        self.assertEqual(mapping[1], "VMUL")

    def test_bank_aware_layout_keeps_two_rows_separate(self):
        nodes = [dict(seq=0, op="VLDS", category="load", memory_ranges=[dict(byte_start=0)]),
                 dict(seq=1, op="VLDS", category="load", memory_ranges=[dict(byte_start=3072)]),
                 dict(seq=2, op="VMUL", category="alu"), dict(seq=3, op="VMUL", category="alu")]
        edges = [dict(src=0, dst=2, kind="register"), dict(src=1, dst=3, kind="register")]
        _, _, mapping = operation_layout(dict(variant="software", case="history-flash_attention_grad_front-u2_concentrated_spill"), nodes, edges, {n["seq"]: n for n in nodes})
        self.assertEqual(mapping[2], "VMUL.0")
        self.assertEqual(mapping[3], "VMUL.1")

    def test_logit_layout_separates_predicate_spills_and_computation(self):
        nodes = [dict(seq=0, op="VCMP_LT", category="alu", writes_predicate=True),
                 dict(seq=1, op="PSTS", category="save"),
                 dict(seq=2, op="PLDS", category="reload", writes_predicate=True),
                 dict(seq=3, op="VSEL", category="alu"),
                 dict(seq=4, op="Membar", category="barrier", reference_only=True)]
        for item in nodes:
            item["iteration_path"] = [dict(loop_id="loop", iteration=0)]
        nodes.insert(0, dict(seq=-1, op="VDUPS", category="alu", iteration_path=[]))
        shown, lanes, mapping = operation_layout(dict(variant="local", case="logit-u4"), nodes, [],
                                                 {n["seq"]: n for n in nodes})
        self.assertEqual(len(shown), 5)
        self.assertNotIn(-1, mapping)
        self.assertEqual(len(nodes), 6)
        self.assertEqual([mapping[i] for i in range(5)],
                         ["predicate", "predicate_store", "predicate_load", "compute", "barrier"])
        self.assertEqual([lane["key"] for lane in lanes],
                         ["predicate", "predicate_store", "predicate_load", "compute", "barrier"])
        u3_shown, _, u3_mapping = operation_layout(dict(variant="local", case="logit-u3"), nodes, [],
                                                  {n["seq"]: n for n in nodes})
        self.assertEqual(u3_shown, shown)
        self.assertEqual(u3_mapping, mapping)

    def test_logit_u3_omits_separate_tail_loop_from_first_group_plot(self):
        nodes = [dict(seq=0, op="VLDS", category="load",
                      iteration_path=[dict(loop_id="main", iteration=0)]),
                 dict(seq=1, op="VLDS", category="load",
                      iteration_path=[dict(loop_id="tail", iteration=0)])]
        shown, _, mapping = operation_layout(dict(variant="software", case="logit-u3"), nodes, [],
                                             {n["seq"]: n for n in nodes})
        self.assertEqual([n["seq"] for n in shown], [0])
        self.assertNotIn(1, mapping)

    def test_logit_u8_keeps_both_spill_banks_and_omits_initialization(self):
        nodes = [dict(seq=0, op="VDUPS", category="alu", iteration_path=[]),
                 dict(seq=1, op="VSTS", category="save"),
                 dict(seq=2, op="VLDS", category="reload"),
                 dict(seq=3, op="PSTS", category="save"),
                 dict(seq=4, op="PLDS", category="reload", writes_predicate=True)]
        for item in nodes[1:]:
            item["iteration_path"] = [dict(loop_id="loop", iteration=0)]
        shown, _, mapping = operation_layout(dict(variant="local", case="logit-u8"), nodes, [],
                                             {n["seq"]: n for n in nodes})
        self.assertEqual([n["seq"] for n in shown], [1, 2, 3, 4])
        self.assertEqual([mapping[i] for i in range(1, 5)],
                         ["save", "reload", "predicate_store", "predicate_load"])

    def test_original_membar_is_display_only_and_in_original_position(self):
        nodes = [dict(seq=0, op="VLDS", static_id="a", iteration_path=[], **{"class": "LOAD"}),
                 dict(seq=1, op="VSTS", static_id="b", iteration_path=[], **{"class": "STORE"})]
        edges = [dict(src=0, dst=1, kind="register")]
        instructions = [SimpleNamespace(type="inst", op="VLDS", static_instruction_id="a", iteration_path=[]),
                        SimpleNamespace(type="membar", barrier="VLD_VST", stream_seq=1, static_instruction_id="bar", iteration_path=[]),
                        SimpleNamespace(type="inst", op="VSTS", static_instruction_id="b", iteration_path=[])]
        result = with_reference_membars((dict(variant="local", vf_cycles=123), nodes, edges,
                                        {n["seq"]: n for n in nodes}, "test"), instructions)
        self.assertEqual([n["op"] for n in result[1]], ["VLDS", "Membar", "VSTS"])
        marker = result[1][1]
        self.assertTrue(marker["reference_only"])
        self.assertFalse(marker["scheduled"])
        self.assertEqual(marker["timing_cost"], 0)
        self.assertIs(result[2], edges)
        self.assertEqual(result[0]["vf_cycles"], 123)
        self.assertEqual(len(nodes), 2)
        with self.assertRaises(ValueError):
            with_reference_membars((dict(variant="local"), nodes, edges,
                                    {n["seq"]: n for n in nodes}, "test"), list(reversed(instructions)))


if __name__ == "__main__":
    unittest.main()
