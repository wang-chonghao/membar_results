"""Order-preserving A6 spill DAGs audited against the exact report inputs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from dataclasses import replace

from search_a6_spill_schedules import read, save, setup_model


CASES = {
    "adam_u8": ("history-adam_apply_one-u8", "AdamApplyOne U8"),
    "flash_u2": ("history-flash_attention_grad_front-u2_concentrated_spill", "FlashAttention Grad U2"),
    "logit_u3": ("logit-u3", "Logit Grad U3"),
    "logit_u4": ("logit-u4", "Logit Grad U4"),
    "logit_u8": ("logit-u8", "Logit Grad U8"),
}
META = "metadata/a6_vector_spill_dags_20261027"
CASE_META = {"logit-u3": "metadata/a6_logit_u3_dags_20261027",
             "logit-u4": "metadata/a6_logit_predicate_dags_20261027",
             "logit-u8": "metadata/a6_logit_u8_dags_20261027"}
COLORS = dict(load="#facc15", reload="#facc15", save="#22c55e", output="#22c55e",
              alu="#60a5fa", special="#60a5fa", barrier="#f87171")


def key(instance):
    return instance["definition_id"], json.dumps(instance["iteration_path"], sort_keys=True)


def overlaps(a, b):
    return (a["resolved"] and b["resolved"] and a["base_object_id"] == b["base_object_id"]
            and max(a["byte_start"], b["byte_start"]) < min(a["byte_end"], b["byte_end"]))


def memory_raw(nodes):
    """Nearest preceding writers of each byte; never infer memory from names."""
    stores, edges = [], []
    for node in nodes:
        if node["class"] == "STORE":
            stores.append(node)
        elif node["class"] == "LOAD":
            for access in node["memory_ranges"]:
                if not access["resolved"]:
                    raise ValueError("Unresolved address in diagram")
                remaining = [(access["byte_start"], access["byte_end"])]
                for prior in reversed(stores):
                    for written in prior["memory_ranges"]:
                        if written["base_object_id"] != access["base_object_id"]:
                            continue
                        residual, hit = [], False
                        for left, right in remaining:
                            lo, hi = max(left, written["byte_start"]), min(right, written["byte_end"])
                            if lo < hi:
                                hit = True
                                if left < lo:
                                    residual.append((left, lo))
                                if hi < right:
                                    residual.append((hi, right))
                            else:
                                residual.append((left, right))
                        remaining = residual
                        if hit:
                            edges.append(dict(src=prior["seq"], dst=node["seq"], kind="ub_raw"))
                    if not remaining:
                        break
    return list({(e["src"], e["dst"], e["kind"]): e for e in edges}.values())


def prepare_logit_u8_input(root):
    """Graph-only logged replay; do not publish scalar-excluded report timings."""
    from api.json.adapter import CanonicalJsonVfInfoAdapter
    from api.json.serialization import canonical_vf_info_to_dict
    from multisoc_log_replay import Replay, expanded_instructions, operation, vf_streams
    from predict_logged_vector_replays import audit_scalar_operands, freeze_addresses, scalar_spill
    from refresh_multisoc_predictions import migrate_payload

    case = root / "spill_candidate_alignment/pure_cce/logit-u8"
    stream, = vf_streams(case)
    _, _, address, words, events = stream
    scalar = [e for e in events if scalar_spill(e)]
    selected = [e for e in events if not scalar_spill(e)]
    constants = audit_scalar_operands(selected)
    log = case / "cce/core0.veccore0.instr_popped_log.dump"
    raw = Replay("logit-u8", 0, log).build(address, words, selected)
    frozen, addresses = freeze_addresses(raw, selected)
    payload, migrations = migrate_payload(canonical_vf_info_to_dict(frozen))
    vf = CanonicalJsonVfInfoAdapter.from_payload(payload)
    expected = [e.op for e in selected if operation(e) is not None]
    actual = [n.attributes["rv_opcode"] if hasattr(n, "attributes") else "RV_SMEM_BAR"
              for n, _ in expanded_instructions(vf.context)]
    if actual != expected:
        raise ValueError("Logit U8 replay changed the dynamic opcode order")
    folder = root / CASE_META["logit-u8"]
    save(folder / "global_input.json", payload)
    save(folder / "input_audit.json", dict(scalar_timing_modeled=False,
         scalar_spill_opcodes_excluded=dict(Counter(e.op for e in scalar)),
         dynamic_opcode_order_verified=True, every_dynamic_address_verified=True,
         frozen_compute_scalar_operands=constants, migrations=migrations,
         source=log.relative_to(root).as_posix(), source_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
         dynamic_addresses=addresses, report_tables_modified=False))


def build(root, model, case, variant, db):
    from api.json.adapter import CanonicalJsonVfInfoAdapter
    from api.json.serialization import canonical_vf_info_to_dict
    from api.ub_dependency_experiment import remove_directional_membars
    from core.execution.prepare import prepare_program
    from core.execution.engine import Engine
    from core.params.uarch import resolve_uarch
    from refresh_multisoc_predictions import memory_metadata

    ident, title = CASES[case]
    rows = read(root / "metadata/spill_performance_summary.json")["rows"]
    summary = rows[ident]["A6"] if ident in rows else {
        "local_cycles": None,
        "software_cycles": read(root / "a6_compiler_validation_20261027" / ident / "result.json")["best"]["global_cycles"],
    }
    removed = 0
    if variant == "local":
        source = root / "logged_vector_replay_20261026" / ident / "A6/vf_000/global_input.json"
        if not source.exists():
            source = root / "multisoc_refresh_20261025" / ident / "original/A6/vf_000/global_input.json"
        if case == "logit_u8":
            source = root / CASE_META[ident] / "global_input.json"
        vf = CanonicalJsonVfInfoAdapter.from_payload(read(source))
        vf, removed = remove_directional_membars(vf)
        expected, mode = summary["local_cycles"], "range_overlap"
    else:
        selected = read(root / "a6_compiler_validation_20261027" / ident / "result.json")["best"]
        source = (root / selected["evidence"]).parent / "selected_vf.json"
        if case == "logit_u8":
            if hashlib.sha256(source.with_name("kernel.cce").read_bytes()).hexdigest() != selected["source_sha256"]:
                raise ValueError("A6-verified U8 software source changed")
            source = root / "multisoc_refresh_20261025/logit-u8/software/A6/vf_000/global_input.json"
        vf = CanonicalJsonVfInfoAdapter.from_payload(read(source))
        expected, mode = summary["software_cycles"], "disabled"
    uarch = resolve_uarch(db.base_uarch, vf.uarch)
    prepared = prepare_program(vf, db, uarch, ub_metadata=memory_metadata(vf))
    engine = Engine(prepared, uarch, db, ub_dependency_mode=mode)
    result = engine.run(None)
    if expected is not None and result.vf_end_cycle != expected:
        raise ValueError(f"Prediction drift: {case}/{variant}: {result.vf_end_cycle} != {expected}")
    starts = {n["stream_seq"]: n for n in engine.trace.cyc_start_log}
    done = {n["stream_seq"]: n for n in engine.trace.cyc_done_log}
    if variant == "local" and case != "logit_u8":
        archived = source.parent / "local/start_by_cycle.json.gz"
        with gzip.open(archived, "rt") as stream:
            old = [json.loads(line) for line in stream if line.strip()]
        assert [(n["stream_seq"], n["cy"], n["op"]) for n in old] == [(n["stream_seq"], n["cy"], n["op"]) for n in engine.trace.cyc_start_log]
    nodes, edges, writers, states = [], [], {}, defaultdict(list)
    for inst in prepared.instructions:
        seq = inst.stream_seq
        if inst.type == "membar":
            nodes.append(dict(seq=seq, op="Membar", category="barrier", **{"class": "CONTROL"},
                              iteration_path=inst.iteration_path, barrier=inst.barrier,
                              memory_ranges=[], static_id=inst.static_instruction_id))
            continue
        event = starts[seq]
        node = dict(seq=seq, inst_id=inst.inst_id, static_id=inst.static_instruction_id,
                    op=inst.op, form=inst.form, **{"class": inst.op_class},
                    start=event["cy"], done=done[seq]["cy"], ready=event["ready_cycle"],
                    iteration_path=inst.iteration_path, memory_ranges=event.get("memory_ranges", []),
                    attributes=inst.attributes, exu=event["exu_port"],
                    writes_predicate=any(prepared.values[v["definition_id"]]["storage"] == "PredicateRegister"
                                         for v in inst.dst_value_instances))
        for operand in inst.src_value_instances:
            storage = prepared.values[operand["definition_id"]]["storage"]
            if storage not in ("Register", "PredicateRegister"):
                continue
            identity = key(operand)
            if identity not in writers:
                raise ValueError(f"Unbound model value {identity}")
            edges.append(dict(src=writers[identity], dst=seq, kind="predicate" if storage == "PredicateRegister" else "register"))
        for operand in inst.dst_value_instances:
            storage = prepared.values[operand["definition_id"]]["storage"]
            if storage in ("Register", "PredicateRegister"):
                writers[key(operand)] = seq
        state = inst.attributes.get("align_state_id")
        operation = inst.attributes.get("align_state_operation")
        if state and operation == "append":
            states[state].append(seq)
        elif state and operation == "consume":
            edges.extend(dict(src=before, dst=seq, kind="state") for before in states.pop(state, []))
        nodes.append(node)
    memory = memory_raw(nodes)
    edges += memory
    temporary_writes = {e["src"] for e in memory}
    temporary_reads = {e["dst"] for e in memory}
    by_seq = {n["seq"]: n for n in nodes}
    by_inst = {n["inst_id"]: n for n in nodes if "inst_id" in n}
    actual_blocks = set()
    if engine.lsu.ub_dependency is not None:
        for src, dst in engine.lsu.ub_dependency.edges:
            a, b = by_inst[src]["seq"], by_inst[dst]["seq"]
            actual_blocks.add((a, b))
            if not any(e["src"] == a and e["dst"] == b and e["kind"] == "ub_raw" for e in edges):
                edges.append(dict(src=a, dst=b, kind="ub_war" if by_inst[src]["class"] == "LOAD" else "ub_raw"))
    for edge in edges:
        edge["observed_memory_block"] = (edge["src"], edge["dst"]) in actual_blocks and edge["kind"].startswith("ub_")
        if edge["src"] >= edge["dst"]:
            raise ValueError("Backward DAG edge")
    for node in nodes:
        if node["class"] == "COMPUTE":
            node["category"] = "special" if node["exu"] == 2 else "alu"
        elif node["class"] in ("LOAD", "STORE"):
            node["compiler_spill"] = variant == "local" and (node["op"] in ("PSTS", "PLDS") or
                any(a["byte_start"] >= 192*1024 for a in node["memory_ranges"]))
            if node["class"] == "LOAD":
                node["category"] = "reload" if node["compiler_spill"] or node["seq"] in temporary_reads else "load"
            else:
                node["category"] = "save" if node["compiler_spill"] or node["seq"] in temporary_writes else "output"
    # Show the first dynamic U-group in every loop, keeping all prefix/suffix nodes.
    selected = [n for n in nodes if all(level["iteration"] == 0 for level in n["iteration_path"])]
    chosen = {n["seq"] for n in selected}
    for index, node in enumerate(selected):
        node["view_index"] = index
    visible_edges = [dict(e) for e in edges if e["dst"] in chosen]
    # Every drawn RAW reload must be tied to an earlier store in the full stream.
    chains = []
    consumers = defaultdict(list)
    for edge in edges:
        if edge["kind"] in ("register", "predicate"):
            consumers[edge["src"]].append(by_seq[edge["dst"]])
    for edge in memory:
        producer, reload = by_seq[edge["src"]], by_seq[edge["dst"]]
        if not reload.get("compiler_spill"):
            continue
        useful = [n for n in consumers[reload["seq"]] if n["class"] == "COMPUTE"]
        first = min(useful, key=lambda n: n["start"]) if useful else None
        chains.append(dict(store_seq=producer["seq"], store_start=producer["start"], store_done=producer["done"],
                           reload_seq=reload["seq"], reload_start=reload["start"],
                           consumer_seq=first["seq"] if first else None, consumer_start=first["start"] if first else None,
                           reload_wait_after_store_done=reload["start"]-producer["done"],
                           observed_block=edge["observed_memory_block"]))
    hist = engine.trace.history
    blocked = engine.idu.resource_block_log
    # This is a diagnostic only, never a replacement for the report timings.
    relaxed = replace(uarch, vreg_num=4096)
    if sum(len(i.dst_value_instances) for i in prepared.instructions) >= 4096:
        raise ValueError("Diagnostic vector pool is not an effective upper bound")
    diagnostic = Engine(prepared, relaxed, db, ub_dependency_mode=mode).run(None)
    counts = Counter(n["op"] for n in nodes if n["class"] != "CONTROL")
    reasons = {reason: len({n["cy"] for n in blocked if n["blocked_reason"] == reason})
               for reason in {n["blocked_reason"] for n in blocked}}
    compute = [n for n in nodes if n["class"] == "COMPUTE"]
    lsu = [n for n in nodes if n["class"] in ("LOAD", "STORE")]
    stats = dict(case=ident, title=title, variant=variant, soc="A6 / DV121", vf_cycles=result.vf_end_cycle,
        model_commit="336844700d704bc3afdc4b190b215e736b65ebc9", instruction_count=len(starts),
        op_counts=dict(counts), loads=sum(n["class"] == "LOAD" for n in nodes), stores=sum(n["class"] == "STORE" for n in nodes),
        compute=len(compute), average_compute_ipc=len(compute)/result.vf_end_cycle,
        compute_by_exu=dict(Counter(str(n["exu"]) for n in compute)),
        compiler_spill_loads=sum(n.get("compiler_spill", False) and n["class"] == "LOAD" for n in nodes),
        compiler_spill_stores=sum(n.get("compiler_spill", False) and n["class"] == "STORE" for n in nodes),
        temporary_reads=len(temporary_reads), temporary_writes=len(temporary_writes),
        directional_membars_removed=removed, global_membars_remaining=sum(n["category"] == "barrier" for n in nodes),
        idu_resource_blocked_unique_cycles=reasons, idu_address_blocked_unique_cycles=len({n["cy"] for n in engine.idu.address_block_log}),
        min_vector_free=min((n["vector_phys_free"] for n in hist), default=None),
        last_dispatch_cycle=max(n["cy"] for n in engine.idu.dispatch_log),
        memory_ordering_stats=engine.lsu.ub_dependency.stats() if engine.lsu.ub_dependency else None,
        ub_blocked_unique_cycles=len({n["cy"] for n in hist if n["event"] == "blocked" and n["blocked_reason"] == "ub_address_dependency"}),
        diagnostic_vector4096_cycles=diagnostic.vf_end_cycle,
        shown_nodes=len(selected), omitted_nodes=len(nodes)-len(selected),
        scalar_timing_modeled=False, data_register_raws_are_dynamic_ssa=True,
        prediction_matches_archived_cycles=expected is not None,
        source=source.relative_to(root).as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    audit_folder = root / CASE_META.get(ident, META)
    audit_folder.mkdir(parents=True, exist_ok=True)
    stem = f"a6_{case}_{variant}_instruction_order_20261027"
    save(audit_folder / (stem + ".json"), dict(stats=stats, nodes=nodes, edges=edges, selected_sequences=sorted(chosen),
          spill_reload_chains=chains, idu_resource_block_log=blocked,
          idu_address_block_log=engine.idu.address_block_log,
          actual_ub_block_events=[n for n in hist if n["event"] == "blocked" and n["blocked_reason"] == "ub_address_dependency"]))
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats, selected, visible_edges, by_seq, stem


def operation_layout(stats, nodes, edges, by_seq):
    """Match the archived single-axis order plots, with row lanes for Bank-aware U2."""
    shown = [n for n in nodes if not n["op"].startswith("PSET") and n["op"] != "VSTAS"]
    if stats["case"].startswith("logit-"):
        # The historical A5 Logit plots show only the first loop-body group.
        shown = [n for n in shown if n["iteration_path"]]
        first_loop = next(n["iteration_path"][0]["loop_id"] for n in shown)
        shown = [n for n in shown if n["iteration_path"][0]["loop_id"] == first_loop]
    parents = defaultdict(list)
    for edge in edges:
        if edge["kind"] in ("register", "ub_raw"):
            parents[edge["dst"]].append(edge["src"])
    split_rows = stats["variant"] == "software" and stats["case"].startswith("history-flash_")
    support = {}
    if split_rows:
        # Recorded FlashAttention buffers: two 8 x 3072-byte FP32 row arrays.
        for seq in sorted(by_seq):
            node = by_seq[seq]
            if node["category"] == "load":
                address = node["memory_ranges"][0]["byte_start"]
                origin = 0x8000 if address >= 0x8000 else 0
                support[seq] = {(address-origin)//3072}
            else:
                support[seq] = set().union(*(support.get(p, set()) for p in parents[seq]))
    lane_by_seq, definitions = {}, {}
    operations = ("VMUL", "VADD", "VSQRT", "VDIV", "VSUB", "VCGADDV2", "VCADDV2")
    for node in shown:
        category, op = node["category"], node["op"]
        if stats["case"].startswith("logit-"):
            if op == "PSTS":
                lane, label, rank = "predicate_store", "Predicate spill PSTS", 2
            elif op == "PLDS":
                lane, label, rank = "predicate_load", "Predicate reload PLDS", 3
            elif node.get("writes_predicate"):
                lane, label, rank = "predicate", "谓词生成 / 计算", 1
            elif category == "load":
                lane, label, rank = "load", "输入 VLDS", 0
            elif category == "save":
                lane, label, rank = "save", "Vector spill VSTS", 0.3
            elif category == "reload":
                lane, label, rank = "reload", "Vector reload VLDS", 0.6
            elif category == "output":
                lane, label, rank = "output", "输出 VSTS", 5
            elif category == "barrier":
                lane, label, rank = "barrier", "Membar", 6
            elif category in ("alu", "special"):
                lane, label, rank = "compute", "向量计算 / VSEL", 4
            else:
                raise ValueError(f"Unexpected Logit operation: {op}/{category}")
        elif category == "load":
            lane, label, rank = "load", "输入 VLDS", 0
        elif category == "save":
            lane, label, rank = "save", "Spill VSTS" if stats["variant"] == "local" else "UB 暂存 VSTS", 1
        elif category == "reload":
            lane, label, rank = "reload", "Reload VLDS" if stats["variant"] == "local" else "UB 暂存 VLDS", 2
        elif category == "barrier":
            lane, label, rank = "barrier", "Membar", 30
        elif category == "output":
            lane, label, rank = "output", "输出 " + op, 20
        else:
            if op not in operations:
                raise ValueError(f"Missing operation lane: {op}")
            lane, label, rank = op, op, 3+operations.index(op)
        if split_rows:
            if len(support[node["seq"]]) != 1:
                raise ValueError(f"Ambiguous data-row ownership: {node['seq']}")
            row, = support[node["seq"]]
            if row not in (0, 1):
                raise ValueError(f"Expected first two rows, got {row}")
            lane, label, rank = f"{lane}.{row}", f"{label} · 行 {row}", rank*2+row
        lane_by_seq[node["seq"]] = lane
        definitions[lane] = dict(key=lane, label=label, rank=rank)
    lanes = sorted(definitions.values(), key=lambda lane: lane["rank"])
    return shown, lanes, lane_by_seq


def with_reference_membars(data, original_instructions):
    """Restore original barrier positions for display, never scheduling edges."""
    stats, selected, edges, by_seq, stem = data
    if stats["variant"] != "local":
        return data

    def identity(op, static_id, path):
        return op, static_id, json.dumps(path, sort_keys=True)

    actual = [n for _, n in sorted(by_seq.items()) if n["class"] != "CONTROL"]
    original_actual = [i for i in original_instructions if i.type == "inst"]
    if [identity(n["op"], n["static_id"], n["iteration_path"]) for n in actual] != [
            identity(i.op, i.static_instruction_id, i.iteration_path) for i in original_actual]:
        raise ValueError("Original/local dynamic streams differ; cannot restore barrier positions")
    lookup = {identity(n["op"], n["static_id"], n["iteration_path"]): n for n in actual}
    chosen = {n["seq"] for n in selected}
    shown, references = [], []
    for inst in original_instructions:
        if inst.type == "inst":
            node = lookup[identity(inst.op, inst.static_instruction_id, inst.iteration_path)]
            if node["seq"] in chosen:
                shown.append(node)
        elif inst.type == "membar":
            if inst.barrier not in ("VLD_VST", "VST_VLD"):
                raise ValueError("Only replaced directional barriers may be reference markers")
            if all(level["iteration"] == 0 for level in inst.iteration_path):
                node = dict(seq=f"original_membar:{inst.stream_seq}", op="Membar", category="barrier",
                    **{"class": "CONTROL"}, iteration_path=inst.iteration_path,
                    barrier=inst.barrier, original_stream_seq=inst.stream_seq,
                    static_id=inst.static_instruction_id, reference_only=True,
                    scheduled=False, timing_cost=0, memory_ranges=[])
                shown.append(node)
                references.append(node)
    augmented = dict(by_seq)
    augmented.update({n["seq"]: n for n in references})
    return dict(stats, reference_membars_shown=len(references)), shown, edges, augmented, stem


def dataflow_edges(edges):
    """Display value flow, not storage reuse ordering; keep the model audit intact."""
    return [edge for edge in edges if edge["kind"] in ("register", "predicate", "state", "ub_raw")]


def draw(root, data, font, replace_generated=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch
    stats, nodes, edges, by_seq, stem = data
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "axes.unicode_minus": False})
    shown, lanes, lane_by_seq = operation_layout(stats, nodes, edges, by_seq)
    ypos = {lane["key"]: len(lanes)-i for i, lane in enumerate(lanes)}
    positions = {n["seq"]: (i+1, ypos[lane_by_seq[n["seq"]]]) for i, n in enumerate(shown)}
    counts = Counter(lane_by_seq[n["seq"]] for n in shown)
    numbered = len(shown) <= 125
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.19, right=0.975, top=0.80, bottom=0.23)
    name = "Membar 优化" if stats["variant"] == "local" else "软件优化"
    fig.suptitle(f"{stats['title']} · A6 {name}指令顺序与数据依赖", fontsize=22, y=0.96)
    reference_count = sum(n.get("reference_only", False) for n in shown)
    subtitle = (f"展示首组原始顺序（含 {reference_count} 个原 Membar 标记） · 完整 VF {stats['vf_cycles']} cycles"
                if reference_count else f"展示 {len(shown)} 条指令 · 各循环保留首组 · 完整 VF {stats['vf_cycles']} cycles")
    if stats["case"] == "logit-u3" and not reference_count:
        subtitle = f"展示首组 U3 的 {len(shown)} 条指令 · 完整 VF {stats['vf_cycles']} cycles"
    fig.text(0.59, 0.875, subtitle, ha="center", fontsize=16)
    if stats["case"] == "logit-u8":
        fig.text(0.59, 0.83, "仅预测向量 / 谓词流水；标量 spill 时序未计入", ha="center", fontsize=12, color="#475569")
    for index, lane in enumerate(lanes):
        y = ypos[lane["key"]]
        ax.axhspan(y-0.37, y+0.37, color="#f3f5f7" if index % 2 == 0 else "#e8edf3", zorder=0)
    drawn_edges = []
    for edge in dataflow_edges(edges):
        if edge["src"] not in positions or edge["dst"] not in positions:
            continue
        a, b = positions[edge["src"]], positions[edge["dst"]]
        memory = edge["kind"].startswith("ub_")
        color = "#15803d" if memory else "#9333ea" if edge["kind"] == "predicate" else "#64748b"
        radius = -min(0.17, 2.5/max(1, b[0]-a[0])) if a[1] == b[1] else 0.035
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={radius}",
            arrowstyle="-|>", mutation_scale=7 if numbered else 5,
            shrinkA=4 if numbered else 2, shrinkB=4 if numbered else 2,
            linewidth=0.9 if numbered else 0.65, color=color,
            linestyle="--" if memory else "-", alpha=0.65 if numbered else 0.45, zorder=1))
        drawn_edges.append(edge)
    for node in shown:
        x, y = positions[node["seq"]]
        marker_size = 70 if numbered else 10 if len(shown) > 220 else 15
        if node["category"] == "barrier" and not numbered:
            marker_size = 28
        ax.scatter([x], [y], s=marker_size,
                   color="#fca5a5" if node.get("reference_only") else COLORS[node["category"]],
                   edgecolors="#475569", linewidths=0.5 if not numbered else 0.7,
                   marker="D" if node["category"] == "barrier" else "o", zorder=3)
        if numbered:
            ax.text(x, y, str(int(x)), fontsize=5.8, ha="center", va="center", zorder=4)
    # A phase boundary is not a new dependency: the repeated groups are merely hidden.
    for index, (before, after) in enumerate(zip(shown, shown[1:]), 1):
        if isinstance(after["seq"], int) and isinstance(before["seq"], int) and after["seq"]-before["seq"] > 20:
            x = index+0.5
            ax.axvline(x, color="#94a3b8", linewidth=0.8, linestyle=":", alpha=0.8, zorder=0)
            ax.text(x, len(lanes)+0.36, "省略重复轮次", fontsize=9, ha="center", color="#64748b")
    ax.set_yticks([ypos[lane["key"]] for lane in lanes],
                 [f"{lane['label']} · {counts[lane['key']]}" for lane in lanes], fontsize=12)
    ax.tick_params(axis="y", length=0, pad=9)
    step = 10 if numbered else 20
    ticks = [i for i in range(step, len(shown), step) if i-1 >= step*0.35 and len(shown)-i >= step*0.45]
    ax.set_xticks([1, *ticks, len(shown)])
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(0, len(shown)+1)
    ax.set_ylim(0.35, len(lanes)+0.68)
    ax.set_xlabel("首组原始指令顺序编号（保留 Membar 位置，不是执行 cycle）" if reference_count else
                  "所展示指令组的顺序编号（不是执行 cycle）", fontsize=15, labelpad=14)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    footer = "黄色为加载，蓝色为计算，绿色为写出；灰色曲线为寄存器 RAW，绿色虚线为 UB 数据传递（RAW）。"
    if stats["case"].startswith("logit-"):
        footer = "黄色为加载，蓝色为计算，绿色为写出；灰色为向量 RAW，紫色为谓词 RAW，绿色虚线为 UB 数据传递（RAW）。"
    fig.text(0.59, 0.105, footer, ha="center", fontsize=11, color="#475569")
    detail = "共同 PSET 掩码与最终 VSTAS 不展开；标量时序未建模。完整节点、编号映射和依赖保存在配套数据中。"
    if stats["case"] == "history-adam_apply_one-u8":
        detail = ("阶段间主动 UB 暂存不是编译器自动 spill；红菱形为保留的全局 Membar。" if stats["variant"] == "software" else
                  "已移除方向性 Membar，保留真实 spill 写出与重载；循环外初始化也包含在图中。")
    elif stats["case"] == "logit-u3":
        detail = "软件方案为 U3 内 2+1 分组重排，A6 编译确认无 spill；完整预测包含 5 组 U3 和 1 份 tail。"
    elif stats["case"] == "logit-u4":
        detail = "软件方案为同一 U4 内 2+2 分组重排，A6 编译确认无 spill；共同 PSET 扇出省略，标量时序未建模。"
    elif stats["case"] == "logit-u8":
        detail = "软件方案为同一 U8 内 2+2+2+2 分组重排，A6 编译确认无 spill；图中省略标量指令。"
    if reference_count:
        detail = "红菱形保留原 Membar 位置：优化后已由局部 UB 依赖替代，不参与调度、不计时，也不添加全局同步边。"
    if stats["case"].startswith("logit-"):
        detail += " 循环外常量初始化已省略。"
    if stats["case"] == "logit-u3":
        fig.text(0.59, 0.83, "仅展示首组 U3 循环体；完整 VF 时序包含 5 组 U3 + 1 份 tail", ha="center", fontsize=12, color="#475569")
    fig.text(0.59, 0.065, detail, ha="center", fontsize=10, color="#475569")
    output = root / "figure/DAG" / (stem + ".png")
    if output.exists() and not replace_generated:
        raise FileExistsError(f"Refuse to overwrite {output}")
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    save(root / CASE_META.get(stats["case"], META) / (stem + ".view.json"), dict(style="historical_single_axis_operation_swimlanes",
         source_sequences=[n["seq"] for n in shown],
         nodes=[dict(order=i+1, stream_seq=None if n.get("reference_only") else n["seq"],
                     op=n["op"], lane=lane_by_seq[n["seq"]], reference_only=n.get("reference_only", False),
                     original_stream_seq=n.get("original_stream_seq"), barrier=n.get("barrier")) for i, n in enumerate(shown)],
         reference_membars_are_execution_nodes=False, edge_policy="dataflow_only_no_war_or_waw",
         edges=drawn_edges, cycles=stats["vf_cycles"], layout_only=True))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--replace-generated", action="store_true", help="Replace only the selected dated output images")
    parser.add_argument("--render-only", action="store_true", help="Reuse the saved DAG and timing audit")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=["adam_u8", "flash_u2"])
    args = parser.parse_args()
    root, model = args.root.absolute(), args.model_root.absolute()
    provenance = read(root / "a6_software_search_20261025/provenance.json")
    for name, expected in provenance["model_config_hashes"].items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == expected, name
    db = setup_model(model)
    results = defaultdict(dict)
    for case in args.cases:
        metadata = CASE_META.get(CASES[case][0], META)
        if case == "logit_u8" and not args.render_only:
            prepare_logit_u8_input(root)
        for variant in ("local", "software"):
            if args.render_only:
                stem = f"a6_{case}_{variant}_instruction_order_20261027"
                stored = read(root / metadata / (stem + ".json"))
                by_seq = {n["seq"]: n for n in stored["nodes"]}
                selected = set(stored["selected_sequences"])
                data = (stored["stats"], [by_seq[seq] for seq in sorted(selected)],
                        [e for e in stored["edges"] if e["dst"] in selected], by_seq, stem)
            else:
                data = build(root, model, case, variant, db)
            if variant == "local":
                from api.json.adapter import CanonicalJsonVfInfoAdapter
                from core.execution.prepare import prepare_program
                from core.params.uarch import resolve_uarch
                original = CanonicalJsonVfInfoAdapter.from_payload(read(root / data[0]["source"]))
                original_program = prepare_program(original, db, resolve_uarch(db.base_uarch, original.uarch))
                data = with_reference_membars(data, original_program.instructions)
            path = draw(root, data, args.font, args.replace_generated)
            results[metadata][f"{case}_{variant}"] = dict(stats=data[0], figure=path.relative_to(root).as_posix())
    for metadata, entries in results.items():
        destination = root / metadata / "summary.json"
        previous = read(destination) if destination.exists() else {}
        save(destination, dict(previous, **entries))


if __name__ == "__main__":
    main()
