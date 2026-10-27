"""Replay archived hardware inputs on both SoCs without altering the model."""
import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def migrate_payload(payload):
    payload = copy.deepcopy(payload)
    changes = Counter()
    from api.common.instruction_catalog import DEFAULT_INSTRUCTION_CATALOG
    widths = {"fp32": 4, "b32": 4, "uint32": 4, "int32": 4, "fp16": 2, "bf16": 2,
              "b16": 2, "uint16": 2, "int16": 2, "uint8": 1, "int8": 1}

    def visit(nodes):
        for node in nodes:
            if "dependencies" in node:
                if node["dependencies"]:
                    raise ValueError("Nonempty explicit dependencies cannot be removed")
                del node["dependencies"]
                changes["removed_empty_dependencies"] += 1
            if node["kind"] == "loop":
                visit(node["body"])
            elif node["kind"] == "instruction" and node["opcode"] == "VDUP":
                if node["inputs"] and payload["values"][node["inputs"][0]["value_id"]]["storage"] == "Scalar":
                    node["opcode"] = "VDUPS"
                    changes["scalar_vdup_renamed_vdups"] += 1
            if node["kind"] == "instruction" and node["opcode"] == "PLT" and len(node["outputs"]) == 1:
                counter = payload["values"][node["inputs"][0]["value_id"]]
                if counter["storage"] != "Scalar" or counter.get("producer_node_id") is not None:
                    raise ValueError("Cannot reconstruct a PLT counter with a modeled producer")
                counter["dtype"] = "uint32"
                if node["inputs"][0].get("dtype") is not None:
                    node["inputs"][0]["dtype"] = "uint32"
                identity = node["instruction_id"] + ".remaining"
                if identity in payload["values"]:
                    raise ValueError("PLT remaining definition collision")
                payload["values"][identity] = dict(definition_id=identity, logical_id=counter["logical_id"],
                    storage="Scalar", dtype="uint32", shape=[], producer_node_id=node["instruction_id"],
                    storage_object_id=None, source_location=node.get("source_location"))
                node["outputs"].append(dict(value_id=identity, role="destination", dtype=None, memory_access=None))
                changes["legacy_plt_unused_scalar_result_restored"] += 1
            if node["kind"] == "instruction" and node.get("attributes", {}).get("catalog_mode"):
                base = DEFAULT_INSTRUCTION_CATALOG.lookup(node["opcode"])
                spec = base.memory_modes.get(node["attributes"]["catalog_mode"]) if base else None
                if spec is None:
                    continue
                expected = spec.memory_span_by_form.get(node["form"], spec.memory_span)
                for operand in (*node["inputs"], *node["outputs"]):
                    memory = operand.get("memory_access")
                    if memory is None or expected is None or memory.get("span") == expected:
                        continue
                    old_unit = memory.get("address_unit_bytes") or widths[payload["values"][operand["value_id"]]["dtype"]]
                    new_unit = spec.memory_address_unit_bytes or widths[payload["values"][operand["value_id"]]["dtype"]]
                    if memory["span"] * old_unit != expected * new_unit:
                        raise ValueError("Mode span migration would change the accessed byte count")
                    expression = memory["offset"]
                    values = [expression["constant"], *[term["coefficient"] for term in expression["terms"]]]
                    if any(value * old_unit % new_unit for value in values):
                        raise ValueError("Mode offset cannot be represented without changing the byte address")
                    expression["constant"] = expression["constant"] * old_unit // new_unit
                    for term in expression["terms"]:
                        term["coefficient"] = term["coefficient"] * old_unit // new_unit
                    memory.update(span=expected, address_unit_bytes=new_unit)
                    changes["memory_units_normalized_without_byte_changes"] += 1
    visit(payload["context"])
    return payload, dict(changes)


def find_inputs(root, row, role, output):
    from api.json.adapter import CanonicalJsonVfInfoAdapter
    from api.common.schema import CanonicalLoop
    from multisoc_log_replay import Replay, add_memory_offsets, vf_streams, expanded_count
    categories = ("comparison",) if role == "original" else ("software",)
    links = [entry["path"] for category in categories for entry in row.get("files", {}).get(category, [])]
    candidates = [path for path in links if path.endswith(".json") and "canonical" in path
                  and "local" not in Path(path).name and "audit" not in Path(path).name]
    if role == "original" and row.get("origin") == "historical_membar":
        recent = root / "a6_multisoc_membar_20261022" / row["case"] / row["variant"] / "canonical/global.json"
        if recent.exists() and row["case"] in {"adam_apply_one", "gelu_poly", "swiglu_grad", "gelu_grad"}:
            candidates.insert(0, recent.relative_to(root).as_posix())
    def score(path):
        return (path.startswith("a6_multisoc_membar_20261022/"), "manual_json" in path, "/current/canonical/" in path,
                "/experiment/canonical/" in path, "global" in Path(path).name)
    groups = {}
    for path in candidates:
        try:
            data = read(root / path)
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and "context" in data and "values" in data:
            groups.setdefault(str(Path(path).parent), []).append(path)
    errors = []
    for directory, paths in sorted(groups.items(), key=lambda pair: score(pair[0] + "/"), reverse=True):
        vf_paths = sorted(path for path in paths if re.fullmatch(r"vf_\d+\.json", Path(path).name))
        paths = vf_paths or [max(paths, key=score)]
        try:
            inputs, migrations = [], []
            for path in paths:
                payload, changes = migrate_payload(read(root / path))
                inputs.append(CanonicalJsonVfInfoAdapter.from_payload(payload, source=path))
                migrations.append(changes)
            return inputs, dict(kind="archived_canonical", paths=paths, migrations=migrations)
        except Exception as error:
            errors.append(dict(path=directory, error=str(error)[:1300]))

    log_links = []
    for category in (("original_cce", "comparison") if role == "original" else ("software",)):
        log_links += [entry["path"] for entry in row.get("files", {}).get(category, [])
                      if entry["path"].endswith("instr_popped_log.dump")]
    log_links = list(dict.fromkeys(log_links))
    for popped in log_links:
        log = root / popped
        instr = log.with_name("core0.veccore0.instr_log.dump")
        if not instr.is_file():
            continue
        fixture = output / "decoder_input/cce"
        fixture.mkdir(parents=True, exist_ok=True)
        # Only these two log paths are needed by the existing VF boundary decoder.
        for source in (log, instr):
            dest = fixture / source.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            shutil.copyfile(source, dest)
        try:
            inputs = []
            raw_scalar = Counter()
            for index, (_, _, address, words, events) in enumerate(vf_streams(fixture.parent)):
                raw_scalar.update(e.op for e in events if e.op.startswith(("RV_SST", "RV_SLD")))
                if raw_scalar:
                    raise ValueError(f"Scalar spill timing is unsupported: {dict(raw_scalar)}")
                inputs.append(add_memory_offsets(Replay(row["id"], index, log).build(address, words, events), events))
            if not inputs:
                raise ValueError("No VF found in archived logs")
            # Apply the same public JSON contract at this input boundary.
            from api.json.serialization import canonical_vf_info_to_dict
            inputs = [CanonicalJsonVfInfoAdapter.from_payload(canonical_vf_info_to_dict(v)) for v in inputs]
            return inputs, dict(kind="decoded_hardware_log", paths=[popped, str(instr.relative_to(root))])
        except Exception as error:
            errors.append(dict(path=popped, error=str(error)[:1300]))
    raise ValueError(json.dumps(dict(message="No compatible, lossless archived hardware input", attempts=errors), ensure_ascii=False))


def memory_metadata(vf):
    from api.cce.ub_address import PythonUbAddressExperimentMetadata, UbStaticAccess
    from api.common.schema import AffineExpression, AffineTerm, CanonicalLoop, CanonicalInstruction
    accesses = {}

    def scale(expr, unit):
        return AffineExpression(expr.constant * unit, tuple(AffineTerm(t.variable_id, t.coefficient * unit) for t in expr.terms))

    def visit(nodes):
        for node in nodes:
            if isinstance(node, CanonicalLoop):
                visit(node.body)
            elif isinstance(node, CanonicalInstruction):
                items = []
                for operand in (*node.inputs, *node.outputs):
                    memory = operand.memory_access
                    if memory is None:
                        continue
                    dtype = vf.values[operand.value_id].dtype
                    unit = memory.address_unit_bytes or {"fp32": 4, "b32": 4, "int32": 4, "uint32": 4,
                                                        "fp16": 2, "bf16": 2, "b16": 2, "int16": 2,
                                                        "uint16": 2, "uint8": 1, "int8": 1}.get(dtype)
                    aggregate_consume = node.attributes.get("align_state_operation") == "consume"
                    if unit is None or memory.span is None and not aggregate_consume:
                        raise ValueError(f"Unresolved byte geometry: {node.instruction_id} {dtype}")
                    items.append(UbStaticAccess(memory.base_object_id, memory.address_state_id or node.instruction_id,
                                               AffineExpression(), scale(memory.offset, unit),
                                               memory.post_update_delta_bytes or AffineExpression(),
                                               memory.access_kind.value, memory.span * unit if memory.span is not None else None,
                                               node.attributes.get("catalog_mode")))
                if items:
                    accesses[node.instruction_id] = tuple(items)
    visit(vf.context)
    metadata = PythonUbAddressExperimentMetadata(accesses)
    metadata.validate(vf)
    return metadata


def instruction_limit(vf, expansions):
    from api.common.schema import CanonicalLoop
    from core.execution.lowering import lower_instructions

    def count(nodes):
        return sum(int(vf.params.get(n.count, n.count)) * count(n.body) if isinstance(n, CanonicalLoop) else 1 for n in nodes)

    target, _ = lower_instructions(vf, expansions)
    return max(20_000, count(target.context) + 1)


def run_one(vf, parameters, soc, metadata, folder, cycle_limit=100_000):
    from core.params.database import ParamDB
    from core.params.uarch import resolve_uarch
    from core.execution.prepare import prepare_program
    from core.execution.engine import Engine
    from api.common.validator import require_valid_canonical_vf_info
    from api.ub_dependency_experiment import remove_directional_membars, _dynamic_stream_identity
    from api.json.serialization import canonical_vf_info_to_dict

    require_valid_canonical_vf_info(vf)
    # A run safety limit is not a hardware resource. Raise it only to fit the
    # finite, already known archived workload; both architectures receive it.
    vf = replace(vf, params={**vf.params, "max_cycles": cycle_limit},
                 uarch={**vf.uarch, "canonical_dynamic_instruction_limit": instruction_limit(vf, parameters.instruction_expansions)})
    local, removed = remove_directional_membars(vf)
    save(folder / "global_input.json", canonical_vf_info_to_dict(vf))
    states, results = {}, {}
    for mode, program in (("global", vf), ("local", local)):
        if mode == "local" and metadata is None:
            continue
        db = ParamDB(base_dir=str(MODEL), parameters=parameters)
        uarch = resolve_uarch(db.base_uarch, program.uarch)
        prepared = prepare_program(program, db, uarch, ub_metadata=metadata if mode == "local" else None)
        class ProgressCheckedEngine(Engine):
            def backend_step(self):
                super().backend_step()
                progress = (len(self.trace.cyc_start_log), len(self.trace.cyc_done_log))
                if progress != getattr(self, "_audit_progress", None):
                    self._audit_progress, self._audit_progress_cycle = progress, self.cycle
                if self.cycle - self._audit_progress_cycle > 10_000:
                    raise RuntimeError(f"No instruction start/completion for 10000 cycles at cycle {self.cycle}; pending={len(self.active_uops)}; vector_free={len(self.registers.vector_bank.freelist)}; predicate_free={len(self.registers.predicate_bank.freelist)}")
        engine = ProgressCheckedEngine(prepared, uarch, db, ub_dependency_mode="range_overlap" if mode == "local" else "disabled")
        started = time.perf_counter()
        result = engine.run(None).to_dict()
        result.update(soc=soc, wall_seconds=time.perf_counter() - started,
                      soc_expansion_stats=prepared.soc_expansion_stats,
                      warning_count=len(db.get_warnings()), dynamic_instruction_count=len(engine.trace.cyc_start_log))
        if engine.lsu.ub_dependency is not None:
            result["memory_ordering_stats"] = engine.lsu.ub_dependency.stats()
        folder.joinpath(mode).mkdir(parents=True, exist_ok=True)
        save(folder / mode / "result.json", result)
        save(folder / mode / "warnings.json", db.get_warnings())
        for name, rows in (("start", engine.trace.cyc_start_log), ("done", engine.trace.cyc_done_log)):
            with gzip.open(folder / mode / f"{name}_by_cycle.json.gz", "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        states[mode] = engine.trace.cyc_start_log
        results[mode] = result
    if "local" in states and _dynamic_stream_identity(states["global"]) != _dynamic_stream_identity(states["local"]):
        raise ValueError("Global/local dynamic order differs")
    if "local" in results:
        done = {row["inst_id"]: row["cy"] for row in engine.trace.cyc_done_log}
        memory_rows = sorted((r for r in states["local"] if r.get("memory_ranges")), key=lambda r: r["stream_seq"])
        if any(not access["resolved"] for row in memory_rows for access in row["memory_ranges"]):
            raise ValueError("Unresolved dynamic byte address")
        pending = {}
        for row in memory_rows:
            pending = {key: prior for key, prior in pending.items() if done[key] + 1 > row["cy"]}
            for prior in pending.values():
                for a in row["memory_ranges"]:
                    for b in prior["memory_ranges"]:
                        if (a["base_object_id"] == b["base_object_id"] and a["access_kind"] != b["access_kind"]
                                and max(a["byte_start"], b["byte_start"]) < min(a["byte_end"], b["byte_end"])):
                            raise ValueError("RAW/WAR done+1 release violated")
            pending[row["inst_id"]] = row
        results["local"]["raw_war_release_verified"] = True
    return dict(global_cycles=results["global"]["vf_end_cycle"],
                local_cycles=results.get("local", {}).get("vf_end_cycle"), removed_membars=removed,
                runs=results, dynamic_order_match="local" in states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--arches", nargs="+", choices=("A5", "A6"), default=("A5", "A6"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    global MODEL
    MODEL = args.model_root.resolve()
    root = args.root.absolute()
    sys.path.insert(0, str(MODEL))
    from core.params.loader import load_parameters
    from api.json.serialization import canonical_vf_info_to_dict
    commit = subprocess.check_output(["git", "-C", str(MODEL), "rev-parse", "HEAD"], text=True).strip()
    data = read(root / "metadata/spill_candidate_all_versions.json")
    out = root / "multisoc_refresh_20261025"
    parameters = {arch: load_parameters(MODEL, soc=soc) for arch, soc in (("A5", "dav-3510"), ("A6", "dav-9201"))}
    results = read(out / "summary.json") if (args.resume or args.only or len(args.arches) == 1) and (out / "summary.json").exists() else {}
    save(out / "provenance.json", dict(model_commit=commit, model_branch="MultiSoC-membar-refactor",
        configuration_sha256={str(p.relative_to(MODEL)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (MODEL / "configs").rglob("*.json")},
        scope="All 153 successful archived versions plus their selected software implementations",
        scalar_timing_modeled=False, camodel_architecture="A5 only; no A6 precision is claimed"))
    rows = [row for row in data if not args.only or row["id"] in args.only]
    for index, row in enumerate(rows):
        key = row["id"]
        if args.resume and key in results:
            continue
        print(f"[{index+1}/{len(rows)}] {key}", flush=True)
        result = copy.deepcopy(results.get(key, dict(id=key, case=row["case"], title=row["title"], arches={"A5": {}, "A6": {}})))
        input_cache = {}
        for role in ("original", "software"):
            if role == "software" and (not row.get("files", {}).get("software")):
                for arch in args.arches:
                    result["arches"][arch][role] = dict(result["arches"][arch].get("original", {})) if not row["spill"] else dict(status="unavailable", reason="No archived same-unroll software optimization")
                continue
            try:
                inputs, evidence = find_inputs(root, row, role, out / key / role)
                input_key = json.dumps([canonical_vf_info_to_dict(vf) for vf in inputs], sort_keys=True)
                for arch in args.arches:
                    if role == "software" and (arch, input_key) in input_cache:
                        record = copy.deepcopy(input_cache[arch, input_key])
                        record["reused_identical_original_input"] = True
                        result["arches"][arch][role] = record
                        continue
                    record = dict(status="pending", evidence=evidence, vfs=[])
                    try:
                        for vf_index, vf in enumerate(inputs):
                            try:
                                metadata = memory_metadata(vf) if role == "original" else None
                            except Exception as error:
                                metadata = None
                                record["local_reason"] = str(error)
                            record["vfs"].append(run_one(vf, parameters[arch], arch, metadata,
                                out / key / role / arch / f"vf_{vf_index:03d}",
                                cycle_limit=max(100_000, 5 * int(row.get("camodel") or 0) + 5000)))
                        record.update(status="predicted", global_cycles=sum(v["global_cycles"] for v in record["vfs"]),
                                      local_cycles=sum(v["local_cycles"] for v in record["vfs"]) if all(v["local_cycles"] is not None for v in record["vfs"]) else None)
                    except Exception as error:
                        record.update(status="unsupported", reason=str(error)[:2200], traceback=traceback.format_exc())
                    result["arches"][arch][role] = record
                    if record["status"] == "predicted":
                        input_cache[arch, input_key] = record
            except Exception as error:
                for arch in args.arches:
                    result["arches"][arch][role] = dict(status="unsupported", reason=str(error)[:5000])
        results[key] = result
        save(out / key / "comparison.json", result)
        save(out / "summary.json", results)
        print(json.dumps({arch: {role: (record.get("status"),record.get("global_cycles"),record.get("local_cycles"))
                                for role, record in value.items()} for arch, value in result["arches"].items()}), flush=True)
    print('COMPLETE', len(results), flush=True)


if __name__ == "__main__":
    main()
