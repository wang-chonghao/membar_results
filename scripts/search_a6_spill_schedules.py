"""Target-specific, fixed-workload search with architectural liveness limits.

The model and its physical register counts are not changed. Static liveness is
checked on the fully target-expanded dynamic stream, before simulation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import json
import io
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import time

KINDS = ("Register", "PredicateRegister")
LIMITS = dict(Register=32, PredicateRegister=8)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def setup_model(model, architecture="A6"):
    sys.path.insert(0, str(model))
    from core.params.loader import load_parameters
    from core.params.database import ParamDB
    soc = {"A5": "dav-3510", "A6": "dav-9201"}[architecture]
    return ParamDB(base_dir=str(model), parameters=load_parameters(model, soc=soc))


def instance_key(instance):
    return (instance["definition_id"], json.dumps(instance["iteration_path"], sort_keys=True))


def audit_liveness(prepared):
    """SSA interval coloring; a dying source slot may be reused by this write."""
    sequences = []
    definitions, uses = set(), Counter()
    storage = {}
    for inst in prepared.instructions:
        sources, dests = set(), set()
        for instances, target in ((inst.src_value_instances, sources), (inst.dst_value_instances, dests)):
            for instance in instances:
                key = instance_key(instance)
                kind = prepared.values[instance["definition_id"]]["storage"]
                if kind in KINDS:
                    target.add(key)
                    storage[key] = kind
        if definitions & dests:
            raise ValueError("Duplicate dynamic SSA definition")
        definitions.update(dests)
        uses.update(sources)
        sequences.append((inst, sources, dests))
    live = set(uses) - definitions
    if any(storage[key] == "PredicateRegister" for key in live):
        raise ValueError("Unexpected predicate live-in")
    peaks = Counter(storage[key] for key in live)
    witness = {}
    for inst, sources, dests in sequences:
        if not sources <= live:
            raise ValueError("Dynamic use before definition")
        for key in sources:
            uses[key] -= 1
            if not uses[key]:
                live.remove(key)
        counts = Counter(storage[key] for key in live | dests)
        for kind in KINDS:
            if counts[kind] > peaks[kind]:
                peaks[kind] = counts[kind]
                witness[kind] = dict(static_instruction_id=inst.static_instruction_id,
                                     iteration_path=inst.iteration_path, stream_seq=inst.stream_seq)
        live.update(key for key in dests if uses[key])
    return dict(peaks={kind: peaks[kind] for kind in KINDS}, limits=LIMITS,
                feasible=all(peaks[kind] <= LIMITS[kind] for kind in KINDS),
                dynamic_instructions=len(sequences), witnesses=witness,
                after_soc_expansion=True, includes_prefixes_invariants_and_loop_exits=True)


def node_id(node):
    return node.get("instruction_id", node.get("loop_id", node.get("membar_id")))


def memory_intervals(node, values):
    result = []
    for operand in (*node["inputs"], *node["outputs"]):
        memory = operand.get("memory_access")
        if not memory:
            continue
        value = values[operand["value_id"]]
        unit = memory.get("address_unit_bytes")
        if unit is None:
            unit = {"fp32": 4, "uint32": 4, "int32": 4, "fp16": 2, "bf16": 2,
                    "uint16": 2, "int16": 2, "uint8": 1, "int8": 1}.get(value["dtype"])
        base = memory.get("base_object_id", value.get("storage_object_id"))
        if unit is None or memory.get("span") is None or memory.get("post_update_delta_bytes") is not None:
            result.append((base, None, None, None))
            continue
        offset = memory["offset"]
        coeffs = tuple(sorted((term["variable_id"], term["coefficient"] * unit) for term in offset["terms"]))
        start = offset["constant"] * unit
        result.append((base, coeffs, start, start + memory["span"] * unit))
    return result


def dependencies(nodes, values):
    producers = {out["value_id"]: index for index, node in enumerate(nodes) for out in node["outputs"]}
    parents = [set(producers[o["value_id"]] for o in node["inputs"] if o["value_id"] in producers)
               for node in nodes]
    memory = [memory_intervals(node, values) for node in nodes]
    for index, node in enumerate(nodes):
        for before in range(index):
            previous = nodes[before]
            if any(key.startswith("align_state") for n in (node, previous) for key in n.get("attributes", {})):
                parents[index].add(before)
            if any(o.get("memory_access", {}).get("address_state_id")
                   for n in (node, previous) for o in (*n["inputs"], *n["outputs"]) if o.get("memory_access")):
                parents[index].add(before)
            if "store" not in (node["instruction_class"], previous["instruction_class"]):
                continue
            for a in memory[before]:
                for b in memory[index]:
                    if a[0] is not None and b[0] is not None and a[0] != b[0]:
                        continue
                    if a[1] is None or b[1] is None or a[1] != b[1] or max(a[2], b[2]) < min(a[3], b[3]):
                        parents[index].add(before)
        if any(p >= index for p in parents[index]):
            raise ValueError("Scheduling seed has a forward SSA edge")
    return parents


def reorder_block(nodes, values, traits, weight, random_seed):
    parents = dependencies(nodes, values)
    children = [set() for _ in nodes]
    for index, ps in enumerate(parents):
        for p in ps:
            children[p].add(index)
    ranks = [0] * len(nodes)
    for index in reversed(range(len(nodes))):
        n = nodes[index]
        ranks[index] = traits.get(n["opcode"] + "." + n["form"], 9) + max((ranks[c] for c in children[index]), default=0)
    inputs = [{o["value_id"] for o in node["inputs"] if values[o["value_id"]]["storage"] in KINDS} for node in nodes]
    outputs = [{o["value_id"] for o in node["outputs"] if values[o["value_id"]]["storage"] in KINDS} for node in nodes]
    uses = Counter(v for ins in inputs for v in ins)
    available = {i for i, ps in enumerate(parents) if not ps}
    emitted, done = [], set()
    rng = random.Random(random_seed)
    ties = [rng.random() for _ in nodes]
    while available:
        def score(i):
            delta = sum(4 if values[v]["storage"] == "PredicateRegister" else 1 for v in outputs[i] if uses[v])
            delta -= sum(4 if values[v]["storage"] == "PredicateRegister" else 1 for v in inputs[i] if uses[v] == 1)
            return (weight * delta - ranks[i] / 8, ties[i] if random_seed else i)
        selected = min(available, key=score)
        available.remove(selected)
        emitted.append(selected)
        done.add(selected)
        for value in inputs[selected]:
            uses[value] -= 1
        available.update(c for c in children[selected] if c not in done and parents[c] <= done)
    if len(emitted) != len(nodes):
        raise ValueError("Cyclic scheduling graph")
    position = {old: new for new, old in enumerate(emitted)}
    if any(position[p] >= position[i] for i, ps in enumerate(parents) for p in ps):
        raise AssertionError("Reorder broke a dependency")
    return [nodes[i] for i in emitted]


def reorder(payload, traits, weight, random_seed):
    result = copy.deepcopy(payload)

    def visit(nodes):
        reordered, block = [], []
        def flush():
            reordered.extend(reorder_block(block, result["values"], traits, weight, random_seed))
            block.clear()
        for node in nodes:
            if node["kind"] == "instruction":
                block.append(node)
            else:
                flush()
                if node["kind"] == "loop":
                    node["body"] = visit(node["body"])
                reordered.append(node)
        flush()
        return reordered

    result["context"] = visit(result["context"])
    return result


def semantic_fingerprint(payload):
    def visit(nodes):
        return sorted((dict(node, body=visit(node["body"])) if node["kind"] == "loop" else node
                       for node in nodes), key=lambda n: str(node_id(n)))
    data = copy.deepcopy(payload)
    data["context"] = visit(data["context"])
    return digest(data)


def seed_inputs(root, row, output, architecture="A6"):
    paths = sorted((root / "multisoc_refresh_20261025" / row["id"] / f"software/{architecture}").glob("vf_*/global_input.json"))
    seeds = [(dict(source="same_model_archived_software", files=[str(p.relative_to(root)) for p in paths]),
              [read(path) for path in paths])] if paths else []
    generated = output / row["id"] / "seeds.json.gz"
    if generated.exists():
        with gzip.open(generated, "rt") as stream:
            records = json.load(stream)
        for index, item in enumerate(records["candidates"]):
            seeds.append((dict(source="fresh_fixed_unroll_partition", index=index, plan=item["plan"]), [item["payload"]]))
    for item in row.get("files", {}).get("software", []):
        path = root / item["path"]
        if path.name == "provisional.json" and path.exists():
            seeds.append((dict(source="archived_provisional", file=item["path"]), [read(path)]))
    return seeds


def run_case(arguments):
    model, root, output, row, traits, resume, architecture = arguments
    folder = output / row["id"]
    if resume and (folder / "result.json").exists():
        return read(folder / "result.json")
    db = setup_model(model, architecture)
    from api.json.adapter import CanonicalJsonVfInfoAdapter
    from api.json.serialization import canonical_vf_info_to_dict
    from core.params.uarch import resolve_uarch
    from core.execution.prepare import prepare_program
    from core.execution.engine import Engine
    from refresh_multisoc_predictions import migrate_payload, instruction_limit

    scored, rejected, seen = [], [], set()
    best, best_payloads = None, None
    started = time.monotonic()
    seeds = seed_inputs(root, row, output, architecture)
    for provenance, payloads in seeds:
        try:
            payloads = [migrate_payload(p)[0] for p in payloads]
        except Exception as error:
            rejected.append(dict(seed=provenance, reason=str(error)[:1600]))
            continue
        # Fresh seeds already contain three target-prioritized schedules.
        variants = [(None, 0)] if provenance["source"] == "fresh_fixed_unroll_partition" else [(None, 0), (0, 0), (2, 0), (8, 0), (24, 0), (8, 17), (2, 41)]
        for weight, random_seed in variants:
            plan = dict(seed=provenance, reorder_weight=weight, reorder_seed=random_seed)
            try:
                candidate = [reorder(p, traits, weight, random_seed) for p in payloads] if weight is not None else copy.deepcopy(payloads)
                if any(semantic_fingerprint(a) != semantic_fingerprint(b) for a, b in zip(payloads, candidate)):
                    raise AssertionError("Reordering changed more than instruction order")
                key = digest(candidate)
                if key in seen:
                    continue
                seen.add(key)
                programs, audits, emitted = [], [], []
                for payload in candidate:
                    vf = CanonicalJsonVfInfoAdapter.from_payload(payload)
                    uarch = resolve_uarch(db.base_uarch, vf.uarch)
                    # Safety limits do not change architectural resources or timings.
                    payload["params"]["max_cycles"] = 100000
                    payload["uarch"]["canonical_dynamic_instruction_limit"] = instruction_limit(vf, db.instruction_expansions)
                    vf = CanonicalJsonVfInfoAdapter.from_payload(payload)
                    uarch = resolve_uarch(db.base_uarch, vf.uarch)
                    prepared = prepare_program(vf, db, uarch)
                    audit = audit_liveness(prepared)
                    audits.append(audit)
                    if not audit["feasible"]:
                        raise ValueError("architectural_liveness_limit")
                    programs.append((prepared, uarch))
                    emitted.append(canonical_vf_info_to_dict(vf))
                cycles = sum(Engine(p, u, db, ub_dependency_mode="disabled").run(None).vf_end_cycle for p, u in programs)
                record = dict(plan=plan, digest=key, global_cycles=cycles, audits=audits)
                scored.append(record)
                if best is None or cycles < best["global_cycles"]:
                    best, best_payloads = record, emitted
            except Exception as error:
                rejected.append(dict(plan=plan, reason=str(error)[:1600], audits=locals().get("audits", [])))
    record = dict(id=row["id"], title=row["title"], spill_kind=row["spill_kind"], unroll=row["unroll"],
                  status="predicted" if best else "no_feasible_candidate", best=best,
                  unique_candidates=len(seen), feasible_candidates=len(scored), seed_count=len(seeds),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    save(folder / "search.json", dict(scored=scored, rejected=rejected))
    if best is not None:
        for index, payload in enumerate(best_payloads):
            save(folder / f"best_vf_{index:03d}.json", payload)
        record["evidence"] = str((folder / "result.json").relative_to(root))
    save(folder / "result.json", record)
    print(row["id"], record["status"], best["global_cycles"] if best else None,
          "feasible", len(scored), "/", len(seen), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--architecture", choices=("A5", "A6"), default="A6")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, model = args.root.absolute(), args.model_root.absolute()
    output = args.output.absolute() if args.output else root / f"{args.architecture.lower()}_software_search_20261025"
    provenance = read(root / "multisoc_refresh_20261025/provenance.json")
    commit = subprocess.check_output(["git", "-C", str(model), "rev-parse", "HEAD"], text=True).strip()
    snapshot = None
    if commit != provenance["model_commit"]:
        snapshot = tempfile.TemporaryDirectory(prefix="vfsim-schedule-model-")
        archive = subprocess.check_output(["git", "-C", str(model), "archive", provenance["model_commit"], "api", "core", "configs"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(snapshot.name, filter="data")
        model = Path(snapshot.name)
        print("Using archived report model", provenance["model_commit"], flush=True)
    for name, expected in provenance["configuration_sha256"].items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Search configuration differs from baseline: {name}")
    db = setup_model(model, args.architecture)
    # Use concrete ISA forms plus semantic forms; expansion latency is the sum of stages.
    profile = {"A5": "DV100", "A6": "DV121"}[args.architecture]
    isa = read(model / f"configs/socs/{profile}/isa.json")
    traits = {}
    for op, entry in isa.get("instructions", isa).items():
        if not isinstance(entry, dict):
            continue
        for form in entry.get("forms", {}):
            traits[op + "." + form] = db.resolve_inst(op, form).latency
    for op, rule in read(model / f"configs/socs/{profile}/instruction_expansion.json")["expansions"].items():
        for form in rule["forms"]:
            traits[op + "." + form] = sum(db.resolve_inst(rule[field], form).latency
                                         for field in ("first_opcode", "second_opcode"))
    save(output / "traits.json", traits)
    if not args.skip_generation:
        cmd = [sys.executable, str(root / "scripts/build_a6_search_seeds.py"), "--root", str(root),
               "--output", str(output), "--traits", str(output / "traits.json")]
        if args.ids:
            cmd += ["--ids", *args.ids]
        subprocess.run(cmd, check=True, cwd=root.parents[1])
    rows = [r for r in read(root / "metadata/spill_candidate_alignment.json")["rows"]
            if r["spill"] and r.get("software_cycles") and r.get("local_cycles")]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids]
    save(output / "provenance.json", dict(model_commit=provenance["model_commit"],
         model_config_hashes=provenance["configuration_sha256"],
         limits=LIMITS, architecture={"A5": "dav-3510", "A6": "dav-9201"}[args.architecture], nominal_unroll_preserved=True,
         mode="global_membar", compiler_verified=False,
         algorithm="Target critical path / liveness priorities, up to five loop partitions, lane cohorts, then block-local topological reorder"))
    arguments = [(model, root, output, row, traits, args.resume, args.architecture) for row in rows]
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for future in as_completed([executor.submit(run_case, item) for item in arguments]):
            future.result()
    results = {p.parent.name: read(p) for p in output.glob("*/result.json")}
    save(output / "summary.json", results)
    if snapshot is not None:
        snapshot.cleanup()


if __name__ == "__main__":
    main()
