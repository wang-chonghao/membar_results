"""Publish newly measured pure CCE sources, keeping prior evidence separately."""

import hashlib
import json
from pathlib import Path
import re


def assert_pure_source(source):
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    if re.search(r"\b(?:AscendC|MicroAPI|Reg)::|\bRegTensor\s*<|\busing\s+namespace\s+(?:AscendC|MicroAPI)\b", code):
        raise ValueError("Published CCE still contains an AscendC interface")


def validate_result(record, source):
    if record["status"] not in {"predicted", "unsupported"}:
        raise ValueError("Pure CCE replacement has no audited prediction")
    provenance = record["provenance"]
    check = provenance["numerical_validation"]
    if not (check["bitwise_matches_original"] and check["guard_ok"] and check["golden"]["ok"]
            and provenance["spill_traffic_unchanged"] and provenance["model_stable"]):
        raise ValueError("Pure CCE numerical or compiler evidence failed")
    assert_pure_source((source / "cce/kernel.cce").read_text())
    hashes = provenance["evidence_sha256"]
    required = {"kernel.cce", "input.bin", "output.bin", "commands.json",
                "core0.veccore0.instr_log.dump", "core0.veccore0.instr_popped_log.dump",
                "core0.veccore0.ub.rd_log.dump", "core0.veccore0.ub.wr_log.dump"}
    if not required.issubset(hashes):
        raise ValueError("Pure CCE source/log provenance is incomplete")
    for name, digest in hashes.items():
        if Path(name).name != name or hashlib.sha256((source / "cce" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Measured CCE evidence changed: {name}")
    for key, name in (("source_sha256", "kernel.cce"), ("input_sha256", "input.bin"), ("output_sha256", "output.bin")):
        if provenance[key] != hashes[name]:
            raise ValueError("Source/input/output identity differs from compiled evidence")
    binary = source / "cce/kernel.binary"
    if not binary.exists():
        binary = source / "cce/kernel.o"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != provenance["kernel_sha256"]:
        raise ValueError("Compiled kernel identity differs from CAModel measurement")
    for opcode in ("RV_VSTI", "RV_VLDI", "RV_PSTI", "RV_PLDI", "RV_SSTI", "RV_SLDI", "RV_SMEM_BAR"):
        if provenance["old_rv_opcounts"].get(opcode, 0) != provenance["new_rv_opcounts"].get(opcode, 0):
            raise ValueError("Compiler spill counts changed without reclassification")
    if record["camodel"] != record["measurements"]["total_vf_cycles"]:
        raise ValueError("CAModel cycles differ from the new measured source")
    current = record["prediction"]
    if record["status"] == "unsupported":
        scalar = provenance["new_rv_opcounts"].get("RV_SSTI", 0) or provenance["new_rv_opcounts"].get("RV_SLDI", 0)
        if not scalar or current["status"] != "unsupported" or record["global_cycles"] is not None or record["local_cycles"] is not None:
            raise ValueError("Unsupported scalar spill has fabricated predictions")
        return
    experiment = current["experiment"]
    count = experiment.get("memory_access_count")
    if (current["status"] != "predicted" or experiment["status"] != "predicted"
        or record["global_cycles"] != current["global_cycles"] or record["global_cycles"] != experiment["global_cycles"]
        or record["local_cycles"] != experiment["local_cycles"]
        or not experiment["dynamic_order_match"] or not experiment["raw_war_release_boundary_verified"]
        or type(count) is not int or count <= 0 or experiment["ub_log_verified_accesses"] != count):
        raise ValueError("Pure CCE A/B instruction or address audit failed")


def apply_pure_cce_updates(data, repo, dest, archive, manifest, copy_tree):
    root = repo / "cce_code/spill_candidate_alignment/pure_cce_refresh"
    updated = []
    models = set()
    for row in data["rows"]:
        source = root / row["id"]
        if not (source / "result.json").exists():
            continue
        record = json.loads((source / "result.json").read_text())
        if record["id"] != row["id"]:
            raise ValueError("Pure CCE replacement belongs to another case")
        validate_result(record, source)
        provenance = record["provenance"]
        models.add(json.dumps(provenance["current_model_sha256"], sort_keys=True))
        if len(models) != 1:
            raise ValueError("Pure CCE versions were predicted using different current models")
        original = dest / provenance["original_source_path"]
        if hashlib.sha256(original.read_bytes()).hexdigest() != provenance["original_source_sha256"]:
            raise ValueError("Original mixed-source evidence was modified")
        files = copy_tree(source, archive / "pure_cce" / row["id"], manifest)
        if not row.get("pure_cce_refresh"):
            row["previous_cce_measurements"] = {key: row[key] for key in ("camodel", "global_cycles", "local_cycles")}
            row["files"]["previous_cce"] = row["files"].get("original_cce", [])
            row["files"]["previous_comparison"] = row["files"].get("comparison", [])
        row["files"]["original_cce"] = [f for f in files if f["label"].startswith("cce/")]
        row["files"]["comparison"] = [f for f in files if not f["label"].startswith("cce/")]
        row["pure_cce_refresh"] = record
        row.update(status=record["status"], reason=record.get("reason", ""),
                   **{key: record[key] for key in ("camodel", "global_cycles", "local_cycles")})
        audit = "全部访存核验通过" if record["status"] == "predicted" else "新编译流明确拒绝未建模标量 spill，不填预测周期"
        note = " 纯 CCE 复测：显式底层指令，无 AscendC 运算接口；CAModel 与 VfSim 使用同一源码的新编译流，独立 golden、原输出逐位比较通过；" + audit + "。旧源码和旧测量单独保留。"
        if note not in row["note"]:
            row["note"] += note
        updated.append(row["id"])
    data["pure_cce_updated_versions"] = updated
