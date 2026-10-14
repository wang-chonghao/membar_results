"""Verify the portable report without requiring CANN, VfSim or a browser."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_file(item):
    file = (ROOT / item["path"]).resolve()
    if not file.is_relative_to(ROOT) or not file.is_file():
        raise ValueError(f"Missing or non-local archive file: {file}")
    with file.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != item["sha256"]:
        raise ValueError(f"Archive checksum differs: {file}")
    return file.stat().st_size


def main():
    data = json.loads((ROOT / "metadata/spill_candidate_alignment.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) != len({r["id"] for r in rows}):
        raise ValueError("Wrong experiment coverage or duplicate IDs")
    origins=Counter(r["origin"] for r in rows)
    round3 = data.get("round3_import", {})
    if (origins.get("fixed_unroll_search")!=33 or origins.get("historical_membar")!=24
        or origins.get("supplemental_search")!=10 or origins.get("round3_search",0)!=round3.get("selected_versions",0)):
        raise ValueError("Wrong archive source coverage")
    from build_spill_candidate_report import spill_and_baseline_rows
    all_rows=json.loads((ROOT/"metadata/spill_candidate_all_versions.json").read_text(encoding="utf-8"))
    if len(all_rows)!=116+round3.get("archived_successful_versions",0) or {row["id"] for row in rows}!={row["id"] for row in spill_and_baseline_rows(all_rows)}:
        raise ValueError("Report must retain all register spill and one baseline per comparison group")
    if sum(row["spill"] for row in rows)!=40+round3.get("selected_spill_versions",0):
        raise ValueError("Some register spill versions are missing")
    baselines=[row for row in rows if not row["spill"]]
    if len({(row["case"],row["comparison_group"]) for row in baselines})!=len(baselines):
        raise ValueError("Duplicate baseline within a comparison group")
    if any(row["unroll"]!=1 and row["case"]!="softmax_three_loop" for row in baselines):
        raise ValueError("A non-U1 version was retained as a baseline")
    excluded = json.loads((ROOT / "metadata/spill_candidate_excluded_versions.json").read_text(encoding="utf-8"))
    if len(excluded) != 20+round3.get("stack_failures",0) or any(row["status"] != "vf_stack_exceeded" for row in excluded):
        raise ValueError("Compile-failure records were not preserved separately")
    with (ROOT / "spill_candidate_alignment.csv").open(encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != len(rows):
        raise ValueError("CSV differs from HTML dataset")
    for row, record in zip(rows, records):
        if row.get("pure_cce_refresh"):
            from spill_pure_cce_import import validate_result
            base = ROOT / "spill_candidate_alignment/pure_cce" / row["id"]
            refresh = json.loads((base / "result.json").read_text())
            if refresh != row["pure_cce_refresh"]:
                raise ValueError("Report differs from pure CCE evidence")
            validate_result(refresh, base)
            if row["source_cce_path"] != (base / "cce/kernel.cce").relative_to(ROOT).as_posix():
                raise ValueError("Report links to a source other than the measured pure CCE")
            for key in ("camodel", "global_cycles", "local_cycles"):
                if row[key] != refresh[key]:
                    raise ValueError("Report mixes old and new CCE measurements")
        for key in ("case", "origin", "variant", "selection_role", "loop_count", "loop_count_label", "loop_count_detail", "unroll", "status", "spill_kind", "vector_spill_store", "vector_spill_load", "predicate_spill_store", "predicate_spill_load", "scalar_stack_stores", "scalar_stack_loads", "source_cce_path", "compiled_cce_path", "camodel", "global_cycles", "local_cycles", "software_cycles", "software_camodel", "software_accuracy", "membar_vs_software_performance"):
            expected = "" if row.get(key) is None else str(row[key])
            if record[key] != expected:
                raise ValueError(f"CSV differs: {row['id']} / {key}")
        if row["status"] != "predicted" and (row["global_cycles"] is not None or row["local_cycles"] is not None):
            raise ValueError("Unsupported original stream has fabricated cycles")
        if type(row["loop_count"]) is not int or row["loop_count"]<1:
            raise ValueError("Original loop count missing or invalid")
        if row["origin"]=="fixed_unroll_search" and row["loop_count"]!=16:
            raise ValueError("Loop count was incorrectly divided by unroll")
        if row["origin"]=="supplemental_search" and row["loop_count"]!=48:
            raise ValueError("Supplemental loop count differs from 48")
        counts=[row[key] for key in ("vector_spill_store","vector_spill_load","predicate_spill_store","predicate_spill_load")]
        if any(type(count) is not int or count<0 for count in counts):
            raise ValueError("Spill count is missing or invalid")
        vector,predicate=any(counts[:2]),any(counts[2:])
        scalar=bool(row.get("scalar_stack_stores") or row.get("scalar_stack_loads"))
        kind="vector_predicate" if vector and predicate else "vector" if vector else "predicate" if predicate else "scalar" if scalar else "none"
        if row["spill_kind"]!=kind or row["spill"]!=(kind!="none"):
            raise ValueError("Spill classification differs from recorded counts")
        if not row["source_cce_path"].endswith(".cce") or not (ROOT/row["source_cce_path"]).is_file():
            raise ValueError("Original CCE link missing")
        from spill_pure_cce_import import assert_pure_source
        assert_pure_source((ROOT / row["source_cce_path"]).read_text())
        if row["plan"] and (row["plan"]["unroll"] != row["unroll"] or not row["software_validation"]["ok"]):
            raise ValueError("Software finalist changed unroll or failed numerical validation")
        if row["software_cycles"] is not None and row["software_camodel"] is None:
            raise ValueError("Software-side prediction is missing its measured counterpart")
        if row["software_camodel"] is not None:
            expected_accuracy = 100 * (1 - abs(row["software_cycles"] - row["software_camodel"]) / row["software_camodel"])
            if abs(row["software_accuracy"] - expected_accuracy) > 1e-9:
                raise ValueError("Software-side accuracy differs from its cycles")
        if row["plan"] is None and not row.get("software_comparison") and row["status"] == "predicted" and not row["spill"]:
            if row["software_camodel"] != row["camodel"] or row["software_cycles"] != row["global_cycles"]:
                raise ValueError("Retained version must reuse its original measurements")
        if row["local_cycles"] is None or row["software_cycles"] is None:
            if row["membar_vs_software_performance"] is not None:
                raise ValueError("Missing cycles have fabricated relative performance")
        elif abs(row["membar_vs_software_performance"] - 100*row["software_cycles"]/row["local_cycles"]) > 1e-9:
            raise ValueError("Relative performance must compare two VfSim predictions")
        if row.get("software_comparison"):
            counterpart=row["software_comparison"]
            if counterpart["unroll"] != row["unroll"]:
                raise ValueError("Historical software counterpart changes unroll")
            if row["software_cycles"]!=counterpart["cycles"] or row["software_camodel"]!=counterpart["camodel"]:
                raise ValueError("Historical software counterpart cycles differ")
        if row["origin"] == "round3_search":
            check=row["numeric_validation"]
            if not (check["bitwise_identical"] and check["guard_ok"] and check["cce_vs_formula"]["ok"] and check["ascendc_vs_formula"]["ok"]):
                raise ValueError("Round-three numerical evidence failed")
            if row["spill"] and any(row[k] is not None for k in ("software_cycles","software_camodel","software_accuracy")):
                raise ValueError("Round-three software optimization was not measured")
    historical=json.loads((ROOT/"metadata/data.json").read_text(encoding="utf-8"))
    originals={(case["id"],variant["id"]):variant for case in historical["cases"] for variant in case["variants"]}
    for row in rows:
        if row["origin"]!="historical_membar":continue
        original=originals[(row["case"],row["variant"])]
        update = row.get("compiled_replay_update")
        if update:
            from spill_historical_import import validate_compiled_replay_update
            provenance = row["compiled_replay_provenance"]
            validate_compiled_replay_update(update, provenance, row)
            for key in ("camodel", "global_cycles", "local_cycles", "source_cycles"):
                if row["historical_record"].get(key) != original.get(key):
                    raise ValueError("Original historical record was overwritten")
            base = ROOT / "spill_candidate_alignment/imported" / row["case"] / row["variant"] / "compiled_replay_current"
            if update != json.loads((base / "prediction.json").read_text()) or provenance != json.loads((base / "provenance.json").read_text()):
                raise ValueError("Historical update differs from archived prediction evidence")
            if hashlib.sha256((ROOT / row["source_cce_path"]).read_bytes()).hexdigest() != provenance["original_source_sha256"]:
                raise ValueError("Historical update source checksum differs")
        for key in ("camodel","global_cycles","local_cycles","source_cycles"):
            expected = update[key] if update and key in {"global_cycles", "local_cycles"} else original.get(key)
            if row.get(key)!=expected:raise ValueError("Historical measurements were changed without audited replay")
    manifest = json.loads((ROOT / "metadata/spill_candidate_manifest.json").read_text(encoding="utf-8"))
    if len({item["path"] for item in manifest}) != len(manifest):
        raise ValueError("Duplicate manifest entries")
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(verify_file, manifest))
    print(json.dumps(dict(rows=len(rows), verified_files=len(manifest), verified_bytes=sum(sizes),
                          original_missing_preserved=True, software_finalists=sum(r['plan'] is not None for r in rows),
                          spill_versions=sum(r['spill'] for r in rows),baselines=len(baselines),all_archived_versions=len(all_rows))))


if __name__ == "__main__":
    main()
