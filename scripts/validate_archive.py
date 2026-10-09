"""Verify the bundled evidence and report data before sharing the archive."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import stat

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "metadata/data.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
variants = [v for case in data["cases"] for v in case["variants"]]
assert data['cases'] and variants
assert len({case['id'] for case in data['cases']}) == len(data['cases'])
def verify_file(entry):
    path = root / entry["path"]
    info = path.stat()
    assert stat.S_ISREG(info.st_mode), path
    if info.st_size == 0:
        checksum = hashlib.sha256(b"").hexdigest()
    else:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum == entry["sha256"], path
    return entry["path"], info.st_size

with ThreadPoolExecutor(max_workers=16) as pool:
    verified_sizes = dict(pool.map(verify_file, manifest))
for case in data["cases"]:
    assert len({v['id'] for v in case['variants']}) == len(case['variants'])
    assert case["default"] in {v["id"] for v in case["variants"]}
    if case.get('overlay_figure'):
        assert (root / case['overlay_figure']).is_file()
    for v in case["variants"]:
        folder = root / case["id"] / v["id"]
        assert sorted(p.name for p in (folder / "camodel").iterdir()) == [
            "core0.veccore0.instr_log.dump", "core0.veccore0.instr_popped_log.dump"]
        for item in v["sources"]:
            assert (root / item["path"]).is_file()
        if v.get("camodel_case"):
            bundle = root / v["camodel_case"]["path"]
            config = json.loads((bundle / "case.json").read_text(encoding="utf-8"))
            assert config["expected_vf_cycles"] == v["camodel"]
            for name in ("host.cpp", "run.py", "README.txt", config["kernel_source"]):
                assert (bundle / name).is_file(), bundle / name
            for name in ("core0.veccore0.instr_log.dump", "core0.veccore0.instr_popped_log.dump"):
                assert hashlib.sha256((bundle / "run_output" / name).read_bytes()).hexdigest() == hashlib.sha256((folder / "camodel" / name).read_bytes()).hexdigest()
            for expected, actual in config["golden_pairs"]:
                assert (bundle / "run_output" / expected).is_file()
                assert (bundle / "run_output" / actual).is_file()
        for path in v["logs"] + v["evidence"]:
            assert (root / path).is_file()
        if v["vector_store"] or v["predicate_store"]:
            hardware = v["hardware_equivalent"]
            file = root / hardware["path"]
            mapping = json.loads((root / hardware["mapping"]).read_text(encoding="utf-8"))
            assert mapping["modeled_pc_coverage_complete"]
            assert mapping["dynamic_instruction_counts"]["RV_SMEM_BAR"] == v["membars"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == mapping["sha256"]
            assert v["sources"][0]["path"] == hardware["path"]
        assert v["camodel"] > 0
        assert (folder / "figures/ipc.png").is_file()
        if v["accuracy"] is not None:
            assert abs(v["accuracy"] - 100 * (1 - abs(v["global_cycles"] - v["camodel"]) / v["camodel"])) < 0.0001
        for s in v["series"]:
            assert len(s["samples"]) > 10 and max(s["samples"]) > 0
            assert all(value >= 0 for value in s["samples"])
for name in ("membar_results_summary.html", "assets/report.css", "assets/report.js", "metadata/data.js", "assets/chart.umd.js"):
    assert (root / name).is_file()
print(json.dumps({"cases": len(data["cases"]), "variants": len(variants),
                  "verified_files": len(manifest), "camodel_logs": len(variants) * 2,
                  "verified_bytes": sum(verified_sizes.values())}))
