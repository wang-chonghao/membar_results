"""Verify the bundled evidence and report data before sharing the archive."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "metadata/data.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
variants = [v for case in data["cases"] for v in case["variants"]]
assert len(data["cases"]) == 6 and len(variants) == 25
for entry in manifest:
    path = root / entry["path"]
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
for case in data["cases"]:
    assert case["default"] in {v["id"] for v in case["variants"]}
    for v in case["variants"]:
        folder = root / case["id"] / v["id"]
        assert sorted(p.name for p in (folder / "camodel").iterdir()) == [
            "core0.veccore0.instr_log.dump", "core0.veccore0.instr_popped_log.dump"]
        for item in v["sources"]:
            assert (root / item["path"]).is_file()
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
                  "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}))
