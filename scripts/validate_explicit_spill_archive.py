"""Check published explicit-spill evidence without CANN or VfSimulator."""
import hashlib
import json
from pathlib import Path

from compare_explicit_spill import dynamic_rows, semantic_trace


def main():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "metadata/explicit_spill_roundtrip.json").read_text())
    for item in results:
        base = root / item["case"] / item["variant"]
        folder = base / "explicit_spill_camodel_case"
        for name in ("host.cpp", "run.py", "run.sh", "set_env.sh", "README.txt",
                     "evidence/summary.json", "source/original_pc_mapping.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        original = base / "camodel/core0.veccore0.instr_popped_log.dump"
        explicit = folder / "run_output/core0.veccore0.instr_popped_log.dump"
        expected_hashes = [(original, item["log_sha256"]["original"]),
                           (explicit, item["log_sha256"]["explicit"]),
                           (folder / "source/kernel.cce", item["source_sha256"])]
        for path, digest in expected_hashes:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError((path, "SHA256 changed"))
        if (folder / "source/kernel.cce").read_bytes() != (folder / "run_output/source_used.cce").read_bytes():
            raise ValueError((folder, "Source differs from compilation snapshot"))
        a, b = [semantic_trace(dynamic_rows(p)) for p in (original, explicit)]
        if [r["signature"] for r in a] != [r["signature"] for r in b]:
            raise ValueError((folder, "Dynamic instruction semantics differ"))
        config = json.loads((folder / "case.json").read_text())
        for _, filename in config["golden_pairs"]:
            if (base / "camodel_case/run_output" / filename).read_bytes() != (folder / "run_output" / filename).read_bytes():
                raise ValueError((folder, filename, "Output differs from original CAModel"))
        if not json.loads((folder / "run_output/validation.json").read_text())["passed"]:
            raise ValueError((folder, "Golden failed"))
        print("PASS", item["case"], item["variant"], item["explicit_cycles"], "cycles")
    print("Validated", len(results), "published explicit-spill cases")


if __name__ == "__main__":
    main()
