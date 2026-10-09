"""Bundle original CAModel artifacts and standalone rerun recipes per variant."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = "results/ub_address_dependency_experiment/"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nested_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_strings(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vfsim-root", type=Path, required=True)
    args = parser.parse_args()
    repository = args.vfsim_root.resolve()
    data = read(ROOT / "metadata/data.json")
    manifest = read(ROOT / "metadata/manifest.json")
    origins = {item["path"]: repository / item["source"] for item in manifest if item.get("source")}
    new_manifest, reports = {}, []

    def copy(source, target, *, generated=False):
        if not source.is_file():
            raise FileNotFoundError(source)
        source_digest = digest(source)
        different = target.exists() and digest(target) != source_digest
        if different and not generated:
            raise ValueError(f"Existing archive file differs: {target}")
        if not target.exists() or different:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        relative = str(target.relative_to(ROOT))
        new_manifest[relative] = dict(path=relative, source=str(source), sha256=source_digest)

    def tree(source, target):
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                copy(path, target / path.relative_to(source))

    for case in data["cases"]:
        for variant in case["variants"]:
            folder = ROOT / case["id"] / variant["id"]
            target = folder / "camodel_case"
            log_origin = origins[str(folder.relative_to(ROOT)) + "/camodel/core0.veccore0.instr_log.dump"].parent
            parent = log_origin.parent
            original_source = origins[str(folder.relative_to(ROOT)) + "/source/kernel.cce"]
            candidates = []
            for name in ("summary.json", "commands.json", "evidence/commands.json"):
                if (parent / name).is_file():
                    for value in nested_strings(read(parent / name)):
                        path = Path(value)
                        if path.is_absolute() and str(path).startswith("/tmp/"):
                            candidates.append(path if path.is_dir() else path.parent)
            original_run = next((path for path in candidates if path.is_dir()
                                 and (path / "core0.veccore0.instr_log.dump").is_file()
                                 and digest(path / "core0.veccore0.instr_log.dump")
                                 == digest(folder / "camodel/core0.veccore0.instr_log.dump")), None)
            if original_run:
                tree(original_run, target / "run_output")
            tree(log_origin, target / "run_output")
            for name in ("code", "data", "evidence"):
                if (parent / name).is_dir():
                    tree(parent / name, target / name)
            if (parent / "data").is_dir():
                tree(parent / "data", target / "run_output")
            for name in ("summary.json", "commands.json", "alignment.json", "validation.json"):
                if (parent / name).is_file():
                    copy(parent / name, target / "evidence" / name)
            copy(original_source, target / "source" / original_source.name)
            cce = target / "run_output/kernel.cce"
            if not cce.is_file() and (parent / "code/kernel.cce").is_file():
                copy(parent / "code/kernel.cce", cce)
            kernel_source = str((cce if cce.is_file() else target / "source" / original_source.name).relative_to(target))
            source_text = (target / kernel_source).read_text(encoding="utf-8")
            kernel = re.search(r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+(\w+)', source_text)
            if kernel is None:
                raise ValueError(f"No global kernel entry in {kernel_source}")
            runtime = repository / (EXPERIMENTS + "gelu_poly_i96_unroll_spill_20260911/scripts/native_runtime_softmax_main.cpp")
            script_dirs = []
            host_args, host_defines, input_files = [], [], []
            atol, rtol = 1e-6, 1e-3
            generic = case["id"] in {"gelu_poly", "swiglu_grad", "adam_apply_one", "softmax_three_loop"}
            if case["id"] == "gelu_poly":
                scripts = repository / (EXPERIMENTS + "gelu_poly_i96_unroll_spill_20260911/scripts")
                script_dirs = [scripts]
                host_args = ["2", "1", "6144"]
            elif case["id"] == "swiglu_grad":
                scripts = repository / (EXPERIMENTS + "swiglu_grad_fp32_i96/unroll_scan_20260914/scripts")
                script_dirs, runtime = [scripts], scripts / "native_runtime_swiglu_grad_main.cpp"
                host_args = ["3", "2", "6144"]
            elif case["id"] == "adam_apply_one":
                scripts = repository / (EXPERIMENTS + "adam_apply_one_fused_i48_unroll/scripts")
                script_dirs, runtime = [scripts], scripts / "native_runtime_adam_main.cpp"
                host_args = ["10", "3", "3072"]
            elif case["id"] == "softmax_three_loop":
                host_args = ["2", "1", "8192"]
                script_dirs = [runtime.parent]
            else:
                runtime = repository / (EXPERIMENTS + "GeLU_grad/source/predicate_select_test/host.cpp")
                if original_run and (original_run / "host.cpp").is_file():
                    runtime = original_run / "host.cpp"
                input_files = ["input.bin"]
                input_bytes = (target / "run_output/input.bin").stat().st_size
                output_bytes = (target / "run_output/output.bin").stat().st_size
                host_defines = [f"-DPROBE_INPUT_BYTES={input_bytes}", f"-DPROBE_OUTPUT_BYTES={output_bytes}"]
                atol, rtol = 2e-6, 2e-4
                if (parent / "scripts").is_dir():
                    script_dirs.append(parent / "scripts")
                elif (parent.parent / "scripts").is_dir():
                    script_dirs.append(parent.parent / "scripts")
            copy(runtime, target / "host.cpp")
            for directory in script_dirs:
                tree(directory, target / "original_scripts" / directory.name)
            copy(ROOT / "scripts/run_camodel_case.py", target / "run.py", generated=True)
            pairs = []
            for golden in sorted((target / "run_output").glob("golden*.bin")):
                suffix = golden.stem.removeprefix("golden")
                output_name = f"output{suffix}.bin" if suffix or input_files else "output0.bin"
                if (target / "run_output" / output_name).is_file():
                    pairs.append([golden.name, output_name])
            flags = ["--cce-simd-vf-fusion=false", "-mllvm", "-cce-aicore-vec-misched=0"]
            if generic:
                flags = ["-mllvm", "-cce-aicore-function-stack-size=16000", "-mllvm", "-cce-aicore-record-overflow=false",
                         "-mllvm", "-cce-aicore-addr-transform", "-mllvm", "-cce-aicore-jump-expand=true", *flags]
            binaries = sorted(str(p.relative_to(target / "run_output")) for p in (target / "run_output").glob("*.o"))
            config = dict(case=case["id"], variant=variant["id"], kernel_name=kernel[1],
                          kernel_source=kernel_source, host_args=host_args, host_defines=host_defines,
                          input_files=input_files, golden_pairs=pairs, atol=atol, rtol=rtol,
                          ccec_flags=flags, npu_type="Ascend950PR_9599", expected_vf_cycles=variant["camodel"],
                          original_run_dir=str(original_run) if original_run else None,
                          source_origin=str(original_source), logs_origin=str(log_origin),
                          host_source_origin=str(runtime), archived_binaries=binaries,
                          original_run_dir_available=original_run is not None,
                          archived_golden_available=bool(pairs), rerun_performed=False,
                          archived_source_equals_report=(digest(target / kernel_source) == digest(folder / "source/kernel.cce")))
            (target / "case.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            status = "原临时运行目录完整复制" if original_run else "从仓库保留材料恢复；原临时目录已不存在"
            missing = []
            if not original_run:
                missing.append("原临时目录")
            if not binaries:
                missing.append("原编译二进制（可重新构建）")
            if not pairs:
                missing.append("原 input/output/golden 数据")
            note = (f"{case['title']} / {variant['label']} 的 CAModel 用例\n\n"
                    f"历史 VF 时间：{variant['camodel']} cycles。归档方式：{status}。\n"
                    f"缺失材料：{'、'.join(missing) if missing else '无'}。本次仅归档，没有重新运行 CAModel。\n\n"
                    "目录：host.cpp 为 host 源码；run_output 为已保存的运行日志、数据及仍存在的编译产物；\n"
                    "source/code 为源码；evidence 为历史命令与精度记录；original_scripts 为原实验脚本。\n"
                    "case.json 记录实际运行源码、入口、编译参数与材料来源。原脚本可能包含历史绝对路径。\n\n"
                    "直接复现（Linux/WSL，安装对应 CANN）：\n"
                    "  source ../../../set_env.sh  # 首次使用前编辑结果根目录中的环境配置\n"
                    "  python3 run.py --dry-run\n"
                    "  python3 run.py --check-archived\n"
                    f"  python3 run.py --out-dir /tmp/{case['id']}_{variant['id']}_rerun\n\n"
                    "重新运行的输出目录必须为空且位于本归档目录之外；run.py 会编译并执行，然后进行 golden 校验。\n"
                    "没有历史 golden 的版本不支持 --check-archived，但生成式 host 会在重跑时生成并校验 golden。\n"
                    "报告中的 kernel.cce 可能是后续等效改写；复现始终使用 case.json 中的实际历史运行源码。\n")
            (target / "README.txt").write_text(note, encoding="utf-8")
            for name in ("case.json", "README.txt"):
                path = target / name
                new_manifest[str(path.relative_to(ROOT))] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
            for label, name in (("CAModel 完整用例说明", "README.txt"), ("CAModel 构建运行入口", "run.py"), ("CAModel host 源码", "host.cpp")):
                link = dict(label=label, path=str((target / name).relative_to(ROOT)))
                if not any(item["path"] == link["path"] for item in variant["sources"]):
                    variant["sources"].append(link)
            variant["camodel_case"] = dict(path=str(target.relative_to(ROOT)),
                original_run_dir_available=original_run is not None, archived_golden_available=bool(pairs))
            (folder / "result.json").write_text(json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8")
            files = list((target / "run_output").rglob("*"))
            reports.append(dict(case=case["id"], variant=variant["id"], status=status,
                                missing=missing, config_path=str((target / "case.json").relative_to(ROOT)),
                                original_run_dir_available=original_run is not None,
                                archived_golden_available=bool(pairs), files=sum(p.is_file() for p in files)))
            print(f"Bundled {case['id']}/{variant['id']}: {reports[-1]['files']} run files, {status}", flush=True)
        (ROOT / case["id"] / "summary.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("set_env.sh", "scripts/run_camodel_case.py", "scripts/bundle_camodel_cases.py"):
        new_manifest[name] = dict(path=name, sha256=digest(ROOT / name))
    combined = {entry["path"]: entry for entry in manifest}
    combined.update(new_manifest)
    (ROOT / "metadata/manifest.json").write_text(json.dumps(list(combined.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (ROOT / "metadata/data.json").write_text(serialized, encoding="utf-8")
    (ROOT / "metadata/data.js").write_text("window.MEMBAR_REPORT=" + serialized + ";\n", encoding="utf-8")
    (ROOT / "metadata/camodel_cases.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {len(reports)} variants; {len(new_manifest)} bundled file hashes", flush=True)


if __name__ == "__main__":
    main()
