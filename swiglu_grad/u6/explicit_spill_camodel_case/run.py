"""Build a bundled CAModel case in a new directory, or check archived outputs."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess


def check_outputs(folder, config):
    checks = []
    for expected_name, output_name in config["golden_pairs"]:
        expected_path, output_path = folder / expected_name, folder / output_name
        if not expected_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"Missing golden/output pair: {expected_path}, {output_path}")
        expected_bytes, output_bytes = expected_path.read_bytes(), output_path.read_bytes()
        if len(expected_bytes) != len(output_bytes) or len(expected_bytes) % 4:
            raise ValueError(f"Invalid FP32 output size: {output_path}")
        expected = struct.iter_unpack("<f", expected_bytes)
        actual = struct.iter_unpack("<f", output_bytes)
        mismatches, maximum = 0, 0.0
        for (want,), (got,) in zip(expected, actual):
            error = abs(got - want)
            valid = math.isfinite(want) and math.isfinite(got)
            mismatches += not valid or error > config["atol"] + config["rtol"] * abs(want)
            maximum = max(maximum, error if valid else math.inf)
        checks.append(dict(output=output_name, elements=len(output_bytes) // 4,
                           mismatches=mismatches, max_abs_error=maximum))
    if not checks:
        raise FileNotFoundError("No archived golden files are available for this case")
    if any(item["mismatches"] for item in checks):
        raise ValueError(f"Golden check failed: {checks}")
    return dict(passed=True, checks=checks)


def commands(case, output, toolkit, config):
    host_arch = os.environ.get("MEMBAR_HOST_ARCH") or "x86_64-linux"
    core_arch = os.environ.get("MEMBAR_CORE_ARCH") or "dav-c310-vec"
    includes = [toolkit / f"{host_arch}/include", toolkit / "include",
                toolkit / f"{host_arch}/include/experiment/msprof",
                toolkit / f"{host_arch}/include/experiment/msprof/toolchain",
                toolkit / f"{host_arch}/pkg_inc/profiling", toolkit / f"{host_arch}/pkg_inc",
                toolkit / f"{host_arch}/pkg_inc/runtime"]
    libraries = [toolkit / f"tools/simulator/{config['npu_type']}/lib", toolkit / "lib64",
                 toolkit / f"{host_arch}/lib64", toolkit / f"{host_arch}/devlib",
                 toolkit / f"{host_arch}/devlib/device", toolkit / f"{host_arch}/lib64/device/lib64",
                 toolkit / f"aarch64-linux/simulator/{config['npu_type']}/lib",
                 toolkit / f"simulator/{config['npu_type']}/lib"]
    libraries.extend(Path(p) for p in os.environ.get("MEMBAR_EXTRA_LIBRARY_DIRS", "").split(":") if p)
    overrides = {"ccec": "MEMBAR_CCEC", "bisheng": "MEMBAR_HOST_CXX", "ld.lld": "MEMBAR_LINKER"}

    def tool(name):
        return os.environ.get(overrides[name]) or str(next(
            (p for p in (toolkit / "bin" / name, toolkit / f"{host_arch}/bin" / name) if p.is_file()),
            toolkit / "bin" / name,
        ))
    host = [tool("bisheng"), "-std=c++17", "-O2", str(case / "host.cpp"), "-o", "host",
            "-Wl,--allow-shlib-undefined", *config["host_defines"],
            *(f"-I{p}" for p in includes), *(f"-L{p}" for p in libraries),
            "-lruntime_camodel", "-lstdc++", "-lascendcl", "-lm", "-ltiling_api",
            "-lplatform", "-lc_sec", "-ldl", "-lnnopbase"]
    compile_cmd = [tool("ccec"), "-g", "-std=c++17", "-c", "-O2",
                   str(case / config["kernel_source"]), "-o", "kernel_aiv.o",
                   *(f"-I{p}" for p in os.environ.get("MEMBAR_CXX_INCLUDE_DIRS", "").split(":") if p),
                   f"--cce-aicore-arch={core_arch}", "--cce-aicore-only",
                   *config["ccec_flags"]]
    link = [tool("ld.lld"), "-Ttext=0", "kernel_aiv.o", "-static", "-o", "kernel.o"]
    execute = [str(output / "host"), str(output / "kernel.o"), config["kernel_name"], *config["host_args"]]
    return [(host, "host_build.log"), (compile_cmd, "compile.log"), (link, "link.log"),
            (execute, "camodel.log")], libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show commands without compiling or running")
    parser.add_argument("--check-archived", action="store_true", help="Check existing outputs against saved golden")
    toolkit_default = os.environ.get("MEMBAR_CANN_ROOT") or os.environ.get("ACL_PATH")
    parser.add_argument("--toolkit", type=Path, default=Path(toolkit_default) if toolkit_default else None)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    config = json.loads((case / "case.json").read_text(encoding="utf-8"))
    if args.check_archived:
        print(json.dumps(check_outputs(case / "run_output", config), ensure_ascii=False))
        return
    if args.toolkit is None:
        parser.error("Source membar_results/set_env.sh first, or pass --toolkit <CANN directory>")
    config["npu_type"] = os.environ.get("MEMBAR_NPU_TYPE") or config["npu_type"]
    if args.out_dir is None and not args.dry_run:
        parser.error("--out-dir is required; use a new directory to preserve archived results")
    output = (args.out_dir or Path("/tmp/camodel-case-rerun")).resolve()
    if output == case or case in output.parents:
        parser.error("--out-dir must be outside the archived case directory")
    recipe, libraries = commands(case, output, args.toolkit.resolve(), config)
    if args.dry_run:
        print(json.dumps([dict(argv=cmd, log=log) for cmd, log in recipe], indent=2))
        return
    if output.exists() and any(output.iterdir()):
        parser.error("--out-dir must be empty or not yet exist")
    output.mkdir(parents=True, exist_ok=True)
    for name in config["input_files"]:
        shutil.copy2(case / "run_output" / name, output / name)
    env = dict(os.environ)
    env.update(ASCEND_TOOLKIT_HOME=str(args.toolkit.resolve()), ACL_PATH=str(args.toolkit.resolve()),
               NPU_TYPE=config["npu_type"])
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries)) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    (output / "commands.json").write_text(json.dumps([dict(argv=c, log=l) for c, l in recipe], indent=2))
    for command, log in recipe:
        with (output / log).open("w") as stream:
            subprocess.run(command, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
    if config["input_files"]:
        for expected, _ in config["golden_pairs"]:
            shutil.copy2(case / "run_output" / expected, output / expected)
    validation = check_outputs(output, config)
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
