"""调用现有 run_test.py 运行 A6 Softmax，并强制执行独立 FP16 golden 检查。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from check_softmax import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', type=Path, required=True, help='microbenchmark-cce-direct/run_test.py')
    parser.add_argument('--variant', choices=('no_membar', 'membar'), default='no_membar')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cann-home', type=Path)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.compile_only and args.check_only:
        parser.error('--compile-only 和 --check-only 不能同时使用')
    runner = args.runner.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    kernel_name = 'softmax_8x1024_a6' + ('_membar' if args.variant == 'membar' else '')
    kernel = Path(__file__).resolve().parent / f'{kernel_name}.cce'
    if not args.check_only:
        if not runner.is_file() or not kernel.is_file():
            parser.error('runner 或完整 CCE 文件不存在')
        if destination.exists() and any(destination.iterdir()):
            parser.error('输出目录必须为空或尚不存在，避免把历史文件误当作新运行结果')
        command = [sys.executable, str(runner), '--kernel', str(kernel),
                   '--output', str(destination), '--soc-version', 'Ascend910_9691',
                   '--core-arch', 'dav-920r1-vec', '--core-sim-dir', 'dav_9201',
                   '--block-dim', '1', '--golden', 'none']
        if args.cann_home:
            command += ['--cann-home', str(args.cann_home.expanduser().resolve())]
        if args.compile_only:
            command += ['--compile-only']
        environment = dict(os.environ, SOC_VERSION='Ascend910_9691', NPU_TYPE='Ascend910_9691',
                           CORE_ARCH='dav-920r1-vec', CORE_SIM_DIR='dav_9201')
        # 外部 runner 的 FULL_SIMULATOR_HOME 只配置运行时；独立补齐 host 链接搜索目录。
        simulator = environment.get('FULL_SIMULATOR_HOME')
        if simulator:
            simulator = Path(simulator).expanduser().resolve()
            libraries = [simulator / soc / folder for soc in ('Ascend910_9691', 'dav_9201')
                         for folder in ('lib', 'camodel')]
            available = [str(path) for path in libraries if path.is_dir()]
            if available:
                environment['LIBRARY_PATH'] = ':'.join(available+[environment.get('LIBRARY_PATH', '')])
        subprocess.run(command, cwd=runner.parent, env=environment, check=True)
    if args.compile_only:
        print('编译完成；未执行 CAModel，也未通过数值校验。')
        return 0
    run_dir = destination / 'run'
    report = check(run_dir / 'input_input.bin', run_dir / 'output_output.bin')
    report.update(variant=args.variant, kernel=kernel.name)
    (run_dir / 'softmax_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"[CHECK] {'PASS' if report['passed'] else 'FAIL'} softmax: "
          f"elements={report['elements']} mismatches={report['mismatches']}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'[ERROR] {error}', file=sys.stderr)
        raise SystemExit(1)
