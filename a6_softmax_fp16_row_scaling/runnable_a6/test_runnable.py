"""离线检查 golden、runner 参数及完整 kernel 的接口契约。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from check_softmax import check
import run_softmax_a6

HERE = Path(__file__).resolve().parent


class GoldenTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / 'input.bin'
        self.output = self.root / 'output.bin'

    def values(self, inputs, outputs):
        self.input.write_bytes(struct.pack('<'+'f'*len(inputs), *inputs))
        self.output.write_bytes(struct.pack('<'+'e'*len(outputs), *outputs))

    def test_uniform(self):
        self.values([0.0]*8, [0.25]*8)
        self.assertTrue(check(self.input, self.output, 2, 4)['passed'])

    def test_row_local_and_stable(self):
        self.values([1000, 1000, -1000, -1000], [0.5]*4)
        self.assertTrue(check(self.input, self.output, 2, 2)['passed'])

    def test_nonuniform(self):
        self.values([0, math.log(2), 0, 0], [1/3, 2/3, 0.5, 0.5])
        self.assertTrue(check(self.input, self.output, 2, 2)['passed'])

    def test_wrong_global_normalization_fails(self):
        self.values([0]*4, [0.25]*4)
        self.assertEqual(check(self.input, self.output, 2, 2)['mismatches'], 4)

    def test_nan_output_fails(self):
        self.values([0, 0], [float('nan'), 0.5])
        report = check(self.input, self.output, 1, 2)
        self.assertFalse(report['passed'])
        json.dumps(report, allow_nan=False)

    def test_nonfinite_input_rejected(self):
        self.values([float('inf'), 0], [0.5, 0.5])
        with self.assertRaises(ValueError):
            check(self.input, self.output, 1, 2)

    def test_truncated_output_rejected(self):
        self.values([0, 0], [0.5])
        with self.assertRaises(ValueError):
            check(self.input, self.output, 1, 2)

    def test_wrong_fp32_output_size_rejected(self):
        self.values([0, 0], [0.5, 0.5])
        self.output.write_bytes(struct.pack('<ff', 0.5, 0.5))
        with self.assertRaises(ValueError):
            check(self.input, self.output, 1, 2)

    def test_untouched_output_fails(self):
        self.values([0, 0], [0.5, 0.5])
        self.output.write_bytes(b'\xcd'*4)
        self.assertFalse(check(self.input, self.output, 1, 2)['passed'])

    def test_invalid_configuration_rejected(self):
        for extra in ({'rows': 0}, {'atol': -1}, {'rtol': float('nan')}):
            with self.assertRaises(ValueError):
                check(self.input, self.output, **extra)


class BundleTest(unittest.TestCase):
    def test_shape_kernels_preserve_vf_and_ub_layout(self):
        catalog = json.loads((HERE / 'cases.json').read_text())
        cases = catalog['cases']
        self.assertEqual({(c['shape'][0], c['variant']) for c in cases},
                         {(r, v) for r in (8, 12, 16, 24) for v in ('no_membar', 'three_loop')})
        self.assertEqual(len(cases), 8)
        for case in cases:
            with self.subTest(kernel=case['kernel']):
                code = (HERE / case['kernel']).read_bytes()
                original = (HERE / case['vf_source']).read_bytes()
                self.assertIn(original, code)
                self.assertEqual(hashlib.sha256(code).hexdigest(), case['sha256'])
                self.assertEqual(hashlib.sha256(original).hexdigest(), case['vf_source_sha256'])
                rows, cols = case['shape']
                self.assertIn(f'constexpr uint32_t elements = {rows} * {cols};'.encode(), code)
                self.assertEqual(code.count(b'mem_bar('), case['static_membar_count'])
                self.assertIn(b'get_block_idx() != 0', code)
                regions = case['ub_regions']
                self.assertEqual(regions[0]['start'], 0)
                for before, after in zip(regions, regions[1:]):
                    self.assertEqual(before['end'], after['start'])
                self.assertEqual(regions[0]['end'], rows * cols * 4)
                self.assertEqual(regions[2]['end'] - regions[2]['start'],
                                 rows * 8 if case['variant'] == 'three_loop' else 0)
                self.assertEqual(regions[3]['end'] - regions[3]['start'], rows * cols * 2)
                self.assertEqual(regions[-1]['end'], case['ub_reserved_bytes'])
                self.assertLessEqual(case['ub_reserved_bytes'], catalog['ub_budget_bytes'])
                for name in ('memA', 'memB', 'memOutput', 'offsets'):
                    region_name = 'memB_exp' if name == 'memB' else name
                    start = next(r['start'] for r in regions if r['name'] == region_name)
                    match = re.search(rf'\* {name} = .*get_imm\((0x[0-9a-f]+)\)', code.decode())
                    self.assertIsNotNone(match)
                    self.assertEqual(int(match.group(1), 16), start)
                self.assertIn(b'outputBytes = elements * sizeof(half)', code)
                self.assertIn(b'0, 0, offsets, 0, outputBytes, 0, 1', code)

    def test_provenance_and_original_vf(self):
        provenance = json.loads((HERE / 'source_provenance.json').read_text())
        original = (HERE.parent / 'rows8/gap2/source/kernel.cce').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), provenance['vf_source_sha256'])
        for record in provenance['artifacts']:
            code = (HERE / record['kernel']).read_bytes()
            self.assertEqual(hashlib.sha256(code).hexdigest(), record['sha256'])
            if not record['internal_membar']:
                self.assertIn(original, code)
            self.assertIn(b'nddma_out_to_ub(memA, input', code)
            self.assertIn(b'outputBytes = elements * sizeof(half)', code)
            self.assertIn(b'0, 0, offsets, 0, outputBytes, 0, 1', code)
            self.assertEqual(code.count(b'mem_bar(VST_VLD)'), 4 if record['internal_membar'] else 0)

    def test_external_manifest_parser(self):
        path = os.environ.get('MICROBENCH_RUNNER')
        if not path:
            self.skipTest('设置 MICROBENCH_RUNNER 后验证外部 runner 契约')
        spec = importlib.util.spec_from_file_location('runnable_manifest_reader', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(module)
        for suffix in ('', '_membar'):
            name = 'softmax_8x1024_a6'+suffix
            manifest = module.parse_kernel_manifest(HERE / (name+'.cce'), 256)
            self.assertEqual(manifest.kernel_name, name)
            self.assertEqual([(t.direction, t.name, t.dtype, t.elements) for t in manifest.tensors],
                             [('input', 'input', 'fp32', 8192), ('output', 'output', 'fp16', 8192)])
            self.assertEqual(manifest.goldens, ())
        for case in json.loads((HERE / 'cases.json').read_text())['cases']:
            with self.subTest(kernel=case['kernel']):
                manifest = module.parse_kernel_manifest(HERE / case['kernel'], 256)
                self.assertEqual(manifest.kernel_name, case['kernel_name'])
                elements = math.prod(case['shape'])
                self.assertEqual([(t.direction, t.name, t.dtype, t.elements) for t in manifest.tensors],
                                 [('input', 'input', 'fp32', elements),
                                  ('output', 'output', 'fp16', elements)])
                self.assertEqual(manifest.goldens, ())

    def test_launcher_selects_all_shapes_and_variants(self):
        for rows in (8, 12, 16, 24):
            for variant in ('no_membar', 'three_loop'):
                with self.subTest(rows=rows, variant=variant), tempfile.TemporaryDirectory() as folder:
                    folder = Path(folder)
                    runner = folder / 'run_test.py'
                    runner.touch()
                    output = folder / 'result'

                    def run(command, **kwargs):
                        self.assertEqual(Path(command[command.index('--kernel') + 1]),
                                         HERE / f'rows{rows}/{variant}.cce')
                        self.assertEqual(command[command.index('--block-dim') + 1], '1')
                        run_dir = output / 'run'
                        run_dir.mkdir(parents=True)
                        (run_dir / 'input_input.bin').write_bytes(struct.pack('<f', 0) * rows * 1024)
                        (run_dir / 'output_output.bin').write_bytes(struct.pack('<e', 1/1024) * rows * 1024)

                    with patch.object(sys, 'argv', ['run_softmax_a6.py', '--runner', str(runner),
                                                   '--output', str(output), '--rows', str(rows),
                                                   '--variant', variant]), \
                            patch.object(run_softmax_a6.subprocess, 'run', side_effect=run), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(run_softmax_a6.main(), 0)
                    report = json.loads((output / 'run/softmax_check.json').read_text())
                    self.assertEqual(report['shape'], [rows, 1024])
                    self.assertTrue(report['passed'])

    def test_launcher_forces_a6_and_checks_fp16(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runner = folder / 'run_test.py'
            runner.touch()
            output = folder / 'result'

            def run(command, **kwargs):
                self.assertIn('--golden', command)
                self.assertEqual(command[command.index('--golden')+1], 'none')
                self.assertEqual(kwargs['env']['NPU_TYPE'], 'Ascend910_9691')
                run_dir = output / 'run'
                run_dir.mkdir(parents=True)
                (run_dir / 'input_input.bin').write_bytes(struct.pack('<f', 0)*8192)
                (run_dir / 'output_output.bin').write_bytes(struct.pack('<e', 1/1024)*8192)

            with patch.object(sys, 'argv', ['run_softmax_a6.py', '--runner', str(runner),
                                           '--output', str(output)]), \
                    patch.object(run_softmax_a6.subprocess, 'run', side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_softmax_a6.main(), 0)
            self.assertTrue(json.loads((output / 'run/softmax_check.json').read_text())['passed'])

    def test_launcher_rejects_wrong_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runner = folder / 'run_test.py'
            runner.touch()
            output = folder / 'result'

            def run(command, **kwargs):
                run_dir = output / 'run'
                run_dir.mkdir(parents=True)
                (run_dir / 'input_input.bin').write_bytes(struct.pack('<f', 0)*8192)
                (run_dir / 'output_output.bin').write_bytes(b'\xcd'*(8192*2))

            with patch.object(sys, 'argv', ['run_softmax_a6.py', '--runner', str(runner),
                                           '--output', str(output)]), \
                    patch.object(run_softmax_a6.subprocess, 'run', side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_softmax_a6.main(), 1)
            self.assertFalse(json.loads((output / 'run/softmax_check.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
