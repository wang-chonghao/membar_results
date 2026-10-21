"""Check that an explicit-spill case carries its own runnable environment."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from package_explicit_spill_cases import package_case


class StandaloneCaseTests(unittest.TestCase):
    def make_case(self, root):
        folder = root / "example/u2/explicit_spill_camodel_case"
        (folder / "source").mkdir(parents=True)
        (folder / "run_output").mkdir()
        config = dict(kernel_source="source/kernel.cce", kernel_name="example",
                      host_args=["64"], host_defines=[], ccec_flags=[])
        comparison = dict(golden=dict(passed=True),
                          normalized_dynamic_dataflow_and_addresses_equal=True,
                          output_checks=[dict(byte_equal=True)],
                          original_cycles=100, explicit_cycles=101)
        (folder / "case.json").write_text(json.dumps(config))
        (folder / "comparison.json").write_text(json.dumps(comparison))
        for name in ("source/kernel.cce", "source/original_pc_mapping.json",
                     "run_output/validation.json", "host.cpp"):
            (folder / name).write_text("{}")
        (folder / "run.py").write_text(
            'import json, os, sys\n'
            'print(json.dumps([os.environ["PACKAGE_TEST"], *sys.argv[1:]]))\n')
        (root / "set_env.sh").write_text('export PACKAGE_TEST="local environment"\n')
        return folder

    def test_copied_case_runs_outside_archive_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.make_case(root)
            package_case(root, "example", "u2")
            copied = root / "standalone folder"
            shutil.copytree(folder, copied)
            shutil.rmtree(root / "example")
            (root / "set_env.sh").unlink()
            result = subprocess.run(
                ["bash", str(copied / "run.sh"), "--out-dir", "output with spaces"],
                cwd="/tmp", check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout),
                             ["local environment", "--out-dir", "output with spaces"])
            self.assertTrue((copied / "README.txt").is_file())
            summary = json.loads((copied / "evidence/summary.json").read_text())
            self.assertEqual(summary["explicit_cycles"], 101)

    def test_failed_validation_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.make_case(root)
            comparison = json.loads((folder / "comparison.json").read_text())
            comparison["golden"]["passed"] = False
            (folder / "comparison.json").write_text(json.dumps(comparison))
            with self.assertRaises(ValueError):
                package_case(root, "example", "u2")
            self.assertFalse((folder / "run.sh").exists())

    def test_missing_host_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.make_case(root)
            (folder / "host.cpp").unlink()
            with self.assertRaises(FileNotFoundError):
                package_case(root, "example", "u2")

    def test_no_membar_variant_cannot_inherit_archived_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.make_case(root)
            config = json.loads((folder / "case.json").read_text())
            config["membar_disabled"] = True
            (folder / "case.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                package_case(root, "example", "u2")


if __name__ == "__main__":
    unittest.main()
