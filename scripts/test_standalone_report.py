"""Check single-file packing without depending on the large experiment archive."""

import base64
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from build_standalone_report import build, collect_files, json_script, lean_files, local_path


class PayloadParser(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.payloads = {}
        self.current = None
        self.feed(page)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id", "").startswith("embedded-"):
            self.current = attrs["id"]
            self.payloads[self.current] = ""

    def handle_data(self, value):
        if self.current:
            self.payloads[self.current] += value

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


class SingleFileTests(unittest.TestCase):
    def fixture(self, root):
        (root / "metadata").mkdir()
        (root / "assets").mkdir()
        files = {
            "spill_candidate_alignment_summary.html": '<!doctype html><html><head><link rel="stylesheet" href="assets/test.css"><script defer src="assets/test.js"></script></head><body><header><nav></nav></header><a href="code.cce">code</a><a href="child.html">child</a></body></html>',
            "assets/test.js": 'window.fixtureLoaded = document.body !== null;',
            "assets/test.css": "body {color: black;}",
            "code.cce": "// <script>not executable</script>\nvoid test() {}\n",
            "same.cce": "// <script>not executable</script>\nvoid test() {}\n",
            "child.html": '<html><body><img src="image.svg"><a href="code.cce">source</a></body></html>',
            "image.svg": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="red"/></svg>',
            "metadata/spill_candidate_all_versions.json": "[]",
            "golden.bin": bytes(range(256)),
        }
        for name, content in files.items():
            (root / name).write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
        (root / "metadata/spill_candidate_manifest.json").write_text(json.dumps([dict(path=p) for p in files]), encoding="utf-8")

    def test_roundtrip_deduplication_and_recursive_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            output = root / "single.html"
            index = build(root, output)
            parsed = PayloadParser(output.read_text())
            self.assertIn("image.svg", index["files"])
            self.assertEqual(index["files"]["same.cce"]["blob"], index["files"]["code.cce"]["blob"])
            self.assertNotIn('<script defer src="assets/test.js">', output.read_text())
            self.assertNotIn('<link rel="stylesheet"', output.read_text())
            for name, item in index["files"].items():
                actual = gzip.decompress(base64.b64decode(parsed.payloads[item["blob"]]))
                self.assertEqual(actual, (root / name).read_bytes())
                self.assertEqual(hashlib.sha256(actual).hexdigest(), item["sha256"])
            self.assertEqual(json.loads(parsed.payloads["embedded-index"]), index)

    def test_escape_and_path_contract(self):
        self.assertNotIn("<", json_script({"unsafe": "</script><script>alert(1)</script>"}))
        self.assertEqual(local_path("../code%20file.cce?x=1", "docs/readme.html"), "code file.cce")
        self.assertIsNone(local_path("https://example.com/code.cce"))
        self.assertIsNone(local_path("#section"))
        with self.assertRaises(ValueError):
            local_path("../../private.txt")

    def test_missing_evidence_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / "code.cce").unlink()
            with self.assertRaises(FileNotFoundError):
                collect_files(root)

    def test_lean_keeps_ui_and_source_but_explicitly_omits_binaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            index = build(root, root / "lean.html", mode="lean")
            self.assertEqual(index["bundle_mode"], "lean")
            for name in ("code.cce", "child.html", "assets/test.js", "assets/test.css", "image.svg"):
                self.assertIn(name, index["files"])
            self.assertNotIn("golden.bin", index["files"])
            self.assertIn("golden.bin", index["omitted"])

    def test_lean_excludes_intermediate_search_but_keeps_selected_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            selected = "a5_software_search_20261026/case/camodel_trial_02/cce/kernel.cce"
            rejected = "a5_software_search_20261026/case/camodel_trial_01/cce/kernel.cce"
            (root / "metadata/software_cce_links.json").write_text(json.dumps(dict(source=selected)))
            for name in (selected, rejected):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// candidate\n")
            kept, omitted = lean_files(root, [selected, rejected])
            self.assertEqual(kept, [selected])
            self.assertEqual(omitted[rejected], "中间寻优候选")


if __name__ == "__main__":
    unittest.main()
