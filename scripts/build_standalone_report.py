"""Bundle the report and its archived attachments into one offline HTML file."""

import argparse
import base64
import gzip
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import mimetypes
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit


ENTRY = "spill_candidate_alignment_summary.html"
OUTPUT = "spill_candidate_alignment_standalone.html"
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def json_script(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def local_path(reference, owner=ENTRY):
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    parts = []
    for part in (Path(owner).parent / unquote(parsed.path)).parts:
        if part == "..":
            if not parts:
                raise ValueError(f"Attachment escapes the report directory: {reference}")
            parts.pop()
        elif part not in (".", "/"):
            parts.append(part)
    return "/".join(parts)


class PageParser(HTMLParser):
    def __init__(self, page, owner=ENTRY):
        super().__init__(convert_charrefs=False)
        self.owner = owner
        self.references = set()
        self.scripts = []
        self.styles = []
        self.parts = []
        self.skipped_script = False
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for key in ("src", "href", "poster"):
            path = local_path(values.get(key, ""), self.owner)
            if path:
                self.references.add(path)
        if tag == "script" and values.get("src"):
            self.scripts.append(local_path(values["src"], self.owner))
            if self.scripts[-1] is None:
                raise ValueError("Standalone reports cannot load external scripts")
            self.skipped_script = True
        elif tag == "link" and values.get("rel") == "stylesheet":
            path = local_path(values["href"], self.owner)
            if path is None:
                raise ValueError("Standalone reports cannot load external stylesheets")
            self.styles.append(path)
            self.parts.append(f"<!-- embedded-style:{escape(path)} -->")
        else:
            self.parts.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "script" and self.skipped_script:
            self.skipped_script = False
        else:
            self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skipped_script:
            self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_decl(self, decl):
        self.parts.append(f"<!{decl}>")

    def handle_comment(self, comment):
        self.parts.append(f"<!--{comment}-->")


def collect_files(root):
    resolved_root = root.resolve()
    resolved_parents = {}
    manifest = json.loads((root / "metadata/spill_candidate_manifest.json").read_text())
    paths = {item["path"] for item in manifest}
    paths.add("metadata/spill_candidate_manifest.json")
    for folder in ("figure", "metadata", "scripts", "assets", "multisoc_refresh_20261025", "replay_recompile_20261025", "a6_software_search_20261025", "a6_compiler_validation_20261027", "a5_software_search_20261026", "logged_vector_replay_20261026"):
        paths.update(p.relative_to(root).as_posix() for p in (root / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    rows = json.loads((root / "metadata/spill_candidate_all_versions.json").read_text())
    for row in rows:
        for group in row.get("files", {}).values():
            paths.update(item["path"] for item in group)
        if row.get("evidence_path"):
            paths.add(row["evidence_path"])
    paths.add(ENTRY)
    pending, parsed = list(paths), set()
    while pending:
        name = pending.pop()
        if name == OUTPUT or "__pycache__" in Path(name).parts:
            paths.discard(name)
            continue
        # Do not follow links outside this archive, even through symlinks.
        file = root / name
        parent = resolved_parents.get(file.parent)
        if parent is None:
            parent = resolved_parents[file.parent] = file.parent.resolve()
        resolved = file.resolve() if file.is_symlink() else parent / file.name
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"Attachment outside archive: {name}")
        if not file.is_file():
            raise FileNotFoundError(f"Missing attachment: {name}")
        if file.suffix.lower() == ".html" and name not in parsed:
            parsed.add(name)
            for other in PageParser(file.read_text(encoding="utf-8"), name).references:
                if other not in paths:
                    paths.add(other)
                    pending.append(other)
    return sorted(paths)


def mime_type(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".cce", ".cpp", ".h", ".py", ".sh", ".dump", ".log", ".txt", ".md", ".csv", ".mlir"}:
        return "text/plain; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def lean_files(root, paths):
    """Keep the browser experience and source links, not full experiment dumps."""
    selected_sources = set()

    def sources(value):
        if isinstance(value, dict):
            for item in value.values():
                sources(item)
        elif isinstance(value, list):
            for item in value:
                sources(item)
        elif isinstance(value, str) and value.endswith(".cce"):
            selected_sources.add(value)

    for name in ("metadata/software_cce_links.json", "metadata/spill_performance_summary.json"):
        path = root / name
        if path.exists():
            sources(json.loads(path.read_text()))
    selected_folders = {str(Path(path).parent) + "/" for path in selected_sources}
    dataset = {"metadata/data.json", "metadata/spill_candidate_alignment.json",
               "metadata/spill_performance_summary.json", "metadata/multisoc_predictions.json",
               "metadata/replay_validation.json", "metadata/software_cce_links.json"}
    mandatory = {ENTRY}
    for name in paths:
        if Path(name).suffix.lower() == ".html":
            for reference in PageParser((root / name).read_text(encoding="utf-8"), name).references:
                if Path(reference).suffix.lower() in {".js", ".css", ".html", ".png", ".svg", ".jpg", ".jpeg", ".webp"}:
                    mandatory.add(reference)
    kept, omitted = [], {}
    for name in paths:
        path = Path(name)
        suffix = path.suffix.lower()
        search = name.startswith(("a5_software_search_", "a6_software_search_"))
        intermediate = search and len(path.parts) > 3 and not any(name.startswith(p) for p in selected_folders)
        key_record = (suffix == ".json" and (root / name).stat().st_size <= 256 * 1024
                      and any(word in path.stem.lower() for word in
                              ("result", "validation", "audit", "provenance", "comparison", "summary", "verification", ".view")))
        include = (name in mandatory or name in dataset
                   or suffix in {".js", ".css", ".html", ".md", ".csv"}
                   or name.startswith("figure/") and suffix == ".png"
                   or not intermediate and (suffix in {".cce", ".cpp", ".h", ".sh"} or key_record))
        if include:
            kept.append(name)
        else:
            omitted[name] = ("中间寻优候选" if intermediate else
                             "完整日志或编译产物" if suffix in {".dump", ".log", ".o", ".bin", ".binary", ".gz", ".text"}
                             else "完整复现或详细核验数据")
    return kept, omitted


def build(root, output, mode="full"):
    if mode not in {"full", "lean"}:
        raise ValueError(f"Unsupported bundle mode: {mode}")
    paths = collect_files(root)
    omitted = {}
    if mode == "lean":
        paths, omitted = lean_files(root, paths)
    page = PageParser((root / ENTRY).read_text(encoding="utf-8"))
    document = "".join(page.parts)
    for style in page.styles:
        css = (root / style).read_text(encoding="utf-8")
        if re.search(r"url\s*\(|@import\b", css, re.I):
            raise ValueError(f"CSS resource needs explicit bundling: {style}")
        document = document.replace(f"<!-- embedded-style:{escape(style)} -->", f"<style>{css}</style>")
    document = re.sub(r"</body>\s*</html>\s*$", "", document, flags=re.I)
    css = (ASSETS / "standalone_report.css").read_text(encoding="utf-8")
    runtime = (ASSETS / "standalone_report.js").read_text(encoding="utf-8")
    index, blobs = {}, {}
    raw_total = packed_total = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=".standalone-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(document)
            stream.write(f"\n<style>{css}</style>\n")
            for number, path in enumerate(paths, 1):
                data = (root / path).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                raw_total += len(data)
                if digest not in blobs:
                    ident = f"embedded-{len(blobs)}"
                    packed = gzip.compress(data, compresslevel=6, mtime=0)
                    blobs[digest] = ident
                    packed_total += len(packed)
                    stream.write(f'<script type="application/octet-stream" id="{ident}">')
                    stream.write(base64.b64encode(packed).decode("ascii"))
                    stream.write("</script>\n")
                index[path] = dict(blob=blobs[digest], size=len(data), sha256=digest, mime=mime_type(path))
                if number % 1000 == 0:
                    print(f"Packed {number}/{len(paths)} files; {packed_total / 2**20:.1f} MiB compressed", flush=True)
            info = dict(entry=ENTRY, scripts=page.scripts, files=index, file_count=len(index),
                        bundle_mode=mode, omitted=omitted,
                        unique_blobs=len(blobs), raw_bytes=raw_total, compressed_bytes=packed_total)
            stream.write('<script type="application/json" id="embedded-index">' + json_script(info) + "</script>\n")
            stream.write("<script>" + runtime + "</script>\n</body></html>\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    print(json.dumps(dict(output=str(output), files=len(index), unique_blobs=len(blobs), raw_mib=round(raw_total / 2**20, 2), html_mib=round(output.stat().st_size / 2**20, 2)), ensure_ascii=False), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("full", "lean"), default="full")
    args = parser.parse_args()
    build(args.root, args.output or args.root / OUTPUT, mode=args.mode)


if __name__ == "__main__":
    main()
