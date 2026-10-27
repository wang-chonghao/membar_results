"""Publish the offline HTML as a Release asset, keeping credentials in memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request


def token_for(git):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run([git, "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        env=dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0"))
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not credentials.get("password"):
        raise RuntimeError("No GitHub API credential available; sign in using a credential helper or GH_TOKEN")
    return credentials["password"]


class UploadStream:
    def __init__(self, stream, size):
        self.stream, self.size, self.sent, self.reported = stream, size, 0, 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.sent += len(block)
        if self.sent - self.reported >= 32 * 1024 * 1024 or self.sent == self.size:
            print(f"Uploaded {self.sent / 2**20:.1f} / {self.size / 2**20:.1f} MiB", flush=True)
            self.reported = self.sent
        return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="wang-chonghao/membar_results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--credential-git", default="git")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        raise ValueError("Expected an owner/repository and a full commit SHA")
    token = token_for(args.credential_git)
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "User-Agent": "membar-results-release-publisher", "X-GitHub-Api-Version": "2022-11-28"}
    base = "https://api.github.com/repos/" + args.repo

    def request(url, method="GET", payload=None, data=None, extra_headers=None):
        current = dict(headers, **(extra_headers or {}))
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            current["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=current, method=method)
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    repository = request(base)
    if not repository.get("permissions", {}).get("push"):
        raise RuntimeError("Authenticated account cannot publish releases in this repository")
    remote_commit = request(base + "/git/ref/heads/main")["object"]["sha"]
    if remote_commit != args.commit:
        raise RuntimeError(f"Remote main is {remote_commit}, not the intended source commit")
    if args.preflight:
        print(json.dumps(dict(repo=args.repo, release_permission=True, remote_commit=remote_commit)))
        return
    with args.html.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        release = request(base + "/releases/tags/" + urllib.parse.quote(args.tag, safe=""))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        release = request(base + "/releases", "POST", payload=dict(
            tag_name=args.tag, target_commitish=args.commit,
            name="Membar results 2026-10-27 · A5/A6 与 Spill DAG",
            body=args.notes.read_text() + f"\n\nSource commit: `{args.commit}`\nHTML SHA256: `{digest}`\n",
            draft=True, prerelease=False))
    if not release["draft"] or release["target_commitish"] != args.commit:
        raise RuntimeError("Refuse to replace an existing published or unrelated release")
    existing = {asset["name"]: asset for asset in release["assets"]}
    upload_url = release["upload_url"].split("{", 1)[0]
    if urllib.parse.urlsplit(upload_url).hostname != "uploads.github.com":
        raise RuntimeError("Unexpected release upload host")

    def upload(name, size, checksum, data, content_type):
        if name in existing:
            asset = existing[name]
        else:
            asset = request(upload_url + "?" + urllib.parse.urlencode(dict(name=name)), "POST", data=data,
                            extra_headers={"Content-Type": content_type, "Content-Length": str(size)})
        if asset["state"] != "uploaded" or asset["size"] != size or asset.get("digest") != "sha256:" + checksum:
            raise RuntimeError("Uploaded asset size/digest differs from the local file")
        return asset

    with args.html.open("rb") as stream:
        html_asset = upload(args.html.name, args.html.stat().st_size, digest,
                            UploadStream(stream, args.html.stat().st_size), "text/html")
    checksums = f"{digest}  {args.html.name}\n".encode()
    checksum_asset = upload("SHA256SUMS.txt", len(checksums), hashlib.sha256(checksums).hexdigest(),
                            checksums, "text/plain")
    release = request(base + "/releases/" + str(release["id"]), "PATCH", payload=dict(draft=False, make_latest="true"))
    record = dict(release_url=release["html_url"], tag=args.tag, source_commit=args.commit,
                  html_url=html_asset["browser_download_url"], html_bytes=html_asset["size"], html_sha256=digest,
                  checksum_url=checksum_asset["browser_download_url"], published=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
