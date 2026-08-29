#!/usr/bin/env python3
"""Create a deterministic streaming file manifest for a dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "MANIFEST.sha256.json"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = (args.output or root / MANIFEST_NAME).resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != output
    )
    entries = []
    total_bytes = 0
    for index, path in enumerate(files, start=1):
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
        print(f"[{index}/{len(files)}] {path.relative_to(root)}", flush=True)

    payload = {
        "manifest_version": 1,
        "hash_algorithm": "sha256",
        "root_name": root.name,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output} ({len(entries)} files, {total_bytes} bytes)")


if __name__ == "__main__":
    main()
