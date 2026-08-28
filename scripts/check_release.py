#!/usr/bin/env python3
"""Run lightweight release checks without loading models or dataset pickles."""

from __future__ import annotations

import argparse
import py_compile
import tempfile
from pathlib import Path


REQUIRED = (
    "README.md",
    "LICENSE",
    "MODEL_CARD.md",
    "CITATION.cff",
    "requirements.txt",
    "scripts/build_protocol_pools.py",
    "scripts/build_graphhard_pools.py",
    "scripts/build_magrpo_coordinator_dataset.py",
    "scripts/train_magrpo_coordinator.py",
    "scripts/eval_magrpo_coordinator_graphhard.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    missing = [value for value in REQUIRED if not (root / value).is_file()]
    if missing:
        raise SystemExit(f"missing release files: {missing}")

    checker_path = Path(__file__).resolve()
    forbidden = (
        "/Users/",
        "/home/",
        "/data1/",
        "192.168.",
        "BEGIN OPENSSH " + "PRIVATE KEY",
        "OPENROUTER_" + "API_KEY",
    )
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.resolve() == checker_path
            or path.suffix.lower() not in {".py", ".md", ".txt", ".cff"}
        ):
            continue
        text = path.read_text(errors="replace")
        for marker in forbidden:
            if marker in text:
                findings.append(f"{path.relative_to(root)}: {marker}")
    if findings:
        raise SystemExit("private release markers found:\n" + "\n".join(findings))

    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".pyc"}
    forbidden_names = {".DS_Store", ".env"}
    forbidden_files = [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and (path.suffix.lower() in forbidden_suffixes or path.name in forbidden_names)
    ]
    if forbidden_files:
        raise SystemExit("non-source release files found:\n" + "\n".join(forbidden_files))

    # Compile into a temporary directory so the release tree stays cache-free.
    with tempfile.TemporaryDirectory(prefix="camarl-release-check-") as temp_dir:
        for index, path in enumerate(sorted((root / "scripts").glob("*.py"))):
            cfile = Path(temp_dir) / f"{index}.pyc"
            py_compile.compile(str(path), cfile=str(cfile), doraise=True)

    placeholders = []
    for path in (root / "README.md", root / "MODEL_CARD.md", root / "CITATION.cff"):
        if "REPLACE_WITH_" in path.read_text():
            placeholders.append(str(path.relative_to(root)))
    print(f"release checks passed for {root}")
    if placeholders:
        print("publication placeholders remain in: " + ", ".join(placeholders))


if __name__ == "__main__":
    main()
