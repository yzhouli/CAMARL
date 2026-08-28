#!/usr/bin/env python3
"""Create auditable single-image topic inputs from CAMARL image/video media.

Static images are resized without upscaling.  Videos are represented by a 2x2
contact sheet sampled uniformly at 10%, 35%, 65%, and 90% of duration.  This
keeps the serving input genuinely visual while making repeated evaluation on
the same frozen topic deterministic and tractable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_image(source: Path, output: Path) -> tuple[str, float, str]:
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        mode, seconds = "static-resize", 0.0
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            return mode, seconds, "cv2.imread returned None"
        height, width = frame.shape[:2]
        scale = min(1.0, 672 / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok = cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return mode, seconds, "" if ok else "cv2.imwrite failed"
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            return "video-uniform-2x2-contact-sheet", 0.0, "VideoCapture failed"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        seconds = frame_count / fps if fps > 0 else 0.0
        frames = []
        for fraction in (0.10, 0.35, 0.65, 0.90):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((frame_count - 1) * fraction)))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(336 / max(1, width), 336 / max(1, height))
            resized = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.zeros((336, 336, 3), dtype=np.uint8)
            top = (336 - resized.shape[0]) // 2
            left = (336 - resized.shape[1]) // 2
            canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            frames.append(canvas)
        capture.release()
        mode = "video-uniform-2x2-contact-sheet"
        if not frames:
            return mode, seconds, "no decodable video frames"
        while len(frames) < 4:
            frames.append(frames[-1].copy())
        contact = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])
        ok = cv2.imwrite(str(output), contact, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return mode, seconds, "" if ok else "cv2.imwrite failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--graphhard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args()

    news = load_pickle(args.news)
    news_ids: set[str] = set()
    pool_files = sorted(
        path
        for split in args.splits
        for path in args.graphhard_dir.glob(f"{split}_graphhard_pools_N*.pkl")
    )
    for path in pool_files:
        for record in load_pickle(path):
            news_ids.add(str(record["news_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "method": "static resize or uniform 2x2 video contact sheet",
        "pool_files": {str(path): sha256_file(path) for path in pool_files},
        "records": {},
    }
    for completed, news_id in enumerate(sorted(news_ids), 1):
        relative = Path(str((news.get(news_id, {}) or {}).get("mm_path", "")))
        source = args.source_dir / relative.name
        output = args.output_dir / f"{relative.stem}.jpg"
        error = ""
        mode = "missing"
        seconds = 0.0
        if source.exists():
            if not output.exists():
                mode, seconds, error = make_image(source, output)
            else:
                mode = "existing"
                seconds = 0.0
        manifest["records"][news_id] = {
            "source": str(source),
            "source_exists": source.exists(),
            "source_sha256": sha256_file(source) if source.exists() else None,
            "output": str(output),
            "output_exists": output.exists(),
            "output_sha256": sha256_file(output) if output.exists() else None,
            "source_duration_seconds": seconds,
            "mode": mode,
            "error": error,
        }
        if completed % 25 == 0 or completed == len(news_ids):
            print(f"media {completed}/{len(news_ids)}", flush=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    failures = [key for key, value in manifest["records"].items() if not value["output_exists"]]
    print(f"saved {manifest_path}; topics={len(news_ids)} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(f"Missing outputs for {failures[:10]}")


if __name__ == "__main__":
    main()
