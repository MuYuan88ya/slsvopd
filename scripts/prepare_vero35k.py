#!/usr/bin/env python3
"""Prepare Vero-35K metadata and images for Vision-OPD training.

The safe default is the ``metadata`` command, which downloads only the VeRL
annotation JSONL and small descriptive files.  Image archives are fetched only
by the explicit ``materialize`` command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "muyuho/Vero-35k-VisionSelector"
SOURCE_JSONL = "vero_train_all_35k.verl.jsonl"
OUTPUT_JSONL = "train.verl.jsonl"
METADATA_FILES = (SOURCE_JSONL, "README.md", "summary.json")
IMAGE_PART_GLOB = "images.tar.gz.part_*"


def download_metadata(repo_id: str, data_dir: Path) -> Path:
    """Download only training annotations and small dataset information files."""

    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(data_dir),
            allow_patterns=list(METADATA_FILES),
        )
    )


def download_image_parts(repo_id: str, data_dir: Path) -> list[Path]:
    """Explicitly download the multi-part image archive."""

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=[IMAGE_PART_GLOB],
    )
    parts = sorted(data_dir.glob(IMAGE_PART_GLOB))
    if not parts:
        raise FileNotFoundError(f"No {IMAGE_PART_GLOB} files were downloaded into {data_dir}")
    return parts


def extract_image_parts(parts: list[Path], data_dir: Path) -> None:
    """Stream concatenated gzip parts into tar without creating another archive copy."""

    process = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(data_dir)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, process.stdin, length=8 * 1024 * 1024)
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


def _normalize_image(image: Any, dataset_root: Path, line_number: int) -> dict[str, Any]:
    if isinstance(image, str):
        payload: dict[str, Any] = {"path": image}
    elif isinstance(image, dict):
        payload = dict(image)
    else:
        raise TypeError(f"line {line_number}: image must be a path string or mapping, got {type(image)}")

    image_path = payload.get("path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"line {line_number}: image entry is missing a non-empty path")
    resolved = Path(image_path).expanduser()
    if not resolved.is_absolute():
        resolved = dataset_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"line {line_number}: image not found: {resolved}")
    payload["path"] = str(resolved)
    return payload


def build_training_jsonl(
    source_jsonl: Path,
    output_jsonl: Path,
    dataset_root: Path,
    limit: int | None = None,
) -> int:
    """Keep the VeRL JSONL schema and resolve image paths for the training host."""

    source_jsonl = source_jsonl.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    output_jsonl = output_jsonl.expanduser().resolve()
    if source_jsonl == output_jsonl:
        raise ValueError("source and output JSONL paths must be different")
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"Source annotation file not found: {source_jsonl}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    count = 0
    try:
        with source_jsonl.open(encoding="utf-8") as source, temporary_path.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError(f"line {line_number}: each JSONL record must be an object")
                prompt = item.get("prompt")
                images = item.get("images")
                if not isinstance(prompt, list) or not prompt:
                    raise ValueError(f"line {line_number}: prompt must be a non-empty list")
                if not isinstance(images, list) or not images:
                    raise ValueError(f"line {line_number}: images must be a non-empty list")
                item["images"] = [
                    _normalize_image(image, dataset_root, line_number) for image in images
                ]
                output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
        os.replace(temporary_path, output_jsonl)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("./data/vero35k"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("metadata", help="download annotations only; never downloads images")
    _common_parser(metadata)

    materialize = subparsers.add_parser(
        "materialize",
        help="explicitly download and extract all image archive parts, then build training JSONL",
    )
    _common_parser(materialize)
    materialize.add_argument("--keep-archives", action="store_true")
    materialize.add_argument("--output", type=Path)
    materialize.add_argument("--limit", type=int)

    build = subparsers.add_parser("build", help="build training JSONL from existing metadata and images")
    _common_parser(build)
    build.add_argument("--source", type=Path)
    build.add_argument("--dataset-root", type=Path)
    build.add_argument("--output", type=Path)
    build.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if args.command == "metadata":
        download_metadata(args.repo_id, data_dir)
        print(f"Metadata ready at {data_dir}; no image archives were downloaded.")
        return

    if args.command == "materialize":
        download_metadata(args.repo_id, data_dir)
        parts = download_image_parts(args.repo_id, data_dir)
        print(f"Extracting {len(parts)} image archive parts into {data_dir} ...")
        extract_image_parts(parts, data_dir)
        source_jsonl = data_dir / SOURCE_JSONL
        output_jsonl = args.output or data_dir / OUTPUT_JSONL
        count = build_training_jsonl(source_jsonl, output_jsonl, data_dir, limit=args.limit)
        if not args.keep_archives:
            for part in parts:
                part.unlink()
        print(f"Prepared {count} training records at {output_jsonl.resolve()}")
        return

    source_jsonl = args.source or data_dir / SOURCE_JSONL
    dataset_root = args.dataset_root or data_dir
    output_jsonl = args.output or data_dir / OUTPUT_JSONL
    count = build_training_jsonl(source_jsonl, output_jsonl, dataset_root, limit=args.limit)
    print(f"Prepared {count} training records at {output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
