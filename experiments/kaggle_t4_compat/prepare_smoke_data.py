#!/usr/bin/env python3
"""Create a tiny Vero-shaped multimodal dataset without external image downloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_vero35k.py"


def _load_vero_converter():
    spec = importlib.util.spec_from_file_location("prepare_vero35k", PREPARE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PREPARE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for index, (name, rgb, answer, budget) in enumerate(
        (
            ("red.png", (220, 30, 30), "red", 0.2),
            ("blue.png", (30, 30, 220), "blue", 0.15),
        )
    ):
        image_path = images_dir / name
        Image.new("RGB", (224, 224), rgb).save(image_path)
        samples.append(
            {
                "id": f"t4-smoke-{index}",
                "data_source": "vision_selector_t4_smoke",
                "prompt": [
                    {
                        "role": "user",
                        "content": "<image>\nWhat is the dominant color? Answer with one word.",
                    }
                ],
                "images": [str(image_path.relative_to(output_dir))],
                "ability": "visual_question_answering",
                "reward_model": {"style": "none", "ground_truth": answer},
                "extra_info": {"answer": answer, "smoke_test": True},
                "vision_selector_budget": budget,
            }
        )

    source_path = output_dir / "vero_train_all_35k.verl.jsonl"
    with source_path.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")

    converter = _load_vero_converter()
    training_path = output_dir / "train.verl.jsonl"
    count = converter.build_training_jsonl(source_path, training_path, output_dir)
    if count != len(samples):
        raise RuntimeError(f"expected {len(samples)} records, converted {count}")
    print(training_path)


if __name__ == "__main__":
    main()

