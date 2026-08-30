#!/usr/bin/env python3
"""Build a structurally valid tiny Qwen2.5-VL model for CUDA control-flow tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


MARKER_NAME = "vision_opd_tiny_model.json"


def _is_ready(output_dir: Path) -> bool:
    marker = output_dir / MARKER_NAME
    if not marker.is_file() or not (output_dir / "config.json").is_file():
        return False
    metadata = json.loads(marker.read_text())
    return metadata == {"text_hidden_size": 32, "text_num_attention_heads": 2}


def build_model(base_model: Path, output_dir: Path) -> None:
    if _is_ready(output_dir):
        print(f"tiny_model=READY {output_dir}")
        return

    config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.num_attention_heads = 2
    config.text_config.num_key_value_heads = 1
    config.text_config.rope_parameters = {
        "mrope_section": [2, 2, 4],
        "rope_type": "default",
        "type": "default",
        "rope_theta": 1_000_000.0,
    }
    config.vision_config.hidden_size = 32
    config.vision_config.embed_dim = 32
    config.vision_config.out_hidden_size = 32
    config.vision_config.intermediate_size = 64
    config.vision_config.num_heads = 2
    config.vision_config.fullatt_block_indexes = (0, 1)

    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(config)
    processor = AutoProcessor.from_pretrained(base_model, local_files_only=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    metadata = {"text_hidden_size": 32, "text_num_attention_heads": 2}
    (output_dir / MARKER_NAME).write_text(json.dumps(metadata, indent=2) + "\n")

    reloaded = AutoConfig.from_pretrained(output_dir, local_files_only=True)
    head_dim = reloaded.text_config.hidden_size // reloaded.text_config.num_attention_heads
    if head_dim != 16 or reloaded.vision_config.out_hidden_size != reloaded.text_config.hidden_size:
        raise RuntimeError("generated tiny model has an invalid Qwen2.5-VL shape contract")
    print(f"tiny_model=BUILT {output_dir} head_dim={head_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_model(args.base_model, args.output_dir)


if __name__ == "__main__":
    main()
