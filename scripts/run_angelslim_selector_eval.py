#!/usr/bin/env python3
"""Evaluate a trained Vision-OPD selector with the vendored AngelSlim codebase."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_TASKS = ("mmmu_val", "docvqa_val", "chartqa")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_angelslim_config(
    *,
    strategy: str,
    selector_path: str,
    retention_budget: float,
    mmr_lambda: float = 0.5,
    parallel_k: int = 1,
) -> dict[str, Any]:
    if strategy not in {"vision_selector", "idpruner"}:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if not 0.0 < retention_budget <= 1.0:
        raise ValueError(f"retention_budget must be in (0, 1], got {retention_budget}")

    params: dict[str, Any] = {
        # AngelSlim calls this the pruning ratio. Vision-OPD's budget is the
        # retained fraction, hence the complement here.
        "ratio": 1.0 - retention_budget,
        "selector_path": selector_path,
    }
    requirements = {"inputs_embeds": True}
    if strategy == "vision_selector":
        requirements["vision_token_mask"] = True
    else:
        params.update({"mmr_lambda": mmr_lambda, "parallel_k": parallel_k})

    return {
        "model_mapping": [
            {
                "name": "vl_model",
                "path": "model",
                "wrapper_module": ".models.qwen2_5_vl",
                "wrapper_class": "Prunable_Qwen2_5_VLModel",
            },
            {
                "name": "text_model",
                "path": "model.language_model",
                "wrapper_module": ".models.qwen2_5_vl",
                "wrapper_class": "Prunable_Qwen2_5_VLTextModel",
            },
        ],
        "compressor": {
            "requirements": requirements,
            "strategies": {"global": {"strategy": strategy, "params": params}},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angelslim-root", type=Path, default=PROJECT_ROOT / "third_party" / "AngelSlim")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--selector-path", required=True)
    parser.add_argument("--strategy", choices=("vision_selector", "idpruner"), default="vision_selector")
    parser.add_argument(
        "--retention-budget",
        type=float,
        default=0.2,
        help="fraction of visual tokens retained; 0.2 means keep 20%% and prune 80%%",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "eval_results" / "vopd_vision_selector")
    parser.add_argument("--run-name", default="VOPD-VisionSelector")
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--parallel-k", type=int, default=1)
    parser.add_argument("--log-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    angelslim_root = args.angelslim_root.expanduser().resolve()
    evaluator = angelslim_root / "tools" / "run_token_pruning_evaluation.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"AngelSlim evaluator not found: {evaluator}")

    selector_path = args.selector_path
    local_selector = Path(selector_path).expanduser()
    if local_selector.exists():
        selector_path = str(local_selector.resolve())

    config = build_angelslim_config(
        strategy=args.strategy,
        selector_path=selector_path,
        retention_budget=args.retention_budget,
        mmr_lambda=args.mmr_lambda,
        parallel_k=args.parallel_k,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision-opd-angelslim-") as temporary_dir:
        config_path = Path(temporary_dir) / f"{args.run_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            str(evaluator),
            "--model_path",
            args.model_path,
            "--configs",
            str(config_path),
            "--tasks",
            *args.tasks,
            "--batch_size",
            str(args.batch_size),
            "--num_fewshot",
            str(args.num_fewshot),
            "--output_dir",
            str(args.output_dir.resolve()),
        ]
        if args.log_samples:
            command.append("--log_samples")
        print(
            f"Evaluating {args.run_name}: retain={args.retention_budget:.1%}, "
            f"prune={1.0 - args.retention_budget:.1%}, strategy={args.strategy}",
            flush=True,
        )
        subprocess.run(command, cwd=angelslim_root, check=True)


if __name__ == "__main__":
    main()
