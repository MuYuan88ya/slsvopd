"""Strict AngelSlim-compatible selector checkpoint I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

SCORER_KEYS = frozenset(
    {
        "k_proj.weight",
        "k_proj.bias",
        "q_proj.weight",
        "q_proj.bias",
    }
)

SELECTOR_SCORER_SOURCE = '''import math
import torch
import torch.nn as nn


class TransformerScorer(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=1024, init_scale=0.0001):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.q_proj = nn.Linear(in_features, hidden_dim)
        nn.init.normal_(self.k_proj.weight, std=init_scale)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.q_proj.weight, std=init_scale)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, x):
        keys = self.k_proj(x)
        queries = self.q_proj(x)
        return (queries @ keys.transpose(-2, -1) / math.sqrt(self.hidden_dim)).mean(dim=-1)
'''


def find_importance_scorer(model: Any):
    candidates = [
        getattr(model, "visual", None),
        getattr(getattr(model, "model", None), "visual", None),
    ]
    wrapped = getattr(model, "_fsdp_wrapped_module", None)
    if wrapped is not None:
        candidates.extend(
            [getattr(wrapped, "visual", None), getattr(getattr(wrapped, "model", None), "visual", None)]
        )
    for visual in candidates:
        if visual is not None and hasattr(visual, "importance_scorer"):
            return visual.importance_scorer
    raise ValueError("Cannot find visual.importance_scorer on model")


def normalize_scorer_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for original_name, tensor in state.items():
        name = original_name
        if "importance_scorer." in name:
            name = name.split("importance_scorer.", 1)[1]
        if name not in SCORER_KEYS:
            unexpected.append(original_name)
            continue
        if name in normalized:
            raise ValueError(f"Duplicate selector parameter after normalization: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Selector parameter {original_name} is not a tensor: {type(tensor)}")
        normalized[name] = tensor.detach().cpu()
    missing = SCORER_KEYS.difference(normalized)
    if missing or unexpected:
        raise ValueError(f"Invalid selector state dict: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    return normalized


def _resolve_selector_path(path_or_repo: str | Path) -> Path:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(path_or_repo),
            allow_patterns=["model.safetensors", "config.json", "*.pt", "*.pth"],
        )
    )


def load_selector_state(path_or_repo: str | Path) -> dict[str, torch.Tensor]:
    path = _resolve_selector_path(path_or_repo)
    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        candidates = [path / "trainable_state.pt", path / "model.pt", path / "pytorch_model.bin"]
        if safetensors_path.is_file():
            state = load_file(str(safetensors_path), device="cpu")
        else:
            checkpoint_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No supported selector checkpoint found in {path}")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    elif path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Selector checkpoint must contain a parameter mapping, got {type(state)}")
    return normalize_scorer_state_dict(state)


def load_selector_weights(model: Any, path_or_repo: str | Path) -> None:
    scorer = find_importance_scorer(model)
    state = load_selector_state(path_or_repo)
    scorer.load_state_dict(state, strict=True)


def extract_selector_state(model: Any) -> dict[str, torch.Tensor]:
    return normalize_scorer_state_dict(find_importance_scorer(model).state_dict())


def export_selector(model: Any, output_dir: str | Path, init_scale: float = 1e-4) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = extract_selector_state(model)
    hidden_dim, in_features = state["k_proj.weight"].shape
    if state["q_proj.weight"].shape != (hidden_dim, in_features):
        raise ValueError("q_proj and k_proj shapes do not match")
    config = {
        "in_features": int(in_features),
        "hidden_dim": int(hidden_dim),
        "init_scale": float(init_scale),
        "architecture": "TransformerScorer",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "selector_scorer.py").write_text(SELECTOR_SCORER_SOURCE, encoding="utf-8")
    save_file(state, str(output / "model.safetensors"))
    return output
