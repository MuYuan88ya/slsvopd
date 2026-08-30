from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from vision_opd_selector.checkpoint import export_selector
from vision_opd_selector.core import TransformerScorer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_script():
    path = PROJECT_ROOT / "scripts" / "run_angelslim_selector_eval.py"
    spec = importlib.util.spec_from_file_location("run_angelslim_selector_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = _load_eval_script()


def test_vopd_vision_selector_config_uses_retention_complement():
    config = evaluation.build_angelslim_config(
        strategy="vision_selector",
        selector_path="/tmp/selector",
        retention_budget=0.2,
    )
    global_strategy = config["compressor"]["strategies"]["global"]
    assert global_strategy["strategy"] == "vision_selector"
    assert global_strategy["params"] == {"ratio": 0.8, "selector_path": "/tmp/selector"}
    assert config["compressor"]["requirements"] == {
        "inputs_embeds": True,
        "vision_token_mask": True,
    }


def test_idpruner_reuses_the_same_exported_selector():
    config = evaluation.build_angelslim_config(
        strategy="idpruner",
        selector_path="/tmp/selector",
        retention_budget=0.2,
    )
    params = config["compressor"]["strategies"]["global"]["params"]
    assert params["selector_path"] == "/tmp/selector"
    assert params["ratio"] == 0.8
    assert params["mmr_lambda"] == 0.5
    assert params["parallel_k"] == 1


def test_export_loads_through_vendored_angelslim(tmp_path):
    angelslim_root = PROJECT_ROOT / "third_party" / "AngelSlim"
    sys.path.insert(0, str(angelslim_root))
    try:
        from angelslim.compressor.token_compressor.algorithm.utils.vision_selector_utils import (
            _SELECTOR_REGISTRY,
            get_universal_selector_scores,
        )

        scorer = TransformerScorer(8, hidden_dim=4, init_scale=1e-4)
        model = type("Model", (), {})()
        model.visual = type("Visual", (), {})()
        model.visual.importance_scorer = scorer
        selector_dir = export_selector(model, tmp_path / "selector")
        hidden = torch.randn(1, 5, 8)
        expected = scorer(hidden)
        actual = get_universal_selector_scores(str(selector_dir), hidden, mode="raw")
        assert torch.allclose(actual, expected)
        _SELECTOR_REGISTRY.pop(str(selector_dir), None)
    finally:
        sys.path.remove(str(angelslim_root))
