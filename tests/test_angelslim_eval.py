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
    assert global_strategy["params"] == {"ratio": 0.8, "selector_path": "/tmp/selector", "score_mode": "raw"}
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


def test_checkpoint_step_resolves_and_rejects_bad_config(tmp_path):
    import json
    import pytest
    from vision_opd_selector.checkpoint import export_selector_state, resolve_selector_artifact

    scorer = TransformerScorer(8, hidden_dim=4)
    step = tmp_path / "global_step_2"
    directory = export_selector_state(scorer.state_dict(), step / "actor" / "selector")
    assert resolve_selector_artifact(step) == directory
    assert resolve_selector_artifact(step / "actor") == directory
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["hidden_dim"] = 99
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dimensions"):
        resolve_selector_artifact(step)


def test_trained_export_hard_pruning_and_missing_weight_failure(tmp_path):
    import pytest
    from safetensors.torch import load_file, save_file
    from vision_opd_selector.checkpoint import export_selector_state

    angelslim_root = PROJECT_ROOT / "third_party" / "AngelSlim"
    sys.path.insert(0, str(angelslim_root))
    try:
        from angelslim.compressor.token_compressor.algorithm.visionselector import vision_selector_pruning
        from angelslim.compressor.token_compressor.algorithm.utils.vision_selector_utils import (
            _SELECTOR_REGISTRY, get_universal_selector_scores,
        )
        from angelslim.compressor.token_compressor.base.context import PruningContext

        torch.manual_seed(17)
        scorer = TransformerScorer(8, hidden_dim=4, init_scale=0.1)
        hidden = torch.randn(1, 10, 8)
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.01)
        before = scorer.k_proj.weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad()
            scorer(hidden).square().mean().backward()
            optimizer.step()
        assert not torch.equal(before, scorer.k_proj.weight)
        directory = export_selector_state(scorer.state_dict(), tmp_path / "selector")
        context = PruningContext(
            input_ids=torch.tensor([[1] * 10 + [0] * 2]),
            model_config={"model_type": "qwen2_5_vl", "image_token_id": 1},
            inputs_embeds=torch.cat([hidden, torch.randn(1, 2, 8)], dim=1),
            vision_token_mask=torch.tensor([True] * 10 + [False] * 2),
        )
        keep = vision_selector_pruning(context, selector_path=str(directory), ratio=0.8, score_mode="raw")
        expected = torch.zeros(12, dtype=torch.bool)
        expected[scorer(hidden).squeeze(0).topk(2).indices] = True
        expected[-2:] = True
        assert torch.equal(keep[0], expected)
        loaded = _SELECTOR_REGISTRY.pop(str(directory))["model"]
        for name, parameter in scorer.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], parameter, rtol=0, atol=0)
        state = load_file(str(directory / "model.safetensors"))
        state.pop("q_proj.bias")
        save_file(state, str(directory / "model.safetensors"))
        with pytest.raises(RuntimeError, match="Missing key"):
            get_universal_selector_scores(str(directory), hidden, mode="raw")
    finally:
        sys.path.remove(str(angelslim_root))
