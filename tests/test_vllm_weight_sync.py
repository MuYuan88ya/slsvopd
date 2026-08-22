from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch


def _import_vllm_model_with_fake_upstream(monkeypatch, *, report_loaded: bool):
    class _FakeVision(torch.nn.Module):
        pass

    class _FakeModel(torch.nn.Module):
        def load_weights(self, weights):
            requested = {name.removeprefix("model.") for name, _ in weights}
            return requested if report_loaded else set()

    upstream = ModuleType("vllm.model_executor.models.qwen2_5_vl")
    upstream.Qwen2_5_VisionTransformer = _FakeVision
    upstream.Qwen2_5_VLForConditionalGeneration = _FakeModel

    module_names = ["vllm", "vllm.model_executor", "vllm.model_executor.models"]
    for module_name in module_names:
        module = ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    sys.modules.pop("vision_opd_selector.vllm_model", None)
    return importlib.import_module("vision_opd_selector.vllm_model")


def _fake_selector_model(module):
    model_class = module.Qwen2_5_VLForConditionalGenerationWithVisionSelector
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.visual = torch.nn.Module()
    model.visual.backbone = torch.nn.Parameter(torch.ones(1))
    model.visual.importance_scorer = torch.nn.Linear(1, 1)
    return model


def test_vllm_loader_accepts_a_valid_selector_weight_chunk(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=True)
    model = _fake_selector_model(module)

    loaded = model.load_weights(
        [("model.visual.importance_scorer.q_proj.weight", torch.ones(1))]
    )

    assert loaded == {"visual.importance_scorer.q_proj.weight"}


def test_vllm_loader_rejects_a_requested_selector_weight_not_loaded(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=False)
    model = _fake_selector_model(module)

    with pytest.raises(ValueError, match="did not load selector parameters"):
        model.load_weights(
            [("model.visual.importance_scorer.q_proj.weight", torch.ones(1))]
        )
