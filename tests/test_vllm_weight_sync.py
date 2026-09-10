from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch


def _import_vllm_model_with_fake_upstream(monkeypatch, *, report_loaded: bool):
    class _FakeVision(torch.nn.Module):
        def __init__(self, vision_config, *args, **kwargs):
            super().__init__()
            self.out_hidden_size = 8
            self.spatial_merge_size = 1
            self.backbone = torch.nn.Linear(8, 8)

        def forward(self, x, grid_thw):
            return self.backbone(x)

    class _FakeModel(torch.nn.Module):
        def load_weights(self, weights):
            requested = {name.removeprefix("model.") for name, _ in weights}
            return requested if report_loaded else set()

        def _parse_and_validate_image_input(self, **kwargs):
            class _TensorSchemaLike:
                def __getitem__(self, key):
                    return getattr(self, key)

            image_input = _TensorSchemaLike()
            image_input.image_grid_thw = kwargs["image_grid_thw"]
            return image_input

    class _FakeProcessor:
        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            return {"image_grid_thw": "base-image-field"}

    class _FakeRegistry:
        @staticmethod
        def register_processor(processor, *, info, dummy_inputs):
            return lambda model_cls: model_cls

    class _FakeFieldConfig:
        @staticmethod
        def batched(modality):
            return modality

    upstream = ModuleType("vllm.model_executor.models.qwen2_5_vl")
    upstream.Qwen2_5_VisionTransformer = _FakeVision
    upstream.Qwen2_5_VLForConditionalGeneration = _FakeModel
    upstream.Qwen2_5_VLMultiModalProcessor = _FakeProcessor
    upstream.Qwen2_5_VLProcessingInfo = object
    upstream.Qwen2_5_VLDummyInputsBuilder = object
    upstream.MULTIMODAL_REGISTRY = _FakeRegistry()
    upstream.MultiModalFieldConfig = _FakeFieldConfig

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
    model = _fake_selector_model(module).to(torch.bfloat16)

    loaded = model.load_weights(
        [("model.visual.importance_scorer.q_proj.weight", torch.ones(1))]
    )

    assert loaded == {"visual.importance_scorer.q_proj.weight"}
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.visual.importance_scorer.parameters())


def test_vllm_loader_rejects_a_requested_selector_weight_not_loaded(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=False)
    model = _fake_selector_model(module)

    with pytest.raises(ValueError, match="did not load selector parameters"):
        model.load_weights(
            [("model.visual.importance_scorer.q_proj.weight", torch.ones(1))]
        )


def test_vllm_selector_follows_upstream_construction_dtype(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=True)
    previous_dtype = torch.get_default_dtype()
    try:
        # vLLM constructs ordinary modules inside the model's default dtype context.
        torch.set_default_dtype(torch.bfloat16)
        visual = module.VisionSelectorQwen2_5VisionTransformer(object())
    finally:
        torch.set_default_dtype(previous_dtype)
    assert all(parameter.dtype == torch.bfloat16 for parameter in visual.parameters())
    output = visual(torch.randn(4, 8, dtype=torch.bfloat16), [[1, 1, 4]])
    assert output.dtype == torch.bfloat16
    assert visual._vision_selector_last_mask.dtype == torch.float32


def test_vllm_processor_expands_request_budget_to_each_image(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=True)
    processor = module.VisionSelectorQwen2_5VLMultiModalProcessor()
    processor.info = type(
        "FakeInfo",
        (),
        {"get_hf_config": lambda self: type("FakeConfig", (), {"vision_selector": {"budget": 0.2}})()},
    )()
    hf_inputs = {"image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]])}

    fields = processor._get_mm_fields_config(
        hf_inputs, {module.BUDGET_KEY: 0.15}
    )

    assert fields[module.MODEL_BUDGETS_KEY] == "image"
    assert hf_inputs[module.MODEL_BUDGETS_KEY].dtype == torch.float64
    assert hf_inputs[module.MODEL_BUDGETS_KEY].tolist() == [0.15, 0.15]


def test_vllm_image_parser_attaches_budget_to_tensor_schema(monkeypatch):
    module = _import_vllm_model_with_fake_upstream(monkeypatch, report_loaded=True)
    model = _fake_selector_model(module)
    model.visual.vision_selector_budget = 0.2

    image_input = model._parse_and_validate_image_input(
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        vision_selector_budgets=torch.tensor([0.15], dtype=torch.float64),
    )

    assert image_input[module.MODEL_BUDGETS_KEY].tolist() == [0.15]
