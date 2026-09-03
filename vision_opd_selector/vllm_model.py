"""Native vLLM Qwen2.5-VL model with differentiable visual-token gating."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import vllm.model_executor.models.qwen2_5_vl as upstream

from .budget import BUDGET_KEY, MODEL_BUDGETS_KEY, validate_budget
from .core import DEFAULT_BUDGET, DEFAULT_INIT_SCALE, TransformerScorer, apply_segmented_soft_mask

_UpstreamVision = upstream.Qwen2_5_VisionTransformer
_UpstreamModel = upstream.Qwen2_5_VLForConditionalGeneration


def _config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class VisionSelectorQwen2_5VLMultiModalProcessor(upstream.Qwen2_5_VLMultiModalProcessor):
    """Turn a per-request budget into one batched value for every image."""

    def _default_budget(self) -> float:
        hf_config = self.info.get_hf_config()
        selector_config = getattr(hf_config, "vision_selector", {})
        return validate_budget(_config_get(selector_config, "budget", DEFAULT_BUDGET))

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        hf_mm_kwargs = dict(mm_kwargs)
        hf_mm_kwargs.pop(BUDGET_KEY, None)
        return super()._call_hf_processor(prompt, mm_data, hf_mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        image_grid_thw = hf_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return fields

        budget = validate_budget(hf_processor_mm_kwargs.get(BUDGET_KEY, self._default_budget()))
        hf_inputs[MODEL_BUDGETS_KEY] = torch.full(
            (image_grid_thw.shape[0],), budget, dtype=torch.float64
        )
        fields[MODEL_BUDGETS_KEY] = upstream.MultiModalFieldConfig.batched("image")
        return fields

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        hf_mm_kwargs = dict(hf_processor_mm_kwargs)
        hf_mm_kwargs.pop(BUDGET_KEY, None)
        return super()._get_prompt_updates(mm_items, hf_mm_kwargs, out_mm_kwargs)


class VisionSelectorQwen2_5VisionTransformer(_UpstreamVision):
    def __init__(self, vision_config, *args, **kwargs):
        super().__init__(vision_config, *args, **kwargs)
        selector_config = getattr(vision_config, "_vision_selector_runtime_config", {})
        hidden_dim = int(_config_get(selector_config, "hidden_dim", self.out_hidden_size // 2))
        init_scale = float(_config_get(selector_config, "init_scale", DEFAULT_INIT_SCALE))
        self.vision_selector_budget = float(_config_get(selector_config, "budget", DEFAULT_BUDGET))
        self.vision_selector_enabled = bool(_config_get(selector_config, "enabled", True))
        self.importance_scorer = TransformerScorer(self.out_hidden_size, hidden_dim, init_scale).float()
        self._vision_selector_last_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        hidden_states = super().forward(x, grid_thw)
        if not self.vision_selector_enabled:
            return hidden_states
        hidden_states, mask = apply_segmented_soft_mask(
            hidden_states,
            grid_thw,
            self.spatial_merge_size,
            self.importance_scorer,
            self.vision_selector_budget,
        )
        self._vision_selector_last_mask = mask.detach()
        return hidden_states


@upstream.MULTIMODAL_REGISTRY.register_processor(
    VisionSelectorQwen2_5VLMultiModalProcessor,
    info=upstream.Qwen2_5_VLProcessingInfo,
    dummy_inputs=upstream.Qwen2_5_VLDummyInputsBuilder,
)
class Qwen2_5_VLForConditionalGenerationWithVisionSelector(_UpstreamModel):
    """Version-locked native Qwen2.5-VL wrapper; no generic Transformers fallback."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        # Keep this constructor explicit and version-locked to vLLM 0.18.  In
        # particular, do not replace an upstream class globally while engines
        # are being constructed concurrently in the same process.
        torch.nn.Module.__init__(self)
        hf_config = vllm_config.model_config.hf_config
        selector_config = getattr(hf_config, "vision_selector", {})
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = hf_config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()

        vision_config = hf_config.vision_config
        vision_config._vision_selector_runtime_config = selector_config
        try:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = VisionSelectorQwen2_5VisionTransformer(
                    vision_config=vision_config,
                    norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
                    quant_config=self.quant_config,
                    prefix=upstream.maybe_prefix(prefix, "visual"),
                )
        finally:
            delattr(vision_config, "_vision_selector_runtime_config")

        with self._mark_language_model(vllm_config):
            self.language_model = upstream.init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=upstream.maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        if not isinstance(self.visual, VisionSelectorQwen2_5VisionTransformer):
            raise RuntimeError("vLLM constructed the upstream vision tower instead of VisionSelector tower")

    def _parse_and_validate_image_input(self, **kwargs):
        image_input = super()._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            # vLLM 0.18 returns a TensorSchema object here.  It supports
            # ``obj[key]`` reads through ``getattr`` but intentionally has no
            # item-assignment implementation.
            setattr(
                image_input,
                MODEL_BUDGETS_KEY,
                kwargs.get(MODEL_BUDGETS_KEY, self.visual.vision_selector_budget),
            )
        return image_input

    def _process_image_input(self, image_input):
        selector_enabled = self.visual.vision_selector_enabled
        self.visual.vision_selector_enabled = False
        try:
            image_embeddings = super()._process_image_input(image_input)
        finally:
            self.visual.vision_selector_enabled = selector_enabled

        if not selector_enabled:
            return image_embeddings

        sizes = [embedding.shape[0] for embedding in image_embeddings]
        hidden_states, mask = apply_segmented_soft_mask(
            torch.cat(image_embeddings, dim=0),
            image_input["image_grid_thw"],
            self.visual.spatial_merge_size,
            self.visual.importance_scorer,
            image_input[MODEL_BUDGETS_KEY],
        )
        self.visual._vision_selector_last_mask = mask.detach()
        return hidden_states.split(sizes)

    def _process_video_input(self, video_input):
        previous = self.visual.vision_selector_enabled
        self.visual.vision_selector_enabled = False
        try:
            return super()._process_video_input(video_input)
        finally:
            self.visual.vision_selector_enabled = previous

    def _finalize_selector_device_dtype(self) -> None:
        visual_parameter = next(
            parameter
            for name, parameter in self.visual.named_parameters()
            if not name.startswith("importance_scorer.")
        )
        scorer = self.visual.importance_scorer
        scorer.to(device=visual_parameter.device, dtype=torch.float32)
        for parameter in scorer.parameters():
            if parameter.device != visual_parameter.device or parameter.dtype != torch.float32:
                raise RuntimeError(
                    "VisionSelector dtype/device finalization failed: "
                    f"selector={parameter.device}/{parameter.dtype}, "
                    f"expected={visual_parameter.device}/{torch.float32}"
                )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        requested: set[str] = set()

        def tracked_weights():
            for name, weight in weights:
                if "importance_scorer." in name:
                    normalized = name.removeprefix("model.")
                    requested.add(normalized)
                yield name, weight

        loaded = super().load_weights(tracked_weights())
        if requested:
            missing = requested.difference(loaded)
            if missing:
                raise ValueError(f"vLLM did not load selector parameters: {sorted(missing)}")
            self._finalize_selector_device_dtype()
        return loaded
