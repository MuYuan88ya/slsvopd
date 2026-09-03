"""Role-aware VisionSelector injection before FSDP wrapping."""

from __future__ import annotations

from typing import Any

import torch

from .budget import MODEL_BUDGETS_KEY
from .checkpoint import SCORER_KEYS, load_selector_weights
from .core import (
    DEFAULT_BUDGET,
    DEFAULT_INIT_SCALE,
    TransformerScorer,
    apply_segmented_soft_mask,
    mask_constraint_loss,
)

VALID_LLM_TRAIN_MODES = frozenset({"frozen", "full"})


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _find_visual(model: Any):
    visual = getattr(model, "visual", None)
    if visual is None:
        visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise ValueError("VisionSelector requires a model.visual or model.model.visual module")
    return visual


def _merged_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], torch.Tensor):
            return output[1]
        if output and isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported Qwen2.5-VL vision output type: {type(output)}")


def _replace_merged_hidden(output: Any, hidden_states: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden_states
    if hasattr(output, "pooler_output"):
        output.pooler_output = hidden_states
        return output
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], torch.Tensor):
            return (output[0], hidden_states, *output[2:])
        return (hidden_states, *output[1:])
    raise TypeError(f"Unsupported Qwen2.5-VL vision output type: {type(output)}")


def _grid_identity(grid_thw: Any) -> tuple[Any, ...] | None:
    if grid_thw is None:
        return None
    if isinstance(grid_thw, torch.Tensor):
        return (grid_thw.device.type, grid_thw.data_ptr(), tuple(grid_thw.shape))
    return (id(grid_thw),)


def _selector_model_pre_hook(module, args, kwargs):
    visual = _find_visual(module)
    visual._vision_selector_mask_loss = None
    visual._vision_selector_video_grid_identity = _grid_identity(kwargs.get("video_grid_thw"))
    visual._vision_selector_runtime_budgets = kwargs.pop(MODEL_BUDGETS_KEY, None)
    return args, kwargs


def _selector_model_output_hook(module, args, kwargs, output):
    del args, kwargs
    visual = _find_visual(module)
    visual._vision_selector_video_grid_identity = None
    visual._vision_selector_runtime_budgets = None
    return output


def _selector_forward_hook(module, args, kwargs, output):
    grid_thw = kwargs.get("grid_thw")
    if grid_thw is None and len(args) >= 2:
        grid_thw = args[1]
    if grid_thw is None:
        raise ValueError("VisionSelector could not locate grid_thw in vision forward inputs")
    if _grid_identity(grid_thw) == getattr(module, "_vision_selector_video_grid_identity", None):
        module._vision_selector_last_mask = None
        module._vision_selector_mask_loss = None
        return output
    runtime_budgets = getattr(module, "_vision_selector_runtime_budgets", None)
    budgets = module.vision_selector_budget if runtime_budgets is None else runtime_budgets
    hidden_states, mask = apply_segmented_soft_mask(
        _merged_hidden(output),
        grid_thw,
        module.spatial_merge_size,
        module.importance_scorer,
        budgets,
    )
    module._vision_selector_last_mask = mask.detach()
    if module.training and torch.is_grad_enabled():
        module._vision_selector_mask_loss = mask_constraint_loss(
            mask,
            grid_thw,
            module.spatial_merge_size,
            budgets,
        )
    return _replace_merged_hidden(output, hidden_states)


def apply_vision_selector_before_fsdp(
    *,
    model: Any,
    role: str,
    rank: int,
    model_config: Any,
    selector_config: Any,
) -> Any:
    """Inject the selector into actors while leaving ref/teacher models native."""

    del model_config
    enabled = bool(_config_get(selector_config, "enabled", False))
    if not enabled:
        return model
    if role != "actor":
        visual = _find_visual(model)
        if hasattr(visual, "importance_scorer"):
            raise RuntimeError(f"VisionSelector must not be present on role={role}")
        model.requires_grad_(False)
        return model
    llm_train_mode = str(_config_get(selector_config, "llm_train_mode", "frozen")).lower()
    if llm_train_mode not in VALID_LLM_TRAIN_MODES:
        raise ValueError(
            f"vision_selector.llm_train_mode must be one of {sorted(VALID_LLM_TRAIN_MODES)}, "
            f"got {llm_train_mode!r}"
        )
    if getattr(model, "_vision_selector_hook_applied", False):
        raise RuntimeError("VisionSelector pre-FSDP hook was invoked twice on the same actor")

    visual = _find_visual(model)
    if (
        hasattr(visual, "importance_scorer")
        or hasattr(visual, "_vision_selector_hook_handle")
        or hasattr(model, "_vision_selector_model_pre_hook_handle")
        or hasattr(model, "_vision_selector_model_output_hook_handle")
    ):
        raise RuntimeError("Partial or pre-existing VisionSelector injection detected")
    before_keys = set(model.state_dict())
    in_features = int(getattr(getattr(visual, "config", None), "out_hidden_size", 0))
    if in_features <= 0:
        in_features = int(getattr(visual, "out_hidden_size", 0))
    if in_features <= 0:
        raise ValueError("Cannot determine Qwen2.5-VL merger output dimension")

    hidden_dim = int(_config_get(selector_config, "hidden_dim", in_features // 2))
    init_scale = float(_config_get(selector_config, "init_scale", DEFAULT_INIT_SCALE))
    budget = float(_config_get(selector_config, "budget", DEFAULT_BUDGET))
    visual.importance_scorer = TransformerScorer(in_features, hidden_dim, init_scale)
    # Keep an FP32 master copy for the tiny trainable selector even when the
    # frozen backbone runs in FP16/BF16. Wrapping it as its own FSDP unit avoids
    # mixing parameter dtypes in the parent flat parameter.
    visual.importance_scorer._fsdp_keep_fp32 = True
    visual.importance_scorer._fsdp_wrap_separately = True
    visual.importance_scorer._wrap_overrides = {"mixed_precision": None}
    visual.vision_selector_budget = budget
    visual._vision_selector_mask_loss = None
    visual._vision_selector_hook_handle = visual.register_forward_hook(_selector_forward_hook, with_kwargs=True)
    model._vision_selector_model_pre_hook_handle = model.register_forward_pre_hook(
        _selector_model_pre_hook, with_kwargs=True
    )
    model._vision_selector_model_output_hook_handle = model.register_forward_hook(
        _selector_model_output_hook, with_kwargs=True, always_call=True
    )
    model._vision_selector_hook_applied = True

    selector_path = _config_get(selector_config, "checkpoint", None)
    if selector_path:
        load_selector_weights(model, selector_path)

    # The selector is always trainable and every other parameter owned by the
    # vision tower (encoder and merger included) is always frozen.  The only
    # configurable part is whether the language model remains frozen or is
    # trained in full.
    model.requires_grad_(llm_train_mode == "full")
    visual.requires_grad_(False)
    visual.importance_scorer.requires_grad_(True)
    model._vision_selector_llm_train_mode = llm_train_mode

    after_keys = set(model.state_dict())
    added = after_keys.difference(before_keys)
    expected = {
        f"{name}.importance_scorer.{key}"
        for name, module in model.named_modules()
        if module is visual
        for key in SCORER_KEYS
    }
    if added != expected or before_keys.difference(after_keys):
        raise RuntimeError(
            "VisionSelector changed the model key contract unexpectedly: "
            f"added={sorted(added)}, removed={sorted(before_keys.difference(after_keys))}, expected={sorted(expected)}"
        )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if llm_train_mode == "frozen":
        expected_trainable = expected
    else:
        visual_parameter_ids = {id(parameter) for parameter in visual.parameters()}
        scorer_parameter_ids = {id(parameter) for parameter in visual.importance_scorer.parameters()}
        expected_trainable = {
            name
            for name, parameter in model.named_parameters()
            if id(parameter) not in visual_parameter_ids or id(parameter) in scorer_parameter_ids
        }
        if expected_trainable == expected:
            raise RuntimeError("llm_train_mode=full did not find any trainable non-vision model parameters")
    if trainable != expected_trainable:
        raise RuntimeError(
            "VisionSelector trainability contract mismatch: "
            f"mode={llm_train_mode} expected={sorted(expected_trainable)} actual={sorted(trainable)}"
        )
    if rank == 0:
        print(
            "[VisionSelector FSDP] injected actor selector "
            f"in_features={in_features} hidden_dim={hidden_dim} budget={budget} "
            f"llm_train_mode={llm_train_mode} trainable_tensors={len(trainable)}",
            flush=True,
        )
    return model
