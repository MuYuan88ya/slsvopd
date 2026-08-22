"""VisionSelector integration for Vision-OPD."""

from .core import TransformerScorer, apply_segmented_soft_mask, differentiable_topk

__all__ = ["TransformerScorer", "apply_segmented_soft_mask", "differentiable_topk"]
