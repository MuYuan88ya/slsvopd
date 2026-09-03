"""VisionSelector integration for Vision-OPD."""

from .core import TransformerScorer, apply_segmented_soft_mask, differentiable_topk, mask_constraint_loss

__all__ = ["TransformerScorer", "apply_segmented_soft_mask", "differentiable_topk", "mask_constraint_loss"]
