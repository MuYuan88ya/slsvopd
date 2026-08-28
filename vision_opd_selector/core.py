"""Differentiable visual-token selection shared by Transformers and vLLM."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_BUDGET = 0.2
DEFAULT_INIT_SCALE = 1e-4


class TransformerScorer(nn.Module):
    """AngelSlim-compatible query/key visual-token scorer."""

    def __init__(self, in_features: int, hidden_dim: int | None = None, init_scale: float = DEFAULT_INIT_SCALE):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        hidden_dim = in_features // 2 if hidden_dim is None else int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if init_scale < 0:
            raise ValueError(f"init_scale must be non-negative, got {init_scale}")
        self.in_features = int(in_features)
        self.hidden_dim = hidden_dim
        self.init_scale = float(init_scale)
        self.k_proj = nn.Linear(self.in_features, hidden_dim)
        self.q_proj = nn.Linear(self.in_features, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.k_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.q_proj.weight, std=self.init_scale)
        nn.init.zeros_(self.q_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        keys = self.k_proj(hidden_states)
        queries = self.q_proj(hidden_states)
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        return attention.mean(dim=-1)


@torch.no_grad()
def _find_soft_topk_threshold(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    lower = -scores.amax(dim=-1, keepdim=True) - 10.0
    upper = -scores.amin(dim=-1, keepdim=True) + 10.0
    for _ in range(64):
        middle = (lower + upper) * 0.5
        below_budget = torch.sigmoid(scores + middle).sum(dim=-1, keepdim=True) < keep_count
        lower = torch.where(below_budget, middle, lower)
        upper = torch.where(below_budget, upper, middle)
    return (lower + upper) * 0.5


class _DifferentiableTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, keep_count: int) -> torch.Tensor:
        threshold = _find_soft_topk_threshold(scores, keep_count)
        probabilities = torch.sigmoid(scores + threshold)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probabilities,) = ctx.saved_tensors
        derivative = probabilities * (1.0 - probabilities)
        denominator = derivative.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)
        weighted_grad = grad_output * derivative
        grad_scores = weighted_grad - weighted_grad.sum(dim=-1, keepdim=True) * derivative / denominator
        return grad_scores, None


def differentiable_topk(scores: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Return a differentiable soft mask summing to ``keep_count`` on each row."""

    token_count = scores.shape[-1]
    if token_count == 0:
        return scores
    keep_count = max(1, min(int(keep_count), token_count))
    if keep_count == token_count:
        return torch.ones_like(scores)
    return _DifferentiableTopK.apply(scores.float(), keep_count).to(scores.dtype)


def visual_token_counts(
    grid_thw: torch.Tensor | Sequence[Sequence[int]], spatial_merge_size: int
) -> list[int]:
    grids = grid_thw.detach().cpu().tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
    merge_area = int(spatial_merge_size) ** 2
    if merge_area <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
    return [int(t) * int(h) * int(w) // merge_area for t, h, w in grids]


def apply_segmented_soft_mask(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor | Sequence[Sequence[int]],
    spatial_merge_size: int,
    scorer: TransformerScorer,
    budget: float | torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent differentiable token budget to each visual item."""

    token_counts = visual_token_counts(grid_thw, spatial_merge_size)
    if any(count <= 0 for count in token_counts):
        raise ValueError(f"all visual items must contain at least one merged token, got {token_counts}")
    if sum(token_counts) != hidden_states.shape[0]:
        raise ValueError(
            "visual token counts do not match merger output: "
            f"expected {sum(token_counts)}, got {hidden_states.shape[0]}"
        )
    if not token_counts:
        return hidden_states, hidden_states.new_empty((0,))

    if isinstance(budget, torch.Tensor):
        budgets = budget.detach().cpu().reshape(-1).tolist()
    elif isinstance(budget, Sequence) and not isinstance(budget, str | bytes):
        budgets = list(budget)
    else:
        budgets = [budget]
    if len(budgets) == 1:
        budgets *= len(token_counts)
    if len(budgets) != len(token_counts):
        raise ValueError(
            "vision selector budgets must contain one value or one value per visual item, "
            f"got {len(budgets)} values for {len(token_counts)} items"
        )
    budgets = [float(item) for item in budgets]
    if any(not 0.0 < item <= 1.0 for item in budgets):
        raise ValueError(f"vision selector budgets must be in (0, 1], got {budgets}")

    masked_segments: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for segment, item_budget in zip(hidden_states.split(token_counts, dim=0), budgets, strict=True):
        scores = scorer(segment.unsqueeze(0)).squeeze(0)
        keep_count = max(1, math.floor(segment.shape[0] * item_budget))
        mask = differentiable_topk(scores.unsqueeze(0), keep_count).squeeze(0)
        masks.append(mask)
        masked_segments.append(segment * mask.to(segment.dtype).unsqueeze(-1))
    return torch.cat(masked_segments, dim=0), torch.cat(masks, dim=0)
