"""Reasoning-conditioned retrieval over the dense visual bank."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return residual + x


class SpatialContextEncoder(nn.Module):
    def __init__(self, hidden_size: int = 2560, spatial_dim: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_dim = spatial_dim
        self.down_proj = nn.Conv2d(hidden_size, spatial_dim, kernel_size=1)
        self.down_norm = nn.GroupNorm(1, spatial_dim)
        self.down_act = nn.GELU()
        self.block1 = ConvResidualBlock(spatial_dim)
        self.block2 = ConvResidualBlock(spatial_dim)
        self.up_proj = nn.Conv2d(spatial_dim, hidden_size, kernel_size=1)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, visual_tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        if visual_tokens.ndim != 2:
            raise ValueError(
                f"Expected visual_tokens shape [N, d], got {tuple(visual_tokens.shape)}"
            )
        num_tokens, hidden_size = visual_tokens.shape
        grid_h, grid_w = grid_hw
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: expected {self.hidden_size}, got {hidden_size}"
            )
        if grid_h * grid_w != num_tokens:
            raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")
        x = visual_tokens.transpose(0, 1).reshape(1, hidden_size, grid_h, grid_w)
        x = self.down_act(self.down_norm(self.down_proj(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.up_proj(x)
        context = x.squeeze(0).permute(1, 2, 0).reshape(num_tokens, hidden_size)
        return self.output_norm(visual_tokens + context)


class ProjectionMLP(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.linear1(x)))
        x = self.act(self.norm2(self.linear2(x)))
        return self.linear3(x)


class ContextAggregation(nn.Module):
    def __init__(self, hidden_size: int = 2560, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_t: torch.Tensor, context_hidden_states: torch.Tensor) -> torch.Tensor:
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        if h_t.shape != (1, self.hidden_size):
            raise ValueError(f"Expected h_t shape [d] or [1, d], got {tuple(h_t.shape)}")
        if context_hidden_states.ndim != 2 or context_hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"Expected context_hidden_states shape [N, d], got {tuple(context_hidden_states.shape)}"
            )
        num_context = context_hidden_states.shape[0]
        q = self.q_proj(h_t).reshape(1, self.num_heads, self.head_dim)
        k = self.k_proj(context_hidden_states).reshape(num_context, self.num_heads, self.head_dim)
        v = self.v_proj(context_hidden_states).reshape(num_context, self.num_heads, self.head_dim)
        q_t = q.permute(1, 0, 2)
        k_t = k.permute(1, 2, 0)
        v_t = v.permute(1, 0, 2)
        attn_logits = torch.matmul(q_t, k_t) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(dtype=h_t.dtype)
        attn_out = torch.matmul(attn_weights, v_t)
        attn_out = attn_out.permute(1, 0, 2).reshape(1, self.hidden_size)
        return self.out_proj(attn_out) + h_t


class RetrievalModule(nn.Module):
    def __init__(
        self,
        query_hidden_size: int = 2560,
        key_hidden_size: int = 2560,
        score_dim: int = 2048,
        spatial_dim: int = 512,
        num_heads: int = 16,
        context_num_heads: int = 8,
    ) -> None:
        super().__init__()
        if score_dim % num_heads != 0:
            raise ValueError(f"score_dim={score_dim} must be divisible by num_heads={num_heads}")
        self.query_hidden_size = query_hidden_size
        self.key_hidden_size = key_hidden_size
        self.score_dim = score_dim
        self.spatial_dim = spatial_dim
        self.num_heads = num_heads
        self.head_dim = score_dim // num_heads
        self.context_agg = ContextAggregation(
            hidden_size=query_hidden_size, num_heads=context_num_heads
        )
        self.spatial_context = SpatialContextEncoder(
            hidden_size=key_hidden_size, spatial_dim=spatial_dim
        )
        self.query_mlp = ProjectionMLP(hidden_size=query_hidden_size, output_size=score_dim)
        self.key_mlp = ProjectionMLP(hidden_size=key_hidden_size, output_size=score_dim)
        self.head_weights_raw = nn.Parameter(torch.zeros(num_heads))

    def encode_visual_keys(
        self, visual_tokens: torch.Tensor, grid_hw: tuple[int, int]
    ) -> torch.Tensor:
        if visual_tokens.ndim != 2:
            raise ValueError(
                f"Expected visual_tokens shape [N, d], got {tuple(visual_tokens.shape)}"
            )
        num_tokens = visual_tokens.shape[0]
        grid_h, grid_w = grid_hw
        if grid_h * grid_w != num_tokens:
            raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")
        retrieval_dtype = self.head_weights_raw.dtype
        visual_tokens = visual_tokens.to(dtype=retrieval_dtype)
        visual_tokens_ctx = self.spatial_context(visual_tokens, grid_hw)
        return self.key_mlp(visual_tokens_ctx).reshape(num_tokens, self.num_heads, self.head_dim)

    def score_precomputed_keys(
        self,
        h_t: torch.Tensor,
        visual_keys: torch.Tensor,
        context_hidden_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        if h_t.shape != (1, self.query_hidden_size):
            raise ValueError(
                f"Expected h_t shape [d] or [1, d] with dim={self.query_hidden_size}, got {tuple(h_t.shape)}"
            )
        expected_key_shape = (self.num_heads, self.head_dim)
        if visual_keys.ndim != 3 or tuple(visual_keys.shape[1:]) != expected_key_shape:
            raise ValueError(
                f"Expected visual_keys shape [N, num_heads, head_dim] with tail={expected_key_shape}, got {tuple(visual_keys.shape)}"
            )
        retrieval_dtype = self.head_weights_raw.dtype
        h_t = h_t.to(dtype=retrieval_dtype)
        visual_keys = visual_keys.to(dtype=retrieval_dtype)
        if context_hidden_states is not None and (context_hidden_states.shape[0] > 0):
            h_t = self.context_agg(h_t, context_hidden_states.to(dtype=retrieval_dtype))
        query = self.query_mlp(h_t).reshape(self.num_heads, self.head_dim)
        logits_per_head = torch.einsum("hd,nhd->nh", query, visual_keys)
        logits_per_head = logits_per_head / math.sqrt(self.head_dim)
        head_weights = F.softmax(self.head_weights_raw, dim=0)
        logits = (logits_per_head * head_weights.unsqueeze(0)).sum(dim=-1)
        return {"logits": logits, "scores": torch.sigmoid(logits)}

    def forward(
        self,
        h_t: torch.Tensor,
        visual_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        context_hidden_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_keys = self.encode_visual_keys(visual_tokens, grid_hw)
        return self.score_precomputed_keys(
            h_t, visual_keys, context_hidden_states=context_hidden_states
        )
