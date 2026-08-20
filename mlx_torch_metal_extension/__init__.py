from __future__ import annotations

import mlx.core as mx
import torch

from . import _ext


def silu_and_mul_mlx(
    input: mx.array,
    *,
    stream: mx.Stream | mx.ThreadLocalStream | mx.Device | None = None,
) -> mx.array:
    """Run the shared Metal kernel through an MLX primitive."""
    return _ext.silu_and_mul(input, stream=stream)


def silu_and_mul_torch(input: torch.Tensor) -> torch.Tensor:
    """Run the shared Metal kernel through the PyTorch MPS dispatcher."""
    return torch.ops.mlx_torch_metal.silu_and_mul(input)


__all__ = ["silu_and_mul_mlx", "silu_and_mul_torch"]
