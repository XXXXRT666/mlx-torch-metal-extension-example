import mlx.core as mx
import torch

from mlx_torch_metal_extension import silu_and_mul_mlx, silu_and_mul_torch


def test_mlx() -> None:
    x = (mx.arange(24, dtype=mx.float16).reshape(2, 12) - 12) / 4
    actual = silu_and_mul_mlx(x)
    lhs, rhs = x[..., :6], x[..., 6:]
    expected = (lhs * mx.sigmoid(lhs)) * rhs
    assert bool(mx.allclose(actual, expected, rtol=2e-3, atol=2e-3))
    mx.synchronize()
    print("MLX entry point: OK")


def test_torch() -> None:
    assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable"
    x = (torch.arange(24, dtype=torch.float16, device="mps").reshape(2, 12) - 12) / 4
    actual = silu_and_mul_torch(x)
    lhs, rhs = x[..., :6], x[..., 6:]
    expected = torch.nn.functional.silu(lhs) * rhs
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.mps.synchronize()
    print("PyTorch MPS entry point: OK")


if __name__ == "__main__":
    test_mlx()
    test_torch()
