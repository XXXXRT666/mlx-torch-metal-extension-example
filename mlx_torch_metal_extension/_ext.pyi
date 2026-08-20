import mlx.core

def silu_and_mul(
    input: mlx.core.array,
    *,
    stream: (
        mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | None
    ) = None,
) -> mlx.core.array:
    """Compute ``silu(x) * y`` using the MLX entry point."""
