"""Common correctness-checking utilities for kernelcraft kernels."""

from typing import Optional
import torch


def assert_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    msg: Optional[str] = None,
) -> None:
    """Assert two tensors are close with configurable tolerances.

    Args:
        actual: Tensor from the kernel under test.
        expected: Reference tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional extra message on failure.
    """
    close = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    if not close:
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        max_idx = diff.argmax().item()
        base_msg = f"Max diff: {max_diff:.6e} at flat index {max_idx}"
        if msg:
            base_msg = f"{msg} | {base_msg}"
        raise AssertionError(base_msg)


def default_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Return (rtol, atol) defaults for a given dtype.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Tuple of (rtol, atol).
    """
    if dtype == torch.float32:
        return (1e-3, 1e-4)
    if dtype in (torch.float16, torch.bfloat16):
        return (1e-2, 1e-2)
    return (1e-3, 1e-4)
