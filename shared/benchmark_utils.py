"""Common benchmarking utilities for kernelcraft kernels.

All benchmarks use the same protocol:
- 10+ warmup iterations
- torch.cuda.synchronize() before and after timing
- Median of N=100 runs reported
"""

from typing import Callable, List, Tuple
import time
import torch


def benchmark_kernel(
    fn: Callable,
    *args,
    warmup: int = 10,
    iters: int = 100,
    device: str = "cuda",
) -> float:
    """Benchmark a callable and return median wall-clock time in milliseconds.

    Args:
        fn: Callable to benchmark.
        args: Arguments to pass to fn.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.
        device: CUDA device string.

    Returns:
        Median elapsed time in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize(device)

    # Timed runs
    times: List[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start_evt.record()
        fn(*args)
        end_evt.record()
        torch.cuda.synchronize(device)
        times.append(start_evt.elapsed_time(end_evt))

    times.sort()
    return times[len(times) // 2]


def benchmark_sweep(
    fn: Callable,
    reference_fn: Callable,
    shapes: List[Tuple[int, ...]],
    warmup: int = 10,
    iters: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> List[dict]:
    """Benchmark a kernel against a reference across a sweep of shapes.

    Args:
        fn: Kernel under test.
        reference_fn: PyTorch reference implementation.
        shapes: List of shape tuples to benchmark.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.
        device: CUDA device string.
        dtype: Tensor dtype for inputs.

    Returns:
        List of result dicts with keys: shape, kernel_ms, ref_ms, speedup.
    """
    results = []
    for shape in shapes:
        # Create inputs
        x = torch.randn(shape, dtype=dtype, device=device)

        # Time kernel
        kernel_ms = benchmark_kernel(fn, x, warmup=warmup, iters=iters, device=device)

        # Time reference
        ref_ms = benchmark_kernel(reference_fn, x, warmup=warmup, iters=iters, device=device)

        results.append(
            {
                "shape": shape,
                "kernel_ms": kernel_ms,
                "ref_ms": ref_ms,
                "speedup": ref_ms / kernel_ms if kernel_ms > 0 else float("inf"),
            }
        )
    return results
