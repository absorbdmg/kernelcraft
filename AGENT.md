# AGENTS.md

Context for AI coding assistants working in this repository.

## What this repository is

A collection of GPU kernels written in [Triton](https://triton-lang.org/), each paired with a PyTorch reference implementation, correctness tests, and benchmark suite. The repository is a personal learning and portfolio project; conventions reflect that.

## Tech stack

- Python 3.11+
- Triton 3.x
- PyTorch 2.4+ (with CUDA)
- pytest for testing
- Poetry for dependency management
- GitHub Actions for CI

Hardware target: NVIDIA Ampere or newer. Primary development on RTX 3090 Ti.

## Repository layout

```
kernelcraft/
├── README.md
├── AGENTS.md                    # this file
├── pyproject.toml
├── poetry.lock
├── .github/workflows/ci.yml
├── kernels/
│   └── <op_name>/
│       ├── kernel.py            # Triton implementation
│       ├── reference.py         # PyTorch reference
│       ├── bench.py             # Benchmark script
│       ├── test.py              # Correctness tests
│       └── README.md            # Op description, results
├── shared/
│   ├── benchmark_utils.py
│   └── correctness_utils.py
└── results/
    └── plots/
```

Each kernel directory is self-contained. No cross-kernel imports except through `shared/`.

## Conventions

### Naming
- Kernel directories: `snake_case` matching the op (`rmsnorm/`, `flash_attention/`).
- Kernel functions: `<op>_kernel`, e.g., `rmsnorm_kernel`.
- Python wrappers: `<op>`, e.g., `rmsnorm`.
- Reference implementations: `<op>_ref`, imported from `reference.py`.

### Testing
- Every kernel has a correctness test against the PyTorch reference.
- Tolerances: `rtol=1e-3, atol=1e-4` for FP32; `rtol=1e-2, atol=1e-2` for BF16.
- Tests use `pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")` so CI can skip GPU tests gracefully.
- A pure-NumPy reference may run in CI as a smoke test (no GPU required).

### Benchmarking
- Warm up with at least 10 iterations before timing.
- Always `torch.cuda.synchronize()` before and after timing.
- Report median of N=100 runs, not mean.
- Save raw results as CSV in `results/`, plots as PNG in `results/plots/`. Commit both.

### Commits
- Imperative mood: "Add RMSNorm kernel" not "Added RMSNorm kernel."
- Scope prefix when useful: `rmsnorm: fuse activation into output store`.
- Separate commits for: new kernel, optimization, benchmark addition, doc update.

### Per-kernel README
- One-paragraph description of what the op does.
- Headline result line: "Xx faster than `torch.<op>` for shape (B=N, D=M)."
- Notes on the optimization moves that mattered.

## Adding a new kernel

When asked to add a new kernel:
1. Create `kernels/<op_name>/` with all five files: `kernel.py`, `reference.py`, `bench.py`, `test.py`, `README.md`.
2. Use an existing kernel directory as a structural template.
3. Implement in order: reference first, then Triton kernel, then test, then benchmark.
4. Do not commit until correctness tests pass.

## Guardrails

- **Do not write CUDA C/C++.** This is a Triton repository by design.
- **Do not add new dependencies without asking.** `pyproject.toml` changes require approval.
- **Do not optimize before correctness.** Fast-but-wrong is worth zero. Get the answer right first.
- **Do not copy from existing open-source Triton kernels** (e.g., `triton-lang/kernels`, Liger Kernel, Unsloth, vLLM custom kernels) and present them as new work. Reference them; reimplement from scratch.
- **Do not commit files larger than 5 MB.** No model weights, no datasets, no checkpoints.
- **Do not aggressively refactor working code** for stylistic reasons. If tests pass, leave it alone unless asked.

## Workflow

```bash
# Setup
poetry install

# Lint
ruff check .

# Format
black .

# Tests (skipped without GPU)
pytest

# Benchmark a kernel
python kernels/<op_name>/bench.py
```

CI runs lint, format check, and NumPy-reference smoke tests on every push. GPU tests run locally only.