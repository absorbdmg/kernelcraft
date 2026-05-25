# kernelcraft

Personal repository of Triton GPU kernel implementations, benchmarks, and writeups.

## Tech Stack

- **Language:** Python 3.11+
- **GPU DSL:** Triton 3.x
- **ML Framework:** PyTorch 2.4+ with CUDA
- **Hardware Target:** NVIDIA RTX 3090 Ti (Ampere, SM 8.6)
- **Dependency Management:** Poetry
- **Testing:** pytest
- **CI:** GitHub Actions

## Project Structure

```
kernelcraft/
├── kernels/          # Self-contained kernel directories
│   └── rmsnorm/      # (first kernel — WIP)
├── shared/           # Common benchmark and correctness utilities
├── results/plots/    # Generated benchmark plots
└── notes/            # Learning log and research notes
```

## Setup

```bash
poetry install
```

## Running Tests

```bash
# CPU smoke tests (no GPU required)
pytest -v -k "cpu or numpy"

# Full test suite (requires CUDA)
pytest
```

## Benchmarking

Each kernel directory contains a `bench.py` script:

```bash
python kernels/<kernel_name>/bench.py
```

Results are saved to `results/` (CSV) and `results/plots/` (PNG).

## License

MIT
