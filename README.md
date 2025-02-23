# LLM Pretraining Codebase

This is a starter template for a large-scale LLM pretraining codebase. It includes modules for attention, transformer components, and operations for Triton and C++ CUDA kernels.

## Project Structure
- `src/`: Main source code
  - `model/`: Core model components (attention, transformer, layers)
  - `ops/`: Hardware-specific optimizations
    - `triton/`: Triton kernels for GPU acceleration
    - `cuda/`: CUDA C++ kernels
  - `bindings/`: Python bindings for Triton and CUDA kernels
- `tests/`: Unit tests for all components

## Setup
```bash
pip install -e .
```

## Running Tests
```bash
python -m pytest tests/
``` 