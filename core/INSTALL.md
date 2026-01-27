# Glotta Core - Python Backend

Backend for language learning with constrained LLM generation.

## Installation with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .

# Or install directly without venv activation
uv pip install -e .
```

## Quick Start

```bash
# Run demo
uv run python demo.py 1

# Start API server
uv run python api_server.py
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format code
uv run black .

# Lint
uv run ruff check .
```

## Usage

See [README.md](./README.md) for full documentation.
