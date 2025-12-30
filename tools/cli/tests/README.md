# CLI tests

Python based CLI tests using [pytest](https://docs.pytest.org/en/stable/).

### Install dependencies

```bash
pip3 install pytest
```

### Run tests

1. Build the CLI tool

```bash
cd ../../..
cmake -B build
cmake --build build --target llama-cli
```

2. Run the tests

```bash
cd tools/cli/tests
python3 -m pytest unit/ -v
```

### Test Categories

- **unit/**: Unit tests for specific CLI features
  - `test_kv_cache_persistence.py`: Tests for KV cache persistence feature
