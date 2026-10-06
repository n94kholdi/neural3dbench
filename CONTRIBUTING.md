# Contributing to Neural3DBench

Thank you for helping improve Neural3DBench.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,pyg]"
pytest
```

## Proposing a change

1. Open an issue for substantial features or benchmark-protocol changes.
2. Create a focused branch from `main`.
3. Add or update tests for behavioral changes.
4. Run the complete test suite locally.
5. Open a pull request explaining the motivation, implementation, and test
   results.

New architectures should follow the model, configuration, and registry pattern
described in [docs/adding_a_network.md](docs/adding_a_network.md).

## Benchmark integrity

Benchmark contributions must document the dataset and split, preprocessing,
training budget, random seeds, metrics, and relevant hardware. Do not submit a
headline comparison unless all models were evaluated under the same protocol.

## Generated files

Do not commit datasets, experiment runs, or large checkpoints. Small fixtures
required by automated tests are acceptable when their purpose and origin are
documented.
