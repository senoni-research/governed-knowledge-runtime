# Contributing

Thanks for helping improve this runtime. The project is still an early vertical
slice; small, well-tested changes are easier to review than large redesigns.

## Local setup

On an Apple Silicon Mac:

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
```

Core tests do not require MLX:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest
```

Or, after bootstrap: `make gate0`, `make lint`, and `make test`.

## What to change

- Keep company facts out of model weights. MLX is an optional local generator.
- Authorization must reduce the candidate corpus before retrieval or generation.
- Temporal version resolution happens before status and ACL checks.
- Fail closed when a typed decision, citation, or verifier is ambiguous.
- Do not add a cloud inference provider to the primary runtime.

See [ADR 0001](docs/adr/0001-local-first-governed-runtime.md) before changing
those boundaries.

## Tests

Add or update tests next to the behavior you change. The frozen M0 suite and the
M1 schema contracts are regression gates: do not weaken unauthorized-exposure or
stale-version cases, and do not edit frozen `evaluation/m1/` files in place. A
schema change needs a new contract version. See the
[test programme](docs/test-programme.md) for gate status and exit conditions.
Do not commit M1 case JSONL until it has been validated against the active contract. Public
test manifests contain hashes, not the sealed plaintext test questions used for final
evaluation.

## What not to commit

- Real company knowledge, credentials, personal data, or secrets
- Model weights, adapters, or generated SQLite databases
- `.env` files, caches, or editor state

The included `knowledge/demo_records.jsonl` corpus is synthetic.

## Pull requests

- Keep the change focused and explain why it exists.
- Include test evidence (`pytest`, `ruff check .`).
- Avoid drive-by refactors and generated files.
