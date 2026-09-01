# Security policy

This project is an early local-first research runtime. It is not a production
authorization or inference service.

## What this repository does not provide

- The CLI `--actor` and `--group` flags are a development harness. Anyone who can
  run `gkr` can assert any principal.
- The SQLite hash chain detects ordinary mutation. It is not a digital signature
  and does not protect against replacing the whole database.
- A local model judge is fallible. Citation checks are deterministic; semantic
  support is not a proof of correctness.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Email the maintainer named in `pyproject.toml` with:

- a description of the issue and its impact;
- reproduction steps that stay inside this repository or synthetic data;
- whether any real company knowledge, credentials, or personal data were involved.

If you maintain a fork with a public tracker, prefer a private advisory.

## Safe use of a public clone

- Keep real company knowledge, credentials, and model weights out of Git.
- Treat `artifacts/`, `models/`, and any `*.sqlite` files as local state.
- Do not ingest source artifacts from untrusted JSONL files; import paths are
  confined to the import directory, but the importer still reads local files.
- Network model download is an explicit `--allow-model-download` opt-in.
  Operational inference remains on-device.
