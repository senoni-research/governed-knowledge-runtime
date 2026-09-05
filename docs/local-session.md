# Local interactive session

This is a single-user development interface over the existing authority, context, MLX,
verification, and trace path. Actor and group values are simulated CLI assertions, not
authentication. Do not use secrets or client-confidential material.

Create a local config outside the repository, for example
`~/.config/gkr/session.json`:

```json
{
  "database": "~/gkr-local/authority.sqlite",
  "trace_database": "~/gkr-local/query-traces.sqlite",
  "generator_model": "~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/LOCAL-SNAPSHOT",
  "verifier_model": "~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/LOCAL-SNAPSHOT",
  "actor": "local-owner",
  "groups": ["owner-knowledge"],
  "as_of": "2026-09-05",
  "known_at": null,
  "max_tokens": 512,
  "verifier_max_tokens": 256,
  "evidence_tokens": 12000
}
```

The generator and verifier must be distinct existing local model directories. The authority
database must already exist. Import manually reviewed records with the existing CLI:

```bash
.venv/bin/gkr ingest ~/gkr-local/approved-records.jsonl \
  --db ~/gkr-local/authority.sqlite
```

Start one session; both models remain loaded between questions:

```bash
.venv/bin/python scripts/run_local_session.py \
  --config ~/.config/gkr/session.json
```

Useful commands inside the session:

```text
/as-of 2026-09-01
/known-at 2026-09-05T12:00:00Z
/known-at now
/status
/quit
```

For a reproducible non-interactive check:

```bash
.venv/bin/python scripts/run_local_session.py \
  --config ~/.config/gkr/session.json \
  --question "What must happen before a release?" \
  --question "What applied on the earlier date?"
```

Each result displays the answer or fixed abstention, cited and permitted record versions,
decision date, claim passages, check status, elapsed time, and trace ID. A withheld candidate
is printed only under an explicit diagnostic label. The phrase `local verifier supported; not
certified correct` is intentional.
