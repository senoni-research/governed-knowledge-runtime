# Governed Knowledge Runtime

A local-first runtime for storing, updating, retrieving, and using mutable company knowledge
on Apple Silicon. Company facts remain outside model weights. MLX is the local execution
engine that consumes governed evidence.

This is an early research prototype (`0.1.0`). The CLI is a development harness, not an
authentication boundary. Do not ingest real secrets or production company knowledge into a
public clone.

The design reuses governance and evaluation lessons from a preceding parametric-memory
experiment. It does not assume that current company facts belong in a model adapter.

## Non-negotiable boundaries

- All inference runs locally on the Mac. There is no cloud model provider in the runtime.
- The SQLite authority ledger is append-only and hash-chained.
- Business validity time and system observation time are separate.
- Authorization reduces the candidate corpus before policy matching, retrieval, or model
  execution.
- A later version cannot silently expose an older, now-restricted version.
- The authority ledger accepts only approved or retired events; proposals use a separate
  append-only workflow store and have no authority version.
- Source digests are recomputed from canonical statements or local raw bytes before append.
- The model receives exact record versions, source spans, source hashes, an authority snapshot
  ID, and a distinct evidence-bundle ID.
- Citation integrity is deterministic; semantic claim support remains a separate verifier.
- Candidates with bad citations, unsupported claims, inconclusive support, or verifier errors
  are withheld from the answer field.
- Every `gkr ask` execution is written to a separate append-only query-trace database.

## Current vertical slice

```text
proposal workflow --owner approval--> JSONL authority event
        |
        v
append-only local SQLite ledger
        |
        v
valid-time + known-time resolution
        |
        v
authorization-first candidate set
        |
        +--> typed policy request + approved rule --> deterministic answer
        |
        `--> full authorized context when it fits
                    |
                    `--> local BM25 when it does not
                    |
                    v
          source-addressable prompt
                    |
                    v
             local MLX model
                    |
                    v
       deterministic citation check
                    |
                    v
        separate local semantic judge
                    |
             publish or withhold
                    |
                    v
       append-only execution trace
```

Machine-readable `PolicyRule` objects are evaluated deterministically only after a narrow
parser constructs a typed request against the complete authorized corpus. A matched rule's
authority record is pinned into its evidence bundle, so deterministic execution does not
depend on BM25 selecting that record. Negation, multiple amounts, unclear subjects,
conditionals, unevaluated rule conditions or exceptions, and competing rules fail closed to
the non-deterministic path. The full-context/BM25 router is intentionally simple. Dense
retrieval, reranking, temporal graph traversal, and claim-by-claim provenance tracing will be
added only behind measured evaluation gates.

## Quick start on an Apple Silicon Mac

Core runtime, demo ledger, and tests:

```bash
./scripts/bootstrap_macos.sh
source .venv/bin/activate
```

Compile the complete evidence prompt without loading a model:

```bash
gkr context \
  "Does a £700 travel booking require special approval?" \
  --actor alice \
  --group employees \
  --as-of 2026-09-01
```

The demo has two immutable versions of the travel rule. Querying `2026-08-31` selects the
£500 version; querying `2026-09-01` selects the £750 version. `--known-at` can reconstruct
what the system knew at an earlier transaction time.

When `gkr ask` sees this unambiguous currency request and one matching approved `PolicyRule`,
it produces and cites the answer with the deterministic engine; no model is loaded.

```bash
gkr ask \
  "Does a £700 travel booking require special approval?" \
  --actor alice \
  --group employees \
  --as-of 2026-09-01
```

The current `--actor` and `--group` flags are a development harness, not authentication:
anyone who can run the CLI can assert a group. The later Cline-facing process must derive
principals from a trusted local identity/policy boundary and must not accept client-supplied
authority claims.

Install the optional local MLX execution layer:

```bash
INSTALL_MLX=1 ./scripts/bootstrap_macos.sh
```

Acquire an MLX model into an explicit local directory, then answer locally:

```bash
gkr ask \
  "What evidence is required before a production deployment?" \
  --actor alice \
  --group employees \
  --as-of 2026-09-01 \
  --model models/your-local-mlx-model \
  --verifier-model models/your-distinct-local-verifier
```

By default, `--model` must be an existing local path. `--allow-model-download` is an explicit
one-time opt-in for MLX-LM acquisition; generation still executes on-device. Without
`--verifier-model`, `gkr ask` runs a same-model diagnostic pass but never publishes its
self-approval. A candidate is published only when its citations resolve and a distinct local
judge returns parseable `supported`; all other outcomes are retained as withheld candidates
for audit. Model diversity is defense in depth, not proof of correctness.

Other useful commands:

```bash
gkr doctor
gkr ingest knowledge/demo_records.jsonl
gkr ledger
gkr eval
gkr context "question" --actor alice --group employees --json
```

`gkr eval` runs the frozen M0 suite covering valid-time selection, known-at reconstruction,
and restricted-record non-exposure. The versioned M1 case and metric contract remains frozen
under `evaluation/m1/`. A Grok-authored and Grok-remediated fictional synthetic
authority corpus now exists at `evaluation/m1/corpus/`. Independent ChatGPT content
review concluded `CONTENT_APPROVED`. Independent Cursor code review concluded
`CODE_APPROVED`. That is AI review of fictional benchmark content, not human or
organizational approval. The frozen v2 owner-labelled requirement is unchanged and
unmet. A later versioned contract is required before an AI-reviewed synthetic suite
can be promoted. The 360 scenarios and oracles are unbuilt, and Gate 1 remains
`not_run`. Corpus status `reviewed` is not `frozen`. Validate the corpus with:

```bash
.venv/bin/python scripts/validate_m1_corpus.py
```

Embedding or reranking choices must not be made from the five-record demo.

The [falsifiable test programme](docs/test-programme.md) defines the release gates. `make
gate0` is the deterministic public-push blocker; later benchmark, MLX, security, graph, agent,
and client gates remain explicitly unpassed.

## Authority record

Every version carries:

- stable `record_id` plus an increasing `version`;
- `valid_from` / end-exclusive `valid_to` business time;
- timezone-aware `observed_at` system time;
- `supersedes`, authority status (`approved` or `retired`), owner, sensitivity, and typed ACL
  principals;
- source URI, source span, and SHA-256;
- optional aliases, entities, relations, and reviewed structured policy rules.

`v2` does not rewrite `v1`. At read time, the runtime first resolves the latest applicable
version and only then evaluates approval and access. This ordering prevents stale-version
fallback after retirement or access revocation. System observation timestamps are parsed and
compared as timezone-aware datetimes rather than lexicographically as ISO text.

The included corpus is synthetic. Its hashes cover each canonical demo statement, as declared
by `metadata.hash_scope`. A raw source import uses a record plus `source_artifact` envelope;
the importer reads the local file itself and rejects any supplied digest or artifact metadata
that does not match those bytes.

`gkr ask` stores a content-addressed audit trace in `artifacts/query-traces.sqlite` by default.
The trace records the resolved principals independently of the prompt, request and scope
hashes, temporal scope, both snapshot identities, retriever configuration, evidence
references, decision parse, prompt/candidate hashes, verification results, process peak RSS,
latency, and publication status. It does not duplicate raw prompts or answer bodies; their
hashes verify separately retained content. Actor identity is not sent to the model prompt.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
```

`.venv`, generated SQLite databases, caches, model files, and Git metadata are not source
archive contents. Build archives from tracked files after the repository has an initial
commit, rather than archiving the working directory.

See [ADR 0001](docs/adr/0001-local-first-governed-runtime.md) for the accepted architecture
boundary and [the research basis](docs/research-basis.md) for primary publications behind
the staged roadmap.

## License and contributing

Released under the [MIT License](LICENSE). Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request and [SECURITY.md](SECURITY.md) before reporting a
vulnerability.
