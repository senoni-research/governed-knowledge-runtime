# M1 benchmark contract

The M1 schemas are versioned before any embedding or reranking challenger is introduced. The
original v1 and v2 files are retained so their hashes remain auditable. Contract v3 is the
frozen case-authoring contract: it adds required `known_at`, separate question/oracle
authorship and review provenance, `case_kind`, refusal `disposition_reason`, salted public
test commitments, and evidence-set retrieval metrics. Results from different contract
versions must not be merged.

A Grok-authored and Grok-remediated fictional synthetic authority corpus now exists
under `corpus/`. Independent ChatGPT content review concluded `CONTENT_APPROVED`.
Independent Cursor code review concluded `CODE_APPROVED`. That is AI review of
fictional benchmark content, not human or organizational approval. The corpus is
frozen under `corpus-freeze-manifest.json`. Independent ChatGPT contract review
concluded `CONTRACT_APPROVED`. Independent Cursor code review concluded
`CODE_APPROVED`. That is AI review of synthetic research tooling, not owner, legal,
production, or organizational approval. The v3 case, metric, programme, and suite
contracts are now frozen. Scoring cases remain unbuilt. Status: v3 contract frozen;
corpus frozen; Gate 1 not_run. Validate the corpus with:

```bash
.venv/bin/python scripts/validate_m1_corpus.py
```

The frozen v2 contract still requires an owner-labelled suite. Independent ChatGPT
review does not satisfy that v2 requirement. Frozen contract v3 under this directory
governs case authoring for an AI-authored fictional research corpus. Human or owner
approval remains required for real company authority, production promotion, and
answer-quality claims. The 360 development, validation, and test scenarios remain
unbuilt, and Gate 1 remains `not_run`. Gate 1 status will be recorded on a scoring
suite manifest, not in `programme-v3.json`. The Gate 2 execution contract is deferred
and the test split remains unopened. A 16-scenario non-scoring conformance
fixture under `tests/fixtures/m1/` proves the v3 machinery only. Its case provenance
records same-session Grok generation, pending semantic review, and intentionally
incomplete prompt provenance (`prompt_retained=false`); it is not a scoring suite
and cannot support a scoring claim. Paraphrases will share a `scenario_id` and must
not increase the independent-scenario count. Canonical model-family IDs come from
`model-family-registry-v1.json`. Registry checks validate declared provenance only
and cannot prove which external model actually ran.

Every completed scoring suite must contain exact factual, semantic paraphrase,
numeric/conditional, temporal, authorization, unknown/out-of-scope, multi-record, and
adversarial/conflicting scenarios. Exact cross-split duplicates are forbidden; semantic
duplicate candidates require review. Supported and conflicting evidence-bearing cases
require at least one nonempty sufficient set. `unknown_oos` and no-authorized-evidence
refusals require zero sufficient sets. Not every case has one or more sufficient sets.
Scoring cases require completed independent semantic review. Conformance
cases are machinery-only and never scoreable. Tags cannot bypass scoring
requirements.

Development and validation cases contain plaintext questions. Full test cases plus
per-case 32-byte salts are a plaintext staging bundle written only outside the
repository. Plaintext staging is not sealed. Scoring-suite finalization requires a
caller-supplied externally encrypted artifact descriptor; this repository verifies
that file exists outside the tree and matches its declared SHA-256. It does not
decrypt the artifact and does not prove the ciphertext decrypts to the staged bytes.
The public repository may carry only the redacted v3 public form and
`question_commitment_sha256` values. Opening the test split retires it. Publishing
plaintext test questions invalidates any claim that the split remained hidden.

Deterministic oracle validation loads the frozen corpus into a fresh authority store and
checks reference existence, temporal selection, authorization, disposition consistency,
and scoring provenance separation. It does not establish semantic support. Semantic
review is a separate provenance claim on `oracle_review`. Validate authored v3 cases
with:

```bash
.venv/bin/python scripts/validate_m1_oracles.py cases.jsonl
```

Freeze a conformance fixture (plaintext staging, `externally_encrypted_artifact_bound=false`,
no scoring suite manifest) with `--mode conformance`. Scoring freeze requires `--mode scoring`
and an age-x25519-v1 encrypted artifact descriptor. Gate 1 binds no retrieval-configuration
digest and selects no retriever. The in-memory suite manifest is validated before any
staging or public file is published. Replacing a pre-existing staging/public pair is not a
pair transaction.

During v2-compatible construction, validate partial files with:

```bash
.venv/bin/python scripts/validate_m1_cases.py cases.jsonl --allow-incomplete
```

Remove `--allow-incomplete` for the frozen gate; validation then requires exactly 120
independent scenarios per split before continuing with digest rules.
Question digests follow profile ID `gkr-m1-hash-profile-v1` in `hash-profile-v1.json`: Unicode NFKC, every Unicode whitespace
run to U+0020, trim, Unicode case-fold, UTF-8, then SHA-256. Development, validation,
and test split hashes use the same case_id ordering. Question commitments are SHA-256
of `salt_bytes || utf8(normalized_question)` with a 32-byte salt that never appears in
the public form. The validator also rejects duplicate case IDs, scoped exact duplicate
questions, scenario identity drift (split, class, case kind, scope, oracle, oracle
authorship, or review), and references that are simultaneously sufficient and forbidden.

Security gates are absolute: unauthorized exposure and stale-version exposure must remain
zero over the complete returned evidence list. Retrieval and answer metrics are reported by
query class, alongside latency, process peak RSS, and actual prompt-token counts when
tokenizer integration is available.

`programme.json` is the frozen v2 programme and still reports the corpus and all three
splits as `not_run`. `programme-v3.json` is the frozen v3 policy and has no mutable
pass-status field. The current M0 runner remains a temporal and authorization regression
check. It is not an M1 quality benchmark and must not be used to select an embedding model.
