# M1 benchmark contract

The M1 schemas are frozen before any embedding or reranking challenger is introduced. The
original v1 files are retained so their hashes remain auditable. The current v2 contract adds
independent scenario IDs, linked paraphrase variants, development/validation/test splits,
question hashes, independent-authorship metadata, alternative sufficient oracle-evidence
sets, a sealed-suite manifest, and retrieval-specific metrics. Results from different contract
versions must not be merged.

The benchmark corpus and reviewed cases are intentionally not invented from the five-record
demo. They must be assembled and owner-labelled as the next evaluation task: 120 independent
development scenarios, 120 validation scenarios, and 120 sealed test scenarios over at least
40 synthetic authority records. Paraphrases share a `scenario_id` and do not increase the
independent-scenario count.

Every completed suite must contain exact factual, semantic paraphrase, numeric/conditional,
temporal, authorization, unknown/out-of-scope, multi-record, and adversarial/conflicting
scenarios. Exact cross-split duplicates are forbidden; semantic duplicate candidates require
review. Each case carries one or more sufficient oracle-evidence sets so retrieval failures
remain distinguishable from generation failures.

Development and validation cases contain plaintext questions. A public test case may omit
`question` and retain only `question_sha256`; the runnable sealed test artifact remains outside
the public tuning surface. Publishing plaintext test questions invalidates any claim that the
split remained hidden.

During corpus construction, validate partial files with:

```bash
.venv/bin/python scripts/validate_m1_cases.py cases.jsonl --allow-incomplete
```

Remove `--allow-incomplete` for the frozen gate; validation then requires exactly 120
independent `scenario_id` values in each split and all eight classes. Question digests use
Unicode NFKC normalization, collapsed whitespace, trimming, and case folding before SHA-256.
The validator also rejects duplicate case IDs, exact duplicate questions, scenarios crossing
splits, and references that are simultaneously sufficient and forbidden.

Security gates are absolute: unauthorized exposure and stale-version exposure must remain
zero. Retrieval and answer metrics are reported by query class, alongside latency, process
peak RSS, and actual prompt-token counts when tokenizer integration is available.

`programme.json` deliberately reports the corpus and all three splits as `not_run`. The current
M0 runner remains a temporal and authorization regression check. It is not an M1 quality
benchmark and must not be used to select an embedding model.
