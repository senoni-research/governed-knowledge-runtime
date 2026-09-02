# M1 synthetic authority corpus

This directory holds the M1 authority corpus for Lumenport Civic Works, a fictional
harbor-and-borough services cooperative. The records exist only as benchmark material.
They are not company policy, legal advice, or an operational runbook for any real
organization.

Grok authored and remediated this fictional corpus. Independent ChatGPT content review
concluded `CONTENT_APPROVED`. Independent Cursor code review concluded `CODE_APPROVED`.
That is AI review of fictional benchmark content, not human or organizational approval.
The frozen v2 owner-labelled requirement is unchanged and unmet. A later versioned
contract is required before replacing owner labelling with AI-reviewed synthetic data
for promotion. The 360 scenarios and oracles are unbuilt, and Gate 1 remains `not_run`.
The corpus status is `reviewed`, which is not `frozen`.

## Synthetic organization

Lumenport Civic Works operates a small passenger ferry, quay infrastructure, and shared
civic offices. Eight departments publish authority:

| Domain | Desk |
| --- | --- |
| finance | Finance Ledger Office |
| engineering | Engineering Standards Desk |
| security | Harbor Security |
| people | People Operations |
| procurement | Procurement Desk |
| legal | Counsel's Office and Counsel Risk Unit |
| operations | Quayside Operations and Quayside Archives |
| it | Service Management |

Identifiers, policy titles, aliases, and dollar or sterling figures are invented. Source
URIs use `synthetic://m1/` only.

## Why the corpus is shaped this way

Later M1 questions need more than a five-record demo. This set includes:

- Exact identifiers and deliberately similar titles, such as travel approval versus travel
  insurance, leave accrual versus leave carry-over, and a production release gate versus a
  production access request.
- Versioned records so `as_of` and `known_at` can select different immutable events.
- Future-effective versions that must stay hidden at the 2026-09-02 anchor date, including
  a current summer Friday-night roster superseded by a winter version.
- Approved retirements that state only withdrawal and effective date, with no replacement
  policy in the retired row.
- ACL transitions from employee-visible text to restricted or secret principals.
- Temporal corrections whose business-valid start precedes the observation timestamp.
- Structured `PolicyRule` objects in GBP, USD, and EUR, with mixed comparators. Conditions
  and exceptions stay fail-closed unless the corresponding prose explicitly requires them.
- Typed relations that are direct abstractions of their statements.
- Four intentional conflict groups, discoverable from the claims: retention, vendor
  insurance, overtime premiums, and weekend production changes. Fixture labels stay in
  metadata only.
- Authorized hostile evidence that contains misleading wording plus a natural-language
  disclaimer. Those statements are data, never executable instructions.
- Plausible out-of-scope material such as tide-gate inspections and pontoon lost property.

Record metadata does not encode hidden questions or expected answers. Keys such as
`fixture_kind`, `conflict_group`, and `adversarial` are bookkeeping for validation. Current
BM25 search text and compiled evidence prompts exclude metadata.

## Manifest count units

These coverage fields count distinct stable `record_id` values, not events, rules,
relations, or conflict pairs:

- `relationship_bearing_records`
- `structured_rule_records`
- `conflict_records`
- `adversarial_evidence_records`

The validator also reports event-level relation and rule totals separately.

## Files

- `authority.jsonl` — immutable approved or retired authority events, ordered by
  `record_id` then `version`.
- `corpus-manifest.json` — counts, coverage, the SHA-256 of `authority.jsonl`, and a
  content digest over ordered record references and statement hashes.
- `../corpus-manifest.schema.json` — versioned manifest schema
  `gkr-m1-corpus-manifest-v1`. Frozen v1/v2 case contracts are unchanged.

## Regenerating and validating hashes

Every event uses `metadata.hash_scope = statement`. The `source_hash` is SHA-256 of the
exact statement string after `KnowledgeRecord` trimming.

The content digest is SHA-256 of this canonical JSON, with sorted keys and compact
separators:

```json
{"records":[{"reference":"FIN-EXP-THRESHOLD:v1","source_hash":"..."}]}
```

Regenerate the JSONL, statement hashes, and manifest from the deterministic builder:

```bash
.venv/bin/python scripts/validate_m1_corpus.py --write
```

`--write` stages each file in a same-directory temporary, flushes and `fsync`s it, then
replaces the destination. Authority is published first and the manifest last, so the
manifest is the commit marker. The two replacements are not a transaction: a crash between
them can leave a new authority file beside an old manifest. Validation must reject that
mismatched pair. The guarantee is that no individual file is left half-written.

Validate the committed files, recompute hashes and counts, and ingest the corpus into a
fresh temporary ledger:

```bash
.venv/bin/python scripts/validate_m1_corpus.py
```

Do not hand-edit a hash. Change the statement or builder, then rerun `--write`.
