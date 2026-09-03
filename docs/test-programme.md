# Falsifiable test programme

The runtime is tested as independent gates rather than one subjective chatbot demonstration.
Passing a higher gate never excuses a failure below it. New retrieval complexity is promoted
only when it beats a simpler path on a frozen subset without weakening temporal or
authorization guarantees.

## Release meanings

- A public source release means an early research prototype whose deterministic foundation
  passes Gate 0. It does not mean production readiness.
- A retrieval candidate requires Gates 0–2.
- A credible answer-quality claim requires Gates 0–3 and blinded review.
- A read-only product pilot requires all applicable gates through Gate 6 plus the service
  isolation checks in Gate 8.

Unrun gates are reported as `not_run`, never inferred from unit-test success.

## Gate 0 — Authority correctness

Status: automated public-push blocker.

Run:

```bash
make gate0
```

This gate covers:

- approved, pending, rejected, future-effective, superseded, retired, restricted, and
  retroactively corrected authority states;
- concurrent proposals that cannot alter current authority;
- version-before-ACL resolution and historical reconstruction by `as_of` and `known_at`;
- typed numeric policy requests above and below a threshold;
- fail-closed parsing for explicit negation, false-negation, conditionals, multiple values,
  unclear subjects, unevaluated conditions or exceptions, and competing rules;
- source-byte recomputation, source-version validation, duplicate-event rejection, indexed
  column checks, payload tampering, earlier-event replacement, and serialized hash-chain
  writers;
- reconstructable query traces and CLI failure exit codes.

Explicit negation is intentionally not answered deterministically in the current parser.
Questions such as “Does £800 travel spend not require approval?” must return `ambiguous` and
may proceed to evidence-based MLX reasoning. A deterministic “No” would contradict the
previously accepted fail-closed boundary unless typed negative polarity is implemented and
separately proven.

Exit conditions:

- proposals cannot change approved authority;
- retired or restricted updates cannot expose stale permissive versions;
- ambiguous language cannot publish an inverted deterministic answer;
- source and ledger tampering is detected;
- every `ask` execution has a content-addressed audit trace.

The trace reconstructs request scope, authority/evidence identities, component configuration,
and publication decisions. It stores prompt and candidate hashes rather than duplicating raw
evidence or answer text. Exact output recovery therefore requires the separately retained
content; exact MLX replay is not claimed.

The local hash chain detects mutation or partial event replacement. It cannot detect a
privileged administrator replacing and recomputing the entire unsigned database. Signed
checkpoints or external anchoring belong to the later security threat model; the prototype
must not claim that property.

## Gate 1 — Frozen M1 retrieval benchmark

Status: v3 contract frozen; corpus frozen; Gate 1 not_run.

The v3 contract under `evaluation/m1/` is frozen
(`v3 contract frozen; corpus frozen; Gate 1 not_run`).
The corpus is frozen. The 360 scoring cases are unbuilt, so this gate stays
`not_run`. It requires:

- 120 independent development scenarios;
- 120 independent validation scenarios;
- 120 independent test scenarios, staged as plaintext outside the repository and
  sealed only by an externally encrypted artifact for a scoring suite;
- 15 independent scenarios per query class per split;
- at least 40 synthetic authority records;
- linked paraphrase variants that do not inflate scenario counts;
- exact and reviewed semantic cross-split deduplication;
- independently authored questions with hashes and required `known_at`;
- separate question-author, oracle-author, and review provenance;
- supported and conflicting evidence-bearing cases have at least one nonempty
  sufficient set; OOS and no-authorized-evidence refusals have zero.

The eight required classes are exact factual, semantic paraphrase, numeric/conditional,
temporal, authorization, unknown/out-of-scope, multi-record, and
adversarial/conflicting.

A Grok-authored and Grok-remediated fictional synthetic authority corpus now exists
under `evaluation/m1/corpus/`. Independent ChatGPT content review concluded
`CONTENT_APPROVED`. Independent Cursor code review concluded `CODE_APPROVED`. That
review is of fictional benchmark content. It is not human or organizational approval.
The corpus is frozen under `evaluation/m1/corpus-freeze-manifest.json`. Frozen
contract v3 governs case authoring for this fictional research corpus. Human or
owner approval remains required for real company authority, production promotion,
and answer-quality claims. The 360 scoring scenarios and oracles remain unbuilt, so
this gate stays `not_run`. Gate 1 status belongs on a scoring suite manifest, not
`programme-v3.json`. Gate 1 freezes corpus, cases, oracles, split commitments,
and metrics only; it selects no retriever. The Gate 2 execution contract is deferred
and the test split remains unopened. Deterministic oracle validation checks
corpus-grounded invariants only and is not semantic proof. The 16-case
conformance fixture is non-scoring and records pending semantic review and
intentionally incomplete prompt provenance in its own case files. Plaintext
test staging is not a sealed bundle; sealing requires an external encrypted
artifact. Validate the corpus with `.venv/bin/python scripts/validate_m1_corpus.py`.
Real company authority still requires an organization-defined trusted approval
process.

## Gate 2 — Retrieval bake-off

Status: blocked by Gate 1.

Compare full authorized context, BM25, two local dense candidates, rank fusion, one local
reranker, and oracle evidence. Tune only on validation data. Open the test split only after
thresholds are frozen.

Unauthorized exposure and stale-current-version retrieval have zero tolerance. Other metrics
include recall at 1 and 5, MRR, nDCG, wrong and empty retrieval rates, out-of-scope loading and
rejection, latency, build/update time, and peak memory. Gate 1 defines those metrics but
selects no retriever. Gate 2 must introduce a separately versioned retrieval-execution
contract, freeze it from validation before sealed test opening, and bind its verified digest
in every Gate 2 trace or result. After that validation, one global candidate arm is selected:
the lowest-complexity arm that meets every frozen validation threshold and safety gate.
Per-class metrics are mandatory diagnostics, not separate winners.

## Gate 3 — Answering and verification

Status: blocked by Gate 2.

Use the same pinned MLX model revision, prompt, zero temperature, output budget, question, and
scope across no-evidence, full-context, selected-retrieval, reranked, and oracle arms. Only the
evidence changes.

Grade deterministic numeric/date/negation errors before atomic claim support, citation
coverage, semantic review, and publication decisions. A semantic judge cannot override a
deterministic contradiction. Report the oracle gap to separate retrieval failures from
generation or verification failures.

## Gate 4 — Updates and supersession

Status: not run.

Replay proposal, approval, future-effective activation, current use, and historical lookup
sequences. Measure approval-to-visibility time, stale-current answers, historical accuracy,
incremental indexing latency, impacted regressions, and rollback behavior. Updating authority
must not require model retraining.

## Gate 5 — Security and isolation

Status: partially covered by deterministic unit tests; full gate not run.

Exercise asserted-identity abuse, cross-department access, stale permissive fallback,
restricted data in logs or verifier prompts, source prompt injection, conflicting or
low-trust sources, unapproved extraction, false provenance, malformed inputs, component
failures, and resource exhaustion. The normal safe failure is no publishable answer plus an
authoritative-source escalation.

## Gate 6 — Mac and MLX operation

Status: not run on the candidate model matrix.

On the target M4 Max, record model and tokenizer fingerprints, cold load, warm latency,
throughput, p50/p95 latency, active/cache memory, prompt/output lengths, and power where
available. Test one, two, and four concurrent requests over small, 8K, 32K, and maximum
supported contexts. Missing models, revision mismatch, interruption, out-of-memory, malformed
output, verifier disagreement, and empty output must produce controlled statuses.

## Gate 7 — Graph and agentic challengers

Status: prohibited until simpler frozen baselines reveal a measured gap.

Graph expansion is limited to temporal and multi-record subsets. Bounded agentic retrieval is
limited to complex-navigation scenarios with explicit tool, document, hop, token, and time
budgets. Neither is promoted from an impressive single demonstration.

## Gate 8 — Read-only client pilot

Status: prohibited until identity and process boundaries exist.

A later local service may expose read-only search, answer, excerpt, version comparison, and
evidence explanation operations. The client cannot choose identity or groups, read SQLite,
modify authority, or bypass logging. Turning off the service must remove access completely.
