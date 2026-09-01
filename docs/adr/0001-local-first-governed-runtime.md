# ADR 0001: Local-first governed knowledge runtime

- Status: Accepted
- Date: 2026-09-01

## Context

A preceding parametric-memory experiment tested whether bounded company knowledge
could be acquired reliably in MLX LoRA adapters. Its smoke candidate did not approach the
full-context baseline. More importantly, mutable facts in weights are difficult to update,
authorize, delete, date, and prove during an audit.

The target machine is an Apple Silicon Mac with enough unified memory to run capable local
models. The eventual client is Cline in VS Code, supplied through an optimized local prompt
or tool interface after the primary knowledge runtime is evaluated.

## Decision

Company knowledge is stored as external, immutable, source-bound record versions. MLX is an
optional local compute provider, not the authority store.

The runtime is layered:

1. A local append-only SQLite authority ledger stores only approved/retired record versions;
   proposal workflow events live in a separate append-only store and have no authority version.
2. Temporal resolution evaluates both business valid time and system observation time.
3. Approval and actor authorization reduce the candidate set before policy matching or
   retrieval.
4. A fail-closed parser may construct a typed request from one approved structured policy rule
   in the complete authorized corpus. The matched authority record is pinned into evidence
   before deterministic evaluation and citation verification.
5. Other queries use all authorized context when it fits and local retrieval when it does not.
6. A context compiler emits bounded evidence with exact source and version references.
7. A local MLX model generates an answer when no deterministic decision applies.
8. Deterministic citation and contradiction checks plus a distinct local model judge gate
   publication. Same-model self-approval is diagnostic only.
9. Claim-level semantic support and provenance tracing are added after evaluation.
10. Authority snapshots, selected evidence bundles, and durable query executions have distinct
    identities.

There will be no cloud inference provider in the primary runtime. Model acquisition may use
the network as an explicit setup action; operational inference, indexing, retrieval, storage,
and verification remain on the Mac.

Model weights may later encode stable skills: query decomposition, tool use, temporal
reasoning, evidence discipline, and refusal behavior. They do not become the authoritative
store for changing company facts.

## Authority semantics

Record versions are append-only authority events. A new version names the immediately
preceding version in `supersedes`; old rows are never edited to simulate a closed interval.
At a requested valid date and known-at time, the runtime selects the greatest applicable
version for each stable record ID. It then applies authority status and ACL filtering.
Observation timestamps are parsed and compared as timezone-aware datetimes; ISO text ordering
is not used for transaction-time decisions.

Proposed and rejected content cannot enter this resolution algorithm. A proposal records
candidate content, review events, and an optional base authority reference without allocating
`version`, `supersedes`, `observed_at`, or authority status. Owner approval is a workflow
decision; a separate, source-verified promotion step must still append the resulting authority
event.

This order is deliberate. Filtering before version resolution could let a user fall back to
an older version after a later version retires the fact or removes that user's access.

The ledger's hash chain detects ordinary mutation but is not a digital signature or protection
against an administrator replacing the whole database and all checkpoints. Signed snapshots
and external checkpoint anchoring are future controls if the threat model requires them.
Chain verification also compares every temporal/index column used by queries with its hashed
payload value. Source hashes are recomputed from the canonical statement or a locally opened
raw source artifact before an authority transaction is appended.
SQLite writers acquire an immediate write transaction before reading the chain tip, and a
unique-successor constraint prevents two events from extending the same hash. The proposal
and query-trace chains use the same serialization rule.

An authority snapshot identifies the complete temporally valid, authorized corpus. An
evidence-bundle identity separately covers selected records, scores, retriever identity, and
configuration. Every answer execution records those identities and its request, scope,
decision, model metadata, prompt/candidate hashes, verification outcomes, resource
observations, and publication status in a separate append-only trace store.

## Staged implementation

The accepted sequence is:

1. Authority/proposal separation, source verification, typed rules, traceability, temporal
   resolution, authorization-first corpus, and full-context baseline.
2. Local lexical retrieval and frozen retrieval/answer/security evaluations. The M1 schema is
   frozen before its owner-labelled corpus is assembled.
3. Exact-token full-context budgeting.
4. Local embeddings, rank fusion, and reranking, retained only if M1 evals improve.
5. Source diffing and AI-proposed records with deterministic checks and owner approval.
6. Temporal graph retrieval for measured historical and multi-hop cases.
7. Bounded search/find/open agent tools for cases where one-shot retrieval fails.
8. Claim-level support and provenance tracing.
9. A local interface for Cline, likely MCP or a small loopback service, after runtime behavior
   and prompt contracts stabilize.

Each advanced path is a challenger to simpler baselines, not an assumed replacement.

## Consequences

Knowledge updates do not require model training. Every answer can identify its knowledge
snapshot and source record versions. Authorization and historical reconstruction are
testable without an LLM. The core works before MLX-LM is installed.

The milestone CLI accepts actor and group claims to exercise policy behavior. It is not an
authentication boundary. A serving process must derive principals from trusted local state
before the runtime can protect records from other users or client extensions.

The cost is additional application architecture: connectors, approvals, indexes, graph
maintenance, and verification must be built explicitly. A local model judge is fallible and
does not provide atomic evidence trails, so the first milestone must not be described as
complete claim verification.
