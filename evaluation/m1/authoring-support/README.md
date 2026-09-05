# M1 authoring-support schemas

These schemas are operational tooling for role-separated question, oracle, and
review packets. They are **not** part of the frozen v3 contract and must not be
added to `contract-manifest-v3.json`.

The frozen case, metric, programme, suite, hash, encryption, and model-family
artifacts remain the only normative v3 contract files. Gate 1 remains `not_run`
until a later scoring-suite finalizer publishes the public suite directory.

Semantic review assembly requires
`semantic-review-artifact-v2.schema.json`. Each result binds the canonical
full oracle-draft row it reviewed. Multiple batches may be supplied: assembly
selects exactly one matching approval per current case, ignores stale
content bindings, and fails closed on a matching block, duplicate approval,
missing approval, or split mismatch. Scenario variants must share one review
artifact so their frozen `oracle_review` identity remains equal.

`semantic-review-artifact-v1.schema.json` is retained only to describe
historical authoring output. The assembler rejects v1 because its split-level
question digest did not bind oracle answers, evidence sets, claims, citations,
publication decisions, or dispositions.

Use `scripts/build_m1_review_bindings.py` to produce the case IDs and canonical
content digests for a full or partial v2 review batch. That output is only a
binding scaffold; it is not semantic approval.
