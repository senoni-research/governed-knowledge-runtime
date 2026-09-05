# Local MLX pilot — 2026-09-05

This records an actual synthetic, non-production run on the owner's M4 Max. It is development
evidence, not benchmark evidence, authenticated authorization, human approval, or certification.
M1 Gate 1 remains `not_run`.

## Reproduction

```bash
.venv/bin/python scripts/run_local_pilot.py \
  --generator-model \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --verifier-model \
  ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/96e4e2ca7926fab2eb52b1cdea46a934af630f0a \
  --output-dir artifacts/local-pilot
```

The final clean run used the same command with output directory
`artifacts/local-pilot-20260905-3`. It completed 24 operations in 40.4 seconds: 18 answer
attempts, 15 runtime-published answers, and 3 withheld candidates. Eight published answers used
MLX generation plus the distinct local verifier; seven used the deterministic policy engine.
There were 18 persisted query traces. "Published" below means only that the prototype's current
local checks passed.

## Actual MLX answers

Question:

> What evidence is required before a production deployment?

The Qwen3 generator answered with automated tests, a completed security scan, a named release
owner, a documented rollback plan, and two non-author approvals for high-risk changes. It cited
`ENG-REL-001:v1`. The Llama verifier returned `supported`, producing
`published_local_verifier_supported`. Runtime latency was 4296 ms, peak process RSS was
9.21 GB, and trace ID was
`babc79af3534297ee82584bae560083e072df0a4a4d506c2c0dde3e3632304da`.

The semantic paraphrase:

> Before we ship to production, what proof must the release record contain?

also produced the correct four gates and high-risk approval condition, cited
`ENG-REL-001:v1`, and was published after 3640 ms. The model emitted
`[CITATION: ENG-REL-001:v1]`; the runtime now accepts that common label while still rejecting
references outside the supplied evidence.

## Approved update and history

Before the pilot update, the generated travel-policy answer selected `FIN-EXP-001:v2`: written
approval above £750 and receipt plus business purpose within ten calendar days. After appending
approved `FIN-EXP-001:v3`, effective 2026-10-01, the generated answer selected v3: written
approval above £900 and filing within seven days.

The deterministic £800 checks showed:

- 2026-09-30 after the proposed update was rejected: approval required, citing v2.
- 2026-10-01 after approved ingestion: approval not required, citing v3.
- Historical 2026-09-30 after v3 existed: approval required, citing v2.
- 2026-10-01 with `known_at=2026-09-04`: approval required, citing v2 because v3 was not yet
  known.

The proposed v3 record was rejected at ingestion with `Unsupported status: proposed`. The
approved v3 appended one immutable event; the resulting six-record ledger chain verified.

## Access boundary and abstention

For an actor claiming only `group:employees`, separately compiled context contained
`ENG-REL-001:v1`, `FIN-EXP-001:v2`, and `PUB-HBK-001:v1`; it did not contain the restricted
`SEC-REC-001:v1`. For an actor claiming `group:security-recovery`, context contained
`SEC-REC-001:v1` and the public handbook.

The unauthorized recovery question and an unknown Mars meal-allowance question both produced
diagnostic candidates explaining that the relevant evidence was absent. Both were withheld
because they supplied no citation. The withheld text remained only in `withheld_candidate`.

## Observed limitations

The bounded review found failures that must not be interpreted as certified answers:

- The first deployment answer added the unsupported exhaustive sentence "No other evidence is
  required"; the small local verifier missed it.
- The two-record answer correctly used the handbook and release policy, but cited only
  `ENG-REL-001:v1`, omitting `PUB-HBK-001:v1` for the non-override claim. Citation membership
  does not yet establish claim-to-source completeness.
- The authorized recovery answer broadened "must never be copied into prompts for other actors"
  into the absolute statement "may not be copied"; the verifier missed the broader scope.
- The split-booking answer correctly said splitting is prohibited and cited
  `FIN-EXP-001:v2`, but the local verifier returned `unsupported`, so the runtime withheld it.

These results demonstrate the local runtime, update/history behavior, and fail-closed paths.
They also show that the current small-model publication gate is not reliable enough for
production or accuracy claims.
