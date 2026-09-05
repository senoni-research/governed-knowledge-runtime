# Local claim-contract pilot — 2026-09-05

This is a bounded development rerun on the owner's M4 Max. It fixes observed pilot defects;
it is not an accuracy result, verifier certification, authenticated authorization test, or
production approval. M1 remains paused and Gate 1 remains `not_run`.

## Reproduction

```bash
.venv/bin/python scripts/run_local_pilot.py \
  --generator-model \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --verifier-model \
  ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct/snapshots/96e4e2ca7926fab2eb52b1cdea46a934af630f0a \
  --output-dir artifacts/local-pilot-20260905-9
```

The clean rerun completed 24 operations in 39.5 seconds. It made 18 answer attempts:
16 were runtime-published and two were structured abstentions. Nine published answers used
MLX generation plus a distinct local verifier; seven used deterministic policy evaluation.
No operation had an unexpected exit code. "Published" means only that the prototype's local
checks passed.

## Before and after the four observed failures

### Unsupported exhaustive statement

Before, the release answer added: "No other evidence is required." The source did not
establish that, but the whole-answer verifier returned `supported`.

After, the generator produced one structured claim with `ENG-REL-001:v1` and an exact passage.
The runtime rendered only:

> A production release requires passing automated tests, a completed security scan, a named
> release owner, and a documented rollback plan. [ENG-REL-001:v1]

The unsupported exhaustive sentence was not released. Trace:
`be06a04d966c786cc52e0fc34b56214001384e9e5c06c506cbcedd86397d10bc`.

### Multi-record source coverage

Before, a two-record answer made both handbook and release-policy claims but cited only
`ENG-REL-001:v1`.

After, the direct two-record question produced two separately bound claims:

- handbook behavior and non-override scope cited `PUB-HBK-001:v1`;
- release gates and linked evidence cited `ENG-REL-001:v1`.

Both exact passages were retained in the JSON result, and the verifier returned a result for
each claim index. Trace:
`9a292fb7b6a1aadec62c0b148ec85eba72451348811ec0a4b6a89b9086cf035f`.

### Conditional restriction scope

Before, "must never be copied into prompts for other actors" became an absolute statement that
the material "may not be copied."

After, the released claim preserved the complete condition verbatim:

> Cryptographic recovery material is available only to members of the security recovery group
> and must never be copied into prompts for other actors. [SEC-REC-001:v1]

Trace: `b4d82f86750a71c8723aee9d7db51672ad2cfaebe3bad86d65cdec05fe392179`.

### Correct split-booking prohibition

Before, the correct prohibition was withheld because the whole-answer checker returned
`unsupported`.

After, the verifier evaluated the single paired claim and passage, returned `supported`, and
the runtime released the prohibition with `FIN-EXP-001:v2`. Trace:
`26e3fc8641b94796cbbe155026ba53ba950bf181ec4e06f9b28b1d9e7568fbaf`.

## Abstention, updates, history, and access filtering

The unknown Mars allowance question and unauthorized recovery question both returned exactly:

> I cannot establish that from the evidence available to me.

Their status was `abstained_insufficient_evidence`, citation integrity was `not_applicable`,
and neither result exposed restricted-record existence or model-written explanations. The
unauthorized evidence bundle did not contain `SEC-REC-001:v1`.

The approved travel update still changed the deterministic £800 decision from approval
required under `FIN-EXP-001:v2` to not required under `FIN-EXP-001:v3`. Historical and
pre-observation queries continued to select v2. Generated receipt-deadline answers selected
ten days under v1/v2 and seven days under v3, retaining the receipt and business-purpose
conditions.

## Personal project-document session

The interactive wrapper was exercised against six manually prepared records from five
non-sensitive, public project documents: the README, architecture ADR, test programme,
research basis, and local-session guide. The records, databases, and traces remain under the
ignored local `artifacts/owner-project-session/` directory.

Before the approved local document update, a 2026-09-04 question selected
`GKR-ANSWER-001:v1` and described citation plus distinct-judge publication. After appending
v2 effective 2026-09-05, the current question selected `GKR-ANSWER-001:v2` and described
record-and-passage-bound claims. Repeating the historical question after the update still
selected v1.

The session also answered the MLX/authority-store distinction and unrun-gate reporting, while
displaying passages, dates, non-certifying status labels, elapsed time, and trace IDs.

One concrete failure remains: an early multi-part question asking both where knowledge is
stored and what MLX does answered only the storage part. Asking the MLX part directly
succeeded. The current checks establish support for rendered claims; they do not generally
prove that every requested subpart was answered.

## Remaining limits

- The distinct 3B local verifier remains fallible. `published_local_verifier_supported` is
  not "verified correct."
- Exact passage membership and a conservative gate requiring every normalized material claim
  term to occur in the passage reject obvious wrong bindings, but neither is general semantic
  proof. This can withhold legitimate paraphrases.
- General answer completeness is not certified; the attempted small-model completeness check
  was too unreliable to add as a publication gate.
- Actor/group configuration remains a development simulation.
- No embeddings, reranking, graph, agent, training, Cline integration, or M1 work was added.
