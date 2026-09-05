# Experimental result index

These reports describe a non-production local development milestone. They are not benchmark
scores, human approval, verifier certification, or evidence that an answer is certified
correct.

## Measured source snapshot

The strong-model measurements and article links are pinned to experimental source revision
[`419fcd276a638048e65682146868fc2ae30d7a91`](https://github.com/senoni-research/governed-knowledge-runtime/commit/419fcd276a638048e65682146868fc2ae30d7a91).
Later publication-only cleanup does not change that measured revision. No production release
is implied, and this cleanup PR does not create a source-release tag before review.

## Results in chronological order

1. [Original local MLX pilot](../local-pilot-2026-09-05.md) — proved the local execution,
   update, history, and access-filtering path while exposing four answer/checking failures.
   This is the intentionally retained flawed baseline.
   [Pinned report at the measured revision](https://github.com/senoni-research/governed-knowledge-runtime/blob/419fcd276a638048e65682146868fc2ae30d7a91/docs/local-pilot-2026-09-05.md).
2. [Corrected claim-contract pilot](../local-claim-pilot-2026-09-05.md) — changed the answer
   contract and added per-claim evidence binding plus structured abstention. Comparisons with
   the original pilot therefore include a runtime-contract change, not merely a model change.
   [Pinned report at the measured revision](https://github.com/senoni-research/governed-knowledge-runtime/blob/419fcd276a638048e65682146868fc2ae30d7a91/docs/local-claim-pilot-2026-09-05.md).
3. [Larger-model comparison](../local-strong-model-pilot-2026-09-05.md) — reused the corrected
   claim contract, questions, evidence, and update sequence with Qwen3.8-27B as generator and
   Gemma-4-31B as verifier.
   [Pinned report at the measured revision](https://github.com/senoni-research/governed-knowledge-runtime/blob/419fcd276a638048e65682146868fc2ae30d7a91/docs/local-strong-model-pilot-2026-09-05.md).

## Required interpretation

For the corrected claim-contract and larger-model runs:

Runtime-published is not certified correct.

The 16 published outcomes comprise nine model-generated answers plus seven deterministic
answers.

Two additional outcomes are structured abstentions.

The eight-case replay is a curated regression check.

Pilot wall times include model loading in fresh processes; warm-session latency is not
measured there.

Actor/group assertions are not authentication.

## Portable reproduction material

- [Eight exported synthetic replay cases](../../examples/verifier-replay/cases.jsonl) contain
  the exact historical candidates, public supporting passages, expected diagnostic verdicts,
  and provenance hashes. They are selected regressions, not held-out accuracy evidence.
- [Tested strong-pilot environment](../../reproduction/strong-pilot-2026-09-05.json) records
  the successful Mac, Python, dependency, model-revision, precision, and generation settings.
- [Targeted runtime constraints](../../reproduction/strong-pilot-2026-09-05-constraints.txt)
  pin the packages used for that run without pretending to be a complete operating-system
  lock.

Recreate the tested Python layer on a compatible Apple Silicon Mac:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,mac]" \
  -c reproduction/strong-pilot-2026-09-05-constraints.txt
```

Validate the public fixture without loading a model:

```bash
.venv/bin/python scripts/replay_saved_verifier_cases.py --validate-only
```

Run the model-heavy diagnostic from a fresh clone after separately acquiring the pinned Gemma
checkpoint:

```bash
.venv/bin/python scripts/replay_saved_verifier_cases.py \
  --verifier-model models/gemma-4-31b-it-8bit \
  --verifier-revision f5f3dc92ab4af76724c36c21eb6bedadb3a851be \
  --output artifacts/verifier-replay.json \
  --max-tokens 256
```

An owner retaining the original ignored result directories can additionally supply
`--artifacts-dir artifacts`; the script then verifies those files and candidates against the
published hashes before replaying them.

M1 remains paused and Gate 1 remains `not_run`. Its unfinished research history is preserved
under [`evaluation/m1/`](../../evaluation/m1/).
