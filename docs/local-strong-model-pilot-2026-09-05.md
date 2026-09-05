# Strong-model local pilot — 2026-09-05

This is a bounded quality-first rerun of the existing synthetic pilot on the owner's M4 Max
with 128GB unified memory. It changes no questions, evidence, policy-update sequence, M1
artifact, or publication rule. It is development evidence, not an accuracy result or verifier
certification.

## Pinned local models

- Generator: `mlx-community/Qwen3.8-27B-8bit`, revision
  `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` (29.5GB package).
- Verifier: `mlx-community/gemma-4-31b-it-8bit`, revision
  `f5f3dc92ab4af76724c36c21eb6bedadb3a851be` (33.8GB package).
- Runtime: `mlx-lm` 0.31.3.
- Both models ran with `enable_thinking=false`. Generated reasoning delimiters are separated
  from final content; an unclosed reasoning channel or token-limit finish is an execution
  error.

The exact acquisition and run commands were:

```bash
.venv/bin/hf download mlx-community/Qwen3.8-27B-8bit \
  --revision 815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 \
  --local-dir models/Qwen3.8-27B-8bit
.venv/bin/hf download mlx-community/gemma-4-31b-it-8bit \
  --revision f5f3dc92ab4af76724c36c21eb6bedadb3a851be \
  --local-dir models/gemma-4-31b-it-8bit

.venv/bin/python scripts/run_local_pilot.py \
  --generator-model models/Qwen3.8-27B-8bit \
  --generator-revision 815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 \
  --verifier-model models/gemma-4-31b-it-8bit \
  --verifier-revision f5f3dc92ab4af76724c36c21eb6bedadb3a851be \
  --output-dir artifacts/local-pilot-strong-models
```

The local result directory is separate from and does not alter the small-model baseline at
`artifacts/local-pilot-20260905-9`.

## Execution outcome

The run completed all 24 operations in 351.9 seconds with no unexpected exit code, model
error, or truncated generation. Its 18 answer attempts produced:

- nine claim-bound MLX answers accepted by the distinct local verifier;
- seven deterministic policy-rule answers;
- two structured abstentions;
- no withheld candidate in this particular run.

`published_local_verifier_supported` still means only that the development checks passed. It
does not mean verified correct.

## Actual strong-model answers

- `01_release_requirements` (48.591s):
  - A production release requires passing automated tests, a completed security scan, a named
    release owner, and a documented rollback plan. `[ENG-REL-001:v1]`
  - A high-risk change additionally requires approval from two people who did not author the
    change. `[ENG-REL-001:v1]`
  - Evidence for every gate must be linked in the release record before deployment begins.
    `[ENG-REL-001:v1]`
- `02_release_paraphrase` (47.511s): returned the same three separately bound release claims.
- `03_high_risk_release` (32.203s): a high-risk change additionally requires approval from two
  people who did not author the change. `[ENG-REL-001:v1]`
- `04_travel_policy_before_update` (32.511s): submit an itemised receipt and business purpose
  within ten calendar days after the trip. `[FIN-EXP-001:v2]`
- `05_travel_policy_historical` (32.109s): the same ten-day requirement from
  `[FIN-EXP-001:v1]`.
- `06_two_record_question` (38.569s):
  - the public handbook is public, directs readers to approved policy owners, and does not
    override referenced policies. `[PUB-HBK-001:v1]`
  - evidence for every gate must be linked before deployment. `[ENG-REL-001:v1]`
- `07_unknown_question` (13.964s): “I cannot establish that from the evidence available to
  me.”
- `08_denied_recovery_answer` (10.327s): the same fixed abstention, with no restricted record
  in the permitted evidence.
- `11_authorized_recovery_answer` (29.112s): cryptographic recovery material must never be
  copied into prompts for other actors. `[SEC-REC-001:v1]`
- `15_split_booking_safeguard` (33.070s): an item must not be split into smaller transactions
  to avoid the threshold. `[FIN-EXP-001:v2]`
- `22_travel_policy_after_update` (33.232s): submit an itemised receipt and business purpose
  within seven calendar days after the trip. `[FIN-EXP-001:v3]`

The deterministic update checks continued to select v2 before the approved effective date,
v3 afterward, and v2 for historical and pre-observation queries. The proposed update was
rejected and the access-filtering context remained unchanged.

## Latency and memory versus the smoke baseline

The small Qwen3-4B/Llama-3.2-3B run remains the reproducible smoke baseline. Both runs had the
same status counts: 16 published answers, two abstentions, and no withheld candidate.

- Small-pair total operation time: 39.377s; mean model-case wall time: 3.518s; median: 3.593s;
  maximum: 5.172s.
- Strong-pair total operation time: 351.896s; mean model-case wall time: 31.927s; median:
  32.511s; maximum: 48.591s.
- Strong Qwen decode rate: mean 16.155 tokens/s, range 13.517–17.863 tokens/s.
- Strong Gemma decode rate across 14 claim checks: mean 15.368 tokens/s, range
  15.302–15.412 tokens/s.
- Qwen-only MLX peak: 30.919GB.
- Combined generator-plus-verifier MLX peak: 61.602GB.
- Maximum recorded process RSS: 48.125GiB. This is lower than MLX's Metal allocation metric
  and should not be substituted for it.
- Small-pair maximum recorded process RSS: 8.592GiB. The earlier adapter did not record a
  comparable MLX peak.

The stronger generator was more complete on the direct release question and more concise on
the split-booking question. It did not settle general completeness: the two-record answer
said that every gate's evidence must be linked but did not enumerate those gates, while the
small-model answer did. That remains a concrete “missed requested detail” issue rather than a
reason to add a new benchmark.

## Saved-answer verifier replay

Gemma was also run once, without reloading between cases, against four candidates saved from
the original faulty pilot and four corrected candidates saved from the claim-contract pilot:

```bash
.venv/bin/python scripts/replay_saved_verifier_cases.py \
  --verifier-model models/gemma-4-31b-it-8bit \
  --verifier-revision f5f3dc92ab4af76724c36c21eb6bedadb3a851be \
  --output artifacts/local-pilot-strong-models/verifier-replay.json \
  --max-tokens 256
```

The replay used the complete synthetic records, not statement-only excerpts. Gemma:

- rejected the saved release answer containing “No other evidence is required or implied”;
- rejected the saved two-record answer with no handbook citation;
- rejected the saved recovery answer's unsupported exhaustive conclusion;
- accepted the correct split-booking answer that the old checker rejected;
- accepted all four corrected claim-bound candidates.

All eight matched the expected diagnostic outcome. There was no execution error or
truncation; standalone Gemma replay peaked at 33.981GB MLX memory. This fixed replay is useful
evidence that the larger checker did not repeat the four observed decisions. Eight curated
cases do not establish reliability on new phrasing or domains.

## Remaining limits

- The stronger verifier remains a fallible model and is not independent human approval.
- The strong-pair status counts do not constitute an accuracy comparison because the pilot is
  a fixed development sequence.
- General answer completeness remains unchecked.
- Each pilot question starts a fresh CLI process and reloads the models, so this timing is
  intentionally worse than the existing interactive session, which loads each model once.
- Actor/group flags remain simulated identity.
- M1 remains paused and Gate 1 remains `not_run`.
