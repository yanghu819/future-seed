# RepoBench Char1 Diagnostics

This folder records the `RepoBench-C python_cff char1` mechanism study for the
`rwkv-diff-future-seed` track.

## Why this exists

The main repo already contains strong toy and synthetic evidence for
Future-Seed. This diagnostic study asks a narrower question:

- does Future-Seed help on a real byte-level tagspan task,
- is the gain actually using future information,
- and what kind of future information matters.

The answer supported by the current evidence is:

- `v2_init_gate` helps on `RepoBench char1`
- the gain depends on real future information
- the gain is distance-sensitive: `near > far`
- a more selective collector can rescue the `far` bucket

## Task

Source task: `RepoBench-C python_cff`

Transformation:
- mask the first character of the imported symbol in `# imports`
- keep future evidence in `# code`, `# next_line`, and repeated
  `# target_symbol_tagged_i` anchors
- evaluate byte-level exact match on the masked char span (`spanem`)

## Main Results

### Main result

- run: `rwkv15b_scan20260307_auth_repobench_import_cff_char1_confirm5_v2_v1`
- baseline: `0.00%`
- ours (`v2_init_gate`): `6.67%`
- uplift: `+6.67pp`

### Necessity control

- run: `rwkv15b_scan20260307_auth_repobench_import_cff_char1_shuffled_future_confirm5_v2_v1`
- baseline: `5.46%`
- ours: `2.27%`
- uplift: `-3.19pp`

Interpretation:
- the gain disappears when future information is shuffled
- this is evidence that the improvement is genuinely noncausal

### Distance boundary

- run: `rwkv15b_scan20260309_auth_repobench_import_cff_char1_gap20_v2_v1`
- `near` bucket:
  - baseline: `0.00%`
  - ours: `6.52%`
  - uplift: `+6.52pp`
- `far` bucket:
  - baseline: `5.00%`
  - ours: `2.50%`
  - uplift: `-2.50pp`

Interpretation:
- the useful future signal is not monotonic with distance
- under the current recipe, there is an effective future-distance band

### Collector rescue

- run: `rwkv15b_scan20260309_auth_repobench_char1_gap20_far_anchor_exact_v1`
- `far` baseline: `5.00%`
- `far` v2: `2.50%`
- `far` exact-anchor collector: `7.50%`
- uplift vs baseline: `+2.50pp`
- uplift vs v2: `+5.00pp`

But:

- run: `rwkv15b_scan20260309_auth_repobench_char1_gap20_near_anchor_exact_smoke_v1`
- `near` baseline: `0.00%`
- `near` v2: `6.52%`
- `near` exact-anchor collector: `4.35%`
- exact-anchor is better than baseline but worse than `v2`

Interpretation:
- exact anchors rescue the far bucket
- but they are not globally better than `v2_init_gate`

## Current limitation

- run: `rwkv15b_scan20260309_auth_repobench_char1_datasubset_confirm3_v2_regex_v1`
- baseline mean: `2.33%`
- ours mean: `0.00%`

So the current claim should stay narrow:

- valid: `RepoBench char1` has real, noncausal, distance-sensitive gains
- not yet valid: broad full-distribution robustness under every split

## Key code path

The relevant implementation lives in:

- [`../rwkv_diff_future_seed.py`](../rwkv_diff_future_seed.py)

Important knobs:

- `FUTURE_SEED_INIT_ONLY`
- `FUTURE_SEED_S0_GATE`
- `FUTURE_SEED_COLLECT_MODE`
- `FUTURE_SEED_GATE_MODE`

Collector modes now include:

- `full`
- `suffix`
- `anchors`
- `anchor_window`

## Evidence manifest

See [`evidence_manifest.json`](evidence_manifest.json) for the exact run ids and
paper-facing numbers summarized here.
