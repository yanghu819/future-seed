# Results (2026-02-14)

This repo contains a minimal Future-Seed implementation on RWKV7-CUDA, plus hard toy tasks to stress **in-place constraint repair** (future-aware infill + global consistency).

## Key Additions
- `SUDOKU_TASK=1`: 4x4 Sudoku in-place masked infill (synthetic generator), with `sudoku_eval()` reporting:
  - `solve`: valid 4x4 grid (rows/cols/2x2 blocks each contain {1,2,3,4})
  - `exact`: matches the hidden synthetic solution exactly
- `SUDOKU_CONS_LAMBDA`: optional soft constraint regularizer (row/col/block digit-count ~= 1)
- Fixed behavior: `TRAIN=0` no longer silently trains if `WEIGHTS_PATH` is missing (now raises).

## Summary Findings (Single-GPU 4090, RWKV7 cuda_wind)
- **RIGHTCOPY/CONSTR (sanity):** FS helps future-aware copy / constraint fill.
- **KVSORT (in-place permutation repair):** FS can be the difference between complete failure and perfect exact-match.
- **PERMFILL (in-place permutation fill):** adding a tiny anchor dramatically improves length generalization.
- **SUDOKU (2D constraints):** FS shifts the phase transition to harder puzzles; a simple constraint regularizer further helps (holes=12).

All logs referenced below are under `_autodl_logs/` (not committed).

## RIGHTCOPY / CONSTR (Future-Aware Sanity Tasks)
3 seeds, final-step accuracy (mean ± sd):

| task | FS=0 | FS=1 |
|---|---:|---:|
| rightcopy acc | 0.1046 ± 0.0074 | 0.1550 ± 0.0176 |
| constr acc | 0.0977 ± 0.0028 | 0.2114 ± 0.0179 |

Logs:
- `_autodl_logs/rightcopy_fs{0,1}_s{1,2,3}.log`
- `_autodl_logs/constr_fs{0,1}_s{1,2,3}.log`

## KVSORT (Key-Value Sort, In-Place)
Setting (from log filename): `keys36`, `n_test=20`, `nosep`, `L8`, `SEQ_LEN=256`.

| metric | FS=0 | FS=1 |
|---|---:|---:|
| kvsort_id exact | 0.0 | 1.0 |
| kvsort_ood exact | 0.0 | 1.0 |

Logs:
- `_autodl_logs/kvsort_keys36_n20_nosep_fs0_L8_seq256.log`
- `_autodl_logs/kvsort_keys36_n20_nosep_fs1_L8_seq256.log`

## Attention Baselines (Transformer)
Run date: **2026-02-15** (AutoDL 4090).

We tested:
- `MODEL=transformer` (Transformer-MLM, bidirectional)
- `MODEL=transformer_causal` (decoder-only causal)
- `MODEL=transformer_causal ATTN_FS=1` (attention-side Future-Seed via cross-layer global tokens; `ATTN_FS_K=32`)

**KVSORT (n=20, keys-only, no-sep)**:
- Transformer-MLM: `DECODE=hungarian` enforces uniqueness (`key_valid=1.0`) but **fails ordering** (`key_order=0.0`, `exact=0.0`).
- Transformer-Causal (no future): `exact=0.0` (as expected); Hungarian can't fix ordering.
- Transformer-Causal + ATTN_FS (K=32): **no improvement** over causal baseline (`exact=0.0`, Hungarian still `key_order=0.0`).
- Iterative refinement (`REFINE_STEPS=8`, Mask-Predict style) does **not** fix ordering for Transformer-MLM (`exact=0.0`).

**PERMFILL (train n=24, anchor k=2; eval n_test=24/28/32/36)**:
- Transformer-Causal + Hungarian: `valid=1.0` but **OOD exact=0.0** for all `n_test`.
- Transformer-Causal + ATTN_FS (K=32): **OOD exact=0.0** for all `n_test` (Hungarian or argmax).

Logs:
- `_autodl_logs/attnfs/kvsort_keys36_n20_tfc_{fs0,attnfs}.jsonl`
- `_autodl_logs/attnfs/permfill_anchor2_n24_tfc_{fs0,attnfs}.jsonl`

## PERMFILL (Permutation Fill, In-Place)
We evaluate exact/valid rates under **length extrapolation** by loading saved weights and running `TRAIN=0 PERMFILL_EVAL=1` with different `PERMFILL_N_TEST`.

| weights | anchor | n_test=24 | n_test=28 | n_test=32 | n_test=36 |
|---|---:|---:|---:|---:|---:|
| `permfill_n24_fs1_L12_seq256.pt` | off | 1.000 | 0.000 | 0.000 | 0.000 |
| `permfill_anchor2_n24_fs1_L12_seq256.pt` | k=2 | 0.990 | 0.830 | 0.120 | 0.000 |
| `permfill_anchor2_n32_fs1_L12_seq256.pt` | k=2 | 1.000 | 1.000 | 1.000 | 0.935 |

Interpretation: **a tiny anchor (k=2 visible tokens in the masked span)** + higher training max length pushes OOD generalization strongly.

### Phase Curve (trials=2000)
Weights:
- FS=0: `weights/sudoku_fs0_curr4_12_L4_e128.pt` (L4, e128, curriculum holes∈[4,12])
- FS=1: `weights/sudoku_fs1_curr4_12_L12_e128.pt` (L12, e128, curriculum holes∈[4,12])

| holes | FS=0 solve | FS=1 solve |
|---:|---:|---:|
| 4  | 0.3530 | 1.0000 |
| 6  | 0.1665 | 1.0000 |
| 8  | 0.0520 | 1.0000 |
| 10 | 0.0095 | 0.9510 |
| 12 | 0.0000 | 0.5510 |
| 14 | 0.0000 | 0.0000 |

### Constraint Regularizer (FS=1, L12/e128, holes∈[4,14], trials=2000)
Weights:
- base: `weights/sudoku_fs1_curr4_14_L12_e128_base.pt` (lambda=0.0)
- cons: `weights/sudoku_fs1_curr4_14_L12_e128_cons01.pt` (lambda=0.1)

| setting | holes=10 | holes=12 | holes=14 |
|---|---:|---:|---:|
| lambda=0.0 | 0.9385 | 0.4795 | 0.0000 |
| lambda=0.1 | 0.9515 | 0.5855 | 0.0000 |

## Negative / Lessons
- `FS_MASK_ONLY=1` (train only seed-alpha + mask embedding) fails on Sudoku (solve≈0).
- `SUDOKU_MASK_MODE=prefix` is unstable in current one-shot argmax infill setting (solve≈0).
