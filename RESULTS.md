# Results (2026-02-14)

This repo contains a minimal Future-Seed implementation on RWKV7, plus hard toy tasks to stress “in-place constraint repair”.

## Key Additions
- `SUDOKU_TASK=1`: 4x4 Sudoku in-place masked infill (synthetic generator), with `sudoku_eval()` reporting:
  - `solve`: valid 4x4 grid (rows/cols/2x2 blocks each contain {1,2,3,4})
  - `exact`: matches the hidden synthetic solution exactly
- `SUDOKU_CONS_LAMBDA`: optional soft constraint regularizer (row/col/block digit-count ~= 1)
- Fixed behavior: `TRAIN=0` no longer silently trains if `WEIGHTS_PATH` is missing (now raises).

## Main Finding (Single-GPU 4090, RWKV7 cuda_wind)
Future-Seed strongly improves in-place constraint repair on 4x4 Sudoku, and shifts the “phase transition” to harder puzzles.

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

