# Compute Accounting

This note documents the compute boundary for the paper-facing snapshot.
It is intentionally narrower than the full historical campaign.

## Hardware Boundary

- Post-training audit: single `NVIDIA RTX 4090`
- Search mode: serial quick-to-medium queue execution
- Decode policy: left-to-right causal decode throughout; Future-Seed only changes prompt-time recurrent state initialization
- Paper build: local CPU-side LaTeX build via `tectonic`

The snapshot does **not** include cluster logs, scheduler traces, or exact peak-VRAM telemetry.

## Queue Windows Included In The Snapshot

The anonymous snapshot ships four late-stage queue files:

| Queue file | Rounds | Tasks | Quick budget | Medium budget | Quick steps | Medium steps |
|---|---:|---:|---|---|---|---|
| `_search_queue_round783_790_realtask_exploit_v3.json` | 8 | 16 | `140, 160` | `260, 280` | `160, 220` | `320, 380` |
| `_search_queue_round799_802_final_confirm.json` | 4 | 4 | `120, 140, 160` | `220, 260, 280` | `160, 220` | `320, 380` |
| `_search_queue_round803_804_mbpp_altconfirm.json` | 2 | 2 | `160` | `260` | `160` | `320` |
| `_search_queue_round805_808_breadth_roi.json` | 4 | 8 | `120, 140, 150` | `220, 240, 260` | `160` | `320, 360` |

Across these shipped queues, the late-stage snapshot covers `18` rounds and `30` queued tasks.

## Observed Launch-Span In The Shipped Round Records

The JSONL record files store `run_dir` timestamps. These timestamps allow a conservative lower-bound launch span for each shipped round.
They are **not** the same thing as full GPU occupation time, because they do not include earlier exploratory windows, cache warm-up, or any work that is not reflected in the final `run_dir` timestamps.

| Round range | Shipped rounds | Observed launch-span sum | Median round span |
|---|---:|---:|---:|
| exploit / closure / breadth snapshot | 16 | `64.5` minutes | `5.0` minutes |

Examples from the shipped closure window:

- `round783`: `3.5` minutes from first to last recorded launch timestamp
- `round788`: `6.4` minutes from first to last recorded launch timestamp
- `round805`: `6.7` minutes from first to last recorded launch timestamp
- `round807`: `6.8` minutes from first to last recorded launch timestamp

These values are useful for sanity-checking the scale of the archived queues, but they should not be read as a full compute bill for the entire campaign.

## Storage Boundary

Approximate on-disk sizes in the current repository snapshot:

- `rwkv-diff-future-seed/`: about `228 KB`
- shipped `posttrain_rwkv7/runs/`: about `196 KB`
- tracked `posttrain_rwkv7/results/`: about `6.1 MB` total, although the supplementary package only ships the final queue files
- `paper/neurips2025/`: about `1.2 MB` before build artifacts in `build/` and `dist/`

## What This Supports

This accounting is sufficient to understand the scale of the paper-facing archive:

- one GPU class
- serial queue execution
- explicit quick/medium budgets
- shipped round windows and their lower-bound launch spans
- lightweight local paper rebuild cost

It is **not** a full historical GPU-hour reconstruction of the entire project archive.
