# future-seed-8gpu

Minimal Hugging Face / FLA training entry for running Future-Seed on top of RWKV7 Goose models.

Default base model:
- `RWKV/RWKV7-Goose-World3-2.9B-HF`

This folder is intentionally narrow:
- `setup.sh`: create a `uv` environment and install runtime packages
- `down.sh`: download the base model and normalize ModelScope datasets
- `run1.sh`: single-GPU smoke, defaults to a short `arc_mask` run; supports 4-bit single-card loading
- `run8.sh`: 8-GPU short experiment pack, defaults to baseline vs Future-Seed on `arc_mask` and `mbpp_mask`

## Why these tasks

The default 8-GPU package uses tasks where the target appears before the necessary evidence:

- `arc_mask`: `Answer: [MASK]` comes before the answer options
- `mbpp_mask`: the masked code span sits before its later code suffix

This makes them clean noncausal probes under a same-position masked-token objective.

## Recommended 8-GPU experiments under 4h

1. `arc_mask` baseline vs Future-Seed
2. `mbpp_mask` baseline vs Future-Seed

That gives one low-entropy reasoning probe and one code span-repair probe, both with public datasets.

## Single-card notes

Default single-card smoke is conservative and uses 4-bit loading.

Examples:

```bash
bash setup.sh
bash down.sh
bash run1.sh > log.txt 2>&1
```

To try a larger checkpoint on one card, override `MODEL_ID` and keep `LOAD_IN_4BIT=1`:

```bash
MODEL_ID=RWKV/RWKV7-Goose-World3-7B-HF LOAD_IN_4BIT=1 bash run1.sh > log.txt 2>&1
```

## Output discipline

The shell scripts keep stdout compact and write detailed trainer artifacts under:

- `runs/`
- `artifacts/`

Each run also writes a machine-readable `summary.json`.
