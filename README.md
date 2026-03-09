# Future-Seed

This repository is a unified Future-Seed paper-and-artifact snapshot with two active experimental tracks:

- `rwkv-diff-future-seed/`: core Future-Seed method, toy stress tests, and prefix-infill experiments
- `posttrain_rwkv7/`: single-GPU post-training search on real tasks, with auditable round logs

The paper-facing post-training rows are teacher-forced token-accuracy probes, not benchmark end metrics such as EM, F1, or pass@k.

If you are new here, choose one track first and follow that path end-to-end.

One-shot repo sanity and anonymous supplementary packaging:

```bash
bash check_repo_health.sh
```

## Current Snapshot

The post-training campaign is closed through `round808`.

Current judgment:
- toy and synthetic constraint tasks clearly support Future-Seed
- `rwkv-diff-future-seed` now includes a real-task mechanism diagnostic on `RepoBench char1`; see [`rwkv-diff-future-seed/repobench_char1_diagnostics/`](rwkv-diff-future-seed/repobench_char1_diagnostics)
- the forward-looking benchmark plan now lives in [`NONCAUSAL_TASK_ROADMAP.md`](NONCAUSAL_TASK_ROADMAP.md)
- strongest repeatable family under the current recipe: `protein_ss_spot`
- supporting positive real-task signals: `hotpot_text_restore`, `squad_text_restore`, `punc_restore`
- promising but unconfirmed: `mbpp_longctx_probe`
- high-upside but unstable: `arc_mc_probe`
- low-ROI or negative under the current recipe: `protein_contact`, `wiki`, `hotpot_longctx`, `countdown`, `nqueens`, `zebra`, `sat3`

## Start Here

1. paper hub: [`PAPER.md`](PAPER.md)
2. global results: [`RESULTS.md`](RESULTS.md)
3. quick onboarding: [`GETTING_STARTED.md`](GETTING_STARTED.md)
4. task-to-script map: [`TASK_INDEX.md`](TASK_INDEX.md)
5. forward task roadmap: [`NONCAUSAL_TASK_ROADMAP.md`](NONCAUSAL_TASK_ROADMAP.md)
6. RepoBench char1 method diagnostic: [`rwkv-diff-future-seed/repobench_char1_diagnostics/README.md`](rwkv-diff-future-seed/repobench_char1_diagnostics/README.md)
7. post-training final status: [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
8. post-training archive chronology: [`posttrain_rwkv7/ARCHIVE_ROUNDS.md`](posttrain_rwkv7/ARCHIVE_ROUNDS.md)
9. full experiment ledger: [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)
10. NeurIPS paper package: [`paper/neurips2025/README.md`](paper/neurips2025/README.md)
11. submission-ready upload note: [`paper/neurips2025/SUBMISSION_READY.md`](paper/neurips2025/SUBMISSION_READY.md)

## Most Important Results

| Bucket | Task | Best gain |
|---|---|---:|
| real-task, repeatable | `protein_ss_spot` | `+8.14pp` |
| real-task, repeatable positive | `hotpot_text_restore` | `+4.20pp` |
| real-task, promising and unconfirmed | `mbpp_longctx_probe` | `+10.00pp` |
| real-task, high upside but unstable | `arc_mc_probe` | `+20.83pp` |
| method-track diagnostic | `RepoBench char1` | `+6.67pp` |
| diagnostic constraint task | `graph_color` | `+8.33pp` |
| appendix-only spike | `tsp_mask` | `+25.00pp` |

## Quick Paths

### Path A: toy and method sanity

```bash
bash run.sh
bash run_qa.sh
```

### Path B: prefix infill on real data

```bash
python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
  --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes
python tools/build_hf_bins.py --dataset mbpp \
  --train_split train --val_split test --fields code --out_dir data/mbpp_bytes
bash rwkv-diff-future-seed/run_wikitext_prefix.sh /abs/path/to/data/wikitext2_bytes
bash rwkv-diff-future-seed/run_mbpp_prefix.sh /abs/path/to/data/mbpp_bytes
```

### Path C: post-training runner sanity

```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

## Repository Map

- [`PAPER.md`](PAPER.md): paper-first entry for the unified repo
- [`NONCAUSAL_TASK_ROADMAP.md`](NONCAUSAL_TASK_ROADMAP.md): explicit benchmark and task roadmap for future noncausal work
- [`rwkv-diff-future-seed/`](rwkv-diff-future-seed): method code and toy experiments
- [`future-seed-8gpu/`](future-seed-8gpu): Hugging Face / ModelScope runner for single-GPU smoke and 8-GPU short Future-Seed experiments
- [`rwkv-diff-future-seed/repobench_char1_diagnostics/`](rwkv-diff-future-seed/repobench_char1_diagnostics): real-task noncausal mechanism diagnostic for the method track
- [`posttrain_rwkv7/`](posttrain_rwkv7): post-training experiments, logs, and queues
- [`RESULTS.md`](RESULTS.md): unified results page
- [`paper/neurips2025/README.md`](paper/neurips2025/README.md): anonymous NeurIPS 2025 submission package
- [`paper/neurips2025/REFERENCE_AUDIT.md`](paper/neurips2025/REFERENCE_AUDIT.md): source audit for all citations used in the paper
- [`posttrain_rwkv7/runs/`](posttrain_rwkv7/runs): local snapshot summaries and raw JSONL records
