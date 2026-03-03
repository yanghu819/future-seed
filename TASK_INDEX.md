# Task Index (What to run for what)

## A) Core method / toy stress tests

| Goal | Script | Output location |
|---|---|---|
| rightcopy + constr sanity | `run.sh` | `rwkv-diff-future-seed/logs/*.log` |
| QA bidirectional masking | `run_qa.sh` | `rwkv-diff-future-seed/logs/qa_*.log` |
| sudoku sweep | `rwkv-diff-future-seed/run_sudoku.sh` | `rwkv-diff-future-seed/exp/*.jsonl` |
| kvsort baselines | `rwkv-diff-future-seed/run_kvsort_baselines.sh` | `rwkv-diff-future-seed/exp/*.jsonl` |
| kvsort + sinkhorn | `rwkv-diff-future-seed/run_kvsort_sinkhorn.sh` | `rwkv-diff-future-seed/exp/*.jsonl` |
| permfill anchor sweep | `rwkv-diff-future-seed/run_permfill_anchor_sweep.sh` | `rwkv-diff-future-seed/exp/*.jsonl` |

## B) Real-data prefix infill (the scripts you asked about)

| Goal | Script | Required input |
|---|---|---|
| WikiText prefix infill | `rwkv-diff-future-seed/run_wikitext_prefix.sh` | path to `train.bin`/`val.bin` |
| MBPP prefix infill | `rwkv-diff-future-seed/run_mbpp_prefix.sh` | path to `train.bin`/`val.bin` |

Build bins with:
- `tools/build_hf_bins.py`

## C) Post-training real tasks (ARC/protein/MBPP)

| Goal | Script |
|---|---|
| ARC MC training backend | `posttrain_rwkv7/scripts/train_arc_mc_sft.py` |
| Protein SS backend | `posttrain_rwkv7/scripts/train_protein_ss_spot_sft.py` |
| Protein contact backend | `posttrain_rwkv7/scripts/train_protein_contact_pair_sft.py` |
| MBPP longctx backend | `posttrain_rwkv7/scripts/train_mbpp_longctx_sft.py` |
| fastdiscover orchestrator | `posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py` |
| finite repro packet | `posttrain_rwkv7/scripts/run_repropack_569_574.sh` |

Representative historical launchers:
- ARC: `posttrain_rwkv7/scripts/run_arc_qfirst_stabilized_round13_canonical_s01234.sh`
- protein ss: `posttrain_rwkv7/scripts/run_protein_ss_spot_qfirst_len2048_round1_s012.sh`
- protein contact: `posttrain_rwkv7/scripts/run_protein_contact_pair_qafter_len2048_round4_sched_s012.sh`
- mbpp: `posttrain_rwkv7/scripts/run_mbpp_qfirst_stabilized_len4096_round1_s012.sh`

## D) Where to read outcomes

- method/toy consolidated: `RESULTS.md`
- post-training consolidated: `posttrain_rwkv7/README.md`
- post-training audit ledgers: `posttrain_rwkv7/results/_audit_round77_569_*.{md,csv}`
