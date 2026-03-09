# Task Index

This index maps task family to backend trainer, representative queue, and current judgment.

## Active Post-Training Families

| Task family | Trainer | Representative queue | Current status |
|---|---|---|---|
| `protein_ss_spot` | `posttrain_rwkv7/scripts/train_protein_ss_spot_sft.py` | `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json` | strongest repeatable real-task family; best med `+8.14pp` |
| `hotpot_text_restore` | `posttrain_rwkv7/scripts/train_punc_restore_sft.py` | `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json` | small but repeatable positive line; breadth pivot med `+0.11pp`, `+1.00pp` |
| `mbpp_longctx_probe` | `posttrain_rwkv7/scripts/train_mbpp_longctx_sft.py` | `posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json` | promising but strict confirmation failed in `round799-800` |
| `arc_mc_probe` | `posttrain_rwkv7/scripts/train_arc_mc_sft.py` | `posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json` | high upside, high variance; final confirm failed in `round801` |
| `squad_text_restore` | `posttrain_rwkv7/scripts/train_punc_restore_sft.py` | `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json` | mixed; best spike `+7.31pp`, breadth quicks stalled at `+0.76pp` |
| `punc_restore` | `posttrain_rwkv7/scripts/train_punc_restore_sft.py` | `posttrain_rwkv7/results/_search_queue_round77_82.json` | small stable positive family, but not headline evidence |
| `protein_contact` | `posttrain_rwkv7/scripts/train_protein_contact_pair_sft.py` | `posttrain_rwkv7/results/_search_queue_round77_82.json` | mostly no-op or negative under current recipe |
| `hotpot_longctx` | `posttrain_rwkv7/scripts/train_hotpot_longctx_sft.py` | `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json` | current breadth probe flat |

## Constraint / Diagnostic Families

| Task family | Trainer | Representative queue | Current status |
|---|---|---|---|
| `graph_color` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round743_750_constraint_bfs_v2.json` | useful diagnostic task; med positives up to `+8.33pp`, but not a real-task claim |
| `tsp_mask` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json` | one large spike `+25.00pp`, final confirm flat; appendix only |
| `countdown` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round743_750_constraint_bfs_v2.json` | quick spikes did not convert; current status negative |
| `nqueens` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round735_742_constraint_bfs.json` | hard negative at med |
| `sat3` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round735_742_constraint_bfs.json` | current recipe near zero |
| `zebra` | `posttrain_rwkv7/scripts/train_np_sat_tsp_probe_sft.py` | `posttrain_rwkv7/results/_search_queue_round743_750_constraint_bfs_v2.json` | no useful conversion so far |

## Method / Toy Track

| Goal | Script | Location |
|---|---|---|
| WikiText prefix infill | `run_wikitext_prefix.sh` | `rwkv-diff-future-seed/` |
| MBPP prefix infill | `run_mbpp_prefix.sh` | `rwkv-diff-future-seed/` |
| RepoBench char1 mechanism diagnostic | evidence + code path in `repobench_char1_diagnostics/README.md` and `rwkv_diff_future_seed.py` | `rwkv-diff-future-seed/repobench_char1_diagnostics/` |
| rightcopy + constr sanity | `run.sh` | repo root |
| QA sanity | `run_qa.sh` | repo root |
| Sudoku / KVSORT / PERMFILL | `run_sudoku.sh`, `run_kvsort_baselines.sh`, `run_permfill_anchor_sweep.sh` | `rwkv-diff-future-seed/` |

## Recommended Next Noncausal Targets

This section is not a result table. It is the recommended task stack for future compute budgets.

| Priority | Task family | Concrete target | Why it matters | Current fit |
|---|---|---|---|---|
| P0 | repo-level code infill | `RepoBench-C` / `RepoBench-P` | strongest current mechanism story on a real task; explicit future dependence | already partially implemented in `repobench_char1_diagnostics/` |
| P0 | cross-file code completion | `CrossCodeEval` | real repository completion with later-file evidence | new harness, same mechanism family |
| P0 | executable repository completion | `RepoExec` | upgrades code infill to execution-backed metrics | new harness, high-value benchmark |
| P0 | executable function FIM | `HumanEval-FIM`, `MBPP-FIM` | masked middle code must satisfy both prefix and suffix | builder and metric upgrade from current `mbpp_longctx` line |
| P0 | protein sequence labeling | `protein_ss_spot` | strongest repeatable real-task family already in repo | already active |
| P1 | reasoning with late options | `ARC-Challenge` q-first / options-first | prompt interpretation should use later options | existing `arc_mc_probe`, metric already meaningful |
| P1 | long-context multihop QA | `HotpotQA`, `2Wiki`, `MuSiQue` | answer depends on later supporting evidence in the prompt | `Hotpot` partially explored; builders still needed |
| P1 | support restoration tasks | `hotpot_text_restore`, `squad_text_restore`, `punc_restore` | useful support evidence for future-aware repair | already active |
| P2 | product-style repair | OpenAPI, Docker, GitHub Actions, SQL repair | realistic future-constrained editing and cross-file repair | internal benchmark design, appendix or artifact value |

See [`NONCAUSAL_TASK_ROADMAP.md`](NONCAUSAL_TASK_ROADMAP.md) for the full benchmark roadmap, 8xH100 plan, and 1000-GPU plan.

## Multi-GPU Runner

| Goal | Entry | Location |
|---|---|---|
| single-GPU smoke and 8-GPU short-run RWKV7 Goose experiments | `setup.sh`, `down.sh`, `run1.sh`, `run8.sh` | `future-seed-8gpu/` |
