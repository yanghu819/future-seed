# Global Results (2026-03-06, through round808)

This page is the consolidated result index for the repository.

Primary sources:
- [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
- [`posttrain_rwkv7/results/_rolling_round_log.md`](posttrain_rwkv7/results/_rolling_round_log.md)
- [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)

## Bottom Line

What the current evidence supports:
- Future-Seed clearly helps on toy and synthetic tasks that stress future-aware constraint repair.
- In post-training, the strongest repeatable real-task family under the current recipe is `protein_ss`.
- `hotpot`, `mbpp_longctx`, `squad`, and `punc` show smaller positive pockets.
- `arc_mc` has strong upside but is still too high-variance for a stability claim.
- Under the current recipe, the repo should not claim stable held-out real-task confirmation yet.

## Toy / Synthetic Results

| Task | Baseline | Future-Seed | Gain |
|---|---:|---:|---:|
| `rightcopy` acc | `10.46%` | `15.50%` | `+5.04pp` |
| `constr` acc | `9.77%` | `21.14%` | `+11.37pp` |
| `kvsort` exact | `0.00%` | `100.00%` | `+100.00pp` |
| `permfill` exact at `n_test=36` | `0.00%` | `93.50%` | `+93.50pp` |
| `sudoku` solve at `holes=12` | `0.00%` | `55.10%` | `+55.10pp` |
| `sudoku` constraint regularizer at `holes=12` | `47.95%` | `58.55%` | `+10.60pp` |

## Post-Training Scoreboard

| Task family | Best med gain | Med median | Positive med count | Current judgment |
|---|---:|---:|---:|---|
| `protein_ss` | `+8.14pp` | `+2.18pp` | `103/126` | strongest repeatable real-task family |
| `hotpot` | `+4.20pp` | `+0.54pp` | `35/53` | small but repeatable positive family |
| `mbpp_longctx` | `+10.00pp` | `+0.91pp` | `53/87` | promising, but strict confirmation failed |
| `arc_mc` | `+20.83pp` | `+4.17pp` | `70/116` | high upside, high variance; held-out confirm failed |
| `squad` | `+7.31pp` | `+0.55pp` | `24/35` | mixed; one strong spike, not locked |
| `punc` | `+2.16pp` | `+0.36pp` | `16/24` | small positive, useful support only |
| `graph_color` | `+8.33pp` | `+0.00pp` | `4/10` | useful diagnostic task, not real-task evidence |
| `tsp_mask` | `+25.00pp` | `+2.08pp` | `2/4` | appendix only; spike did not confirm |
| `countdown` | `-4.17pp` | `-4.17pp` | `0/2` | negative under current recipe |
| `nqueens` | `-4.17pp` | `-10.42pp` | `0/2` | negative under current recipe |
| `sat3` | `+0.00pp` | `+0.00pp` | `0/4` | no current signal |

Additional low-ROI families:
- `protein_contact`: mostly no-op or small negative; no useful med-positive line remained
- `wiki`: latest breadth probe failed at baseline in `round808`
- `hotpot_longctx`: latest breadth probe flat in `round806`
- `zebra`: no useful conversion under the current recipe

## Final Closure Window

The final narrow search and breadth pivot did three useful things:

1. `round783-802` showed that the current `mbpp_longctx` and `arc_mc` recipes can still produce positives, but not stable held-out confirmation.
2. `round803-804` killed further spending on repeated `mbpp_longctx` confirmation with the same `head_l8 / scalar_l8_*` family.
3. `round805-808` breadth-first exploration found small new positive `hotpot` conversions and no new strong wins elsewhere.

Latest final-window highlights:
- `mbpp_longctx_seed38_depthmix_anchor`: `+5.01pp`
- `arc_mc_seed260_depthmix_qfirst`: `+9.38pp`
- `hotpot_seed96_breadth`: `+1.00pp`
- `squad_seed104_breadth`: quick `+0.76pp`, below promote gate
- `protein_ss_seed283/284_breadth`: near-gate quick positives, but no promote
- `wiki_seed41_breadth`: baseline failed

## Where To Verify Raw Artifacts

- latest summaries and records: [`posttrain_rwkv7/runs/`](posttrain_rwkv7/runs)
- rolling status log: [`posttrain_rwkv7/results/_rolling_round_log.md`](posttrain_rwkv7/results/_rolling_round_log.md)
- full experiment ledger: [`posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md`](posttrain_rwkv7/paper/DETAILED_EXPERIMENT_LOG.md)
