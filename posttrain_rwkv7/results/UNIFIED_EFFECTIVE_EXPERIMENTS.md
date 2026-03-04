# Unified Effective Experiment Report

This document unifies all **effective** experiments in one place, from earliest logged rounds to latest.

## Scope & Rule
- Parsed record files: **291**
- Round span: **20-569**
- Comparable items (best FS med + anchor baseline exists): **277**
- Effective items (`best_med_delta_vs_anchor_pp > 0`): **201**
- Non-effective comparable items: **76**
- Comparable med mean/median: **+2.09pp / +1.33pp**
- Strict `med_vs_med` comparable items: **259**
- Legacy `med_vs_quick` fallback items: **18**

Effective = for the same `(round, task)`, best FS `med` token-accuracy is above anchor baseline. Anchor uses `med baseline` when present; otherwise falls back to `quick baseline`.

## Family Stability (All Comparable Med)

| family | comparable_n | effective_n | effective_rate | mean_delta_pp | median_delta_pp | best_pp | worst_pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| sudoku | 2 | 2 | 100.0% | +17.40pp | +17.40pp | +33.29pp | +1.51pp |
| protein_ss | 63 | 56 | 88.9% | +2.69pp | +2.39pp | +13.31pp | -2.26pp |
| punc | 30 | 25 | 83.3% | +1.73pp | +1.36pp | +10.63pp | -2.55pp |
| squad | 17 | 14 | 82.4% | +0.68pp | +0.71pp | +1.73pp | -2.13pp |
| hotpot | 30 | 22 | 73.3% | +0.86pp | +0.65pp | +4.20pp | -2.74pp |
| mbpp | 59 | 40 | 67.8% | +1.29pp | +0.52pp | +19.17pp | -4.68pp |
| arc_mc | 51 | 29 | 56.9% | +3.45pp | +4.17pp | +20.83pp | -12.50pp |
| mbpp_longctx | 25 | 13 | 52.0% | +1.32pp | +0.33pp | +10.00pp | -6.45pp |

## Top Effective Runs (Global Top 50 by med delta)

| round | task | family | best_med_cfg | baseline_med | best_med | delta_pp | source |
|---:|---|---|---|---:|---:|---:|---|
| 157 | arc_mc_seed13_discovery | arc_mc | scalar_l8_train8e5 | 29.17% | 50.00% | +20.83pp | _round157_fastdiscover_records.jsonl |
| 163 | arc_mc_seed16_discovery | arc_mc | scalar_l8_sched_cos | 33.33% | 50.00% | +16.67pp | _round163_fastdiscover_records.jsonl |
| 136 | arc_mc_seed3_discovery | arc_mc | scalar_l8_sched_cos | 29.17% | 41.67% | +12.50pp | _round136_fastdiscover_records.jsonl |
| 405 | arc_mc_seed115_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 45.83% | +12.50pp | _round405_fastdiscover_records.jsonl |
| 417 | arc_mc_seed120_discovery | arc_mc | scalar_l8_sched_cos | 37.50% | 50.00% | +12.50pp | _round417_fastdiscover_records.jsonl |
| 222 | mbpp_longctx_seed13_repair | mbpp_longctx | scalar_l8_train8e5 | 17.13% | 27.13% | +10.00pp | _round222_fastdiscover_records.jsonl |
| 534 | mbpp_longctx_seed106_repair | mbpp_longctx | head_l8 | 19.06% | 27.75% | +8.69pp | _round534_fastdiscover_records.jsonl |
| 160 | arc_mc_seed14_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | _round160_fastdiscover_records.jsonl |
| 171 | arc_mc_seed20_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | _round171_fastdiscover_records.jsonl |
| 193 | arc_mc_seed31_discovery | arc_mc | scalar_l8_sched_cos | 33.33% | 41.67% | +8.33pp | _round193_fastdiscover_records.jsonl |
| 209 | arc_mc_seed39_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | _round209_fastdiscover_records.jsonl |
| 425 | arc_mc_seed122_discovery | arc_mc | scalar_l8_sched_cos | 33.33% | 41.67% | +8.33pp | _round425_fastdiscover_records.jsonl |
| 557 | arc_mc_seed205_discovery | arc_mc | scalar_l8_train8e5 | 33.33% | 41.67% | +8.33pp | _round557_fastdiscover_records.jsonl |
| 569 | arc_mc_seed211_repro | arc_mc | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | _round569_fastdiscover_records.jsonl |
| 168 | arc_mc_seed18_discovery | arc_mc | scalar_l8_train1e4 | 37.50% | 45.83% | +8.33pp | _round168_fastdiscover_records.jsonl |
| 195 | arc_mc_seed32_discovery | arc_mc | scalar_l8_sched_cos | 37.50% | 45.83% | +8.33pp | _round195_fastdiscover_records.jsonl |
| 528 | protein_ss_seed261_discovery | protein_ss | scalar_l8_sched_cos | 27.86% | 36.00% | +8.14pp | _round528_fastdiscover_records.jsonl |
| 507 | mbpp_longctx_seed99_repair | mbpp_longctx | head_l8 | 19.58% | 27.02% | +7.44pp | _round507_fastdiscover_records.jsonl |
| 466 | protein_ss_seed239_discovery | protein_ss | head_l8 | 27.84% | 35.17% | +7.33pp | _round466_fastdiscover_records.jsonl |
| 158 | protein_ss_seed25_discovery | protein_ss | scalar_l8_train8e5 | 28.00% | 34.04% | +6.04pp | _round158_fastdiscover_records.jsonl |
| 121 | protein_ss_seed10_discovery | protein_ss | scalar_l8_train8e5 | 27.66% | 33.62% | +5.96pp | _round121_fastdiscover_records.jsonl |
| 526 | mbpp_longctx_seed104_repair | mbpp_longctx | head_l8 | 18.57% | 24.53% | +5.96pp | _round526_fastdiscover_records.jsonl |
| 227 | protein_ss_seed60_discovery | protein_ss | head_l8 | 26.68% | 32.10% | +5.42pp | _round227_fastdiscover_records.jsonl |
| 522 | protein_ss_seed259_discovery | protein_ss | scalar_l8_sched_cos | 30.11% | 35.09% | +4.98pp | _round522_fastdiscover_records.jsonl |
| 99 | protein_ss_seed4_discovery | protein_ss | scalar_l8_train8e5 | 30.30% | 35.20% | +4.90pp | _round99_fastdiscover_records.jsonl |
| 444 | protein_ss_seed189_discovery | protein_ss | scalar_l8_train1e4 | 26.68% | 31.57% | +4.90pp | _round444_fastdiscover_records.jsonl |
| 219 | protein_ss_seed56_discovery | protein_ss | scalar_l8_train1e4 | 25.64% | 30.39% | +4.75pp | _round219_fastdiscover_records.jsonl |
| 226 | mbpp_seed45_headprobe | mbpp | head_l10_strong | 46.69% | 51.33% | +4.64pp | _round226_fastdiscover_records.jsonl |
| 414 | protein_ss_seed174_discovery | protein_ss | scalar_l8_sched_cos | 31.06% | 35.63% | +4.58pp | _round414_fastdiscover_records.jsonl |
| 438 | protein_ss_seed186_discovery | protein_ss | scalar_l8_sched_cos | 29.68% | 34.21% | +4.53pp | _round438_fastdiscover_records.jsonl |
| 214 | mbpp_longctx_seed12_repair | mbpp_longctx | scalar_l8_train8e5 | 18.46% | 22.84% | +4.37pp | _round214_fastdiscover_records.jsonl |
| 157 | hotpot_seed26_discovery | hotpot | scalar_l8_train8e5 | 9.10% | 13.30% | +4.20pp | _round157_fastdiscover_records.jsonl |
| 169 | arc_mc_seed19_discovery | arc_mc | scalar_l8_train8e5 | 37.50% | 41.67% | +4.17pp | _round169_fastdiscover_records.jsonl |
| 187 | arc_mc_seed28_discovery | arc_mc | scalar_l8_train8e5 | 33.33% | 37.50% | +4.17pp | _round187_fastdiscover_records.jsonl |
| 211 | arc_mc_seed40_discovery | arc_mc | scalar_l8_sched_cos | 37.50% | 41.67% | +4.17pp | _round211_fastdiscover_records.jsonl |
| 213 | arc_mc_seed41_discovery | arc_mc | scalar_l8_sched_cos | 33.33% | 37.50% | +4.17pp | _round213_fastdiscover_records.jsonl |
| 221 | arc_mc_seed45_discovery | arc_mc | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | _round221_fastdiscover_records.jsonl |
| 225 | arc_mc_seed47_discovery | arc_mc | scalar_l8_sched_cos | 33.33% | 37.50% | +4.17pp | _round225_fastdiscover_records.jsonl |
| 401 | arc_mc_seed114_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 37.50% | +4.17pp | _round401_fastdiscover_records.jsonl |
| 449 | arc_mc_seed164_discovery | arc_mc | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | _round449_fastdiscover_records.jsonl |
| 465 | arc_mc_seed170_discovery | arc_mc | scalar_l8_train1e4 | 33.33% | 37.50% | +4.17pp | _round465_fastdiscover_records.jsonl |
| 505 | arc_mc_seed184_discovery | arc_mc | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | _round505_fastdiscover_records.jsonl |
| 441 | arc_mc_seed126_discovery | arc_mc | scalar_l8_train8e5 | 41.67% | 45.83% | +4.17pp | _round441_fastdiscover_records.jsonl |
| 551 | arc_mc_seed202_discovery | arc_mc | scalar_l8_train1e4 | 41.67% | 45.83% | +4.17pp | _round551_fastdiscover_records.jsonl |
| 441 | protein_ss_seed188_discovery | protein_ss | scalar_l8_sched_cos | 28.55% | 32.68% | +4.13pp | _round441_fastdiscover_records.jsonl |
| 155 | protein_ss_seed24_discovery | protein_ss | scalar_l8_train8e5 | 26.68% | 30.59% | +3.92pp | _round155_fastdiscover_records.jsonl |
| 425 | protein_ss_seed180_discovery | protein_ss | scalar_l8_sched_cos | 29.13% | 32.88% | +3.74pp | _round425_fastdiscover_records.jsonl |
| 134 | mbpp_seed11_anchor | mbpp | scalar_l8_sched_cos | 46.07% | 49.75% | +3.68pp | _round134_fastdiscover_records.jsonl |
| 408 | protein_ss_seed170_discovery | protein_ss | scalar_l8_train1e4 | 28.88% | 32.56% | +3.68pp | _round408_fastdiscover_records.jsonl |
| 224 | protein_ss_seed58_discovery | protein_ss | head_l8 | 28.12% | 31.75% | +3.63pp | _round224_fastdiscover_records.jsonl |

## Full Effective Timeline (Chronological, All Positive med)

| round | task | family | compare_mode | best_med_cfg | anchor_baseline | best_med | delta_pp | quick_delta_pp | source |
|---:|---|---|---|---:|---:|---:|---:|---|
| 20 | protein_ss | protein_ss | med_vs_quick | scalar_l10_norm_node | 24.66% | 32.69% | +8.02pp | +4.45pp | _round20_serial_earlystop_records.jsonl |
| 21 | mbpp_fix | mbpp | med_vs_quick | scalar_l8_trainable | 10.46% | 24.07% | +13.61pp | +1.00pp | _round21_targeted_search_records.jsonl |
| 21 | protein_ss_refine | protein_ss | med_vs_quick | scalar_l10_sched_cos | 21.14% | 34.45% | +13.31pp | +4.48pp | _round21_targeted_search_records.jsonl |
| 22 | sudoku4_refine | sudoku | med_vs_quick | scalar_l6_trainable | 58.89% | 92.18% | +33.29pp | +5.58pp | _round22_adaptive_search_records.jsonl |
| 22 | sudoku9_probe | sudoku | med_vs_quick | scalar_l6_trainable | 1.35% | 2.86% | +1.51pp | +0.24pp | _round22_adaptive_search_records.jsonl |
| 24 | protein_ss_rt | protein_ss | med_vs_quick | scalar_l10_sched_cos | 30.17% | 34.31% | +4.14pp | +0.51pp | _round24_punc_protein_records.jsonl |
| 26 | mbpp_low | mbpp | med_vs_quick | scalar_l8_trainable | 10.46% | 29.64% | +19.17pp | +1.00pp | _round26_mbpp_hotpot_lowthroughput_records.jsonl |
| 27 | mbpp_low | mbpp | med_vs_quick | scalar_l8_trainable | 13.99% | 25.39% | +11.40pp | +0.32pp | _round27_seedcheck_positive_s012_records.jsonl |
| 27 | punc_restore | punc | med_vs_quick | head_l8 | 9.18% | 11.69% | +2.50pp | +0.80pp | _round27_seedcheck_positive_s012_records.jsonl |
| 28 | mbpp | mbpp | med_vs_quick | scalar_l8_trainable | 14.50% | 25.39% | +10.89pp | -2.08pp | _round28_mbpp_bsz_sweep_s0_records.jsonl |
| 29 | punc_restore | punc | med_vs_quick | head_l8 | 9.81% | 11.69% | +1.87pp | +0.64pp | _round29_punc_seed5_s01234_records.jsonl |
| 34 | punc_hotpot | punc | med_vs_quick | scalar_l8_sched_cos | 9.18% | 11.89% | +2.70pp | +0.80pp | _round34_punc_transfer_s0_records.jsonl |
| 34 | punc_mbpp | punc | med_vs_quick | scalar_l8_sched_cos | 37.62% | 48.25% | +10.63pp | +3.06pp | _round34_punc_transfer_s0_records.jsonl |
| 35 | punc_hotpot | punc | med_vs_quick | head_l8 | 8.50% | 10.80% | +2.30pp | +1.41pp | _round35_punc_transfer_prune_s0_records.jsonl |
| 35 | punc_mbpp | punc | med_vs_quick | scalar_l8_sched_cos | 40.83% | 46.80% | +5.97pp | +1.28pp | _round35_punc_transfer_prune_s0_records.jsonl |
| 36 | punc_hotpot | punc | med_vs_quick | head_l8 | 8.50% | 10.80% | +2.30pp | +1.41pp | _round36_punc_scene_discovery_s0_records.jsonl |
| 36 | punc_mbpp | punc | med_vs_quick | scalar_l8_sched_cos | 40.83% | 45.28% | +4.45pp | +1.28pp | _round36_punc_scene_discovery_s0_records.jsonl |
| 36 | punc_squad | punc | med_vs_quick | scalar_l8_sched_cos | 11.22% | 15.74% | +4.52pp | +1.90pp | _round36_punc_scene_discovery_s0_records.jsonl |
| 61 | mbpp_strict_seed1 | mbpp | med_vs_med | head_l10_strong | 46.67% | 47.01% | +0.35pp | +3.43pp | _round61_strict_seed1_frontier_records.jsonl |
| 62 | punc_restore_seed0_scout | punc | med_vs_med | head_l8 | 10.59% | 11.09% | +0.51pp | +1.66pp | _round62_3h_finishpack_records.jsonl |
| 63 | mbpp_seed2_regrescue | mbpp | med_vs_med | head_l10_clip07 | 46.69% | 48.97% | +2.28pp | +3.41pp | _round63_useful_followup_records.jsonl |
| 63 | punc_restore_seed1_confirm | punc | med_vs_med | scalar_l8_sched_cos | 12.20% | 14.32% | +2.12pp | +2.42pp | _round63_useful_followup_records.jsonl |
| 65 | squad_strict_seed2 | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round65_mbpp_squad_seed23_records.jsonl |
| 66 | squad_seed0_train1e4_confirm | squad | med_vs_med | scalar_l8_train1e4 | 17.27% | 19.00% | +1.73pp | +0.72pp | _round66_squad_mbpp_frontier_records.jsonl |
| 67 | mbpp_seed0_reconfirm_strict | mbpp | med_vs_med | head_l10_strong | 48.39% | 49.94% | +1.54pp | +2.27pp | _round67_squad3_mbpp0_records.jsonl |
| 69 | mbpp_seed3_dualmed_rescue | mbpp | med_vs_med | head_l8_nodetach | 48.23% | 48.28% | +0.05pp | +0.95pp | _round69_mbpp_squad_rescue_records.jsonl |
| 70 | mbpp_seed2_dualmed_recheck | mbpp | med_vs_med | head_l10_clip07 | 46.69% | 48.97% | +2.28pp | +3.41pp | _round70_squad3_mbpp2_records.jsonl |
| 71 | squad_seed2_scalar_reconfirm | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round71_mbpp3_squad2_records.jsonl |
| 73 | mbpp_seed1_strict_recheck | mbpp | med_vs_med | head_l10_strong | 46.67% | 47.01% | +0.35pp | +3.43pp | _round73_mbpp1_squad0_recheck_records.jsonl |
| 73 | squad_seed0_frontier_recheck | squad | med_vs_med | scalar_l8_train1e4 | 17.27% | 19.00% | +1.73pp | +0.72pp | _round73_mbpp1_squad0_recheck_records.jsonl |
| 74 | punc_seed1_frontier_recheck | punc | med_vs_med | scalar_l8_train1e4 | 12.20% | 13.04% | +0.84pp | +5.12pp | _round74_punc1_squad2_frontier_records.jsonl |
| 74 | squad_seed2_frontier_recheck | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round74_punc1_squad2_frontier_records.jsonl |
| 75 | mbpp_seed3_l8_refine | mbpp | med_vs_med | head_l8_nodetach | 48.23% | 48.28% | +0.05pp | +0.56pp | _round75_mbpp3_punc0_targeted_records.jsonl |
| 75 | punc_seed0_frontier_recheck | punc | med_vs_med | head_l8 | 10.59% | 11.09% | +0.51pp | +1.66pp | _round75_mbpp3_punc0_targeted_records.jsonl |
| 76 | punc_seed1_dualmed_compare | punc | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round76_punc1_squad0_dualmed_records.jsonl |
| 76 | squad_seed0_dualmed_compare | squad | med_vs_med | scalar_l8_train1e4 | 17.27% | 19.00% | +1.73pp | +0.72pp | _round76_punc1_squad0_dualmed_records.jsonl |
| 77 | arc_seed0_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 12.33% | 12.63% | +0.29pp | +1.00pp | _round77_fastdiscover_records.jsonl |
| 78 | protein_ss_seed0_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 27.17% | 30.77% | +3.60pp | +3.60pp | _round78_fastdiscover_records.jsonl |
| 79 | hotpot_seed1_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round79_fastdiscover_records.jsonl |
| 81 | punc_seed1_anchor | punc | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round81_fastdiscover_records.jsonl |
| 81 | squad_seed2_anchor | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round81_fastdiscover_records.jsonl |
| 82 | mbpp_seed2_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 46.69% | 47.58% | +0.89pp | +2.08pp | _round82_fastdiscover_records.jsonl |
| 92 | protein_ss_seed1_discovery | protein_ss | med_vs_med | head_l8 | 32.23% | 34.62% | +2.39pp | +9.39pp | _round92_fastdiscover_records.jsonl |
| 93 | mbpp_seed0_anchor | mbpp | med_vs_med | scalar_l8_train8e5 | 48.07% | 50.10% | +2.03pp | +2.86pp | _round93_fastdiscover_records.jsonl |
| 96 | protein_ss_seed3_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 31.29% | 32.97% | +1.68pp | +1.68pp | _round96_fastdiscover_records.jsonl |
| 97 | squad_seed2_anchor_recheck | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round97_fastdiscover_records.jsonl |
| 98 | punc_seed1_anchor_recheck | punc | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round98_fastdiscover_records.jsonl |
| 99 | mbpp_seed2_headprobe | mbpp | med_vs_med | head_l10_strong | 46.69% | 48.97% | +2.28pp | +3.41pp | _round99_fastdiscover_records.jsonl |
| 99 | protein_ss_seed4_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 30.30% | 35.20% | +4.90pp | +4.90pp | _round99_fastdiscover_records.jsonl |
| 100 | arc_seed0_headprobe | arc_mc | med_vs_med | scalar_l8_train8e5 | 12.33% | 12.63% | +0.29pp | +1.00pp | _round100_fastdiscover_records.jsonl |
| 100 | hotpot_seed1_headprobe | hotpot | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round100_fastdiscover_records.jsonl |
| 101 | punc_seed1_anchor_recheck2 | punc | med_vs_med | scalar_l8_train1e4 | 10.88% | 13.04% | +2.16pp | +5.12pp | _round101_fastdiscover_records.jsonl |
| 101 | squad_seed2_anchor_recheck2 | squad | med_vs_med | scalar_l8_train1e4 | 18.33% | 19.04% | +0.71pp | +1.01pp | _round101_fastdiscover_records.jsonl |
| 102 | mbpp_seed0_anchor_recheck2 | mbpp | med_vs_med | scalar_l8_sched_cos | 48.07% | 49.26% | +1.20pp | +1.43pp | _round102_fastdiscover_records.jsonl |
| 102 | protein_ss_seed1_anchor_recheck | protein_ss | med_vs_med | head_l8 | 32.23% | 34.62% | +2.39pp | +9.39pp | _round102_fastdiscover_records.jsonl |
| 103 | protein_ss_seed5_discovery | protein_ss | med_vs_med | head_l8 | 26.68% | 28.30% | +1.62pp | +5.22pp | _round103_fastdiscover_records.jsonl |
| 104 | arc_seed4_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 10.34% | 13.07% | +2.73pp | +2.83pp | _round104_fastdiscover_records.jsonl |
| 104 | mbpp_seed4_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 47.53% | 48.82% | +1.29pp | +2.25pp | _round104_fastdiscover_records.jsonl |
| 109 | hotpot_seed6_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 9.01% | 12.59% | +3.59pp | +2.86pp | _round109_fastdiscover_records.jsonl |
| 112 | hotpot_longctx_seed1_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 6.67% | 8.89% | +2.22pp | +2.22pp | _round112_fastdiscover_records.jsonl |
| 115 | punc_seed3_anchor | punc | med_vs_med | scalar_l8_train1e4 | 8.88% | 10.38% | +1.50pp | +1.20pp | _round115_fastdiscover_records.jsonl |
| 118 | hotpot_seed8_discovery | hotpot | med_vs_med | scalar_l8_train8e5 | 10.59% | 11.31% | +0.72pp | +1.54pp | _round118_fastdiscover_records.jsonl |
| 121 | hotpot_seed13_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 9.40% | 10.97% | +1.57pp | +5.41pp | _round121_fastdiscover_records.jsonl |
| 121 | protein_ss_seed10_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 27.66% | 33.62% | +5.96pp | +9.69pp | _round121_fastdiscover_records.jsonl |
| 123 | punc_seed4_anchor | punc | med_vs_med | scalar_l8_train8e5 | 8.48% | 9.28% | +0.80pp | +0.91pp | _round123_fastdiscover_records.jsonl |
| 127 | mbpp_longctx_seed2_repair | mbpp_longctx | med_vs_med | scalar_l8_train1e4 | 19.45% | 20.77% | +1.32pp | +4.71pp | _round127_fastdiscover_records.jsonl |
| 134 | mbpp_seed11_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 46.07% | 49.75% | +3.68pp | +0.85pp | _round134_fastdiscover_records.jsonl |
| 134 | punc_seed7_anchor | punc | med_vs_med | scalar_l8_train1e4 | 9.77% | 10.90% | +1.13pp | +1.10pp | _round134_fastdiscover_records.jsonl |
| 136 | arc_mc_seed3_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 29.17% | 41.67% | +12.50pp | +12.50pp | _round136_fastdiscover_records.jsonl |
| 154 | hotpot_seed25_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 10.88% | 11.50% | +0.62pp | +6.11pp | _round154_fastdiscover_records.jsonl |
| 154 | mbpp_seed18_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 46.40% | 49.35% | +2.95pp | +1.10pp | _round154_fastdiscover_records.jsonl |
| 155 | protein_ss_seed24_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 26.68% | 30.59% | +3.92pp | +5.41pp | _round155_fastdiscover_records.jsonl |
| 156 | mbpp_seed19_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 46.40% | 47.00% | +0.60pp | +2.89pp | _round156_fastdiscover_records.jsonl |
| 157 | arc_mc_seed13_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 29.17% | 50.00% | +20.83pp | +20.83pp | _round157_fastdiscover_records.jsonl |
| 157 | hotpot_seed26_discovery | hotpot | med_vs_med | scalar_l8_train8e5 | 9.10% | 13.30% | +4.20pp | +7.16pp | _round157_fastdiscover_records.jsonl |
| 158 | protein_ss_seed25_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 28.00% | 34.04% | +6.04pp | +2.30pp | _round158_fastdiscover_records.jsonl |
| 159 | mbpp_seed20_headprobe | mbpp | med_vs_med | head_l10_strong | 45.43% | 48.30% | +2.87pp | +1.71pp | _round159_fastdiscover_records.jsonl |
| 159 | punc_seed14_anchor | punc | med_vs_med | head_l8 | 10.88% | 11.05% | +0.17pp | +1.06pp | _round159_fastdiscover_records.jsonl |
| 160 | arc_mc_seed14_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | +4.17pp | _round160_fastdiscover_records.jsonl |
| 161 | protein_ss_seed27_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 27.61% | 29.56% | +1.95pp | +1.95pp | _round161_fastdiscover_records.jsonl |
| 163 | arc_mc_seed16_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 33.33% | 50.00% | +16.67pp | +4.17pp | _round163_fastdiscover_records.jsonl |
| 163 | protein_ss_seed28_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.13% | 30.53% | +1.40pp | +1.40pp | _round163_fastdiscover_records.jsonl |
| 165 | hotpot_seed28_discovery | hotpot | med_vs_med | scalar_l8_train8e5 | 11.71% | 12.28% | +0.57pp | +2.68pp | _round165_fastdiscover_records.jsonl |
| 167 | punc_seed16_anchor | punc | med_vs_med | scalar_l8_train8e5 | 9.80% | 10.03% | +0.22pp | +1.34pp | _round167_fastdiscover_records.jsonl |
| 168 | arc_mc_seed18_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 37.50% | 45.83% | +8.33pp | +4.17pp | _round168_fastdiscover_records.jsonl |
| 169 | arc_mc_seed19_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 37.50% | 41.67% | +4.17pp | +12.50pp | _round169_fastdiscover_records.jsonl |
| 170 | mbpp_seed24_headprobe | mbpp | med_vs_med | scalar_l8_train8e5 | 46.74% | 47.26% | +0.52pp | +2.86pp | _round170_fastdiscover_records.jsonl |
| 171 | arc_mc_seed20_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | +8.33pp | _round171_fastdiscover_records.jsonl |
| 171 | protein_ss_seed32_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 29.41% | 30.09% | +0.68pp | +1.75pp | _round171_fastdiscover_records.jsonl |
| 173 | hotpot_seed30_discovery | hotpot | med_vs_med | head_l8 | 10.88% | 11.35% | +0.48pp | +1.78pp | _round173_fastdiscover_records.jsonl |
| 174 | mbpp_longctx_seed7_repair | mbpp_longctx | med_vs_med | scalar_l8_train1e4 | 22.49% | 22.82% | +0.33pp | +1.64pp | _round174_fastdiscover_records.jsonl |
| 174 | protein_ss_seed33_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 28.78% | 30.20% | +1.42pp | +1.42pp | _round174_fastdiscover_records.jsonl |
| 175 | punc_seed18_anchor | punc | med_vs_med | head_l8 | 9.65% | 10.87% | +1.22pp | +1.33pp | _round175_fastdiscover_records.jsonl |
| 181 | hotpot_seed32_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 10.18% | 10.88% | +0.70pp | +2.17pp | _round181_fastdiscover_records.jsonl |
| 182 | mbpp_longctx_seed8_repair | mbpp_longctx | med_vs_med | head_l8 | 19.64% | 20.18% | +0.54pp | +3.07pp | _round182_fastdiscover_records.jsonl |
| 182 | protein_ss_seed37_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.39% | 31.55% | +2.15pp | +1.56pp | _round182_fastdiscover_records.jsonl |
| 184 | protein_ss_seed38_discovery | protein_ss | med_vs_med | head_l8 | 28.97% | 29.54% | +0.57pp | +1.78pp | _round184_fastdiscover_records.jsonl |
| 187 | arc_mc_seed28_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 33.33% | 37.50% | +4.17pp | +4.17pp | _round187_fastdiscover_records.jsonl |
| 188 | punc_seed21_anchor | punc | med_vs_med | head_l8 | 10.94% | 11.48% | +0.54pp | +1.54pp | _round188_fastdiscover_records.jsonl |
| 189 | hotpot_seed34_discovery | hotpot | med_vs_med | head_l8 | 10.24% | 10.86% | +0.62pp | +2.44pp | _round189_fastdiscover_records.jsonl |
| 191 | mbpp_seed32_headprobe | mbpp | med_vs_med | head_l10_strong | 47.18% | 48.86% | +1.68pp | +0.82pp | _round191_fastdiscover_records.jsonl |
| 193 | arc_mc_seed31_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 33.33% | 41.67% | +8.33pp | +4.17pp | _round193_fastdiscover_records.jsonl |
| 193 | protein_ss_seed43_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 31.16% | 33.60% | +2.44pp | +1.31pp | _round193_fastdiscover_records.jsonl |
| 195 | arc_mc_seed32_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 37.50% | 45.83% | +8.33pp | +8.33pp | _round195_fastdiscover_records.jsonl |
| 196 | punc_seed23_anchor | punc | med_vs_med | head_l8 | 10.41% | 10.72% | +0.31pp | +0.83pp | _round196_fastdiscover_records.jsonl |
| 198 | mbpp_longctx_seed10_repair | mbpp_longctx | med_vs_med | head_l8 | 16.30% | 19.64% | +3.34pp | +6.96pp | _round198_fastdiscover_records.jsonl |
| 198 | protein_ss_seed45_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 27.61% | 31.19% | +3.58pp | +3.69pp | _round198_fastdiscover_records.jsonl |
| 203 | protein_ss_seed48_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 29.92% | 32.38% | +2.46pp | +5.39pp | _round203_fastdiscover_records.jsonl |
| 204 | mbpp_seed37_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 46.25% | 48.80% | +2.55pp | +2.36pp | _round204_fastdiscover_records.jsonl |
| 206 | mbpp_longctx_seed11_repair | mbpp_longctx | med_vs_med | scalar_l8_train8e5 | 20.36% | 22.45% | +2.09pp | +6.65pp | _round206_fastdiscover_records.jsonl |
| 207 | mbpp_seed38_headprobe | mbpp | med_vs_med | scalar_l8_train8e5 | 46.61% | 47.58% | +0.97pp | +1.89pp | _round207_fastdiscover_records.jsonl |
| 209 | arc_mc_seed39_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | +8.33pp | _round209_fastdiscover_records.jsonl |
| 210 | hotpot_seed39_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 10.24% | 13.31% | +3.07pp | +1.26pp | _round210_fastdiscover_records.jsonl |
| 211 | arc_mc_seed40_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 37.50% | 41.67% | +4.17pp | +4.17pp | _round211_fastdiscover_records.jsonl |
| 211 | protein_ss_seed52_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.68% | 30.76% | +1.08pp | +1.08pp | _round211_fastdiscover_records.jsonl |
| 213 | arc_mc_seed41_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 33.33% | 37.50% | +4.17pp | +4.17pp | _round213_fastdiscover_records.jsonl |
| 214 | mbpp_longctx_seed12_repair | mbpp_longctx | med_vs_med | scalar_l8_train8e5 | 18.46% | 22.84% | +4.37pp | +2.31pp | _round214_fastdiscover_records.jsonl |
| 216 | protein_ss_seed54_discovery | protein_ss | med_vs_med | scalar_l8_train8e5 | 29.56% | 31.74% | +2.18pp | +2.18pp | _round216_fastdiscover_records.jsonl |
| 219 | protein_ss_seed56_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 25.64% | 30.39% | +4.75pp | +2.44pp | _round219_fastdiscover_records.jsonl |
| 221 | arc_mc_seed45_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | +8.33pp | _round221_fastdiscover_records.jsonl |
| 221 | hotpot_seed42_discovery | hotpot | med_vs_med | scalar_l8_train8e5 | 10.88% | 13.12% | +2.24pp | +1.78pp | _round221_fastdiscover_records.jsonl |
| 222 | mbpp_longctx_seed13_repair | mbpp_longctx | med_vs_med | scalar_l8_train8e5 | 17.13% | 27.13% | +10.00pp | +1.86pp | _round222_fastdiscover_records.jsonl |
| 222 | protein_ss_seed57_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.68% | 30.97% | +1.29pp | +1.29pp | _round222_fastdiscover_records.jsonl |
| 223 | mbpp_seed44_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 45.94% | 46.07% | +0.13pp | +3.21pp | _round223_fastdiscover_records.jsonl |
| 224 | protein_ss_seed58_discovery | protein_ss | med_vs_med | head_l8 | 28.12% | 31.75% | +3.63pp | +4.96pp | _round224_fastdiscover_records.jsonl |
| 225 | arc_mc_seed47_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 33.33% | 37.50% | +4.17pp | +4.17pp | _round225_fastdiscover_records.jsonl |
| 226 | hotpot_seed43_discovery | hotpot | med_vs_med | scalar_l8_train8e5 | 11.87% | 12.26% | +0.38pp | +1.14pp | _round226_fastdiscover_records.jsonl |
| 226 | mbpp_seed45_headprobe | mbpp | med_vs_med | head_l10_strong | 46.69% | 51.33% | +4.64pp | +2.83pp | _round226_fastdiscover_records.jsonl |
| 227 | protein_ss_seed60_discovery | protein_ss | med_vs_med | head_l8 | 26.68% | 32.10% | +5.42pp | +6.00pp | _round227_fastdiscover_records.jsonl |
| 229 | hotpot_seed44_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 11.71% | 12.40% | +0.69pp | +2.94pp | _round229_fastdiscover_records.jsonl |
| 401 | arc_mc_seed114_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 37.50% | +4.17pp | +4.17pp | _round401_fastdiscover_records.jsonl |
| 401 | protein_ss_seed166_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 26.77% | 26.93% | +0.16pp | +3.00pp | _round401_fastdiscover_records.jsonl |
| 403 | squad_seed48_discovery | squad | med_vs_med | scalar_l8_sched_cos | 17.75% | 18.48% | +0.73pp | +1.01pp | _round403_fastdiscover_records.jsonl |
| 405 | arc_mc_seed115_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 45.83% | +12.50pp | +12.50pp | _round405_fastdiscover_records.jsonl |
| 406 | hotpot_seed67_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 10.50% | 10.86% | +0.36pp | +3.98pp | _round406_fastdiscover_records.jsonl |
| 406 | protein_ss_seed169_discovery | protein_ss | med_vs_med | head_l8 | 26.69% | 30.23% | +3.54pp | +3.30pp | _round406_fastdiscover_records.jsonl |
| 408 | protein_ss_seed170_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 28.88% | 32.56% | +3.68pp | +8.27pp | _round408_fastdiscover_records.jsonl |
| 409 | protein_ss_seed171_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 24.95% | 27.44% | +2.49pp | +3.37pp | _round409_fastdiscover_records.jsonl |
| 410 | mbpp_seed92_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 47.12% | 49.01% | +1.89pp | +1.25pp | _round410_fastdiscover_records.jsonl |
| 412 | protein_ss_seed173_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.68% | 32.66% | +2.98pp | +2.98pp | _round412_fastdiscover_records.jsonl |
| 414 | protein_ss_seed174_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 31.06% | 35.63% | +4.58pp | +4.58pp | _round414_fastdiscover_records.jsonl |
| 417 | arc_mc_seed120_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 37.50% | 50.00% | +12.50pp | +4.17pp | _round417_fastdiscover_records.jsonl |
| 417 | protein_ss_seed176_discovery | protein_ss | med_vs_med | head_l8 | 31.41% | 31.46% | +0.04pp | +2.32pp | _round417_fastdiscover_records.jsonl |
| 419 | squad_seed52_discovery | squad | med_vs_med | scalar_l8_sched_cos | 18.68% | 18.75% | +0.07pp | +2.00pp | _round419_fastdiscover_records.jsonl |
| 420 | protein_ss_seed177_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 23.77% | 26.25% | +2.48pp | +0.90pp | _round420_fastdiscover_records.jsonl |
| 422 | mbpp_seed95_headprobe | mbpp | med_vs_med | head_l10_strong | 46.11% | 48.48% | +2.37pp | +2.28pp | _round422_fastdiscover_records.jsonl |
| 425 | arc_mc_seed122_discovery | arc_mc | med_vs_med | scalar_l8_sched_cos | 33.33% | 41.67% | +8.33pp | +8.33pp | _round425_fastdiscover_records.jsonl |
| 425 | protein_ss_seed180_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 29.13% | 32.88% | +3.74pp | +3.74pp | _round425_fastdiscover_records.jsonl |
| 427 | squad_seed54_discovery | squad | med_vs_med | scalar_l8_train1e4 | 18.12% | 19.72% | +1.61pp | +1.25pp | _round427_fastdiscover_records.jsonl |
| 428 | mbpp_longctx_seed62_repair | mbpp_longctx | med_vs_med | scalar_l8_sched_cos | 22.84% | 23.61% | +0.77pp | +4.33pp | _round428_fastdiscover_records.jsonl |
| 430 | protein_ss_seed182_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 26.68% | 29.94% | +3.27pp | +2.85pp | _round430_fastdiscover_records.jsonl |
| 436 | protein_ss_seed185_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 26.56% | 29.68% | +3.12pp | +3.12pp | _round436_fastdiscover_records.jsonl |
| 438 | protein_ss_seed186_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 29.68% | 34.21% | +4.53pp | +4.53pp | _round438_fastdiscover_records.jsonl |
| 440 | protein_ss_seed187_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 29.68% | 30.79% | +1.11pp | +1.11pp | _round440_fastdiscover_records.jsonl |
| 441 | arc_mc_seed126_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 41.67% | 45.83% | +4.17pp | +4.17pp | _round441_fastdiscover_records.jsonl |
| 441 | protein_ss_seed188_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 28.55% | 32.68% | +4.13pp | +1.87pp | _round441_fastdiscover_records.jsonl |
| 444 | protein_ss_seed189_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 26.68% | 31.57% | +4.90pp | +0.87pp | _round444_fastdiscover_records.jsonl |
| 449 | arc_mc_seed164_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | +8.33pp | _round449_fastdiscover_records.jsonl |
| 449 | protein_ss_seed232_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 29.68% | 32.92% | +3.24pp | +3.24pp | _round449_fastdiscover_records.jsonl |
| 450 | mbpp_seed126_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 45.74% | 46.25% | +0.50pp | +3.12pp | _round450_fastdiscover_records.jsonl |
| 452 | protein_ss_seed233_discovery | protein_ss | med_vs_med | head_l8 | 27.84% | 29.56% | +1.72pp | +1.72pp | _round452_fastdiscover_records.jsonl |
| 455 | squad_seed81_anchor | squad | med_vs_med | scalar_l8_train1e4 | 17.82% | 18.51% | +0.69pp | +1.51pp | _round455_fastdiscover_records.jsonl |
| 456 | protein_ss_seed235_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 31.92% | 33.26% | +1.33pp | +1.33pp | _round456_fastdiscover_records.jsonl |
| 465 | arc_mc_seed170_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 37.50% | +4.17pp | +4.17pp | _round465_fastdiscover_records.jsonl |
| 466 | protein_ss_seed239_discovery | protein_ss | med_vs_med | head_l8 | 27.84% | 35.17% | +7.33pp | +7.33pp | _round466_fastdiscover_records.jsonl |
| 503 | hotpot_seed84_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 11.44% | 14.33% | +2.88pp | +1.16pp | _round503_fastdiscover_records.jsonl |
| 504 | protein_ss_seed252_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 30.41% | 32.94% | +2.53pp | +2.76pp | _round504_fastdiscover_records.jsonl |
| 505 | arc_mc_seed184_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 37.50% | 41.67% | +4.17pp | +4.17pp | _round505_fastdiscover_records.jsonl |
| 507 | mbpp_longctx_seed99_repair | mbpp_longctx | med_vs_med | head_l8 | 19.58% | 27.02% | +7.44pp | +5.72pp | _round507_fastdiscover_records.jsonl |
| 509 | mbpp_seed143_anchor | mbpp | med_vs_med | head_l10_strong | 49.03% | 49.09% | +0.06pp | +1.79pp | _round509_fastdiscover_records.jsonl |
| 509 | protein_ss_seed254_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 30.53% | 32.85% | +2.32pp | +2.32pp | _round509_fastdiscover_records.jsonl |
| 510 | squad_seed95_anchor | squad | med_vs_med | head_l8 | 16.76% | 18.11% | +1.35pp | +2.26pp | _round510_fastdiscover_records.jsonl |
| 511 | hotpot_seed86_discovery | hotpot | med_vs_med | head_l8 | 11.09% | 13.21% | +2.11pp | +4.10pp | _round511_fastdiscover_records.jsonl |
| 521 | mbpp_seed146_headprobe | mbpp | med_vs_med | head_l10_strong | 45.64% | 45.98% | +0.34pp | +1.75pp | _round521_fastdiscover_records.jsonl |
| 522 | protein_ss_seed259_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 30.11% | 35.09% | +4.98pp | +4.98pp | _round522_fastdiscover_records.jsonl |
| 525 | protein_ss_seed260_discovery | protein_ss | med_vs_med | head_l8 | 30.13% | 31.02% | +0.89pp | +0.89pp | _round525_fastdiscover_records.jsonl |
| 526 | mbpp_longctx_seed104_repair | mbpp_longctx | med_vs_med | head_l8 | 18.57% | 24.53% | +5.96pp | +4.49pp | _round526_fastdiscover_records.jsonl |
| 527 | hotpot_seed90_discovery | hotpot | med_vs_med | head_l8 | 12.07% | 12.66% | +0.59pp | +2.18pp | _round527_fastdiscover_records.jsonl |
| 528 | protein_ss_seed261_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 27.86% | 36.00% | +8.14pp | +8.14pp | _round528_fastdiscover_records.jsonl |
| 529 | mbpp_seed148_headprobe | mbpp | med_vs_med | head_l10_strong | 49.89% | 52.73% | +2.84pp | +3.00pp | _round529_fastdiscover_records.jsonl |
| 530 | protein_ss_seed262_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.10% | 31.49% | +2.38pp | +2.38pp | _round530_fastdiscover_records.jsonl |
| 531 | hotpot_seed91_discovery | hotpot | med_vs_med | scalar_l8_train1e4 | 9.41% | 10.10% | +0.69pp | +2.44pp | _round531_fastdiscover_records.jsonl |
| 531 | mbpp_longctx_seed105_repair | mbpp_longctx | med_vs_med | scalar_l8_sched_cos | 21.11% | 23.85% | +2.73pp | +7.43pp | _round531_fastdiscover_records.jsonl |
| 534 | mbpp_longctx_seed106_repair | mbpp_longctx | med_vs_med | head_l8 | 19.06% | 27.75% | +8.69pp | +1.86pp | _round534_fastdiscover_records.jsonl |
| 543 | hotpot_seed94_discovery | hotpot | med_vs_med | scalar_l8_sched_cos | 9.47% | 11.35% | +1.88pp | +4.42pp | _round543_fastdiscover_records.jsonl |
| 544 | protein_ss_seed267_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 25.69% | 27.99% | +2.31pp | +2.01pp | _round544_fastdiscover_records.jsonl |
| 545 | mbpp_seed152_headprobe | mbpp | med_vs_med | scalar_l8_train1e4 | 46.74% | 47.13% | +0.39pp | +4.97pp | _round545_fastdiscover_records.jsonl |
| 550 | protein_ss_seed270_discovery | protein_ss | med_vs_med | scalar_l8_sched_cos | 27.87% | 29.34% | +1.47pp | +1.47pp | _round550_fastdiscover_records.jsonl |
| 551 | arc_mc_seed202_discovery | arc_mc | med_vs_med | scalar_l8_train1e4 | 41.67% | 45.83% | +4.17pp | +8.33pp | _round551_fastdiscover_records.jsonl |
| 552 | protein_ss_seed271_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 28.75% | 31.08% | +2.34pp | +2.34pp | _round552_fastdiscover_records.jsonl |
| 554 | protein_ss_seed272_discovery | protein_ss | med_vs_med | scalar_l8_train1e4 | 29.87% | 31.92% | +2.05pp | +4.35pp | _round554_fastdiscover_records.jsonl |
| 556 | mbpp_seed159_headprobe | mbpp | med_vs_med | scalar_l8_train1e4 | 46.61% | 48.42% | +1.82pp | +1.01pp | _round556_fastdiscover_records.jsonl |
| 557 | arc_mc_seed205_discovery | arc_mc | med_vs_med | scalar_l8_train8e5 | 33.33% | 41.67% | +8.33pp | +8.33pp | _round557_fastdiscover_records.jsonl |
| 557 | mbpp_longctx_seed113_repair | mbpp_longctx | med_vs_med | head_l8 | 16.01% | 19.58% | +3.57pp | +1.64pp | _round557_fastdiscover_records.jsonl |
| 558 | mbpp_seed160_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 45.91% | 47.47% | +1.57pp | +1.87pp | _round558_fastdiscover_records.jsonl |
| 559 | mbpp_seed161_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 45.71% | 45.86% | +0.15pp | +1.51pp | _round559_fastdiscover_records.jsonl |
| 561 | mbpp_seed162_headprobe | mbpp | med_vs_med | scalar_l8_train1e4 | 45.79% | 46.98% | +1.19pp | +1.63pp | _round561_fastdiscover_records.jsonl |
| 563 | mbpp_seed163_anchor | mbpp | med_vs_med | head_l10_strong | 46.97% | 47.58% | +0.61pp | +3.13pp | _round563_fastdiscover_records.jsonl |
| 566 | mbpp_seed165_anchor | mbpp | med_vs_med | scalar_l8_sched_cos | 47.30% | 48.71% | +1.41pp | +1.50pp | _round566_fastdiscover_records.jsonl |
| 567 | mbpp_seed166_headprobe | mbpp | med_vs_med | scalar_l8_sched_cos | 45.56% | 47.14% | +1.58pp | +1.52pp | _round567_fastdiscover_records.jsonl |
| 569 | arc_mc_seed211_repro | arc_mc | med_vs_med | scalar_l8_train1e4 | 33.33% | 41.67% | +8.33pp | +8.33pp | _round569_fastdiscover_records.jsonl |

## Reference Files
- Full comparable table: `results/_unified_med_comparisons.csv`
- Effective-only table: `results/_unified_effective_med_runs.csv`
- Existing rolling log: `results/_rolling_round_log.md`
