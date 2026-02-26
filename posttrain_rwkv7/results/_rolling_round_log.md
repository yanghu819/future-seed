# Future-Seed Post-Training Rolling Round Log

This file tracks round-level execution outcomes with three fixed items:
- best config and gain vs baseline (pp)
- failed/pruned configs and reasons
- next round plan

---

## Round60 - strict-quick (seed0, completed 2026-02-25)

Best config vs baseline:
- `mbpp_strict`: `head_l10_strong` med `49.94%` vs med baseline `48.39%` (**`+1.55pp`**)
- `squad_strict`: `scalar_l8_sched_cos` med `17.47%` vs med baseline `17.27%` (**`+0.20pp`**)

Failed/pruned:
- none pruned in this round (both tasks passed quick gate and entered med)

Next round plan:
1. Run strict seed1 frontier search around Round60 winners only (no broad re-sweep).
2. Keep quick->med gate (`>= +0.30pp`) to avoid low-value long runs.

---

## Round61 - strict frontier (seed1, completed 2026-02-25)

Best config vs baseline:
- `mbpp_strict_seed1`: `head_l10_strong` med `47.01%` vs med baseline `46.67%` (**`+0.35pp`**)
- `squad_strict_seed1`: best quick FS was `scalar_l8_train1e4` `14.58%` vs quick baseline `14.83%` (**`-0.25pp`**)

Failed/pruned:
- `squad_strict_seed1` all tested FS quick variants were negative:
  - `scalar_l8_train1e4`: `-0.25pp`
  - `scalar_l8_sched_cos_highfloor`: `-0.50pp`
  - `scalar_l8_sched_cos`: `-0.75pp`
- SQuAD med pruned due quick gate:
  - reason: `quick_d_acc=-0.25pp < prune_pp=+0.30pp`
- `mbpp_strict_seed1` quick alternatives were not promoted:
  - `head_l10_midlr` and `head_l10_sched_cos` trailed `head_l10_strong` quick score

Next round plan:
1. SQuAD strict rescue with conservative schedules and lower prompt-note pressure.
2. MBPP strict seed2 confirmation on `head_l10_strong` to extend cross-seed evidence.

---

## Round62 - 3h finishpack (completed 2026-02-25)

Best config vs baseline:
- `punc_restore_seed0_scout`: `head_l8` med `11.09%` vs med baseline `10.59%` (**`+0.51pp`**)
- `mbpp_strict_seed2_confirm`: `head_l10_strong` med `45.17%` vs med baseline `47.15%` (**`-1.98pp`**)
- `squad_strict_seed1_rescue`: best quick FS `14.82%` vs quick baseline `14.83%` (**`-0.01pp`**)

Failed/pruned:
- `mbpp_strict_seed2_confirm`:
  - `head_l10_midlr` quick pruned (`-1.26pp < -0.50pp`)
  - winner `head_l10_strong` failed med confirmation (`-1.98pp`)
- `squad_strict_seed1_rescue`:
  - all rescue quick candidates non-positive (`-0.01pp`, `-0.50pp`, `-1.00pp`)
  - med pruned due gate (`best_quick -0.01pp < +0.20pp`)

Next round plan:
1. Prioritize `punc_restore` as current most reliable useful task branch.
2. Freeze or deprioritize unstable branches (`squad_strict`, current `mbpp_strict` recipe) until new hypotheses are ready.

---

## Round63 - useful-followup (completed 2026-02-25)

Best config vs baseline:
- `mbpp_seed2_regrescue`: `head_l10_clip07` med `48.97%` vs med baseline `46.69%` (**`+2.28pp`**)
- `punc_restore_seed1_confirm`: `scalar_l8_sched_cos` med `14.32%` vs med baseline `12.20%` (**`+2.12pp`**)
- `punc_restore_seed2_confirm`: `scalar_l8_sched_cos` med `10.24%` vs med baseline `12.78%` (**`-2.55pp`**)

Failed/pruned:
- `punc_restore_seed2_confirm`:
  - `head_l8` quick pruned (`-0.61pp < -0.50pp`)
  - `scalar_l8_sched_cos` failed med confirmation (`-2.55pp`)
- no crash failures in this round; all failures are metric-based pruning/regression

Next round plan:
1. MBPP: validate rescue winner (`head_l10_clip07`) on additional seeds.
2. PUNC: run one extra seed confirmation before deciding to keep or freeze the branch.

---

## Round64 - mbpp+punc multiseed (completed 2026-02-25)

Best config vs baseline:
- `mbpp_seed0_clip07_confirm`: best quick `head_l10_clip07` `40.71%` vs quick baseline `40.69%` (**`+0.02pp`**)
- `mbpp_seed1_clip07_confirm`: best quick `head_l10_clip07` `39.64%` vs quick baseline `40.86%` (**`-1.22pp`**)
- `punc_seed3_scalar_confirm`: best quick `scalar_l8_sched_cos` `10.71%` vs quick baseline `11.76%` (**`-1.04pp`**)

Failed/pruned:
- `mbpp_seed0_clip07_confirm`: med skipped (`best_quick +0.02pp < +0.20pp gate`)
- `mbpp_seed1_clip07_confirm`:
  - `head_l10_clip07` quick-pruned (`-1.22pp < -0.50pp`)
  - `head_l10_strong` quick-pruned (`-1.22pp < -0.50pp`)
  - med skipped
- `punc_seed3_scalar_confirm`:
  - `scalar_l8_sched_cos` quick-pruned (`-1.04pp < -0.50pp`)
  - `head_l8` quick-pruned (`-1.80pp < -0.50pp`)
  - med skipped

Next round plan:
1. Shift focus from PUNC to real-task SQuAD branch for higher practical value.
2. Continue MBPP only with strict early prune and no broad seed expansion.

---

## Round65 - mbpp+squad seed2/seed3 (completed 2026-02-25)

Best config vs baseline:
- `squad_strict_seed2`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)
- `mbpp_seed3_regrescue`: best quick `head_l10_strong` `42.35%` vs quick baseline `43.11%` (**`-0.76pp`**)

Failed/pruned:
- `mbpp_seed3_regrescue`:
  - `head_l10_clip07` quick-pruned (`-2.89pp < -0.50pp`)
  - `head_l10_strong` quick-pruned (`-0.76pp < -0.50pp`)
  - med skipped
- `squad_strict_seed2`:
  - `scalar_l8_sched_cos` quick-pruned (`-0.51pp < -0.50pp`)

Next round plan:
1. Reconfirm SQuAD train1e4 on another seed to test stability.
2. Retry MBPP seed3 using strict-base alternatives to check if quick-positive can survive med.

---

## Round66 - squad+mbpp frontier (completed 2026-02-25)

Best config vs baseline:
- `squad_seed0_train1e4_confirm`: `scalar_l8_train1e4` med `19.00%` vs med baseline `17.27%` (**`+1.73pp`**)
- `mbpp_strict_seed3_alt`: `head_l10_strong` med `46.76%` vs med baseline `48.23%` (**`-1.47pp`**)

Failed/pruned:
- `mbpp_strict_seed3_alt`:
  - no quick prune (`head_l10_strong +0.95pp`, `head_l8_nodetach +0.56pp`)
  - med regression (`head_l10_strong -1.47pp`)

Next round plan:
1. Extend SQuAD train1e4 to seed3 to test whether med gains continue.
2. Reconfirm MBPP strict on seed0 to compare against seed3 failure pattern.

---

## Round67 - squad3+mbpp0 (completed 2026-02-25)

Best config vs baseline:
- `mbpp_seed0_reconfirm_strict`: `head_l10_strong` med `49.94%` vs med baseline `48.39%` (**`+1.54pp`**)
- `squad_seed3_train1e4_frontier`: best quick `scalar_l8_train1e4` `14.06%` vs quick baseline `14.67%` (**`-0.61pp`**)

Failed/pruned:
- `squad_seed3_train1e4_frontier`:
  - `scalar_l8_train1e4` quick-pruned (`-0.61pp < -0.50pp`)
  - `scalar_l8_sched_cos` quick-pruned (`-0.84pp < -0.50pp`)
  - med skipped (`best_quick -0.61pp < med_gate -0.50pp`)
- `mbpp_seed0_reconfirm_strict`:
  - no prune; both quick candidates positive and winner passed med with `+1.54pp`

Next round plan:
1. SQuAD: run targeted rescue on seed1/seed3 to reduce seed split (train1e4 neighborhood only).
2. MBPP: verify whether seed2/seed3 can be repaired or freeze MBPP as seed-conditional positive branch.

---

## Round68 - squad head rescue (completed 2026-02-25)

Best config vs baseline:
- `squad_seed1_head_rescue`: best quick `head_l8_nodetach` `14.58%` vs quick baseline `14.83%` (**`-0.25pp`**)
- `squad_seed3_head_rescue`: best quick `head_l8_nodetach` `13.90%` vs quick baseline `14.67%` (**`-0.76pp`**)

Failed/pruned:
- `squad_seed1_head_rescue`:
  - `head_l8` quick-pruned (`-0.76pp < -0.50pp`)
  - med skipped (`best_quick -0.25pp < med_gate -0.20pp`)
- `squad_seed3_head_rescue`:
  - `head_l8` quick-pruned (`-1.35pp < -0.50pp`)
  - `head_l8_nodetach` quick-pruned (`-0.76pp < -0.50pp`)
  - med skipped (`best_quick -0.76pp < med_gate -0.20pp`)

Next round plan:
1. Drop SQuAD head rescue direction and try scalar micro adjustments on seed1.
2. For MBPP seed3, run dual med confirmation to avoid quick-only false positives.

---

## Round69 - mbpp+squad rescue (completed 2026-02-25)

Best config vs baseline:
- `mbpp_seed3_dualmed_rescue`: `head_l8_nodetach` med `48.28%` vs med baseline `48.23%` (**`+0.05pp`**)
- `squad_seed1_scalar_micro`: best quick `scalar_l8_train1e4` `14.58%` vs quick baseline `14.83%` (**`-0.25pp`**)

Failed/pruned:
- `mbpp_seed3_dualmed_rescue`:
  - `head_l10_strong` med regression (`-1.47pp`)
  - only `head_l8_nodetach` remained near-neutral positive (`+0.05pp`)
- `squad_seed1_scalar_micro`:
  - all scalar micro candidates non-positive (`-0.25pp`, `-0.25pp`, `-0.42pp`)
  - med skipped (`best_quick -0.25pp < med_gate -0.10pp`)

Next round plan:
1. SQuAD: attempt seed3 scalar micro rescue with strict prune.
2. MBPP: recheck seed2 rescue (`clip07` + `strong`) with dual med promotion.

---

## Round70 - squad3+mbpp2 (completed 2026-02-25)

Best config vs baseline:
- `mbpp_seed2_dualmed_recheck`:
  - `head_l10_clip07` med `48.97%` vs med baseline `46.69%` (**`+2.28pp`**)
  - `head_l10_strong` med `48.97%` vs med baseline `46.69%` (**`+2.28pp`**)
- `squad_seed3_scalar_micro`: best quick `scalar_l8_train8e5` `14.41%` vs quick baseline `14.67%` (**`-0.26pp`**)

Failed/pruned:
- `squad_seed3_scalar_micro`:
  - `scalar_l8_train1e4` quick-pruned (`-0.61pp < -0.50pp`)
  - `scalar_l8_train1e4_clip07` quick-pruned (`-0.59pp < -0.50pp`)
  - med skipped (`best_quick -0.26pp < med_gate -0.10pp`)

Next round plan:
1. MBPP: move to seed3 rescue recheck under note-pool-384 base to reduce seed split.
2. SQuAD: reconfirm seed2 scalar positives and decide whether to freeze seed1/3 rescue branch.

---

## Round71 - mbpp3+squad2 (completed 2026-02-25)

Best config vs baseline:
- `squad_seed2_scalar_reconfirm`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)
- `mbpp_seed3_rescue384`: best quick `head_l10_strong` `42.35%` vs quick baseline `43.11%` (**`-0.76pp`**, med pruned)

Failed/pruned:
- `mbpp_seed3_rescue384`:
  - `head_l10_clip07` quick-pruned (`-2.89pp < -0.50pp`)
  - `head_l10_strong` quick-pruned (`-0.76pp < -0.50pp`)
  - med skipped (`best_quick -0.76pp < med_gate +0.20pp`)
- `squad_seed2_scalar_reconfirm`:
  - `scalar_l8_train1e4_clip07` quick-pruned (`-1.00pp < -0.50pp`)

Next round plan:
1. keep SQuAD seed2/seed0 as positive anchor branch.
2. attempt MBPP seed3 nodetach rescue to test whether detach setting caused collapse.

---

## Round72 - mbpp3+squad1 rescue (completed 2026-02-26)

Best config vs baseline:
- `squad_seed1_lowpressure_rescue`: `scalar_l8_train1e4` med `19.20%` vs med baseline `19.38%` (**`-0.18pp`**)
- `mbpp_seed3_nodetach_rescue384`: best quick `head_l10_nodetach_clip07` `41.35%` vs quick baseline `43.11%` (**`-1.76pp`**, med pruned)

Failed/pruned:
- `mbpp_seed3_nodetach_rescue384`:
  - `head_l8_nodetach` quick-pruned (`-2.01pp < -0.50pp`)
  - `head_l10_nodetach_clip07` quick-pruned (`-1.76pp < -0.50pp`)
  - med skipped (`best_quick -1.76pp < med_gate +0.00pp`)
- `squad_seed1_lowpressure_rescue`:
  - `scalar_l8_train1e4_clip07` quick-pruned (`-0.67pp < -0.50pp`)

Next round plan:
1. return to confirmed positive seeds to maintain useful-task hit-rate.
2. run strict reconfirm pairs (MBPP seed1 + SQuAD seed0) for stronger stability evidence.

---

## Round73 - mbpp1+squad0 recheck (completed 2026-02-26)

Best config vs baseline:
- `squad_seed0_frontier_recheck`: `scalar_l8_train1e4` med `19.00%` vs med baseline `17.27%` (**`+1.73pp`**)
- `mbpp_seed1_strict_recheck`: `head_l10_strong` med `47.01%` vs med baseline `46.67%` (**`+0.35pp`**)

Failed/pruned:
- no quick/med prune in this round; both tasks passed quick gates and produced positive med deltas.

Next round plan:
1. expand beyond MBPP/SQuAD with one additional real-task branch (`punc_restore`) while keeping strict prune.
2. reconfirm SQuAD seed2 under same frontier to check cross-round consistency.

---

## Round74 - punc1+squad2 frontier (completed 2026-02-26)

Best config vs baseline:
- `punc_seed1_frontier_recheck`: `scalar_l8_train1e4` med `13.04%` vs med baseline `12.20%` (**`+0.84pp`**)
- `squad_seed2_frontier_recheck`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)

Failed/pruned:
- `squad_seed2_frontier_recheck`:
  - `scalar_l8_sched_cos` quick-pruned (`-0.51pp < -0.50pp`)
- `punc_seed1_frontier_recheck`:
  - `head_l8` quick under baseline (`-0.12pp`), not promoted

Next round plan:
1. continue high-value branch around `scalar_l8_train1e4` for SQuAD/PUNC, test minimal LR/clip perturbations only.
2. launch one targeted MBPP seed3 rescue with strict quick gate; stop immediately if quick < baseline by `0.5pp`.

---

## Round75 - mbpp3+punc0 targeted (completed 2026-02-26)

Best config vs baseline:
- `punc_seed0_frontier_recheck`: `head_l8` med `11.09%` vs med baseline `10.59%` (**`+0.51pp`**)
- `mbpp_seed3_l8_refine`: `head_l8_nodetach` med `48.28%` vs med baseline `48.23%` (**`+0.05pp`**)

Failed/pruned:
- `punc_seed0_frontier_recheck`:
  - `scalar_l8_train1e4` quick-pruned (`-4.17pp < -0.50pp`)
  - `scalar_l8_sched_cos` quick `-0.26pp`, not promoted
- `mbpp_seed3_l8_refine`:
  - `head_l8_nodetach_midlr` quick `-0.23pp` (not promoted)

Next round plan:
1. keep `head_l8_nodetach` as MBPP seed3 near-neutral fallback, but reduce further budget on this branch.
2. prioritize higher-yield SQuAD/PUNC positive branches for useful-task discovery speed.

---

## Round76 - punc1+squad0 dualmed (completed 2026-02-26)

Best config vs baseline:
- `punc_seed1_dualmed_compare`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)
- `squad_seed0_dualmed_compare`: `scalar_l8_train1e4` med `19.00%` vs med baseline `17.27%` (**`+1.73pp`**)

Failed/pruned:
- `punc_seed1_dualmed_compare`:
  - `head_l8` quick `-0.12pp` (not promoted)
  - `scalar_l8_sched_cos` med `-0.41pp` vs med baseline
- `squad_seed0_dualmed_compare`:
  - `scalar_l8_train8e5` quick `+0.25pp` but未进入 top2 med

Next round plan:
1. freeze `scalar_l8_train1e4` as PUNC/SQuAD default high-yield recipe under current budget.
2. continue only one低优先 MBPP seed3 near-neutral track，主预算转向高收益任务。

---

## Round77 - fastdiscover (new tasks, completed 2026-02-26)

Best config vs baseline:
- `arc_seed0_discovery`: `scalar_l8_train8e5` med `12.63%` vs med baseline `12.33%` (**`+0.29pp`**)
- `hotpot_seed0_discovery`: `scalar_l8_train8e5` med `10.36%` vs med baseline `10.48%` (**`-0.12pp`**)

Failed/pruned:
- `hotpot_seed0_discovery`:
  - `scalar_l8_train1e4` quick-pruned (`-4.17pp < -0.50pp`)
  - promoted candidate failed med confirmation (`-0.12pp`)

Next round plan:
1. Continue broad search with protein and wiki branches.
2. Keep strict quick prune and single-top1 med promotion.

---

## Round78 - fastdiscover (new tasks, completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed0_discovery`: `scalar_l8_train1e4` med `30.77%` vs med baseline `27.17%` (**`+3.60pp`**)

Failed/pruned:
- `wiki_seed0_discovery` baseline failed (offline dataset cache miss).
- `protein_ss_seed0_discovery`:
  - `scalar_l8_train8e5` quick-pruned (`-1.64pp < -0.50pp`)

Next round plan:
1. Keep protein branch expansion (new seeds).
2. Investigate wiki/protein_contact failure causes and patch queue/env.

---

## Round79 - fastdiscover (new tasks, completed 2026-02-26)

Best config vs baseline:
- `hotpot_seed1_discovery`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)

Failed/pruned:
- `protein_contact_seed0_discovery` baseline failed:
  - dataset built only `168` val examples vs requested `200`.

Next round plan:
1. Run ARC seed extension with same quick->med policy.
2. Patch `protein_contact` `n_val` to a feasible value.

---

## Round80 - fastdiscover (new tasks, completed 2026-02-26)

Best config vs baseline:
- `arc_seed1_discovery`: best quick `scalar_l8_sched_cos` `9.49%` vs quick baseline `9.09%` (**`+0.41pp`**, med skipped)

Failed/pruned:
- `arc_seed1_discovery`:
  - `scalar_l8_train1e4` quick-pruned (`-1.45pp < -0.50pp`)
  - `scalar_l8_train8e5` quick-pruned (`-1.16pp < -0.50pp`)
  - med skipped (`best_quick +0.41pp < +0.80pp`)
- `wiki_seed1_discovery` baseline failed (offline dataset cache miss).

Next round plan:
1. Start 30% anchor calibration block.
2. Preserve strict +0.8pp promotion to control wasted med budget.

---

## Round81 - fastdiscover (anchor calibration, completed 2026-02-26)

Best config vs baseline:
- `punc_seed1_anchor`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)
- `squad_seed2_anchor`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)

Failed/pruned:
- `squad_seed2_anchor`:
  - `scalar_l8_sched_cos` quick-pruned (`-0.51pp < -0.50pp`)

Next round plan:
1. Continue anchor calibration with MBPP/SQuAD seed0.
2. Keep new-task expansion as main budget after anchor check.

---

## Round82 - fastdiscover (anchor calibration, completed 2026-02-26)

Best config vs baseline:
- `mbpp_seed2_anchor`: `scalar_l8_sched_cos` med `47.58%` vs med baseline `46.69%` (**`+0.89pp`**)
- `squad_seed0_anchor`: best quick `scalar_l8_train1e4` `14.05%` vs quick baseline `13.33%` (**`+0.72pp`**, med skipped)

Failed/pruned:
- `squad_seed0_anchor` med skipped (`best_quick +0.72pp < +0.80pp promote gate`).

Next round plan:
1. Launch round83-88 queue with failure-fix paths (`wiki` online mode, `protein_contact` n_val fix).
2. Expand to new seeds (`hotpot/arc/protein_ss`) while keeping 70/30 new-vs-anchor mix.

---

## Round89 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `protein_contact_seed0_discovery_fix`: best quick `+0.00pp`, med skipped (`< +0.80pp gate`)

Failed/pruned:
- no crash failures.
- all FS quick scores tied baseline; no candidate reached med promotion gate.

Next round plan:
1. run seed1 contact fix once to confirm whether tie ceiling persists.
2. if still tied, reduce budget on protein_contact branch.

---

## Round90 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `protein_contact_seed1_discovery_fix`: best quick `+0.00pp`, med skipped (`< +0.80pp gate`)

Failed/pruned:
- no crash failures.
- same tie ceiling as seed0 (`96.88%` baseline and FS all tied).

Next round plan:
1. mark protein_contact as low-value in this setup.
2. shift budget to hotpot/arc/protein_ss new-seed tasks.

---

## Round91 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `hotpot_seed2_discovery`: best quick `scalar_l8_train8e5` `+1.10pp`, but med `-2.74pp` vs med baseline
- `arc_seed2_discovery`: best quick `-0.51pp`, med skipped

Failed/pruned:
- `hotpot_seed2_discovery`:
  - `scalar_l8_train1e4` quick-pruned (`-4.18pp < -0.50pp`)
  - promoted candidate failed med confirmation (`-2.74pp`) and flagged strong negative
- `arc_seed2_discovery`:
  - all candidates quick-pruned (`-0.51pp`, `-0.82pp`, `-1.65pp`)

Next round plan:
1. deprioritize hotpot/arc seed2 neighborhood due strong negatives.
2. prioritize protein_ss seed1 + anchor checks.

---

## Round92 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed1_discovery`: `head_l8` med `34.62%` vs med baseline `32.23%` (**`+2.39pp`**)

Failed/pruned:
- `arc_easy_seed0_discovery` baseline failed (current trainer/data path incompatibility).
- `protein_ss_seed1_discovery`:
  - `scalar_l8_train8e5` quick `-0.24pp` (not promoted)

Next round plan:
1. keep protein_ss branch as highest-value new-task direction.
2. continue anchor calibration on MBPP/SQuAD.

---

## Round93 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `mbpp_seed0_anchor`: `scalar_l8_train8e5` med `50.10%` vs med baseline `48.07%` (**`+2.03pp`**)
- `squad_seed3_anchor`: best quick `-0.26pp`, med skipped

Failed/pruned:
- `squad_seed3_anchor`:
  - quick-pruned: `scalar_l8_train1e4` (`-0.61pp`), `scalar_l8_sched_cos` (`-0.84pp`)
  - med skipped (`best_quick -0.26pp < +0.80pp gate`)

Next round plan:
1. finalize round94 anchor check (`squad_seed1`, `punc_seed0`).
2. freeze squad seed3 branch unless new hypotheses appear.

---

## Round94 - fastdiscover continuation (completed 2026-02-26)

Best config vs baseline:
- `punc_seed0_anchor`: best quick `scalar_l8_train8e5` `+1.92pp`, med `-0.12pp` vs med baseline
- `squad_seed1_anchor`: best quick `-0.25pp`, med skipped

Failed/pruned:
- `punc_seed0_anchor`:
  - quick-positive/med-negative reversal (`+1.92pp` -> `-0.12pp`)
- `squad_seed1_anchor`:
  - `scalar_l8_sched_cos` quick-pruned (`-0.75pp < -0.50pp`)
  - med skipped (`best_quick -0.25pp < +0.80pp gate`)

Next round plan:
1. keep MBPP + protein_ss as primary high-yield branches.
2. deprioritize protein_contact/arc_seed2/squad_seed1|3 and only revisit with new recipe class.

---

## Round95 - focus queue (completed 2026-02-26)

Best config vs baseline:
- `mbpp_seed1_anchor`: best quick `scalar_l8_train8e5` `+2.25pp`, but med `-1.51pp` vs med baseline
- `protein_ss_seed2_discovery`: best quick `-0.25pp`, med skipped

Failed/pruned:
- `mbpp_seed1_anchor`:
  - quick-positive/med-negative reversal (`+2.25pp -> -1.51pp`)
- `protein_ss_seed2_discovery`:
  - `scalar_l8_train8e5` quick-pruned (`-1.34pp < -0.50pp`)
  - `head_l8` quick-pruned (`-2.02pp < -0.50pp`)

Next round plan:
1. continue protein_ss new-seed discovery to test whether seed2 is outlier.
2. probe MBPP seed3 quickly and prune on first negative signal.

---

## Round96 - focus queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed3_discovery`: `scalar_l8_train1e4` med `32.97%` vs med baseline `31.29%` (**`+1.68pp`**)

Failed/pruned:
- `mbpp_seed3_anchor`:
  - all quick candidates negative (`-0.51pp`, `-1.12pp`, `-3.01pp`), all pruned
- `protein_ss_seed3_discovery`:
  - `scalar_l8_train8e5` quick-pruned (`-1.60pp < -0.50pp`)

Next round plan:
1. run anchor reconfirm pair (`squad_seed2`, `hotpot_seed3`) for stability map.
2. keep protein_ss as top discovery branch.

---

## Round97 - focus queue (completed 2026-02-26)

Best config vs baseline:
- `squad_seed2_anchor_recheck`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)

Failed/pruned:
- `hotpot_seed3_discovery`:
  - all quick candidates negative (`-0.52pp`, `-1.04pp`, `-1.43pp`), med skipped
- `squad_seed2_anchor_recheck`:
  - `scalar_l8_sched_cos` quick-pruned (`-0.51pp < -0.50pp`)

Next round plan:
1. run final anchor round (`arc_seed3`, `punc_seed1`) to close this queue.
2. freeze hotpot seed3 branch under current recipe family.

---

## Round98 - focus queue (completed 2026-02-26)

Best config vs baseline:
- `punc_seed1_anchor_recheck`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)
- `arc_seed3_discovery`: best quick `-0.51pp`, med skipped

Failed/pruned:
- `arc_seed3_discovery`:
  - all quick candidates negative (`-0.51pp`, `-0.87pp`, `-1.92pp`), all pruned

Next round plan:
1. prioritize high-yield pool (`protein_ss`, `punc_seed1`, `mbpp_seed0`, `squad_seed2`) for compact confirmation.
2. stop spending budget on `arc_seed2/3` and `hotpot_seed2/3` unless new config family is introduced.

---

## Round99 - focus2 queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed4_discovery`: `scalar_l8_train8e5` med `35.20%` vs med baseline `30.30%` (**`+4.90pp`**)
- `mbpp_seed2_headprobe`: `head_l10_strong` med `48.97%` vs med baseline `46.69%` (**`+2.28pp`**)

Failed/pruned:
- `protein_ss_seed4_discovery`:
  - `scalar_l8_train1e4` quick-pruned (`-1.16pp < -0.50pp`)
  - `head_l8` quick-pruned (`-1.01pp < -0.50pp`)

Next round plan:
1. continue round100-102 to expand novel-task hits under same gate policy.
2. prioritize `protein_ss`-family follow-ups for seed transfer checks if later rounds stall.

---

## Round100 - focus2 queue (completed 2026-02-26)

Best config vs baseline:
- `hotpot_seed1_headprobe`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)
- `arc_seed0_headprobe`: `scalar_l8_train8e5` med `12.63%` vs med baseline `12.33%` (**`+0.29pp`**)

Failed/pruned:
- none (all quick candidates stayed above prune threshold; only top1 promoted to med by gate)

Next round plan:
1. execute round101 anchor recheck (`squad_seed2`, `punc_seed1`) and keep same quick/med gates.
2. if anchor rounds flatten, move budget back to new-task pool with protein/hotpot variants.

---

## Round101 - focus2 queue (completed 2026-02-26)

Best config vs baseline:
- `punc_seed1_anchor_recheck2`: `scalar_l8_train1e4` med `13.04%` vs med baseline `10.88%` (**`+2.16pp`**)
- `squad_seed2_anchor_recheck2`: `scalar_l8_train1e4` med `19.04%` vs med baseline `18.33%` (**`+0.71pp`**)

Failed/pruned:
- none (all quick candidates above prune threshold; top1 promoted by gate)

Next round plan:
1. finish round102 (`mbpp_seed0_anchor_recheck2`, `protein_ss_seed1_anchor_recheck`) and close this queue.
2. promote high-yield branches (`protein_ss`, `hotpot/punc`) for next discovery package.

---

## Round102 - focus2 queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed1_anchor_recheck`: `head_l8` med `34.62%` vs med baseline `32.23%` (**`+2.39pp`**)
- `mbpp_seed0_anchor_recheck2`: `scalar_l8_sched_cos` med `49.26%` vs med baseline `48.07%` (**`+1.20pp`**)

Failed/pruned:
- `protein_ss_seed1_anchor_recheck`: `scalar_l8_train8e5` quick `-0.24pp` (non-winning, not promoted)

Next round plan:
1. build next discovery queue around high-yield families (`protein_ss` + `head_l8`, `mbpp` + `scalar_l8_sched_cos`).
2. keep 30% anchor calibration and spend 70% on new tasks to expand useful-task pool.
