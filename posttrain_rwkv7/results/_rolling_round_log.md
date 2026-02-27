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

---

## Round103 - expand queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed5_discovery`: `head_l8` med `28.30%` vs med baseline `26.68%` (**`+1.62pp`**)

Failed/pruned:
- `hotpot_seed4_discovery`:
  - `scalar_l8_train8e5` quick-pruned (`-0.75pp < -0.50pp`)
  - med skipped: best quick `+0.79pp` below promote gate `+0.80pp`

Next round plan:
1. complete round104 (`arc_seed4_discovery`, `mbpp_seed4_headprobe`) under same gate/prune policy.
2. keep `protein_ss + head_l8` as primary positive branch for new-task expansion.

---

## Round104 - expand queue (completed 2026-02-26)

Best config vs baseline:
- `arc_seed4_discovery`: `scalar_l8_train1e4` med `13.07%` vs med baseline `10.34%` (**`+2.73pp`**)
- `mbpp_seed4_headprobe`: `scalar_l8_sched_cos` med `48.82%` vs med baseline `47.53%` (**`+1.29pp`**)

Failed/pruned:
- `arc_seed4_discovery`: `head_l8` quick-pruned (`-0.86pp < -0.50pp`)
- `mbpp_seed4_headprobe`: `head_l10_strong` and `head_l10_clip07` quick-pruned (`-0.51pp`)

Next round plan:
1. continue new-task expansion, prioritizing `arc + scalar_l8_train1e4` and `mbpp + scalar_l8_sched_cos` on fresh seeds.
2. keep `protein_ss + head_l8` as primary branch and reduce budget on near-threshold hotpot lines.

---

## Round105 - expand2 queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed6_discovery`: best quick `scalar_l8_train1e4` `31.82%` vs baseline `31.37%` (**`+0.44pp`**, med skipped)
- `arc_seed5_discovery`: best quick `scalar_l8_train1e4` `8.54%` vs baseline `10.72%` (**`-2.19pp`**, med skipped)

Failed/pruned:
- `arc_seed5_discovery`: all candidates quick-pruned (`-2.19pp`, `-2.25pp`, `-3.42pp`)
- `protein_ss_seed6_discovery`: `scalar_l8_train8e5` quick-pruned (`-3.50pp`)

Next round plan:
1. complete round106 (`mbpp_seed5_headprobe`, `hotpot_seed5_discovery`) and check if scalar cosine remains robust.
2. freeze arc seed5 route under current recipe family (strong negative quick).

---

## Round106 - expand2 queue (completed 2026-02-26)

Best config vs baseline:
- `mbpp_seed5_headprobe`: best quick `head_l10_strong` `40.35%` vs baseline `41.36%` (**`-1.01pp`**, med skipped)
- `hotpot_seed5_discovery`: best quick `scalar_l8_train1e4` `7.13%` vs baseline `8.41%` (**`-1.28pp`**, med skipped)

Failed/pruned:
- `mbpp_seed5_headprobe`: all candidates quick-pruned (`-1.01pp`, `-1.01pp`, `-1.27pp`)
- `hotpot_seed5_discovery`: all candidates quick-pruned (`-1.28pp`, `-1.37pp`, `-1.57pp`)

Next round plan:
1. expand to new task families (`arc_mc`, `hotpot_longctx`, `mbpp_longctx`) to avoid local optimum loops.
2. keep a light anchor lane (`squad_seed4`, `punc_seed2`) while prioritizing fresh-seed discovery.

---

## Round107 - new-task expansion (completed 2026-02-26)

Best config vs baseline:
- `hotpot_longctx_seed0_discovery`: best quick `scalar_l8_train1e4` `8.89%` vs baseline `8.89%` (**`+0.00pp`**, med skipped)
- `arc_mc_seed0_discovery`: best quick `scalar_l8_train1e4` `33.33%` vs baseline `37.50%` (**`-4.17pp`**, med skipped)

Failed/pruned:
- `arc_mc_seed0_discovery`: all candidates quick-pruned (`-4.17pp`, `-4.17pp`, `-4.17pp`)
- `hotpot_longctx_seed0_discovery`: no gain (all candidates `+0.00pp`), failed promote gate

Next round plan:
1. continue queue to round108/109 while keeping strict quick-prune.
2. watch `mbpp_longctx` data-build feasibility and cut failing configs quickly.

---

## Round108 - new-task expansion (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed7_discovery`: best quick `head_l8` `30.35%` vs baseline `32.91%` (**`-2.56pp`**, med skipped)
- `mbpp_longctx_seed0_discovery`: quick baseline failed (med skipped)

Failed/pruned:
- `mbpp_longctx_seed0_discovery`: baseline failed (`Only built 374 examples (wanted 600)`)
- `protein_ss_seed7_discovery`: all candidates quick-pruned (`-2.56pp`, `-3.02pp`, `-4.91pp`)

Next round plan:
1. continue round109 (`hotpot_seed6`, `arc_seed6`) and keep aggressive pruning.
2. reduce or replace unstable longctx MBPP settings in next queue package.

Execution continuity:
- queued `results/_search_queue_round113_120_iter.json` and deployed remote auto-chainer so next queue starts immediately after round112.

---

## Round109 - new-task expansion (completed 2026-02-26)

Best config vs baseline:
- `hotpot_seed6_discovery`: `scalar_l8_train1e4` med `12.59%` vs med baseline `9.01%` (**`+3.59pp`**)
- `arc_seed6_discovery`: best quick `head_l8` `9.26%` vs baseline `9.02%` (**`+0.24pp`**, med skipped)

Failed/pruned:
- `arc_seed6_discovery`:
  - `scalar_l8_train8e5` quick-pruned (`-0.76pp < -0.50pp`)
  - med skipped (`best_quick +0.24pp < +0.80pp gate`)

Next round plan:
1. continue round110 (`mbpp_seed6_headprobe`, `squad_seed4_anchor`) with same quick->med gates.
2. keep longctx lines in repaired configuration only (`mbpp_longctx_*_repair`) for queued round116/120.

---

## Round110 - new-task expansion (completed 2026-02-26)

Best config vs baseline:
- `mbpp_seed6_headprobe`: best quick `scalar_l8_sched_cos` `42.54%` vs baseline `42.61%` (**`-0.07pp`**, med skipped)
- `squad_seed4_anchor`: quick baseline failed, med skipped

Failed/pruned:
- `mbpp_seed6_headprobe`: `head_l10_strong` and `head_l10_clip07` quick-pruned (`-1.89pp`, `-2.42pp`)
- `squad_seed4_anchor`: baseline run failed (task skipped)

Next round plan:
1. proceed to round111 anchor check (`punc_seed2`, `arc_mc_seed1`) for quick filtering.
2. keep MBPP headprobe on new seeds only if quick passes +0.80pp gate.

---

## Round111 - iter queue (completed 2026-02-26)

Best config vs baseline:
- `arc_mc_seed1_discovery`: best quick tie (`+0.00pp`), med skipped
- `punc_seed2_anchor`: best quick `-0.63pp`, med skipped

Failed/pruned:
- `punc_seed2_anchor`: all candidates quick-pruned (`-0.63pp`, `-0.69pp`, `-0.70pp`)
- `arc_mc_seed1_discovery`: no gain (all candidates `+0.00pp`)

Next round plan:
1. continue round112 longctx check (`hotpot_longctx_seed1`, `mbpp_longctx_seed1`).
2. keep only longctx variants that pass buildability + quick gate.

---

## Round112 - iter queue (completed 2026-02-26)

Best config vs baseline:
- `hotpot_longctx_seed1_discovery`: `scalar_l8_train1e4` med `8.89%` vs med baseline `6.67%` (**`+2.22pp`**)
- `mbpp_longctx_seed1_discovery`: quick baseline failed (med skipped)

Failed/pruned:
- `hotpot_longctx_seed1_discovery`: `scalar_l8_train8e5` and `scalar_l8_sched_cos` quick-pruned (`-1.11pp`)
- `mbpp_longctx_seed1_discovery`: baseline failed under longctx example-build constraints

Next round plan:
1. continue iter queue round113 (`hotpot_seed7`, `protein_ss_seed8`) and prune aggressively.
2. keep `mbpp_longctx_*_repair` only in later rounds (116/120) with reduced build pressure.

---

## Round113 - iter queue (completed 2026-02-26)

Best config vs baseline:
- `hotpot_seed7_discovery`: best quick tie (`+0.00pp`), med skipped
- `protein_ss_seed8_discovery`: best quick tie (`+0.00pp`), med skipped

Failed/pruned:
- none by quick-prune threshold, but both tasks failed promote gate (no deltas)

Next round plan:
1. run round114 (`mbpp_seed7_headprobe`, `arc_seed7_discovery`) and keep top1 med policy.
2. continue nonstop iteration to round120 via active chainer.

---

## Round121 - fastloop queue (completed 2026-02-26)

Best config vs baseline:
- `protein_ss_seed10_discovery`: `scalar_l8_train8e5` med `33.62%` vs med baseline `27.66%` (**`+5.96pp`**)
- `hotpot_seed13_discovery`: `scalar_l8_train1e4` med `10.97%` vs med baseline `9.40%` (**`+1.57pp`**)

Failed/pruned:
- none (both tasks passed quick gates and produced positive med gains)

Next round plan:
1. continue round122-128 on the active queue (`hotpot/protein_ss + longctx + anchors`).
2. auto-chain into round129-136 without idle time for broader task discovery.

---

## Round122-136 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed3_discovery`: `scalar_l8_sched_cos` med **`+12.50pp`**
- `mbpp_seed11_anchor`: `scalar_l8_sched_cos` med **`+3.68pp`**
- `mbpp_longctx_seed2_repair`: `scalar_l8_train1e4` med **`+1.32pp`**
- `punc_seed7_anchor`: `scalar_l8_train1e4` med **`+1.13pp`**
- `punc_seed4_anchor`: `scalar_l8_train8e5` med **`+0.80pp`**

Failed/pruned highlights:
- `hotpot_longctx_seed7/8/9/10`: mostly quick `+0.00pp` or strong negative, med skipped.
- `protein_ss_seed11/12/13/14/15`: quick mostly negative, med skipped.
- multiple fresh `hotpot_seed14/15/16/17/18/19/20` failed promote gate under strict `+0.80pp`.

Next round plan:
1. run `round137-144` immediately, prioritizing `arc_mc` transfer on new seeds.
2. keep broad 70/30 mix with limited anchors and strict quick prune/promo gating.

---

## Round151-158 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed13_discovery`: `scalar_l8_train8e5` med **`+20.83pp`**
- `protein_ss_seed25_discovery`: `scalar_l8_train8e5` med **`+6.04pp`**
- `hotpot_seed26_discovery`: `scalar_l8_train8e5` med **`+4.20pp`**
- `protein_ss_seed24_discovery`: `scalar_l8_train8e5` med **`+3.92pp`**
- `mbpp_longctx_seed4_repair`: `scalar_l8_train1e4` med **`+3.00pp`**
- `mbpp_seed18_headprobe`: `scalar_l8_sched_cos` med **`+2.95pp`**

Failed/pruned highlights:
- `punc_seed12/13_anchor` mostly quick negative or near-flat, med often skipped.
- `mbpp_seed17_headprobe` underperformed at quick stage; no med promotion.
- `arc_mc_seed11/12` mostly flat around baseline; `seed13` produced major jump.

Next round plan:
1. finish `round159-160` and close queue153-160.
2. continue automatically into queue161-168 with same prune/promote gates.

---

## Round159-162 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed14_discovery`: `scalar_l8_train1e4` med **`+8.33pp`**
- `mbpp_seed20_headprobe`: `head_l10_strong` med **`+2.87pp`**
- `protein_ss_seed27_discovery`: `scalar_l8_train1e4` med **`+1.95pp`**
- `punc_seed14_anchor`: `head_l8` med **`+0.17pp`**

Failed/pruned highlights:
- `hotpot_seed27_discovery`: best quick `-0.26pp`, below promote gate `+0.80pp`; med skipped.
- `hotpot_seed27_discovery`: `scalar_l8_train1e4` (`-2.94pp`) and `head_l8` (`-3.08pp`) quick-pruned.
- `mbpp_seed21_headprobe`: quick winner `scalar_l8_train8e5` promoted (`+0.92pp`) but med regressed **`-0.67pp`** vs baseline; marked strong negative and cooldown to round170.
- `arc_mc_seed15_discovery`: quick uplift was positive but med landed flat (`+0.00pp`), no net gain.

Next round plan:
1. continue queue `round161-168` to completion and keep strict quick prune/promote policy.
2. auto-chain queue `round169-176` then `round177-184` for nonstop broad discovery.

---

## Round163 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed16_discovery`: `scalar_l8_sched_cos` med **`+16.67pp`**
- `protein_ss_seed28_discovery`: `scalar_l8_train1e4` med **`+1.40pp`**

Failed/pruned:
- no quick-prune on this round; both tasks cleared promote gates and produced positive med gains.

Next round plan:
1. continue `round164-168` and keep quick prune (`-0.50pp`) / promote (`+0.80pp`) gates unchanged.
2. keep nonstop chain active through `round169-176`, `round177-184`, and prepared `round185-192`.

---

## Round164 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `punc_seed15_anchor`: best quick `head_l8` **`+0.07pp`** (below promote gate; med skipped)
- `mbpp_seed22_anchor`: quick winner `head_l10_strong` **`+2.15pp`**, but med **`-0.58pp`** vs baseline

Failed/pruned:
- `mbpp_seed22_anchor`: `scalar_l8_sched_cos` quick-pruned (`-0.52pp < -0.50pp`).
- `mbpp_seed22_anchor`: head branch showed quick-overfit pattern (quick positive, med negative).
- `punc_seed15_anchor`: all candidates clustered near baseline; no candidate reached `+0.80pp` promote gate.

Next round plan:
1. continue `round165-168` with emphasis on discovery tasks (`arc_mc`, `hotpot`, `protein_ss`) over weak anchors.
2. keep chained queues active: `169-176` -> `177-184` -> `185-192` without manual restart.

---

## Round165 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `hotpot_seed28_discovery`: `scalar_l8_train8e5` med **`+0.57pp`**
- `arc_mc_seed17_discovery`: best quick `-8.33pp` (all FS candidates tied negative)

Failed/pruned:
- `arc_mc_seed17_discovery`: `scalar_l8_train1e4`, `scalar_l8_sched_cos`, `scalar_l8_train8e5` all quick-pruned (`-8.33pp` each), med skipped.
- `hotpot_seed28_discovery`: `head_l8` quick-pruned (`-1.30pp`).
- `hotpot_seed28_discovery`: med gain exists but magnitude is small (`+0.57pp`) and below strong-promote threshold.

Next round plan:
1. continue `round166-168` and prioritize discovery tasks with stronger historical hit-rate (`arc_mc`, `protein_ss`).
2. keep nonstop auto-chain through `169-176`, `177-184`, `185-192`, `193-200`.

---

## Round166-176 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed18_discovery`: `scalar_l8_train1e4` med **`+8.33pp`**
- `arc_mc_seed20_discovery`: `scalar_l8_train1e4` med **`+8.33pp`**
- `arc_mc_seed19_discovery`: `scalar_l8_train8e5` med **`+4.17pp`**
- `protein_ss_seed33_discovery`: `scalar_l8_train1e4` med **`+1.42pp`**
- `punc_seed18_anchor`: `head_l8` med **`+1.22pp`**

Failed/pruned highlights:
- `arc_mc_seed17_discovery`: all FS candidates quick `-8.33pp`, med skipped.
- `arc_mc_seed22_discovery`: `scalar_l8_train1e4` `-8.33pp`, `scalar_l8_train8e5` `-16.67pp`, med skipped.
- `protein_ss_seed34_discovery`: all candidates quick negative (`-3.43pp` to `-3.53pp`), med skipped.
- `mbpp_seed25_anchor`: quick winner (`+1.40pp`) but med large regression **`-4.68pp`**.
- `mbpp_seed23_headprobe` and `mbpp_seed26_headprobe`: quick positive but med regressions (`-1.32pp`, `-1.50pp`).

Next round plan:
1. continue active queue `177-184` and keep strict quick prune (`-0.50pp`) and promote (`+0.80pp`) gates.
2. maintain nonstop chain through `185-192`, `193-200`, `201-208` and extend farther as needed.

---

## Round177 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed23_discovery`: best quick `scalar_l8_sched_cos` **`+0.00pp`**
- `protein_ss_seed35_discovery`: best quick `scalar_l8_train8e5` **`-0.47pp`**

Failed/pruned:
- `arc_mc_seed23_discovery`: `scalar_l8_train1e4` quick-pruned (`-8.33pp`), remaining candidates did not pass promote gate.
- `protein_ss_seed35_discovery`: `head_l8` and `scalar_l8_train1e4` quick-pruned (`-2.24pp`, `-3.13pp`).
- both tasks failed `quick >= +0.80pp` promote gate, so med stage skipped for this round.

Next round plan:
1. continue `round178-184` and watch whether `mbpp/hotpot` can convert quick gains into med-positive gains.
2. keep nonstop chain active through `185-192`, `193-200`, `201-208`, `209-216`.

---

## Round178 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `mbpp_seed27_headprobe`: best quick `head_l10_strong` **`+0.04pp`** (med skipped)
- `hotpot_seed31_discovery`: quick best `scalar_l8_train8e5` **`+1.40pp`**, med **`-0.55pp`**

Failed/pruned:
- `mbpp_seed27_headprobe`: `scalar_l8_sched_cos` and `scalar_l8_train8e5` quick-pruned (`-0.77pp`, `-0.81pp`).
- `mbpp_seed27_headprobe`: no candidate reached promote gate (`+0.80pp`), med skipped.
- `hotpot_seed31_discovery`: `head_l8` quick-pruned (`-2.80pp`); med stage showed regression despite quick gain.

Next round plan:
1. continue `round179-184`, prioritize `arc_mc/protein_ss` where med-positive hit rate is higher.
2. maintain nonstop chain through `185-192`, `193-200`, `201-208`, `209-216`, `217-224`, `225-232`, `233-240`.

---

## Round179 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed24_discovery`: `scalar_l8_train1e4` med **`+0.00pp`**
- `protein_ss_seed36_discovery`: best quick **`+0.00pp`** (no promote)

Failed/pruned:
- no strong quick-prune failures in this round, but both tasks lacked meaningful uplift.
- `arc_mc_seed24_discovery`: quick gains (`+4.17pp`) did not transfer to med (flat at `+0.00pp`).
- `protein_ss_seed36_discovery`: all candidates tied baseline at quick stage; med skipped by promote gate.

Next round plan:
1. continue `round180-184`, watch anchor tasks for regression while preserving discovery throughput.
2. keep nonstop chain active through `185-192`, `193-200`, `201-208`, `209-216`, `217-224`, `225-232`, `233-240`.

---

## Round180-207 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed31_discovery`: `scalar_l8_sched_cos` med **`+8.33pp`**
- `arc_mc_seed32_discovery`: `scalar_l8_sched_cos` med **`+8.33pp`**
- `arc_mc_seed28_discovery`: `scalar_l8_train8e5` med **`+4.17pp`**
- `protein_ss_seed45_discovery`: `scalar_l8_train1e4` med **`+3.58pp`**
- `mbpp_seed37_anchor`: `scalar_l8_sched_cos` med **`+2.55pp`**
- `protein_ss_seed48_discovery`: `scalar_l8_train8e5` med **`+2.46pp`**

Failed/pruned highlights:
- `arc_mc` remains high-variance across seeds: strong positive rounds mixed with hard quick collapses (`-8.33pp` / `-16.67pp`) and med skips.
- `hotpot` and `mbpp_longctx` often show quick positives but med regressions (for example `round190` med `-1.59pp`).
- `round199-202` mostly failed promote gate and were med-skipped due to weak quick deltas.
- anchor rounds (`mbpp`/`punc`) are frequently near-flat and can consume budget without reliable gains.

Next round plan:
1. finish active queue `201-208` (`round208` in progress) and auto-chain into `209-216`.
2. keep nonstop chain active through `217-224`, `225-232`, and `233-240`; prioritize `arc_mc/protein_ss` lines with proven med-positive transfer.

---

## Round208 - fastloop queue (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed38_discovery`: best quick `scalar_l8_train8e5` **`+0.00pp`** (med skipped)
- `protein_ss_seed50_discovery`: best quick `head_l8` **`+0.00pp`** (med skipped)

Failed/pruned:
- `arc_mc_seed38_discovery`: `scalar_l8_train1e4` and `scalar_l8_sched_cos` quick-pruned (`-8.33pp`, `-4.17pp`).
- both tasks failed promote gate (`+0.80pp`) with flat quick deltas, resulting in no med stage.

Next round plan:
1. continue active queue `209-216` with focus on converting quick gains into med-positive gains.
2. keep nonstop chain ready through `217-224`, `225-232`, `233-240`.

---

## Round209-214 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `arc_mc_seed39_discovery`: `scalar_l8_train1e4` med **`+8.33pp`**
- `mbpp_longctx_seed12_repair`: `scalar_l8_train8e5` med **`+4.37pp`**
- `arc_mc_seed40_discovery`: `scalar_l8_sched_cos` med **`+4.17pp`**
- `arc_mc_seed41_discovery`: `scalar_l8_sched_cos` med **`+4.17pp`**
- `hotpot_seed39_discovery`: `scalar_l8_train1e4` med **`+3.07pp`**

Failed/pruned highlights:
- `hotpot_seed40_discovery`: all candidates quick negative (`-1.52pp` to `-3.20pp`), med skipped.
- `mbpp_seed39_headprobe`: quick uplift did not transfer; med `-1.23pp`.
- `mbpp_seed40_anchor` / `punc_seed27_anchor`: below promote gate (`+0.37pp`, `+0.64pp`), med skipped.
- `protein_ss_seed51/53`: near-flat quick, med skipped.

Next round plan:
1. finish active queue `209-216` (`round215+` in progress), then auto-chain into `217-224`.
2. keep nonstop chain active through `225-232`, `233-240`, and prepared extension `241-248`.

---

## Round215-227 - fastloop queues (completed 2026-02-27)

Best config vs baseline:
- `mbpp_longctx_seed13_repair`: `scalar_l8_train8e5` med **`+10.00pp`**
- `protein_ss_seed60_discovery`: `head_l8` med **`+5.42pp`**
- `protein_ss_seed56_discovery`: `scalar_l8_train1e4` med **`+4.75pp`**
- `mbpp_seed45_headprobe`: `head_l10_strong` med **`+4.64pp`**
- `arc_mc_seed45_discovery`: `scalar_l8_train1e4` med **`+4.17pp`**
- `arc_mc_seed47_discovery`: `scalar_l8_sched_cos` med **`+4.17pp`**

Failed/pruned highlights:
- several anchor rounds (`215`, `217`, `218`, `220`) failed promote gate and were med-skipped.
- `hotpot` lines still show inconsistent med transfer; many runs stop at quick stage.
- `arc_mc` remains high-variance by seed, though positive med transfer continues to appear regularly.

Next round plan:
1. continue active queue `225-232` (`round228` in progress), then auto-chain into `233-240`.
2. keep nonstop chain armed through `241-248`; maintain same quick prune/promote gates for throughput.

---

## Chain Extension (prepared 2026-02-27)

Best config vs baseline:
- N/A (orchestration-only update)

Failed/pruned:
- none; this step only extends queue coverage for nonstop execution.

Next round plan:
1. run prepared fastloop queues continuously through `round368` (`249-256` ... `361-368`) after existing chain completes.
2. keep per-round sync cadence unchanged: remote runs -> local snapshot -> README/rolling log -> backup branch push.
