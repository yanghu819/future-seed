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
