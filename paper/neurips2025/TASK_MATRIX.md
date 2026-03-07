# Task Matrix

This matrix maps the paper's post-training rows to the actual trainer scripts and probe semantics used in the repository.

All reported post-training numbers are teacher-forced token-accuracy deltas against same-task baselines.

| Paper row | Actual probe | Dataset / config | Trainer | Metric | Representative queue |
|---|---|---|---|---|---|
| `protein_ss_spot` | queried-residue secondary-structure spot labeling | `lamm-mit/protein_secondary_structure_from_PDB` | `train_protein_ss_spot_sft.py` | token accuracy on queried residue labels | `_search_queue_round805_808_breadth_roi.json` |
| `hotpot_text_restore` | clean-text restoration from Hotpot-derived contexts | `hotpot_qa` / `distractor` | `train_punc_restore_sft.py` | token accuracy on restored clean text | `_search_queue_round805_808_breadth_roi.json` |
| `mbpp_longctx_probe` | long-context MBPP code-completion probe | `mbpp` | `train_mbpp_longctx_sft.py` | token accuracy on solution code span | `_search_queue_round783_790_realtask_exploit_v3.json` |
| `arc_mc_probe` | ARC multiple-choice answer-token probe | `ai2_arc` / `ARC-Challenge` | `train_arc_mc_sft.py` | token accuracy on answer option label | `_search_queue_round783_790_realtask_exploit_v3.json`, `_search_queue_round799_802_final_confirm.json` |
| `squad_text_restore` | clean-text restoration from SQuAD-derived passages | `squad` | `train_punc_restore_sft.py` | token accuracy on restored clean text | `_search_queue_round805_808_breadth_roi.json` |
| `punc_restore` | generic punctuation/case restoration proxy | mixed text corpora across earlier rounds | `train_punc_restore_sft.py` | token accuracy on restored clean text | earlier `round24/33/35/36/39/41/42/74/75` search scripts |

Notes:

- `hotpot_longctx` is a separate QA-style long-context answer-span probe implemented by `train_hotpot_longctx_sft.py`, but it is not a main-table positive family in the paper.
- `hotpot_text_restore` and `squad_text_restore` should not be read as benchmark EM/F1 results.
- `mbpp_longctx_probe` and `arc_mc_probe` should not be read as pass@k or benchmark leaderboard claims.
