# Future-Seed Noncausal Task Roadmap

This note is the repository's forward-looking task map for Future-Seed.
It is not a result table. It is a decision document for what to run next if the goal is a strong paper on real noncausal utility.

## Selection Rule

A task belongs on the main Future-Seed roadmap only if all five conditions hold.

1. The target token or span should depend on information that appears on its right or later in the prompt.
2. The task should have a benchmark end metric such as exact match, pass@1, execution success, or F1, not only teacher-forced token accuracy.
3. A necessity control should be possible by shuffling, deleting, or moving the future evidence.
4. The task should admit a distance split such as near versus far future evidence, or local versus global future anchors.
5. The task should be meaningful outside the paper itself: code completion, repository repair, long-context QA, biological labeling, or document restoration.

## Current Repository Signal

The current repository already suggests a clear ranking.

- strongest repeatable real-task family under the current single-GPU recipe: `protein_ss_spot`
- strongest clean mechanism story on a real task: `RepoBench char1`
- promising but not yet stable enough for headline claims: `mbpp_longctx_probe`
- high upside but high variance: `arc_mc_probe`
- useful support tasks but not final headline metrics: `hotpot_text_restore`, `squad_text_restore`, `punc_restore`
- low ROI under the current recipe: `wiki`, `protein_contact`, `hotpot_longctx`, generic prefix-only probes

## Priority Table

| Priority | Task family | Concrete benchmark or build | Why noncausal matters | End metric to report | Repository fit | 8xH100 | 1000 GPU |
|---|---|---|---|---|---|---|---|
| P0 | repo-level code infill | `RepoBench-C` / `RepoBench-P` symbol or span infill | missing code must agree with right-context code and later tagged anchors | span exact match, completion exact match, execution when possible | already partially implemented through `repobench_char1_diagnostics/` | yes | yes |
| P0 | cross-file code completion | `CrossCodeEval` repository completion | the missing code depends on later repository evidence and cross-file definitions | exact match, pass@1, retrieval-conditioned completion | new benchmark harness, same mechanism story | yes | yes |
| P0 | executable repo completion | `RepoExec` or equivalent execution-backed repo completion | later code and tests constrain the missing implementation | pass@1, execution success | new harness needed | yes | yes |
| P0 | executable function-level FIM | `HumanEval-FIM` and `MBPP-FIM` style builds | a masked middle span must match both prefix and suffix | pass@1, executable tests, exact span match | builder upgrade needed; current `mbpp_longctx` is the nearest line | yes | yes |
| P0 | protein residue labeling | `protein_ss_spot` | a residue label depends on both left and right sequence context | queried-position accuracy, macro-F1 if label balance matters | already active and strongest repeatable family | yes | yes |
| P1 | long-context code reasoning | `mbpp_longctx_probe` upgraded from token probe to executable eval | the answer code depends on long prompt details that appear earlier and later in the prompt scaffold | pass@1, execution success | existing trainer, metric upgrade needed | yes | yes |
| P1 | multiple-choice reasoning with future options | `ARC-Challenge` question-first and options-first variants | early prompt understanding should use answer options that appear later | accuracy on final option label | already active as `arc_mc_probe` | yes | yes |
| P1 | multihop long-context QA | `HotpotQA`, `2WikiMultihopQA`, `MuSiQue` in question-first order | question tokens must be interpreted using later supporting evidence | EM, F1, supporting-fact F1 if available | `Hotpot` is already partially explored; new builders needed for cleaner end metrics | yes | yes |
| P1 | long-document QA | `Qasper` or long-context QA slices from `LongBench` | answer generation depends on evidence distributed later in the document | EM, F1, answer exactness | new harness needed | maybe | yes |
| P1 | repository-level issue-to-patch | `SWE-bench Verified` issue-first, repo-after formatting | issue tokens should be reinterpreted after later repo context arrives | resolved rate, test pass rate | new large harness; only worth doing at higher budget | no | yes |
| P1 | document restoration | `hotpot_text_restore`, `squad_text_restore`, `punc_restore` | restoring text or punctuation depends on later tokens and sentence closure | exact restoration accuracy, edit accuracy | already present; useful support evidence | yes | yes |
| P2 | config and schema repair | OpenAPI ref repair, Docker Compose dependency repair, GitHub Actions `needs`, cross-file env wiring | masked keys or references are constrained by later fields and other files | exact match, file-validity, execution if available | product-style internal benchmark; good for appendix or artifact | yes | yes |
| P2 | SQL structure repair | join-key completion, group-by / having consistency, alias restoration | valid SQL often requires later clauses to constrain earlier spans | exact match, execution accuracy on held-out queries | internal benchmark, useful if grounded in real SQL corpora | maybe | yes |
| P2 | structured biomedical sequence repair | harder protein contact or masked pair relation prediction | pair labels depend on the whole sequence and distant residues | pair accuracy, F1, contact precision | current recipe weak on `protein_contact`; requires reformulation | no | yes |
| P2 | long-form editing and rewrite | controlled text editing where conclusion or constraints appear later | the correct middle rewrite depends on later target style or facts | exact edit success, factual consistency | new benchmark design needed | maybe | yes |

## Immediate Recommendation: 8xH100

If the budget is eight H100s, the paper should stay narrow and benchmark-first.

### Headline set

1. `RepoBench` repo-level symbol or span infill
2. executable `MBPP-FIM` / `HumanEval-FIM`
3. `protein_ss_spot`

### Secondary set

1. `ARC-Challenge` question-first and options-first
2. `HotpotQA` or `2WikiMultihopQA` question-first long-context QA
3. `punc_restore` or text-restore as support evidence only

### What not to spend on first

- generic `WikiText` or plain prefix language modeling
- `protein_contact` under the current recipe
- toy constraint tasks as headline claims
- unstable long-context code probes without executable metrics

### Why this is the right 8xH100 package

- it gives one strong code story, one strong biological sequence story, and one reasoning support line
- each main task admits shuffled-future and near-vs-far controls
- all headline tasks can be evaluated with end metrics rather than only token probes

## Scale Recommendation: 1000 GPU

At thousand-GPU scale, the strategy should change from post-training search to model-scale training plus benchmark evaluation.

### Train

1. baseline RWKV checkpoint continued at larger scale
2. `Future-Seed v2` as the minimum mechanism line
3. one selective collector or token-conditioned gate line only if it stays implementation-clean

### Evaluate

1. repo-level code infill: `RepoBench`, `CrossCodeEval`, `RepoExec`
2. function-level FIM: `HumanEval-FIM`, `MBPP-FIM`
3. long-context reasoning: `HotpotQA`, `2Wiki`, `MuSiQue`, `Qasper`, selected `LongBench` slices
4. biological labeling: `protein_ss_spot`, then harder residue or pair tasks only after the first line is stable
5. internal mechanism controls: shuffled future, near-vs-far, selective anchor collectors

### What a thousand-GPU paper should answer

1. Does Future-Seed still help after scaling, or is it a small-model phenomenon?
2. Which benchmarks improve only on infill or repair, and which also improve on repository or reasoning tasks?
3. Does the gain persist under benchmark end metrics rather than teacher-forced token accuracy?
4. Which future information is most useful: local suffix, repeated anchors, or broad document context?

## Minimal Publication Packages

### Publication package for 8xH100

- main: `RepoBench` plus executable `MBPP-FIM` or `HumanEval-FIM`
- second domain: `protein_ss_spot`
- support: one long-context reasoning task such as `ARC-Challenge` or `HotpotQA`
- controls: shuffled future, near-versus-far, one collector ablation

### Publication package for 1000 GPU

- model scale curve: at least three sizes
- code benchmark suite: `RepoBench`, `CrossCodeEval`, `RepoExec`, `HumanEval-FIM`, `MBPP-FIM`
- long-context reasoning suite: `HotpotQA`, `2Wiki`, `MuSiQue`, `Qasper`, selected `LongBench` slices
- non-language validation: `protein_ss_spot`
- mechanism appendix: shuffled future, distance split, collector and gate ablations

## Tasks To Deprioritize Under The Current Evidence

These are still useful as sanity or appendix material, but they should not absorb primary compute until a stronger recipe exists.

- plain `WikiText` prefix infill
- generic prefix-only probes without explicit future dependence
- `protein_contact` under the current trainer and objective
- long-context tasks that do not yet have end metrics
- toy tasks promoted as real-task headline evidence

## Repository Pointers

- current task map: [`TASK_INDEX.md`](TASK_INDEX.md)
- current main result summary: [`RESULTS.md`](RESULTS.md)
- real-task method diagnostic: [`rwkv-diff-future-seed/repobench_char1_diagnostics/README.md`](rwkv-diff-future-seed/repobench_char1_diagnostics/README.md)
- current paper entry: [`PAPER.md`](PAPER.md)
- post-training track status: [`posttrain_rwkv7/README.md`](posttrain_rwkv7/README.md)
