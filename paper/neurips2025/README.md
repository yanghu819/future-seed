# NeurIPS 2025 Paper Package

This directory contains the anonymous NeurIPS main-track draft for Future-Seed, backed by a local archival artifact snapshot.

## Template Source

Official NeurIPS 2025 style bundle:
- `https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip`

Included here:
- `neurips_2025.sty`
- `neurips_2025.tex`
- `neurips_2025.pdf`

## Build

This package uses `tectonic`.

```bash
cd paper/neurips2025
./build.sh submission
./build.sh preprint
```

Outputs:
- `build/neurips2025-submission.pdf`
- `build/neurips2025-preprint.pdf`

The current submission build keeps the main content safely below the NeurIPS nine-page content limit; references, appendix, and checklist follow after the main text.

## Full Submission Check

```bash
bash check_repo_health.sh
```

That command runs:
- markdown link validation
- fastdiscover orchestrator self-test
- non-mutating fastdiscover dry-run on the shipped breadth queue
- paper metrics verification
- paper reference audit verification
- submission PDF build
- NeurIPS page-budget verification
- source anonymity verification
- anonymous supplementary ZIP packaging and ZIP-level anonymity verification

## Supplementary Packaging

```bash
cd paper/neurips2025
python3 package_submission_bundle.py
```

Outputs:
- `dist/future-seed-neurips2025-supplementary.zip`
- `dist/SHA256SUMS.txt`
- `SUBMISSION_READY.md`

Default checkpoint policy in the current snapshot:
- `omit`
- no checkpoint blob is bundled unless you explicitly provide one with `--checkpoint-path` or point to an anonymous external URL with `--checkpoint-url`
- because no anonymous checkpoint is currently shipped, reproducibility and open-access checklist items remain conservative

The package script enforces the NeurIPS 2025 supplementary ZIP policy boundary documented at:
- `https://neurips.cc/Conferences/2025/PaperInformation/CodeSubmissionPolicy`

## Files

Core paper files:
- `main.tex`: submission-ready anonymous paper source
- `main_preprint.tex`: preprint-ready paper source
- `appendix.tex`: appendices and reproducibility notes
- `checklist.tex`: filled NeurIPS 2025 checklist
- `references.bib`: bibliography

Paper-side utilities:
- `render_tables.py`: generates LaTeX tables from `data/metrics.json`
- `verify_metrics_snapshot.py`: verifies that the historical archive summary in `data/metrics.json` matches the committed README scoreboard and that the shipped closure/breadth subset recomputes from raw round artifacts
- `verify_reference_audit.py`: verifies that every citation used in `main.tex` has a detailed audit entry, an approved official-source URL, and a matching `url` field in `references.bib`
- `verify_submission_layout.py`: verifies total pages, content-page budget, and bibliography/appendix/checklist boundaries from the built logs
- `verify_anonymity_snapshot.py`: scans the curated source snapshot and supplementary ZIP for anonymity leaks
- `package_submission_bundle.py`: assembles the anonymous supplementary ZIP
- `artifact_manifest.py`: single source of truth for the supplementary file list

Paper-side documentation:
- `TASK_MATRIX.md`: mapping from paper rows to trainer scripts, datasets, and probe metrics
- `METRICS_PROVENANCE.md`: provenance notes for the curated paper-side metrics snapshot
- `LOCAL_REPRO.md`: exact boundary of what this local snapshot can and cannot reproduce
- `ARTIFACT_GUIDE.md`: high-level description of the anonymous artifact boundary
- `SUPPLEMENTARY_MANIFEST.md`: human-readable manifest for the anonymous supplementary snapshot
- `COMPUTE_ACCOUNTING.md`: compute and storage boundary for the paper-facing snapshot
- `ASSET_LICENSE_MATRIX.md`: paper-relevant asset and license audit matrix
- `REPRO_MATRIX.md`: per-claim reproducibility status matrix
- `REFERENCE_AUDIT.md`: official-source audit for all citations used in `main.tex`
- `SUBMISSION_READY.md`: generated upload checklist once the package script runs

## Notes

- The paper is written as an anonymous submission draft.
- The strongest headline claim is synthetic mechanism evidence plus repeatable `protein_ss_spot` gains.
- The clearest released secondary positive is `hotpot_text_restore`.
- `squad_text_restore` and `punc_restore` should be read as smaller historical positive pockets in the broader archive.
- The draft intentionally keeps `mbpp_longctx_probe` and `arc_mc_probe` as mixed evidence, not stable confirmation.
- The paper tables are regenerated from committed `data/metrics.json` and do not depend on live experiment logs at build time.
- `python3 verify_metrics_snapshot.py` checks internal consistency for the committed historical archive summary and also recomputes the shipped closure/breadth subset from raw round records.
- `python3 verify_reference_audit.py` checks that every cited paper has an explicit paper-side usage note, an official-source URL in the audit file, and a matching official URL in `references.bib`.
- `python3 verify_submission_layout.py` checks that the submission/preprint builds remain within the NeurIPS content-page budget.
- `python3 verify_anonymity_snapshot.py` checks that the curated anonymous snapshot does not leak local paths, legacy remote commands, or owner identifiers.
- The shipped post-training round artifacts in this snapshot are `783-788` and `799-808`; the paper-side family counts remain a curated internal archive summary rather than a raw recomputation from that subset.
- The default post-training checkpoint path is referenced in the scripts but the checkpoint blob is not included in this anonymous snapshot.
