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

## Files

- `main.tex`: submission-ready anonymous paper source
- `main_preprint.tex`: preprint-ready paper source
- `appendix.tex`: appendices and reproducibility notes
- `checklist.tex`: filled NeurIPS 2025 checklist
- `references.bib`: bibliography
- `render_tables.py`: generates LaTeX tables from `data/metrics.json`
- `data/metrics.json`: curated metrics used by the paper draft
- `tables/`: generated LaTeX tables
- `REFERENCE_AUDIT.md`: official-source audit for all citations used in `main.tex`
- `TASK_MATRIX.md`: mapping from paper rows to trainer scripts, datasets, and probe metrics
- `METRICS_PROVENANCE.md`: provenance notes for the curated paper-side metrics snapshot
- `LOCAL_REPRO.md`: exact boundary of what this local snapshot can and cannot reproduce

## Notes

- The paper is written as an anonymous submission draft.
- The strongest headline claim is synthetic mechanism evidence plus repeatable `protein_ss_spot` gains.
- The supporting real-task signals are `hotpot_text_restore`, `squad_text_restore`, and `punc_restore`.
- The draft intentionally keeps `mbpp_longctx_probe` and `arc_mc_probe` as mixed evidence, not stable confirmation.
- The paper tables are regenerated from committed `data/metrics.json` and do not depend on live experiment logs at build time.
- For the supplementary layout, see `ARTIFACT_GUIDE.md`.
- For reproduction boundaries and local sanity commands, see `LOCAL_REPRO.md`.
- `references.bib` is manually audited against official venue or archive pages (`proceedings.neurips.cc`, `aclanthology.org`, `jmlr.org`, `pnas.org`, and `arxiv.org`).
- The shipped post-training round artifacts in this snapshot are `783-788` and `799-808`.
- The default post-training checkpoint path is referenced in the scripts but the checkpoint blob is not included in this anonymous snapshot.
