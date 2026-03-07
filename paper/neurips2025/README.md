# NeurIPS 2025 Paper Package

This directory contains the anonymous NeurIPS main-track draft for Future-Seed.

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

- `main.tex`: paper body with build-time placeholders for submission vs preprint style
- `appendix.tex`: appendices and reproducibility notes
- `checklist.tex`: filled NeurIPS 2025 checklist
- `references.bib`: bibliography
- `render_tables.py`: generates LaTeX tables from `data/metrics.json`
- `data/metrics.json`: curated metrics used by the paper draft
- `tables/`: generated LaTeX tables

## Notes

- The paper is written as an anonymous submission draft.
- The strongest headline claim is synthetic mechanism evidence plus repeatable `protein_ss` gains.
- The draft intentionally keeps `mbpp_longctx` and `arc_mc` as mixed evidence, not stable confirmation.
- The paper tables are regenerated from committed `data/metrics.json` and do not depend on live experiment logs at build time.
- For the supplementary layout, see `ARTIFACT_GUIDE.md`.
- `references.bib` is manually audited against official venue or archive pages (`proceedings.neurips.cc`, `aclanthology.org`, `jmlr.org`, `pnas.org`, and `arxiv.org`).
