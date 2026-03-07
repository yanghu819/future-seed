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
