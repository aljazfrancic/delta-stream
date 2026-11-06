# Paper

LaTeX source of
**"How Residual-Stream Updates Travel Through Depth: A Pre-Registered Study of
Cross-Layer Dynamics in Transformer Factual Recall"** (v1.0, 6 November 2026).
The built PDF is committed as [`paper.pdf`](paper.pdf); for how to cite it, see the
[root README](../README.md#citation).

## Build

```bash
cd paper
latexmk -pdf paper.tex     # -> paper.pdf
```

That is the whole build. The committed [`.latexmkrc`](.latexmkrc) adds
`-shell-escape` (required by the `svg` package, which converts the figures via
Inkscape at build time) and wires bibtex, so no extra flags are needed.

Clean up build artefacts (keeps the PDF: use `-C` to remove it too):

```bash
latexmk -c
```

## Prerequisites

TeX Live with `latexmk`, the `svg` package, and Inkscape (the `svg` package
shells out to it). On Ubuntu/Debian:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-science latexmk inkscape
```

## Layout

```
paper.tex            main file: preamble, macros, \input of all sections
.latexmkrc           build config (shell-escape, bibtex, clean rules)
references.bib       bibliography (natbib, plainnat author-year)
sections/            one file per section, in the pre-registered outline order
  00-abstract.tex      09-conclusion.tex
  01-introduction.tex  appendix.tex (pre-registration & deviations, operating
  ...                  points, stats details, cross-model inventory, repro)
svg-inkscape/        Inkscape conversion cache (generated; git-ignored)
```

The section skeleton follows the pre-registered outline in
[docs/implementation-plan/phase-4-evaluation.md](../docs/implementation-plan/phase-4-evaluation.md)
(Step 4) exactly: Introduction · Related work · Phase-1 findings · Phase-2
findings · The decomposition · Evaluation · Limitations · Discussion ·
Conclusion.

## Figures

Figures are **not** stored here. The paper includes the curated SVGs from
[`../figures/`](../figures/) by relative path (`\svgpath{{../figures/}}` in
`paper.tex`), so the build must run from inside `paper/`. All 16 figures
regenerate byte-identically from the tracked run tables:

```bash
uv run python scripts/make_paper_figures.py   # from the repo root
```

Note on numbering: the SVG file names (`fig01`…`fig16`) are stable pipeline
IDs and do **not** match the figure numbers LaTeX assigns in the PDF (floats
are numbered by order of appearance; e.g. `fig14`/`fig15` render as Figures
4–5). Match files to figures by caption/`\label`, not by number.

## Numbers

Every quantitative claim traces to a run memo under [`../runs/`](../runs/)
(each memo carries its reproduce command) and to the tracker
[progress.md](../progress.md). The operating-points table in the appendix is
the map from quoted numbers to protocols.
