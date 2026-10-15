# Rail-V paper

LaTeX draft of the Rail-V paper (IEEE conference format, `IEEEtran`); `rail-v-paper.pdf` is
the compiled copy. It builds on
Overleaf as is: upload `main.tex`, `refs.bib` and the `figures/` folder.

## Adding the figures

`vibration/paper_results.py` (notebook step 10) writes the figures to
`rail-v-runs/paper/figures/` and zips them into `paper_results.zip`. Copy these into
`figures/` here:

    scores.pdf  confusion.pdf  training_curves.pdf  multitask_runs.pdf

They are already here for the current draft. If a file is missing, the PDF shows a labelled
box in its place.

## Building locally

    pdflatex main && bibtex main && pdflatex main && pdflatex main
