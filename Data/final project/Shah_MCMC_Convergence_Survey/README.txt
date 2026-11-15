Exact Draws, Approximate Draws: A Survey of Convergence Theory for
Gibbs and Metropolis-within-Gibbs Samplers  --  LaTeX source

Compile (local TeX Live or Overleaf, pdfLaTeX):
    pdflatex survey.tex
    bibtex   survey
    pdflatex survey.tex
    pdflatex survey.tex
On Overleaf: upload this whole folder (keep figs/ as a subfolder); the
default pdfLaTeX compiler works as is.

Files
    survey.tex           main text (16 pages including references)
    refs.bib             bibliography (57 cited works)
    figs/                the two simulation figures (PDF)

Reproducing the figures (Python 3 with numpy, scipy, matplotlib):
    python3 run_iat.py      # autocorrelation times for d = 2, 4 and A = 4..32
    python3 run_iat64.py    # adds the A = 64 point for d = 2 (about 1.5 min)
    python3 make_figs.py    # writes figs/fig_gibbs_geometry.pdf, figs/fig_almost_exch.pdf
sim_ax.py contains the random-scan Gibbs sampler for de Finetti's prior and
the integrated-autocorrelation-time estimator; iat_results.json holds the
values used in the paper (IAT / (d A^2) stays near 0.34-0.39 as A grows,
consistent with the Theta(A^2) rate of Gerencser and Ottolini).
