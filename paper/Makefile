# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build the CliffordClock methods-and-validation paper (WP15 draft).
#
#   make figures   -- regenerate every PDF figure and \input-able .tex
#                      value/table file under generated/ and figures/,
#                      by calling the REAL cliffordclock pipeline and
#                      benchmarks/ code (paper/figures/make_figures.py).
#   make pdf       -- build main.pdf, trying tectonic, then latexmk,
#                      then pdflatex+bibtex, in that order (first
#                      available engine wins; see PDF_ENGINE_USED below
#                      after a build for which one ran).
#   make all       -- figures, then pdf.
#   make clean     -- remove build artifacts (not figures/generated/,
#                      since those are committed pipeline outputs, only
#                      LaTeX intermediates and the PDF itself).
#   make distclean -- also remove figures/*.pdf and generated/*.tex
#                      (forces a full `make all` regeneration).
#
# Uses the repository .venv's python for the figures target (falls back
# to `python3` on PATH if .venv is not present -- e.g. a fresh checkout
# before `python -m venv .venv && pip install -e .[dev,notebooks]`).

REPO_ROOT := $(abspath ..)
VENV_PYTHON := $(REPO_ROOT)/.venv/bin/python
PYTHON := $(shell test -x $(VENV_PYTHON) && echo $(VENV_PYTHON) || echo python3)

.PHONY: all figures pdf clean distclean

all: figures pdf

figures:
	$(PYTHON) figures/make_figures.py

# Try tectonic first (self-contained, no separate bibtex step, no local
# TeX Live install needed), then latexmk (drives pdflatex+bibtex to a
# fixed point automatically), then a manual pdflatex/bibtex/pdflatex/pdflatex
# sequence as a last resort. Whichever succeeds first wins; this target
# never tries more than one engine per invocation once one succeeds.
pdf:
	@if command -v tectonic >/dev/null 2>&1; then \
		echo "Building with tectonic..."; \
		tectonic main.tex && echo "PDF_ENGINE_USED=tectonic"; \
	elif command -v latexmk >/dev/null 2>&1; then \
		echo "Building with latexmk..."; \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && echo "PDF_ENGINE_USED=latexmk"; \
	elif command -v pdflatex >/dev/null 2>&1; then \
		echo "Building with pdflatex+bibtex (manual sequence)..."; \
		pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex && \
		echo "PDF_ENGINE_USED=pdflatex+bibtex"; \
	else \
		echo "No LaTeX engine found (tried tectonic, latexmk, pdflatex)." >&2; \
		exit 1; \
	fi

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.fls \
	      main.fdb_latexmk main.synctex.gz main.pdf mainNotes.bib

distclean: clean
	rm -f figures/*.pdf
	rm -f generated/*.tex
