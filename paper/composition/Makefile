# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Build the "Composing the error budget of an optical clock" companion
# paper (paper/composition/). Patterned on paper/Makefile.
#
#   make pdf   -- build main.pdf, trying tectonic, then latexmk, then
#                 pdflatex+bibtex, in that order (first available
#                 engine wins; see PDF_ENGINE_USED below after a build
#                 for which one ran).
#   make all   -- same as `make pdf` (this paper has no generated
#                 figures/value files of its own: every number in it
#                 is copied by hand from the committed companion paper
#                 and repository docs, never recomputed by this
#                 paper's own build; see main.tex's header comment).
#   make clean -- remove build artifacts (main.pdf included; this
#                 paper does not commit its own PDF the way paper/
#                 does, see the repository's build report for that
#                 decision).
#
# No figures target: this companion paper has no paper/figures/*.py
# pipeline of its own to run.

.PHONY: all pdf clean

all: pdf

# Try tectonic first (self-contained, no separate bibtex step, no
# local TeX Live install needed), then latexmk (drives pdflatex+bibtex
# to a fixed point automatically), then a manual
# pdflatex/bibtex/pdflatex/pdflatex sequence as a last resort.
# Whichever succeeds first wins; this target never tries more than one
# engine per invocation once one succeeds.
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
