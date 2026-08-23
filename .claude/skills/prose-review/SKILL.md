---
name: prose-review
description: Review user-facing prose (notebooks, docs, README, benchmark md) against the project's writing standards. Run before any prose-bearing commit is declared review-clean; a reviewer agent follows this procedure and reports blockers. The mechanical scanner catches only a fraction of these rules; the read-through pass is the review.
---

# Prose review

Every reader-facing sentence in this project is judged by one test: a busy
experimentalist scanning the page gets the point of each section, paragraph,
and table before reading the details, and nothing on the page talks about the
page. This skill turns the owner's accumulated review feedback into a
procedure a reviewer agent runs on any prose-bearing change.

## Scope

Markdown cells of notebooks, docs/*.md, README.md, benchmarks/**/*.md,
docstrings that render in reports, error messages, and figure titles or
captions. Code comments follow the banned-pattern rules but are exempt from
the orientation rules.

## Procedure

1. Run the mechanical layer first, from the repo root:
   `python tools/release_checks.py --only prose-scan` plus a grep of the
   changed files for the banned patterns below. Zero new findings is the
   entry bar, never the verdict: every round of review in this project's
   history found blockers the scanner cannot see.
2. Read every changed or new prose block start to finish, as a first-time
   reader, applying the checklist below. Judge each paragraph by what a
   scanning reader takes away, and quote every violation verbatim in the
   report.
3. Report PASS or FAIL with blockers. Each blocker gives the file, the cell
   or line, the quoted text, which rule it breaks, and a suggested rewrite.
   Prose blockers are real blockers: a change with a prose FAIL is one fix
   loop from done, never done.

## Banned outright (blocker on sight)

- Em dashes, in any prose, comment, docstring, or error message. Numeric
  ranges in LaTeX are the only exception.
- The words honest, honesty, honestly, in any user-facing text.
- "rather than" and "instead of" constructions. State what the thing is.
- "not X but Y", "not X; it is Y", "does not X, but Y" pivots, including
  spread across two sentences. If X genuinely needs ruling out, rule it out
  in its own sentence with a reason.
- Sentences that lead with what something does not do, is not, or cannot do
  when describing a capability. Scope limits are stated after the capability,
  as facts with reasons, in their own sentences.
- Trailing "not a X" appositive tails ("a validation, not a guarantee").
  Either the distinction deserves its own sentence or it goes.

## Meta-commentary and throat-clearing (blocker on sight)

A sentence whose subject is the document instead of the physics is noise.
This includes announcing what is about to be said, labeling the act of
saying it, and summarizing what was just said.

- "One frontier is worth naming directly, since it sits closest to this
  notebook's subject." The fix: delete the sentence and name the frontier.
- "Label this section plainly before anything else: the geometry below is
  illustrative..." The fix: "The chamber drawn below uses the thesis's
  published dimensions and orientation; the remaining distances are
  estimates, and the text says so where each enters."
- Other shapes of the same disease: "This section shows/demonstrates...",
  "The following...", "It is worth noting...", "the key point is",
  "stated plainly", "to be clear", a closing paragraph that restates the
  section.

Orientation is the opposite of meta-commentary and is REQUIRED: before any
dense paragraph, table, or derivation, one plain sentence states the point
directly, in the physics, so the reader knows why the details matter before
meeting them. Orientation states the conclusion; meta-commentary announces
that a conclusion is coming.

## Word salad (blocker when found)

A sentence carrying three ideas, or a paragraph where every sentence has
subordinate clauses stacked on qualifications, forces the reader to study
each sentence to extract any meaning. The fix is one idea per sentence,
short declaratives, with each technical term appearing next to its plain
meaning. Keep every number, hedge, and scope qualifier; redistribute them
into their own sentences. Paragraphs are flowing, 3 to 5 sentences, and
build on each other; no one-sentence drama paragraphs.

## Context before content (blocker when a reader would be lost)

Every table row, figure element, and named quantity gets introduced before
or immediately beside its first appearance: what it is, where it comes
from, and, when two rows agree or disagree, whether that outcome is
expected and why. A comparison whose agreement is expected by construction
must say so; unexplained agreement reads as magic and unexplained
disagreement reads as failure. Acronyms and institution names get one
plain-words introduction per document.

## Scannability

Enumerable content (what is validated, what a lab can do, known limits)
belongs in labeled bullet groups with a short narrative around them, never
buried in paragraph runs. Wide tables render as Markdown tables with
footnotes, never fixed-width text prints. Section titles are plain human
titles ("What an experimental team can use today"); titles shaped like
"What X does, and what it does not" or other symmetrical AI constructions
are blockers. The opening of a notebook leads with why the reader should
care, in the reader's own terms, before any project vocabulary.

## Machine-accent sweep

Flag and rewrite: hollow intensifiers (genuinely, truly, simply,
fundamentally, crucially, notably, actually), inflated vocabulary (delve,
leverage, robust, seamless, comprehensive, pivotal, showcase, landscape
figuratively), hedge stacks ("may potentially"), rule-of-three reflexes,
closing tics ("In conclusion", "Ultimately"), and uniform sentence rhythm.
Numbers in prose byte-match the computed outputs they describe; a prose
number that drifts from its computation is a correctness blocker, not a
style note.

## Calibration examples from past reviews

- Word salad, before: "A lab or systems integrator without PTB-grade
  thermal engineering reads this the same way any uncertainty budget line
  is read: sensor readings and their calibration uncertainties go in, a
  defensible BBR shift and band come out, with no requirement to first
  build a shielded enclosure or run a finite-element thermal model."
  After: three sentences, one claim each, ending "Sensor readings and
  their calibration uncertainties are the only required inputs; a shielded
  enclosure and a finite-element thermal model both stay optional
  refinements."
- Pivot, before: "That is not a weakness of the demonstration; it is a
  real, independent confirmation of section 2's headline point."
  After: "That behavior independently confirms section 2's headline
  point." (If the weakness reading needs addressing, address it in its own
  sentence.)
- Leading negation, before: "This engine does not walk a classical
  mm-scale trajectory through a field map." After: state what the engine
  does evaluate, then the scope sentence with its reason.
