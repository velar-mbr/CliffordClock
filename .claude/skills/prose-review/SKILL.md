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
- Trailing negated appositive tails: ", not a X", ", not Y", ", never Z"
  hung on the end of a statement ("a validation, not a guarantee";
  "classified as arithmetic reproduction, not reproducibility"). No
  exception for tails that seem informative, and no appeal to existing
  documents that still carry the pattern. If the ruled-out alternative
  matters, it gets its own sentence stating what it is and why this
  case is not it: "classified as arithmetic reproduction. The stronger
  reproducibility class requires inputs measured independently of the
  published result, which this case does not have." A tacked-on half
  sentence is fine only when it adds positive content ("classified as
  arithmetic reproduction, a total-level check against published
  inputs"); a negation adds none. One sentence per idea worth having.

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

## Decorative adverbs and empty emphasis (blocker on sight)

Adverbs that add emphasis without changing the claim get deleted:
"straight from" (from), "reconstructed directly from" (reconstructed
from), "the field map itself" (the field map), "precisely/exactly"
when the number already carries the precision. Test: if deleting the
word changes nothing checkable, delete it. Prefer plain verbs over
reaching ones: gives or reports over supplies, uses over utilizes,
shows over reveals.

## Absolute claims (blocker unless cashable on the spot)

Words like nothing, zero, every, all, none, and never make a reader
stop and audit. An absolute survives only when the text beside it lets
the reader verify it: "zero free parameters" is legitimate exactly
where the sentence or table lists every input and its published
source, so the reader can count. "With nothing tuned" floating free is
a red flag; either name the complete input list in the same breath or
drop the claim.

## Unprompted defense (blocker on sight)

A sentence that rebuts an objection the text has not raised reads as
nervousness: "None of this is new physics" appearing without any
surrounding novelty claim defends against a thought the reader was
not having. Delete it, or state the positive fact it was guarding
("every formula here is standard, with its source cited") only where
the reader would genuinely ask.

## Opaque punchy labels (blocker when found)

A bold label or bullet heading must inform a cold reader, never just
sound decisive. "The Al+ reproduction" and "The per-mode cross-check"
name nothing a first-time reader can hold; "Reproducing the published
Al+ time-dilation total" and "Checking each motional mode against the
published row" do. The test: covered up to the label alone, does a
newcomer know what the item is about?

## Story before inventory (blocker when missing)

Before any bullet list or dense paragraph that enumerates results, one
or two plain sentences answer: what question is this settling, and why
does the reader care? An inventory of achievements ("adds X,
completed by Y, closes most Z, passes W") without that setup says many
things while telling no story, and the reader has no reason to slow
down for the details. Context, then question, then the details.

## Figurative language (blocker on sight)

Analogies, metaphors, and poetic framing distract in science writing:
"that is the picture these clocks live in, and this notebook walks
five capabilities against exactly that picture" reaches for an image
where the literal statement is shorter and clearer ("Five capabilities
build on that model:"). State what the thing is. A physical analogy
earns its place only when it carries a real correspondence the reader
computes with (an integrating sphere for a mirror-walled chamber);
decorative imagery never does.

## The clarity read (its own pass, not a side effect)

Correct prose can still be unreadable. After every rule above, read
each paragraph once more with a different question: could a strong
graduate student from outside this subfield follow it on first read,
without backtracking? A sentence that stacks a claim, its cause, a
qualifier on the cause, and a comparison into one breath fails this
even when every clause is true. The repair: one claim per sentence,
cause in its own sentence, and a plain-English guide sentence ahead of
any dense stretch so the reader knows what the details are about to
establish. Example of the failure, from a real review: "The
participation-corrected total above does not, because the two radial
STR modes carry the largest published per-mode magnitudes, exactly
where the closed form's mass-ratio-only approximation departs most
from the true, RF/DC-geometry-dependent radial eigenvector." The
repair: "The participation-corrected total misses, and for a specific
reason. The radial STR modes carry the largest published per-mode
values. They are also the modes where the mass-ratio formula is least
reliable, because the true radial eigenvector depends on trap geometry
the formula does not include."

## "Exactly" and "precisely" (blocker by default)

These two words almost always either inflate a point into a
superlative or glue an extra clause onto a sentence that was already
done. Delete them wherever they modify emphasis ("exactly where the
approximation departs most", "precisely the tool a lab needs"). The
survivors are technical uses that change a checkable claim: "exact" as
the antonym of approximate ("the exact Floquet solution"), and
"exactly" stating a mathematical identity the reader can verify
("participations sum to one exactly"). When in doubt, delete and
reread; the sentence is stronger.

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

## Stale status claims (blocker; check the WHOLE document)

A claim about what is open, pending, unmodeled, or future must be
checked against the entire document and the current repository state,
never only against the cells or paragraphs the change touched. Review
rounds scoped to a diff systematically miss these: a capability
completed in a later section falsifies an earlier "filed as an open
item" that nobody edited, and the reader who meets both loses trust in
every other claim. Sweep every reviewed document for open-item,
roadmap, not-yet, and future-work statements and verify each against
what the document itself, and the repo, now deliver.

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

## Antecedent-less demonstratives (blocker on sight)

"That", "this", "it", and "those" opening a sentence must resolve to a
named thing the reader can point to, not to the vague shape of the
previous sentence. "That addition is not a separate approximation
layered on top of the pivot picture above. It falls out of the same
product, kept to first order." leaves both "That" and "It" pointing at
an unnamed blend of what came before. The fix names the actual subject
in each sentence: "The pivot's own product, `prod(1+p_k)`, becomes that
row-by-row sum once it is expanded and truncated at first order." A
demonstrative immediately after the sentence that defines its referent
("...three kinds of term appear... these cross terms...") is fine; a
demonstrative standing in for a whole prior discussion is not.

## Announced precision (blocker on sight)

A phrase that asserts its own exactness reads as a claim the writer
felt pressure to reassure the reader about, and a reader who checks it
loses trust in every other claim on the page: "coordinated word for
word with the methods paper it accompanies" cannot possibly be checked
word for word by a reader, and is not literally true. The fix is the
checkable fact itself, with a link: "checked to match the listing
printed in Section VI of the companion paper,
[`paper/composition/main.pdf`](../paper/composition/main.pdf)." The
same disease shows up as a bare "exactly" reassuring a claim that
needed no reassurance ("is exactly the first sum" to "is the first
sum"). The surviving "exactly"/"exact" uses are the ones already
carved out above: the antonym of approximate, and a mathematical
identity the reader can verify on the spot ("averages to exactly zero
in an isotropic thermal field" survives, because the reader can verify
an isotropic vector field's mean is zero).

## Compression claims must show their work (blocker unless cashed)

A claim that one thing reduces to, stands in for, or compresses
another must show enough of the actual content that the reader can
judge the reduction, or it gets cut: "with two terms in place of six"
asserts a big simplification without saying which two, which six, or
why four terms vanish. The fix makes it concrete in the same breath:
"GPS's own two terms, the gravitational-potential factor and the
special-relativistic velocity factor, correspond to two of
CliffordClock's own six, the gravitational redshift and the motional
time-dilation correction. CliffordClock's other four terms... correct
for a trapped atom's own local field environment. A free-flying GPS
satellite's orbit carries no such fields to correct for." If the claim
cannot be made concrete in two or three sentences, cut it rather than
leave it as an assertion the reader has to take on faith.

## Current version only (blocker on sight)

Reader-facing documentation describes what the code does now, never a
past broken version, even when the fixed history is reassuring: "An
earlier version of the same code clamped the invalid, negative variance
to zero and printed a small, confident-looking number there... The
eigenvalue check now catches that failure mode directly" spends two
sentences on a bug that no longer exists. The fix states the current
safeguard positively: "The eigenvalue check catches this directly and
blocks the inversion; CliffordClock reports that row's uncertainty as
`nan`, the correct output when the Laplace approximation's own
precondition fails." The one exception is a notebook whose own subject
is a gated demonstration of a bug being found and fixed by the
project's own test suite; that story is the point of the notebook, by
owner approval, not a general license to narrate history elsewhere.

## Meaningless abstraction sentences (blocker on sight)

A sentence that names a category for what was just said, without
saying what it means, gives the reader nothing to check: "The point of
stating these bounds is structural." asserts an abstraction and moves
on. The fix states the actual meaning: "The literature's published
budget totals lose nothing measurable by keeping only the linear sum:
every cross term this section bounds sits at least four orders of
magnitude below the floor those totals report." If a sentence can be
deleted and the paragraph loses no checkable content, it was this kind
of filler.

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
