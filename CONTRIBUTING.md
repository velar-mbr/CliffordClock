# Contributing to CliffordClock

CliffordClock gets better with every lab's case that runs through it. NPL's
stray-field measurement and Bothwell et al.'s mm-scale redshift measurement
are the two reproducibility cases the validation record stands on today, and
both exist because that data was public enough to reconstruct end to end. We
want more of that: your species, your chamber, your trap, your published
evaluation, run through this engine and written up as a real case the next
person can check.

This file is that invitation. Whether you run a clock, work on the physics
this engine touches, or want to fix a sentence that reads wrong, we want you
contributing, and the rest of this document explains how.

## What helps most

The clearest way to help is the same thing this project has needed since
notebooks 06 and 09 turned NPL's and Bothwell's data into the two
reproducibility cases `docs/validation.md` records: a real case from your own
lab. Short of that, contributions land in a few different places depending on
what you have to give.

- **A benchmark case from your own lab.** If you have a published or
  shareable evaluation, both a characterized field and the shift it
  produced, `benchmarks/partner_case_template/` walks through what plugs in
  where. A case with both halves closes the gap in this project's validation
  record: the blind-prediction case that does not exist yet.
- **New species or coefficient registrations.** Every coefficient this
  engine ships carries its source, which paper, which value, in the output.
  A new species entry needs the same: a primary-source citation for every
  number you add.
- **Chamber or trap geometry cases.** A real chamber's own ray-traced
  exchange-factor table, the way notebook 12 uses Bothwell's Table 2.7, or a
  real trap's measured mode structure, turns an illustrative sweep into a
  validated case.
- **Roadmap items.** `docs/roadmap.md` lists what is queued and why: N>2 ion
  crystals, RF trap dynamics and micromotion, non-Gaussian motional states,
  time-varying fields. Pick whichever one already matters to your own work.
- **Documentation and prose fixes.** A confusing sentence, a stale link, a
  missing unit: these are exactly as welcome as a new physics term, and
  often faster to review.

Open an issue first for anything large enough to need a design conversation,
a new benchmark case, a new species, a new physics term. A small fix, a
typo, a broken link, a clear bug, can go straight to a pull request.

## How we collaborate, and why

Every physics change here gets an independent review. The reviewer checks
the physics against primary sources, works through the derivation, and
either signs off or sends the change back. The public record of what that
looks like lives in `plan/reviews/`, gate by gate, including the reviews
that found a real blocker and sent a draft back for another pass.
Every number that appears in this project's prose, in a notebook, a
docstring, a README claim, is computed by the code that ships and traced
back to its source, never typed in from a calculator.

The prose itself follows a written standard, `.claude/skills/prose-review/SKILL.md`,
built from the owner's own accumulated review feedback, and every review
applies it end to end, catching everything from stray meta-commentary to a
prose number that drifted from what the code printed. CI and the
release-checks battery (`python tools/release_checks.py`) run the mechanical
layer of that same standard on every pull request, alongside the lint,
type, and test gates below. AGPLv3 keeps the whole chain open: if you build
on this code and serve its results over a network, your changes stay open
too.

That rigor exists because of what this project is for. A lab adopts a
clock-systematics tool because its numbers can be trusted enough to go
straight into a paper's own uncertainty budget. The review process, the
citation discipline, and the prose standard are the product: the reason a
lab can hand this engine a field map and trust the shift that comes back is
that every step between the map and that number has already been checked by
someone else, and checked again by CI on every change. Every contribution
that touches the physics goes through that same process, whichever lab or
person sends it.

## Practicalities

PRs from forks are welcome. Approval needs a green CI run, a clean
release-checks pass, and a reviewer's sign-off, the same bar this project
holds its own commits to; a physics-touching PR gets the independent review
described above, and a documentation or prose-only PR gets the equivalent
prose review. If you are not sure whether something counts as large, open
an issue and ask; that conversation is itself a welcome contribution.

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Quality bar

Every change must pass, locally and in CI, before merge:

- `ruff check .`: lint clean.
- `ruff format --check .`: formatting clean.
- `mypy src/`: strict type checking on `src/`.
- `pytest --cov --cov-fail-under=90`: all tests green; coverage gate 90%.
  (CI splits this into a fast job and a `slow`-marked job that run in
  parallel; `pytest -m "not slow"` runs just the fast subset locally.)

## Closed dependency rule

Runtime dependencies are a closed set: `jax`, `numpy`, `scipy`, `pyyaml`.
Test-only additions: `pytest`, `hypothesis`, `clifford` (the third-party
[pygae/clifford](https://github.com/pygae/clifford) geometric-algebra
package from PyPI, used purely as an independent cross-check reference, a
"test oracle", for this project's own Cl(1,3) kernel), `ruff`, `mypy`.
Notebook-only: `matplotlib`, `jupyter`. Adding any other
runtime dependency requires an owner-approved architecture decision record
before it lands in `pyproject.toml`.

## Frozen module map

The package has exactly these top-level modules: `cl13`, `fields`,
`ensemble`, `integrator`, `analytics`, plus `constants.py` and `cli.py`.
New top-level modules require an owner-approved architecture decision
record.

## Docstring style

Public functions and classes use NumPy-style docstrings, including units and
array shapes for anything physics-related.

## License and headers

The project is licensed AGPLv3 (see `LICENSE`). Every `.py` file must
start with:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

## Physics conventions

All physics code must cite the equation number(s) it implements from
`docs/CONVENTIONS.md` in its docstring. Code that disagrees with
`docs/CONVENTIONS.md` is treated as a defect regardless of whether its tests
pass.
