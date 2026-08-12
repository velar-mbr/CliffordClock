## What this changes and why

<!-- One or two sentences. Link an issue if there is one. -->

## Quality bar (see CONTRIBUTING.md)

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy src/` passes
- [ ] `pytest --cov` passes locally, including any new/updated tests
- [ ] Any touched notebook was re-executed (`jupyter nbconvert --to
      notebook --execute --inplace ...`)
- [ ] Physics changes cite the relevant `docs/CONVENTIONS.md` equation
      number(s), and `docs/CONVENTIONS.md` was updated if the equation
      itself changed
- [ ] Docs updated for any user-visible behavior change (README, the
      relevant `docs/*.md`, or `CHANGELOG.md`)

## Notes for the reviewer

<!-- Anything non-obvious: deviations from an original spec, deliberate
     scope cuts, open questions. -->
