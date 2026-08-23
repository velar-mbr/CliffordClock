# Contributing to CliffordClock

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Quality bar

Every change must pass, locally and in CI, before merge:

- `ruff check .` — lint clean.
- `ruff format --check .` — formatting clean.
- `mypy src/` — strict type checking on `src/`.
- `pytest --cov --cov-fail-under=90` — all tests green; coverage gate 90%.
  (CI splits this into a fast job and a `slow`-marked job that run in
  parallel; `pytest -m "not slow"` runs just the fast subset locally.)

## Closed dependency rule

Runtime dependencies are a closed set: `jax`, `numpy`, `scipy`, `pyyaml`.
Test-only additions: `pytest`, `hypothesis`, `clifford` (the third-party
[pygae/clifford](https://github.com/pygae/clifford) geometric-algebra
package from PyPI, used purely as an independent cross-check reference —
"test oracle" — for this project's own Cl(1,3) kernel), `ruff`, `mypy`.
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
