# SPDX-License-Identifier: AGPL-3.0-or-later
"""CliffordClock: Spacetime Algebra Cl(1,3) rotor dynamics for optical-atomic-clock
fractional frequency shift calculations.

This package targets fractional frequency shift precision at the 1e-18 level.
Achieving that precision requires 64-bit floating point throughout every JAX
computation graph in this package (JAX defaults to 32-bit for speed on
accelerators, which is insufficient by many orders of magnitude here). We
therefore enable x64 mode as an import-time side effect, before any other
submodule can construct a `jax.numpy` array.
"""

from importlib.metadata import PackageNotFoundError, version

import jax

# Must run before any jax.numpy array is created anywhere in this package or
# by downstream code that imports it. See module docstring: the 1e-18 target
# precision is unreachable with JAX's default 32-bit dtype.
# `Config.update` ships without type annotations upstream, hence the ignore.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

try:
    __version__ = version("cliffordclock")
except PackageNotFoundError:  # pragma: no cover - only hit if package not installed
    __version__ = "0.0.0+unknown"

# Re-exported after the x64 config above (WP6): the one-call pipeline
# façade, for `import cliffordclock; cliffordclock.run_pipeline(...)`.
from cliffordclock.pipeline import PipelineConfig, run_pipeline  # noqa: E402

__all__ = ["PipelineConfig", "__version__", "run_pipeline"]
