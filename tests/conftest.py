# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures for the CliffordClock test suite."""

import jax
import pytest


@pytest.fixture
def prng_key() -> jax.Array:
    """A fixed, reproducible JAX PRNG key for randomized tests.

    Returns
    -------
    jax.Array
        A JAX PRNG key seeded with a fixed value (0), so tests using it are
        deterministic and reproducible across runs.
    """
    return jax.random.PRNGKey(0)
