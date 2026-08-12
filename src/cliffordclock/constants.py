# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physical constants used throughout CliffordClock.

All values are CODATA 2022 recommended values (source: CODATA Internationally
recommended 2022 values of the Fundamental Physical Constants,
https://physics.nist.gov/cuu/Constants/, published 2024). Constants that are
exact by SI definition (``PLANCK_H``, ``SPEED_OF_LIGHT``,
``ELEMENTARY_CHARGE``, ``BOLTZMANN_K``) are reproduced to their defined exact
values.
"""

import math

#: Planck constant, exact by SI definition. Unit: J s.
PLANCK_H = 6.62607015e-34

#: Reduced Planck constant hbar = h / (2 pi). Unit: J s.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?hbar
HBAR = 1.054571817e-34

#: Speed of light in vacuum, exact by SI definition. Unit: m / s.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?c
SPEED_OF_LIGHT = 299792458.0

#: Electron mass. Unit: kg.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?me
ELECTRON_MASS = 9.1093837139e-31

#: Elementary charge, exact by SI definition. Unit: C.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?e
ELEMENTARY_CHARGE = 1.602176634e-19

#: Boltzmann constant, exact by SI definition. Unit: J / K.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?k
BOLTZMANN_K = 1.380649e-23

#: Unified atomic mass unit. Unit: kg.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?u
ATOMIC_MASS_UNIT = 1.66053906892e-27

#: Bohr radius a0. Unit: m.
#: CODATA 2022: https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
#: Same CODATA vintage already used for `cliffordclock.ensemble.species.ALPHA_AU_TO_SI`
#: (WP21: also feeds `cliffordclock.ensemble.species.EA0_SQUARED_SI`, the
#: atomic-unit-of-electric-quadrupole-moment conversion, CONVENTIONS.md E34).
BOHR_RADIUS = 5.29177210544e-11

#: Electron Compton duration, tau_c = hbar / (m_e c^2) (E7). Unit: s.
#: This is the natural non-dimensionalization timescale for the rotor path
#: integrator (docs/CONVENTIONS.md). Numerically
#: approximately 1.288e-21 s.
TAU_COMPTON = HBAR / (ELECTRON_MASS * SPEED_OF_LIGHT**2)

#: Electron Compton angular rate, omega_C = m_e c^2 / hbar = 1 / tau_c (E8).
#: Unit: rad / s. Satisfies ``OMEGA_COMPTON * TAU_COMPTON == 1.0`` exactly in
#: fp64 (docs/CONVENTIONS.md section 3). Numerically
#: approximately 7.76344e20 rad/s.
OMEGA_COMPTON = 1.0 / TAU_COMPTON

#: Compton *cycle* period, T_C = 2 pi tau_c = h / (m_e c^2) -- the source
#: spec's tabulated ``8.0908e-21 s`` value (a 2*pi-different quantity from
#: TAU_COMPTON; see docs/CONVENTIONS.md section 3). Unit: s.
T_COMPTON_CYCLE = 2.0 * math.pi * TAU_COMPTON

#: Reduced Compton length, lambda_bar_C = c * tau_c (docs/CONVENTIONS.md
#: section 6, E18: the length scale non-dimensionalizing the spin
#: connection, ``omega_tilde_0k = lambda_bar_C * d_k ln P``). Unit: m.
#: Numerically approximately 3.86159e-13 m.
LAMBDA_BAR_COMPTON = SPEED_OF_LIGHT * TAU_COMPTON

#: Standard gravity g_n, exact by international definition (3rd CGPM,
#: 1901; reproduced in the BIPM SI Brochure and CODATA's "non-SI units
#: accepted for use" table -- not a measured/fitted CODATA quantity like
#: the constants above, but an exact convention, so it is reproduced here
#: to its full defined precision rather than truncated). Unit: m/s^2. Used
#: as `cliffordclock`'s default input for CONVENTIONS.md section 15's
#: (E36) uniform gravitational-redshift pivot term (WP22). At the
#: 1e-19-fractional level standard gravity is only a placeholder: a real
#: lab's own surveyed local g differs by parts in 1e3 (e.g. Boulder, CO's
#: 9.796 m/s^2 -- see the Bothwell et al. 2022 benchmark case,
#: `benchmarks/run_bothwell_redshift.py`), so any 1e-19-class comparison
#: against a real site must supply the surveyed local value explicitly
#: (`environment.gravity.g_m_s2`) rather than rely on this default.
STANDARD_GRAVITY = 9.80665
