# DC-Stark coupling (E14b)

`cliffordclock.integrator.omega.pivot_perturbation_stark` implements the
**physical** pivot coupling for optical-lattice clocks: the second-order DC
Stark shift, CONVENTIONS.md E14b. It exists alongside the
linear validation coupling E14a
(`cliffordclock.integrator.omega.pivot_perturbation`), which is untouched.

Clock states carry no permanent dipole moment, so the physically meaningful
stray-field systematic is quadratic in the field: E14a is a
closed-form coupling used to validate the integrator/phase-accumulation
pipeline (CONVENTIONS.md §5). It makes no claim about the real physics of
a clock transition.
`pivot_perturbation_stark` (E14b) is what makes a `cliffordclock` report a physically
meaningful clock-shift number.

**Production path vs. general engine (updated, WP16).** `coupling.type:
stark_dc` (the path every physically-meaningful example in this
repository uses) runs through a scalar phase accumulator
(`cliffordclock.pipeline._stark_scalar_ensemble`, E14b/E21) for
`integration.mode: direct`/`fast_path`/`secular`, and the **true Cl(1,3)
rotor** (`cliffordclock.integrator.omega.build_omega_stark`, E16/E18
instantiated for E14b, via `cliffordclock.pipeline._stark_rotor_ensemble`)
for `integration.mode: worldline`. This is a directly verified
equivalence: `tests/test_integrator_stark_rotor.py` runs the rotor
against the `stark_dc` scalar formulation head-to-head (first-order
agreement at realistic parameters, the permitted O(ω_boost²) divergence
under an exaggerated boost, a uniform-field null, and a v=0 static-node
check), and `benchmarks/run_benchmarks.py`'s NPL reproducibility case
re-runs through the rotor path with the same MET verdict as the scalar
path. This *replaces* the older three-step-chain-only story (rotor
checked against scalar for E14a by E24; E14b checked against E14a as its
linearization by the bridge-identity test; independent theory review
bounding the boost content at second order) as the headline verification
for `stark_dc`↔rotor agreement; that chain remains valid derivation
background (see CONVENTIONS.md), but a direct test now exists where
previously there was none. The Cl(1,3) rotor integrator remains the
general formalism this library is built around and the natural home for
geometric effects beyond DC-Stark; `integration.mode: direct` (classical
trajectories) still uses the scalar accumulator, since a rotor
cross-check is not this WP's target there (see the mode table in
`cliffordclock.pipeline`'s module docstring).

**Status:** current and stable. `coupling: {type: stark_dc, ...}` is a
regular `config.yaml` block: see `docs/cli.md`'s "Pivot coupling"
section for the schema and `examples/lattice_sr87_stark.yaml` for a
runnable example. This document's "The physics"/"Species data"/
`pivot_perturbation_stark` sections below are the current Python API
reference; the design history of how the pipeline wiring was built is
collected in the "Historical design notes" appendix at the end, clearly
marked superseded.

## The physics

CONVENTIONS.md E14b:

```
P(r) - 1 = Δν(r)/ν₀ = -(Δα/2) · |E(r)|² / (h ν₀)
```

with `Δα` the transition's differential static scalar polarizability
(`α(excited) - α(ground)`) and `ν₀` the clock transition frequency,
**not** `m_e c²`, which is the E14a MVP denominator. Equivalently, in terms
of a Stark coefficient `k_S` (Hz·m²·V⁻²):

```
P(r) - 1 = k_S |E(r)|² / ν₀ ,        k_S = -Δα / (2h)
```

Scope (CONVENTIONS.md E14b scope note): for J=0→J=0 lattice clocks (Sr,
Yb) the scalar `Δα` is the whole story. Ion clocks (Al27+) need
tensor/electric-quadrupole polarizability terms, out of MVP scope; the
registry leaves `Al27+` unpopulated and callers get a clear `ValueError`.

### Bridge identity

Expanding `|E|² = |E₀|² + 2E₀·δE + |δE|²` about the E11 baseline/residual
decomposition, the part linear in `δE` is `δE·μ_eff/(hν₀)` with
`μ_eff = -Δα·E₀`: i.e. E14a is exactly the linearization of E14b about
the bias field `E₀`, once E14a's denominator is read as `hν₀` and its `μ`
as `-Δα·E₀`. `tests/test_stark_pivot.py` proves this by taking
`jax.grad` of `pivot_perturbation_stark` with respect to `δE` at `δE = 0`
and comparing against the **unmodified** `pivot_perturbation` (E14a),
fed a rescaled dipole so its `m_e c²` denominator reproduces the `hν₀`
one, literal reuse of the validation machinery, per
CONVENTIONS.md E14b's own suggestion.

## Species data (`cliffordclock.ensemble.species`)

`Species` gained two optional fields:

| Field | Units | Meaning |
|---|---|---|
| `delta_alpha_dc_si` | C²·m²·J⁻¹ (= C·m²·V⁻¹) | `Δα`, literature-native units |
| `stark_coefficient_hz_per_v2_m2` | Hz·m²·V⁻² | `k_S = -Δα/(2h)`, derived automatically from `delta_alpha_dc_si` |

Both are `None` for a species with no DC-Stark data. Read the resolved
coefficient through
`Species.resolve_stark_coefficient_hz_per_v2_m2()`: it prefers `stark_coefficient_hz_per_v2_m2` if set,
falls back to deriving it from `delta_alpha_dc_si`, and raises a clear
`ValueError` (citing E14b and the species name) if neither is populated.
The resolver is what applies that precedence and derivation consistently
for every caller. A caller reading `species.delta_alpha_dc_si` or
`species.stark_coefficient_hz_per_v2_m2` directly gets whichever field
happens to be populated (often `None`), with no fallback and no
validation.

Populated species and their literature sources:

- **Sr87**: `Δα = 4.07873(11) × 10⁻³⁹ C²m²J⁻¹`. T. Middelmann, S. Falke,
  C. Lisdat, U. Sterr, "High Accuracy Correction of Blackbody Radiation
  Shift in an Optical Lattice Clock", Phys. Rev. Lett. 109, 263004
  (2012), arXiv:1208.2848. Cross-checked against the theoretical value
  `Δα = 4.305(59) × 10⁻³⁹` (S.G. Porsev et al., cited in the same paper).
- **Yb171**: `Δα = 2.40269(5) × 10⁻³⁹ C²m²J⁻¹` (= 145.726(3) a.u.).
  J.A. Sherman, N.D. Lemke, N. Hinkley, M. Pizzocaro, R.W. Fox,
  A.D. Ludlow, C.W. Oates, "High-Accuracy Measurement of Atomic
  Polarizability in an Optical Lattice Clock", Phys. Rev. Lett. 108,
  153002 (2012), arXiv:1112.2766, Table II. Cross-checked against the
  theoretical value `Δα = 2.56(26) × 10⁻³⁹` (S.G. Porsev, A. Derevianko,
  Phys. Rev. A 74, 020502 (2006), cited in the same table).
- **Al27+**: `Δα(0) = 0.416(14) a.u.` (Wei et al., Phys. Rev. Lett. 133,
  033001 (2024), secondary; Brewer et al., Phys. Rev. Lett. 123, 033201
  (2019), 0.426(58) a.u., primary text, recorded as the fallback, WP21,
  the project's theory sign-off record (G8), B1). No `BbrCoefficients`
  (see below); micromotion-boundary and hyperfine-E2 budget-line report
  notes carried on every run of this species.
- **In115+**: `Δα(0) = 2.01 a.u.` theory (Safronova, Porsev, Safronova,
  Phys. Rev. Lett. 107, 143006 (2011), primary text, WP21, G8 sign-off
  B2). Same no-`BbrCoefficients`/report-notes treatment as Al27+.

### a.u. → SI conversion

Literature polarizabilities are frequently quoted in atomic units.
`cliffordclock.ensemble.species.ALPHA_AU_TO_SI` converts:

```python
alpha_si = alpha_au * ALPHA_AU_TO_SI  # C^2 m^2 J^-1
```

`ALPHA_AU_TO_SI = 1.64877727436e-41` (= `4π ε₀ a₀³`, CODATA 2022
`ε₀`/`a₀`, pinned at full precision per an independent theory review).
Value confirmed: this module's value has been independently recomputed
from CODATA and matches. CONVENTIONS.md previously carried a
transcription digit-swap in this constant (`1.648772e-41`, in the sixth
significant figure); that slip was caught and has since been corrected
directly in CONVENTIONS.md, which now also reads `1.648777e-41`; the
full-precision value was independently reviewed and confirmed
(CONVENTIONS.md §11): see the module docstring.

### Explicit overrides: `StarkCoefficients`

For a species not in the registry, or a value newer than the pinned
registry entry, `cliffordclock.ensemble.species.StarkCoefficients` bypasses
the registry entirely:

```python
from cliffordclock.ensemble.species import StarkCoefficients

coeffs = StarkCoefficients(
    clock_frequency_hz=1_121_015_393_207_857.4,
    delta_alpha_dc_si=1.0e-40,  # or stark_coefficient_hz_per_v2_m2=...
)
```

At least one of `delta_alpha_dc_si` / `stark_coefficient_hz_per_v2_m2`
is required; if both are given they must agree (`__post_init__` checks
`k_S == -Δα/(2h)` to a loose relative tolerance) or construction raises.
`StarkCoefficients` implements the same
`resolve_stark_coefficient_hz_per_v2_m2()` method as `Species`, so both
are interchangeable wherever `pivot_perturbation_stark` expects a
`Species | StarkCoefficients`.

## `pivot_perturbation_stark`

```python
def pivot_perturbation_stark(
    e0: jax.Array,  # shape (..., 3), V/m, E11 baseline field E_0(r)
    delta_e: jax.Array,  # shape (..., 3), V/m, E11 perturbation field δE(r)
    species_or_coeffs: Species | StarkCoefficients,
) -> jax.Array:  # shape (...,), dimensionless, P(r) - 1
    ...
```

```python
from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import pivot_perturbation_stark

species = get_species("Sr87")
p_minus_1 = pivot_perturbation_stark(e0, delta_e, species)
```

**API note:** this function takes the E11
baseline/perturbation split as two separate arguments, `e0` and
`delta_e`, matching the existing E14a functions' established convention
at this module boundary and matching what the precision discipline below
requires.

### Precision discipline (E10)

`|E|² = |E₀|² + 2E₀·δE + |δE|²` is evaluated **term-by-term**, with each
term individually scaled by the (very small) `k_S/ν₀` prefactor *before*
summing:

```
baseline_term  = (k_S/ν₀) · |E₀|²          # the "no gradient" DC-Stark shift
gradient_term  = (k_S/ν₀) · 2(E₀·δE)       # precision-critical
residual_term  = (k_S/ν₀) · |δE|²          # second order in δE, usually negligible
P(r) - 1       = baseline_term + gradient_term + residual_term
```

This never forms a combined total-field vector `E₀ + δE` before squaring.
`tests/test_stark_pivot.py` demonstrates why: at `|E₀| = 1e5 V/m` with a
`δE` sized to produce a `~1e-19`-level `gradient_term`, computing the
naive `|E₀ + δE|²` and subtracting a separately-computed baseline already
loses several significant digits relative to a 50-digit `decimal`
reference, while the term-by-term form matches that reference to
`~1e-16` relative. All functions are batched (leading `...` axes),
`float64`, and jit/vmap/grad-safe.

## Blackbody-radiation shift (E32/E33, WP20)

Every real optical-lattice clock sits inside a room-temperature (or
actively temperature-controlled) enclosure. That enclosure emits a
thermal (blackbody) infrared radiation field, and this field perturbs
the clock transition by exactly the same second-order Stark mechanism as
a stray DC field, with the field's mean-square magnitude set by the
Stefan-Boltzmann law. It is, in practice, the dominant systematic in a
real strontium or ytterbium lattice clock, so this project gives it its
own explicit model with a dedicated formula and coefficient provenance.

`cliffordclock.integrator.omega.bbr_pivot_perturbation` computes the BBR
pivot term (CONVENTIONS.md E32) for a given radiation temperature and
species:

```python
from cliffordclock.ensemble.species import get_species
from cliffordclock.integrator.omega import bbr_pivot_perturbation

species = get_species("Sr87")
bbr_shift = bbr_pivot_perturbation(300.0, species)  # ~-5.32e-15 at 300 K
```

It composes additively into the DC-Stark pivot (E33: `P-1 = (P-1)_stark +
(P-1)_BBR`), via a keyword-only `bbr_pivot_perturbation` parameter on
`pivot_perturbation_stark`/`spin_connection_stark`/
`scalar_rate_perturbation_stark`/`build_omega_stark` (all default `0.0`,
an exact no-op): see CONVENTIONS.md §13 for the full formula, coefficient
provenance, and uncertainty-propagation details.

**Config (`docs/cli.md`'s "Environment" section has the full schema):**

```yaml
environment:
  radiation_temperature_K: 300.0                    # kelvin; absent = BBR off
  radiation_temperature_uncertainty_K: 0.004         # optional, kelvin
```

Requires `coupling: {type: stark_dc, ...}` (BBR needs the species'
registry-published static/dynamic BBR fit) and a temperature in
`[50, 350]` K (hard `PipelineConfigError`
outside that range, the published fit's support). Composed into every
evaluation mode: `fast_path`, `secular`, classical `direct` (batched and
streaming), and the rotor worldline. Every shipped example omits this
section entirely, so BBR is off (and every example's output byte-identical
to before WP20) unless a config opts in.

**Species data:** `Species.bbr_coefficients`
(`cliffordclock.ensemble.species.BbrCoefficients`) carries the static
shift at 300 K, its dynamic-term polynomial coefficients, and both
quantities' uncertainties, for `Sr87` and `Yb171` (unpopulated for
`Al27+`, same pattern as `delta_alpha_dc_si`). See CONVENTIONS.md §13 for
the citations and shape-vs-anchor reasoning behind each coefficient.

**Uncertainty and labeling.** The pipeline report's BBR
uncertainty is *arithmetic-reproduction fidelity*: does the code evaluate
the published formula correctly. It is not an independent claim about
*BBR accuracy*, since the physical uncertainty is set by the registry's
published coefficient uncertainties (combined in quadrature, `≈8e-19`
for Sr87 at 300 K), optionally widened by a
`radiation_temperature_uncertainty_K`.
The report also carries an explicit "modeled-out, `≈6e-20` each" M1/E2
multipole budget line and, for `300 < T ≤ 350 K`, a note that the run is
in-fit-range but beyond the PTB↔JILA 1e-19-class cross-verification band.

**Non-goals (this MVP; see WP20):** no `T(r)`
spatial maps or solid-angle effective-temperature computation from chamber
geometry (uniform `T` only), no stochastic/Monte-Carlo BBR field sampling
(the bath self-averages over ~1e14 independent cycles per interrogation,
deterministic, matching every published evaluation), no
hyperpolarizability (`β`) terms, no BBR-Zeeman coupling. A benchmark case
reproducing a published BBR-row measurement (JILA 2024) is a separate,
later work package with its own review.

## Ion-clock electric-quadrupole shift (E34/E35, WP21 Tier 2)

For a D/F-state ion clock (Ca+, Sr+, Ba+, Yb+), the clock state's electric-
quadrupole moment `Θ` couples to the LOCAL electric-field gradient:
first order in the gradient, distinct from the second-order-in-field
DC-Stark/BBR mechanisms above, and structurally the largest engineering
payoff of this project's field-gradient machinery: the smoother already
delivers the full gradient tensor (E13) everywhere, so no new field
capability was needed to add this term.

`cliffordclock.integrator.omega.quadrupole_pivot_perturbation` computes
the pivot term (CONVENTIONS.md E34/E35) from a LOCAL gradient tensor:

```python
import jax.numpy as jnp
from cliffordclock.ensemble.species import get_quadrupole_moment
from cliffordclock.integrator.omega import quadrupole_pivot_perturbation

ca = get_quadrupole_moment("Ca+:D5/2")  # Theta = 1.83(1) a.u., primary
grad_e = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0e8]])  # V/m^2
axis = jnp.array([0.0, 0.0, 1.0])
# nu_0: Ca+ 4S1/2-3D5/2, Chwalla et al., PRL 102, 023002 (2009)
nu_0_hz = 411_042_129_776_393.0
shift = quadrupole_pivot_perturbation(grad_e, axis, ca.theta_au, ca.j, m_j=2.5, nu_0_hz=nu_0_hz)
```

It composes additively into the DC-Stark pivot (E35, the same additive
pattern E33 already establishes for BBR), via a keyword-only
`quadrupole_pivot_perturbation` parameter on the same four
`pivot_perturbation_stark`/`spin_connection_stark`/
`scalar_rate_perturbation_stark`/`build_omega_stark` functions BBR
extends (default `0.0`, exact no-op), but unlike BBR (a single per-run
scalar), the quadrupole term is per-point: `cliffordclock.pipeline`
evaluates it fresh from each point's local gradient tensor on every
call. See CONVENTIONS.md §14 for the full formula (including the
coordinate-free `n_hat^T . G(r) . n_hat` reduction this project
implements, algebraically equivalent to the literal Itano/Roos
axial-plus-asymmetric form; the derivation is in that section), sign-
verification discipline, and the exact three-orthogonal-orientation
cancellation proof.

**Config (`docs/cli.md`'s "Quadrupole shift" section has the full
schema):**

```yaml
quadrupole:
  state: "Ca+:D5/2"       # a QUADRUPOLE_MOMENTS registry key
  nu_0_hz: 411042129776393.0  # Ca+ 4S1/2-3D5/2, Chwalla et al., PRL 102, 023002 (2009)
  m_j: 2.5
  quantization_axis: [0.0, 0.0, 1.0]
```

Requires `coupling: {type: stark_dc, ...}`. Every shipped example omits
this section, so the quadrupole term is off (byte-identical output)
unless a config opts in.

**Sign discipline.** The leading sign is transcribed directly from Roos
et al.'s primary-text Eq. 1 (an owner-supplied preprint, quant-ph/0701215v1,
read in full during the G8 gate), reconciled against Itano 2000 Eq. 46's
hyperfine-form sign in CONVENTIONS.md §14. The convention-free m_J ratio
(`-1.25` for a D5/2 state) and Yb+ F7/2's negative-Theta relative-sign
anchor are pinned regressions; the sign form is additionally confirmed by
Dube et al. 2005 in primary text (owner-supplied; ion-clock dossier
section 7), which also contributes the magic-m_J^2 intercept regression.
Caution recorded there: Dube's measured slope is ~95% micromotion tensor
Stark, so it is never cited here as a pure quadrupole absolute anchor.

**Species data:** `cliffordclock.ensemble.species.QUADRUPOLE_MOMENTS`
carries `Theta(J)` for five D/F states across four ion species, each
tagged `verification="primary"`/`"secondary"`/`"secondary-paywalled"`.
None of these ions are registered `Species` entries (no clock-transition
frequency is pinned for them in this WP's scope): `quadrupole.nu_0_hz`
is always an explicit config input, unlike the Stark/BBR terms' species-
resolved `nu_0`.

**Non-goals (WP21):** no RF/
micromotion dynamics (the dominant ion stray-field pathway, every
Al27+/In115+ report carries an explicit micromotion-boundary note, "a
quadrupole map is not a stray-field budget"), no tensor polarizability
/ m_J-dependent Stark, no rotor/bivector content beyond the
scalar pivot composition (a fixed `(J, m_J, axis)` collapses the tensor
character to a scalar coefficient, G8 sign-off A3), no spin-connection
gradient contribution from the quadrupole term (would need
third-order field derivatives the smoother does not expose, a
documented, bounded scope limit, CONVENTIONS.md §14). A benchmark case
reproducing a published quadrupole-shift measurement (Roos et al. 2006 /
Barwood et al. 2004) is a separate, later work package with its own
review.

## Gravitational redshift and the extended-lattice ensemble (E36, WP22)

A clock's rate also depends on its position in a gravitational
potential. Every term above is electromagnetic; gravity is a separate
effect: a clock a millimetre higher in a lab ticks faster, by a
fractional amount `g*Delta_h/c^2` (General Relativity's weak-field
gravitational redshift). Real optical-lattice clocks now measure this
directly across a single, extended (millimetre-scale) atomic sample.
Earlier redshift measurements relied on comparing two separated clocks;
Bothwell et al. (Nature 602, 420 (2022)) resolves the gradient within
one sample and is the showpiece measurement this section targets
(`benchmarks/run_bothwell_redshift.py`).

`cliffordclock.integrator.omega.grav_pivot_perturbation` computes the
gravitational pivot term (CONVENTIONS.md section 15, E36) for a given
height and local gravitational acceleration:

```python
from cliffordclock.constants import STANDARD_GRAVITY
from cliffordclock.integrator.omega import grav_pivot_perturbation

grav_shift = grav_pivot_perturbation(1.0, STANDARD_GRAVITY)  # ~+1.09e-16 per metre of height
```

It composes additively into the DC-Stark pivot (E36, E33's pattern:
`P-1 = (P-1)_stark + (P-1)_BBR + (P-1)_Q + (P-1)_grav`), via a keyword-only
`grav_pivot_perturbation` parameter on `pivot_perturbation_stark`/
`spin_connection_stark`/`scalar_rate_perturbation_stark`/
`build_omega_stark` (all default `0.0`, an exact no-op), mirroring the
BBR term's threading pattern exactly, but (like the quadrupole term) a
per-position value computed by the caller from each point's height via
`cliffordclock.integrator.omega.height_along_axis`. See CONVENTIONS.md
§15 for the full formula, sign convention, validity bounds, and the G9
theory sign-off record.

**Config (`docs/cli.md`'s "Gravitational redshift" section has the full
schema):**

```yaml
environment:
  gravity:
    g_m_s2: 9.80665           # optional (default STANDARD_GRAVITY); use the
                               #   lab's own surveyed local g for 1e-19-class work
    up_axis: [0.0, 0.0, 1.0]  # optional (default [0,0,1])
    reference_height_m: 0.0   # optional (default 0.0)
```

Requires `coupling: {type: stark_dc, ...}` (mirrors BBR's cross-field
validation exactly). Composed into every evaluation mode: `fast_path`,
`secular`, classical `direct` (batched and streaming), and the rotor
worldline. Through the rotor, gravity enters only via the SCALAR pivot
(the `B_hat_C` rotation-plane coefficient). `omega_boost`'s gradient
term is provably inconsequential for every configuration this project
ships, since every lattice/lattice_extended node is static (`v=0`), and
`omega_boost`'s coefficient carries an explicit factor of `v`. Every
shipped example
omits this section entirely, so gravity is off (byte-identical output)
unless a config opts in.

**The `lattice_extended` ensemble regime** (`ensemble.regime:
lattice_extended`, `cliffordclock.ensemble.lattice.extended_lattice_nodes`)
is what makes the gravitational term's SPATIAL variation observable: `n`
copies of the `lattice` regime's own single-site Hermite-Gauss quadrature,
distributed along a configured axis with a Gaussian-or-uniform
site-occupation envelope. Every site's own position feeds every pivot
term already in play (local field/Stark, uniform BBR, height-dependent
gravity) through the SAME `fast_path`/`worldline` accumulators the
`lattice` regime uses: no new evaluation-mode machinery. The result
(`PipelineResult.site_map`,
`cliffordclock.pipeline.LatticeExtendedSiteMap`) is the per-site frequency
map (the observable Bothwell-class measurements report) plus a
weighted-least-squares linear-gradient fit (`slope_per_m`) and the
dispersion-labeling split gate edit 4 requires (below).

**Dispersion labeling (a required design discipline).** For an
extended sample the frequency spread across sites is dominated by the
DETERMINISTIC linear gradient (higher/lower sites tick at a
systematically different rate). Reporting a map, a spread, and a
T2*/linewidth together without separating that gradient out risks
double-counting it or misreading it as decoherence.
`LatticeExtendedSiteMap`
reports BOTH the total spread (`total_spread_fractional`, the same
combined number `MetrologyReport.t2_star_s` is derived from, in
fractional-shift units) AND the gradient-removed residual spread
(`gradient_removed_residual_spread_fractional`, what remains after
subtracting the best-fit linear gradient from each site's own mean),
extending the showcase's existing SEM-vs-T2* discipline to the
deterministic-vs-stochastic axis. Every `lattice_extended` report carries
a test-pinned note stating this explicitly
(`cliffordclock.pipeline.LATTICE_EXTENDED_DISPERSION_LABEL_NOTE`).

**Coordinate-sign convention is this project's own** (a HIGHER clock along
`up_axis` runs FASTER, so a positive-slope map means the sample's
"upward" direction is up); a published paper's own axis
convention may run the opposite way (Bothwell's z-axis increases toward
LOWER physical height, so their published gradient is negative); mapping
between the two conventions is the comparing script's job, and
`benchmarks/run_bothwell_redshift.py` documents and applies
that specific mapping explicitly.

**Validity bounds.** The uniform-`g` approximation is exact to `<<1e-19`
at lab/mm scale and stays below the 1e-19 floor out to `~76 m` of height
extent; `cliffordclock.pipeline.GRAVITY_EXTENT_WARN_M` (10 m) triggers a
runtime warning note well before that
bound, recommending a surveyed potential difference or height-dependent
g/geoid model beyond it. `g_m_s2` defaults to standard gravity, a
placeholder at the 1e-19 level: the lab's own surveyed local value is
the physically correct input there (CONVENTIONS.md §15).

**Non-goals (this WP; WP22):** no
tidal/higher-order gravity terms (uniform `g` only, the validity bound
above states exactly where that stops being exact), no lattice light
shifts/density/collisional/Zeeman effects in the extended-sample mode
(recorded post-beta), no conveyor-belt/moving-ensemble dynamics (the
mode's geometry is static; trajectory/rotor support already exists
unchanged for anyone who wants it later), no changes to the chamber-scale
showcase or the existing `lattice` regime (byte-identical). The Bothwell
2022 benchmark case (`benchmarks/run_bothwell_redshift.py`) is this WP's
own scope; Zheng et al. (Nature 602,
425 (2022)) is recorded as the named second extended-lattice candidate,
post-beta.

## Historical design notes (superseded, kept for record)

The rest of this document is design history: the
sketch that guided wiring `pivot_perturbation_stark` into
`cliffordclock.pipeline`/`cliffordclock.cli`, and how the shipped
implementation ended up differing from that sketch. Everything a user or
contributor needs today is above this point ("The physics" through
"Precision discipline"), plus `docs/cli.md`'s "Pivot coupling" section for
the current `config.yaml` schema.

**Original sketch** for the `coupling:` config block:

```yaml
coupling:
  type: linear_mu | stark_dc          # linear_mu keeps E14a
  # linear_mu (existing, unchanged):
  mu: [1.0e-25, -2.0e-25, 1.5e-25]     # C.m
  # stark_dc:
  delta_alpha_dc_si: 4.07873e-39       # optional explicit override, C^2 m^2 J^-1
  stark_coefficient_hz_per_v2_m2: ...  # optional explicit override, Hz m^2 V^-2 (alternative to delta_alpha_dc_si)
```

**How the shipped implementation differs from that sketch**
(`src/cliffordclock/pipeline.py`), each a documented decision:

- **The default stays `linear_mu`** when `coupling.type`
  is omitted; every config written before this coupling existed (which
  never wrote a `type` key at all) keeps its exact behavior unchanged.
  `docs/cli.md` documents `stark_dc` as the recommended choice for new
  configs.
- **No pipeline-level E11 baseline/perturbation split.** The original
  sketch assumed a separate baseline/residual field would already be
  available inside the pipeline; in the implementation as built, every
  field abstraction the pipeline actually uses returns one combined total
  field, so the implementation calls
  `pivot_perturbation_stark(e0=E(r), delta_e=0, ...)`. The
  `baseline` term alone then evaluates the exact E14b formula (`cross`/
  `quadratic` are identically zero); this costs nothing in accuracy,
  since the pipeline already works with one combined total field, and
  the exact E14b formula still comes out whole through the `baseline`
  term.
- **No direct swap inside the rotor integrator's field-to-`Ω` step**
  *(superseded by WP16 for `integration.mode: worldline`: see the
  "Production path vs. general engine" note above; this bullet describes
  the pre-WP16 state as originally shipped)*. As originally built,
  `integration.mode: direct`/`worldline` with `coupling.type: stark_dc`
  both routed through a coupling-agnostic scalar phase accumulator
  (`cliffordclock.pipeline._stark_scalar_ensemble`) that never built a
  Cl(1,3) rotor at all: E14b had no `Ω`-bivector construction in this
  codebase, a genuine gap flagged for future work. WP16 closed that gap
  for `integration.mode: worldline`
  specifically (`cliffordclock.integrator.omega.build_omega_stark` +
  `cliffordclock.pipeline._stark_rotor_ensemble`); `integration.mode:
  direct` still uses the scalar accumulator (classical-ensemble
  trajectories are not this WP's target). `integration.mode:
  fast_path`/`secular` needed no change at all, then or now: both already
  consume the coupling-agnostic `fastpath.RateFn` seam.
- **Provenance is recorded in `uncertainty_notes` (free text).** Free
  text is already the documented extension point for this kind of note,
  so no dedicated `report-schema.md` field exists for it.
