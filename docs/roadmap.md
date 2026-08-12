# Roadmap

CliffordClock's current physics package covers the field-driven and
relativistic terms of an optical-clock frequency budget end to end: DC
Stark shifts from static and gradient fields, electric-quadrupole shifts
for ion clocks, blackbody-radiation shifts, second-order Doppler, and
gravitational redshift across an extended atomic sample. Every one of
these terms runs through the same numerics proven at 1e-18 against
adversarial tests, so a lab can add a new field geometry or species today
and get a full dispersion budget rather than a single hand-computed
number. The sections below describe the physics packages that extend
this coverage next, organized by area, each can be re-prioritized based 
on community interest and need.

## Lattice light shifts

Lattice light shifts are the next physics package on the queue, because
they complete the lattice-clock systematic triad alongside BBR and the
field-driven terms this tool already covers. Modeling them at the magic
wavelength requires species-specific AC polarizability data: the
hyperpolarizability, the E2/M1 polarizabilities, and the motional-state
(intensity-sampling) dependence that recent dual-ensemble evaluations
resolve directly.

## Density and collisional shifts

CliffordClock's ensembles are independent particles by construction,
which is what lets every atom's trajectory and shift be computed
separately and combined into a dispersion budget. Density and
collisional shifts are a different physics class: they need
an interaction model between atoms (s-wave and p-wave collision
physics, excitation-fraction dependence), not an added term on a single
atom's pivot. This package waits for a partner lab to make it the
binding constraint on their evaluation; some labs manage it entirely
through density control and extrapolation, in which case it may stay
outside this tool's scope indefinitely.

## Magnetic coupling (Zeeman shifts)

First- and second-order Zeeman shifts, and the associated BBR-Zeeman
and M1/E2 multipole corrections, are magnetic physics, and nothing
magnetic is computed in the current release: every shipped term enters
the calculation as a scalar contribution to the pivot. Magnetic coupling 
belongs to the bivector sector of the Cl(1,3) rotor engine, and building 
it there is future work with minimal extension to the framework.

## Ion-trap dynamics: tensor polarizability and micromotion

Ion clocks today ship with the electric-quadrupole shift. Tensor
polarizability and RF/micromotion dynamics are the next ion
capability, and they arrive together: published measurements in a
real Sr+ trap show micromotion-driven tensor Stark shifts dominating
the m_J-dependent budget (roughly 95% of it, against roughly 5% from
the quadrupole term alone), so shipping tensor polarizability without
the time-dependent RF field that drives micromotion would model the
smaller term while implying full coverage of the larger one. Every ion
report in the current release carries a note marking this boundary, so
a reader always knows where the modeled budget ends.

## Thermal field maps for blackbody radiation

The current BBR package takes a single, uniform ambient temperature,
which matches how most labs already run their own budget: they compute
an effective temperature themselves, often via thermal finite-element
modeling in tools like ANSYS, and feed CliffordClock that single number.
Importing a spatial temperature map (T(r) across the chamber) and
computing the solid-angle effective temperature it implies is a natural
next step once a partner lab wants to hand over the FEA output directly
instead of pre-reducing it.

## Time-dependent fields

RF trap fields and probe-beam AC Stark shifts share their underlying
machinery with the micromotion work above: both need a field that
varies during the interrogation, not a static snapshot. Building that
time-dependent field machinery once, for the RF/micromotion package,
sets up AC-field support broadly.

## Other clock platforms

Nuclear clocks and highly charged ions are emerging platforms built
specifically to be field-insensitive, which is their scientific
selling point. Because their whole design goal is to suppress the terms
this tool models, there is no near-term case for extending coverage to
them.

## Blind-prediction validation

The validation record today holds two reproducibility cases: published
measurements reconstructed from published inputs with zero fitted
parameters. The stronger claim, predicting a measurement nobody has
already converted to a shift, needs a partner: a lab that shares a
field characterization before (or independently of) its own shift
measurement.

## Transportable and accelerating clocks

A laboratory clock sits still; a transportable clock on a vehicle, a
ship, or a satellite accumulates phase through motion, and an
accelerating frame adds geometric effects that a single shift value per
point in space cannot express. The Cl(1,3) rotor engine was built with
exactly this physics in mind. 

## Engineering and distribution

A handful of items are pure engineering: Allan
deviation and multi-shot statistics for turning a shift budget into a
stability estimate, VTK/Ansys field import so a lab's own FEA output
loads directly, and a PyPI release with a citable DOI so the tool can be
referenced the way any other clock-analysis package is. These are
scoped and queued.

