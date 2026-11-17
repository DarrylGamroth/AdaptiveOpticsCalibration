# Focal-plane sharpening: numerical extraction review

## Scope and disposition

Independent design review, 2026-10-02. **Proposed numerical contracts, ready
for primary-agent adjudication and implementation; no implementation or
instrument qualification is claimed.** The initial target is a deterministic
scalar CPU implementation through the existing `prepare` / `process` /
`process!` interface. A single `FocalPlaneSharpening` domain module is sufficient.
No general optimization framework is needed.

The reusable operations are an image metric, generation of an explicit
two-sided probe pair, an SPSA gradient estimate from two supplied metrics, and
one Adam step from supplied state. Frame acquisition, averaging policy,
command constraints, scheduling, application and acceptance belong to the
composing application. The package never receives an equipment callback.

## Source and authority

- AOC baseline: `3c11576fe65ca4998f1357d0210bb399d68e7b12` in
  `AdaptiveOpticsCalibration-reference-frames`; authority is [DESIGN.md](../DESIGN.md),
  [AGENTS.md](../AGENTS.md), [MIGRATION.md](../MIGRATION.md), and
  [src/interface.jl](../src/interface.jl).
- Observed source: `focal-plane-sharpen.jl`,
  SHA-256 `fe2dc48de82539bfbd0c4a6a67ee5032e3c2374de00d7d364ce13663ea8361c5`.
  A Git revision and reproducible SPIDERS package environment were unavailable
  in that directory. This review did not execute the script or contact equipment.
- Relevant script lines: SPSA 41–56; Adam 62–107; optimization policy 123–167;
  command construction 198–203; image accumulation 212–217; metric 242–278;
  exploratory runs 291–357; best-flat activation after line370.
- Locally installed ImageFiltering **0.7.12**, `src/mapwindow.jl:75`, declares
  `border="replicate"` and a centered `(3,3)` footprint. This establishes the
  inspected dependency's behavior, not the missing historical environment.
  The new contract makes boundary behavior explicit.
- Preserve source attribution and determine applicable licensing before copying
  implementation text. The equations below do not imply permission to import
  the equipment script wholesale.

## Observed objective and proposed metric contract

The script first averages camera images, then median-filters that average.
It uses the entire image: the smaller ROI expressions are commented out and
the `xc,yc` arguments do not select the active ROI. Its center is the maximum
of the **filtered** image. The background accumulator is unused in the return
value. The actual objective is minimized and has no flux normalization.

For supplied finite real ROI I, let F be its centered 3×3 median with replicated
edge samples. Let p be the first maximum of F in Julia column-major order.
Using row/column pixel coordinates:

```text
r(i,j) = hypot(i − p.row, j − p.column)
J(I)   = −Σ F[i,j] / (1 + r(i,j))², over ROI pixels with r(i,j) < R
SPIDERS value: R = 7 pixels
```

Proposed method type: `WeightedCoreIntensity`; prepared specification fixes
the ROI dimensions and explicit positive finite radius. Kernel width3,
replicated boundaries, argmax centering, exponent2 and the negative sign are
part of this method's definition. Do not add selectable filter/normalization
registries for the initial extraction. Radius7 is the source-compatible case;
other radii are explicit scientific choices.

Input is one complete ROI, which may already be an arithmetic mean supplied
by the caller. Applying this metric to a mean image generally differs from
averaging metrics of individual images. It is not an acquisition or running
mean API. Use Float64 numerical storage/accumulation for the CPU reference;
retain input values without hidden background subtraction or rescaling.

Product: objective, peak Cartesian index, filtered peak value, aperture pixel
count, whether the discrete aperture extends beyond the ROI, and typed
numerical status. A filtered-image result is unnecessary for the minimal API;
it can remain replaceable workspace. A finite objective establishes numerical
completion only, not successful sharpening or a valid physical PSF.

Required corner conditions:

- Empty ROI, nonfinite input, incompatible dimensions or unsupported axes are
  invalid inputs. The initial implementation may explicitly require one-based
  CPU matrices and support ordinary views; it must reject unsupported axes
  before indexing rather than silently reinterpret coordinates.
- Median uses nine samples with replicated edges, including 1×1 and 1×N
  inputs. It is not a shrinking-window median or zero padding.
- Tied maxima select the first Cartesian index in column-major traversal.
  An isolated hot pixel can disappear under the median; this is intentional.
- The aperture uses strict `r < R`. At R7, offsets exactly seven pixels away
  are excluded. Image edges truncate the sum without renormalization.
- Signed finite intensities are permitted. Zero and constant images have a
  defined metric and deterministic center, without being accepted as useful
  physical observations. No guessed signal threshold belongs in this method.
- Radius and distances are in pixels. Intensity units remain the supplied
  radiometric units (e.g. detector counts per averaged exposure); J has those
  units times dimensionless pixel weights. It is not a Strehl ratio, encircled
  energy fraction, calibrated flux or λ/D metric. Exposure/gain/illumination
  changes can change J even with an unchanged PSF.
- Finite inputs can still overflow accumulation. Return an explicit invalid
  numerical product rather than a success status with Inf/NaN.

## SPSA: supplied perturbations and measurements

Two complete operations avoid passing an evaluation callback into AOC:

1. `SPSAProbes`: supplied coefficient vector q, perturbation magnitude c and
   Rademacher direction Δ produce candidate vectors q⁺ and q⁻.
2. `SPSAGradient`: supplied Δ, c, J⁺ and J⁻ produce an estimated gradient.

```text
q⁺ = q + c Δ
q⁻ = q − c Δ
ĝᵢ = (J⁺ − J⁻) / (2 c Δᵢ)
```

The specification fixes coordinate count, ordered basis and coefficient units.
The smallest contract uses one scalar c for homogeneous coefficient units.
Keep c explicit for each complete operation so the caller owns its schedule.
Require nonempty matching one-based vectors, finite q, finite positive c,
and each Δᵢ exactly −1 or +1. Booleans/zeros/arbitrary nonzero directions are
not Rademacher inputs. Do not generate random directions inside the method.
Reject nonfinite candidate coordinates and candidate perturbations lost to
the declared working precision. The caller separately validates conversion
to the eventual command representation and actual adopted commands.

The gradient product has units objective/coefficient. J⁺ and J⁻ must be finite
scalars from the same metric definition, radiometry and supplied probe pair.
Telemetry association and proof that commanded perturbations were adopted are
external responsibilities. AOC does not infer those facts from two numbers.
Overflow of subtraction or scaling produces typed numerical failure.

A single SPSA direction is not a coordinate-wise finite-difference Jacobian.
Its components contain cross-coordinate terms; deterministic tests must use
the directional formula, not demand exact equality to an arbitrary full
gradient. Statistical claims require a declared independent Rademacher
ensemble. No convergence or noise-rejection claim follows from one pair.

## Adam: explicit state and a pure numerical step

Proposed method: `AdamStep`, with explicit α, β₁, β₂ and ε. Supplied input
contains q, ĝ and an `AdamState(m,v,t)` snapshot. A product contains candidate
coefficients and the complete next state; neither input arrays nor state are
mutated. The step minimizes the supplied objective:

```text
t′ = t + 1
m′ = β₁ m + (1 − β₁) ĝ
v′ = β₂ v + (1 − β₂) ĝ²
m̂ = m′ / (1 − β₁^t′)
v̂ = v′ / (1 − β₂^t′)
q_candidate = q − α m̂ / (sqrt(v̂) + ε)
```

Require matching nonempty dimensions, finite q/ĝ/m/v, componentwise v ≥ 0,
integer t ≥ 0, representable t+1, α > 0, ε > 0 and 0 ≤ β₁,β₂ < 1, all finite.
Fresh state has t0 and zero moments. Reject inconsistent nonzero moments at
t0. The caller must retain fixed β values over a history or explicitly reset
it; bias correction is not valid for an arbitrary time-varying β schedule.
Choose a stable evaluation of `1 − β^t` for β near1, including β0, and test it.

With homogeneous coefficient units, m has objective/coefficient units, v their
square, ε the gradient units, and α the coefficient units. Do not silently
mix mode normalizations or accept a volts-to-OPD conversion inferred from a
comment. The source's ε placement is outside the square root.

The next state is persistent scientific data, **not workspace**. The caller
decides whether the candidate and its associated next state enter the next
iteration. Rejected candidates must not leave a hidden updated history in the
plan or workspace. Changing coefficients independently requires an explicit
caller policy for retaining/resetting optimizer history. No `commit!`,
`discard!`, activation or transport method is added to AOC.

Workspace holds only replaceable intermediates. Recreating it must produce
the same step. `process!` result/workspace must not alias q, ĝ or input state
storage; reject unsupported aliasing before writes. Failure must not mutate
the supplied state or expose a valid-status candidate containing partial or
nonfinite values. Reused product storage needs an invalid status on numerical
failure so an earlier successful candidate cannot be mistaken for this step.

## Extraction limits and legacy policy

| Observed script behavior | AOC disposition |
| --- | --- |
| Modal-to-actuator multiplication using an external FITS matrix | Caller supplies coefficients in an explicit basis; command construction remains outside these methods. |
| Clamp to ±0.25, then subtract actuator mean | Do not extract. Mean subtraction can move values outside the previous clamp bounds; constraints require their own validated authority. |
| Poll5ms, discard2frames, live Aeron I/O, sleeps, best-flat file replacement | Equipment/campaign operations remain outside AOC. |
| Random Δ generated internally | Caller supplies the realized direction, enabling deterministic numerical processing. |
| Reduce c and multiply averaging count by1.025 after a lower objective | Exploratory policy, not SPSA definition; the noninteger averaging count must not become an AOC sample-count contract. |
| Two coordinate scans, then100 and1000 Adam iterations | Exploratory campaign; no stopping rule or convergence evidence is implied. |
| c0.02 then0.01, α0.001, 17modes then expanded basis | Historical settings only. No universal safe amplitude, physical conversion or convergence default follows. |
| Third metric evaluation after each step | Optional caller validation, separate from the two measurements defining the SPSA estimate. |

## Implementation and validation priorities

1. Implement and test the exact image metric first: hand-computed median
   fixtures, ties, replicated boundaries, tiny/rectangular ROIs, signed and
   zero images, strict radius boundary, cropped apertures, and brightness
   scaling. Test median-of-mean versus mean-of-metrics distinction explicitly.
2. Add pure probe generation and SPSA estimates: ±1 validation, equal-metric
   zero gradient, sign reversal, one-dimensional quadratic central derivative,
   multidimensional directional formula, overflow and lost perturbations.
3. Add pure Adam step with supplied state: closed-form first/second steps,
   ε placement, β0/near1, zero gradient with nonzero prior momentum, invalid
   moments/counters, overflow, input immutability and alias rejection. Reusing
   versus recreating workspace must not alter any numerical result.
4. Compare extracted operations against isolated numerical portions of the
   source on deterministic finite fixtures; do not execute its top-level
   equipment actions. Float64 accumulation may differ from the legacy Float32
   camera average, so document the input boundary and justified tolerances.
5. Require a separate application-level command/adoption/acquisition test
   before claiming closed-loop or instrumental improvement. There is no
   measured cadence, optimizer convergence, hardware safety, or sharpening
   performance evidence in this design review.

No new package dependency is required for the nine-value scalar median or
these equations. Accelerator paths and additional merit functions are deferred
until a measured use case establishes their contracts.

## Implemented scalar CPU slice (2026-10-02)

The four complete operations are implemented in `src/focal_plane_sharpening/`
and exported through the `FocalPlaneSharpening` namespace. They use the common
`prepare`, `process`, and `process!` interface. No dependency, physical command
transform, equipment action, or adaptive campaign policy was added.

`ImageSpecification` fixes ROI dimensions. `CoefficientSpecification` fixes
vector length; the caller supplies all vectors in the same declared basis and
coordinate order, with homogeneous coefficient units. Image and coefficient
products use Float64 numerical storage. Ordinary one-based scalar CPU arrays
and views are supported; no accelerator execution is claimed.

`WeightedCoreProduct` exposes `objective`, `center`, `peak_value`,
`aperture_count`, `cropped`, and `status`. Its workspace holds only the
replaceable filtered image. The implementation uses a fixed nine-scalar
insertion sorting network and explicit replicated row/column edge indices.

`SPSAProbeInputs` and `SPSAGradientInputs` carry complete supplied probe/metric
inputs. Directions must contain exact ±1, with Bool rejected. Both Float64
working precision and native coefficient precision are checked for collapsed
perturbations. `LostPerturbation` and `NumericalFailure` zero the entire
output. Native precision overflow also invalidates a probe. Float64 subtraction
overflow invalidates a gradient, while a representable subnormal metric
difference is preserved when scaling by `2c`.

`AdamInputs` supplies an `AdamState` snapshot with moments, UInt64 iteration,
and explicit beta history tags. Count overflow, negative second moments,
nonzero initial moments, and history tag mismatches are rejected before writes.
Bias correction uses `-expm1(t*log(beta))`, with beta zero handled explicitly.
The returned `AdamProduct.next_state` has independent moment storage. A
numerical failure zeros its candidate/moments and resets its output iteration;
the invalid product cannot serve as accepted history. Product reuse overwrites
that product's next-state storage, so the caller retains a separate product or
copies a snapshot if it must survive reuse. Plans and workspaces retain no
optimizer history. Unsupported output/workspace/input aliases are rejected
before writes. Recreating workspace reproduces the same supplied-state step.

Validation used Julia 1.12.7 on CPUs 6 and 14 with one OpenBLAS thread:

```text
OPENBLAS_NUM_THREADS=1 taskset -c 6,14 julia --startup-file=no --project=. \
    -e 'using Pkg; Pkg.test()'
```

The full package suite passed **1,866/1,866 assertions in 83 test sets**, including
**336 focal assertions** (199 metric, 52 SPSA, 85 Adam), all public documentation
checks, and method ambiguity detection. Equation references in
`test/focal_plane_sharpening.jl` are independently written: a dynamic sorted
nine-sample median reference, direct quadratic SPSA identities, and an Adam
reference with BigFloat bias correction. No legacy script text was copied and
no top-level equipment script was executed. The legacy source SHA256 above was
verified unchanged. These checks establish the specified numerical behavior.
Physical safety, convergence, scientific improvement, and execution cadence
remain unvalidated.

Independent review identified and verified correction of one intermediate
implementation defect (CM-R2): a high-precision coefficient rounded to its
Float64 baseline could produce identical plus/minus probes while native-only
comparisons appeared changed. The same regression failed before the working
precision check and passed afterward with `LostPerturbation`. Evidence logs
are preserved under `~/.cache/aoc-focal-sharpening-20261002/`:
`working-precision-fail-before.log`, `working-precision-pass-after.log`,
`focused-final.log`, and `pkg-test.log`. No further confirmed numerical or
ownership blocker was reported in the independent final source review.
