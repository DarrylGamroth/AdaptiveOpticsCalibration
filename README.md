# AdaptiveOpticsCalibration.jl

[![CI](https://github.com/DarrylGamroth/AdaptiveOpticsCalibration/actions/workflows/ci.yml/badge.svg)](https://github.com/DarrylGamroth/AdaptiveOpticsCalibration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DarrylGamroth/AdaptiveOpticsCalibration/branch/main/graph/badge.svg)](https://codecov.io/gh/DarrylGamroth/AdaptiveOpticsCalibration)

[Algorithm references](docs/REFERENCES.md) · [Release provenance](docs/RELEASE_PROVENANCE.md)

## Installation

Add the public release directly from Julia's package manager.

```julia
using Pkg
Pkg.add(url="https://github.com/DarrylGamroth/AdaptiveOpticsCalibration.git",
        rev="v0.17.7")
```

With the [DarrylGamrothRegistry](https://github.com/DarrylGamroth/PackageRegistry)
installed, use `Pkg.add("AdaptiveOpticsCalibration")`.

`AdaptiveOpticsCalibration.jl` is a transport-independent numerical toolbox
for adaptive-optics calibration and commissioning.

The package supplies numerical methods shared by simulation, soft-real-time
calibration workers, commissioning workflows, and bounded RTC adapters.

The [soft-real-time calibration guide](docs/SOFT_REALTIME_CALIBRATION.md)
lists the CPU methods, instrument worker entry points, and replay evidence.

## Interface

The routine interface uses three generic functions with multiple dispatch:

```julia
plan = prepare(method, specification)
result = process(plan, inputs)
process!(result, plan, inputs)
```

The method selects the equations and scientific policy. The specification,
array types, and any explicitly supported execution target select compatible
implementations. A plan is run-immutable, workspace is replaceable scratch,
and the result is a caller-visible calibration product.

`process(plan, inputs)` allocates a compatible result and workspace.
`process!(result, plan, inputs)` allocates workspace scratch; retain a
workspace and use the four-argument form for repeatable scratch, allocation
control, or KernelAbstractions execution. `allocate_result` and
`allocate_workspace` are advanced public, non-exported extension seams.

```julia
result = AdaptiveOpticsCalibration.allocate_result(plan)
workspace = AdaptiveOpticsCalibration.allocate_workspace(plan)
process!(result, workspace, plan, inputs)
```

Supported data-parallel operations can instead be prepared for an explicit
KernelAbstractions backend:

```julia
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.InteractionMatrices
using KernelAbstractions

method = ZonalPushPull(Float32[0.5])
specification = ResponseSpecification(2)
responses = Float32[2 4; 0 2]
execution = KernelExecution(specification, CPU())
plan = prepare(method, execution)
result = process(plan, responses)
```

The same form accepts a loaded CUDA or AMDGPU backend. Array-valued scientific
storage and prepared coefficients remain on that backend. `process!` is
synchronous: device work is complete when it returns. Compact validity or
diagnostic summaries may cross to host memory.

Concrete algorithm families use domain modules such as `Reconstructors`,
`InteractionMatrices`, `PhaseRetrieval`, `DOCRIME`, `Diagnostics`, and
`ProbeBases`. The module supplies the namespace; algorithms do not acquire
parallel, algorithm-qualified preparation or execution entry points.

See [DESIGN.md](DESIGN.md) for the ownership contract and
[MIGRATION.md](MIGRATION.md) for the source inventory and migration gates.

For example, a covariance-form MAP reconstructor uses the same prepared
operation shape as every other family:

```julia
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.Reconstructors

method = MAPReconstructor(prior_weight=1.0)
specification = ReconstructorSpecification(size(D)...)
plan = prepare(method, specification)
inputs = MAPReconstructorInputs(D, noise_covariance, command_prior_covariance)
result = process(plan, inputs)

R = reconstructor(result)
```

`D` is ordered measurement by corrector coordinate, while `R` is ordered
corrector by measurement coordinate.

Spatial sine/cosine DM probes accept explicit physical actuator positions and
spatial frequency vectors, with no inferred geometry or pupil mask:

```julia
using AdaptiveOpticsCalibration.ProbeBases

method = SpatialSinusoidalPushPull(amplitudes_per_sine_cosine_mode;
    normalization=:peak, minimum_sampled_peak=sampled_peak_guard)
specification = SpatialSinusoidalSpecification(actuator_positions, spatial_frequencies)
plan = prepare(method, specification)
commands = probe_commands(process(plan, nothing))
```

Positions are `(N, 2)` in the required actuator order; frequencies are `(K, 2)`
in cycles per matching coordinate unit. The `2K` positive amplitudes follow
`(sine₁, cosine₁, sine₂, cosine₂, …)`; command rows follow
`(+sine₁, −sine₁, +cosine₁, −cosine₁, …)`. `:none` preserves raw sinusoidal
samples and `:peak` divides each mode by its sampled maximum absolute value.
The required dimensionless guard rejects DC and sampled near-null modes,
without claiming independence across frequencies. `plan.basis`,
`plan.sampled_peak`, and `plan.sampled_rms` expose the normalized modes and raw
sampled diagnostics. These are spatial modes; a partial Fourier probe set
calibrates modal responses and does not replace complete zonal identification.

Complete-image optical gains use the same prepared operation shape. The CPU
reference accepts a finite complex focal-plane mask, an already aligned real
modal basis, and complete reference/current detector frames. It preserves the AOS
centered-FFT convention and reports signed modal ratios, complex
sensitivities, weak-mode flags, and flux diagnostics:

```julia
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.OpticalGains

specification = GainSensingSpecification(mask, modal_basis, reference_frame)
plan = prepare(GainSensing(), specification)
product = process(plan, GainSensingInputs(current_frame))
gains = optical_gains(product)
```

With the optional KernelAbstractions and CUDA or AMDGPU extensions loaded,
`prepare(GainSensing(), KernelExecution(specification, backend))` prepares
device-resident mask, basis, and reference products. Pass a device-resident
current frame to `GainSensingInputs` and retain the result and workspace for
repeated `process!` calls. Gain, sensitivity, and weak-mode arrays remain on
the selected device; compact flux diagnostics are returned on the host.
Processing synchronizes and checks finite inputs before publishing a result.
This calibration-time path does not claim CUDA Graph or HIP Graph capture, or
zero host allocation during direct execution.

## Current status

Pure focal-plane sharpening calculations use `FocalPlaneSharpening`:

```julia
using AdaptiveOpticsCalibration.FocalPlaneSharpening

metric_plan = prepare(WeightedCoreIntensity(7), ImageSpecification(rows, columns))
metric = process(metric_plan, image)
metric.status == FocalPlaneSharpening.Valid || error("invalid image calculation")

spec = CoefficientSpecification(length(q)) # caller's basis/order and homogeneous units
probe_plan = prepare(SPSAProbes(), spec)
pair = process(probe_plan, SPSAProbeInputs(q, c, direction)) # supplied exact ±1
pair.status == FocalPlaneSharpening.Valid || error("invalid probe pair")
# The caller supplies Jplus/Jminus associated with the respective probes.
gradient_plan = prepare(SPSAGradient(), spec)
gradient = process(gradient_plan, SPSAGradientInputs(direction, c, Jplus, Jminus))
gradient.status == FocalPlaneSharpening.Valid || error("invalid gradient")

adam_plan = prepare(AdamStep(alpha, beta1, beta2, epsilon), spec)
state = AdamState(zeros(length(q)), zeros(length(q)), 0; beta1, beta2)
step = process(adam_plan, AdamInputs(q, gradient.gradient, state))
step.status == FocalPlaneSharpening.Valid || error("invalid Adam calculation")
# The caller owns acceptance and may retain step.next_state after accepting.
```

The image objective is the negative weighted sum within strict `r < radius`
of a replicated-edge 3×3 median image, centered on its first column-major
maximum. Weights are `1/(1+r)^2` and radius is explicit detector pixels.
Intensity units, sign, background, and brightness are unchanged. The product
includes center, filtered peak, aperture count, and whether the discrete
aperture is cropped by the ROI. It is not a normalized Strehl metric.

Probe and gradient calculations use the supplied SPSA pair order; a single
direction includes cross terms. Adam uses supplied moment history with fixed
beta tags, stable bias correction, and ε outside the square root. Processing
never mutates inputs or advances supplied state. Products use Float64;
`NumericalFailure` and `LostPerturbation` invalidate and zero outputs. SPSA
checks perturbations in both Float64 working and native coefficient precision;
the caller also checks any later command conversion. Structural/nonfinite
input errors and unsupported aliases throw before writes. These scalar CPU
operations support ordinary one-based arrays and views. They do not own random
directions, image acquisition/averaging, clipping, constraints, activation, or
convergence policy. See [the sharpening design](docs/DESIGN_SHARPENING.md) for
equations, units, source attribution, and validation limits.

Complete detector reference batches use `ReferenceFrames`:

```julia
using AdaptiveOpticsCalibration.ReferenceFrames

dark_plan = prepare(DarkFrameMoments(), DarkFrameSpecification(rows, columns, N))
dark = process(dark_plan, raw_dark_adc) # UInt16 row × column × exposure
# dark.mean: ADC counts; dark.variance: unbiased squared ADC counts (N - 1)

lamp_plan = prepare(LampReference(minimum_flux_per_roi, adc_upper_rail),
    LampReferenceSpecification(rows, columns, roi_count, N))
lamp_inputs = LampReferenceInputs(absolute_centroids, flux, intrinsic_valid,
    raw_lamp_adc; centroids_are_absolute=true)
lamp = process(lamp_plan, lamp_inputs)
```

Lamp inputs preserve the deployed estimator's interleaved x/y centroid rows in
detector pixels, two rows per ROI. Absolute coordinates require acquisition
with zero subtraction reference, or adding the explicitly known deployed
reference back to each residual sample. Every ROI retains minimum-flux and
all-training-valid diagnostics. Eligibility is per ROI; no fixed eligible
count is required. An ADC value at or above the declared upper rail, or a
non-finite input/calculation, produces `ReferenceFrames.Invalid` for the whole
lamp batch. Inactive reference rows are zero; a finite batch with no eligible
ROI produces `ReferenceFrames.NoEligible`. The scalar CPU implementations use
one-based arrays. Acquisition, association, acceptance,
storage, and activation remain with the composing application.

For arriving samples, retain explicit scientific state instead of a frame cube:

```julia
dark_state = DarkFrameState(dark_plan)
process!(dark_state, dark_plan, raw_dark_frame) # unsigned row × column matrix
# Repeat for exactly N exposures, or supply bounded row × column × exposure chunks.
dark = process(dark_plan, dark_state)

lamp_state = LampReferenceState(lamp_plan)
process!(lamp_state, lamp_plan, lamp_chunk) # LampReferenceInputs, one or more exposures
# Repeat for exactly N exposures; centroids must remain absolute detector pixels.
lamp = process(lamp_plan, lamp_state)
```

Pyramid references consume ordered successful FGA reconstruction outputs after
orchestration has discarded priming exposures:

```julia
pyramid_plan = prepare(PyramidReference(adc_upper_rail),
    PyramidReferenceSpecification(rows, columns, selected_pixel_count, N,
        reconstructor; row_units=modal_coordinate_units))
pyramid_inputs = PyramidReferenceInputs(normalized_pixels, current_pupil_means,
    raw_frames) # (4, K, N), (N,), (rows, columns, N)
pyramid = process(pyramid_plan, pyramid_inputs)
state = PyramidReferenceState(pyramid_plan)
process!(state, pyramid_plan, pyramid_chunk) # same shapes with one or more exposures
pyramid = process(pyramid_plan, state) # only when count == N
```

The dimensionless pixel reference is an unweighted Float64 mean; the modal
reference is `−R * vec(permutedims(reference_pixels))` in the explicitly
configured row units. AOC preserves the input normalization: FGA uses the
previous successful frame's full-pupil mean, while the supplied diagnostic
means describe current frames, including masked pixels. Nonfinite pixels or
calculations, nonpositive/nonfinite diagnostic means, or raw ADC at/above the
rail invalidate the whole product. Diagnostics retain the ADC maximum and
minimum current mean. The plan owns one Float64 R copy, shared by state; state
requires that exact plan, retains no raw history and resets by construction.

States own their numerical arrays and retain no input arrays. Dark and lamp
states copy their prepared configuration; pyramid state shares its exact plan.
Every update uses the global exposure index, preserving batch results for ordered
partitions. Shape, alias, plan mismatch and overrun errors
throw before writes; final extraction rejects incomplete state. Numerical
invalidity remains sticky while later exposures complete all diagnostics.
Extraction leaves state unchanged. Construct fresh state to reset; independent
state merging is unsupported. Source settings, sample association and
acceptance remain caller-owned. These APIs are scalar CPU only.

The package currently provides scalar CPU reference implementations for the
following families, with additional accelerator targets noted below:

- covariance-form MAP and stroke-metric Tikhonov reconstructors;
- interaction-matrix estimation from complete zonal and balanced Hadamard
  response cycles;
- complete modal, zonal, balanced Hadamard and spatial sine/cosine probe command bases;
- weighted core image intensity, SPSA probe/gradient operations, explicit-state Adam steps and quadratic modal scan fits;
- sampled influence-function eigenbases, finite-corrector atmospheric
  Karhunen–Loève bases, pure modal fitting diagnostics, and explicit modal OPD
  expansion;
- LiFT nonlinear phase retrieval over a typed, package-neutral physical-model
  contract;
- interaction-matrix stability diagnostics; and
- complete-batch and explicit-state dark mean/unbiased variance, lamp centroid
  references and normalized pyramid pixel/modal references;
- incremental centered DO-CRIME covariance and open- and closed-loop solves;
- complete-window temporal dither gain and phase estimation;
- one-sided modal power spectra and delay-aware integrator gains with predicted
  residual variance and gain margin
  ([held-out replay](docs/SOFT_REALTIME_CALIBRATION.md#temporal-gain-selection-on-held-out-modal-opd))
- complete-image gain sensing with a CPU reference and CUDA/AMDGPU execution
  against frozen AOS parity; and
- CPU meta-sensitivity misregistration estimation, explicit-coordinate
  von Kármán tomography covariance assembly with an optional KA backend,
  and CPU tomography covariance reconstruction.

For `Cxx` or `Cox` on a KA backend, pass already transformed, one-based host
coordinates and host layer weights through
`KernelExecution(specification, backend)`. Preparation validates and copies
them into owned backend storage. The covariance product stays on that backend;
`process!` synchronizes and publishes it only after a finite-candidate check.
The CPU path uses `SpecialFunctions.besselk` as its reference, while the KA
path uses the AOS scalar approximation from revision `b4d2da2`. This path
does not derive optical geometry, perform fit-source weighting or a GPU solve,
or claim graph capture or zero allocation.

Soft-real-time callers can update reference and covariance state on associated
samples or process fixed observation windows on a calibration worker. Use an
independent workspace and result for each concurrent operation. PipeWireAORTC
or the composing SRT owns exposure association, scheduling and activation.
See [soft-real-time calibration](docs/SOFT_REALTIME_CALIBRATION.md) for usage,
accuracy comparisons and qualification work.

An optional KernelAbstractions extension provides backend execution for
structured interaction-matrix estimation, probe-basis construction,
interaction-matrix stability, and complete-image gain sensing.
It has been exercised with scalar indexing disabled on AMDGPU and CUDA
hardware. Reconstructor and DO-CRIME preparation continue to use the scalar
CPU implementation; dense factorization should eventually use a vendor linear
algebra library rather than a custom elementwise kernel. CUDA Graph and HIP
Graph capture are intentionally outside this calibration package.

The [calibration catalogue](docs/CALIBRATION_CATALOGUE.md) maps implemented
methods to pyRTC, SPECULA, MagAO-X and SPIDERS references. The
[numerical review](docs/CALIBRATION_METHODS_REVIEW.md) records the spatial and
sharpening contracts, corrected defects and validation limits.

The CUDA and AMDGPU test entry points accept one domain selector. Without an
argument, they run the full accelerator suite. For example,
`julia --startup-file=no --project=test/cuda test/cuda/runtests.jl tomography`
runs both tomography covariance testsets. Available selectors are `tomography`, `gain`, `structured`,
`probes`, `stability`, and `lift`.
