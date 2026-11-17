# Calibration methods: independent implementation review

## Scope and provenance

Review date: 2026-10-02. Reviewer owns this document only. Production edits
belong to the primary agent and method implementers.

- Spatial worktree: `AdaptiveOpticsCalibration-spatial-sinusoidal`, branch
  `calibration-spatial-sinusoidal`, baseline
  `3c11576fe65ca4998f1357d0210bb399d68e7b12`.
- Focal worktree: `AdaptiveOpticsCalibration-focal-sharpening`, same baseline.
  Its implementation was not present at the initial spatial review.
- Existing spatial draft changes comprise method source, focused tests,
  namespace/test registration, README and DESIGN updates. They were retained.
- Authority: repository AGENTS, DESIGN and the common `prepare` / `process`
  interface. Focal mathematical authority additionally includes
  `docs/DESIGN_SHARPENING.md` and its exact SPIDERS source hash.

This review covers scalar CPU numerical equations, representations, prepared
ownership and failure behavior. It does not qualify hardware, command limits,
closed-loop correction, optimizer convergence or physical interaction-matrix
rank. Spatial sinusoidal probing is independent of temporal modulation.

## Spatial basis: source conclusions

Observed equations match the declared contract: `sincospi(2 k⋅r)` implements
positive-phase sine/cosine in cycles per supplied coordinate unit. Raw rows
are sine then cosine per frequency. The existing modal operation constructs
positive/negative command rows in that order. Positions retain exact supplied
actuator ordering; no centering, piston subtraction, pupil selection or
orthogonalization is introduced.

The caller supplies both normalization policy and a strict raw sampled-peak
guard. DC and either sampled near-null row are rejected. `:peak` normalizes
each row independently; `:none` preserves samples. Duplicate frequencies are
accepted and explicitly tested without implying rank or independence.
Represented nonfinite phases/samples and entirely zero amplitude-scaled modes
are rejected. Checked extents prevent integer dimension-product overflow;
they do not establish a fixed memory budget.

Preparation copies caller arrays and delegates a copied command matrix to the
modal plan. Processing checks result aliasing against all prepared numerical
arrays before delegation. Explicit one-based requirements match the loops.
The mathematical basis and output ordering have no confirmed defect in this
pass. The raw RMS diagnostic has the precision defect below.

## CM-R1 — Raw RMS accumulation loses samples at accepted precision

- Severity: medium. Confidence: confirmed by source and execution.
- Affected code: `src/probe_bases/spatial_sinusoidal.jl`, raw
  `sampled_rms[mode] += abs2(sample)` accumulation and final division.
- Observation: the public promoted `AbstractFloat` contract accepts Float16.
  A running sum in Float16 stops accepting small contributions well before
  overflow. All inputs, basis values and the incorrect final RMS remain
  finite, so existing checks accept misleading diagnostics.
- Reproduction (fresh Julia, this worktree):

```julia
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.ProbeBases
p = hcat(fill(Float16(0.125), 4096), zeros(Float16, 4096))
f = Float16[1 0]
plan = prepare(SpatialSinusoidalPushPull(Float16[1, 1];
    normalization=:none, minimum_sampled_peak=1e-5),
    SpatialSinusoidalSpecification(p, f))
plan.sampled_rms
sqrt.(sum(abs2, Float64.(plan.basis); dims=2) ./ 4096)
```

- Observed: reported `Float16[0.5,0.5]`; recomputed RMS of the actual stored
  samples `[0.70703125,0.70703125]`. The command basis is unaffected by this
  specific diagnostic defect.
- Proposed remedy: accumulate squared raw samples in a sufficiently widened
  type, then convert final diagnostics to the declared result type if desired.
  Preserve arbitrary higher-precision input support when choosing the wider
  type. An explicit narrower supported-type contract is another possible
  adjudication, but silent Float16 acceptance is not.
- Required validation: fail-before/pass-after fixture above; ordinary Float32
  and Float64 RMS cases; large repeated-sample sums; verify command values and
  row ordering remain unchanged.
- Disposition: open, sent to primary agent before integration. No production
  change made by reviewer.

## Verification ledger

- Independently executed the CM-R1 reproduction: Julia exit0, incorrect finite
  diagnostics observed as recorded above.
- Reviewed focused spatial tests covering equations, translation/unit scaling,
  oblique frequencies, row/actuator ordering, explicit normalization/guard,
  Float32 and views, input copying, finite/shape/axis rejection, and redundant
  frequencies. Full package test evidence is pending the implementer.
- Focal implementation review pending source availability.
- No instrument, transport, campaign or equipment action was performed.

Initial spatial source SHA-256: `a3fb6862740658172df1d004373db4a65415b8a653eb0fb81f21589c464e5a00`.
Initial focused-test SHA-256: `7608711fbd664d68bd05b3c5bac809f35f4b60b53a286c1b57daebc00a1b3bd9`.

## Focal source review: first pass

The source now contains the proposed four complete operations. The metric
uses a replicated-edge nine-sample median, first column-major filtered maximum,
strict pixel-radius aperture, and an unnormalized negative weighted sum. It
retains signed finite intensities and reports cropped discrete apertures. The
source avoids equipment callbacks, RNG, acquisition policy and command clipping.

Adam consumes a supplied beta-tagged state and returns independent candidate
and next-state storage. It validates dimensions, finite input conversion,
nonnegative second moments, zero initial moments, matching beta history and
counter overflow before writes. Output/workspace alias checks cover every
input and other output buffer. Bias correction uses `expm1` with a beta-zero
branch. Numerical overflow yields a typed failed product with cleared output
state; no mutable history is retained by the plan or workspace. No confirmed
Adam or metric defect was identified in this source pass.

### CM-R2 — Float64 SPSA probe collapse can be accepted as valid

- Severity: medium. Confidence: confirmed source and execution.
- Affected code: focal `src/focal_plane_sharpening/spsa.jl`, probe validation.
- Reviewed source SHA-256:
  `766f2eaf4df45cf0425cb824f3a90fe331626d298056674121c1d4a18f5c44cd`.
- The input coefficient may be higher precision than the Float64 working
  representation. Comparing the converted probe only with the original
  coefficient misses collapse against its rounded working baseline.

```julia
plan = prepare(SPSAProbes(), CoefficientSpecification(1))
q = BigFloat[BigFloat(1) + eps(BigFloat)]
result = process(plan, SPSAProbeInputs(q, eps(Float64)/16, [1]))
# Observed: Valid, plus == [1.0], minus == [1.0].
```

- Remedy: require both finite working probes to differ from the Float64
  baseline, in addition to checking native-representation finiteness and
  unchanged perturbations. Identical probe pairs cannot be accepted as valid.
- Required validation: reproduction above returns `LostPerturbation` with
  cleared output; preserve ordinary Float32/Float64 and representable BigFloat
  probe behavior, plus native-overflow rejection.
- Disposition: open; primary agent and focal implementer notified.

Targeted execution loaded the focal module into the instantiated spatial
project via `Base.include(AdaptiveOpticsCalibration, focal_module_path)`; no
production files were changed. The first standalone focal load failed because
its dependency environment lacked SpecialFunctions; that environment failure
is separate from the reproduced method behavior. In the same targeted run,
current source correctly rejected native Float32 overflow, rejected overflowing
Float64 candidates with BigFloat direction values, and returned gradient0.5
for `c=Jplus=nextfloat(0.0), Jminus=0, Δ=[1]`. Earlier draft arithmetic in those
paths had already been corrected by the implementer before execution. They
are not remaining review defects. Full focal suite evidence remains pending.

### CM-R1 verification update

**Corrected; independently verified.** The source now uses
`D=promote_type(T,Float64)` for squared-sample accumulation and RMS storage,
while retaining T for sampled peaks, the basis and commands. The exact
fail-before reproduction now reports `[0.70703125,0.70703125]`, matching the
Float64 RMS of the stored basis, and commands remain Float16. This avoids
narrowing BigFloat inputs. Focused tests add the regression, Float32 diagnostic
widening, BigFloat128 preservation and abstract-element rejection. Implementer
reports89/89 focused assertions; independent full-suite log inspection remains
pending. This supersedes the initial open CM-R1 disposition.

### CM-R2 verification update

**Corrected; independently verified.** Probe validation now checks changes
against the Float64 working baseline and rejects collapsed pairs, retaining
native finiteness and unchanged-probe checks. The exact failing input now
returns `LostPerturbation`, plus/minus both zero. The same BigFloat baseline
with c0.25 returns `Valid`, plus1.25 and minus0.75. This supersedes the initial
open CM-R2 disposition.

The focused focal tests were independently inspected: an explicit sorted-window
metric reference covers replicated edges, tiny/signed/zero images, filtered
argmax ties, strict apertures and radiometric scaling; SPSA tests cover
directional rather than full-gradient semantics, lost working/native precision,
subnormal arithmetic and alias rejection; Adam tests cover first/second steps,
BigFloat bias-correction references, beta-history tags, stale workspace, input
immutability, aliases and numerical failure. No additional confirmed blocker
was found. Completed full-suite evidence is still required before integration.

## Branch verification evidence

- Spatial production commit: `d2c27a86eaf40668a4045e5c5edefe69a6d9094c`.
  Its six-file diff was inspected after commit, including the corrected
  diagnostic representation and concrete-element checks. CM-R1 remains closed.
- Spatial implementer reports the required `Pkg.test()` completed with exit0,
  89 focused spatial assertions,246 public-documentation checks and a clean
  ambiguity test. **No durable full-suite log was saved**; that full-run claim
  is implementer-reported from its tool session24434, whose earlier output was
  truncated. Independent CM-R1 execution and source/test inspection are
  distinguished above. The parent plans a combined integration suite.
- Focal focused log independently inspected:
  `focused.log`:197 metric,
  49 SPSA and44 Adam assertions, **290/290 passed**.
- Focal source SHA-256 at final source review:
  - `adam.jl`: `fe59a8dc0647a33eb04e35ba4e565429a3059472a6fff17ae78559031b29e29c`.
  - `focal_plane_sharpening.jl`: `25dc0f077acf971d223c08f478a439bdae0a4c898400b902f09504ac5387c2fe`.
  - `spsa.jl`: `3a2a65d58e65985046edcfd284b8125eab6afc76dff59a94c6aa691ab289509c`.
  - `weighted_core_intensity.jl`: `ec7aa33a5d20d37433f79adaf16894eef8327cbb7d343fbe260953ee8c670cf8`.
  - `test/focal_plane_sharpening.jl`: `61207c62396341aa219f62c336b9f7ebc8d63b258bc6aa27d5a16d7717b29e7f`.

No remaining confirmed numerical/ownership blocker is known at this source
revision. Final focal documentation/commit, its full package-suite result and
the parent-owned combined integration suite remain to be recorded.

## Final focal package verification

The completed `Pkg.test()` log was independently inspected at
`pkg-test.log`, SHA-256
`288508ecc0c5efcfd49d35cbf449b11c4ce94822162e4b47bef32e34f63243ee`. It ends
with `Testing AdaptiveOpticsCalibration tests passed`. Expanded focal coverage
passes **199 metric +52 SPSA +85 Adam =336 assertions**, alongside269 public
documentation checks, the ambiguity check and existing package/KA CPU tests.
The final production source hashes remain those recorded above. The expanded
focused test file SHA-256 is
`ff7662cda2cd5c84b8a625181f2f28b989ffba6c0553bdd67bf634e865bb5866`; added
all-slot alias and counter-edge tests were inspected.

The final README now checks each product status before consuming it in another
operation, especially the gradient before Adam. This closes a documentation
hazard in the earlier draft example, where a zeroed failed gradient could
otherwise feed a numerically valid Adam step. The method documentation retains
caller-owned measurement association, radiometry, units, constraints and
acceptance. No production numerical edits were made by the reviewer.

**Disposition:** both confirmed numerical findings, CM-R1 and CM-R2, are closed
with independently reproduced fail-before/pass-after behavior. No remaining
confirmed mathematical or ownership blocker was found. The two branches are
ready for parent-owned integration and the planned combined suite. These are
software numerical results, not hardware, optimizer convergence, physical
rank, cadence or correction qualification.

## Combined integration verification

Production merge `78abc2e` combines the reference-window, spatial and focal
methods without a merge conflict. The primary agent inspected the merged
production diff and ran the required complete `Pkg.test()` in the isolated
`AdaptiveOpticsCalibration-integration` worktree. The run passed
1,958 assertions across 89 test sets, including all three additions,
272 public documentation checks and the ambiguity check.

Log: `aoc-integration-tests.log`;
SHA-256 `82c098234363bdc6935220d7abb998f14cb0bea06238701235c0dca349785404`.
The tests used one BLAS thread and disabled Julia startup files. The
experimental limits stated above still apply.
