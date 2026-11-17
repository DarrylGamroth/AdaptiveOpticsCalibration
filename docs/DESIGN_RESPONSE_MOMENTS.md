# Repeated response moments

This is a pure numerical diagnostic for complete, equally weighted response
vectors. It adds no acquisition, exposure association, mask selection, frame
aggregation, artifact writer, acceptance policy or covariance inversion.
The scalar CPU implementation and fixtures are independently written from
the equations below. No external implementation or translated fixture is
used; no dependency or numerical backend is added.

## Public contract

```julia
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.Diagnostics

plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 4))
result = process(plan, [1.0 3 5 7; 2 4 6 8])
# result.mean == [4.0, 5.0]
# result.sample_variance ≈ [20/3, 20/3]
# result.mean_standard_error ≈ [sqrt(5/3), sqrt(5/3)]
```

Input is an explicit real measurement × repeat matrix, with positive
measurement extent, at least two repeats and one-based axes. Ordinary strided
views are supported. Each repeat can be a response vector, an already
averaged response batch, a central-difference estimate, or a caller-composed
contrast. All columns must share coordinates, homogeneous units and the same
scientific meaning. The repeat count is not the number of underlying detector
exposures. No hidden scaling by an exposure count is performed.

Products and scratch are concrete parametric containers of Float64 arrays.
Default allocation uses ordinary vectors and matrices. Compatible caller-owned
views can also be supplied. The common allocating `process` and output-first
three/four-argument `process!` interfaces apply. Plans retain dimensions only;
scratch is overwritten or reset before use and retains no scientific history. Input and all
seven writable product/scratch arrays must have one-based axes and must not
alias. Conservative `Base.mightalias` checks can reject shared-parent views.

Invalid shape, axes or aliases throw before changing the product. An original
non-finite input publishes `ResponseMomentsNonFiniteInput`. Float64 conversion
or arithmetic overflow from finite input publishes
`ResponseMomentsNonFiniteCalculation`. Both outcomes zero every product array;
a later call with valid inputs remains usable. Newly allocated products have
`ResponseMomentsNotProcessed`; successful calculations have
`ResponseMomentsValid`. These statuses describe numerical completion, not
scientific acceptance or verified statistical independence.

## Equations and units

For responses xᵣ ∈ ℝᴹ and R repeats:

```text
μ          = (1/R) ∑ᵣ xᵣ
sⱼ²        = (1/(R−1)) ∑ᵣ (xⱼᵣ − μⱼ)²
SEⱼ        = √(sⱼ²/R)
F[:,r]     = (xᵣ − μ)/√(R−1)
Σ̂          = F Fᵀ
```

The implementation uses Float64 Welford updates in fixed-origin coordinates.
Set o to the first converted column and zᵣ = xᵣ − o. Initialize the offset
mean ν₁ = 0, then use δ = zᵣ − νᵣ₋₁, νᵣ = νᵣ₋₁ + δ/r, and
M₂ᵣ = M₂ᵣ₋₁ + δ(zᵣ − νᵣ). Variance is M₂/(R−1), factor columns are
(zᵣ − νᴿ)/√(R−1), and returned mean is the rounded Float64 sum o + νᴿ.
Factor columns retain the original repeat order. Centering in offset
coordinates preserves small represented differences even if the absolute
mean rounds; re-subtracting the rounded returned mean is not equivalent.
Every input is converted individually to Float64; wider input precision is
not retained. Opposite extreme finite inputs can overflow their fixed-origin
subtraction and return the typed non-finite-calculation outcome.

For represented samples `[1e16+2, 1e16, 1e16, 1e16]`, the exact mean is
1e16 + 0.5, which rounds to 1e16 in Float64. The unbiased variance is 1 and
mean standard error is 0.5. Offset centering preserves these values and the
factor energy 1, while using the rounded absolute mean to center the samples
would give factor energy 4/3. This case is checked against an independently
calculated BigFloat scalar oracle and every distinct ordering of the samples.

Mean, standard error and factor entries have the input unit; sample variance
has its square. For centroid vectors this can be detector pixels. For a
central difference it can be pixels per explicitly declared command unit.
Interleaved x/y rows and inactive positions remain in the supplied row order;
the diagnostic does not select or interpret them.

The covariance representation is compact M × R rather than an asserted
invertible M × M matrix. In exact arithmetic its rank is at most R−1, and can
be smaller. Rounded centering can leave a small nonzero column sum; this factor
is not a numerical rank or independence certificate. Welford variance and
the factor diagonal agree up to floating-point rounding. It is inappropriate
to invert a high-dimensional empirical covariance from four repeats.

## Statistical interpretation

Unbiased sample variance and the standard-error interpretation require
independent repeats with a common mean and covariance, and finite second
moments. Formula evaluation does not establish these assumptions. Drift,
replayed detector seeds, overlapping frame windows, or cross-repeat
correlation can invalidate that interpretation. A normal confidence interval
also needs distributional assumptions and a declared confidence level. No
confidence interval or significance threshold is calculated here.
The [NIST mean-interval description](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)
uses a Student t multiplier with R−1 degrees of freedom, rather than treating
an estimated standard error as a confidence bound. NIST also describes
[normal-population standard-deviation intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm)
based on the chi-square distribution.

R = 4 supplies only three covariance degrees of freedom. Apparent amplitude
dependence should be compared with independent repeat uncertainty; it is not
established by a large response norm or repeat cosine alone. Correlations among
measurement rows are represented by F, rather than an assumption that all
rows are independent.

## Composition with push-pull and held-out measurements

Use the existing public `InteractionMatrices.ZonalPushPull` for each complete
signed response pair. Its response input is probe × measurement, ordered
positive then negative. For symmetric adopted amplitudes ±a it computes
dᵣ = (ȳ₊ᵣ − ȳ₋ᵣ)/(2a). Supply the actual represented amplitude in matching
command units; converting an adopted Float32 value to Float64 does not restore
the unrounded decimal request. Association, unclipped commands, validity,
settling, stable reference subtraction and mask consistency remain caller-owned.
Place the resulting derivative vectors in measurement × repeat order for
this diagnostic. It does not reacquire or re-estimate the central difference.

For a batch of N chronological frames, the exact variance-of-mean identity is
Cov(ȳ) = (1/N²) ∑ₙ,ₘ Cov(yₙ,yₘ). Only independent, identically distributed
frames reduce this to Σ/N. For equal independent positive/negative batches,
Cov(dᵣ) = Σ/(2Na²). In general it is
(Cov(ȳ₊) + Cov(ȳ₋) − Cov(ȳ₊,ȳ₋) − Cov(ȳ₋,ȳ₊))/(4a²).
Increasing N therefore does not imply a universal 1/√N improvement.

Linearity can be inspected with explicit amplitude contrasts between repeat
derivatives. Matched repeat contrasts preserve any cross-amplitude covariance;
unmatched independent means require both estimates' uncertainty. Symmetric
central differences cancel even response terms, so also retain the contrast
(ȳ₊ + ȳ₋)/2 − ȳ₀ when checking nonlinear response. A null response and any
shared reference introduce their own uncertainty. This method summarizes
supplied contrasts without fitting, selecting amplitudes or choosing thresholds.

For an independently fitted response D̂ and held-out command u, a prediction
residual ȳ_test − b̂ − D̂u includes held-out detector/WFS noise, reference
uncertainty when applicable, and interaction-matrix estimation uncertainty.
Treating D̂ as fixed omits its calibration uncertainty. For independent
column estimates the latter term is ∑ⱼ uⱼ² Cov(D̂[:,j]); shared measurements or
reference estimates add cross terms. Paired held-out signed differences can
remove a constant subtraction reference. They do not establish independence
or remove detector/WFS noise.

Inspect ordered individual null frames, disjoint batch means at the declared
averaging counts, sign order and chronological repeat order before interpreting
mean standard errors. Fixed coordinate projections belong to the composing
application. A diagnostic over two selected actuators does not qualify the
other columns of a full physical interaction matrix or its controller basis.
