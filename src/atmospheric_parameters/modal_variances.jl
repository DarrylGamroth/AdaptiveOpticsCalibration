# Winker (1991) motivates outer-scale fits from modal variances. See docs/REFERENCES.md.
"""
    ModalVarianceFit(; relative_rms_tolerance=0)

Fit the Fried parameter `r₀` and outer scale `L₀` by profiling the positive
turbulence amplitude over the supplied `L₀` grid. `relative_rms_tolerance` is
the allowable increase above the minimum weighted relative RMS for reporting
the supported `L₀` range. It is a model-discrepancy tolerance, in units of
fractional RMS, and has no statistical coverage interpretation.
"""
struct ModalVarianceFit <: AbstractCalibrationMethod
    relative_rms_tolerance::Float64

    function ModalVarianceFit(; relative_rms_tolerance::Real=0)
        tolerance = Float64(relative_rms_tolerance)
        0 <= tolerance <= 1 || throw(ArgumentError(
            "relative_rms_tolerance must be in [0, 1]",
        ))
        return new(tolerance)
    end
end

"""
    ModalVarianceSpecification(L0_candidates_m, reference_variance_rad2;
        reference_r0_m, reference_wavelength_m, weights=ones(size(reference_variance_rad2, 1)))

`reference_variance_rad2[mode, candidate]` is the positive phase variance of
each selected mode for the specified `L₀` and at a common `reference_r0_m` and
`reference_wavelength_m`. All lengths and wavelengths are in m. Supply
strictly increasing positive `L0_candidates_m` with optional `Inf` for a
Kolmogorov template. One candidate holds `L₀` fixed and permits one or more
modes. Searching `L₀` requires at least two modes with distinct sensitivity
to outer scale to give an identifiable fit. Weights are strictly positive
and dimensionless. Exclude a mode by omitting its row from both specification
and observation.

The caller supplies reference variances for the exact selected basis, aperture,
normalization, and transfer function in the observation, in the same mode
order. The template may be an analytic turbulence prediction or independently
calibrated through a linear sensor and reconstruction model. Preparation copies
the one-based CPU reference arrays into owned Float64 storage and converts
the template to OPD variance with `(reference_wavelength_m / 2π)²`.
"""
struct ModalVarianceSpecification{L<:AbstractVector,S<:AbstractMatrix,W<:AbstractVector}
    L0_candidates_m::L
    reference_variance_rad2::S
    reference_r0_m::Float64
    reference_wavelength_m::Float64
    weights::W

    function ModalVarianceSpecification(
        L0_candidates_m::L,
        reference_variance_rad2::S;
        reference_r0_m::Real,
        reference_wavelength_m::Real,
        weights::W=ones(size(reference_variance_rad2, 1)),
    ) where {L<:AbstractVector{<:Real},S<:AbstractMatrix{<:Real},W<:AbstractVector{<:Real}}
        return new{L,S,W}(L0_candidates_m, reference_variance_rad2,
            Float64(reference_r0_m), Float64(reference_wavelength_m), weights)
    end
end

"""
    ModalVarianceInputs(observed_variance_m2, noise_variance_m2)

Observed and additive measurement-noise variances in m² OPD, in the selected
mode order. Each observed variance must exceed its finite nonnegative noise
variance. Noise must be uncorrelated with the turbulence. The caller supplies
complete open-loop or calibrated pseudo-open-loop modal variances and corrects
vibration, residual loop filtering, nonlinearity, and unsensed modes before
forming them. Use zero noise variances for data explicitly corrected for noise
or a noiseless input. Use one-based CPU arrays or views.
"""
struct ModalVarianceInputs{V<:AbstractVector{<:Real},N<:AbstractVector{<:Real}}
    observed_variance_m2::V
    noise_variance_m2::N
end

"""
Prepared OPD reference templates and fit weights. Reentrant with separate
workspace and product storage for each invocation.
"""
struct ModalVariancePlan <: AbstractCalibrationPlan
    L0_candidates_m::Vector{Float64}
    reference_variance_m2::Matrix{Float64}
    reference_r0_m::Float64
    weights::Vector{Float64}
    relative_rms_tolerance::Float64
end

"""
Caller-owned Fried parameter `r0_m` at the reference wavelength and outer
scale `L0_m`, both in m. `fitted_variance_m2` contains fitted turbulence-only
OPD variances in input mode order. `relative_rms_by_L0` and `r0_by_L0_m`
retain the minimized score and associated Fried parameter for every candidate.

`L0_support_m` is the minimum and maximum grid value within the declared
score tolerance, including numerical ties. A range reaching a search
endpoint leaves that side unresolved by the grid. Inspect the score profile
for disconnected support. `L0_at_grid_boundary` identifies an endpoint
optimum of a multi-candidate search. A fixed-`L₀` fit has singleton support
and a false boundary flag.
"""
mutable struct ModalVarianceProduct
    r0_m::Float64
    L0_m::Float64
    L0_support_m::Tuple{Float64,Float64}
    L0_at_grid_boundary::Bool
    relative_rms::Float64
    fitted_variance_m2::Vector{Float64}
    relative_rms_by_L0::Vector{Float64}
    r0_by_L0_m::Vector{Float64}
end

struct ModalVarianceWorkspace
    turbulence_variance_m2::Vector{Float64}
    ratios::Vector{Float64}
    relative_rms_by_L0::Vector{Float64}
    r0_by_L0_m::Vector{Float64}
end

function prepare(method::ModalVarianceFit, specification::ModalVarianceSpecification)
    (; L0_candidates_m, reference_variance_rad2, reference_r0_m,
        reference_wavelength_m, weights) = specification
    _require_cpu_arrays(L0_candidates_m, reference_variance_rad2, weights)
    n_modes, n_candidates = size(reference_variance_rad2)
    n_modes > 0 && n_candidates > 0 || throw(ArgumentError(
        "the reference must contain at least one mode and one L0 candidate",
    ))
    n_candidates == length(L0_candidates_m) && n_modes == length(weights) || throw(
        DimensionMismatch("reference variance axes must match the mode weights and L0 candidates"),
    )
    n_candidates > 1 && n_modes < 2 && throw(ArgumentError(
        "an L0 search requires at least two modes",
    ))
    isfinite(reference_r0_m) && reference_r0_m > 0 || throw(ArgumentError(
        "reference_r0_m must be finite and positive",
    ))
    isfinite(reference_wavelength_m) && reference_wavelength_m > 0 || throw(ArgumentError(
        "reference_wavelength_m must be finite and positive",
    ))
    candidates = Vector{Float64}(L0_candidates_m)
    all(>(0), candidates) && all(i -> candidates[i] > candidates[i - 1], 2:n_candidates) ||
        throw(ArgumentError("L0 candidates must be strictly increasing and positive"))
    opd_variance = Matrix{Float64}(reference_variance_rad2)
    opd_variance .*= (reference_wavelength_m / (2π))^2
    all(x -> isfinite(x) && x > 0, opd_variance) || throw(ArgumentError(
        "converted reference variances must be finite and positive",
    ))
    normalized_weights = Vector{Float64}(weights)
    all(x -> isfinite(x) && x > 0, normalized_weights) || throw(ArgumentError(
        "mode weights must be finite and positive",
    ))
    normalized_weights ./= maximum(normalized_weights)
    normalized_weights ./= sum(normalized_weights)
    all(>(0), normalized_weights) || throw(ArgumentError(
        "normalized mode weights must be representable",
    ))
    return ModalVariancePlan(candidates, opd_variance, reference_r0_m,
        normalized_weights, method.relative_rms_tolerance)
end

function allocate_result(plan::ModalVariancePlan)
    n_modes, n_candidates = size(plan.reference_variance_m2)
    return ModalVarianceProduct(NaN, NaN, (NaN, NaN), false, NaN,
        zeros(n_modes), zeros(n_candidates), zeros(n_candidates))
end

function allocate_workspace(plan::ModalVariancePlan)
    n_modes, n_candidates = size(plan.reference_variance_m2)
    return ModalVarianceWorkspace(
        zeros(n_modes), zeros(n_modes), zeros(n_candidates), zeros(n_candidates),
    )
end

function _validate_storage(result, workspace, plan, inputs)
    (; observed_variance_m2, noise_variance_m2) = inputs
    _require_cpu_arrays(observed_variance_m2, noise_variance_m2)
    n_modes, n_candidates = size(plan.reference_variance_m2)
    mode_arrays = (observed_variance_m2, noise_variance_m2, result.fitted_variance_m2,
        workspace.turbulence_variance_m2, workspace.ratios)
    profile_arrays = (result.relative_rms_by_L0, result.r0_by_L0_m,
        workspace.relative_rms_by_L0, workspace.r0_by_L0_m)
    all(a -> length(a) == n_modes, mode_arrays) &&
        all(a -> length(a) == n_candidates, profile_arrays) || throw(
        DimensionMismatch("fit storage does not match the prepared modes and L0 candidates"),
    )

    readonly = (observed_variance_m2, noise_variance_m2, plan.L0_candidates_m,
        plan.reference_variance_m2, plan.weights)
    writable = (mode_arrays[3:end]..., profile_arrays...)
    for i in eachindex(writable)
        a = writable[i]
        (any(b -> Base.mightalias(a, b), readonly) ||
            any(j -> Base.mightalias(a, writable[j]), 1:i-1)) && throw(ArgumentError(
            "modal fit writable arrays must be disjoint from other fit storage",
        ))
    end
    return nothing
end

# Profile a scalar amplitude, reducing ratios by their peak to keep the dot
# products representable even when the reference and observation scales differ.
function _profile!(ratios, turbulence, plan, candidate)
    for i in eachindex(ratios)
        ratios[i] = plan.reference_variance_m2[i, candidate] / turbulence[i]
    end
    peak = maximum(ratios)
    isfinite(peak) && all(>(0), ratios) || throw(ArgumentError(
        "reference-to-observation variance ratios must be representable",
    ))
    numerator, denominator = 0.0, 0.0
    for i in eachindex(ratios)
        ratios[i] /= peak
        weighted_ratio = plan.weights[i] * ratios[i]
        numerator += weighted_ratio
        denominator += weighted_ratio * ratios[i]
    end
    amplitude = numerator / denominator
    score = 0.0
    for i in eachindex(ratios)
        ratios[i] *= amplitude
        score += plan.weights[i] * abs2(1 - ratios[i])
    end
    r0 = plan.reference_r0_m * (peak / amplitude)^(3 / 5)
    isfinite(r0) && r0 > 0 && isfinite(score) || throw(ArgumentError(
        "profiled amplitude or score must be representable",
    ))
    return (r0, sqrt(score))
end

"""
    process!(result, workspace, plan::ModalVariancePlan, inputs::ModalVarianceInputs)

With `yᵢ = observedᵢ − noiseᵢ` and `vᵢ(L₀)` the prepared reference OPD
variance, minimize `Σᵢ wᵢ(1 − a vᵢ(L₀)/yᵢ)² / Σᵢ wᵢ`. For each `L₀`,
letting `qᵢ = vᵢ(L₀)/yᵢ` gives `a = Σᵢ wᵢ qᵢ / Σᵢ wᵢ qᵢ²` and
`r₀ = r₀,ref a^(−3/5)`. Select the first minimum on the ordered grid.
Fractions give each mode comparable influence despite differing variance
scales. Estimated noise, temporal correlation, and model discrepancy can
all affect the profile. A malformed or unrepresentable fit throws before
changing the product.
"""
function process!(result::ModalVarianceProduct, workspace::ModalVarianceWorkspace,
    plan::ModalVariancePlan, inputs::ModalVarianceInputs)
    _validate_storage(result, workspace, plan, inputs)
    turbulence = workspace.turbulence_variance_m2
    for i in eachindex(turbulence)
        observed = Float64(inputs.observed_variance_m2[i])
        noise = Float64(inputs.noise_variance_m2[i])
        isfinite(observed) && 0 <= noise < observed || throw(ArgumentError(
            "each observed variance must be finite and exceed its nonnegative noise variance",
        ))
        turbulence[i] = observed - noise
    end

    scores, r0_profile = workspace.relative_rms_by_L0, workspace.r0_by_L0_m
    for candidate in eachindex(scores)
        r0_profile[candidate], scores[candidate] =
            _profile!(workspace.ratios, turbulence, plan, candidate)
    end
    score, best = findmin(scores)
    # At zero user tolerance, roundoff-equivalent profiles still share support.
    limit = score + max(plan.relative_rms_tolerance, 32eps(Float64))
    supported = <=(limit)
    lower, upper = findfirst(supported, scores), findlast(supported, scores)
    _profile!(workspace.ratios, turbulence, plan, best)
    turbulence .*= workspace.ratios
    all(x -> isfinite(x) && x > 0, turbulence) || throw(ArgumentError(
        "fitted modal variances must be finite and positive",
    ))

    copyto!(result.fitted_variance_m2, turbulence)
    copyto!(result.relative_rms_by_L0, scores)
    copyto!(result.r0_by_L0_m, r0_profile)
    result.r0_m = r0_profile[best]
    result.L0_m = plan.L0_candidates_m[best]
    result.L0_support_m = (plan.L0_candidates_m[lower], plan.L0_candidates_m[upper])
    result.L0_at_grid_boundary = length(scores) > 1 && (best == 1 || best == length(scores))
    result.relative_rms = score
    return result
end
