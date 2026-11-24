"""
    ModalScan()

Fit an equal-weight least-squares quadratic to a larger-is-better metric
sampled at supplied amplitudes of one mode. Pass one metric per amplitude in
the prepared order. A metric that is minimized, such as the `objective` from
[`WeightedCoreIntensity`](@ref), must be negated by the caller. Each repeat
contributes one observation to the fit.

The CPU reference uses Float64. With `x = (amplitude - center) / scale`, fit
`c₀ + c₁x + c₂x²` to metrics after subtracting the first metric and dividing by
the largest absolute resulting difference. This affine transformation preserves
the quadratic vertex. A peak requires negative resolved curvature and samples
on both sides of `x = -c₁ / (2c₂)`. A margin of `32eps(Float64)` in normalized
amplitude excludes numerical endpoint ties. The curvature guard is
`32eps(Float64) * ‖p₂‖₁ * max(1, maxabs(metrics) / metric_scale)`, where `p₂` is
the row of the prepared pseudoinverse estimating `c₂`. This is a Float64
roundoff guard, with no statistical confidence interpretation.

The four-argument `process!` uses `nothing` as workspace. For warmed calls
with ordinary Float64 CPU vectors and views, the fit uses bounded linear work
in the prepared sample count and no Julia heap allocations. The caller owns
the mode, amplitude units, metric definition, and measurement association.
"""
struct ModalScan <: AbstractCalibrationMethod end

"""
    ModalScanSpecification(amplitudes)

Declare a one-based vector of amplitudes in one mode's caller-defined units.
The prepared plan copies the amplitudes in supplied order. Repeated amplitudes
are allowed. Preparation requires finite Float64 coordinates and a design
matrix of numerical rank three after centering and scaling.
"""
struct ModalScanSpecification{A<:AbstractVector{<:Real}}
    amplitudes::A
end

"""
Prepared amplitude coordinates and quadratic least-squares operator for
[`ModalScan`](@ref). Treat the retained arrays as read-only. The plan is
reentrant with separate caller-owned products.
"""
struct ModalScanPlan <: AbstractCalibrationPlan
    amplitudes::Vector{Float64}
    normalized_amplitudes::Vector{Float64}
    fit_operator::Matrix{Float64}
    center::Float64
    scale::Float64
    minimum_amplitude::Float64
    maximum_amplitude::Float64
    curvature_row_norm::Float64
end

function prepare(::ModalScan, spec::ModalScanSpecification)
    count = length(spec.amplitudes)
    count >= 3 || throw(ArgumentError("modal scan needs at least three amplitudes"))
    _vector(spec.amplitudes, count, "amplitudes")
    amplitudes = collect(Float64, spec.amplitudes)
    lo, hi = extrema(amplitudes)
    width = hi - lo
    center = isfinite(width) ? lo + width / 2 : lo / 2 + hi / 2
    scale = max(center - lo, hi - center)
    scale > 0 || throw(ArgumentError("modal scan amplitudes do not span a nonzero interval"))
    x = (amplitudes .- center) ./ scale
    decomposition = svd(hcat(ones(count), x, x .^ 2))
    singular_values = decomposition.S
    singular_values[3] > count * eps(Float64) * singular_values[1] ||
        throw(ArgumentError("modal scan amplitudes do not support a rank-three quadratic fit"))
    fit_operator = decomposition.V * Diagonal(inv.(singular_values)) * decomposition.U'
    curvature_row_norm = sum(abs, view(fit_operator, 3, :))
    return ModalScanPlan(amplitudes, x, fit_operator, center, scale, lo, hi,
                         curvature_row_norm)
end

"""
Caller-owned result of [`ModalScan`](@ref). `best_sample_index` is the first
index with the greatest sampled metric. `best_sample_amplitude` and
`best_sample_metric` retain the input units. `peak_amplitude` is the fitted
peak, in the supplied amplitude units, when `status` is `Valid`. `residual_rms`
is `sqrt(sum(residual²) / count)` in metric units for a finite fit, including
fits with no peak. Unavailable fitted quantities are `nothing`. A finite fit
with an interior peak leaves scientific acceptance to the caller.
"""
mutable struct ModalScanProduct
    status::SharpeningStatus
    best_sample_index::Int
    best_sample_amplitude::Float64
    best_sample_metric::Float64
    peak_amplitude::Union{Nothing,Float64}
    residual_rms::Union{Nothing,Float64}
end

allocate_result(::ModalScanPlan) = ModalScanProduct(NumericalFailure, 0, 0., 0., nothing, nothing)
allocate_workspace(::ModalScanPlan) = nothing

function process!(result::ModalScanProduct, ::Nothing, plan::ModalScanPlan,
                  metrics::AbstractVector{<:Real})
    count = length(plan.amplitudes)
    length(plan.normalized_amplitudes) == count && size(plan.fit_operator) == (3, count) ||
        throw(DimensionMismatch("modal scan coefficients must match the prepared amplitudes"))
    _storage(metrics, count, "metrics")
    offset = Float64(metrics[1])
    best_metric = lowest_metric = offset
    best_index = 1
    @inbounds for i in 1:count
        metric = Float64(metrics[i])
        isfinite(metric) || throw(ArgumentError("metrics must be finite in Float64"))
        lowest_metric = min(lowest_metric, metric)
        if metric > best_metric
            best_metric, best_index = metric, i
        end
    end
    result.best_sample_index = best_index
    result.best_sample_amplitude = plan.amplitudes[best_index]
    result.best_sample_metric = best_metric
    result.peak_amplitude = result.residual_rms = nothing
    metric_scale = max(best_metric - offset, offset - lowest_metric)
    if !isfinite(metric_scale)
        result.status = NumericalFailure
    elseif iszero(metric_scale)
        result.status = UnresolvedCurvature
        result.residual_rms = 0.0
    else
        max_abs_metric = max(abs(lowest_metric), abs(best_metric))
        _fit_modal_scan!(result, plan, metrics, offset, metric_scale, max_abs_metric)
    end
    return result
end

function _fit_modal_scan!(result, plan, metrics, offset, metric_scale, max_abs_metric)
    c₀ = c₁ = c₂ = 0.0
    operator = plan.fit_operator
    x = plan.normalized_amplitudes
    @inbounds for i in eachindex(x)
        y = (Float64(metrics[i]) - offset) / metric_scale
        c₀ = muladd(operator[1, i], y, c₀)
        c₁ = muladd(operator[2, i], y, c₁)
        c₂ = muladd(operator[3, i], y, c₂)
    end
    sum_squared_residuals = 0.0
    @inbounds for i in eachindex(x)
        y = (Float64(metrics[i]) - offset) / metric_scale
        residual = evalpoly(x[i], (c₀, c₁, c₂)) - y
        sum_squared_residuals = muladd(residual, residual, sum_squared_residuals)
    end
    residual_rms = sqrt(sum_squared_residuals / length(x)) * metric_scale
    if !isfinite(residual_rms)
        result.status = NumericalFailure
        return nothing
    end
    result.residual_rms = residual_rms

    margin = 32eps(Float64)
    curvature_guard = margin * plan.curvature_row_norm * max(1.0, max_abs_metric / metric_scale)
    if abs(c₂) <= curvature_guard
        result.status = UnresolvedCurvature
    elseif c₂ > 0
        result.status = NonconcaveFit
    else
        vertex = -c₁ / (2c₂)
        lo = (plan.minimum_amplitude - plan.center) / plan.scale
        hi = (plan.maximum_amplitude - plan.center) / plan.scale
        peak = muladd(plan.scale, vertex, plan.center)
        if lo + margin < vertex < hi - margin &&
           plan.minimum_amplitude < peak < plan.maximum_amplitude
            result.status = Valid
            result.peak_amplitude = peak
        else
            result.status = UnbracketedPeak
        end
    end
    return nothing
end
