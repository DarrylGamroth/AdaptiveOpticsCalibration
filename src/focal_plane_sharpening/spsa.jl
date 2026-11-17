"""Prepare the pure supplied-direction SPSA pair `q ± cΔ`; no RNG or command conversion."""
struct SPSAProbes <: AbstractCalibrationMethod end
"""Immutable coefficient count for SPSA probe generation."""
struct SPSAProbePlan <: AbstractCalibrationPlan
    count::Int
end
prepare(::SPSAProbes, spec::CoefficientSpecification) = SPSAProbePlan(spec.count)

"""
    SPSAProbeInputs(coefficients, perturbation, direction)

One-based floating coefficient vector, positive scalar `c` in coefficient
units, and realized Rademacher direction (exact ±1; Bool is rejected). Results
use Float64. Each probe must remain changed when represented in each input
coefficient's native precision; downstream command conversion is caller-owned.
"""
struct SPSAProbeInputs{Q<:AbstractVector{<:AbstractFloat},C<:Real,D<:AbstractVector{<:Real}}
    coefficients::Q
    perturbation::C
    direction::D
end

"""Caller-owned `plus`/`minus` coefficient vectors and typed numerical status."""
mutable struct SPSAProbeProduct
    status::SharpeningStatus
    plus::Vector{Float64}
    minus::Vector{Float64}
end
allocate_result(plan::SPSAProbePlan) = SPSAProbeProduct(
    NumericalFailure, zeros(plan.count), zeros(plan.count))
allocate_workspace(::SPSAProbePlan) = nothing

_rademacher(::Bool) = false
_rademacher(x::Real) = x == 1 || x == -1
function _direction(d::AbstractVector{<:Real}, n::Int)
    Base.require_one_based_indexing(d)
    length(d) == n || throw(DimensionMismatch("direction length"))
    all(_rademacher, d) || throw(ArgumentError("direction must contain exact ±1, not Bool"))
end

function process!(result::SPSAProbeProduct, ::Nothing, plan::SPSAProbePlan,
                  inputs::SPSAProbeInputs)
    q, d = inputs.coefficients, inputs.direction
    _vector(q, plan.count, "coefficients")
    _direction(d, plan.count)
    c = _positive(inputs.perturbation, "perturbation")
    _storage(result.plus, plan.count, "plus output")
    _storage(result.minus, plan.count, "minus output")
    _noalias((result.plus, result.minus), (q, d))
    status = Valid
    @inbounds for i in 1:plan.count
        plus, minus = Float64(q[i]) + c * Float64(d[i]), Float64(q[i]) - c * Float64(d[i])
        if !(isfinite(plus) && isfinite(minus))
            status = NumericalFailure
            break
        else
            native_plus = convert(typeof(q[i]), plus)
            native_minus = convert(typeof(q[i]), minus)
            if !(isfinite(native_plus) && isfinite(native_minus))
                status = NumericalFailure
                break
            elseif plus == Float64(q[i]) || minus == Float64(q[i]) ||
                   plus == minus || native_plus == q[i] || native_minus == q[i]
                status = LostPerturbation
            end
        end
    end
    if status == Valid
        @inbounds for i in 1:plan.count
            result.plus[i] = Float64(q[i]) + c * Float64(d[i])
            result.minus[i] = Float64(q[i]) - c * Float64(d[i])
        end
    else
        fill!(result.plus, 0.0)
        fill!(result.minus, 0.0)
    end
    result.status = status
    return result
end

"""Prepare the two-metric SPSA estimate `(Jplus-Jminus)/(2cΔᵢ)` in supplied order."""
struct SPSAGradient <: AbstractCalibrationMethod end
"""Immutable coefficient count for a supplied SPSA metric pair."""
struct SPSAGradientPlan <: AbstractCalibrationPlan
    count::Int
end
prepare(::SPSAGradient, spec::CoefficientSpecification) = SPSAGradientPlan(spec.count)

"""
    SPSAGradientInputs(direction, perturbation, plus_metric, minus_metric)

Supply exact ±1 direction, positive `c`, and two finite metrics with identical
definition/radiometry. Association and pair ordering belong to the caller.
Gradient units are objective/coefficient. A single pair estimates a directional
gradient including cross terms, not a complete coordinate-wise gradient.
"""
struct SPSAGradientInputs{D<:AbstractVector{<:Real},C<:Real,P<:Real,M<:Real}
    direction::D
    perturbation::C
    plus_metric::P
    minus_metric::M
end
"""Caller-owned Float64 gradient vector and typed numerical outcome."""
mutable struct SPSAGradientProduct
    status::SharpeningStatus
    gradient::Vector{Float64}
end
allocate_result(plan::SPSAGradientPlan) = SPSAGradientProduct(NumericalFailure, zeros(plan.count))
allocate_workspace(::SPSAGradientPlan) = nothing

function process!(result::SPSAGradientProduct, ::Nothing, plan::SPSAGradientPlan,
                  inputs::SPSAGradientInputs)
    _direction(inputs.direction, plan.count)
    c = _positive(inputs.perturbation, "perturbation")
    p, m = Float64(inputs.plus_metric), Float64(inputs.minus_metric)
    isfinite(p) && isfinite(m) || throw(ArgumentError("metrics must be finite in Float64"))
    _storage(result.gradient, plan.count, "gradient output")
    _noalias((result.gradient,), (inputs.direction,))
    difference = p - m
    # Retain subnormal differences when 2c is representable; only rescale when
    # the denominator would overflow. An overflowing difference is invalid.
    denominator = 2.0 * c
    scale = isfinite(denominator) ? difference / denominator : (difference * 0.5) / c
    if isfinite(difference) && isfinite(scale)
        @inbounds for i in 1:plan.count
            result.gradient[i] = scale / Float64(inputs.direction[i])
        end
        result.status = Valid
    else
        fill!(result.gradient, 0.0)
        result.status = NumericalFailure
    end
    return result
end
