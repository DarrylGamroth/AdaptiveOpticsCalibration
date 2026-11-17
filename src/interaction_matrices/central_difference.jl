"""
    CentralDifference(amplitude)

Select a central difference for one signed probe pair at `±amplitude`.
The derivative has measurement units per command unit.
"""
struct CentralDifference{T<:AbstractFloat} <: AbstractCalibrationMethod
    amplitude::T

    function CentralDifference(amplitude::T) where {T<:AbstractFloat}
        isfinite(2amplitude) && amplitude > 0 ||
            throw(ArgumentError("probe amplitude must be positive and finite when doubled"))
        return new{T}(amplitude)
    end
end

struct CentralDifferencePlan{T} <: AbstractCalibrationPlan
    measurements::Int
    span::T
end

function prepare(method::CentralDifference, measurements::Integer)
    count = _checked_extent("measurement count", measurements)
    count > 0 || throw(ArgumentError("measurement count must be positive"))
    return CentralDifferencePlan(count, 2 * method.amplitude)
end

allocate_result(plan::CentralDifferencePlan{T}) where {T} = Vector{T}(undef, plan.measurements)
allocate_workspace(::CentralDifferencePlan) = nothing

function process!(
    derivative::AbstractVector,
    ::Nothing,
    plan::CentralDifferencePlan,
    (positive, negative)::Tuple{AbstractVector,AbstractVector},
)
    axes(derivative) == axes(positive) == axes(negative) == (Base.OneTo(plan.measurements),) ||
        throw(DimensionMismatch("probe responses must match the prepared measurement axis"))
    derivative .= (positive .- negative) ./ plan.span
    return derivative
end
