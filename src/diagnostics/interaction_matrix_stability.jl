# Package-defined interaction-matrix stability checks. See docs/REFERENCES.md.
"""
    StabilityStatus

Classify one comparison of independently identified interaction matrices.
"""
@enum StabilityStatus::UInt8 begin
    Stable
    InsufficientColumnNorm
    ExcessiveRelativeColumnDifference
    ExcessiveRelativeFrobeniusDifference
    NonFiniteStabilityCalculation
end

@doc "Both interaction matrices satisfy every prepared stability criterion." Stable
@doc "At least one interaction-matrix column has insufficient norm." InsufficientColumnNorm
@doc "At least one column exceeds the relative-difference criterion." ExcessiveRelativeColumnDifference
@doc "The complete matrices exceed the relative Frobenius-difference criterion." ExcessiveRelativeFrobeniusDifference
@doc "Finite inputs produced a non-finite intermediate stability calculation." NonFiniteStabilityCalculation

"""
    InteractionMatrixStability(minimum_column_norm,
                               maximum_relative_column_difference,
                               maximum_relative_frobenius_difference)

Define numerical criteria for comparing two independently identified
interaction matrices. Relative-difference limits are dimensionless values in
`[0,2]`; `minimum_column_norm` has the interaction-matrix column unit.
"""
struct InteractionMatrixStability{T<:AbstractFloat} <: AbstractCalibrationMethod
    minimum_column_norm::T
    maximum_relative_column_difference::T
    maximum_relative_frobenius_difference::T

    function InteractionMatrixStability{T}(
        minimum_column_norm::T,
        maximum_relative_column_difference::T,
        maximum_relative_frobenius_difference::T,
    ) where {T<:AbstractFloat}
        isfinite(minimum_column_norm) && minimum_column_norm > zero(T) ||
            throw(ArgumentError("minimum_column_norm must be finite and positive"))
        isfinite(maximum_relative_column_difference) &&
            zero(T) <= maximum_relative_column_difference <= T(2) ||
            throw(ArgumentError(
                "maximum_relative_column_difference must be finite and in [0,2]",
            ))
        isfinite(maximum_relative_frobenius_difference) &&
            zero(T) <= maximum_relative_frobenius_difference <= T(2) ||
            throw(ArgumentError(
                "maximum_relative_frobenius_difference must be finite and in [0,2]",
            ))
        return new{T}(
            minimum_column_norm,
            maximum_relative_column_difference,
            maximum_relative_frobenius_difference,
        )
    end
end

function InteractionMatrixStability(
    minimum_column_norm::Real,
    maximum_relative_column_difference::Real,
    maximum_relative_frobenius_difference::Real,
)
    T = float(promote_type(
        typeof(minimum_column_norm),
        typeof(maximum_relative_column_difference),
        typeof(maximum_relative_frobenius_difference),
    ))
    T <: AbstractFloat || throw(ArgumentError(
        "interaction-matrix stability requires an AbstractFloat representation",
    ))
    minimum = T(minimum_column_norm)
    maximum_column = T(maximum_relative_column_difference)
    maximum_frobenius = T(maximum_relative_frobenius_difference)
    isfinite(minimum) && minimum > zero(T) || throw(ArgumentError(
        "minimum_column_norm must be finite and positive",
    ))
    isfinite(maximum_column) && zero(T) <= maximum_column <= T(2) ||
        throw(ArgumentError(
            "maximum_relative_column_difference must be finite and in [0,2]",
        ))
    isfinite(maximum_frobenius) && zero(T) <= maximum_frobenius <= T(2) ||
        throw(ArgumentError(
            "maximum_relative_frobenius_difference must be finite and in [0,2]",
        ))
    return InteractionMatrixStability{T}(minimum, maximum_column, maximum_frobenius)
end

"""
    InteractionMatrixPair(first_window, second_window)

Bundle two independently identified measurement-by-corrector interaction
matrices without copying them.
"""
struct InteractionMatrixPair{
    T<:AbstractFloat,
    A<:AbstractMatrix{T},
    B<:AbstractMatrix{T},
}
    first_window::A
    second_window::B
end

"""
    InteractionMatrixStabilitySpecification(measurement_count,
                                            corrector_coordinate_count)

Define the positive measurement-row and corrector-column extents of both
interaction-matrix windows without binding transient matrix storage.
"""
struct InteractionMatrixStabilitySpecification
    measurement_count::Int
    corrector_coordinate_count::Int

    function InteractionMatrixStabilitySpecification(
        measurement_count::Integer,
        corrector_coordinate_count::Integer,
    )
        measurements = Int(measurement_count)
        correctors = Int(corrector_coordinate_count)
        measurements > 0 || throw(ArgumentError(
            "measurement_count must be positive",
        ))
        correctors > 0 || throw(ArgumentError(
            "corrector_coordinate_count must be positive",
        ))
        _checked_extent("interaction matrix", measurements, correctors)
        return new(measurements, correctors)
    end
end

"""Prepared interaction-matrix stability comparison."""
struct InteractionMatrixStabilityPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    criteria::InteractionMatrixStability{T}
end

function prepare(
    method::InteractionMatrixStability{T},
    specification::InteractionMatrixStabilitySpecification,
) where {T}
    return InteractionMatrixStabilityPlan(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        method,
    )
end

function prepare(
    method::InteractionMatrixStability{T},
    input::InteractionMatrixPair{T},
) where {T}
    size(input.first_window) == size(input.second_window) || throw(DimensionMismatch(
        "interaction-matrix windows must have the same shape",
    ))
    all(>(0), size(input.first_window)) || throw(DimensionMismatch(
        "interaction-matrix windows must be nonempty",
    ))
    return prepare(
        method,
        InteractionMatrixStabilitySpecification(size(input.first_window)...),
    )
end

"""
    InteractionMatrixStabilityProduct{T}

Caller-owned diagnostics for an interaction-matrix stability comparison.
Corrector coordinates are one-based and are zero only when no coordinate
applies to the corresponding field.
"""
mutable struct InteractionMatrixStabilityProduct{T<:AbstractFloat}
    status::StabilityStatus
    status_corrector_coordinate::Int
    minimum_first_window_column_norm::T
    minimum_second_window_column_norm::T
    maximum_relative_column_difference::T
    maximum_relative_column_difference_coordinate::Int
    relative_frobenius_difference::T
end

function allocate_result(plan::InteractionMatrixStabilityPlan{T}) where {T}
    return InteractionMatrixStabilityProduct{T}(
        InsufficientColumnNorm,
        0,
        zero(T),
        zero(T),
        zero(T),
        0,
        zero(T),
    )
end

allocate_workspace(::InteractionMatrixStabilityPlan) = nothing

function _publish_nonfinite_stability_calculation!(
    result::InteractionMatrixStabilityProduct{T},
) where {T}
    result.status = NonFiniteStabilityCalculation
    result.status_corrector_coordinate = 0
    result.minimum_first_window_column_norm = zero(T)
    result.minimum_second_window_column_norm = zero(T)
    result.maximum_relative_column_difference = zero(T)
    result.maximum_relative_column_difference_coordinate = 0
    result.relative_frobenius_difference = zero(T)
    return result
end

function _validate_stability_input(plan, input)
    expected = (plan.measurement_count, plan.corrector_coordinate_count)
    size(input.first_window) == expected || throw(DimensionMismatch(
        "first interaction-matrix window must have shape $expected",
    ))
    size(input.second_window) == expected || throw(DimensionMismatch(
        "second interaction-matrix window must have shape $expected",
    ))
    all(isfinite, input.first_window) || throw(ArgumentError(
        "first interaction-matrix window must contain only finite values",
    ))
    all(isfinite, input.second_window) || throw(ArgumentError(
        "second interaction-matrix window must contain only finite values",
    ))
    return nothing
end

"""
    process!(result, nothing, plan::InteractionMatrixStabilityPlan, input)

Compare two independently identified interaction matrices.

For each corrector column and for the complete matrices, the symmetric relative
difference is `2norm(A-B)/(norm(A)+norm(B))`. Status precedence is insufficient
column norm, excessive maximum column difference, excessive Frobenius
difference, then stable. Ties retain the lowest one-based corrector coordinate.
Invalid input throws without changing the result. A non-finite calculation
from finite input publishes `NonFiniteStabilityCalculation` with zero-valued
diagnostics and no applicable coordinate.
"""
function process!(
    result::InteractionMatrixStabilityProduct{T},
    ::Nothing,
    plan::InteractionMatrixStabilityPlan{T},
    input::InteractionMatrixPair{T},
) where {T}
    _validate_stability_input(plan, input)
    first_matrix = input.first_window
    second_matrix = input.second_window
    two = T(2)
    minimum_first = T(Inf)
    minimum_second = T(Inf)
    maximum_relative = zero(T)
    maximum_coordinate = 1
    first_insufficient_coordinate = 0
    total_first_squares = zero(T)
    total_second_squares = zero(T)
    total_difference_squares = zero(T)

    first_columns = axes(first_matrix, 2)
    second_columns = axes(second_matrix, 2)
    for (corrector_coordinate, (first_column, second_column)) in
        enumerate(zip(first_columns, second_columns))
        first_squares = zero(T)
        second_squares = zero(T)
        difference_squares = zero(T)
        for (first_row, second_row) in zip(axes(first_matrix, 1), axes(second_matrix, 1))
            first_value = first_matrix[first_row, first_column]
            second_value = second_matrix[second_row, second_column]
            difference = first_value - second_value
            first_squares += abs2(first_value)
            second_squares += abs2(second_value)
            difference_squares += abs2(difference)
        end
        all(isfinite, (first_squares, second_squares, difference_squares)) ||
            return _publish_nonfinite_stability_calculation!(result)
        total_first_squares += first_squares
        total_second_squares += second_squares
        total_difference_squares += difference_squares
        first_norm = sqrt(first_squares)
        second_norm = sqrt(second_squares)
        difference_norm = sqrt(difference_squares)
        minimum_first = min(minimum_first, first_norm)
        minimum_second = min(minimum_second, second_norm)
        if first_norm < plan.criteria.minimum_column_norm ||
           second_norm < plan.criteria.minimum_column_norm
            iszero(first_insufficient_coordinate) &&
                (first_insufficient_coordinate = corrector_coordinate)
            continue
        end
        relative = two * difference_norm / (first_norm + second_norm)
        isfinite(relative) ||
            return _publish_nonfinite_stability_calculation!(result)
        if relative > maximum_relative
            maximum_relative = relative
            maximum_coordinate = corrector_coordinate
        end
    end

    all(isfinite, (
        total_first_squares,
        total_second_squares,
        total_difference_squares,
    )) || return _publish_nonfinite_stability_calculation!(result)
    total_norm = sqrt(total_first_squares) + sqrt(total_second_squares)
    relative_frobenius = iszero(total_norm) ? zero(T) :
        two * sqrt(total_difference_squares) / total_norm
    isfinite(relative_frobenius) ||
        return _publish_nonfinite_stability_calculation!(result)

    status, status_coordinate = if !iszero(first_insufficient_coordinate)
        (InsufficientColumnNorm, first_insufficient_coordinate)
    elseif maximum_relative > plan.criteria.maximum_relative_column_difference
        (ExcessiveRelativeColumnDifference, maximum_coordinate)
    elseif relative_frobenius > plan.criteria.maximum_relative_frobenius_difference
        (ExcessiveRelativeFrobeniusDifference, 0)
    else
        (Stable, 0)
    end

    result.status = status
    result.status_corrector_coordinate = status_coordinate
    result.minimum_first_window_column_norm = minimum_first
    result.minimum_second_window_column_norm = minimum_second
    result.maximum_relative_column_difference = maximum_relative
    result.maximum_relative_column_difference_coordinate = maximum_coordinate
    result.relative_frobenius_difference = relative_frobenius
    return result
end
