"""
    ConditioningCriteria(minimum_corrector_variance,
                         maximum_absolute_corrector_correlation,
                         minimum_factorization_pivot_ratio)

Define DO-CRIME covariance-conditioning criteria. Variance is in squared
corrector units; correlation and pivot ratio are dimensionless.
"""
struct ConditioningCriteria{T<:AbstractFloat}
    minimum_corrector_variance::T
    maximum_absolute_corrector_correlation::T
    minimum_factorization_pivot_ratio::T

    function ConditioningCriteria{T}(
        minimum_corrector_variance::Real,
        maximum_absolute_corrector_correlation::Real,
        minimum_factorization_pivot_ratio::Real,
    ) where {T<:AbstractFloat}
        minimum_variance = T(minimum_corrector_variance)
        maximum_correlation = T(maximum_absolute_corrector_correlation)
        minimum_pivot = T(minimum_factorization_pivot_ratio)
        isfinite(minimum_variance) && minimum_variance > zero(T) ||
            throw(ArgumentError("minimum_corrector_variance must be finite and positive"))
        isfinite(maximum_correlation) && zero(T) <= maximum_correlation < one(T) ||
            throw(ArgumentError(
                "maximum_absolute_corrector_correlation must be finite and in [0,1)",
            ))
        isfinite(minimum_pivot) && zero(T) < minimum_pivot <= one(T) ||
            throw(ArgumentError(
                "minimum_factorization_pivot_ratio must be finite and in (0,1]",
            ))
        return new{T}(minimum_variance, maximum_correlation, minimum_pivot)
    end
end

function ConditioningCriteria(
    minimum_corrector_variance::Real,
    maximum_absolute_corrector_correlation::Real,
    minimum_factorization_pivot_ratio::Real,
)
    T = float(promote_type(
        typeof(minimum_corrector_variance),
        typeof(maximum_absolute_corrector_correlation),
        typeof(minimum_factorization_pivot_ratio),
    ))
    return ConditioningCriteria{T}(
        minimum_corrector_variance,
        maximum_absolute_corrector_correlation,
        minimum_factorization_pivot_ratio,
    )
end

"""Dimensions and representation of a batch DO-CRIME solve."""
struct CovarianceSpecification{T<:AbstractFloat}
    measurement_count::Int
    corrector_coordinate_count::Int
    function CovarianceSpecification{T}(measurements::Integer, correctors::Integer) where {T}
        m = Int(measurements)
        c = Int(correctors)
        m > 0 || throw(ArgumentError("measurement_count must be positive"))
        c > 0 || throw(ArgumentError("corrector_coordinate_count must be positive"))
        _checked_extent("DO-CRIME covariance", m, c)
        _checked_extent("DO-CRIME corrector covariance", c, c)
        return new{T}(m, c)
    end
end

CovarianceSpecification(
    measurements::Integer,
    correctors::Integer,
    ::Type{T}=Float64,
) where {T<:AbstractFloat} = CovarianceSpecification{T}(measurements, correctors)

"""Select the open-loop estimator `D = C_mc * C_cc⁻¹`."""
struct OpenLoop{T<:AbstractFloat} <: AbstractCalibrationMethod
    criteria::ConditioningCriteria{T}
end

"""
    ClosedLoop(loop_gain, criteria)

Select the closed-loop estimator `D = g * C_mΔc * C_ΔcΔc⁻¹`, where `Δc`
is the complete exposure-associated corrector command increment.
"""
struct ClosedLoop{T<:AbstractFloat} <: AbstractCalibrationMethod
    loop_gain::T
    criteria::ConditioningCriteria{T}

    function ClosedLoop{T}(
        loop_gain::T,
        criteria::ConditioningCriteria{T},
    ) where {T<:AbstractFloat}
        isfinite(loop_gain) && loop_gain > zero(T) ||
            throw(ArgumentError("loop_gain must be finite and positive"))
        return new{T}(loop_gain, criteria)
    end
end

function ClosedLoop(loop_gain::Real, criteria::ConditioningCriteria{T}) where {T}
    gain = T(loop_gain)
    isfinite(gain) && gain > zero(T) ||
        throw(ArgumentError("loop_gain must be finite and positive"))
    return ClosedLoop{T}(gain, criteria)
end

"""Prepared complete-batch DO-CRIME solve."""
struct InteractionMatrixPlan{T<:AbstractFloat,M<:AbstractCalibrationMethod} <:
       AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    response_scale::T
    criteria::ConditioningCriteria{T}
    method::M
end

function prepare(method::OpenLoop, specification::CovarianceSpecification{T}) where {T}
    criteria = ConditioningCriteria{T}(
        method.criteria.minimum_corrector_variance,
        method.criteria.maximum_absolute_corrector_correlation,
        method.criteria.minimum_factorization_pivot_ratio,
    )
    prepared_method = OpenLoop(criteria)
    return InteractionMatrixPlan(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        one(T),
        criteria,
        prepared_method,
    )
end

function prepare(method::ClosedLoop, specification::CovarianceSpecification{T}) where {T}
    criteria = ConditioningCriteria{T}(
        method.criteria.minimum_corrector_variance,
        method.criteria.maximum_absolute_corrector_correlation,
        method.criteria.minimum_factorization_pivot_ratio,
    )
    gain = T(method.loop_gain)
    isfinite(gain) && gain > zero(T) ||
        throw(ArgumentError("loop_gain must be finite and positive"))
    prepared_method = ClosedLoop(gain, criteria)
    return InteractionMatrixPlan(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        gain,
        criteria,
        prepared_method,
    )
end

"""Frozen centered covariance sums from one coherent accumulation window."""
struct CovarianceSums{
    N<:Integer,
    C<:AbstractMatrix{<:Real},
    V<:AbstractMatrix{<:Real},
}
    sample_count::N
    measurement_corrector_cross_covariance_sum::C
    corrector_covariance_sum::V

    function CovarianceSums(sample_count::Integer, cross::C, corrector::V) where {
        C<:AbstractMatrix{<:Real},
        V<:AbstractMatrix{<:Real},
    }
        sample_count >= 0 || throw(ArgumentError("sample_count must be nonnegative"))
        return new{typeof(sample_count),C,V}(sample_count, cross, corrector)
    end
end

"""Expected scientific outcome of a DO-CRIME covariance solve."""
@enum InteractionMatrixStatus::UInt8 begin
    Valid
    InsufficientSamples
    InsufficientCorrectorVariance
    ExcessiveCorrectorCorrelation
    NonPositiveDefinite
    InsufficientFactorizationPivot
end

@doc "The covariance window produced a valid interaction matrix." Valid
@doc "The covariance window contains fewer than two samples." InsufficientSamples
@doc "At least one corrector coordinate has insufficient variance." InsufficientCorrectorVariance
@doc "At least one corrector-coordinate pair has excessive absolute correlation." ExcessiveCorrectorCorrelation
@doc "The corrector covariance is not positive definite in the declared ordering." NonPositiveDefinite
@doc "A positive Cholesky pivot is too small relative to the maximum variance." InsufficientFactorizationPivot

"""Numerical or representational rejection from a direct DO-CRIME solve."""
@enum InteractionMatrixFailure::UInt8 begin
    NonFiniteCrossCovariance = 1
    NonFiniteCorrectorCovariance = 2
    NonSymmetricCorrectorCovariance = 3
    SampleCountNotRepresentable = 4
    SampleCountOutOfRange = 5
    NonFiniteNormalization = 6
    NonFiniteCorrelation = 7
    NonFiniteFactorization = 8
    NonFiniteSolve = 9
end

@doc "The measurement--corrector covariance sum contains a non-finite value." NonFiniteCrossCovariance
@doc "The corrector covariance sum contains a non-finite value." NonFiniteCorrectorCovariance
@doc "The corrector covariance sum is not symmetric in the plan representation." NonSymmetricCorrectorCovariance
@doc "The sample covariance divisor is not exactly representable." SampleCountNotRepresentable
@doc "The sample count is outside Julia's native `Int` range." SampleCountOutOfRange
@doc "Sample-covariance normalization produced a non-finite value." NonFiniteNormalization
@doc "Corrector-correlation calculation produced a non-finite value." NonFiniteCorrelation
@doc "Cholesky factorization produced a non-finite value." NonFiniteFactorization
@doc "Triangular interaction-matrix solution produced a non-finite value." NonFiniteSolve

"""Conditioning diagnostics, with one-based corrector coordinates."""
struct InteractionMatrixDiagnostics{T<:AbstractFloat}
    status::InteractionMatrixStatus
    sample_count::Int
    minimum_corrector_variance::T
    maximum_corrector_variance::T
    maximum_absolute_corrector_correlation::T
    minimum_factorization_pivot_ratio::T
    first_corrector_coordinate::Int
    second_corrector_coordinate::Int
end

_initial_diagnostics(::Type{T}) where {T} = InteractionMatrixDiagnostics{T}(
    InsufficientSamples,
    0,
    zero(T),
    zero(T),
    zero(T),
    zero(T),
    0,
    0,
)

"""Caller-owned DO-CRIME interaction matrix and conditioning diagnostics."""
mutable struct InteractionMatrixProduct{T<:AbstractFloat}
    matrix::Matrix{T}
    diagnostics::InteractionMatrixDiagnostics{T}
end

"""Return the caller-owned measurement-by-corrector interaction matrix."""
interaction_matrix(result::InteractionMatrixProduct) = result.matrix

"""Replaceable scratch for one complete DO-CRIME solve."""
mutable struct InteractionMatrixWorkspace{T<:AbstractFloat}
    corrector_covariance::Matrix{T}
    factor::Matrix{T}
    staged_interaction_matrix::Matrix{T}
    solve_scratch::Vector{T}
    diagnostics::InteractionMatrixDiagnostics{T}
end

function allocate_result(plan::InteractionMatrixPlan{T}) where {T}
    return InteractionMatrixProduct(
        zeros(T, plan.measurement_count, plan.corrector_coordinate_count),
        _initial_diagnostics(T),
    )
end

function allocate_workspace(plan::InteractionMatrixPlan{T}) where {T}
    correctors = plan.corrector_coordinate_count
    return InteractionMatrixWorkspace(
        zeros(T, correctors, correctors),
        zeros(T, correctors, correctors),
        zeros(T, plan.measurement_count, correctors),
        zeros(T, correctors),
        _initial_diagnostics(T),
    )
end

function _validate_storage(output, workspace, plan)
    expected = (plan.measurement_count, plan.corrector_coordinate_count)
    Base.require_one_based_indexing(output)
    size(output) == expected ||
        throw(DimensionMismatch("DO-CRIME result matrix must have shape $expected"))
    correctors = plan.corrector_coordinate_count
    size(workspace.corrector_covariance) == (correctors, correctors) &&
        size(workspace.factor) == (correctors, correctors) &&
        size(workspace.staged_interaction_matrix) == expected &&
        length(workspace.solve_scratch) == correctors ||
        throw(DimensionMismatch("DO-CRIME workspace has incompatible extents"))
    Base.mightalias(output, workspace.corrector_covariance) &&
        throw(ArgumentError("DO-CRIME result must not alias workspace storage"))
    Base.mightalias(output, workspace.factor) &&
        throw(ArgumentError("DO-CRIME result must not alias workspace storage"))
    Base.mightalias(output, workspace.staged_interaction_matrix) &&
        throw(ArgumentError("DO-CRIME result must not alias workspace storage"))
    Base.mightalias(output, workspace.solve_scratch) &&
        throw(ArgumentError("DO-CRIME result must not alias workspace storage"))
    return nothing
end

function _reject_workspace_alias(input, workspace)
    Base.mightalias(input, workspace.corrector_covariance) && throw(ArgumentError(
        "DO-CRIME covariance inputs must not alias workspace storage",
    ))
    Base.mightalias(input, workspace.factor) && throw(ArgumentError(
        "DO-CRIME covariance inputs must not alias workspace storage",
    ))
    Base.mightalias(input, workspace.staged_interaction_matrix) && throw(ArgumentError(
        "DO-CRIME covariance inputs must not alias workspace storage",
    ))
    Base.mightalias(input, workspace.solve_scratch) && throw(ArgumentError(
        "DO-CRIME covariance inputs must not alias workspace storage",
    ))
    return nothing
end

function _stage_inputs!(workspace, input, plan, ::Type{T}) where {T}
    cross = input.measurement_corrector_cross_covariance_sum
    corrector = input.corrector_covariance_sum
    Base.require_one_based_indexing(cross, corrector)
    size(cross) == (plan.measurement_count, plan.corrector_coordinate_count) ||
        throw(DimensionMismatch("measurement-corrector covariance sum has invalid shape"))
    size(corrector) ==
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count) ||
        throw(DimensionMismatch("corrector covariance sum has invalid shape"))
    _reject_workspace_alias(cross, workspace)
    _reject_workspace_alias(corrector, workspace)

    staged_cross = workspace.staged_interaction_matrix
    for column in 1:plan.corrector_coordinate_count
        for measurement in 1:plan.measurement_count
            value = T(cross[measurement, column])
            isfinite(value) || return NonFiniteCrossCovariance
            staged_cross[measurement, column] = value
        end
    end

    staged_corrector = workspace.corrector_covariance
    for column in 1:plan.corrector_coordinate_count
        for row in 1:plan.corrector_coordinate_count
            value = T(corrector[row, column])
            isfinite(value) || return NonFiniteCorrectorCovariance
            staged_corrector[row, column] = value
        end
    end
    for column in 1:plan.corrector_coordinate_count
        for row in 1:(column - 1)
            staged_corrector[row, column] == staged_corrector[column, row] ||
                return NonSymmetricCorrectorCovariance
        end
    end
    return nothing
end

function _diagnostics(
    ::Type{T}, status, input, minimum_variance=zero(T), maximum_variance=zero(T),
    maximum_correlation=zero(T), minimum_pivot=zero(T), first_coordinate=0,
    second_coordinate=0,
) where {T}
    return InteractionMatrixDiagnostics{T}(
        status,
        input.sample_count,
        minimum_variance,
        maximum_variance,
        maximum_correlation,
        minimum_pivot,
        first_coordinate,
        second_coordinate,
    )
end

function _publish_status!(output, workspace, diagnostics)
    fill!(workspace.staged_interaction_matrix, zero(eltype(workspace.staged_interaction_matrix)))
    copyto!(output, workspace.staged_interaction_matrix)
    workspace.diagnostics = diagnostics
    return nothing
end

"""
    process!(output, workspace, plan::InteractionMatrixPlan, covariances)

Estimate an interaction matrix from a complete coherent covariance window.
Expected conditioning rejections publish a zero matrix and typed diagnostics.
Numerical or representational rejections return [`InteractionMatrixFailure`](@ref)
without changing `output`. Malformed shapes and storage aliases throw without
changing `output`.

Covariance inputs must have one-based axes and must not alias workspace
storage. They may alias `output`: all converted inputs are staged and validated
before `output` is mutated. On successful publication the method returns
`nothing`; diagnostics are available in `workspace.diagnostics`.
"""
function process!(
    output::AbstractMatrix{T},
    workspace::InteractionMatrixWorkspace{T},
    plan::InteractionMatrixPlan{T},
    input::CovarianceSums,
) where {T}
    _validate_storage(output, workspace, plan)
    input.sample_count <= typemax(Int) || return SampleCountOutOfRange
    staging_failure = _stage_inputs!(workspace, input, plan, T)
    staging_failure === nothing || return staging_failure
    cross = workspace.staged_interaction_matrix
    corrector_sum = workspace.corrector_covariance
    if input.sample_count < 2
        return _publish_status!(
            output,
            workspace,
            _diagnostics(T, InsufficientSamples, input),
        )
    end

    divisor_count = input.sample_count - one(input.sample_count)
    divisor = T(divisor_count)
    isfinite(divisor) && divisor == divisor_count || return SampleCountNotRepresentable
    normalization = inv(divisor)
    @. workspace.corrector_covariance = corrector_sum * normalization
    covariance = workspace.corrector_covariance
    all(isfinite, covariance) || return NonFiniteNormalization
    minimum_variance = covariance[1, 1]
    maximum_variance = minimum_variance
    insufficient_coordinate = 0
    for coordinate in axes(covariance, 1)
        variance = covariance[coordinate, coordinate]
        minimum_variance = min(minimum_variance, variance)
        maximum_variance = max(maximum_variance, variance)
        if iszero(insufficient_coordinate) &&
           variance < plan.criteria.minimum_corrector_variance
            insufficient_coordinate = coordinate
        end
    end
    if !iszero(insufficient_coordinate)
        return _publish_status!(output, workspace, _diagnostics(
            T,
            InsufficientCorrectorVariance,
            input,
            minimum_variance,
            maximum_variance,
            zero(T),
            zero(T),
            insufficient_coordinate,
            0,
        ))
    end

    maximum_correlation = zero(T)
    maximum_row = 0
    maximum_column = 0
    for row in axes(covariance, 1)
        row_deviation = sqrt(covariance[row, row])
        for column in (row + 1):lastindex(covariance, 2)
            column_deviation = sqrt(covariance[column, column])
            correlation = (abs(covariance[row, column]) / row_deviation) /
                column_deviation
            isfinite(correlation) || return NonFiniteCorrelation
            if correlation > maximum_correlation
                maximum_correlation = correlation
                maximum_row = row
                maximum_column = column
            end
        end
    end
    if maximum_correlation > plan.criteria.maximum_absolute_corrector_correlation
        return _publish_status!(output, workspace, _diagnostics(
            T,
            ExcessiveCorrectorCorrelation,
            input,
            minimum_variance,
            maximum_variance,
            maximum_correlation,
            zero(T),
            maximum_row,
            maximum_column,
        ))
    end

    fill!(workspace.factor, zero(T))
    minimum_pivot = one(T)
    for row in axes(workspace.factor, 1)
        for column in firstindex(workspace.factor, 2):row
            factor_sum = zero(T)
            for index in firstindex(workspace.factor, 2):(column - 1)
                factor_sum += workspace.factor[row, index] * workspace.factor[column, index]
            end
            coefficient = covariance[row, column] - factor_sum
            isfinite(coefficient) || return NonFiniteFactorization
            if row == column
                if coefficient <= zero(T)
                    return _publish_status!(output, workspace, _diagnostics(
                        T,
                        NonPositiveDefinite,
                        input,
                        minimum_variance,
                        maximum_variance,
                        maximum_correlation,
                        zero(T),
                        row,
                        0,
                    ))
                end
                pivot_ratio = coefficient / maximum_variance
                isfinite(pivot_ratio) || return NonFiniteFactorization
                minimum_pivot = min(minimum_pivot, pivot_ratio)
                if pivot_ratio < plan.criteria.minimum_factorization_pivot_ratio
                    return _publish_status!(output, workspace, _diagnostics(
                        T,
                        InsufficientFactorizationPivot,
                        input,
                        minimum_variance,
                        maximum_variance,
                        maximum_correlation,
                        minimum_pivot,
                        row,
                        0,
                    ))
                end
                workspace.factor[row, column] = sqrt(coefficient)
            else
                workspace.factor[row, column] =
                    coefficient / workspace.factor[column, column]
            end
            isfinite(workspace.factor[row, column]) || return NonFiniteFactorization
        end
    end

    staged = workspace.staged_interaction_matrix
    for measurement in axes(staged, 1)
        for corrector in axes(staged, 2)
            value = cross[measurement, corrector] *
                normalization * plan.response_scale
            isfinite(value) || return NonFiniteSolve
            workspace.solve_scratch[corrector] = value
        end
        for row in eachindex(workspace.solve_scratch)
            value = workspace.solve_scratch[row]
            for column in firstindex(workspace.solve_scratch):(row - 1)
                value -= workspace.factor[row, column] * workspace.solve_scratch[column]
            end
            value /= workspace.factor[row, row]
            isfinite(value) || return NonFiniteSolve
            workspace.solve_scratch[row] = value
        end
        for row in reverse(eachindex(workspace.solve_scratch))
            value = workspace.solve_scratch[row]
            for column in (row + 1):lastindex(workspace.solve_scratch)
                value -= workspace.factor[column, row] * staged[measurement, column]
            end
            value /= workspace.factor[row, row]
            isfinite(value) || return NonFiniteSolve
            staged[measurement, row] = value
        end
    end
    all(isfinite, staged) || return NonFiniteSolve
    diagnostics = _diagnostics(
        T,
        Valid,
        input,
        minimum_variance,
        maximum_variance,
        maximum_correlation,
        minimum_pivot,
    )
    copyto!(output, staged)
    workspace.diagnostics = diagnostics
    return nothing
end

function _failure_message(failure::InteractionMatrixFailure)
    failure === NonFiniteCrossCovariance &&
        return "measurement-corrector covariance sum must contain only finite values"
    failure === NonFiniteCorrectorCovariance &&
        return "corrector covariance sum must contain only finite values"
    failure === NonSymmetricCorrectorCovariance &&
        return "corrector covariance sum must be symmetric"
    failure === SampleCountNotRepresentable &&
        return "sample count is not represented exactly by the plan element type"
    failure === SampleCountOutOfRange && return "sample count must fit Int"
    failure === NonFiniteNormalization &&
        return "DO-CRIME covariance normalization produced a non-finite value"
    failure === NonFiniteCorrelation &&
        return "DO-CRIME correlation calculation produced a non-finite value"
    failure === NonFiniteFactorization &&
        return "DO-CRIME factorization produced a non-finite value"
    return "DO-CRIME solve produced non-finite values"
end

function process!(
    result::InteractionMatrixProduct{T},
    workspace::InteractionMatrixWorkspace{T},
    plan::InteractionMatrixPlan{T},
    input::CovarianceSums,
) where {T}
    failure = process!(result.matrix, workspace, plan, input)
    failure === nothing || throw(ArgumentError(_failure_message(failure)))
    result.diagnostics = workspace.diagnostics
    return result
end
