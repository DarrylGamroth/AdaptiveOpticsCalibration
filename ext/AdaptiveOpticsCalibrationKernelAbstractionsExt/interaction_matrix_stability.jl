using KernelAbstractions: @Const, @index, @kernel

struct InteractionMatrixStabilityKernelPlan{T,P,B,K} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    workgroup_size::Int
end

function InteractionMatrixStabilityKernelPlan(
    plan::Diagnostics.InteractionMatrixStabilityPlan{T},
    backend,
    kernel,
    workgroup_size,
) where {T}
    return InteractionMatrixStabilityKernelPlan{
        T,
        typeof(plan),
        typeof(backend),
        typeof(kernel),
    }(plan, backend, kernel, workgroup_size)
end

struct InteractionMatrixStabilityKernelWorkspace{T,D,H}
    first_column_squares::D
    second_column_squares::D
    difference_column_squares::D
    host_first_column_squares::H
    host_second_column_squares::H
    host_difference_column_squares::H
end

@kernel function interaction_matrix_column_squares_kernel!(
    first_column_squares,
    second_column_squares,
    difference_column_squares,
    @Const(first_window),
    @Const(second_window),
    measurement_count,
)
    corrector = @index(Global, Linear)
    first_squares = zero(eltype(first_column_squares))
    second_squares = zero(eltype(second_column_squares))
    difference_squares = zero(eltype(difference_column_squares))
    for measurement in 1:measurement_count
        first_value = @inbounds first_window[measurement, corrector]
        second_value = @inbounds second_window[measurement, corrector]
        difference = first_value - second_value
        first_squares += abs2(first_value)
        second_squares += abs2(second_value)
        difference_squares += abs2(difference)
    end
    @inbounds begin
        first_column_squares[corrector] = first_squares
        second_column_squares[corrector] = second_squares
        difference_column_squares[corrector] = difference_squares
    end
end

function AOC.prepare(
    method::Diagnostics.InteractionMatrixStability{T},
    execution::AOC.KernelExecution{S,B},
) where {
    T<:AbstractFloat,
    S<:Diagnostics.InteractionMatrixPair{T},
    B<:KernelAbstractions.Backend,
}
    plan = AOC.prepare(method, execution.specification)
    return InteractionMatrixStabilityKernelPlan(
        plan,
        execution.backend,
        interaction_matrix_column_squares_kernel!(
            execution.backend,
            execution.workgroup_size,
        ),
        execution.workgroup_size,
    )
end

function AOC.allocate_result(plan::InteractionMatrixStabilityKernelPlan)
    return AOC.allocate_result(plan.plan)
end

function AOC.allocate_workspace(
    plan::InteractionMatrixStabilityKernelPlan{T},
) where {T}
    count = plan.plan.corrector_coordinate_count
    first_squares = KernelAbstractions.allocate(plan.backend, T, count)
    second_squares = KernelAbstractions.allocate(plan.backend, T, count)
    difference_squares = KernelAbstractions.allocate(plan.backend, T, count)
    return InteractionMatrixStabilityKernelWorkspace{T,typeof(first_squares),Vector{T}}(
        first_squares,
        second_squares,
        difference_squares,
        zeros(T, count),
        zeros(T, count),
        zeros(T, count),
    )
end

function validate_stability_kernel_storage(result, workspace, plan, input)
    expected = (
        plan.plan.measurement_count,
        plan.plan.corrector_coordinate_count,
    )
    size(input.first_window) == expected || throw(DimensionMismatch(
        "first interaction-matrix window must have shape $expected",
    ))
    size(input.second_window) == expected || throw(DimensionMismatch(
        "second interaction-matrix window must have shape $expected",
    ))
    count = plan.plan.corrector_coordinate_count
    for values in (
        workspace.first_column_squares,
        workspace.second_column_squares,
        workspace.difference_column_squares,
        workspace.host_first_column_squares,
        workspace.host_second_column_squares,
        workspace.host_difference_column_squares,
    )
        length(values) == count || throw(DimensionMismatch(
            "interaction-matrix stability workspace has an incompatible extent",
        ))
    end
    require_one_based(
        input.first_window,
        input.second_window,
        workspace.first_column_squares,
        workspace.second_column_squares,
        workspace.difference_column_squares,
    )
    require_backend(
        plan.backend,
        input.first_window,
        input.second_window,
        workspace.first_column_squares,
        workspace.second_column_squares,
        workspace.difference_column_squares,
    )
    return nothing
end

function finish_stability_kernel!(result, workspace, plan)
    T = eltype(workspace.host_first_column_squares)
    first_squares = workspace.host_first_column_squares
    second_squares = workspace.host_second_column_squares
    difference_squares = workspace.host_difference_column_squares
    all(isfinite, first_squares) && all(isfinite, second_squares) &&
        all(isfinite, difference_squares) || throw(ArgumentError(
            "interaction-matrix stability calculation produced a non-finite value",
        ))

    minimum_first = T(Inf)
    minimum_second = T(Inf)
    maximum_relative = zero(T)
    maximum_coordinate = 1
    first_insufficient_coordinate = 0
    total_first_squares = zero(T)
    total_second_squares = zero(T)
    total_difference_squares = zero(T)
    two = T(2)
    for coordinate in eachindex(first_squares, second_squares, difference_squares)
        first_value = first_squares[coordinate]
        second_value = second_squares[coordinate]
        difference_value = difference_squares[coordinate]
        total_first_squares += first_value
        total_second_squares += second_value
        total_difference_squares += difference_value
        first_norm = sqrt(first_value)
        second_norm = sqrt(second_value)
        minimum_first = min(minimum_first, first_norm)
        minimum_second = min(minimum_second, second_norm)
        if first_norm < plan.plan.criteria.minimum_column_norm ||
           second_norm < plan.plan.criteria.minimum_column_norm
            iszero(first_insufficient_coordinate) &&
                (first_insufficient_coordinate = coordinate)
            continue
        end
        relative = two * sqrt(difference_value) / (first_norm + second_norm)
        isfinite(relative) || throw(ArgumentError(
            "interaction-matrix stability calculation produced a non-finite ratio",
        ))
        if relative > maximum_relative
            maximum_relative = relative
            maximum_coordinate = coordinate
        end
    end

    all(isfinite, (
        total_first_squares,
        total_second_squares,
        total_difference_squares,
    )) || throw(ArgumentError(
        "interaction-matrix stability calculation produced a non-finite total",
    ))
    total_norm = sqrt(total_first_squares) + sqrt(total_second_squares)
    relative_frobenius = iszero(total_norm) ? zero(T) :
        two * sqrt(total_difference_squares) / total_norm
    isfinite(relative_frobenius) || throw(ArgumentError(
        "interaction-matrix stability calculation produced a non-finite Frobenius ratio",
    ))

    status, status_coordinate = if !iszero(first_insufficient_coordinate)
        (Diagnostics.InsufficientColumnNorm, first_insufficient_coordinate)
    elseif maximum_relative >
           plan.plan.criteria.maximum_relative_column_difference
        (Diagnostics.ExcessiveRelativeColumnDifference, maximum_coordinate)
    elseif relative_frobenius >
           plan.plan.criteria.maximum_relative_frobenius_difference
        (Diagnostics.ExcessiveRelativeFrobeniusDifference, 0)
    else
        (Diagnostics.Stable, 0)
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

function AOC.process!(
    result::Diagnostics.InteractionMatrixStabilityProduct{T},
    workspace::InteractionMatrixStabilityKernelWorkspace{T},
    plan::InteractionMatrixStabilityKernelPlan{T},
    input::Diagnostics.InteractionMatrixPair{T},
) where {T}
    validate_stability_kernel_storage(result, workspace, plan, input)
    plan.kernel(
        workspace.first_column_squares,
        workspace.second_column_squares,
        workspace.difference_column_squares,
        input.first_window,
        input.second_window,
        plan.plan.measurement_count;
        ndrange=plan.plan.corrector_coordinate_count,
    )
    KernelAbstractions.synchronize(plan.backend)
    copyto!(
        workspace.host_first_column_squares,
        workspace.first_column_squares,
    )
    copyto!(
        workspace.host_second_column_squares,
        workspace.second_column_squares,
    )
    copyto!(
        workspace.host_difference_column_squares,
        workspace.difference_column_squares,
    )
    KernelAbstractions.synchronize(plan.backend)
    return finish_stability_kernel!(result, workspace, plan)
end
