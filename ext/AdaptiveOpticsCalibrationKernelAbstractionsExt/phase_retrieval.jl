using KernelAbstractions: @index, @kernel

struct LiFTKernelExecution{B,I,K}
    backend::B
    mode_indices::I
    kernels::K
    workgroup_size::Int
end

@kernel function lift_copy_rate_kernel!(destination, source, scale, count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds destination[index] = source[index] * scale
    end
end

@kernel function lift_residual_kernel!(destination, observation, model, count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds destination[index] = observation[index] - model[index]
    end
end

@kernel function lift_inverse_sqrt_variance_kernel!(
    destination,
    values,
    scale,
    offset,
    floor_value,
    count,
)
    index = @index(Global, Linear)
    if index <= count
        variance = @inbounds values[index] * scale + offset
        @inbounds destination[index] = sqrt(inv(max(variance, floor_value)))
    end
end

@kernel function lift_apply_weights_kernel!(jacobian, residual, weights, rows, columns)
    row, column = @index(Global, NTuple)
    if row <= rows && column <= columns
        weight = @inbounds weights[row]
        @inbounds jacobian[row, column] *= weight
        if column == 1
            @inbounds residual[row] *= weight
        end
    end
end

@kernel function lift_scale_array_kernel!(values, scale, count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds values[index] *= scale
    end
end

@kernel function lift_shift_coefficient_kernel!(coefficients, mode_indices, column, delta)
    if @index(Global, Linear) == 1
        coefficient_index = @inbounds mode_indices[column]
        @inbounds coefficients[coefficient_index] += delta
    end
end

@kernel function lift_jacobian_column_kernel!(jacobian, plus_rate, minus_rate, column, divisor, rows)
    row = @index(Global, Linear)
    if row <= rows
        @inbounds jacobian[row, column] = (plus_rate[row] - minus_rate[row]) / divisor
    end
end

@kernel function lift_scatter_update_kernel!(coefficients, update, mode_indices, count)
    index = @index(Global, Linear)
    if index <= count
        coefficient_index = @inbounds mode_indices[index]
        @inbounds coefficients[coefficient_index] += update[index]
    end
end

@kernel function lift_gather_kernel!(destination, coefficients, mode_indices, count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds destination[index] = coefficients[mode_indices[index]]
    end
end

@kernel function lift_map_reduction_kernel!(destination, source, operation, count)
    index = @index(Global, Linear)
    if index <= count
        value = @inbounds source[index]
        mapped = operation == UInt8(1) ? value :
            operation == UInt8(2) ? value * value : abs(value)
        @inbounds destination[index] = mapped
    end
end

@kernel function lift_map_diagonal_abs_kernel!(destination, matrix, count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds destination[index] = abs(matrix[index, index])
    end
end

@kernel function lift_map_scaled_square_kernel!(destination, source, scale, count)
    index = @index(Global, Linear)
    if index <= count
        value = @inbounds source[index] / scale
        @inbounds destination[index] = value * value
    end
end

@kernel function lift_pair_reduction_kernel!(destination, source, operation, count)
    index = @index(Global, Linear)
    source_index = 2 * index - 1
    first_value = @inbounds source[source_index]
    value = first_value
    if source_index < count
        second_value = @inbounds source[source_index + 1]
        value = operation == UInt8(1) ? first_value + second_value :
            operation == UInt8(2) ? max(first_value, second_value) :
            min(first_value, second_value)
    end
    @inbounds destination[index] = value
end

@kernel function lift_invalid_kernel!(destination, source, require_nonnegative, count)
    index = @index(Global, Linear)
    if index <= count
        value = @inbounds source[index]
        valid = isfinite(value) && (!require_nonnegative || value >= zero(value))
        @inbounds destination[index] = valid ? UInt32(0) : UInt32(1)
    end
end

@kernel function lift_invalid_pair_kernel!(destination, source, count)
    index = @index(Global, Linear)
    source_index = 2 * index - 1
    value = @inbounds source[source_index]
    if source_index < count
        value |= @inbounds source[source_index + 1]
    end
    @inbounds destination[index] = value
end

function _lift_kernel_set(backend, workgroup_size)
    return (
        copy_rate=lift_copy_rate_kernel!(backend, workgroup_size),
        residual=lift_residual_kernel!(backend, workgroup_size),
        inverse_sqrt_variance=
            lift_inverse_sqrt_variance_kernel!(backend, workgroup_size),
        apply_weights=lift_apply_weights_kernel!(backend, workgroup_size),
        scale_array=lift_scale_array_kernel!(backend, workgroup_size),
        shift_coefficient=lift_shift_coefficient_kernel!(backend, 1),
        jacobian_column=lift_jacobian_column_kernel!(backend, workgroup_size),
        scatter_update=lift_scatter_update_kernel!(backend, workgroup_size),
        gather=lift_gather_kernel!(backend, workgroup_size),
        map_reduction=lift_map_reduction_kernel!(backend, workgroup_size),
        map_diagonal_abs=lift_map_diagonal_abs_kernel!(backend, workgroup_size),
        map_scaled_square=lift_map_scaled_square_kernel!(backend, workgroup_size),
        pair_reduction=lift_pair_reduction_kernel!(backend, workgroup_size),
        invalid=lift_invalid_kernel!(backend, workgroup_size),
        invalid_pair=lift_invalid_pair_kernel!(backend, workgroup_size),
    )
end

_validate_lift_kernel_solve_mode(::PhaseRetrieval.AbstractLiFTSolveMode) = nothing
_validate_lift_kernel_solve_mode(::PhaseRetrieval.LiFTSolveQR) = throw(
    ArgumentError("LiFT KernelExecution does not support an explicit QR solve"),
)

_require_lift_weighting_backend(backend, ::PhaseRetrieval.AbstractLiFTWeighting) =
    nothing
_require_lift_weighting_backend(
    backend,
    weighting::PhaseRetrieval.LiFTVarianceMapWeighting,
) = require_backend(backend, weighting.variance)

function AOC.prepare(
    method::PhaseRetrieval.LiFT,
    execution::AOC.KernelExecution{S,B},
) where {
    S<:PhaseRetrieval.LiFTSpecification,
    B<:KernelAbstractions.Backend,
}
    _validate_lift_kernel_solve_mode(method.solve_mode)
    plan = AOC.prepare(method, execution.specification)
    output = PhaseRetrieval.allocate_photon_rate(plan.model)
    require_backend(execution.backend, output)
    _require_lift_weighting_backend(execution.backend, plan.weighting)
    mode_indices = copy_to_backend(execution.backend, plan.mode_indices)
    KernelAbstractions.synchronize(execution.backend)
    prepared_execution = LiFTKernelExecution(
        execution.backend,
        mode_indices,
        _lift_kernel_set(execution.backend, execution.workgroup_size),
        execution.workgroup_size,
    )
    return PhaseRetrieval.LiFTKernelPlan(plan, prepared_execution)
end

AOC.allocate_result(plan::PhaseRetrieval.LiFTKernelPlan) =
    AOC.allocate_result(plan.plan)

function AOC.allocate_workspace(plan::PhaseRetrieval.LiFTKernelPlan{P}) where {T,P<:PhaseRetrieval.LiFTPlan{T}}
    inverse = AOC.allocate_workspace(plan.plan)
    reduction_capacity = max(
        plan.plan.observation_count,
        plan.plan.coefficient_count,
        length(plan.plan.mode_indices),
    )
    invalid_capacity = max(
        reduction_capacity,
        length(inverse.jacobian),
    )
    reduction_first = KernelAbstractions.allocate(
        plan.execution.backend,
        T,
        reduction_capacity,
    )
    reduction_second = similar(reduction_first)
    invalid_first = KernelAbstractions.allocate(
        plan.execution.backend,
        UInt32,
        invalid_capacity,
    )
    invalid_second = similar(invalid_first)
    return PhaseRetrieval.LiFTKernelWorkspace(
        inverse,
        reduction_first,
        reduction_second,
        invalid_first,
        invalid_second,
        Vector{T}(undef, 1),
        Vector{UInt32}(undef, 1),
    )
end

function _lift_reduce!(workspace, plan, source, count, map_operation, pair_operation)
    plan.execution.kernels.map_reduction(
        workspace.reduction_first,
        source,
        map_operation,
        count;
        ndrange=count,
    )
    KernelAbstractions.synchronize(plan.execution.backend)
    first = workspace.reduction_first
    second = workspace.reduction_second
    while count > 1
        next_count = cld(count, 2)
        plan.execution.kernels.pair_reduction(
            second,
            first,
            pair_operation,
            count;
            ndrange=next_count,
        )
        KernelAbstractions.synchronize(plan.execution.backend)
        first, second = second, first
        count = next_count
    end
    copyto!(workspace.host_scalar, 1, first, 1, 1)
    KernelAbstractions.synchronize(plan.execution.backend)
    return workspace.host_scalar[1]
end

_lift_sum!(workspace, plan, source, count=length(source)) =
    _lift_reduce!(workspace, plan, source, count, UInt8(1), UInt8(1))
function _lift_norm!(workspace, plan, source, count=length(source))
    maximum_absolute = _lift_reduce!(
        workspace,
        plan,
        source,
        count,
        UInt8(3),
        UInt8(2),
    )
    iszero(maximum_absolute) && return zero(maximum_absolute)
    plan.execution.kernels.map_scaled_square(
        workspace.reduction_first,
        source,
        maximum_absolute,
        count;
        ndrange=count,
    )
    scaled_sum = _lift_reduce!(
        workspace,
        plan,
        workspace.reduction_first,
        count,
        UInt8(1),
        UInt8(1),
    )
    return maximum_absolute * sqrt(scaled_sum)
end
_lift_maximum!(workspace, plan, source, count=length(source)) =
    _lift_reduce!(workspace, plan, source, count, UInt8(1), UInt8(2))

function _lift_diagonal_reduce!(workspace, plan, matrix, pair_operation)
    count = min(size(matrix)...)
    plan.execution.kernels.map_diagonal_abs(
        workspace.reduction_first,
        matrix,
        count;
        ndrange=count,
    )
    KernelAbstractions.synchronize(plan.execution.backend)
    return _lift_reduce!(
        workspace,
        plan,
        workspace.reduction_first,
        count,
        UInt8(1),
        pair_operation,
    )
end

function _lift_all_valid!(workspace, plan, source; nonnegative::Bool=false)
    count = length(source)
    plan.execution.kernels.invalid(
        workspace.invalid_first,
        source,
        nonnegative,
        count;
        ndrange=count,
    )
    KernelAbstractions.synchronize(plan.execution.backend)
    first = workspace.invalid_first
    second = workspace.invalid_second
    while count > 1
        next_count = cld(count, 2)
        plan.execution.kernels.invalid_pair(
            second,
            first,
            count;
            ndrange=next_count,
        )
        KernelAbstractions.synchronize(plan.execution.backend)
        first, second = second, first
        count = next_count
    end
    copyto!(workspace.host_invalid, 1, first, 1, 1)
    KernelAbstractions.synchronize(plan.execution.backend)
    return iszero(workspace.host_invalid[1])
end

function _lift_validate_kernel_storage(result, workspace, plan, inputs)
    inverse_plan = plan.plan
    inverse_workspace = workspace.inverse
    PhaseRetrieval._validate_result_workspace_structure(
        inverse_plan,
        result,
        inverse_workspace,
        inputs,
    )
    inverse_arrays = PhaseRetrieval._workspace_arrays(inverse_workspace)
    device_scratch = (
        workspace.reduction_first,
        workspace.reduction_second,
        workspace.invalid_first,
        workspace.invalid_second,
    )
    host_scratch = (workspace.host_scalar, workspace.host_invalid)
    scratch = (inverse_arrays..., device_scratch..., host_scratch...)
    plan_arrays = (
        PhaseRetrieval._plan_arrays(inverse_plan.weighting)...,
        plan.execution.mode_indices,
    )
    Base.require_one_based_indexing(
        result.coefficients,
        inputs.observation,
        inverse_arrays...,
        plan.execution.mode_indices,
        device_scratch...,
        host_scratch...,
    )
    eltype(workspace.reduction_first) === eltype(inverse_workspace.normal) &&
        eltype(workspace.reduction_second) === eltype(inverse_workspace.normal) ||
        throw(ArgumentError("LiFT reduction scratch must use the prepared numeric type"))
    eltype(workspace.invalid_first) === UInt32 &&
        eltype(workspace.invalid_second) === UInt32 &&
        eltype(workspace.host_invalid) === UInt32 || throw(ArgumentError(
        "LiFT validity scratch must use UInt32 elements",
    ))
    eltype(workspace.host_scalar) === eltype(inverse_workspace.normal) || throw(
        ArgumentError("LiFT host scalar scratch must use the prepared numeric type"),
    )
    length(workspace.host_scalar) == 1 && length(workspace.host_invalid) == 1 ||
        throw(DimensionMismatch("LiFT host scalar scratch must have length one"))
    KernelAbstractions.get_backend(workspace.host_scalar) ==
        KernelAbstractions.CPU() || throw(ArgumentError(
        "LiFT host scalar scratch must use CPU storage",
    ))
    KernelAbstractions.get_backend(workspace.host_invalid) ==
        KernelAbstractions.CPU() || throw(ArgumentError(
        "LiFT host validity scratch must use CPU storage",
    ))
    require_backend(
        plan.execution.backend,
        result.coefficients,
        inputs.observation,
        inverse_arrays...,
    )
    if !isnothing(inputs.initial_coefficients)
        require_backend(plan.execution.backend, inputs.initial_coefficients)
    end
    require_backend(plan.execution.backend, plan.execution.mode_indices, device_scratch...)
    reduction_capacity = max(
        inverse_plan.observation_count,
        inverse_plan.coefficient_count,
        length(inverse_plan.mode_indices),
    )
    invalid_capacity = max(reduction_capacity, length(inverse_workspace.jacobian))
    length(workspace.reduction_first) >= reduction_capacity &&
        length(workspace.reduction_second) >= reduction_capacity || throw(
        DimensionMismatch("LiFT reduction scratch is smaller than the prepared execution extent"),
    )
    length(workspace.invalid_first) >= invalid_capacity &&
        length(workspace.invalid_second) >= invalid_capacity || throw(
        DimensionMismatch("LiFT validity scratch is smaller than the prepared execution extent"),
    )
    PhaseRetrieval._mightalias_any(result.coefficients, plan_arrays) && throw(
        ArgumentError("LiFT product storage must not alias kernel-plan storage"),
    )
    PhaseRetrieval._any_cross_alias(scratch, plan_arrays) && throw(
        ArgumentError("LiFT workspace storage must not alias kernel-plan storage"),
    )
    PhaseRetrieval._mightalias_any(result.coefficients, device_scratch) && throw(
        ArgumentError("LiFT product storage must not alias kernel scratch"),
    )
    PhaseRetrieval._mightalias_any(inputs.observation, device_scratch) && throw(
        ArgumentError("LiFT observation storage must not alias kernel scratch"),
    )
    if !isnothing(inputs.initial_coefficients)
        PhaseRetrieval._mightalias_any(inputs.initial_coefficients, device_scratch) &&
            throw(ArgumentError(
                "LiFT initial coefficients must not alias kernel scratch",
            ))
    end
    PhaseRetrieval._any_pair_alias(scratch) && throw(
        ArgumentError("LiFT inverse and kernel scratch arrays must not alias"),
    )
    return nothing
end

function _lift_model_scale!(workspace, plan, ::PhaseRetrieval.LiFTTotalRateMatching)
    observation_sum = _lift_sum!(workspace, plan, workspace.inverse.observation_rate)
    model_sum = _lift_sum!(workspace, plan, workspace.inverse.model_rate)
    T = eltype(workspace.inverse.model_rate)
    scale = model_sum > zero(T) ? observation_sum / model_sum : one(T)
    plan.execution.kernels.scale_array(
        workspace.inverse.model_rate,
        scale,
        plan.plan.observation_count;
        ndrange=plan.plan.observation_count,
    )
    return scale, max(abs(model_sum * scale), one(T))
end


function _lift_model_scale!(workspace, plan, ::PhaseRetrieval.LiFTPeakRateMatching)
    observation_peak = _lift_maximum!(workspace, plan, workspace.inverse.observation_rate)
    model_peak = _lift_maximum!(workspace, plan, workspace.inverse.model_rate)
    T = eltype(workspace.inverse.model_rate)
    scale = model_peak > zero(T) ? observation_peak / model_peak : one(T)
    plan.execution.kernels.scale_array(
        workspace.inverse.model_rate,
        scale,
        plan.plan.observation_count;
        ndrange=plan.plan.observation_count,
    )
    model_sum = _lift_sum!(workspace, plan, workspace.inverse.model_rate)
    return scale, max(abs(model_sum), one(T))
end

function _lift_model_scale!(workspace, plan, ::PhaseRetrieval.LiFTPhysicalRatePreservation)
    T = eltype(workspace.inverse.model_rate)
    model_sum = _lift_sum!(workspace, plan, workspace.inverse.model_rate)
    return one(T), max(abs(model_sum), one(T))
end

function _lift_static_weights!(workspace, plan, ::Union{
    PhaseRetrieval.LiFTInitialModelWeighting,
    PhaseRetrieval.LiFTIterativeModelWeighting,
})
    return nothing
end

function _lift_static_weights!(workspace, plan, ::PhaseRetrieval.LiFTReadNoiseWeighting)
    T = eltype(workspace.inverse.sqrt_weights)
    fill!(
        workspace.inverse.sqrt_weights,
        sqrt(inv(max(plan.plan.read_noise_rate^2, eps(T)))),
    )
    return nothing
end

function _lift_static_weights!(workspace, plan, weighting::PhaseRetrieval.LiFTVarianceMapWeighting)
    count = plan.plan.observation_count
    T = eltype(workspace.inverse.sqrt_weights)
    plan.execution.kernels.inverse_sqrt_variance(
        workspace.inverse.sqrt_weights,
        weighting.variance,
        plan.plan.observation_to_rate_scale^2,
        zero(T),
        eps(T),
        count;
        ndrange=count,
    )
    return nothing
end

function _lift_update_weights!(workspace, plan, ::Union{
    PhaseRetrieval.LiFTReadNoiseWeighting,
    PhaseRetrieval.LiFTVarianceMapWeighting,
}, ::Int)
    return nothing
end

function _lift_update_weights!(workspace, plan, ::PhaseRetrieval.LiFTInitialModelWeighting, iteration::Int)
    isone(iteration) || return nothing
    return _lift_model_weights!(workspace, plan)
end

_lift_update_weights!(workspace, plan, ::PhaseRetrieval.LiFTIterativeModelWeighting, ::Int) =
    _lift_model_weights!(workspace, plan)

function _lift_model_weights!(workspace, plan)
    count = plan.plan.observation_count
    T = eltype(workspace.inverse.sqrt_weights)
    plan.execution.kernels.inverse_sqrt_variance(
        workspace.inverse.sqrt_weights,
        workspace.inverse.model_rate,
        plan.plan.shot_variance_rate_scale,
        plan.plan.read_noise_rate^2,
        eps(T),
        count;
        ndrange=count,
    )
    return nothing
end

function _lift_jacobian!(workspace, plan, ::PhaseRetrieval.LiFTAnalyticJacobian)
    return PhaseRetrieval.analytic_photon_rate_jacobian!(
        workspace.inverse.jacobian,
        plan.plan.model,
        workspace.inverse.model_workspace,
        workspace.inverse.coefficients,
        plan.plan.mode_indices,
    )
end

function _lift_jacobian!(workspace, plan, method::PhaseRetrieval.LiFTNumericalJacobian)
    T = eltype(workspace.inverse.jacobian)
    step = T(method.step_m)
    divisor = T(2) * step
    for column in eachindex(plan.plan.mode_indices)
        plan.execution.kernels.shift_coefficient(
            workspace.inverse.coefficients,
            plan.execution.mode_indices,
            column,
            step;
            ndrange=1,
        )
        PhaseRetrieval.predict_photon_rate!(
            workspace.inverse.plus_rate,
            plan.plan.model,
            workspace.inverse.model_workspace,
            workspace.inverse.coefficients,
        )
        plan.execution.kernels.shift_coefficient(
            workspace.inverse.coefficients,
            plan.execution.mode_indices,
            column,
            -divisor;
            ndrange=1,
        )
        PhaseRetrieval.predict_photon_rate!(
            workspace.inverse.minus_rate,
            plan.plan.model,
            workspace.inverse.model_workspace,
            workspace.inverse.coefficients,
        )
        plan.execution.kernels.shift_coefficient(
            workspace.inverse.coefficients,
            plan.execution.mode_indices,
            column,
            step;
            ndrange=1,
        )
        plan.execution.kernels.jacobian_column(
            workspace.inverse.jacobian,
            workspace.inverse.plus_rate,
            workspace.inverse.minus_rate,
            column,
            divisor,
            plan.plan.observation_count;
            ndrange=plan.plan.observation_count,
        )
    end
    return workspace.inverse.jacobian
end

function _lift_normal_metrics!(workspace, plan)
    maximum_diagonal = _lift_diagonal_reduce!(
        workspace, plan, workspace.inverse.normal, UInt8(2))
    minimum_diagonal = _lift_diagonal_reduce!(
        workspace, plan, workspace.inverse.normal, UInt8(3))
    T = eltype(workspace.inverse.normal)
    condition = iszero(maximum_diagonal) ? T(Inf) :
        maximum_diagonal / max(minimum_diagonal, eps(T))
    diagonal_sum = _lift_diagonal_reduce!(
        workspace, plan, workspace.inverse.normal, UInt8(1))
    mean_diagonal = diagonal_sum / T(size(workspace.inverse.normal, 1))
    regularization_load = max(sqrt(eps(T)) * max(mean_diagonal, one(T)), eps(T))
    return condition, regularization_load
end

function AOC.process!(
    result::PhaseRetrieval.LiFTProduct{T},
    workspace::PhaseRetrieval.LiFTKernelWorkspace,
    plan::PhaseRetrieval.LiFTKernelPlan{P},
    inputs::PhaseRetrieval.LiFTInputs{T},
) where {T<:AbstractFloat,P<:PhaseRetrieval.LiFTPlan{T}}
    _lift_validate_kernel_storage(result, workspace, plan, inputs)
    inverse = workspace.inverse
    count = plan.plan.observation_count
    _lift_all_valid!(workspace, plan, inputs.observation; nonnegative=true) ||
        throw(ArgumentError("LiFT observation must contain only finite nonnegative values"))
    if !isnothing(inputs.initial_coefficients)
        _lift_all_valid!(workspace, plan, inputs.initial_coefficients) ||
            throw(ArgumentError("LiFT initial_coefficients must contain only finite values"))
    end
    plan.execution.kernels.copy_rate(
        inverse.observation_rate,
        inputs.observation,
        plan.plan.observation_to_rate_scale,
        count;
        ndrange=count,
    )
    _lift_all_valid!(workspace, plan, inverse.observation_rate; nonnegative=true) ||
        throw(ArgumentError(
            "LiFT rate conversion must produce finite nonnegative observations",
        ))
    if isnothing(inputs.initial_coefficients)
        fill!(inverse.coefficients, zero(T))
    else
        copyto!(inverse.coefficients, inputs.initial_coefficients)
    end
    _lift_static_weights!(workspace, plan, plan.plan.weighting)
    damping_state = PhaseRetrieval._damping_state(plan.plan.damping, T)
    previous_weighted_norm = T(Inf)
    diagnostics = PhaseRetrieval.LiFTDiagnostics{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), 0, false, false, false,
    )
    for iteration in 1:plan.plan.iterations
        PhaseRetrieval.predict_photon_rate!(
            inverse.model_rate,
            plan.plan.model,
            inverse.model_workspace,
            inverse.coefficients,
        )
        _lift_all_valid!(workspace, plan, inverse.model_rate; nonnegative=true) ||
            throw(ArgumentError("LiFT forward model must predict finite nonnegative photon rates"))
        scale, objective_scale =
            _lift_model_scale!(workspace, plan, plan.plan.model_scaling)
        isfinite(scale) && scale >= zero(T) || throw(ArgumentError(
            "LiFT model scaling must produce a finite nonnegative scale",
        ))
        isfinite(objective_scale) && objective_scale > zero(T) ||
            throw(ArgumentError("LiFT objective scale must be finite and positive"))
        _lift_all_valid!(workspace, plan, inverse.model_rate; nonnegative=true) ||
            throw(ArgumentError(
                "LiFT model scaling must preserve finite nonnegative photon rates",
            ))
        plan.execution.kernels.residual(
            inverse.residual,
            inverse.observation_rate,
            inverse.model_rate,
            count;
            ndrange=count,
        )
        residual_norm = _lift_norm!(workspace, plan, inverse.residual)
        isfinite(residual_norm) ||
            throw(ArgumentError("LiFT residual norm must be finite"))
        _lift_update_weights!(workspace, plan, plan.plan.weighting, iteration)
        _lift_all_valid!(workspace, plan, inverse.sqrt_weights; nonnegative=true) ||
            throw(ArgumentError("LiFT weighting must produce finite nonnegative square-root weights"))
        _lift_jacobian!(workspace, plan, plan.plan.jacobian_method)
        _lift_all_valid!(workspace, plan, inverse.jacobian) || throw(ArgumentError(
            "LiFT forward-model Jacobian must contain only finite values",
        ))
        plan.execution.kernels.scale_array(
            inverse.jacobian,
            scale / objective_scale,
            length(inverse.jacobian);
            ndrange=length(inverse.jacobian),
        )
        _lift_all_valid!(workspace, plan, inverse.jacobian) || throw(ArgumentError(
            "LiFT Jacobian scaling must produce finite values",
        ))
        plan.execution.kernels.apply_weights(
            inverse.jacobian,
            inverse.residual,
            inverse.sqrt_weights,
            count,
            length(plan.plan.mode_indices);
            ndrange=(count, length(plan.plan.mode_indices)),
        )
        weighted_norm = _lift_norm!(workspace, plan, inverse.residual)
        isfinite(weighted_norm) || throw(ArgumentError(
            "LiFT weighted residual norm must be finite",
        ))
        _lift_all_valid!(workspace, plan, inverse.jacobian) &&
            _lift_all_valid!(workspace, plan, inverse.residual) ||
            throw(ArgumentError("LiFT weighted system must contain only finite values"))
        plan.execution.kernels.scale_array(
            inverse.residual,
            inv(objective_scale),
            count;
            ndrange=count,
        )
        mul!(inverse.normal, transpose(inverse.jacobian), inverse.jacobian)
        mul!(inverse.rhs, transpose(inverse.jacobian), inverse.residual)
        _lift_all_valid!(workspace, plan, inverse.normal) &&
            _lift_all_valid!(workspace, plan, inverse.rhs) ||
            throw(ArgumentError("LiFT normal system must contain only finite values"))
        damping_state = PhaseRetrieval._next_damping_state(
            plan.plan.damping,
            damping_state,
            previous_weighted_norm,
            weighted_norm,
        )
        condition, regularization_load = _lift_normal_metrics!(workspace, plan)
        !isnan(condition) && condition >= zero(T) || throw(ArgumentError(
            "LiFT condition ratio must be nonnegative and not NaN",
        ))
        isfinite(regularization_load) && regularization_load >= zero(T) ||
            throw(ArgumentError(
                "LiFT regularization load must be finite and nonnegative",
            ))
        regularization, used_fallback = PhaseRetrieval.solve_lift_normal_device!(
            workspace,
            plan,
            plan.plan.damping,
            damping_state,
            regularization_load,
        )
        isfinite(regularization) && regularization >= zero(T) ||
            throw(ArgumentError(
                "LiFT regularization must be finite and nonnegative",
            ))
        _lift_all_valid!(workspace, plan, inverse.rhs) || throw(ArgumentError(
            "LiFT solve must produce a finite coefficient update",
        ))
        update_norm = _lift_norm!(workspace, plan, inverse.rhs)
        isfinite(update_norm) ||
            throw(ArgumentError("LiFT update norm must be finite"))
        plan.execution.kernels.scatter_update(
            inverse.coefficients,
            inverse.rhs,
            plan.execution.mode_indices,
            length(plan.plan.mode_indices);
            ndrange=length(plan.plan.mode_indices),
        )
        _lift_all_valid!(workspace, plan, inverse.coefficients) || throw(ArgumentError(
            "LiFT coefficient update must produce finite values",
        ))
        coefficient_norm = _lift_norm!(workspace, plan, inverse.coefficients)
        isfinite(coefficient_norm) ||
            throw(ArgumentError("LiFT coefficient norm must be finite"))
        converged = plan.plan.check_convergence &&
            update_norm / max(coefficient_norm, eps(T)) <
            plan.plan.relative_update_tolerance
        diagnostics = PhaseRetrieval.LiFTDiagnostics{T}(
            residual_norm,
            weighted_norm,
            update_norm,
            condition,
            regularization,
            iteration,
            converged,
            false,
            used_fallback,
        )
        previous_weighted_norm = weighted_norm
        converged && break
    end
    plan.execution.kernels.gather(
        result.coefficients,
        inverse.coefficients,
        plan.execution.mode_indices,
        length(plan.plan.mode_indices);
        ndrange=length(plan.plan.mode_indices),
    )
    KernelAbstractions.synchronize(plan.execution.backend)
    result.diagnostics = diagnostics
    return result
end
