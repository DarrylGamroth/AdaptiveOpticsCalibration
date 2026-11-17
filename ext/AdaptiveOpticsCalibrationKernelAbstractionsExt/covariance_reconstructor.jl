struct CovarianceReconstructorKernelPlan{T<:AbstractFloat,B,F,I,X,U,S,Q,P} <:
       AOC.AbstractCalibrationPlan
    phase_count::Int
    measurement_count::Int
    fit_count::Int
    backend::B
    finite_kernel::F
    invalid_reduction_kernel::I
    maximum_reduction_kernel::X
    sum_reduction_kernel::U
    symmetry_kernel::S
    transpose_kernel::Q
    publish_kernel::P
    workgroup_size::Int
end

struct CovarianceReconstructorKernelWorkspace{M,V,U,I,H}
    phase_covariance::M
    measurement_noise_covariance::M
    multiplication::M
    system::M
    factor::M
    right_hand_side::M
    solution::M
    candidate::M
    invalid_first::U
    invalid_second::U
    pivot::I
    metric_first::V
    metric_second::V
    host_invalid::Vector{UInt32}
    host_metric::Vector{H}
end

@kernel function _covariance_finite_kernel!(invalid, @Const(matrix))
    index = @index(Global, Linear)
    @inbounds invalid[index] = isfinite(matrix[index]) ? UInt32(0) : UInt32(1)
end

@kernel function _covariance_maximum_abs_kernel!(destination, @Const(matrix))
    index = @index(Global, Linear)
    @inbounds destination[index] = abs(matrix[index])
end

@kernel function _covariance_scaled_square_kernel!(destination, @Const(matrix), scale)
    index = @index(Global, Linear)
    @inbounds destination[index] = abs2(matrix[index] / scale)
end

@kernel function _covariance_max_reduction_kernel!(destination, @Const(source), count)
    index = @index(Global, Linear)
    source_index = 2 * index - 1
    value = @inbounds source[source_index]
    if source_index < count
        value = max(value, @inbounds(source[source_index + 1]))
    end
    @inbounds destination[index] = value
end

@kernel function _covariance_sum_reduction_kernel!(destination, @Const(source), count)
    index = @index(Global, Linear)
    source_index = 2 * index - 1
    value = @inbounds source[source_index]
    if source_index < count
        value += @inbounds(source[source_index + 1])
    end
    @inbounds destination[index] = value
end

@kernel function _covariance_symmetrize_kernel!(invalid, matrix, tolerance, count)
    row, column = @index(Global, NTuple)
    index = row + (column - 1) * count
    if row < column
        first = @inbounds matrix[row, column]
        second = @inbounds matrix[column, row]
        if abs(first - second) <= tolerance
            average = first / 2 + second / 2
            @inbounds begin
                matrix[row, column] = average
                matrix[column, row] = average
                invalid[index] = UInt32(0)
            end
        else
            @inbounds invalid[index] = UInt32(1)
        end
    else
        @inbounds invalid[index] = UInt32(0)
    end
end

@kernel function _covariance_transpose_kernel!(destination, @Const(source))
    row, column = @index(Global, NTuple)
    @inbounds destination[row, column] = source[column, row]
end

function AOC.prepare(
    ::Tomography.CovarianceReconstructor,
    execution::AOC.KernelExecution{S,B},
) where {
    T<:AbstractFloat,
    S<:Tomography.CovarianceReconstructorSpecification{T},
    B<:KernelAbstractions.Backend,
}
    execution.backend isa KernelAbstractions.CPU && throw(ArgumentError(
        "KernelExecution covariance reconstruction requires an accelerator backend",
    ))
    (T === Float32 || T === Float64) || throw(ArgumentError(
        "accelerator covariance reconstruction supports Float32 and Float64",
    ))
    specification = execution.specification
    backend = execution.backend
    finite_kernel = _covariance_finite_kernel!(backend, execution.workgroup_size)
    invalid_reduction_kernel = invalid_or_reduction_kernel!(backend, execution.workgroup_size)
    maximum_reduction_kernel = _covariance_max_reduction_kernel!(backend,
        execution.workgroup_size)
    sum_reduction_kernel = _covariance_sum_reduction_kernel!(backend,
        execution.workgroup_size)
    symmetry_kernel = _covariance_symmetrize_kernel!(backend, execution.workgroup_size)
    transpose_kernel = _covariance_transpose_kernel!(backend, execution.workgroup_size)
    publish_kernel = publish_matrix_kernel!(backend, execution.workgroup_size)
    return CovarianceReconstructorKernelPlan{
        T,typeof(backend),typeof(finite_kernel),typeof(invalid_reduction_kernel),
        typeof(maximum_reduction_kernel),typeof(sum_reduction_kernel),
        typeof(symmetry_kernel),typeof(transpose_kernel),typeof(publish_kernel),
    }(
        specification.phase_count,
        specification.measurement_count,
        specification.fit_count,
        backend,
        finite_kernel,
        invalid_reduction_kernel,
        maximum_reduction_kernel,
        sum_reduction_kernel,
        symmetry_kernel,
        transpose_kernel,
        publish_kernel,
        execution.workgroup_size,
    )
end

function AOC.allocate_result(plan::CovarianceReconstructorKernelPlan{T}) where {T}
    return Tomography.CovarianceReconstructorProduct(
        KernelAbstractions.allocate(plan.backend, T, plan.fit_count, plan.measurement_count),
        Tomography.CovarianceReconstructorDiagnostics(false, zero(T), zero(T)),
    )
end

function _covariance_workspace_largest(plan)
    phase = plan.phase_count
    measurements = plan.measurement_count
    fit = plan.fit_count
    phase_covariance_count = AOC._checked_extent(
        "tomographic covariance phase workspace", phase, phase)
    measurement_covariance_count = AOC._checked_extent(
        "tomographic covariance measurement workspace", measurements, measurements)
    multiplication_count = AOC._checked_extent(
        "tomographic covariance multiplication workspace", measurements, phase)
    fit_phase_count = AOC._checked_extent(
        "tomographic covariance fit-phase workspace", fit, phase)
    reconstructor_count = AOC._checked_extent(
        "tomographic covariance reconstructor workspace", fit, measurements)
    return max(phase_covariance_count, measurement_covariance_count,
        multiplication_count, fit_phase_count, reconstructor_count)
end

function AOC.allocate_workspace(plan::CovarianceReconstructorKernelPlan{T}) where {T}
    backend = plan.backend
    phase = plan.phase_count
    measurements = plan.measurement_count
    fit = plan.fit_count
    largest = _covariance_workspace_largest(plan)
    workspace = CovarianceReconstructorKernelWorkspace(
        KernelAbstractions.allocate(backend, T, phase, phase),
        KernelAbstractions.allocate(backend, T, measurements, measurements),
        KernelAbstractions.allocate(backend, T, measurements, phase),
        KernelAbstractions.allocate(backend, T, measurements, measurements),
        KernelAbstractions.allocate(backend, T, measurements, measurements),
        KernelAbstractions.allocate(backend, T, fit, measurements),
        KernelAbstractions.allocate(backend, T, measurements, fit),
        KernelAbstractions.allocate(backend, T, fit, measurements),
        KernelAbstractions.allocate(backend, UInt32, largest),
        KernelAbstractions.allocate(backend, UInt32, max(1, cld(largest, 2))),
        KernelAbstractions.allocate(backend, Cint, measurements),
        KernelAbstractions.allocate(backend, T, largest),
        KernelAbstractions.allocate(backend, T, max(1, cld(largest, 2))),
        zeros(UInt32, 1),
        zeros(T, 1),
    )
    KernelAbstractions.synchronize(backend)
    return workspace
end

function _covariance_validate_matrix(matrix, shape, name)
    Base.require_one_based_indexing(matrix)
    size(matrix) == shape || throw(DimensionMismatch("$name must have shape $shape"))
    return nothing
end

function _covariance_validate_storage(result, workspace, plan, input)
    _covariance_validate_matrix(result.matrix,
        (plan.fit_count, plan.measurement_count), "tomographic covariance result")
    _covariance_validate_matrix(input.projection,
        (plan.measurement_count, plan.phase_count), "projection")
    _covariance_validate_matrix(input.phase_covariance,
        (plan.phase_count, plan.phase_count), "phase_covariance")
    _covariance_validate_matrix(input.fit_phase_covariance,
        (plan.fit_count, plan.phase_count), "fit_phase_covariance")
    _covariance_validate_matrix(input.measurement_noise_covariance,
        (plan.measurement_count, plan.measurement_count), "measurement_noise_covariance")
    _covariance_validate_matrix(workspace.phase_covariance,
        (plan.phase_count, plan.phase_count), "tomographic covariance phase workspace")
    _covariance_validate_matrix(workspace.measurement_noise_covariance,
        (plan.measurement_count, plan.measurement_count),
        "tomographic covariance measurement-noise workspace")
    _covariance_validate_matrix(workspace.multiplication,
        (plan.measurement_count, plan.phase_count),
        "tomographic covariance multiplication workspace")
    _covariance_validate_matrix(workspace.system,
        (plan.measurement_count, plan.measurement_count),
        "tomographic covariance system workspace")
    _covariance_validate_matrix(workspace.factor,
        (plan.measurement_count, plan.measurement_count),
        "tomographic covariance factor workspace")
    _covariance_validate_matrix(workspace.right_hand_side,
        (plan.fit_count, plan.measurement_count),
        "tomographic covariance right-hand-side workspace")
    _covariance_validate_matrix(workspace.solution,
        (plan.measurement_count, plan.fit_count),
        "tomographic covariance solution workspace")
    _covariance_validate_matrix(workspace.candidate,
        (plan.fit_count, plan.measurement_count),
        "tomographic covariance candidate workspace")
    numerical_arrays = (
        result.matrix, input.projection, input.phase_covariance,
        input.fit_phase_covariance, input.measurement_noise_covariance,
        workspace.phase_covariance, workspace.measurement_noise_covariance,
        workspace.multiplication, workspace.system, workspace.factor,
        workspace.right_hand_side, workspace.solution, workspace.candidate,
        workspace.metric_first, workspace.metric_second,
    )
    flag_arrays = (workspace.invalid_first, workspace.invalid_second)
    arrays = (numerical_arrays..., flag_arrays..., workspace.pivot)
    require_one_based(arrays...)
    require_backend(plan.backend, arrays...)
    for array in numerical_arrays
        eltype(array) === eltype(result.matrix) || throw(ArgumentError(
            "tomographic covariance storage must match the plan precision",
        ))
    end
    largest = _covariance_workspace_largest(plan)
    eltype(workspace.invalid_first) === UInt32 || throw(ArgumentError(
        "tomographic covariance validity storage must use UInt32",
    ))
    eltype(workspace.invalid_second) === UInt32 || throw(ArgumentError(
        "tomographic covariance validity storage must use UInt32",
    ))
    length(workspace.invalid_first) == largest &&
        length(workspace.invalid_second) == max(1, cld(largest, 2)) || throw(
        DimensionMismatch("tomographic covariance validity storage has an incompatible extent"),
    )
    length(workspace.metric_first) == largest &&
        length(workspace.metric_second) == max(1, cld(largest, 2)) || throw(
        DimensionMismatch("tomographic covariance metric storage has an incompatible extent"),
    )
    eltype(workspace.pivot) === Cint && length(workspace.pivot) == plan.measurement_count ||
        throw(DimensionMismatch("tomographic covariance pivot storage has an incompatible extent"))
    length(workspace.host_invalid) == 1 && length(workspace.host_metric) == 1 || throw(
        DimensionMismatch("tomographic covariance host validation storage has an incompatible extent"),
    )
    eltype(workspace.host_metric) === eltype(result.matrix) || throw(ArgumentError(
        "tomographic covariance metric storage must match the plan precision",
    ))
    for first in 1:(length(arrays) - 1), second in (first + 1):length(arrays)
        first_root = Tomography._covariance_reconstructor_alias_root(arrays[first])
        second_root = Tomography._covariance_reconstructor_alias_root(arrays[second])
        require_backend(plan.backend, first_root, second_root)
        Tomography._covariance_reconstructor_root_mightalias(
            plan.backend, first_root, second_root) && throw(ArgumentError(
            "tomographic covariance result, inputs, and workspace buffers must not alias",
        ))
    end
    return nothing
end

function _covariance_invalid!(matrix, workspace, plan, name)
    count = length(matrix)
    plan.finite_kernel(workspace.invalid_first, matrix; ndrange=count)
    KernelAbstractions.synchronize(plan.backend)
    source = workspace.invalid_first
    destination = workspace.invalid_second
    while count > 1
        next_count = cld(count, 2)
        plan.invalid_reduction_kernel(destination, source, count; ndrange=next_count)
        KernelAbstractions.synchronize(plan.backend)
        source, destination = destination, source
        count = next_count
    end
    copyto!(workspace.host_invalid, 1, source, 1, 1)
    KernelAbstractions.synchronize(plan.backend)
    iszero(workspace.host_invalid[1]) || throw(ArgumentError("$name must contain only finite values"))
    return nothing
end

function _covariance_metric!(kernel, reduction_kernel, matrix, workspace, plan, arguments...)
    count = length(matrix)
    kernel(workspace.metric_first, matrix, arguments...; ndrange=count)
    KernelAbstractions.synchronize(plan.backend)
    source = workspace.metric_first
    destination = workspace.metric_second
    while count > 1
        next_count = cld(count, 2)
        reduction_kernel(destination, source, count; ndrange=next_count)
        KernelAbstractions.synchronize(plan.backend)
        source, destination = destination, source
        count = next_count
    end
    copyto!(workspace.host_metric, 1, source, 1, 1)
    KernelAbstractions.synchronize(plan.backend)
    return workspace.host_metric[1]
end

function _covariance_maximum_abs!(matrix, workspace, plan)
    return _covariance_metric!(_covariance_maximum_abs_kernel!(
        plan.backend, plan.workgroup_size), plan.maximum_reduction_kernel,
        matrix, workspace, plan)
end

function _covariance_norm!(matrix, workspace, plan)
    scale = _covariance_maximum_abs!(matrix, workspace, plan)
    isfinite(scale) || return scale
    iszero(scale) && return zero(eltype(matrix))
    squared = _covariance_metric!(_covariance_scaled_square_kernel!(
        plan.backend, plan.workgroup_size), plan.sum_reduction_kernel,
        matrix, workspace, plan, scale)
    return scale * sqrt(squared)
end

function _covariance_symmetrize!(matrix, count, workspace, plan, name)
    _covariance_invalid!(matrix, workspace, plan, name)
    tolerance = sqrt(eps(eltype(matrix))) * _covariance_maximum_abs!(matrix, workspace, plan)
    plan.symmetry_kernel(workspace.invalid_first, matrix, tolerance, count;
        ndrange=size(matrix))
    KernelAbstractions.synchronize(plan.backend)
    source = workspace.invalid_first
    destination = workspace.invalid_second
    length_ = length(matrix)
    while length_ > 1
        next_count = cld(length_, 2)
        plan.invalid_reduction_kernel(destination, source, length_; ndrange=next_count)
        KernelAbstractions.synchronize(plan.backend)
        source, destination = destination, source
        length_ = next_count
    end
    copyto!(workspace.host_invalid, 1, source, 1, 1)
    KernelAbstractions.synchronize(plan.backend)
    iszero(workspace.host_invalid[1]) || throw(ArgumentError(
        "$name must be symmetric within numerical tolerance",
    ))
    _covariance_invalid!(matrix, workspace, plan, name)
    return nothing
end

function Tomography._covariance_reconstructor_lu_solve!(factor, solution, workspace)
    ldiv!(lu!(factor), solution)
    return nothing
end

function AOC.process!(
    result::Tomography.CovarianceReconstructorProduct{T},
    workspace::CovarianceReconstructorKernelWorkspace,
    plan::CovarianceReconstructorKernelPlan{T},
    input::Tomography.CovarianceReconstructorInputs{T},
) where {T<:AbstractFloat}
    _covariance_validate_storage(result, workspace, plan, input)
    _covariance_invalid!(input.projection, workspace, plan, "projection")
    _covariance_invalid!(input.fit_phase_covariance, workspace, plan,
        "fit_phase_covariance")
    copyto!(workspace.phase_covariance, input.phase_covariance)
    copyto!(workspace.measurement_noise_covariance, input.measurement_noise_covariance)
    _covariance_symmetrize!(workspace.phase_covariance, plan.phase_count,
        workspace, plan, "phase_covariance")
    _covariance_symmetrize!(workspace.measurement_noise_covariance,
        plan.measurement_count, workspace, plan, "measurement_noise_covariance")
    mul!(workspace.multiplication, input.projection, workspace.phase_covariance)
    mul!(workspace.system, workspace.multiplication, transpose(input.projection))
    workspace.system .+= workspace.measurement_noise_covariance
    _covariance_symmetrize!(workspace.system, plan.measurement_count,
        workspace, plan, "tomographic covariance system")
    mul!(workspace.right_hand_side, input.fit_phase_covariance,
        transpose(input.projection))
    _covariance_invalid!(workspace.right_hand_side, workspace, plan,
        "tomographic covariance right-hand side")
    system_norm = _covariance_norm!(workspace.system, workspace, plan)
    isfinite(system_norm) || throw(ArgumentError(
        "tomographic covariance system norm is non-finite",
    ))
    plan.transpose_kernel(workspace.solution, workspace.right_hand_side;
        ndrange=size(workspace.solution))
    copyto!(workspace.factor, workspace.system)
    factorization = cholesky!(Hermitian(workspace.factor), check=false)
    used_lu_fallback = !issuccess(factorization)
    if used_lu_fallback
        copyto!(workspace.factor, workspace.system)
        Tomography._covariance_reconstructor_lu_solve!(workspace.factor, workspace.solution,
            workspace)
    else
        ldiv!(factorization, workspace.solution)
    end
    plan.transpose_kernel(workspace.candidate, workspace.solution;
        ndrange=size(workspace.candidate))
    _covariance_invalid!(workspace.candidate, workspace, plan,
        "tomographic covariance reconstructor")
    reconstructor_norm = _covariance_norm!(workspace.candidate, workspace, plan)
    isfinite(reconstructor_norm) || throw(ArgumentError(
        "tomographic covariance reconstructor norm is non-finite",
    ))
    diagnostics = Tomography.CovarianceReconstructorDiagnostics(
        used_lu_fallback, T(system_norm), T(reconstructor_norm))
    plan.publish_kernel(result.matrix, workspace.candidate; ndrange=length(result.matrix))
    KernelAbstractions.synchronize(plan.backend)
    result.diagnostics = diagnostics
    return result
end
