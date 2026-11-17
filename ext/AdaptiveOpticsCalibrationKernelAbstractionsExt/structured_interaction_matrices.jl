using KernelAbstractions: @Const, @index, @kernel

@inline static_value(::Val{N}) where {N} = N

struct ZonalPushPullKernelPlan{T,P,B,K,C,KP,KR} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    reciprocal_scales::C
    publish_kernel::KP
    reduction_kernel::KR
    workgroup_size::Int
end

struct HadamardPushPullKernelPlan{T,P,B,K,C,KP,KR} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    reciprocal_scales::C
    publish_kernel::KP
    reduction_kernel::KR
    workgroup_size::Int
end

function ZonalPushPullKernelPlan(
    plan::InteractionMatrices.ZonalPushPullPlan{T},
    backend,
    kernel,
    reciprocal_scales,
    publish_kernel,
    reduction_kernel,
    workgroup_size,
) where {T}
    return ZonalPushPullKernelPlan{
        T,
        typeof(plan),
        typeof(backend),
        typeof(kernel),
        typeof(reciprocal_scales),
        typeof(publish_kernel),
        typeof(reduction_kernel),
    }(
        plan,
        backend,
        kernel,
        reciprocal_scales,
        publish_kernel,
        reduction_kernel,
        workgroup_size,
    )
end

function HadamardPushPullKernelPlan(
    plan::InteractionMatrices.HadamardPushPullPlan{T},
    backend,
    kernel,
    reciprocal_scales,
    publish_kernel,
    reduction_kernel,
    workgroup_size,
) where {T}
    return HadamardPushPullKernelPlan{
        T,
        typeof(plan),
        typeof(backend),
        typeof(kernel),
        typeof(reciprocal_scales),
        typeof(publish_kernel),
        typeof(reduction_kernel),
    }(
        plan,
        backend,
        kernel,
        reciprocal_scales,
        publish_kernel,
        reduction_kernel,
        workgroup_size,
    )
end

struct StructuredKernelWorkspace{T,M,F}
    staged_matrix::M
    invalid_first::F
    invalid_second::F
    host_invalid::Vector{UInt32}
end

@kernel function zonal_push_pull_estimate_kernel!(
    staged,
    invalid,
    @Const(responses),
    @Const(reciprocal_scales),
    measurement_extent,
)
    index = @index(Global, Linear)
    measurement_count = static_value(measurement_extent)
    measurement = (index - 1) % measurement_count + 1
    corrector = (index - 1) ÷ measurement_count + 1
    positive_row = 2 * corrector - 1
    @inbounds begin
        positive_response = responses[positive_row, measurement]
        negative_response = responses[positive_row + 1, measurement]
        response_is_finite = isfinite(positive_response) && isfinite(negative_response)
        difference = positive_response - negative_response
        coefficient = difference * reciprocal_scales[corrector]
        calculation_is_finite = isfinite(difference) && isfinite(coefficient)
        if response_is_finite && calculation_is_finite
            staged[measurement, corrector] = coefficient
            invalid[index] = UInt32(0)
        else
            invalid[index] =
                (!response_is_finite ? UInt32(1) : UInt32(0)) |
                (!calculation_is_finite ? UInt32(2) : UInt32(0))
        end
    end
end

@kernel function hadamard_push_pull_estimate_kernel!(
    staged,
    invalid,
    @Const(responses),
    @Const(reciprocal_scales),
    measurement_extent,
    order,
)
    index = @index(Global, Linear)
    measurement_count = static_value(measurement_extent)
    measurement = (index - 1) % measurement_count + 1
    corrector = (index - 1) ÷ measurement_count + 1
    accumulation = zero(eltype(staged))
    response_is_finite = true
    calculation_is_finite = true
    for row in 0:(order - 1)
        positive_row = 2 * row + 1
        positive_response = @inbounds responses[positive_row, measurement]
        negative_response = @inbounds responses[positive_row + 1, measurement]
        response_is_finite &= isfinite(positive_response) && isfinite(negative_response)
        difference = positive_response - negative_response
        calculation_is_finite &= isfinite(difference)
        accumulation += iseven(count_ones(row & corrector)) ? difference : -difference
        calculation_is_finite &= isfinite(accumulation)
    end
    coefficient = accumulation * (@inbounds reciprocal_scales[corrector])
    calculation_is_finite &= isfinite(coefficient)
    @inbounds if response_is_finite && calculation_is_finite
        staged[measurement, corrector] = coefficient
        invalid[index] = UInt32(0)
    else
        invalid[index] =
            (!response_is_finite ? UInt32(1) : UInt32(0)) |
            (!calculation_is_finite ? UInt32(2) : UInt32(0))
    end
end

@kernel function invalid_or_reduction_kernel!(destination, @Const(source), count)
    index = @index(Global, Linear)
    source_index = 2 * index - 1
    value = @inbounds source[source_index]
    if source_index < count
        value |= @inbounds source[source_index + 1]
    end
    @inbounds destination[index] = value
end

@kernel function publish_matrix_kernel!(destination, @Const(source))
    index = @index(Global, Linear)
    @inbounds destination[index] = source[index]
end

function AOC.prepare(
    method::InteractionMatrices.ZonalPushPull{T},
    execution::AOC.KernelExecution{S,B},
) where {
    T<:AbstractFloat,
    S<:InteractionMatrices.ResponseSpecification,
    B<:KernelAbstractions.Backend,
}
    plan = AOC.prepare(method, execution.specification)
    scales = copy_to_backend(execution.backend, plan.reciprocal_scales)
    KernelAbstractions.synchronize(execution.backend)
    return ZonalPushPullKernelPlan(
        plan,
        execution.backend,
        zonal_push_pull_estimate_kernel!(execution.backend, execution.workgroup_size),
        scales,
        publish_matrix_kernel!(execution.backend, execution.workgroup_size),
        invalid_or_reduction_kernel!(execution.backend, execution.workgroup_size),
        execution.workgroup_size,
    )
end

function AOC.prepare(
    method::InteractionMatrices.HadamardPushPull{T},
    execution::AOC.KernelExecution{S,B},
) where {
    T<:AbstractFloat,
    S<:InteractionMatrices.ResponseSpecification,
    B<:KernelAbstractions.Backend,
}
    plan = AOC.prepare(method, execution.specification)
    scales = copy_to_backend(execution.backend, plan.reciprocal_scales)
    KernelAbstractions.synchronize(execution.backend)
    return HadamardPushPullKernelPlan(
        plan,
        execution.backend,
        hadamard_push_pull_estimate_kernel!(execution.backend, execution.workgroup_size),
        scales,
        publish_matrix_kernel!(execution.backend, execution.workgroup_size),
        invalid_or_reduction_kernel!(execution.backend, execution.workgroup_size),
        execution.workgroup_size,
    )
end

const StructuredKernelPlan{T} = Union{
    ZonalPushPullKernelPlan{T},
    HadamardPushPullKernelPlan{T},
}

function AOC.allocate_result(plan::StructuredKernelPlan{T}) where {T}
    matrix = KernelAbstractions.allocate(
        plan.backend,
        T,
        plan.plan.measurement_count,
        plan.plan.corrector_coordinate_count,
    )
    return InteractionMatrices.InteractionMatrixProduct(matrix)
end

function AOC.allocate_workspace(plan::StructuredKernelPlan{T}) where {T}
    staged = KernelAbstractions.allocate(
        plan.backend,
        T,
        plan.plan.measurement_count,
        plan.plan.corrector_coordinate_count,
    )
    coefficient_count = plan.plan.measurement_count *
        plan.plan.corrector_coordinate_count
    invalid_first = KernelAbstractions.allocate(
        plan.backend,
        UInt32,
        coefficient_count,
    )
    invalid_second = KernelAbstractions.allocate(
        plan.backend,
        UInt32,
        max(1, cld(coefficient_count, 2)),
    )
    KernelAbstractions.synchronize(plan.backend)
    return StructuredKernelWorkspace{T,typeof(staged),typeof(invalid_first)}(
        staged,
        invalid_first,
        invalid_second,
        zeros(UInt32, 1),
    )
end

function validate_structured_kernel_storage(output, workspace, plan, responses)
    scientific_plan = plan.plan
    expected_output = (
        scientific_plan.measurement_count,
        scientific_plan.corrector_coordinate_count,
    )
    size(output) == expected_output || throw(DimensionMismatch(
        "interaction-matrix result must have shape $expected_output",
    ))
    size(workspace.staged_matrix) == expected_output || throw(DimensionMismatch(
        "structured kernel workspace must have shape $expected_output",
    ))
    size(responses) ==
        (scientific_plan.response_row_count, scientific_plan.measurement_count) ||
        throw(DimensionMismatch("responses have an incompatible shape"))
    coefficient_count = prod(expected_output)
    length(workspace.invalid_first) == coefficient_count &&
        length(workspace.invalid_second) == max(1, cld(coefficient_count, 2)) &&
        length(workspace.host_invalid) == 1 ||
        throw(DimensionMismatch("structured kernel validity storage is invalid"))
    require_one_based(
        output,
        workspace.staged_matrix,
        workspace.invalid_first,
        workspace.invalid_second,
        responses,
    )
    require_backend(
        plan.backend,
        output,
        workspace.staged_matrix,
        workspace.invalid_first,
        workspace.invalid_second,
        responses,
        plan.reciprocal_scales,
    )
    Base.mightalias(output, workspace.staged_matrix) &&
        throw(ArgumentError("result matrix must not alias structured workspace"))
    Base.mightalias(responses, workspace.staged_matrix) &&
        throw(ArgumentError("responses must not alias structured workspace"))
    return nothing
end

function reduce_invalid_flags!(workspace, plan, count)
    source = workspace.invalid_first
    destination = workspace.invalid_second
    while count > 1
        next_count = cld(count, 2)
        plan.reduction_kernel(
            destination,
            source,
            count;
            ndrange=next_count,
        )
        KernelAbstractions.synchronize(plan.backend)
        source, destination = destination, source
        count = next_count
    end
    copyto!(workspace.host_invalid, 1, source, 1, 1)
    KernelAbstractions.synchronize(plan.backend)
    return workspace.host_invalid[1]
end

function process_structured_kernel!(operation, output, workspace, plan, responses)
    validate_structured_kernel_storage(output, workspace, plan, responses)
    operation()
    KernelAbstractions.synchronize(plan.backend)
    invalid = reduce_invalid_flags!(workspace, plan, length(output))
    if !iszero(invalid)
        return iszero(invalid & UInt32(1)) ?
            InteractionMatrices.StructuredNonFiniteCalculation :
            InteractionMatrices.StructuredNonFiniteResponse
    end
    plan.publish_kernel(
        output,
        workspace.staged_matrix;
        ndrange=length(output),
    )
    KernelAbstractions.synchronize(plan.backend)
    return InteractionMatrices.StructuredValid
end

function AOC.process!(
    output::AbstractMatrix{T},
    workspace::StructuredKernelWorkspace{T},
    plan::ZonalPushPullKernelPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    return process_structured_kernel!(output, workspace, plan, responses) do
        plan.kernel(
            workspace.staged_matrix,
            workspace.invalid_first,
            responses,
            plan.reciprocal_scales,
            Val(plan.plan.measurement_count);
            ndrange=length(workspace.staged_matrix),
        )
    end
end

function AOC.process!(
    output::AbstractMatrix{T},
    workspace::StructuredKernelWorkspace{T},
    plan::HadamardPushPullKernelPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    return process_structured_kernel!(output, workspace, plan, responses) do
        plan.kernel(
            workspace.staged_matrix,
            workspace.invalid_first,
            responses,
            plan.reciprocal_scales,
            Val(plan.plan.measurement_count),
            plan.plan.order;
            ndrange=length(workspace.staged_matrix),
        )
    end
end

function AOC.process!(
    result::InteractionMatrices.InteractionMatrixProduct{T},
    workspace::StructuredKernelWorkspace{T},
    plan::ZonalPushPullKernelPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    result.status = AOC.process!(result.matrix, workspace, plan, responses)
    return result
end

function AOC.process!(
    result::InteractionMatrices.InteractionMatrixProduct{T},
    workspace::StructuredKernelWorkspace{T},
    plan::HadamardPushPullKernelPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    result.status = AOC.process!(result.matrix, workspace, plan, responses)
    return result
end
