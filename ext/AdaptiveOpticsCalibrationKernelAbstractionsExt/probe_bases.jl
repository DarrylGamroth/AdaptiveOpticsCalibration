using KernelAbstractions: @Const, @index, @kernel

struct ModalPushPullKernelPlan{T,P,B,K,C} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    positive_commands::C
    workgroup_size::Int
end

struct ZonalProbeKernelPlan{T,P,B,K,C} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    amplitudes::C
    workgroup_size::Int
end

struct HadamardProbeKernelPlan{T,P,B,K,C} <: AOC.AbstractCalibrationPlan
    plan::P
    backend::B
    kernel::K
    amplitudes::C
    workgroup_size::Int
end

for (kernel_plan, scalar_plan) in (
    (:ModalPushPullKernelPlan, :(ProbeBases.ModalPushPullPlan)),
    (:ZonalProbeKernelPlan, :(ProbeBases.ZonalPushPullPlan)),
    (:HadamardProbeKernelPlan, :(ProbeBases.HadamardPushPullPlan)),
)
    @eval function $(kernel_plan)(
        plan::$scalar_plan{T},
        backend,
        kernel,
        coefficients,
        workgroup_size,
    ) where {T}
        return $(kernel_plan){
            T,
            typeof(plan),
            typeof(backend),
            typeof(kernel),
            typeof(coefficients),
        }(plan, backend, kernel, coefficients, workgroup_size)
    end
end

@kernel function modal_push_pull_probe_kernel!(
    commands,
    @Const(positive_commands),
    row_extent,
)
    index = @index(Global, Linear)
    row_count = static_value(row_extent)
    row = (index - 1) % row_count + 1
    corrector = (index - 1) ÷ row_count + 1
    mode = (row + 1) ÷ 2
    coefficient = @inbounds positive_commands[mode, corrector]
    @inbounds commands[row, corrector] = isodd(row) ? coefficient : -coefficient
end

@kernel function zonal_push_pull_probe_kernel!(commands, @Const(amplitudes), row_extent)
    index = @index(Global, Linear)
    row_count = static_value(row_extent)
    row = (index - 1) % row_count + 1
    corrector = (index - 1) ÷ row_count + 1
    coordinate = (row + 1) ÷ 2
    amplitude = @inbounds amplitudes[corrector]
    coefficient = coordinate == corrector ?
        amplitude : zero(eltype(commands))
    @inbounds commands[row, corrector] = isodd(row) ? coefficient : -coefficient
end

@kernel function hadamard_push_pull_probe_kernel!(commands, @Const(amplitudes), row_extent)
    index = @index(Global, Linear)
    row_count = static_value(row_extent)
    row = (index - 1) % row_count + 1
    corrector = (index - 1) ÷ row_count + 1
    pattern = (row - 1) ÷ 2
    amplitude = @inbounds amplitudes[corrector]
    coefficient = iseven(count_ones(pattern & corrector)) ?
        amplitude : -amplitude
    @inbounds commands[row, corrector] = isodd(row) ? coefficient : -coefficient
end

function AOC.prepare(
    method::ProbeBases.ModalPushPull{T},
    execution::AOC.KernelExecution{Nothing,B},
) where {T<:Real,B<:KernelAbstractions.Backend}
    plan = AOC.prepare(method, nothing)
    commands = copy_to_backend(execution.backend, plan.positive_commands)
    KernelAbstractions.synchronize(execution.backend)
    return ModalPushPullKernelPlan(
        plan,
        execution.backend,
        modal_push_pull_probe_kernel!(execution.backend, execution.workgroup_size),
        commands,
        execution.workgroup_size,
    )
end

function AOC.prepare(
    method::ProbeBases.ZonalPushPull{T},
    execution::AOC.KernelExecution{Nothing,B},
) where {T<:Real,B<:KernelAbstractions.Backend}
    plan = AOC.prepare(method, nothing)
    amplitudes = copy_to_backend(execution.backend, plan.amplitudes)
    KernelAbstractions.synchronize(execution.backend)
    return ZonalProbeKernelPlan(
        plan,
        execution.backend,
        zonal_push_pull_probe_kernel!(execution.backend, execution.workgroup_size),
        amplitudes,
        execution.workgroup_size,
    )
end

function AOC.prepare(
    method::ProbeBases.HadamardPushPull{T},
    execution::AOC.KernelExecution{Nothing,B},
) where {T<:Real,B<:KernelAbstractions.Backend}
    plan = AOC.prepare(method, nothing)
    amplitudes = copy_to_backend(execution.backend, plan.amplitudes)
    KernelAbstractions.synchronize(execution.backend)
    return HadamardProbeKernelPlan(
        plan,
        execution.backend,
        hadamard_push_pull_probe_kernel!(execution.backend, execution.workgroup_size),
        amplitudes,
        execution.workgroup_size,
    )
end

const ProbeKernelPlan{T} = Union{
    ModalPushPullKernelPlan{T},
    ZonalProbeKernelPlan{T},
    HadamardProbeKernelPlan{T},
}

function AOC.allocate_result(plan::ProbeKernelPlan{T}) where {T}
    commands = KernelAbstractions.allocate(
        plan.backend,
        T,
        2 * plan.plan.pattern_count,
        plan.plan.corrector_coordinate_count,
    )
    return ProbeBases.ProbeCommandBasisProduct(commands)
end

AOC.allocate_workspace(::ProbeKernelPlan) = nothing

function validate_probe_kernel_product(result, plan)
    expected = (
        2 * plan.plan.pattern_count,
        plan.plan.corrector_coordinate_count,
    )
    size(result.commands) == expected || throw(DimensionMismatch(
        "probe-command result must have shape $expected",
    ))
    require_one_based(result.commands)
    require_backend(plan.backend, result.commands)
    return nothing
end

function process_probe_kernel!(result, plan, coefficients)
    validate_probe_kernel_product(result, plan)
    require_backend(plan.backend, coefficients)
    plan.kernel(
        result.commands,
        coefficients,
        Val(size(result.commands, 1));
        ndrange=length(result.commands),
    )
    KernelAbstractions.synchronize(plan.backend)
    return result
end

function AOC.process!(
    result::ProbeBases.ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::ModalPushPullKernelPlan{T},
    ::Nothing,
) where {T}
    return process_probe_kernel!(result, plan, plan.positive_commands)
end

function AOC.process!(
    result::ProbeBases.ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::ZonalProbeKernelPlan{T},
    ::Nothing,
) where {T}
    return process_probe_kernel!(result, plan, plan.amplitudes)
end

function AOC.process!(
    result::ProbeBases.ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::HadamardProbeKernelPlan{T},
    ::Nothing,
) where {T}
    return process_probe_kernel!(result, plan, plan.amplitudes)
end
