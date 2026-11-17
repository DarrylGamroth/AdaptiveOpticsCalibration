using KernelAbstractions: @Const, @index, @kernel

# The device scalar approximation is adapted from AdaptiveOpticsSim.jl
# src/core/kv56.jl at b4d2da2. Its gamma-dependent series coefficients are
# computed during preparation; the CPU reference continues to use besselk.
struct VonKarmanKernelPlan{
    T<:AbstractFloat,
    G<:AbstractArray{Complex{T},3},
    F<:Union{Nothing,AbstractArray{Complex{T},3}},
    W<:AbstractVector{T},
    B<:KernelAbstractions.Backend,
    K,R,P,
} <: AOC.AbstractCalibrationPlan
    guide_coordinates::G
    fit_coordinates::F
    fractional_cn2::W
    covariance_constant::T
    zero_separation_covariance::T
    inverse_outer_scale_m::T
    series_coefficients::NTuple{7,T}
    backend::B
    covariance_kernel::K
    reduction_kernel::R
    publish_kernel::P
    workgroup_size::Int
end

const AutoVonKarmanKernelPlan{T} = VonKarmanKernelPlan{T,G,Nothing} where {G}
const CrossVonKarmanKernelPlan{T} = VonKarmanKernelPlan{T,G,F} where {G,F<:AbstractArray}

struct VonKarmanKernelWorkspace{A,F}
    candidate::A
    invalid_first::F
    invalid_second::F
    host_invalid::Vector{UInt32}
end

function _von_karman_series_coefficients(::Type{T}) where {T<:AbstractFloat}
    nu = T(5) / T(6)
    a0 = T(0.5) * T(Tomography.gamma(nu)) * T(2)^nu
    b0 = T(0.5) * T(Tomography.gamma(-nu)) * T(2)^(-nu)
    c1 = inv(T(4) * (one(T) - nu))
    c2 = inv(T(32) * (one(T) - nu) * (T(2) - nu))
    d1 = inv(T(4) * (one(T) + nu))
    d2 = inv(T(32) * (one(T) + nu) * (T(2) + nu))
    return (nu, a0, b0, c1, c2, d1, d2)
end

@inline function _scaled_kv56_device(x::T, coefficients::NTuple{7,T}) where {T}
    nu, a0, b0, c1, c2, d1, d2 = coefficients
    if x < T(0.1)
        x2 = x * x
        lead = a0 * (one(T) + x2 * c1 + (x2 * x2) * c2)
        tail = b0 * x^(T(2) * nu) * (one(T) + x2 * d1 + (x2 * x2) * d2)
        return lead + tail
    elseif x >= T(20)
        xinv = inv(x)
        series = one(T) + (T(2) / T(9)) * xinv +
            (-T(7) / T(81)) * xinv^2 + (T(175) / T(2187)) * xinv^3 +
            (-T(2275) / T(19683)) * xinv^4 + (T(5005) / T(177147)) * xinv^5
        return x^nu * sqrt(T(pi) / (T(2) * x)) * exp(-x) * series
    end
    dt = T(9) / T(64)
    acc = zero(T)
    for k in 0:63
        t = (T(k) + T(0.5)) * dt
        acc += exp(-x * cosh(t)) * cosh(nu * t)
    end
    return x^nu * dt * acc
end

@inline function _von_karman_device_layer(first::Complex{T}, second::Complex{T},
    weight::T, covariance_constant::T, zero_separation_covariance::T,
    inverse_outer_scale_m::T, coefficients::NTuple{7,T}) where {T}
    separation = abs(first - second)
    if iszero(separation)
        return zero_separation_covariance * weight
    end
    argument = T(2π) * separation * inverse_outer_scale_m
    isfinite(argument) && argument > zero(T) || return T(NaN)
    return covariance_constant * _scaled_kv56_device(argument, coefficients) * weight
end

@kernel function _auto_von_karman_kernel!(candidate, invalid, @Const(guide),
    @Const(weights), covariance_constant, zero_separation_covariance,
    inverse_outer_scale_m, coefficients, guide_sample_count, guide_count)
    first_sample, first_source, second_sample, second_source =
        @index(Global, NTuple)
    total = guide_count
    row = first_sample + (first_source - 1) * guide_sample_count
    column = second_sample + (second_source - 1) * guide_sample_count
    index = row + (column - 1) * total
    value = zero(eltype(candidate))
    @inbounds for layer in 1:size(guide, 3)
        value += _von_karman_device_layer(
            guide[first_sample, first_source, layer],
            guide[second_sample, second_source, layer], weights[layer],
            covariance_constant, zero_separation_covariance,
            inverse_outer_scale_m, coefficients)
    end
    @inbounds begin
        candidate[index] = value
        invalid[index] = isfinite(value) ? UInt32(0) : UInt32(1)
    end
end

@kernel function _cross_von_karman_kernel!(candidate, invalid, @Const(guide),
    @Const(fit), @Const(weights), covariance_constant, zero_separation_covariance,
    inverse_outer_scale_m, coefficients, guide_sample_count, fit_source_count,
    fit_sample_count)
    fit_source, fit_sample, guide_sample, guide_source = @index(Global, NTuple)
    flattened_guide = guide_sample + (guide_source - 1) * guide_sample_count
    index = fit_source + (fit_sample - 1) * fit_source_count +
        (flattened_guide - 1) * fit_source_count * fit_sample_count
    value = zero(eltype(candidate))
    @inbounds for layer in 1:size(guide, 3)
        value += _von_karman_device_layer(
            fit[fit_sample, fit_source, layer],
            guide[guide_sample, guide_source, layer], weights[layer],
            covariance_constant, zero_separation_covariance,
            inverse_outer_scale_m, coefficients)
    end
    @inbounds begin
        candidate[index] = value
        invalid[index] = isfinite(value) ? UInt32(0) : UInt32(1)
    end
end

function _prepare_von_karman_kernel(method, execution, covariance_kernel)
    specification = execution.specification
    for array in (specification.guide_coordinates, specification.fractional_cn2)
        KernelAbstractions.get_backend(array) == KernelAbstractions.CPU() ||
            throw(ArgumentError("von Kármán preparation requires host coordinates and weights"))
    end
    if specification isa Tomography.VonKarmanCrossCovarianceSpecification
        KernelAbstractions.get_backend(specification.fit_coordinates) ==
            KernelAbstractions.CPU() || throw(ArgumentError(
                "von Kármán preparation requires host fit coordinates"))
    end
    cpu = AOC.prepare(method, specification)
    backend = execution.backend
    guide = copy_to_backend(backend, cpu.guide_coordinates)
    fit = cpu isa Tomography.VonKarmanCrossCovariancePlan ?
        copy_to_backend(backend, cpu.fit_coordinates) : nothing
    weights = copy_to_backend(backend, cpu.fractional_cn2)
    coefficients = _von_karman_series_coefficients(typeof(cpu.covariance_constant))
    all(isfinite, coefficients) || throw(ArgumentError(
        "von Kármán series coefficients must be finite"))
    KernelAbstractions.synchronize(backend)
    return VonKarmanKernelPlan(guide, fit, weights, cpu.covariance_constant,
        cpu.zero_separation_covariance, cpu.inverse_outer_scale_m, coefficients,
        backend, covariance_kernel(backend, execution.workgroup_size),
        invalid_or_reduction_kernel!(backend, execution.workgroup_size),
        publish_matrix_kernel!(backend, execution.workgroup_size),
        execution.workgroup_size)
end

function AOC.prepare(method::Tomography.VonKarmanAutoCovariance,
    execution::AOC.KernelExecution{S,B}) where
    {S<:Tomography.VonKarmanAutoCovarianceSpecification,B<:KernelAbstractions.Backend}
    return _prepare_von_karman_kernel(method, execution, _auto_von_karman_kernel!)
end

function AOC.prepare(method::Tomography.VonKarmanCrossCovariance,
    execution::AOC.KernelExecution{S,B}) where
    {S<:Tomography.VonKarmanCrossCovarianceSpecification,B<:KernelAbstractions.Backend}
    return _prepare_von_karman_kernel(method, execution, _cross_von_karman_kernel!)
end

function AOC.allocate_result(plan::AutoVonKarmanKernelPlan{T}) where {T}
    sample_count, source_count, _ = size(plan.guide_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    return Tomography.VonKarmanAutoCovarianceProduct(
        KernelAbstractions.allocate(plan.backend, T, guide_count, guide_count))
end

function AOC.allocate_result(plan::CrossVonKarmanKernelPlan{T}) where {T}
    sample_count, source_count, _ = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    AOC._checked_extent("von Kármán cross covariance", fit_source_count,
        fit_sample_count, guide_count)
    return Tomography.VonKarmanCrossCovarianceProduct(KernelAbstractions.allocate(
        plan.backend, T, fit_source_count, fit_sample_count, guide_count))
end

function _allocate_von_karman_workspace(plan::VonKarmanKernelPlan{T}, dims) where {T}
    candidate = KernelAbstractions.allocate(plan.backend, T, dims)
    count = length(candidate)
    first = KernelAbstractions.allocate(plan.backend, UInt32, count)
    second = KernelAbstractions.allocate(plan.backend, UInt32, max(1, cld(count, 2)))
    KernelAbstractions.synchronize(plan.backend)
    return VonKarmanKernelWorkspace(candidate, first, second, zeros(UInt32, 1))
end

function AOC.allocate_workspace(plan::AutoVonKarmanKernelPlan)
    sample_count, source_count, _ = size(plan.guide_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    return _allocate_von_karman_workspace(plan, (guide_count, guide_count))
end

function AOC.allocate_workspace(plan::CrossVonKarmanKernelPlan)
    sample_count, source_count, _ = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    AOC._checked_extent("von Kármán cross covariance", fit_source_count,
        fit_sample_count, guide_count)
    return _allocate_von_karman_workspace(plan,
        (fit_source_count, fit_sample_count, guide_count))
end

function _process_von_karman_kernel!(operation, output::AbstractArray{T},
    workspace, plan::VonKarmanKernelPlan{T}) where {T}
    size(output) == size(workspace.candidate) || throw(DimensionMismatch(
        "von Kármán result has an incompatible shape"))
    eltype(workspace.candidate) == T || throw(ArgumentError(
        "von Kármán candidate element type must match the plan precision"))
    eltype(workspace.invalid_first) == UInt32 &&
        eltype(workspace.invalid_second) == UInt32 &&
        eltype(workspace.host_invalid) == UInt32 || throw(ArgumentError(
            "von Kármán validity buffers must store UInt32"))
    length(workspace.invalid_first) == length(output) &&
        length(workspace.invalid_second) == max(1, cld(length(output), 2)) &&
        length(workspace.host_invalid) == 1 || throw(DimensionMismatch(
            "von Kármán validity storage has an incompatible extent"))
    require_one_based(output, workspace.candidate, workspace.invalid_first,
        workspace.invalid_second)
    require_backend(plan.backend, output, workspace.candidate,
        workspace.invalid_first, workspace.invalid_second,
        plan.guide_coordinates, plan.fractional_cn2)
    if plan.fit_coordinates !== nothing
        require_backend(plan.backend, plan.fit_coordinates)
    end
    buffers = (output, workspace.candidate, workspace.invalid_first,
        workspace.invalid_second, workspace.host_invalid)
    for first in 1:(length(buffers) - 1), second in (first + 1):length(buffers)
        Base.mightalias(buffers[first], buffers[second]) && throw(ArgumentError(
            "von Kármán result and workspace buffers must not alias"))
    end
    for storage in (plan.guide_coordinates, plan.fractional_cn2)
        for buffer in buffers
            Base.mightalias(buffer, storage) && throw(ArgumentError(
                "von Kármán result and workspace must not alias plan storage"))
        end
    end
    if plan.fit_coordinates !== nothing
        for buffer in buffers
            Base.mightalias(buffer, plan.fit_coordinates) && throw(ArgumentError(
                "von Kármán result and workspace must not alias fit coordinates"))
        end
    end
    operation()
    KernelAbstractions.synchronize(plan.backend)
    source = workspace.invalid_first
    destination = workspace.invalid_second
    count = length(output)
    while count > 1
        next_count = cld(count, 2)
        plan.reduction_kernel(destination, source, count; ndrange=next_count)
        KernelAbstractions.synchronize(plan.backend)
        source, destination = destination, source
        count = next_count
    end
    copyto!(workspace.host_invalid, 1, source, 1, 1)
    KernelAbstractions.synchronize(plan.backend)
    iszero(workspace.host_invalid[1]) || throw(ArgumentError(
        "von Kármán covariance contains non-finite values"))
    plan.publish_kernel(output, workspace.candidate; ndrange=length(output))
    KernelAbstractions.synchronize(plan.backend)
    return nothing
end

function AOC.process!(result::Tomography.VonKarmanAutoCovarianceProduct{T},
    workspace::VonKarmanKernelWorkspace, plan::AutoVonKarmanKernelPlan{T},
    ::Nothing) where {T}
    sample_count, source_count, _ = size(plan.guide_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    size(result.Cxx) == (guide_count, guide_count) || throw(DimensionMismatch(
        "Cxx has an incompatible guide-sample extent"))
    _process_von_karman_kernel!(result.Cxx, workspace, plan) do
        plan.covariance_kernel(workspace.candidate, workspace.invalid_first,
            plan.guide_coordinates, plan.fractional_cn2, plan.covariance_constant,
            plan.zero_separation_covariance, plan.inverse_outer_scale_m,
            plan.series_coefficients, sample_count, guide_count;
            ndrange=(sample_count, source_count, sample_count, source_count))
    end
    return result
end

function AOC.process!(result::Tomography.VonKarmanCrossCovarianceProduct{T},
    workspace::VonKarmanKernelWorkspace, plan::CrossVonKarmanKernelPlan{T},
    ::Nothing) where {T}
    sample_count, source_count, _ = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    guide_count = AOC._checked_extent("von Kármán guide sample", sample_count, source_count)
    size(result.Cox) == (fit_source_count, fit_sample_count, guide_count) ||
        throw(DimensionMismatch("Cox has an incompatible fit-by-guide extent"))
    _process_von_karman_kernel!(result.Cox, workspace, plan) do
        plan.covariance_kernel(workspace.candidate, workspace.invalid_first,
            plan.guide_coordinates, plan.fit_coordinates, plan.fractional_cn2,
            plan.covariance_constant, plan.zero_separation_covariance,
            plan.inverse_outer_scale_m, plan.series_coefficients,
            sample_count, fit_source_count, fit_sample_count;
            ndrange=(fit_source_count, fit_sample_count,
                sample_count, source_count))
    end
    return result
end
