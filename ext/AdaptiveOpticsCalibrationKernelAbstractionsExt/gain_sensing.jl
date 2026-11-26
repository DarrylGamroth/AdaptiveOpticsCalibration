using KernelAbstractions: @Const, @index, @kernel

struct GainSensingKernelPlan{
    T<:AbstractFloat,
    B<:KernelAbstractions.Backend,
    C<:AbstractMatrix{Complex{T}},
    R<:AbstractMatrix{T},
    S<:AbstractVector{Complex{T}},
    W<:AbstractVector{Bool},
    K,
} <: AOC.AbstractCalibrationPlan
    side::Int
    sample_count::Int
    mode_count::Int
    sensitivity_floor::T
    backend::B
    mask::C
    mask_fft::C
    basis_products::C
    reference_impulse_response::R
    reference_sensitivities::S
    weak_mode_mask::W
    reference_total_flux::T
    normalized_reference_total_flux::T
    kernels::K
end

struct GainSensingKernelWorkspace{
    T<:AbstractFloat,
    C<:AbstractMatrix{Complex{T}},
    V<:AbstractVector{Complex{T}},
    R<:AbstractMatrix{T},
    S<:AbstractVector{Complex{T}},
    G<:AbstractVector{T},
    W<:AbstractVector{Bool},
    I<:AbstractVector{UInt32},
    PF,
    PI,
}
    buffer::C
    buffer_vector::V
    forward::C
    inverse::C
    normalized_current_frame::R
    current_impulse_response::R
    current_sensitivities::S
    candidate_gains::G
    candidate_weak_mode_mask::W
    sum_first::G
    sum_second::G
    invalid_first::I
    invalid_second::I
    host_sum::Vector{T}
    host_invalid::Vector{UInt32}
    fft_plan::PF
    ifft_plan::PI
end

@kernel function gain_sensing_shift_kernel!(destination, @Const(source), side, half_side)
    row, column = @index(Global, NTuple)
    if row <= side && column <= side
        source_row = row + half_side
        source_column = column + half_side
        source_row > side && (source_row -= side)
        source_column > side && (source_column -= side)
        @inbounds destination[row, column] = source[source_row, source_column]
    end
end

@kernel function gain_sensing_mode_plane_kernel!(destination, @Const(basis), mode, count)
    sample = @index(Global, Linear)
    if sample <= count
        @inbounds destination[sample] = basis[sample + (mode - 1) * count]
    end
end

@kernel function gain_sensing_basis_products_kernel!(
    products, @Const(forward), @Const(inverse), mode, count)
    sample = @index(Global, Linear)
    if sample <= count
        @inbounds products[sample, mode] = forward[sample] * inverse[sample]
    end
end

@kernel function gain_sensing_map_sum_kernel!(sums, invalid, @Const(source), count)
    sample = @index(Global, Linear)
    if sample <= count
        value = @inbounds source[sample]
        @inbounds begin
            sums[sample] = value
            invalid[sample] = isfinite(value) ? UInt32(0) : UInt32(1)
        end
    end
end

@kernel function gain_sensing_pair_sum_kernel!(
    destination_sum, destination_invalid, @Const(source_sum),
    @Const(source_invalid), count)
    index = @index(Global, Linear)
    first = 2 * index - 1
    if first <= count
        value = @inbounds source_sum[first]
        invalid = @inbounds source_invalid[first]
        if first < count
            value += @inbounds source_sum[first + 1]
            invalid |= @inbounds source_invalid[first + 1]
        end
        @inbounds begin
            destination_sum[index] = value
            destination_invalid[index] = invalid
        end
    end
end

@kernel function gain_sensing_map_finite_kernel!(invalid, @Const(source), count)
    index = @index(Global, Linear)
    if index <= count
        @inbounds invalid[index] = isfinite(source[index]) ? UInt32(0) : UInt32(1)
    end
end

@kernel function gain_sensing_pair_invalid_kernel!(destination, @Const(source), count)
    index = @index(Global, Linear)
    first = 2 * index - 1
    if first <= count
        value = @inbounds source[first]
        if first < count
            value |= @inbounds source[first + 1]
        end
        @inbounds destination[index] = value
    end
end

@kernel function gain_sensing_candidate_kernel!(
    gains, weak_modes, @Const(current), @Const(reference), @Const(reference_weak),
    count)
    mode = @index(Global, Linear)
    if mode <= count
        weak = @inbounds reference_weak[mode]
        gain = weak ? one(eltype(gains)) :
            real((@inbounds current[mode]) / (@inbounds reference[mode]))
        @inbounds begin
            gains[mode] = gain
            weak_modes[mode] = weak
        end
    end
end

function _gain_sensing_kernels(backend, workgroup_size)
    return (
        shift=gain_sensing_shift_kernel!(backend, workgroup_size),
        mode_plane=gain_sensing_mode_plane_kernel!(backend, workgroup_size),
        basis_products=gain_sensing_basis_products_kernel!(backend, workgroup_size),
        map_sum=gain_sensing_map_sum_kernel!(backend, workgroup_size),
        pair_sum=gain_sensing_pair_sum_kernel!(backend, workgroup_size),
        map_finite=gain_sensing_map_finite_kernel!(backend, workgroup_size),
        pair_invalid=gain_sensing_pair_invalid_kernel!(backend, workgroup_size),
        candidate=gain_sensing_candidate_kernel!(backend, workgroup_size),
    )
end

@inline function _gain_sensing_launch!(backend, kernel, args...; ndrange)
    kernel(args...; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _gain_sensing_kernel_workspace(backend, ::Type{T}, side, mode_count) where {T<:AbstractFloat}
    sample_count = side * side
    reduction_count = max(sample_count, mode_count)
    buffer = KernelAbstractions.allocate(backend, Complex{T}, side, side)
    buffer_vector = vec(buffer)
    forward = similar(buffer)
    inverse = similar(buffer)
    normalized_current_frame = KernelAbstractions.allocate(backend, T, side, side)
    current_impulse_response = similar(normalized_current_frame)
    current_sensitivities = KernelAbstractions.allocate(backend, Complex{T}, mode_count)
    candidate_gains = KernelAbstractions.allocate(backend, T, mode_count)
    candidate_weak_mode_mask = KernelAbstractions.allocate(backend, Bool, mode_count)
    sum_first = KernelAbstractions.allocate(backend, T, reduction_count)
    sum_second = similar(sum_first)
    invalid_first = KernelAbstractions.allocate(backend, UInt32, reduction_count)
    invalid_second = similar(invalid_first)
    fft_plan = FFTW.plan_fft!(buffer, (1, 2))
    ifft_plan = FFTW.plan_ifft!(buffer, (1, 2))
    return GainSensingKernelWorkspace(
        buffer, buffer_vector, forward, inverse, normalized_current_frame,
        current_impulse_response, current_sensitivities, candidate_gains,
        candidate_weak_mode_mask, sum_first, sum_second, invalid_first,
        invalid_second, zeros(T, 1), zeros(UInt32, 1), fft_plan, ifft_plan,
    )
end

function _gain_sensing_centered_fft!(
    output, workspace, backend, kernels, input, side)
    _gain_sensing_launch!(backend, kernels.shift, workspace.buffer, input,
        side, side ÷ 2; ndrange=(side, side))
    OpticalGains._execute_gain_sensing_fft_plan!(workspace.buffer,
        workspace.fft_plan)
    KernelAbstractions.synchronize(backend)
    _gain_sensing_launch!(backend, kernels.shift, output, workspace.buffer,
        side, side ÷ 2; ndrange=(side, side))
    output ./= real(eltype(output))(side)
    KernelAbstractions.synchronize(backend)
    return output
end

function _gain_sensing_centered_ifft!(
    output, workspace, backend, kernels, input, side)
    _gain_sensing_launch!(backend, kernels.shift, workspace.buffer, input,
        side, side ÷ 2; ndrange=(side, side))
    OpticalGains._execute_gain_sensing_fft_plan!(workspace.buffer,
        workspace.ifft_plan)
    KernelAbstractions.synchronize(backend)
    _gain_sensing_launch!(backend, kernels.shift, output, workspace.buffer,
        side, side ÷ 2; ndrange=(side, side))
    output .*= real(eltype(output))(side)
    KernelAbstractions.synchronize(backend)
    return output
end

function _gain_sensing_impulse_response!(
    output, workspace, backend, kernels, mask, mask_fft, normalized_frame, side)
    @. workspace.forward = mask * normalized_frame
    KernelAbstractions.synchronize(backend)
    _gain_sensing_centered_fft!(workspace.inverse, workspace, backend, kernels,
        workspace.forward, side)
    @. output = 2 * imag(conj(mask_fft) * workspace.inverse)
    KernelAbstractions.synchronize(backend)
    return output
end

function _gain_sensing_sensitivity!(
    output, workspace, backend, kernels, basis_products, first_ir, second_ir, side)
    _gain_sensing_centered_fft!(workspace.forward, workspace, backend, kernels,
        first_ir, side)
    _gain_sensing_centered_ifft!(workspace.inverse, workspace, backend, kernels,
        second_ir, side)
    @. workspace.buffer = workspace.forward * workspace.inverse
    KernelAbstractions.synchronize(backend)
    mul!(output, adjoint(basis_products), workspace.buffer_vector)
    KernelAbstractions.synchronize(backend)
    return output
end

function _gain_sensing_sum_finite!(workspace, plan, source, count)
    kernels = plan.kernels
    backend = plan.backend
    _gain_sensing_launch!(backend, kernels.map_sum, workspace.sum_first,
        workspace.invalid_first, source, count; ndrange=count)
    sums, next_sums = workspace.sum_first, workspace.sum_second
    invalid, next_invalid = workspace.invalid_first, workspace.invalid_second
    while count > 1
        next_count = cld(count, 2)
        _gain_sensing_launch!(backend, kernels.pair_sum, next_sums,
            next_invalid, sums, invalid, count; ndrange=next_count)
        sums, next_sums = next_sums, sums
        invalid, next_invalid = next_invalid, invalid
        count = next_count
    end
    copyto!(workspace.host_sum, 1, sums, 1, 1)
    copyto!(workspace.host_invalid, 1, invalid, 1, 1)
    KernelAbstractions.synchronize(backend)
    return workspace.host_sum[1], iszero(workspace.host_invalid[1])
end

function _gain_sensing_all_finite(workspace, plan, source, count)
    kernels = plan.kernels
    backend = plan.backend
    _gain_sensing_launch!(backend, kernels.map_finite, workspace.invalid_first,
        source, count; ndrange=count)
    invalid, next_invalid = workspace.invalid_first, workspace.invalid_second
    while count > 1
        next_count = cld(count, 2)
        _gain_sensing_launch!(backend, kernels.pair_invalid, next_invalid,
            invalid, count; ndrange=next_count)
        invalid, next_invalid = next_invalid, invalid
        count = next_count
    end
    copyto!(workspace.host_invalid, 1, invalid, 1, 1)
    KernelAbstractions.synchronize(backend)
    return iszero(workspace.host_invalid[1])
end

function AOC.prepare(
    method::OpticalGains.GainSensing,
    execution::AOC.KernelExecution{S,B},
) where {S<:OpticalGains.GainSensingSpecification,B<:KernelAbstractions.Backend}
    execution.backend isa KernelAbstractions.CPU && throw(ArgumentError(
        "gain sensing kernel execution requires an accelerator backend",
    ))
    return GainSensingKernelPlan(method, execution)
end

function GainSensingKernelPlan(
    method::OpticalGains.GainSensing,
    execution::AOC.KernelExecution{<:OpticalGains.GainSensingSpecification},
)
    specification = execution.specification
    Base.require_one_based_indexing(specification.mask, specification.basis,
        specification.reference_frame)
    T = eltype(specification.reference_frame)
    host_specification = OpticalGains.GainSensingSpecification(
        Array(specification.mask), Array(specification.basis),
        Array(specification.reference_frame),
    )
    side, mode_count, reference_total =
        OpticalGains._validate_gain_sensing_specification(host_specification)
    sensitivity_floor = OpticalGains._prepared_sensitivity_floor(method, T)
    normalized_reference = Matrix{T}(undef, side, side)
    OpticalGains._normalize_frame!(normalized_reference,
        host_specification.reference_frame, reference_total)
    all(isfinite, normalized_reference) || throw(ArgumentError(
        "gain-sensing reference normalization produced non-finite values",
    ))
    normalized_reference_total = sum(normalized_reference)
    isfinite(normalized_reference_total) || throw(ArgumentError(
        "gain-sensing normalized reference total flux must be finite",
    ))
    backend = execution.backend
    kernels = _gain_sensing_kernels(backend, execution.workgroup_size)
    mask = copy_to_backend(backend, host_specification.mask)
    basis = copy_to_backend(backend, host_specification.basis)
    reference = copy_to_backend(backend, normalized_reference)
    workspace = _gain_sensing_kernel_workspace(backend, T, side, mode_count)
    mask_fft = similar(mask)
    _gain_sensing_centered_fft!(mask_fft, workspace, backend, kernels, mask, side)
    all(isfinite, Array(mask_fft)) || throw(ArgumentError(
        "gain-sensing mask transform produced non-finite values",
    ))
    sample_count = side * side
    basis_products = KernelAbstractions.allocate(backend, Complex{T},
        sample_count, mode_count)
    mode_plane = KernelAbstractions.allocate(backend, T, side, side)
    for mode in 1:mode_count
        _gain_sensing_launch!(backend, kernels.mode_plane, mode_plane,
            basis, mode, sample_count; ndrange=sample_count)
        _gain_sensing_centered_fft!(workspace.forward, workspace, backend,
            kernels, mode_plane, side)
        _gain_sensing_centered_ifft!(workspace.inverse, workspace, backend,
            kernels, mode_plane, side)
        _gain_sensing_launch!(backend, kernels.basis_products, basis_products,
            workspace.forward, workspace.inverse, mode, sample_count;
            ndrange=sample_count)
    end
    all(isfinite, Array(basis_products)) || throw(ArgumentError(
        "gain-sensing basis products contain non-finite values",
    ))
    reference_ir = KernelAbstractions.allocate(backend, T, side, side)
    _gain_sensing_impulse_response!(reference_ir, workspace, backend, kernels,
        mask, mask_fft, reference, side)
    all(isfinite, Array(reference_ir)) || throw(ArgumentError(
        "gain-sensing reference impulse response contains non-finite values",
    ))
    reference_sensitivities = KernelAbstractions.allocate(backend, Complex{T},
        mode_count)
    _gain_sensing_sensitivity!(reference_sensitivities, workspace, backend,
        kernels, basis_products, reference_ir, reference_ir, side)
    host_sensitivities = Array(reference_sensitivities)
    all(isfinite, host_sensitivities) || throw(ArgumentError(
        "gain-sensing reference sensitivities contain non-finite values",
    ))
    weak_modes = copy_to_backend(backend,
        Vector{Bool}(abs.(host_sensitivities) .<= sensitivity_floor))
    KernelAbstractions.synchronize(backend)
    return GainSensingKernelPlan(
        side, sample_count, mode_count, sensitivity_floor, backend, mask,
        mask_fft, basis_products, reference_ir, reference_sensitivities,
        weak_modes, reference_total, T(normalized_reference_total), kernels,
    )
end

function AOC.allocate_result(plan::GainSensingKernelPlan{T}) where {T<:AbstractFloat}
    backend = plan.backend
    gains = KernelAbstractions.allocate(backend, T, plan.mode_count)
    sensitivities = KernelAbstractions.allocate(backend, Complex{T}, plan.mode_count)
    weak_modes = KernelAbstractions.allocate(backend, Bool, plan.mode_count)
    fill!(gains, one(T))
    fill!(sensitivities, zero(Complex{T}))
    copyto!(weak_modes, plan.weak_mode_mask)
    KernelAbstractions.synchronize(backend)
    diagnostics = OpticalGains.GainSensingFluxDiagnostics(
        plan.reference_total_flux, plan.normalized_reference_total_flux,
        zero(T), zero(T),
    )
    return OpticalGains.GainSensingProduct(
        gains, sensitivities, weak_modes, diagnostics)
end

function AOC.allocate_workspace(plan::GainSensingKernelPlan{T}) where {T<:AbstractFloat}
    return _gain_sensing_kernel_workspace(plan.backend, T, plan.side, plan.mode_count)
end

function _gain_sensing_validate_execution(product, workspace, plan, inputs)
    frame = inputs.current_frame
    size(frame) == (plan.side, plan.side) || throw(DimensionMismatch(
        "gain-sensing current_frame must have shape $((plan.side, plan.side))",
    ))
    length(product.optical_gains) == plan.mode_count &&
        length(product.current_sensitivities) == plan.mode_count &&
        length(product.weak_mode_mask) == plan.mode_count || throw(
        DimensionMismatch("gain-sensing product has incompatible mode count"),
    )
    size(workspace.buffer) == (plan.side, plan.side) &&
        size(workspace.forward) == (plan.side, plan.side) &&
        size(workspace.inverse) == (plan.side, plan.side) &&
        size(workspace.normalized_current_frame) == (plan.side, plan.side) &&
        size(workspace.current_impulse_response) == (plan.side, plan.side) &&
        length(workspace.buffer_vector) == plan.sample_count &&
        length(workspace.current_sensitivities) == plan.mode_count &&
        length(workspace.candidate_gains) == plan.mode_count &&
        length(workspace.candidate_weak_mode_mask) == plan.mode_count &&
        length(workspace.sum_first) >= max(plan.sample_count, plan.mode_count) &&
        length(workspace.sum_second) >= max(plan.sample_count, plan.mode_count) &&
        length(workspace.invalid_first) >= max(plan.sample_count, plan.mode_count) &&
        length(workspace.invalid_second) >= max(plan.sample_count, plan.mode_count) &&
        length(workspace.host_sum) == 1 &&
        length(workspace.host_invalid) == 1 || throw(DimensionMismatch(
        "gain-sensing kernel workspace has incompatible shape",
    ))
    Base.require_one_based_indexing(frame, product.optical_gains,
        product.current_sensitivities, product.weak_mode_mask,
        workspace.buffer, workspace.buffer_vector, workspace.forward,
        workspace.inverse, workspace.normalized_current_frame,
        workspace.current_impulse_response, workspace.current_sensitivities,
        workspace.candidate_gains, workspace.candidate_weak_mode_mask,
        workspace.sum_first, workspace.sum_second, workspace.invalid_first,
        workspace.invalid_second)
    plan_arrays = (plan.mask, plan.mask_fft, plan.basis_products,
        plan.reference_impulse_response, plan.reference_sensitivities,
        plan.weak_mode_mask)
    product_arrays = (product.optical_gains, product.current_sensitivities,
        product.weak_mode_mask)
    workspace_arrays = (workspace.buffer, workspace.forward, workspace.inverse,
        workspace.normalized_current_frame, workspace.current_impulse_response,
        workspace.current_sensitivities, workspace.candidate_gains,
        workspace.candidate_weak_mode_mask, workspace.sum_first,
        workspace.sum_second, workspace.invalid_first, workspace.invalid_second)
    require_backend(plan.backend, frame, plan_arrays..., product_arrays...,
        workspace_arrays..., workspace.buffer_vector)
    Base.mightalias(workspace.buffer_vector, workspace.buffer) &&
        pointer(workspace.buffer_vector) == pointer(workspace.buffer) || throw(
        ArgumentError("gain-sensing FFT buffer vector must start at its buffer"),
    )
    OpticalGains._require_gain_sensing_pairwise_disjoint(product_arrays,
        "gain-sensing product arrays must not alias")
    OpticalGains._require_gain_sensing_pairwise_disjoint(workspace_arrays,
        "gain-sensing workspace arrays must not alias")
    OpticalGains._require_gain_sensing_disjoint(product_arrays, workspace_arrays,
        "gain-sensing product and workspace arrays must not alias")
    OpticalGains._require_gain_sensing_disjoint(product_arrays, plan_arrays,
        "gain-sensing product and plan arrays must not alias")
    OpticalGains._require_gain_sensing_disjoint(workspace_arrays, plan_arrays,
        "gain-sensing workspace and plan arrays must not alias")
    OpticalGains._require_gain_sensing_disjoint(product_arrays, (frame,),
        "gain-sensing product and current_frame must not alias")
    OpticalGains._require_gain_sensing_disjoint(workspace_arrays, (frame,),
        "gain-sensing workspace and current_frame must not alias")
    return nothing
end

function AOC.process!(
    product::OpticalGains.GainSensingProduct{T},
    workspace::GainSensingKernelWorkspace{T},
    plan::GainSensingKernelPlan{T},
    inputs::OpticalGains.GainSensingInputs{T},
) where {T<:AbstractFloat}
    _gain_sensing_validate_execution(product, workspace, plan, inputs)
    current_total, finite_frame = _gain_sensing_sum_finite!(workspace, plan,
        inputs.current_frame, plan.sample_count)
    finite_frame || throw(ArgumentError(
        "gain-sensing current_frame must contain only finite values",
    ))
    isfinite(current_total) && current_total > zero(T) || throw(ArgumentError(
        "gain-sensing current_frame total flux must be finite and positive",
    ))
    inverse_total = inv(current_total)
    @. workspace.normalized_current_frame = inputs.current_frame * inverse_total
    KernelAbstractions.synchronize(plan.backend)
    normalized_total, finite_normalized = _gain_sensing_sum_finite!(workspace,
        plan, workspace.normalized_current_frame, plan.sample_count)
    finite_normalized && isfinite(normalized_total) || throw(ArgumentError(
        "gain-sensing current normalization produced non-finite values",
    ))
    _gain_sensing_impulse_response!(workspace.current_impulse_response,
        workspace, plan.backend, plan.kernels, plan.mask, plan.mask_fft,
        workspace.normalized_current_frame, plan.side)
    _gain_sensing_all_finite(workspace, plan,
        workspace.current_impulse_response, plan.sample_count) || throw(
        ArgumentError("gain-sensing current impulse response contains non-finite values"),
    )
    _gain_sensing_sensitivity!(workspace.current_sensitivities, workspace,
        plan.backend, plan.kernels, plan.basis_products,
        workspace.current_impulse_response, plan.reference_impulse_response,
        plan.side)
    _gain_sensing_all_finite(workspace, plan,
        workspace.current_sensitivities, plan.mode_count) || throw(
        ArgumentError("gain-sensing current sensitivities contain non-finite values"),
    )
    _gain_sensing_launch!(plan.backend, plan.kernels.candidate,
        workspace.candidate_gains, workspace.candidate_weak_mode_mask,
        workspace.current_sensitivities, plan.reference_sensitivities,
        plan.weak_mode_mask, plan.mode_count; ndrange=plan.mode_count)
    _gain_sensing_all_finite(workspace, plan,
        workspace.candidate_gains, plan.mode_count) || throw(ArgumentError(
        "gain-sensing optical-gain calculation produced a non-finite value",
    ))
    copyto!(product.optical_gains, workspace.candidate_gains)
    copyto!(product.current_sensitivities, workspace.current_sensitivities)
    copyto!(product.weak_mode_mask, workspace.candidate_weak_mode_mask)
    KernelAbstractions.synchronize(plan.backend)
    product.diagnostics = OpticalGains.GainSensingFluxDiagnostics(
        plan.reference_total_flux, plan.normalized_reference_total_flux,
        current_total, normalized_total,
    )
    return product
end
