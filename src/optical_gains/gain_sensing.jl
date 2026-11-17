"""
    GainSensing()
    GainSensing(sensitivity_floor)

Select complete-image modal optical-gain estimation. On an even `N × N` grid,
the CPU reference uses `F(x) = fftshift(fft(fftshift(x))) / N` and
`I(x) = fftshift(ifft(fftshift(x))) * N`. For normalized frame `p`, its real
impulse response is `2 * imag(conj(F(mask)) .* F(mask .* p))`. Each modal
sensitivity contracts `conj(F(basisₖ) .* I(basisₖ))` with
`F(IR_current) .* I(IR_reference)`; a weak reference sensitivity has absolute
value at most `sensitivity_floor` and returns unity rather than a ratio.
"""
struct GainSensing{F} <: AbstractCalibrationMethod
    sensitivity_floor::F
end

"""Use the floating-point representation's source-parity sensitivity floor."""
GainSensing() = GainSensing{Nothing}(nothing)

function GainSensing(sensitivity_floor::Real)
    T = float(typeof(sensitivity_floor))
    T <: AbstractFloat || throw(ArgumentError(
        "gain-sensing sensitivity_floor must use an AbstractFloat representation",
    ))
    floor = T(sensitivity_floor)
    isfinite(floor) && floor >= zero(T) || throw(ArgumentError(
        "gain-sensing sensitivity_floor must be finite and nonnegative",
    ))
    return GainSensing{T}(floor)
end

"""
    GainSensingSpecification(mask, basis, reference_frame)

Declare a complete CPU-host gain-sensing calibration input. `mask` is a finite
complex focal-plane transmission, `basis[row, column, mode]` is a finite real
modal basis already aligned with the mask and frames, and `reference_frame` is
one complete finite modulation-cycle image. All arrays use the same Float32 or
Float64 representation, have one-based axes, and are copied during preparation.
The initial target is an even square CPU grid; neither padding nor modal-basis
normalization is performed.
"""
struct GainSensingSpecification{
    T<:AbstractFloat,
    M<:AbstractMatrix{Complex{T}},
    B<:AbstractArray{T,3},
    R<:AbstractMatrix{T},
}
    mask::M
    basis::B
    reference_frame::R
end

function GainSensingSpecification(
    ::AbstractMatrix,
    ::AbstractArray,
    ::AbstractMatrix,
)
    throw(ArgumentError(
        "gain-sensing mask, basis, and reference_frame must use one shared Float32 or Float64 representation; mask elements must be Complex{T}",
    ))
end

"""Bundle one complete current modulation-cycle image without copying it."""
struct GainSensingInputs{T<:AbstractFloat,F<:AbstractMatrix{T}}
    current_frame::F
end

GainSensingInputs(::AbstractMatrix) = throw(ArgumentError(
    "gain-sensing current_frame must use Float32 or Float64 storage",
))

"""Raw and normalized flux totals from one complete gain-sensing operation."""
struct GainSensingFluxDiagnostics{T<:AbstractFloat}
    reference_total_flux::T
    normalized_reference_total_flux::T
    current_total_flux::T
    normalized_current_total_flux::T
end

"""
Prepared complete-image gain-sensing contract.

The plan owns copied mask and reference-derived numerical products. It does not
retain caller-visible products, transient current-frame storage, FFT workspace,
or physical camera state.
"""
struct GainSensingPlan{
    T<:AbstractFloat,
    M<:Matrix{Complex{T}},
    BP<:Matrix{Complex{T}},
    R<:Matrix{T},
    S<:Vector{Complex{T}},
    W<:Vector{Bool},
} <: AbstractCalibrationPlan
    side::Int
    sample_count::Int
    mode_count::Int
    sensitivity_floor::T
    mask::M
    mask_fft::M
    basis_products::BP
    reference_impulse_response::R
    reference_sensitivities::S
    weak_mode_mask::W
    reference_total_flux::T
    normalized_reference_total_flux::T
end

"""Caller-owned complete-image optical-gain product and diagnostics."""
mutable struct GainSensingProduct{
    T<:AbstractFloat,
    G<:AbstractVector{T},
    S<:AbstractVector{Complex{T}},
    W<:AbstractVector{Bool},
}
    optical_gains::G
    current_sensitivities::S
    weak_mode_mask::W
    diagnostics::GainSensingFluxDiagnostics{T}
end

"""Return ordered dimensionless signed modal optical-gain ratios."""
optical_gains(product::GainSensingProduct) = product.optical_gains
"""Return ordered complex sensitivities of the current complete image."""
current_sensitivities(product::GainSensingProduct) = product.current_sensitivities
"""Return reference-conditioned weak-mode flags in modal-basis order."""
weak_mode_mask(product::GainSensingProduct) = product.weak_mode_mask
"""Return raw and normalized reference/current flux totals."""
flux_diagnostics(product::GainSensingProduct) = product.diagnostics

"""
Replaceable FFTW CPU scratch for one gain-sensing operation.

The FFTW plans and all mutable transform/staging storage belong here so a plan
can be reused concurrently with independent workspaces.
"""
struct GainSensingWorkspace{
    T<:AbstractFloat,
    C<:Matrix{Complex{T}},
    BV<:Vector{Complex{T}},
    R<:Matrix{T},
    S<:Vector{Complex{T}},
    G<:Vector{T},
    W<:Vector{Bool},
    PF,
    PI,
}
    buffer::C
    buffer_vector::BV
    forward::C
    inverse::C
    normalized_current_frame::R
    current_impulse_response::R
    current_sensitivities::S
    candidate_gains::G
    candidate_weak_mode_mask::W
    fft_plan::PF
    ifft_plan::PI
end

_gain_sensing_cpu_storage(::Array) = true
_gain_sensing_cpu_storage(array::SubArray) = _gain_sensing_cpu_storage(parent(array))
_gain_sensing_cpu_storage(::AbstractArray) = false

function _require_gain_sensing_cpu_storage(name::AbstractString, array)
    _gain_sensing_cpu_storage(array) || throw(ArgumentError(
        "$name must use CPU host Array storage or a view of CPU host Array storage",
    ))
    return nothing
end

function _gain_sensing_float_type(::Type{T}) where {T<:AbstractFloat}
    (T === Float32 || T === Float64) || throw(ArgumentError(
        "gain sensing supports Float32 and Float64 CPU FFTW execution",
    ))
    return T
end

function _prepared_sensitivity_floor(method::GainSensing{Nothing}, ::Type{T}) where {T<:AbstractFloat}
    return sqrt(eps(T))
end

function _prepared_sensitivity_floor(method::GainSensing, ::Type{T}) where {T<:AbstractFloat}
    floor = T(method.sensitivity_floor)
    isfinite(floor) && floor >= zero(T) || throw(ArgumentError(
        "gain-sensing sensitivity_floor must be finite and nonnegative after conversion",
    ))
    return floor
end

function _validate_gain_sensing_specification(specification::GainSensingSpecification{T}) where {T<:AbstractFloat}
    _gain_sensing_float_type(T)
    mask = specification.mask
    basis = specification.basis
    reference = specification.reference_frame
    Base.require_one_based_indexing(mask, basis, reference)
    _require_gain_sensing_cpu_storage("gain-sensing mask", mask)
    _require_gain_sensing_cpu_storage("gain-sensing basis", basis)
    _require_gain_sensing_cpu_storage("gain-sensing reference_frame", reference)
    side_rows, side_columns = size(mask)
    side_rows > 0 || throw(ArgumentError("gain-sensing mask must be nonempty"))
    side_rows == side_columns || throw(DimensionMismatch(
        "gain-sensing mask must be square",
    ))
    iseven(side_rows) || throw(ArgumentError(
        "gain-sensing mask side length must be even",
    ))
    mode_count = size(basis, 3)
    mode_count > 0 || throw(ArgumentError("gain-sensing basis must have at least one mode"))
    size(basis) == (side_rows, side_rows, mode_count) || throw(DimensionMismatch(
        "gain-sensing basis must have shape $((side_rows, side_rows, mode_count))",
    ))
    size(reference) == (side_rows, side_rows) || throw(DimensionMismatch(
        "gain-sensing reference_frame must have shape $((side_rows, side_rows))",
    ))
    _checked_extent("gain-sensing samples", side_rows, side_rows)
    _checked_extent("gain-sensing basis", side_rows, side_rows, mode_count)
    all(isfinite, mask) || throw(ArgumentError(
        "gain-sensing mask must contain only finite values",
    ))
    all(isfinite, basis) || throw(ArgumentError(
        "gain-sensing basis must contain only finite values",
    ))
    all(isfinite, reference) || throw(ArgumentError(
        "gain-sensing reference_frame must contain only finite values",
    ))
    reference_total = sum(reference)
    isfinite(reference_total) && reference_total > zero(T) || throw(ArgumentError(
        "gain-sensing reference_frame total flux must be finite and positive",
    ))
    return side_rows, mode_count, T(reference_total)
end

function _allocate_gain_sensing_workspace(
    ::Type{T},
    side::Int,
    mode_count::Int,
) where {T<:AbstractFloat}
    buffer = Matrix{Complex{T}}(undef, side, side)
    buffer_vector = vec(buffer)
    forward = similar(buffer)
    inverse = similar(buffer)
    normalized_current_frame = Matrix{T}(undef, side, side)
    current_impulse_response = similar(normalized_current_frame)
    current_sensitivities = Vector{Complex{T}}(undef, mode_count)
    candidate_gains = Vector{T}(undef, mode_count)
    candidate_weak_mode_mask = Vector{Bool}(undef, mode_count)
    fft_plan = FFTW.plan_fft!(buffer)
    ifft_plan = FFTW.plan_ifft!(buffer)
    return GainSensingWorkspace(
        buffer,
        buffer_vector,
        forward,
        inverse,
        normalized_current_frame,
        current_impulse_response,
        current_sensitivities,
        candidate_gains,
        candidate_weak_mode_mask,
        fft_plan,
        ifft_plan,
    )
end

function allocate_workspace(plan::GainSensingPlan{T}) where {T<:AbstractFloat}
    return _allocate_gain_sensing_workspace(T, plan.side, plan.mode_count)
end

function _fftshift2d!(destination::Matrix, source::AbstractMatrix)
    side = size(destination, 1)
    half_side = div(side, 2)
    for column in 1:side
        source_column = mod1(column + half_side, side)
        for row in 1:side
            source_row = mod1(row + half_side, side)
            destination[row, column] = source[source_row, source_column]
        end
    end
    return destination
end

_execute_gain_sensing_fft_plan!(buffer, plan) =
    (mul!(buffer, plan, buffer); buffer)

function _centered_fft!(
    output::Matrix{Complex{T}},
    workspace::GainSensingWorkspace{T},
    input::AbstractMatrix,
) where {T<:AbstractFloat}
    _fftshift2d!(workspace.buffer, input)
    mul!(workspace.buffer, workspace.fft_plan, workspace.buffer)
    _fftshift2d!(output, workspace.buffer)
    output ./= T(size(output, 1))
    return output
end

function _centered_ifft!(
    output::Matrix{Complex{T}},
    workspace::GainSensingWorkspace{T},
    input::AbstractMatrix,
) where {T<:AbstractFloat}
    _fftshift2d!(workspace.buffer, input)
    mul!(workspace.buffer, workspace.ifft_plan, workspace.buffer)
    _fftshift2d!(output, workspace.buffer)
    output .*= T(size(output, 1))
    return output
end

function _normalize_frame!(destination::Matrix{T}, frame::AbstractMatrix{T}, total::T) where {T<:AbstractFloat}
    inverse_total = inv(total)
    @. destination = frame * inverse_total
    return destination
end

function _impulse_response!(
    output::Matrix{T},
    workspace::GainSensingWorkspace{T},
    mask::Matrix{Complex{T}},
    mask_fft::Matrix{Complex{T}},
    normalized_frame::AbstractMatrix{T},
) where {T<:AbstractFloat}
    @. workspace.forward = mask * normalized_frame
    _centered_fft!(workspace.inverse, workspace, workspace.forward)
    @. output = T(2) * imag(conj(mask_fft) * workspace.inverse)
    return output
end

function _sensitivity!(
    output::Vector{Complex{T}},
    workspace::GainSensingWorkspace{T},
    basis_products::Matrix{Complex{T}},
    first_ir::AbstractMatrix{T},
    second_ir::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _centered_fft!(workspace.forward, workspace, first_ir)
    _centered_ifft!(workspace.inverse, workspace, second_ir)
    @. workspace.buffer = workspace.forward * workspace.inverse
    mul!(output, adjoint(basis_products), workspace.buffer_vector)
    return output
end

function prepare(
    method::GainSensing,
    specification::GainSensingSpecification{T},
) where {T<:AbstractFloat}
    side, mode_count, reference_total =
        _validate_gain_sensing_specification(specification)
    sensitivity_floor = _prepared_sensitivity_floor(method, T)
    mask = Matrix{Complex{T}}(specification.mask)
    basis = Array{T,3}(specification.basis)
    normalized_reference_frame = Matrix{T}(undef, side, side)
    _normalize_frame!(normalized_reference_frame, specification.reference_frame,
        reference_total)
    all(isfinite, normalized_reference_frame) || throw(ArgumentError(
        "gain-sensing reference normalization produced non-finite values",
    ))
    normalized_reference_total = sum(normalized_reference_frame)
    isfinite(normalized_reference_total) || throw(ArgumentError(
        "gain-sensing normalized reference total flux must be finite",
    ))
    workspace = _allocate_gain_sensing_workspace(T, side, mode_count)
    mask_fft = similar(mask)
    _centered_fft!(mask_fft, workspace, mask)
    all(isfinite, mask_fft) || throw(ArgumentError(
        "gain-sensing mask transform produced non-finite values",
    ))
    basis_products = Matrix{Complex{T}}(undef, side * side, mode_count)
    for mode in 1:mode_count
        basis_mode = @view basis[:, :, mode]
        _centered_fft!(workspace.forward, workspace, basis_mode)
        _centered_ifft!(workspace.inverse, workspace, basis_mode)
        for sample in eachindex(workspace.forward, workspace.inverse)
            basis_products[sample, mode] =
                workspace.forward[sample] * workspace.inverse[sample]
        end
    end
    all(isfinite, basis_products) || throw(ArgumentError(
        "gain-sensing basis products contain non-finite values",
    ))
    reference_ir = Matrix{T}(undef, side, side)
    _impulse_response!(reference_ir, workspace, mask, mask_fft,
        normalized_reference_frame)
    all(isfinite, reference_ir) || throw(ArgumentError(
        "gain-sensing reference impulse response contains non-finite values",
    ))
    reference_sensitivities = Vector{Complex{T}}(undef, mode_count)
    _sensitivity!(reference_sensitivities, workspace, basis_products,
        reference_ir, reference_ir)
    all(isfinite, reference_sensitivities) || throw(ArgumentError(
        "gain-sensing reference sensitivities contain non-finite values",
    ))
    weak_modes = Vector{Bool}(abs.(reference_sensitivities) .<= sensitivity_floor)
    return GainSensingPlan(
        side,
        _checked_extent("gain-sensing samples", side, side),
        mode_count,
        sensitivity_floor,
        mask,
        mask_fft,
        basis_products,
        reference_ir,
        reference_sensitivities,
        weak_modes,
        reference_total,
        T(normalized_reference_total),
    )
end

function allocate_result(plan::GainSensingPlan{T}) where {T<:AbstractFloat}
    diagnostics = GainSensingFluxDiagnostics(
        plan.reference_total_flux,
        plan.normalized_reference_total_flux,
        zero(T),
        zero(T),
    )
    return GainSensingProduct(
        ones(T, plan.mode_count),
        zeros(Complex{T}, plan.mode_count),
        copy(plan.weak_mode_mask),
        diagnostics,
    )
end

function _validate_gain_sensing_workspace(
    plan::GainSensingPlan{T},
    workspace::GainSensingWorkspace{T},
) where {T<:AbstractFloat}
    expected_matrix = (plan.side, plan.side)
    size(workspace.buffer) == expected_matrix || throw(DimensionMismatch(
        "gain-sensing FFT buffer has incompatible shape",
    ))
    length(workspace.buffer_vector) == plan.sample_count || throw(DimensionMismatch(
        "gain-sensing FFT buffer vector has incompatible length",
    ))
    pointer(workspace.buffer_vector) == pointer(workspace.buffer) || throw(ArgumentError(
        "gain-sensing FFT buffer vector must start at the FFT buffer",
    ))
    length(workspace.buffer_vector) == plan.sample_count || throw(DimensionMismatch(
        "gain-sensing FFT buffer vector has incompatible length",
    ))
    Base.mightalias(workspace.buffer_vector, workspace.buffer) || throw(ArgumentError(
        "gain-sensing FFT buffer vector must alias the FFT buffer",
    ))
    size(workspace.forward) == expected_matrix || throw(DimensionMismatch(
        "gain-sensing forward workspace has incompatible shape",
    ))
    size(workspace.inverse) == expected_matrix || throw(DimensionMismatch(
        "gain-sensing inverse workspace has incompatible shape",
    ))
    size(workspace.normalized_current_frame) == expected_matrix || throw(DimensionMismatch(
        "gain-sensing normalized-frame workspace has incompatible shape",
    ))
    size(workspace.current_impulse_response) == expected_matrix || throw(DimensionMismatch(
        "gain-sensing impulse-response workspace has incompatible shape",
    ))
    length(workspace.current_sensitivities) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing sensitivity workspace has incompatible length",
    ))
    length(workspace.candidate_gains) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing gain workspace has incompatible length",
    ))
    length(workspace.candidate_weak_mode_mask) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing weak-mode workspace has incompatible length",
    ))
    return nothing
end

@inline function _gain_sensing_storage_parent(array::AbstractArray)
    owner = parent(array)
    return owner === array ? array : _gain_sensing_storage_parent(owner)
end

@inline _gain_sensing_storage_mightalias(first::AbstractArray,
    second::AbstractArray) = Base.mightalias(first, second)

@inline function _gain_sensing_mightalias(first::AbstractArray,
    second::AbstractArray)
    Base.mightalias(first, second) && return true
    return _gain_sensing_storage_mightalias(
        _gain_sensing_storage_parent(first),
        _gain_sensing_storage_parent(second))
end

@inline _require_gain_sensing_disjoint_one(::AbstractArray, ::Tuple{}, ::AbstractString) = nothing

@inline function _require_gain_sensing_disjoint_one(
    array::AbstractArray,
    others::Tuple,
    message::AbstractString,
)
    _gain_sensing_mightalias(array, first(others)) && throw(ArgumentError(message))
    _require_gain_sensing_disjoint_one(array, Base.tail(others), message)
    return nothing
end

@inline _require_gain_sensing_disjoint(::Tuple{}, ::Tuple, ::AbstractString) = nothing

@inline function _require_gain_sensing_disjoint(
    arrays::Tuple,
    others::Tuple,
    message::AbstractString,
)
    _require_gain_sensing_disjoint_one(first(arrays), others, message)
    _require_gain_sensing_disjoint(Base.tail(arrays), others, message)
    return nothing
end

@inline _require_gain_sensing_pairwise_disjoint(::Tuple{}, ::AbstractString) = nothing

@inline function _require_gain_sensing_pairwise_disjoint(
    arrays::Tuple,
    message::AbstractString,
)
    remaining = Base.tail(arrays)
    _require_gain_sensing_disjoint_one(first(arrays), remaining, message)
    _require_gain_sensing_pairwise_disjoint(remaining, message)
    return nothing
end

function _validate_gain_sensing_processing(
    product::GainSensingProduct{T},
    workspace::GainSensingWorkspace{T},
    plan::GainSensingPlan{T},
    inputs::GainSensingInputs{T},
) where {T<:AbstractFloat}
    frame = inputs.current_frame
    Base.require_one_based_indexing(
        product.optical_gains,
        product.current_sensitivities,
        product.weak_mode_mask,
        frame,
    )
    _require_gain_sensing_cpu_storage("gain-sensing product gains", product.optical_gains)
    _require_gain_sensing_cpu_storage("gain-sensing product sensitivities",
        product.current_sensitivities)
    _require_gain_sensing_cpu_storage("gain-sensing product weak-mode mask",
        product.weak_mode_mask)
    _require_gain_sensing_cpu_storage("gain-sensing current_frame", frame)
    size(frame) == (plan.side, plan.side) || throw(DimensionMismatch(
        "gain-sensing current_frame must have shape $((plan.side, plan.side))",
    ))
    length(product.optical_gains) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing product gains have incompatible length",
    ))
    length(product.current_sensitivities) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing product sensitivities have incompatible length",
    ))
    length(product.weak_mode_mask) == plan.mode_count || throw(DimensionMismatch(
        "gain-sensing product weak-mode mask has incompatible length",
    ))
    _validate_gain_sensing_workspace(plan, workspace)
    product_arrays = (
        product.optical_gains, product.current_sensitivities,
        product.weak_mode_mask)
    workspace_arrays = (
        workspace.buffer, workspace.forward, workspace.inverse,
        workspace.normalized_current_frame, workspace.current_impulse_response,
        workspace.current_sensitivities, workspace.candidate_gains,
        workspace.candidate_weak_mode_mask)
    plan_arrays = (
        plan.mask, plan.mask_fft, plan.basis_products,
        plan.reference_impulse_response, plan.reference_sensitivities,
        plan.weak_mode_mask)
    _require_gain_sensing_pairwise_disjoint(product_arrays,
        "gain-sensing product arrays must not alias")
    _require_gain_sensing_pairwise_disjoint(workspace_arrays,
        "gain-sensing workspace arrays must not alias")
    _require_gain_sensing_disjoint(product_arrays, workspace_arrays,
        "gain-sensing product and workspace arrays must not alias")
    _require_gain_sensing_disjoint(product_arrays, plan_arrays,
        "gain-sensing product and plan arrays must not alias")
    _require_gain_sensing_disjoint(workspace_arrays, plan_arrays,
        "gain-sensing workspace and plan arrays must not alias")
    _require_gain_sensing_disjoint(product_arrays, (frame,),
        "gain-sensing product and current_frame must not alias")
    _require_gain_sensing_disjoint(workspace_arrays, (frame,),
        "gain-sensing workspace and current_frame must not alias")
    return nothing
end

"""
    process!(product, workspace, plan, inputs)

Estimate signed modal optical gains from one complete current modulation-cycle
image. The current frame may contain finite negative samples when its total
flux is positive. Invalid input or a non-finite intermediate throws before any
caller-visible product field is published. The explicit CPU workspace form is
the allocation-control boundary for one-based host `Array` inputs and views;
it makes no accelerator or captured-graph claim.
"""
function process!(
    product::GainSensingProduct{T},
    workspace::GainSensingWorkspace{T},
    plan::GainSensingPlan{T},
    inputs::GainSensingInputs{T},
) where {T<:AbstractFloat}
    _validate_gain_sensing_processing(product, workspace, plan, inputs)
    current_frame = inputs.current_frame
    all(isfinite, current_frame) || throw(ArgumentError(
        "gain-sensing current_frame must contain only finite values",
    ))
    current_total = sum(current_frame)
    isfinite(current_total) && current_total > zero(T) || throw(ArgumentError(
        "gain-sensing current_frame total flux must be finite and positive",
    ))
    _normalize_frame!(workspace.normalized_current_frame, current_frame,
        T(current_total))
    all(isfinite, workspace.normalized_current_frame) || throw(ArgumentError(
        "gain-sensing current normalization produced non-finite values",
    ))
    _impulse_response!(workspace.current_impulse_response, workspace,
        plan.mask, plan.mask_fft, workspace.normalized_current_frame)
    all(isfinite, workspace.current_impulse_response) || throw(ArgumentError(
        "gain-sensing current impulse response contains non-finite values",
    ))
    _sensitivity!(workspace.current_sensitivities, workspace,
        plan.basis_products, workspace.current_impulse_response,
        plan.reference_impulse_response)
    all(isfinite, workspace.current_sensitivities) || throw(ArgumentError(
        "gain-sensing current sensitivities contain non-finite values",
    ))
    for mode in eachindex(workspace.candidate_gains,
            workspace.current_sensitivities, plan.reference_sensitivities,
            plan.weak_mode_mask)
        weak = plan.weak_mode_mask[mode]
        gain = weak ? one(T) : real(
            workspace.current_sensitivities[mode] /
            plan.reference_sensitivities[mode],
        )
        isfinite(gain) || throw(ArgumentError(
            "gain-sensing optical-gain calculation produced a non-finite value",
        ))
        workspace.candidate_gains[mode] = gain
        workspace.candidate_weak_mode_mask[mode] = weak
    end
    normalized_current_total = sum(workspace.normalized_current_frame)
    isfinite(normalized_current_total) || throw(ArgumentError(
        "gain-sensing normalized current total flux must be finite",
    ))
    copyto!(product.optical_gains, workspace.candidate_gains)
    copyto!(product.current_sensitivities, workspace.current_sensitivities)
    copyto!(product.weak_mode_mask, workspace.candidate_weak_mode_mask)
    product.diagnostics = GainSensingFluxDiagnostics(
        plan.reference_total_flux,
        plan.normalized_reference_total_flux,
        T(current_total),
        T(normalized_current_total),
    )
    return product
end
