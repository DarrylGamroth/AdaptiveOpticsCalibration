# Two-sided response probes with an AO Hadamard precedent in Meimon et al. (2015). See docs/REFERENCES.md.
abstract type AbstractPushPullBasis <: AbstractCalibrationMethod end

"""Select positive/negative probing of an explicit modal command matrix."""
struct ModalPushPull{T<:Real,M<:AbstractMatrix{T}} <: AbstractPushPullBasis
    positive_commands::M
end

"""Select positive/negative single-coordinate probing."""
struct ZonalPushPull{T<:Real,V<:AbstractVector{T}} <: AbstractPushPullBasis
    amplitudes::V
end

"""Select balanced positive/negative Sylvester-Hadamard probing."""
struct HadamardPushPull{T<:Real,V<:AbstractVector{T}} <: AbstractPushPullBasis
    amplitudes::V
end

function _prepared_amplitudes(amplitudes::AbstractVector{T}, name) where {T<:Real}
    isempty(amplitudes) && throw(ArgumentError("$name amplitudes must be nonempty"))
    F = _float_type(T, "$name probe-basis construction")
    prepared = Vector{F}(amplitudes)
    for amplitude in prepared
        isfinite(amplitude) && amplitude > zero(F) || throw(ArgumentError(
            "$name amplitudes must be finite and positive",
        ))
    end
    return prepared
end

"""Prepared explicit modal push-pull command basis."""
struct ModalPushPullPlan{T<:AbstractFloat,M<:AbstractMatrix{T}} <:
       AbstractCalibrationPlan
    positive_commands::M
    pattern_count::Int
    corrector_coordinate_count::Int
end

"""Prepared zonal push-pull command basis."""
struct ZonalPushPullPlan{T<:AbstractFloat,V<:AbstractVector{T}} <:
       AbstractCalibrationPlan
    amplitudes::V
    pattern_count::Int
    corrector_coordinate_count::Int
end

"""Prepared balanced Sylvester-Hadamard push-pull command basis."""
struct HadamardPushPullPlan{T<:AbstractFloat,V<:AbstractVector{T}} <:
       AbstractCalibrationPlan
    amplitudes::V
    pattern_count::Int
    corrector_coordinate_count::Int
end

function prepare(method::ModalPushPull{T}, ::Nothing=nothing) where {T}
    Base.require_one_based_indexing(method.positive_commands)
    modes, correctors = size(method.positive_commands)
    modes > 0 && correctors > 0 || throw(ArgumentError(
        "modal positive-command dimensions must be positive",
    ))
    _checked_extent("modal command basis", 2, modes, correctors)
    F = _float_type(T, "modal probe-basis construction")
    commands = Matrix{F}(method.positive_commands)
    for mode in axes(commands, 1)
        all(isfinite, @view(commands[mode, :])) || throw(ArgumentError(
            "modal positive commands must contain only finite coefficients",
        ))
        any(!iszero, @view(commands[mode, :])) || throw(ArgumentError(
            "each modal positive command must be nonzero",
        ))
    end
    return ModalPushPullPlan(commands, modes, correctors)
end

function prepare(method::ZonalPushPull, ::Nothing=nothing)
    Base.require_one_based_indexing(method.amplitudes)
    amplitudes = _prepared_amplitudes(method.amplitudes, "zonal")
    count = length(amplitudes)
    _checked_extent("zonal command basis", 2, count, count)
    return ZonalPushPullPlan(amplitudes, count, count)
end

function prepare(method::HadamardPushPull, ::Nothing=nothing)
    Base.require_one_based_indexing(method.amplitudes)
    amplitudes = _prepared_amplitudes(method.amplitudes, "Hadamard")
    correctors = length(amplitudes)
    order = _hadamard_order(correctors)
    _checked_extent("Hadamard command basis", 2, order, correctors)
    return HadamardPushPullPlan(amplitudes, order, correctors)
end

"""
    ProbeCommandBasisProduct(commands)

Wrap caller-owned command-sequence storage. Rows are ordered probe patterns;
columns are corrector coordinates. Positive and negative rows are interleaved.
"""
struct ProbeCommandBasisProduct{T<:AbstractFloat,M<:AbstractMatrix{T}}
    commands::M
end

"""Return the caller-owned command matrix in a probe-basis result."""
probe_commands(result::ProbeCommandBasisProduct) = result.commands

function allocate_result(
    plan::Union{ModalPushPullPlan{T},ZonalPushPullPlan{T},HadamardPushPullPlan{T}},
) where {T}
    return ProbeCommandBasisProduct(
        zeros(T, 2 * plan.pattern_count, plan.corrector_coordinate_count),
    )
end

allocate_workspace(::Union{ModalPushPullPlan,ZonalPushPullPlan,HadamardPushPullPlan}) = nothing

function _validate_probe_product(result::ProbeCommandBasisProduct, plan)
    Base.require_one_based_indexing(result.commands)
    expected = (2 * plan.pattern_count, plan.corrector_coordinate_count)
    size(result.commands) == expected || throw(DimensionMismatch(
        "probe-command result must have shape $expected",
    ))
    return nothing
end

"""Construct the complete interleaved modal push-pull command sequence."""
function process!(
    result::ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::ModalPushPullPlan{T},
    ::Nothing,
) where {T}
    _validate_probe_product(result, plan)
    for mode in 1:plan.pattern_count
        positive_row = 2 * mode - 1
        negative_row = positive_row + 1
        for corrector in 1:plan.corrector_coordinate_count
            coefficient = plan.positive_commands[mode, corrector]
            result.commands[positive_row, corrector] = coefficient
            result.commands[negative_row, corrector] = -coefficient
        end
    end
    return result
end

"""Construct the complete interleaved zonal push-pull command sequence."""
function process!(
    result::ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::ZonalPushPullPlan{T},
    ::Nothing,
) where {T}
    _validate_probe_product(result, plan)
    fill!(result.commands, zero(T))
    for corrector in 1:plan.corrector_coordinate_count
        positive_row = 2 * corrector - 1
        negative_row = positive_row + 1
        result.commands[positive_row, corrector] = plan.amplitudes[corrector]
        result.commands[negative_row, corrector] = -plan.amplitudes[corrector]
    end
    return result
end

"""Construct the complete interleaved balanced Hadamard push-pull sequence."""
function process!(
    result::ProbeCommandBasisProduct{T},
    ::Nothing,
    plan::HadamardPushPullPlan{T},
    ::Nothing,
) where {T}
    _validate_probe_product(result, plan)
    for row in 0:(plan.pattern_count - 1)
        positive_row = 2 * row + 1
        negative_row = positive_row + 1
        for corrector in 1:plan.corrector_coordinate_count
            coefficient = iseven(count_ones(row & corrector)) ?
                plan.amplitudes[corrector] : -plan.amplitudes[corrector]
            result.commands[positive_row, corrector] = coefficient
            result.commands[negative_row, corrector] = -coefficient
        end
    end
    return result
end
