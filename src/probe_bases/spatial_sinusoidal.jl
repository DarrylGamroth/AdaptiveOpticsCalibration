# Conventional real Fourier basis evaluated in supplied pupil coordinates. See docs/REFERENCES.md.
"""
    SpatialSinusoidalPushPull(amplitudes; normalization, minimum_sampled_peak)

Select spatial sine/cosine corrector modes with one finite positive command
amplitude for each ordered mode: `(sine₁, cosine₁, sine₂, cosine₂, …)`.
`normalization=:none` preserves raw sinusoidal samples; `:peak` divides each
mode by its sampled maximum absolute value. Both options are explicit.
The dimensionless `minimum_sampled_peak` must be finite and in `(0, 1)`;
preparation rejects a raw sampled mode with peak at or below this caller-owned
guard. This guards sampled near-null modes, not dependence across frequencies.
Amplitudes retain caller-declared command units; no gain or unit conversion is
inferred. This is a spatial mode construction, independent of exposure timing.
"""
struct SpatialSinusoidalPushPull{T<:AbstractFloat} <: AbstractPushPullBasis
    amplitudes::Vector{T}
    normalization::Symbol
    minimum_sampled_peak::Float64

    function SpatialSinusoidalPushPull(amplitudes::AbstractVector{<:Real};
                                      normalization::Symbol,
                                      minimum_sampled_peak::Real)
        Base.require_one_based_indexing(amplitudes)
        normalization in (:none, :peak) || throw(ArgumentError(
            "sinusoidal normalization must be :none or :peak",
        ))
        peak_guard = Float64(minimum_sampled_peak)
        isfinite(peak_guard) && 0.0 < peak_guard < 1.0 || throw(ArgumentError(
            "minimum_sampled_peak must be finite and in (0, 1)",
        ))
        prepared = _prepared_amplitudes(amplitudes, "spatial sinusoidal")
        isconcretetype(eltype(prepared)) || throw(ArgumentError(
            "spatial sinusoidal amplitudes require a concrete floating representation",
        ))
        return new{eltype(prepared)}(prepared, normalization, peak_guard)
    end
end

"""
    SpatialSinusoidalSpecification(actuator_positions, spatial_frequencies)

Declare one-based real arrays of shape `(N, 2)` and `(K, 2)`. The first array
contains physical `(x, y)` positions in the exact requested corrector order;
the second contains ordered `(kₓ, kᵧ)` spatial frequencies in cycles per
matching coordinate unit. Preparation copies these arrays for calculation,
without centering, masking, inferring DM geometry, or selecting actuators.
Every frequency must be nonzero; DC would produce a null sine mode.
The prepared floating-point representation promotes the amplitude, position,
and frequency element types through `float(promote_type(...))`, which must
yield a concrete floating-point type.
"""
struct SpatialSinusoidalSpecification{
    P<:AbstractMatrix{<:Real},
    F<:AbstractMatrix{<:Real},
}
    actuator_positions::P
    spatial_frequencies::F
end

"""
    SpatialSinusoidalPushPullPlan

Prepared spatial modes, scaled modal commands, and dimensionless diagnostics.
`basis` has `(2K, N)` rows ordered sine/cosine per supplied frequency, after
the declared normalization. `sampled_peak` and `sampled_rms` describe the raw
sinusoidal rows before normalization; RMS is `sqrt(Σⱼ bⱼ²/N)` over the supplied
actuator positions, with no pupil or influence-function weighting.
RMS accumulation and storage use `promote_type(T, Float64)`, where `T` is
the prepared basis representation; sampled peaks retain `T` for normalization.
`modal_plan` owns the amplitude-scaled positive commands and delegates the
ordinary push-pull operation. The arrays are logically immutable. A partial
Fourier basis calibrates modal directional responses; it does not establish
the complete physical zonal interaction matrix or full basis independence.
"""
struct SpatialSinusoidalPushPullPlan{
    T<:AbstractFloat,D<:AbstractFloat,M<:AbstractMatrix{T},
} <:
       AbstractCalibrationPlan
    modal_plan::ModalPushPullPlan{T,M}
    basis::Matrix{T}
    sampled_peak::Vector{T}
    sampled_rms::Vector{D}
    normalization::Symbol
end

function prepare(method::SpatialSinusoidalPushPull{A},
                 specification::SpatialSinusoidalSpecification{P,F}) where {A,P,F}
    positions = specification.actuator_positions
    frequencies = specification.spatial_frequencies
    Base.require_one_based_indexing(positions, frequencies)
    size(positions, 2) == 2 || throw(DimensionMismatch(
        "actuator_positions must have two coordinate columns",
    ))
    size(frequencies, 2) == 2 || throw(DimensionMismatch(
        "spatial_frequencies must have two coordinate columns",
    ))
    correctors, frequency_count = size(positions, 1), size(frequencies, 1)
    correctors > 0 && frequency_count > 0 || throw(ArgumentError(
        "spatial sinusoidal positions and frequencies must be nonempty",
    ))
    _checked_extent("spatial sinusoidal push-pull basis", 4, frequency_count, correctors)
    modes = 2 * frequency_count
    length(method.amplitudes) == modes || throw(DimensionMismatch(
        "spatial sinusoidal amplitudes must have two entries per frequency",
    ))
    T = _float_type(promote_type(A, eltype(P), eltype(F)), "spatial sinusoidal basis")
    isconcretetype(T) || throw(ArgumentError(
        "spatial sinusoidal inputs require a concrete floating representation",
    ))
    D = promote_type(T, Float64)
    isconcretetype(D) || throw(ArgumentError(
        "spatial sinusoidal RMS requires a concrete floating representation",
    ))
    coordinates = Matrix{T}(positions)
    wavevectors = Matrix{T}(frequencies)
    all(isfinite, coordinates) && all(isfinite, wavevectors) || throw(ArgumentError(
        "spatial sinusoidal coordinates and frequencies must be finite",
    ))
    amplitudes = Vector{T}(_prepared_amplitudes(method.amplitudes, "spatial sinusoidal"))
    basis = Matrix{T}(undef, modes, correctors)
    sampled_peak = zeros(T, modes)
    sampled_rms = zeros(D, modes)
    @inbounds for frequency in 1:frequency_count
        kx, ky = wavevectors[frequency, 1], wavevectors[frequency, 2]
        iszero(kx) && iszero(ky) && throw(ArgumentError(
            "DC spatial frequency is excluded because its sine mode is zero",
        ))
        sine_row, cosine_row = 2 * frequency - 1, 2 * frequency
        for corrector in 1:correctors
            # sinpi/cospi evaluate sin/cos(2π k⋅r), retaining exact sampled
            # half/integer-cycle zeros without rounding a stored 2π constant.
            phase = T(2) * (kx * coordinates[corrector, 1] +
                            ky * coordinates[corrector, 2])
            isfinite(phase) || throw(ArgumentError(
                "spatial sinusoidal phase calculation must be finite",
            ))
            sine, cosine = sincospi(phase)
            isfinite(sine) && isfinite(cosine) || throw(ArgumentError(
                "spatial sinusoidal samples must be finite",
            ))
            basis[sine_row, corrector] = sine
            basis[cosine_row, corrector] = cosine
            sampled_peak[sine_row] = max(sampled_peak[sine_row], abs(sine))
            sampled_peak[cosine_row] = max(sampled_peak[cosine_row], abs(cosine))
            sampled_rms[sine_row] += abs2(D(sine))
            sampled_rms[cosine_row] += abs2(D(cosine))
        end
    end
    positive_commands = similar(basis)
    @inbounds for mode in 1:modes
        peak = sampled_peak[mode]
        peak > method.minimum_sampled_peak || throw(ArgumentError(
            "sampled sinusoidal mode $mode is at or below minimum_sampled_peak",
        ))
        sampled_rms[mode] = sqrt(sampled_rms[mode] / correctors)
        isfinite(sampled_rms[mode]) || throw(ArgumentError(
            "spatial sinusoidal RMS calculation must be finite",
        ))
        divisor = method.normalization === :peak ? peak : one(T)
        for corrector in 1:correctors
            basis[mode, corrector] /= divisor
            positive_commands[mode, corrector] =
                amplitudes[mode] * basis[mode, corrector]
        end
    end
    modal_plan = prepare(ModalPushPull(positive_commands))
    return SpatialSinusoidalPushPullPlan(modal_plan, basis, sampled_peak,
        sampled_rms, method.normalization)
end

allocate_result(plan::SpatialSinusoidalPushPullPlan) = allocate_result(plan.modal_plan)
allocate_workspace(::SpatialSinusoidalPushPullPlan) = nothing

"""
    process!(result, nothing, plan::SpatialSinusoidalPushPullPlan, nothing)

Construct complete spatial push-pull commands in rows
`(+sine₁, −sine₁, +cosine₁, −cosine₁, +sine₂, …)` and columns in the
supplied actuator order, using the existing modal push-pull implementation.
With `:none`, each row's absolute peak is at most its declared amplitude;
with `:peak`, each row's absolute peak equals that amplitude. Output storage
must be one-based and must not alias prepared numerical arrays. The scalar CPU
implementation does not sequence equipment, acquire observations, choose
frequencies, or establish a zonal response from an incomplete modal basis.
"""
function process!(result::ProbeCommandBasisProduct{T}, ::Nothing,
                  plan::SpatialSinusoidalPushPullPlan{T}, ::Nothing) where {T}
    for prepared in (plan.basis, plan.modal_plan.positive_commands,
                     plan.sampled_peak, plan.sampled_rms)
        Base.mightalias(result.commands, prepared) && throw(ArgumentError(
            "spatial sinusoidal result must not alias prepared arrays",
        ))
    end
    return process!(result, nothing, plan.modal_plan, nothing)
end
