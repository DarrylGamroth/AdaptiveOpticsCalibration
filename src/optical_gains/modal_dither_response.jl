"""
    ModalDitherResponse(; command_amplitude_floor,
        relative_rank_tolerance=sqrt(eps(Float64)))

Estimate the synchronous modal response over one complete dither window.
Independently fit each command and response column to `D + C cos(θ) + S sin(θ)`
by least squares. The phase is `θ = 2π f (t - t₁)`, where the window starts
at `t₁` and `f` is the mode's declared frequency. The fitted command amplitude
must exceed `command_amplitude_floor`, a finite nonnegative threshold in the
supplied command coordinates. Use command coordinates with common units and a
meaningful common threshold across modes.

The smallest singular value of each three-column design must exceed the largest
singular value times `relative_rank_tolerance ∈ [√eps(Float64), 1)`.
Preparation and processing use Float64 arithmetic.
"""
struct ModalDitherResponse <: AbstractCalibrationMethod
    command_amplitude_floor::Float64
    relative_rank_tolerance::Float64

    function ModalDitherResponse(;
        command_amplitude_floor::Real,
        relative_rank_tolerance::Real=sqrt(eps(Float64)),
    )
        floor = Float64(command_amplitude_floor)
        tolerance = Float64(relative_rank_tolerance)
        isfinite(floor) && floor >= 0 || throw(ArgumentError(
            "command_amplitude_floor must be finite and nonnegative",
        ))
        sqrt(eps(Float64)) <= tolerance < 1 || throw(ArgumentError(
            "relative_rank_tolerance must be in [sqrt(eps(Float64)), 1)",
        ))
        return new(floor, tolerance)
    end
end

"""
    ModalDitherResponseSpecification(frequencies, frame_times; sample_rate)

Declare one carrier frequency in Hz per ordered command/response mode and one
time in seconds per frame of the complete window. Supply at least three finite
strictly increasing times on the regular grid declared by positive
`sample_rate` in Hz. Preparation allows rounding at the precision of each
timestamp, capped at 1/1024 of a frame interval, and uses the regular
grid originating at the first frame. Carriers must lie strictly between DC
and Nyquist.

Use one-based CPU arrays or views of CPU arrays. Coordinates may also be ranges.
Each mode is associated one-to-one with its carrier. The fit assumes the
component at that carrier in the reconstructed mode is due to the corresponding
command mode. Callers supply paired frames and account for cross-mode coupling,
coherent contamination, and timing association.
"""
struct ModalDitherResponseSpecification{F<:AbstractVector,T<:AbstractVector}
    frequencies::F
    frame_times::T
    sample_rate::Float64

    function ModalDitherResponseSpecification(
        frequencies::F,
        frame_times::T;
        sample_rate::Real,
    ) where {F<:AbstractVector{<:Real},T<:AbstractVector{<:Real}}
        return new{F,T}(frequencies, frame_times, Float64(sample_rate))
    end
end

"""
    ModalDitherResponseInputs(realized_commands, reconstructed_responses)

Bundle complete, paired frame × mode matrices in specification order. Commands
are realized modal amplitudes, and responses are raw reconstructed amplitudes
in the corresponding modal basis and sign convention. Both matrices use finite
real values and one-based CPU Array storage or views of that storage.
"""
struct ModalDitherResponseInputs{C<:AbstractMatrix{<:Real},R<:AbstractMatrix{<:Real}}
    realized_commands::C
    reconstructed_responses::R
end

"""
Prepared synchronous projection onto the cosine and negative sine coefficients.
The plan is reentrant with independent result and workspace storage.
"""
struct ModalDitherResponsePlan <: AbstractCalibrationPlan
    projection::Matrix{ComplexF64}
    command_amplitude_floor::Float64
end

"""
Caller-owned complete-window modal response and diagnostics in specification order.

The `command_amplitudes` and `response_amplitudes` are the magnitudes of the
fitted phasors `C - iS`. The complex `response_ratios` are response phasors
divided by command phasors. Their magnitudes are `gains` in response units per
command unit, with no calibration-reference normalization. The `phase_lags`
are the negative arguments of those ratios in radians in `[-π, π]`. A delayed
response has positive lag within this interval, matching the sign in
FilterGraphAlgorithms' synchronous optical-gain estimator.

`valid` is false when the command amplitude is at or below its declared floor
or the response ratio vanishes. Such modes retain their fitted amplitudes and
report zero ratio, gain, and phase lag.
"""
struct ModalDitherResponseProduct
    command_amplitudes::Vector{Float64}
    response_amplitudes::Vector{Float64}
    response_ratios::Vector{ComplexF64}
    gains::Vector{Float64}
    phase_lags::Vector{Float64}
    valid::Vector{Bool}
end

function _check_dither_times(frame_times, sample_rate)
    interval = inv(sample_rate)
    isfinite(interval) && interval > 0 || throw(ArgumentError(
        "sample_rate must have a finite positive frame interval",
    ))
    origin = first(frame_times)
    first_time = Float64(origin)
    origin_roundoff = origin isa AbstractFloat ? Float64(eps(abs(origin))) : 0.0
    previous = first_time
    for frame in 2:length(frame_times)
        stamp = frame_times[frame]
        time = Float64(stamp)
        expected = muladd(frame - 1, interval, first_time)
        scale = max(abs(first_time), abs(expected), interval)
        stamp_roundoff = stamp isa AbstractFloat ? Float64(eps(abs(stamp))) : 0.0
        tolerance = min(interval / 1024,
            max(origin_roundoff + stamp_roundoff, 32eps(Float64) * scale))
        isfinite(time) && isfinite(expected) && time > previous &&
            abs(time - expected) <= tolerance || throw(ArgumentError(
                "frame times must follow the declared regular grid",
            ))
        previous = time
    end
    return nothing
end

function prepare(method::ModalDitherResponse, specification::ModalDitherResponseSpecification)
    frequencies = specification.frequencies
    frame_times = specification.frame_times
    sample_rate = specification.sample_rate
    Base.require_one_based_indexing(frequencies, frame_times)
    for coordinates in (frequencies, frame_times)
        coordinates isa AbstractRange ||
            _require_gain_sensing_cpu_storage("modal dither coordinates", coordinates)
    end
    frame_count = length(frame_times)
    mode_count = length(frequencies)
    frame_count >= 3 || throw(ArgumentError("dither window must have at least three frames"))
    mode_count > 0 || throw(ArgumentError("at least one dither mode is required"))
    isfinite(sample_rate) && sample_rate > 0 ||
        throw(ArgumentError("sample_rate must be finite and positive"))
    _check_dither_times(frame_times, sample_rate)

    _checked_extent("modal dither projection", frame_count, mode_count)
    _checked_extent("modal dither design", frame_count, 3)
    projection = Matrix{ComplexF64}(undef, frame_count, mode_count)
    design = ones(Float64, frame_count, 3)
    for (mode, frequency) in pairs(frequencies)
        cycles_per_frame = Float64(frequency) / sample_rate
        0 < cycles_per_frame < 0.5 ||
            throw(ArgumentError("dither carrier $mode must lie strictly between DC and Nyquist"))
        for frame in axes(design, 1)
            sine, cosine = sincospi(2 * cycles_per_frame * (frame - 1))
            design[frame, 2] = cosine
            design[frame, 3] = sine
        end
        factorization = svd(design)
        singular_values = factorization.S
        last(singular_values) > method.relative_rank_tolerance * first(singular_values) ||
            throw(ArgumentError("dither carrier $mode is ill-conditioned over the window"))
        coefficients = factorization.V * Diagonal(inv.(singular_values)) * factorization.U'
        for frame in axes(projection, 1)
            projection[frame, mode] = complex(coefficients[2, frame], -coefficients[3, frame])
        end
    end
    return ModalDitherResponsePlan(projection, method.command_amplitude_floor)
end

function allocate_result(plan::ModalDitherResponsePlan)
    mode_count = size(plan.projection, 2)
    return ModalDitherResponseProduct(
        zeros(mode_count), zeros(mode_count), zeros(ComplexF64, mode_count),
        zeros(mode_count), zeros(mode_count), fill(false, mode_count),
    )
end

allocate_workspace(plan::ModalDitherResponsePlan) = allocate_result(plan)

_dither_arrays(product::ModalDitherResponseProduct) = (
    product.command_amplitudes, product.response_amplitudes, product.response_ratios,
    product.gains, product.phase_lags, product.valid,
)

function _dither_phasor(samples, projection, mode)
    # The fitted carrier coefficients are invariant to a constant. Subtracting
    # the first sample before projection avoids leaking a large offset through
    # the small numerical sum of the projection weights.
    origin = Float64(samples[1, mode])
    cosine = 0.0
    negative_sine = 0.0
    for frame in axes(projection, 1)
        value = Float64(samples[frame, mode]) - origin
        weight = projection[frame, mode]
        cosine = muladd(value, real(weight), cosine)
        negative_sine = muladd(value, imag(weight), negative_sine)
    end
    return complex(cosine, negative_sine)
end

function process!(
    product::ModalDitherResponseProduct,
    workspace::ModalDitherResponseProduct,
    plan::ModalDitherResponsePlan,
    inputs::ModalDitherResponseInputs,
)
    commands = inputs.realized_commands
    responses = inputs.reconstructed_responses
    projection = plan.projection
    Base.require_one_based_indexing(commands, responses)
    _require_gain_sensing_cpu_storage("realized commands", commands)
    _require_gain_sensing_cpu_storage("reconstructed responses", responses)
    size(commands) == size(responses) == size(projection) || throw(DimensionMismatch(
        "modal dither inputs must have the prepared frame × mode shape",
    ))
    mode_count = size(projection, 2)
    outputs = _dither_arrays(product)
    staged = _dither_arrays(workspace)
    buffers = (outputs..., staged...)
    all(buffer -> length(buffer) == mode_count, buffers) ||
        throw(DimensionMismatch("modal dither product and workspace must match the mode count"))
    (_any_pair_alias(buffers) || _any_cross_alias(buffers, (projection, commands, responses))) &&
        throw(ArgumentError("modal dither product and workspace must own disjoint storage"))

    for mode in axes(projection, 2)
        command = _dither_phasor(commands, projection, mode)
        response = _dither_phasor(responses, projection, mode)
        command_amplitude = abs(command)
        response_amplitude = abs(response)
        isfinite(command_amplitude) && isfinite(response_amplitude) ||
            throw(ArgumentError("modal dither samples and fitted amplitudes must be finite"))
        ratio = command_amplitude > plan.command_amplitude_floor ? response / command : 0.0im
        gain = abs(ratio)
        isfinite(gain) || throw(ArgumentError("modal dither response ratio must be finite"))
        valid = gain > 0
        workspace.command_amplitudes[mode] = command_amplitude
        workspace.response_amplitudes[mode] = response_amplitude
        workspace.response_ratios[mode] = ratio
        workspace.gains[mode] = gain
        workspace.phase_lags[mode] = valid ? -angle(ratio) : 0.0
        workspace.valid[mode] = valid
    end
    map(copyto!, outputs, staged)
    return product
end
