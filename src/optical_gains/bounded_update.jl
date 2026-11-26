# Package-defined gain limits and acceptance follow a supplied temporal probe. See docs/REFERENCES.md.
"""
    OpticalGainUpdate(; integrator_gain, response_floor, reference_floor, phase_tolerance)

Update the dimensionless modal optical-gain correction from one complete dither
response. For a measured complex response ratio `R` and a path-specific reference
ratio `R₀`, the observed optical gain is the in-phase component
`h = real(R / R₀)`. Given the previous positive correction `c`, the update is
`clamp((1 - α)c + α/h, c_min, c_max)` with integrator gain `0 < α ≤ 1`.
The updated multiplier increases when the corrected in-phase response `c*h`
is below unity and decreases when it is above unity, unless limited by a bound.

`reference_floor` is a nonnegative lower bound on the magnitude of `R₀` in
response units per command unit. `response_floor` is a nonnegative lower bound
on the dimensionless `h`. Both limits are exclusive. `phase_tolerance` is the
maximum absolute residual phase in radians, in `[0, π/2)`. A mode holds its
previous correction when the dither fit is invalid, the normalized ratio is
nonfinite, its response is at or below the floor, its residual phase exceeds
the tolerance, or its inverse gain cannot be represented in Float64.
"""
struct OpticalGainUpdate <: AbstractCalibrationMethod
    integrator_gain::Float64
    response_floor::Float64
    reference_floor::Float64
    phase_tolerance::Float64

    function OpticalGainUpdate(;
        integrator_gain::Real,
        response_floor::Real,
        reference_floor::Real,
        phase_tolerance::Real,
    )
        α = Float64(integrator_gain)
        response_floor = Float64(response_floor)
        reference_floor = Float64(reference_floor)
        phase_tolerance = Float64(phase_tolerance)
        0 < α <= 1 || throw(ArgumentError("integrator_gain must be in (0, 1]"))
        isfinite(response_floor) && response_floor >= 0 ||
            throw(ArgumentError("response_floor must be finite and nonnegative"))
        isfinite(reference_floor) && reference_floor >= 0 ||
            throw(ArgumentError("reference_floor must be finite and nonnegative"))
        0 <= phase_tolerance < π / 2 ||
            throw(ArgumentError("phase_tolerance must be in [0, π/2) radians"))
        return new(α, response_floor, reference_floor, phase_tolerance)
    end
end

"""
    OpticalGainUpdateSpecification(reference_ratios, correction_min, correction_max)

Declare one complex calibration-reference ratio and two positive, finite,
inclusive correction bounds per dither mode. The reference and observations
must use the same modal order, units, sign, and exposure association. Common-path
dither uses the realized DM carrier. Non-common-path dither uses the calibrated
realized WFS-side carrier. Each injection path requires its own reference
ratios, including the phase of the reference transfer. Use one-based CPU
arrays, views of CPU arrays, or ranges. Preparation copies the vectors.
"""
struct OpticalGainUpdateSpecification{
    R<:AbstractVector{<:Number},L<:AbstractVector{<:Real},U<:AbstractVector{<:Real},
}
    reference_ratios::R
    correction_min::L
    correction_max::U
end

"""
    OpticalGainUpdateInputs(response_ratios, valid, correction)
    OpticalGainUpdateInputs(dither, correction)

Supply the raw complex response/command ratios and Boolean dither validity in
specification order, or pass a `ModalDitherResponseProduct` directly. Ratios
use reconstructed response units per realized probe unit. With the phasor
convention `C - iS`, the negative argument of the ratio is the phase lag in
radians, modulo `2π`. Use the same convention for the reference. The supplied
command and response must use their raw (unscaled) values.

Supply the previous dimensionless correction explicitly, including on the
first call. Each correction must be within its prepared bounds. Use one-based
CPU arrays or views of those arrays. Validity can also be a `BitVector` and
the correction can also be a range.
"""
struct OpticalGainUpdateInputs{
    R<:AbstractVector{<:Complex},V<:AbstractVector{Bool},C<:AbstractVector{<:Real},
}
    response_ratios::R
    valid::V
    correction::C
end

OpticalGainUpdateInputs(dither::ModalDitherResponseProduct, correction) =
    OpticalGainUpdateInputs(dither.response_ratios, dither.valid, correction)

"""Prepared reference ratios and bounded optical-gain correction update."""
struct OpticalGainUpdatePlan <: AbstractCalibrationPlan
    method::OpticalGainUpdate
    reference_ratios::Vector{ComplexF64}
    correction_min::Vector{Float64}
    correction_max::Vector{Float64}
end

"""
Per-mode updated `correction`, observed signed `optical_gains`, and `valid`
flags in specification order. The gains are zero when the fitted ratio is invalid
or normalization is nonfinite. A finite signed gain is retained when the
response or phase fails an update criterion. A valid recommendation can coincide
with a correction bound or the previous correction.
"""
struct OpticalGainUpdateProduct
    correction::Vector{Float64}
    optical_gains::Vector{Float64}
    valid::Vector{Bool}
end

function prepare(method::OpticalGainUpdate, specification::OpticalGainUpdateSpecification)
    reference = specification.reference_ratios
    lower = specification.correction_min
    upper = specification.correction_max
    Base.require_one_based_indexing(reference, lower, upper)
    for vector in (reference, lower, upper)
        vector isa AbstractRange ||
            _require_gain_sensing_cpu_storage("optical gain update specification", vector)
    end
    length(reference) == length(lower) == length(upper) ||
        throw(DimensionMismatch("reference ratios and correction bounds must match the mode count"))
    isempty(reference) && throw(ArgumentError("at least one dither mode is required"))

    reference_ratios = collect(ComplexF64, reference)
    correction_min = collect(Float64, lower)
    correction_max = collect(Float64, upper)
    for mode in eachindex(reference_ratios)
        amplitude = abs(reference_ratios[mode])
        isfinite(amplitude) && amplitude > method.reference_floor || throw(ArgumentError(
            "reference ratio $mode must have finite amplitude above reference_floor",
        ))
        0 < correction_min[mode] <= correction_max[mode] < Inf || throw(ArgumentError(
            "correction bounds for mode $mode must be positive, ordered, and finite",
        ))
    end
    return OpticalGainUpdatePlan(method, reference_ratios, correction_min, correction_max)
end

function allocate_result(plan::OpticalGainUpdatePlan)
    count = length(plan.reference_ratios)
    return OpticalGainUpdateProduct(zeros(count), zeros(count), fill(false, count))
end

allocate_workspace(::OpticalGainUpdatePlan) = nothing

function process!(
    product::OpticalGainUpdateProduct,
    ::Nothing,
    plan::OpticalGainUpdatePlan,
    inputs::OpticalGainUpdateInputs,
)
    correction = inputs.correction
    ratios = inputs.response_ratios
    dither_valid = inputs.valid
    Base.require_one_based_indexing(ratios, dither_valid, correction)
    _require_gain_sensing_cpu_storage("response ratios", ratios)
    dither_valid isa BitVector || _require_gain_sensing_cpu_storage("dither validity", dither_valid)
    correction isa AbstractRange || _require_gain_sensing_cpu_storage("correction", correction)
    outputs = (product.correction, product.optical_gains, product.valid)
    references = (plan.reference_ratios, plan.correction_min, plan.correction_max)
    observations = (ratios, dither_valid, correction)
    mode_count = length(plan.reference_ratios)
    all(v -> length(v) == mode_count, (outputs..., observations...)) ||
        throw(DimensionMismatch("optical gain update vectors must match the prepared mode count"))
    (_any_pair_alias(outputs) ||
        _any_cross_alias(outputs, (references..., observations...))) &&
        throw(ArgumentError("optical gain update products must own disjoint storage"))

    # Check the full explicit state before writing any product entries.
    for mode in eachindex(plan.reference_ratios)
        plan.correction_min[mode] <= Float64(correction[mode]) <= plan.correction_max[mode] ||
            throw(ArgumentError("previous correction for mode $mode is outside the prepared bounds"))
    end

    method = plan.method
    α = method.integrator_gain
    for mode in eachindex(plan.reference_ratios)
        previous = Float64(correction[mode])
        ratio = dither_valid[mode] ?
                ComplexF64(ratios[mode]) / plan.reference_ratios[mode] : 0.0im
        gain = isfinite(ratio) ? real(ratio) : 0.0
        target = gain > method.response_floor ? inv(gain) : Inf
        valid = isfinite(target) && abs(angle(ratio)) <= method.phase_tolerance
        product.optical_gains[mode] = gain
        product.valid[mode] = valid
        product.correction[mode] = valid ? clamp(muladd(α, target, (1 - α) * previous),
            plan.correction_min[mode], plan.correction_max[mode]) : previous
    end
    return product
end
