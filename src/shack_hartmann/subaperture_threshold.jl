"""
    SubapertureThreshold(fraction)

Compute the absolute pixel threshold `fraction * median(peak_pixels[1:N])`
for `N` active subapertures. `fraction` is finite and nonnegative after
Float64 conversion. The threshold has the units of the calibrated peaks.
"""
struct SubapertureThreshold <: AbstractCalibrationMethod
    fraction::Float64

    function SubapertureThreshold(fraction::Real)
        value = Float64(fraction)
        isfinite(value) && fraction >= 0 || throw(ArgumentError(
            "fraction must be finite and nonnegative",
        ))
        return new(value)
    end
end

"""
    SubapertureThresholdSpecification(subaperture_count)

Declare the positive length of the peak vector, including any inactive padding.
"""
struct SubapertureThresholdSpecification
    subaperture_count::Int

    function SubapertureThresholdSpecification(subaperture_count::Integer)
        subaperture_count > 0 || throw(ArgumentError("subaperture_count must be positive"))
        return new(Int(subaperture_count))
    end
end

"""Prepared threshold fraction and peak vector length."""
struct SubapertureThresholdPlan <: AbstractCalibrationPlan
    fraction::Float64
    subaperture_count::Int
end

prepare(method::SubapertureThreshold, spec::SubapertureThresholdSpecification) =
    SubapertureThresholdPlan(method.fraction, spec.subaperture_count)

"""
    SubapertureThresholdInputs(peak_pixels, active_count)

One-based CPU vector of peak calibrated pixel intensities, with one maximum
per subaperture and a common intensity unit. The first `active_count` entries
are the active subapertures in caller order. Remaining entries are ignored.
Active peaks must be finite after Float64 conversion. Signed calibrated
intensities are allowed.
"""
struct SubapertureThresholdInputs{P<:AbstractVector{<:Real}}
    peak_pixels::P
    active_count::Int
end

SubapertureThresholdInputs(peak_pixels::P, active_count::Integer) where
    {P<:AbstractVector{<:Real}} = SubapertureThresholdInputs{P}(peak_pixels, Int(active_count))

"""
    SubapertureThresholdProduct

Float64 `threshold` and `median_peak`, supplied `active_count`, and numerical
validity flag `valid`. A zero active count or a nonfinite active peak yields
NaN for both scalars. An overflowing threshold yields NaN for `threshold`
and preserves the finite `median_peak`. Both outcomes set `valid` to false.
"""
mutable struct SubapertureThresholdProduct
    threshold::Float64
    median_peak::Float64
    active_count::Int
    valid::Bool
end

"""Replaceable Float64 scratch for selection of the active peak median."""
struct SubapertureThresholdWorkspace
    peak_pixels::Vector{Float64}
end

allocate_result(::SubapertureThresholdPlan) =
    SubapertureThresholdProduct(NaN, NaN, 0, false)

allocate_workspace(plan::SubapertureThresholdPlan) =
    SubapertureThresholdWorkspace(Vector{Float64}(undef, plan.subaperture_count))

function process!(result::SubapertureThresholdProduct, workspace::SubapertureThresholdWorkspace,
                  plan::SubapertureThresholdPlan, inputs::SubapertureThresholdInputs)
    peak_pixels, count = inputs.peak_pixels, inputs.active_count
    _require_cpu_arrays(peak_pixels)
    length(peak_pixels) == length(workspace.peak_pixels) == plan.subaperture_count || throw(
        DimensionMismatch("peak vector and scratch lengths must match the plan"),
    )
    _validate_active_count(count, plan.subaperture_count)
    Base.mightalias(workspace.peak_pixels, peak_pixels) && throw(ArgumentError(
        "threshold scratch must not alias input peaks",
    ))

    result.active_count = count
    result.valid = false
    result.threshold = NaN
    result.median_peak = NaN
    count == 0 && return result

    for subaperture in 1:count
        peak = Float64(peak_pixels[subaperture])
        isfinite(peak) || return result
        workspace.peak_pixels[subaperture] = peak
    end
    peaks = view(workspace.peak_pixels, 1:count)
    lower, upper = cld(count, 2), count ÷ 2 + 1
    sort!(peaks; alg=Base.Sort.PartialQuickSort(lower:upper))
    low, high = peaks[lower], peaks[upper]
    total = low + high
    result.median_peak = isfinite(total) ? total / 2 : low / 2 + high / 2
    threshold = plan.fraction * result.median_peak
    if isfinite(threshold)
        result.threshold = threshold
        result.valid = true
    end
    return result
end
