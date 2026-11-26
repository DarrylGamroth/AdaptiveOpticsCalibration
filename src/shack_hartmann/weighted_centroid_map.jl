# Thomas et al. (2006) compare weighted centroid estimates. See docs/REFERENCES.md.
"""
    WeightedCentroidMap(; gaussian_fwhm=0)

Prepare dimensionless weights for a weighted center-of-gravity centroid.
For each active subaperture, optionally smooth its mean calibrated pixels,
clip negative values to zero, and divide by the peak clipped intensity.

`gaussian_fwhm` is a finite nonnegative full width at half maximum in pixels,
converted to Float64. Zero disables smoothing. For positive widths, use
zero-padded separable correlation with `g(δ) = exp(-4log(2)(δ/fwhm)²)` along
each axis over integer offsets `|δ| ≤ floor(axis_size/2)`. The symmetric
support centers the peak at zero lag on both even and odd dimensions. The
kernel peaks at one and the correlation is computed before clipping.
"""
struct WeightedCentroidMap <: AbstractCalibrationMethod
    gaussian_fwhm::Float64

    function WeightedCentroidMap(; gaussian_fwhm::Real=0)
        fwhm = Float64(gaussian_fwhm)
        isfinite(fwhm) && gaussian_fwhm >= 0 || throw(ArgumentError(
            "gaussian_fwhm must be finite and nonnegative",
        ))
        return new(fwhm)
    end
end

"""
    WeightedCentroidMapSpecification(rows, columns, subaperture_count)

Declare positive subaperture pixel dimensions and subaperture count, including
any inactive padding. Maps keep input row, column and subaperture order.
"""
struct WeightedCentroidMapSpecification
    rows::Int
    columns::Int
    subaperture_count::Int

    function WeightedCentroidMapSpecification(rows::Integer, columns::Integer,
                                               subaperture_count::Integer)
        rows > 0 || throw(ArgumentError("rows must be positive"))
        columns > 0 || throw(ArgumentError("columns must be positive"))
        subaperture_count > 0 || throw(ArgumentError("subaperture_count must be positive"))
        _checked_extent("weighted centroid map", rows, columns, subaperture_count)
        return new(Int(rows), Int(columns), Int(subaperture_count))
    end
end

"""Run-immutable map dimensions and optional separable Gaussian coefficients."""
struct WeightedCentroidMapPlan{K<:Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}} <:
       AbstractCalibrationPlan
    rows::Int
    columns::Int
    subaperture_count::Int
    kernels::K
end

function _gaussian_kernel(axis_size, fwhm)
    radius = axis_size ÷ 2
    kernel = Vector{Float64}(undef, radius + 1)
    kernel[1] = 1.0
    for offset in 1:radius
        kernel[offset + 1] = exp(-4 * log(2.0) * abs2(offset / fwhm))
    end
    return kernel
end

function prepare(method::WeightedCentroidMap, spec::WeightedCentroidMapSpecification)
    fwhm = method.gaussian_fwhm
    kernels = iszero(fwhm) ? nothing :
        (_gaussian_kernel(spec.rows, fwhm), _gaussian_kernel(spec.columns, fwhm))
    return WeightedCentroidMapPlan(spec.rows, spec.columns, spec.subaperture_count, kernels)
end

"""
    WeightedCentroidMapInputs(mean_pixels, active_count)

One-based CPU array of calibrated mean pixel intensities in `row × column ×
subaperture` order. The caller supplies the mean for each pixel in a common
intensity unit, with subapertures packed in caller order. Only the first
`active_count` subapertures are read. Every active pixel must be finite after
Float64 conversion for its map to be valid.
"""
struct WeightedCentroidMapInputs{P<:AbstractArray{<:Real,3}}
    mean_pixels::P
    active_count::Int
end

WeightedCentroidMapInputs(mean_pixels::P, active_count::Integer) where
    {P<:AbstractArray{<:Real,3}} = WeightedCentroidMapInputs{P}(mean_pixels, Int(active_count))

"""
    WeightedCentroidMapProduct

Float64 `weights` in input order, per-subaperture `peak` clipped intensities
in the input intensity unit, per-subaperture `finite` and `positive_peak`
flags, and supplied `active_count`. An active, finite subaperture with a
positive peak yields weights in `[0,1]` with maximum one. An active, finite
subaperture with no positive peak yields uniform unit weights and a zero
peak. An active subaperture with invalid input or nonfinite smoothing yields
zero weights, a NaN peak, and both flags false. Inactive maps and peaks are
zero and both flags are false.
"""
mutable struct WeightedCentroidMapProduct
    weights::Array{Float64,3}
    peak::Vector{Float64}
    finite::Vector{Bool}
    positive_peak::Vector{Bool}
    active_count::Int
end

"""Replaceable Float64 scratch for one subaperture's Gaussian correlation."""
struct WeightedCentroidMapWorkspace
    pixels::Matrix{Float64}
end

allocate_result(plan::WeightedCentroidMapPlan) = WeightedCentroidMapProduct(
    zeros(plan.rows, plan.columns, plan.subaperture_count), zeros(plan.subaperture_count),
    fill(false, plan.subaperture_count), fill(false, plan.subaperture_count), 0,
)

allocate_workspace(::WeightedCentroidMapPlan{Nothing}) = nothing
allocate_workspace(plan::WeightedCentroidMapPlan) =
    WeightedCentroidMapWorkspace(Matrix{Float64}(undef, plan.rows, plan.columns))

function _validate_centroid_storage(result, workspace, plan, mean_pixels)
    dimensions = (plan.rows, plan.columns, plan.subaperture_count)
    size(mean_pixels) == size(result.weights) == dimensions || throw(DimensionMismatch(
        "mean pixel and weight dimensions must match the plan",
    ))
    length(result.peak) == length(result.finite) == length(result.positive_peak) ==
        plan.subaperture_count || throw(DimensionMismatch(
        "weight diagnostics must have one entry per subaperture",
    ))
    outputs = (result.weights, result.peak, result.finite, result.positive_peak)
    if isnothing(plan.kernels)
        isnothing(workspace) || throw(ArgumentError("unsmoothed maps require no scratch"))
        inputs = (mean_pixels,)
    else
        isnothing(workspace) && throw(ArgumentError("Gaussian smoothing requires scratch"))
        size(workspace.pixels) == (plan.rows, plan.columns) || throw(DimensionMismatch(
            "Gaussian smoothing scratch dimensions must match the plan",
        ))
        outputs = (outputs..., workspace.pixels)
        inputs = (mean_pixels, plan.kernels...)
    end
    _any_cross_alias(outputs, inputs) && throw(ArgumentError(
        "weight product and scratch must not alias inputs or Gaussian coefficients",
    ))
    _any_pair_alias(outputs) && throw(ArgumentError("weight product and scratch must not alias"))
    return nothing
end

_smooth_spot!(weights, ::Nothing, ::Nothing) = nothing

function _smooth_spot!(weights, workspace::WeightedCentroidMapWorkspace, kernels)
    scratch = workspace.pixels
    row_kernel, column_kernel = kernels
    row_radius, column_radius = length(row_kernel) - 1, length(column_kernel) - 1
    rows, columns = size(weights)
    for column in 1:columns, row in 1:rows
        intensity = 0.0
        for source in max(1, row - row_radius):min(rows, row + row_radius)
            intensity += row_kernel[abs(row - source) + 1] * weights[source, column]
        end
        scratch[row, column] = intensity
    end
    for column in 1:columns, row in 1:rows
        intensity = 0.0
        for source in max(1, column - column_radius):min(columns, column + column_radius)
            intensity += column_kernel[abs(column - source) + 1] * scratch[row, source]
        end
        weights[row, column] = intensity
    end
    return nothing
end

function _spot_weights!(weights, workspace, kernels, pixels)
    for pixel in eachindex(weights, pixels)
        intensity = Float64(pixels[pixel])
        isfinite(intensity) || return NaN
        weights[pixel] = intensity
    end
    _smooth_spot!(weights, workspace, kernels)
    peak = 0.0
    for pixel in eachindex(weights)
        intensity = weights[pixel]
        isfinite(intensity) || return NaN
        clipped = max(0.0, intensity)
        weights[pixel] = clipped
        peak = max(peak, clipped)
    end
    if iszero(peak)
        fill!(weights, 1.0)
    else
        weights ./= peak
    end
    return peak
end

function process!(result::WeightedCentroidMapProduct,
                  workspace::Union{Nothing,WeightedCentroidMapWorkspace},
                  plan::WeightedCentroidMapPlan, inputs::WeightedCentroidMapInputs)
    mean_pixels, count = inputs.mean_pixels, inputs.active_count
    _require_cpu_arrays(mean_pixels)
    _validate_active_count(count, plan.subaperture_count)
    _validate_centroid_storage(result, workspace, plan, mean_pixels)

    result.active_count = count
    for subaperture in 1:plan.subaperture_count
        weights = view(result.weights, :, :, subaperture)
        active = subaperture <= count
        peak = active ? _spot_weights!(weights, workspace, plan.kernels,
            view(mean_pixels, :, :, subaperture)) : 0.0
        finite = active && isfinite(peak)
        finite || fill!(weights, 0.0)
        result.peak[subaperture] = peak
        result.finite[subaperture] = finite
        result.positive_peak[subaperture] = finite && peak > 0
    end
    return result
end
