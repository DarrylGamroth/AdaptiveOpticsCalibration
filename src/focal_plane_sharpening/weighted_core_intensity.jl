"""
    WeightedCoreIntensity(radius)

Negative weighted core sum of the replicated-edge centered 3×3 median image.
The center is its first column-major maximum. Include pixels at strict distance
`r < radius` with weight `1/(1+r)^2`. Radius is explicit detector pixels;
intensities retain their supplied units, sign, background, and scale.
"""
struct WeightedCoreIntensity <: AbstractCalibrationMethod
    radius::Float64
    WeightedCoreIntensity(radius::Real) = new(_positive(radius, "radius"))
end

"""Declare positive ROI `(rows, columns)` dimensions and one-based indexing."""
struct ImageSpecification
    rows::Int
    columns::Int
    function ImageSpecification(rows::Integer, columns::Integer)
        r, c = Int(rows), Int(columns)
        r > 0 && c > 0 || throw(ArgumentError("image dimensions must be positive"))
        _checked_extent("sharpening image", r, c)
        return new(r, c)
    end
end

"""Immutable ROI dimensions and strict aperture radius in pixels."""
struct WeightedCorePlan <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    radius::Float64
end
prepare(method::WeightedCoreIntensity, spec::ImageSpecification) =
    WeightedCorePlan(spec.rows, spec.columns, method.radius)

"""
Float64 `objective`, filtered `peak_value`, `center::CartesianIndex{2}`,
`aperture_count`, and `cropped` (a discrete aperture pixel lies outside the ROI).
`NumericalFailure` zeroes objective and diagnostics. Finite signed or zero
images can be `Valid`; this is no assertion of physical image quality.
"""
mutable struct WeightedCoreProduct
    status::SharpeningStatus
    objective::Float64
    center::CartesianIndex{2}
    peak_value::Float64
    aperture_count::Int
    cropped::Bool
end

"""Replaceable Float64 filtered image scratch; overwritten each invocation."""
struct WeightedCoreWorkspace
    filtered::Matrix{Float64}
end
allocate_result(::WeightedCorePlan) = WeightedCoreProduct(
    NumericalFailure, 0.0, CartesianIndex(0, 0), 0.0, 0, false)
allocate_workspace(plan::WeightedCorePlan) = WeightedCoreWorkspace(
    zeros(plan.rows, plan.columns))

# A fixed insertion sorting network on nine scalar values. No image library,
# variable-sized window, or extra dependency is needed for this exact median.
function _median9(values::NTuple{9,Float64})
    sorted = values
    for i in 2:9
        for j in (i - 1):-1:1
            a, b = sorted[j], sorted[j + 1]
            sorted = Base.setindex(sorted, min(a, b), j)
            sorted = Base.setindex(sorted, max(a, b), j + 1)
        end
    end
    return sorted[5]
end

function process!(result::WeightedCoreProduct, workspace::WeightedCoreWorkspace,
                  plan::WeightedCorePlan, image::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(image, workspace.filtered)
    size(image) == (plan.rows, plan.columns) || throw(DimensionMismatch("image shape"))
    size(workspace.filtered) == size(image) || throw(DimensionMismatch("filtered shape"))
    _noalias((workspace.filtered,), (image,))
    all(x -> isfinite(x) && isfinite(Float64(x)), image) ||
        throw(ArgumentError("image must be finite and representable in Float64"))
    f = workspace.filtered
    @inbounds for j in 1:plan.columns, i in 1:plan.rows
        r0, r1 = max(i - 1, 1), min(i + 1, plan.rows)
        c0, c1 = max(j - 1, 1), min(j + 1, plan.columns)
        f[i, j] = _median9((Float64(image[r0, c0]), Float64(image[i, c0]),
            Float64(image[r1, c0]), Float64(image[r0, j]), Float64(image[i, j]),
            Float64(image[r1, j]), Float64(image[r0, c1]), Float64(image[i, c1]),
            Float64(image[r1, c1])))
    end
    center = CartesianIndex(1, 1)
    peak = f[1, 1]
    @inbounds for j in 1:plan.columns, i in 1:plan.rows
        if f[i, j] > peak
            peak = f[i, j]
            center = CartesianIndex(i, j)
        end
    end
    total, count = 0.0, 0
    @inbounds for j in 1:plan.columns, i in 1:plan.rows
        distance = hypot(Float64(i - center[1]), Float64(j - center[2]))
        if distance < plan.radius
            total += f[i, j] / (1.0 + distance)^2
            count += 1
        end
    end
    if isfinite(total)
        result.status = Valid
        result.objective = -total
        result.center = center
        result.peak_value = peak
        result.aperture_count = count
        outside_distance = min(center[1], center[2],
            plan.rows - center[1] + 1, plan.columns - center[2] + 1)
        result.cropped = outside_distance < plan.radius
    else
        result.status = NumericalFailure
        result.objective = 0.0
        result.center = CartesianIndex(0, 0)
        result.peak_value = 0.0
        result.aperture_count = 0
        result.cropped = false
    end
    return result
end
