"""
    ShackHartmann

Calibration products for Shack–Hartmann wavefront sensors. The caller supplies
calibrated subaperture measurements in deployment order. Inputs use one-based
Julia CPU arrays, ranges or wrappers backed by these, including views.
"""
module ShackHartmann

import ..AdaptiveOpticsCalibration: AbstractCalibrationMethod, AbstractCalibrationPlan,
    prepare, allocate_result, allocate_workspace, process!, _checked_extent,
    _any_cross_alias, _any_pair_alias, _require_cpu_arrays

export SubapertureThreshold, SubapertureThresholdSpecification, SubapertureThresholdPlan
export SubapertureThresholdInputs, SubapertureThresholdProduct, SubapertureThresholdWorkspace
export WeightedCentroidMap, WeightedCentroidMapSpecification, WeightedCentroidMapPlan
export WeightedCentroidMapInputs, WeightedCentroidMapProduct, WeightedCentroidMapWorkspace

function _validate_active_count(active_count, subaperture_count)
    0 <= active_count <= subaperture_count || throw(ArgumentError(
        "active_count must be between zero and the subaperture count",
    ))
    return nothing
end

include("subaperture_threshold.jl")
include("weighted_centroid_map.jl")

end
