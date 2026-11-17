"""
    ProbeBases

Pure construction of complete adaptive-optics calibration probe-command
sequences. Equipment sequencing and exposure association remain caller-owned.
"""
module ProbeBases

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    _float_type,
    _hadamard_order,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("push_pull.jl")
include("spatial_sinusoidal.jl")

export ModalPushPull, ZonalPushPull, HadamardPushPull
export ModalPushPullPlan, ZonalPushPullPlan, HadamardPushPullPlan
export ProbeCommandBasisProduct, probe_commands
export SpatialSinusoidalPushPull, SpatialSinusoidalSpecification
export SpatialSinusoidalPushPullPlan

end
