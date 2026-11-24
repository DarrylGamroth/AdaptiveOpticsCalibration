"""
    AtmosphericParameters

Fried parameter and turbulence outer-scale estimation from explicitly
calibrated, ordered optical-path-difference modal variances. Templates and
observations use one-based Julia CPU arrays, ranges, or views and other
wrappers backed by these.
"""
module AtmosphericParameters

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod, AbstractCalibrationPlan, allocate_result,
    allocate_workspace, prepare, process!, _require_cpu_arrays

export ModalVarianceFit, ModalVarianceSpecification, ModalVarianceInputs
export ModalVariancePlan, ModalVarianceProduct

include("modal_variances.jl")

end
