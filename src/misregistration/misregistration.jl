"""
    Misregistration

Complete-data meta-sensitivity estimation from explicit response matrices.
This numerical module does not acquire physical responses or interpret its
ordered offsets as physical registrations.
"""
module Misregistration

using LinearAlgebra

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process,
    process!
import ..Reconstructors

include("meta_sensitivity_estimate.jl")

export MetaSensitivityEstimate, MetaSensitivityEstimateSpecification
export MetaSensitivityEstimateInputs, MetaSensitivityEstimatePlan
export MetaSensitivityEstimateProduct, MetaSensitivityEstimateWorkspace
export MetaSensitivityEstimateDiagnostics
export parameter_offsets, condition_number, effective_rank

end
