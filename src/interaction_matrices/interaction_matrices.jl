"""
    InteractionMatrices

Complete-batch construction and estimation of adaptive-optics interaction
matrices from explicitly associated calibration observations.

This module does not acquire observations, sequence probes, or activate
calibration artifacts.
"""
module InteractionMatrices

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

include("structured.jl")
include("central_difference.jl")

export CentralDifference

export ZonalPushPull, HadamardPushPull, ResponseSpecification
export ZonalPushPullPlan, HadamardPushPullPlan
export InteractionMatrixProduct, StructuredWorkspace, interaction_matrix
export StructuredInteractionMatrixStatus
export StructuredNotProcessed, StructuredValid
export StructuredNonFiniteResponse, StructuredNonFiniteCalculation

end
