"""
    Diagnostics

Package-neutral numerical qualification of adaptive-optics calibration
products. Acceptance and activation policy remain caller-owned.
"""
module Diagnostics

using LinearAlgebra: norm

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    _any_cross_alias,
    _any_pair_alias,
    allocate_result,
    allocate_workspace,
    prepare,
    process,
    process!

include("interaction_matrix_stability.jl")
include("repeated_response_moments.jl")
include("held_out.jl")

export HeldOutReconstruction, PupilResidualRatio

export InteractionMatrixStability, InteractionMatrixPair
export InteractionMatrixStabilitySpecification
export InteractionMatrixStabilityPlan, InteractionMatrixStabilityProduct
export StabilityStatus, Stable, InsufficientColumnNorm
export ExcessiveRelativeColumnDifference, ExcessiveRelativeFrobeniusDifference
export NonFiniteStabilityCalculation
export RepeatedResponseMoments, RepeatedResponseSpecification
export RepeatedResponseMomentsPlan, RepeatedResponseMomentsProduct
export RepeatedResponseMomentsWorkspace, ResponseMomentsStatus
export ResponseMomentsNotProcessed, ResponseMomentsValid
export ResponseMomentsNonFiniteInput, ResponseMomentsNonFiniteCalculation

end
