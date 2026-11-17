"""
    DOCRIME

Complete-batch covariance-to-interaction-matrix estimation for open- and
closed-loop DO-CRIME system identification. Covariance accumulation and
per-frame excitation belong to the calling workflow or RTC implementation.
"""
module DOCRIME

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("interaction_matrix.jl")

export ConditioningCriteria, CovarianceSpecification, CovarianceSums
export OpenLoop, ClosedLoop, InteractionMatrixPlan
export InteractionMatrixStatus, Valid, InsufficientSamples
export InsufficientCorrectorVariance, ExcessiveCorrectorCorrelation
export NonPositiveDefinite, InsufficientFactorizationPivot
export InteractionMatrixFailure, NonFiniteCrossCovariance
export NonFiniteCorrectorCovariance, NonSymmetricCorrectorCovariance
export SampleCountNotRepresentable, SampleCountOutOfRange
export NonFiniteNormalization, NonFiniteCorrelation
export NonFiniteFactorization, NonFiniteSolve
export InteractionMatrixDiagnostics, InteractionMatrixProduct
export InteractionMatrixWorkspace, interaction_matrix

end
