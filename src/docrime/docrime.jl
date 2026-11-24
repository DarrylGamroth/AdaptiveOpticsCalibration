"""
    DOCRIME

Centered covariance reduction and covariance-to-interaction-matrix estimation
for open- and closed-loop DO-CRIME system identification. The caller supplies
exposure-associated samples and chooses when to freeze a covariance window.
"""
module DOCRIME

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _any_cross_alias,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("interaction_matrix.jl")
include("covariance.jl")

export ConditioningCriteria, CovarianceSpecification, CovarianceSums
export CenteredCovariance, CovariancePlan, CovarianceSample, CovarianceState, CovarianceWorkspace
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
