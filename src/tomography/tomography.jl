"""
    Tomography

Cold preparation of explicit-array tomographic covariance products. This
module accepts already ordered, consistently scaled numerical arrays; it does
not derive physical geometry, convert wavelengths, reorder slopes, fit a DM,
or apply a reconstructor to per-frame measurements.
"""
module Tomography

using LinearAlgebra
using SpecialFunctions: besselk, gamma

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("covariance_reconstructor.jl")
include("von_karman_covariance.jl")

export CovarianceReconstructor
export CovarianceReconstructorSpecification, CovarianceReconstructorInputs
export CovarianceReconstructorPlan, CovarianceReconstructorProduct
export CovarianceReconstructorDiagnostics, reconstructor
export VonKarmanAutoCovariance, VonKarmanAutoCovarianceSpecification
export VonKarmanAutoCovariancePlan, VonKarmanAutoCovarianceProduct
export VonKarmanCrossCovariance, VonKarmanCrossCovarianceSpecification
export VonKarmanCrossCovariancePlan, VonKarmanCrossCovarianceProduct
export auto_covariance, cross_covariance

end
