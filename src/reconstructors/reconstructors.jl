"""
    Reconstructors

Noise-weighted linear reconstructor preparation for adaptive-optics
calibration. The module distinguishes a command-prior covariance from a
corrector stroke metric because they are inverse parameterizations with
different scientific meanings.
"""
module Reconstructors

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

include("weighted.jl")
include("svd_inverse.jl")
include("modal.jl")

export ModalReconstructor, MeasurementNulling

export ReconstructorSpecification
export MAPReconstructor, MAPReconstructorInputs
export StrokeWeightedTikhonov, StrokeWeightedReconstructorInputs
export ReconstructorPlan, ReconstructorProduct, ReconstructorDiagnostics
export reconstructor, singular_values, tikhonov_gains
export AbstractSVDInverse, ExactPseudoInverse, TSVDInverse, TikhonovInverse
export SVDReconstructorInputs, SVDReconstructorPlan
export SVDReconstructorProduct, SVDReconstructorDiagnostics
export measurement_singular_vectors, corrector_singular_vectors
export spectral_gains, condition_number, effective_rank, truncation_count

end
