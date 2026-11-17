module AdaptiveOpticsCalibrationKernelAbstractionsExt

using AdaptiveOpticsCalibration
using FFTW
using LinearAlgebra

import KernelAbstractions

const AOC = AdaptiveOpticsCalibration
const Diagnostics = AOC.Diagnostics
const InteractionMatrices = AOC.InteractionMatrices
const OpticalGains = AOC.OpticalGains
const PhaseRetrieval = AOC.PhaseRetrieval
const ProbeBases = AOC.ProbeBases
const Tomography = AOC.Tomography

include("AdaptiveOpticsCalibrationKernelAbstractionsExt/backend_storage.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/structured_interaction_matrices.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/probe_bases.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/interaction_matrix_stability.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/phase_retrieval.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/gain_sensing.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/von_karman_covariance.jl")
include("AdaptiveOpticsCalibrationKernelAbstractionsExt/covariance_reconstructor.jl")

end
