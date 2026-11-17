"""
    PhaseRetrieval

Reusable nonlinear focal-plane phase-retrieval calculations.  The module owns
the LiFT inverse iteration; a concrete [`AbstractLiFTForwardModel`](@ref)
provides the physical photon-rate model without making this package depend on
a plant simulator.
"""
module PhaseRetrieval

using LinearAlgebra

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _mightalias_any,
    _any_pair_alias,
    _any_cross_alias,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("types.jl")
include("solver.jl")
include("device_solver.jl")

export AbstractLiFTForwardModel
export LiFT, LiFTSpecification, LiFTInputs, LiFTPlan, LiFTWorkspace
export LiFTProduct, LiFTDiagnostics
export LiFTAnalyticJacobian, LiFTNumericalJacobian
export LiFTSolveAuto, LiFTSolveQR, LiFTSolveNormalEquations
export LiFTDampingNone, LiFTLevenbergMarquardt, LiFTAdaptiveLevenbergMarquardt
export LiFTPhotonRate, LiFTExpectedCounts, LiFTNormalizedIntensity
export LiFTInitialModelWeighting, LiFTIterativeModelWeighting
export LiFTReadNoiseWeighting, LiFTVarianceMapWeighting
export LiFTTotalRateMatching, LiFTPeakRateMatching, LiFTPhysicalRatePreservation
export lift_coefficients, lift_diagnostics

public coefficient_count, observation_axes, allocate_model_workspace
public allocate_photon_rate, predict_photon_rate!, analytic_photon_rate_jacobian!
public LiFTKernelPlan, LiFTKernelWorkspace, solve_lift_normal_device!

end
