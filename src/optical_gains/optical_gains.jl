"""
    OpticalGains

Complete-image modal optical-gain estimation from explicit focal-plane frames.
This module does not form physical modulation images, model a camera, retain
history, or apply gains to a real-time controller.
"""
module OpticalGains

using FFTW
using LinearAlgebra

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("gain_sensing.jl")

export GainSensing, GainSensingSpecification, GainSensingInputs
export GainSensingPlan, GainSensingWorkspace, GainSensingProduct
export GainSensingFluxDiagnostics
export optical_gains, current_sensitivities, weak_mode_mask, flux_diagnostics

end
