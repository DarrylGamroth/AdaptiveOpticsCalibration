"""
    OpticalGains

Modal optical-gain estimation from complete focal-plane images, synchronous
modal response estimation from complete dither windows, and bounded updates
of optical-gain corrections.
"""
module OpticalGains

using FFTW
using LinearAlgebra

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _any_cross_alias,
    _any_pair_alias,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("gain_sensing.jl")
include("modal_dither_response.jl")
include("bounded_update.jl")

export GainSensing, GainSensingSpecification, GainSensingInputs
export GainSensingPlan, GainSensingWorkspace, GainSensingProduct
export GainSensingFluxDiagnostics
export optical_gains, current_sensitivities, weak_mode_mask, flux_diagnostics
export ModalDitherResponse, ModalDitherResponseSpecification, ModalDitherResponseInputs
export ModalDitherResponsePlan, ModalDitherResponseProduct
export OpticalGainUpdate, OpticalGainUpdateSpecification, OpticalGainUpdateInputs
export OpticalGainUpdatePlan, OpticalGainUpdateProduct

end
