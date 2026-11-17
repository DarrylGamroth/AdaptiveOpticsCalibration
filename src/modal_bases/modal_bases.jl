"""
    ModalBases

Explicit-array construction of sampled modal bases and pure modal fitting
diagnostics. This module neither extracts physical influence functions nor
places an OPD product in an optical plant.
"""
module ModalBases

using LinearAlgebra

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    _mightalias_any,
    _any_pair_alias,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

include("influence_function_eigenbasis.jl")
include("karhunen_loeve_basis.jl")
include("modal_fitting.jl")
include("modal_opd_expansion.jl")

export AbstractModalBasisMethod, InfluenceFunctionEigenbasis
export SampledInfluenceBasisSpecification, InfluenceFunctionEigenbasisPlan
export SampledInfluenceEigenbasisProduct, SampledInfluenceEigenbasisWorkspace
export gram_eigenvalues, modal_to_command, sampled_modes
export KarhunenLoeveBasis, KarhunenLoeveBasisSpecification
export KarhunenLoeveBasisInputs, KarhunenLoeveBasisPlan
export KarhunenLoeveBasisProduct, KarhunenLoeveBasisWorkspace
export atmospheric_variances
export ModalFitting, ModalFittingSpecification, ModalFittingInputs
export ModalFittingPlan, ModalFittingProduct, ModalFittingWorkspace
export input_opd, fitted_opd, residual_opd
export ModalOPDExpansion, ModalOPDExpansionSpecification, ModalOPDExpansionInputs
export ModalOPDExpansionPlan, ModalOPDExpansionProduct, opd

end
