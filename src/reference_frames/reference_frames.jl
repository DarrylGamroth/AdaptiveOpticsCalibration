"""
    ReferenceFrames

Complete-batch and explicit-state detector dark moments, lamp centroid
references and normalized pyramid references. These scalar CPU algorithms
consume explicit one-based arrays and return numerical products and diagnostics.
Acquisition, exposure association, acceptance, persistence and activation belong
to the caller.
"""
module ReferenceFrames

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _checked_extent,
    _any_cross_alias,
    _any_pair_alias,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

export ReferenceStatus, Valid, Invalid, NoEligible
export DarkFrameMoments, DarkFrameSpecification, DarkFramePlan
export DarkFrameProduct, DarkFrameWorkspace, DarkFrameState
export LampReference, LampReferenceSpecification, LampReferencePlan
export LampReferenceInputs, LampReferenceProduct, LampReferenceWorkspace, LampReferenceState
export PyramidReference, PyramidReferenceSpecification, PyramidReferencePlan
export PyramidReferenceInputs, PyramidReferenceProduct, PyramidReferenceWorkspace, PyramidReferenceState

"""Numerical outcome of a complete reference-frame batch."""
@enum ReferenceStatus::UInt8 begin
    Valid
    Invalid
    NoEligible
end

@doc "The batch produced a finite numerical reference product." Valid
@doc "A numerical failure, nonpositive pyramid mean or ADC rail violation invalidated the product." Invalid
@doc "A finite lamp batch below the ADC rail has no eligible ROI." NoEligible

include("dark_frame_moments.jl")
include("lamp_reference.jl")
include("pyramid_reference.jl")

end
