"""
Pure scalar CPU image objectives, modal scans, supplied SPSA probe pairs and
gradients, and explicit-state Adam steps. One-based ordinary arrays and views
are supported. The caller owns coefficient order, homogeneous units,
measurement association, equipment, acceptance, and optimizer history.
"""
module FocalPlaneSharpening

using LinearAlgebra: Diagonal, svd

import ..AdaptiveOpticsCalibration: AbstractCalibrationMethod,
    AbstractCalibrationPlan, _checked_extent, prepare, process!,
    allocate_result, allocate_workspace

export SharpeningStatus, Valid, NumericalFailure, LostPerturbation
export UnresolvedCurvature, NonconcaveFit, UnbracketedPeak
export WeightedCoreIntensity, ImageSpecification, WeightedCorePlan
export WeightedCoreProduct, WeightedCoreWorkspace
export ModalScan, ModalScanSpecification, ModalScanPlan, ModalScanProduct
export CoefficientSpecification, SPSAProbes, SPSAProbePlan, SPSAProbeInputs
export SPSAProbeProduct, SPSAGradient, SPSAGradientPlan, SPSAGradientInputs
export SPSAGradientProduct, AdamStep, AdamPlan, AdamState, AdamInputs
export AdamProduct, AdamWorkspace

"""Numerical outcome; structural and nonfinite input errors throw before writes."""
@enum SharpeningStatus::UInt8 begin
    Valid
    NumericalFailure
    LostPerturbation
    UnresolvedCurvature
    NonconcaveFit
    UnbracketedPeak
end
@doc "A complete finite numerical product; acceptance remains caller-owned." Valid
@doc "Finite inputs overflowed the Float64 numerical calculation. See the product for available diagnostics." NumericalFailure
@doc "An SPSA probe is unchanged in the supplied coefficient's floating precision; outputs are zeroed." LostPerturbation
@doc "The modal scan quadratic coefficient is within its Float64 roundoff guard." UnresolvedCurvature
@doc "The modal scan quadratic opens upward." NonconcaveFit
@doc "The modal scan quadratic peak lacks sampled amplitudes on both sides at working precision." UnbracketedPeak

"""
    CoefficientSpecification(count)

Declare a positive coefficient count. Every coefficient, direction, gradient
and moment vector uses this same caller-declared coordinate order and basis.
Coefficients must have homogeneous units for a scalar perturbation or Adam
step size. No physical unit conversion or normalization is applied. Numerical
products use Float64 storage; only ordinary one-based scalar CPU arrays and
views are supported.
"""
struct CoefficientSpecification
    count::Int
    function CoefficientSpecification(count::Integer)
        n = Int(count)
        n > 0 || throw(ArgumentError("coefficient count must be positive"))
        return new(n)
    end
end

function _vector(v::AbstractVector{<:Real}, n::Int, name::String)
    _storage(v, n, name)
    all(x -> isfinite(x) && isfinite(Float64(x)), v) ||
        throw(ArgumentError("$name must be finite and representable in Float64"))
    return nothing
end

function _storage(v::AbstractVector, n::Int, name::String)
    Base.require_one_based_indexing(v)
    length(v) == n || throw(DimensionMismatch("$name length"))
    return nothing
end

function _noalias(outputs::Tuple, inputs::Tuple)
    for (i, a) in enumerate(outputs)
        for b in inputs
            Base.mightalias(a, b) && throw(ArgumentError("output/workspace aliases input"))
        end
        for j in 1:(i - 1)
            Base.mightalias(a, outputs[j]) && throw(ArgumentError("output/workspace storage aliases"))
        end
    end
    return nothing
end

function _positive(x::Real, name::String)
    y = Float64(x)
    isfinite(y) && y > 0 || throw(ArgumentError("$name must be finite and positive in Float64"))
    return y
end

include("weighted_core_intensity.jl")
include("modal_scan.jl")
include("spsa.jl")
include("adam.jl")

end
