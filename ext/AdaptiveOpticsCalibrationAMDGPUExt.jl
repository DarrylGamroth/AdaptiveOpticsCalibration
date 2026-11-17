module AdaptiveOpticsCalibrationAMDGPUExt

using AdaptiveOpticsCalibration
using AMDGPU
using LinearAlgebra

const PhaseRetrieval = AdaptiveOpticsCalibration.PhaseRetrieval
const OpticalGains = AdaptiveOpticsCalibration.OpticalGains
const Tomography = AdaptiveOpticsCalibration.Tomography

function Tomography._covariance_reconstructor_lu_solve!(
    factor::AMDGPU.ROCArray{T,2},
    solution::AMDGPU.ROCArray{T,2},
    workspace,
) where {T<:AbstractFloat}
    _, _, info = AMDGPU.rocSOLVER.getrf!(factor, workspace.pivot)
    info > 0 && throw(SingularException(info))
    AMDGPU.rocSOLVER.getrs!('N', factor, workspace.pivot, solution)
    return nothing
end

function _covariance_roc_storage_range(root::AMDGPU.ROCArray)
    isempty(root) && return nothing
    start = UInt(pointer(root))
    return start, start + UInt(sizeof(root))
end

function Tomography._covariance_reconstructor_root_mightalias(
    ::AMDGPU.ROCBackend,
    first_root::AMDGPU.ROCArray,
    second_root::AMDGPU.ROCArray,
)
    first_range = _covariance_roc_storage_range(first_root)
    second_range = _covariance_roc_storage_range(second_root)
    (first_range === nothing || second_range === nothing) && return false
    first_start, first_end = first_range
    second_start, second_end = second_range
    return first_start < second_end && second_start < first_end
end

Tomography._covariance_reconstructor_root_mightalias(
    ::AMDGPU.ROCBackend, ::AbstractArray, ::AbstractArray) = true
Tomography._covariance_reconstructor_root_mightalias(
    ::AMDGPU.ROCBackend, ::Any, ::Any) = true

# ROCArray's dataids compares only starting pointers. Offset views can share
# storage while reporting different identities, so check their byte ranges.
function OpticalGains._gain_sensing_storage_mightalias(
    first::AMDGPU.ROCArray, second::AMDGPU.ROCArray)
    isempty(first) && return false
    isempty(second) && return false
    first_start = UInt(pointer(first))
    second_start = UInt(pointer(second))
    return first_start <= second_start ?
        second_start - first_start < sizeof(first) :
        first_start - second_start < sizeof(second)
end

function OpticalGains._execute_gain_sensing_fft_plan!(
    buffer::AMDGPU.ROCArray, plan)
    plan * buffer
    return buffer
end

function PhaseRetrieval._validate_weighting(
    weighting::PhaseRetrieval.LiFTVarianceMapWeighting{A},
    axes_::Tuple,
    ::Type{T},
) where {T<:AbstractFloat,A<:AMDGPU.AnyROCArray}
    axes(weighting.variance) == axes_ || throw(DimensionMismatch(
        "LiFT variance-map axes must match the model observation axes",
    ))
    eltype(weighting.variance) === T || throw(ArgumentError(
        "LiFT variance map must use the LiFT numeric type",
    ))
    # AMDGPU 2.8's generic map-reduction is not valid for this predicate.
    # Preparation is cold, so validate a host snapshot and leave repeated
    # execution device-resident.
    variance = Array(weighting.variance)
    all(value -> isfinite(value) && value >= zero(T), variance) || throw(
        ArgumentError(
            "LiFT variance map must contain finite nonnegative values",
        ),
    )
    return nothing
end

function PhaseRetrieval._device_normal_fallback!(
    workspace::PhaseRetrieval.LiFTKernelWorkspace{W,R}, regularization,
) where {W,R<:AMDGPU.AnyROCVector}
    inverse = workspace.inverse
    factorization = svd(AMDGPU.ROCArray(inverse.jacobian); full=false)
    return PhaseRetrieval._device_svd_step!(inverse, factorization, regularization)
end

function PhaseRetrieval.solve_lift_normal_device!(
    workspace::PhaseRetrieval.LiFTKernelWorkspace{W,R},
    ::PhaseRetrieval.LiFTKernelPlan,
    damping::D,
    state::T,
    regularization_load::T,
) where {
    T<:AbstractFloat,
    W,
    R<:AMDGPU.AnyROCVector,
    D<:Union{
        PhaseRetrieval.LiFTDampingNone,
        PhaseRetrieval.LiFTLevenbergMarquardt,
        PhaseRetrieval.LiFTAdaptiveLevenbergMarquardt,
    },
}
    return PhaseRetrieval._solve_device_normal!(
        workspace, damping, state, regularization_load)
end

end
