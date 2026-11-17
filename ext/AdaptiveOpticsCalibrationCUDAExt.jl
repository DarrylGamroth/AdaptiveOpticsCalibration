module AdaptiveOpticsCalibrationCUDAExt

using AdaptiveOpticsCalibration
using CUDA
using LinearAlgebra

const PhaseRetrieval = AdaptiveOpticsCalibration.PhaseRetrieval
const Tomography = AdaptiveOpticsCalibration.Tomography

Tomography._covariance_reconstructor_root_mightalias(
    ::CUDA.CUDABackend,
    first_root::CUDA.CuArray,
    second_root::CUDA.CuArray,
) = Base.mightalias(first_root, second_root)
Tomography._covariance_reconstructor_root_mightalias(
    ::CUDA.CUDABackend, ::AbstractArray, ::AbstractArray) = true
Tomography._covariance_reconstructor_root_mightalias(
    ::CUDA.CUDABackend, ::Any, ::Any) = true

function PhaseRetrieval._device_normal_fallback!(
    workspace::PhaseRetrieval.LiFTKernelWorkspace{W,R}, regularization,
) where {W,R<:CUDA.AnyCuVector}
    inverse = workspace.inverse
    factorization = svd(CUDA.CuArray(inverse.jacobian); full=false)
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
    R<:CUDA.AnyCuVector,
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
