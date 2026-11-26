# Estimator inputs and outputs for the LIFT method of Meimon et al. (2010). See docs/REFERENCES.md.
"""
    AbstractLiFTForwardModel{T<:AbstractFloat}

Typed physical-model seam for LiFT. A concrete model has a fixed complete
metre-OPD coefficient order and a fixed photon-rate observation layout. It is
owned by the package that owns the physical model; this package only invokes
the documented numerical seams below. Model implementations must keep
run-immutable model storage disjoint from the writable arrays returned by
`allocate_model_workspace` and `allocate_photon_rate`; opaque model-owned
storage cannot be inspected by the inverse solver's alias checks.
"""
abstract type AbstractLiFTForwardModel{T<:AbstractFloat} end

"""Return the positive count of complete metre-OPD coefficients for `model`."""
function coefficient_count end

"""Return the exact axes of a photon-rate observation produced by `model`."""
function observation_axes end

"""Allocate replaceable model-specific scratch for one forward evaluation."""
function allocate_model_workspace end

"""Allocate photon-rate output storage with [`observation_axes`](@ref)."""
function allocate_photon_rate end

"""
    predict_photon_rate!(out, model, workspace, coefficients)

Write an unscaled photon-arrival-rate observation for the complete
metre-OPD `coefficients` vector. `out` must have `observation_axes(model)`.
"""
function predict_photon_rate! end

"""
    analytic_photon_rate_jacobian!(jacobian, model, workspace, coefficients,
                                   mode_indices)

Write derivatives of unscaled photon-arrival rate with respect to the selected
metre-OPD coefficient indices. Concrete models that support analytic LiFT
Jacobian evaluation extend this typed seam.
"""
function analytic_photon_rate_jacobian! end

abstract type AbstractLiFTJacobian end
"""Select a forward model's analytic photon-rate Jacobian seam."""
struct LiFTAnalyticJacobian <: AbstractLiFTJacobian end
"""Select a centered finite-difference photon-rate Jacobian."""
struct LiFTNumericalJacobian{T<:AbstractFloat} <: AbstractLiFTJacobian
    step_m::T
end

function LiFTNumericalJacobian(step_m::Real=1e-9)
    T = float(typeof(step_m))
    value = T(step_m)
    isfinite(value) && value > zero(T) ||
        throw(ArgumentError("LiFT numerical Jacobian step_m must be finite and positive"))
    return LiFTNumericalJacobian{T}(value)
end

abstract type AbstractLiFTSolveMode end
"""Select in-place QR on the CPU reference."""
struct LiFTSolveAuto <: AbstractLiFTSolveMode end
"""Select in-place QR."""
struct LiFTSolveQR <: AbstractLiFTSolveMode end
"""Select normal equations with Cholesky and a diagnosed SVD fallback."""
struct LiFTSolveNormalEquations <: AbstractLiFTSolveMode end

abstract type AbstractLiFTDamping end
"""Select no requested Levenberg--Marquardt damping."""
struct LiFTDampingNone <: AbstractLiFTDamping end
"""Fixed Levenberg--Marquardt damping policy."""
struct LiFTLevenbergMarquardt{T<:AbstractFloat} <: AbstractLiFTDamping
    lambda0::T
    growth::T
    condition_rtol::T
end

function LiFTLevenbergMarquardt(
    ; lambda0::Real=1e-6,
    growth::Real=10,
    condition_rtol::Real=sqrt(eps(Float64)),
)
    T = float(promote_type(typeof(lambda0), typeof(growth), typeof(condition_rtol)))
    return LiFTLevenbergMarquardt{T}(T(lambda0), T(growth), T(condition_rtol))
end

"""Adaptive Levenberg--Marquardt damping policy."""
struct LiFTAdaptiveLevenbergMarquardt{T<:AbstractFloat} <: AbstractLiFTDamping
    lambda0::T
    growth::T
    shrink::T
    min_lambda::T
    condition_rtol::T
end

function LiFTAdaptiveLevenbergMarquardt(
    ; lambda0::Real=1e-6,
    growth::Real=10,
    shrink::Real=2,
    min_lambda::Real=1e-10,
    condition_rtol::Real=sqrt(eps(Float64)),
)
    T = float(promote_type(
        typeof(lambda0), typeof(growth), typeof(shrink), typeof(min_lambda),
        typeof(condition_rtol),
    ))
    return LiFTAdaptiveLevenbergMarquardt{T}(
        T(lambda0), T(growth), T(shrink), T(min_lambda), T(condition_rtol),
    )
end

abstract type AbstractLiFTObservationDomain end
"""Photon-arrival-rate observations; noise exposure and QE define shot noise."""
struct LiFTPhotonRate{T<:AbstractFloat} <: AbstractLiFTObservationDomain
    noise_equivalent_exposure_s::T
    quantum_efficiency::T
end

function LiFTPhotonRate(
    ; noise_equivalent_exposure_s::Real=1,
    quantum_efficiency::Real=1,
)
    T = float(promote_type(typeof(noise_equivalent_exposure_s), typeof(quantum_efficiency)))
    return LiFTPhotonRate{T}(T(noise_equivalent_exposure_s), T(quantum_efficiency))
end

"""Expected detected-count observations formed with an explicit exposure and QE."""
struct LiFTExpectedCounts{T<:AbstractFloat} <: AbstractLiFTObservationDomain
    exposure_duration_s::T
    quantum_efficiency::T
end

function LiFTExpectedCounts(exposure_duration_s::Real; quantum_efficiency::Real=1)
    T = float(promote_type(typeof(exposure_duration_s), typeof(quantum_efficiency)))
    return LiFTExpectedCounts{T}(T(exposure_duration_s), T(quantum_efficiency))
end

"""Dimensionless observations with an explicit photon-rate-per-unit scale."""
struct LiFTNormalizedIntensity{T<:AbstractFloat} <: AbstractLiFTObservationDomain
    photon_rate_per_unit::T
    noise_equivalent_exposure_s::T
    quantum_efficiency::T
end

function LiFTNormalizedIntensity(
    photon_rate_per_unit::Real;
    noise_equivalent_exposure_s::Real=1,
    quantum_efficiency::Real=1,
)
    T = float(promote_type(
        typeof(photon_rate_per_unit), typeof(noise_equivalent_exposure_s),
        typeof(quantum_efficiency),
    ))
    return LiFTNormalizedIntensity{T}(
        T(photon_rate_per_unit), T(noise_equivalent_exposure_s), T(quantum_efficiency),
    )
end

abstract type AbstractLiFTWeighting end
"""Freeze model-based inverse-variance weights after the first iteration."""
struct LiFTInitialModelWeighting <: AbstractLiFTWeighting end
"""Refresh model-based inverse-variance weights on every iteration."""
struct LiFTIterativeModelWeighting <: AbstractLiFTWeighting end
"""Use only the specified read-noise variance."""
struct LiFTReadNoiseWeighting <: AbstractLiFTWeighting end
"""Use an explicit nonnegative variance map in the supplied observation domain."""
struct LiFTVarianceMapWeighting{A<:AbstractArray{<:AbstractFloat}} <: AbstractLiFTWeighting
    variance::A
end

abstract type AbstractLiFTModelScaling end
"""Match model and observation total photon rate."""
struct LiFTTotalRateMatching <: AbstractLiFTModelScaling end
"""Match model and observation peak photon rate."""
struct LiFTPeakRateMatching <: AbstractLiFTModelScaling end
"""Keep the physical photon-rate scale supplied by the forward model."""
struct LiFTPhysicalRatePreservation <: AbstractLiFTModelScaling end

"""
    LiFT(; iterations=5, jacobian_method=LiFTAnalyticJacobian(),
         solve_mode=LiFTSolveAuto(), damping=LiFTDampingNone(),
         mode_indices=nothing, weighting=LiFTReadNoiseWeighting(),
         model_scaling=LiFTTotalRateMatching(), check_convergence=true,
         relative_update_tolerance=1e-3)

Cold selection of LiFT inverse equations and numerical policies. Selected
coefficient indices use the forward model's one-based complete basis order.
"""
struct LiFT{J<:AbstractLiFTJacobian,S<:AbstractLiFTSolveMode,D<:AbstractLiFTDamping,
            I,W<:AbstractLiFTWeighting,M<:AbstractLiFTModelScaling,T<:AbstractFloat} <:
       AbstractCalibrationMethod
    iterations::Int
    jacobian_method::J
    solve_mode::S
    damping::D
    mode_indices::I
    weighting::W
    model_scaling::M
    check_convergence::Bool
    relative_update_tolerance::T
end

function LiFT(
    ; iterations::Integer=5,
    jacobian_method::AbstractLiFTJacobian=LiFTAnalyticJacobian(),
    solve_mode::AbstractLiFTSolveMode=LiFTSolveAuto(),
    damping::AbstractLiFTDamping=LiFTDampingNone(),
    mode_indices=nothing,
    weighting::AbstractLiFTWeighting=LiFTReadNoiseWeighting(),
    model_scaling::AbstractLiFTModelScaling=LiFTTotalRateMatching(),
    check_convergence::Bool=true,
    relative_update_tolerance::Real=1e-3,
)
    count = Int(iterations)
    count > 0 || throw(ArgumentError("LiFT iterations must be positive"))
    T = float(typeof(relative_update_tolerance))
    tolerance = T(relative_update_tolerance)
    isfinite(tolerance) && tolerance >= zero(T) ||
        throw(ArgumentError("LiFT relative_update_tolerance must be finite and nonnegative"))
    indices = isnothing(mode_indices) ? nothing : Tuple(Int(i) for i in mode_indices)
    return LiFT(
        count, jacobian_method, solve_mode, damping, indices, weighting,
        model_scaling, check_convergence, tolerance,
    )
end

"""
    LiFTSpecification(model, observation_domain; read_noise_std=0)

Bind a typed, run-immutable forward model and the representation of complete
observations. `read_noise_std` is expressed in that observation domain.
"""
struct LiFTSpecification{T<:AbstractFloat,M<:AbstractLiFTForwardModel{T},D<:AbstractLiFTObservationDomain}
    model::M
    observation_domain::D
    read_noise_std::T
end

function LiFTSpecification(
    model::M,
    observation_domain::D;
    read_noise_std::Real=0,
) where {T<:AbstractFloat,M<:AbstractLiFTForwardModel{T},D<:AbstractLiFTObservationDomain}
    return LiFTSpecification{T,M,D}(model, observation_domain, T(read_noise_std))
end

"""One complete LiFT observation and an optional complete initial metre-OPD vector."""
struct LiFTInputs{T<:AbstractFloat,O<:AbstractArray{T},I}
    observation::O
    initial_coefficients::I
end

LiFTInputs(observation::O; initial_coefficients=nothing) where {T<:AbstractFloat,O<:AbstractArray{T}} =
    _lift_inputs(observation, initial_coefficients)

_lift_inputs(observation::O, ::Nothing) where {T<:AbstractFloat,O<:AbstractArray{T}} =
    LiFTInputs{T,O,Nothing}(observation, nothing)

_lift_inputs(observation::O, initial_coefficients::I) where {
    T<:AbstractFloat,O<:AbstractArray{T},I<:AbstractVector{T},
} = LiFTInputs{T,O,I}(observation, initial_coefficients)

_lift_inputs(::AbstractArray{T}, initial_coefficients) where {T<:AbstractFloat} =
    throw(ArgumentError("LiFT initial_coefficients must be an AbstractVector with the observation element type"))

"""Prepared LiFT dimensions, typed forward model, and numerical policies."""
struct LiFTPlan{
    T<:AbstractFloat,
    M<:AbstractLiFTForwardModel{T},
    A<:Tuple,
    J<:AbstractLiFTJacobian,
    S<:AbstractLiFTSolveMode,
    D<:AbstractLiFTDamping,
    I<:AbstractVector{Int},
    W<:AbstractLiFTWeighting,
    MS<:AbstractLiFTModelScaling,
} <: AbstractCalibrationPlan
    model::M
    coefficient_count::Int
    observation_axes::A
    observation_count::Int
    mode_indices::I
    iterations::Int
    jacobian_method::J
    solve_mode::S
    damping::D
    weighting::W
    model_scaling::MS
    observation_to_rate_scale::T
    shot_variance_rate_scale::T
    read_noise_rate::T
    check_convergence::Bool
    relative_update_tolerance::T
end

"""Numerical diagnostics from a complete LiFT inverse operation."""
struct LiFTDiagnostics{T<:AbstractFloat}
    residual_norm::T
    weighted_residual_norm::T
    update_norm::T
    condition_ratio::T
    regularization::T
    iterations::Int
    converged::Bool
    used_qr::Bool
    used_fallback::Bool
end

"""Caller-owned selected metre-OPD coefficients and LiFT diagnostics."""
mutable struct LiFTProduct{T<:AbstractFloat,V<:AbstractVector{T}}
    coefficients::V
    diagnostics::LiFTDiagnostics{T}
end

"""Return selected reconstructed metre-OPD coefficients."""
lift_coefficients(product::LiFTProduct) = product.coefficients
"""Return diagnostics for the most recently completed LiFT operation."""
lift_diagnostics(product::LiFTProduct) = product.diagnostics

"""Replaceable LiFT and forward-model scratch for explicit-workspace processing."""
struct LiFTWorkspace{T<:AbstractFloat,MW,OR<:AbstractArray{T},RV<:AbstractVector{T},
    JV<:AbstractMatrix{T},NV<:AbstractMatrix{T},FV<:AbstractMatrix{T},
    CV<:AbstractVector{T},SV<:AbstractVector{T},WV<:AbstractVector{T}}
    model_workspace::MW
    observation_rate::OR
    model_rate::OR
    plus_rate::OR
    minus_rate::OR
    residual::RV
    sqrt_weights::RV
    jacobian::JV
    jacobian_backup::JV
    normal::NV
    factor::FV
    rhs::CV
    coefficients::SV
    fallback_work::WV
end

"""KernelAbstractions-prepared LiFT plan with a backend-specific execution contract."""
struct LiFTKernelPlan{P<:LiFTPlan,E} <: AbstractCalibrationPlan
    plan::P
    execution::E
end

"""Replaceable device LiFT scratch plus prepared reduction and solver storage."""
struct LiFTKernelWorkspace{W,R,U,HS,HU}
    inverse::W
    reduction_first::R
    reduction_second::R
    invalid_first::U
    invalid_second::U
    host_scalar::HS
    host_invalid::HU
end

"""Vendor extension seam for an in-place device normal-equation solve."""
function solve_lift_normal_device! end
