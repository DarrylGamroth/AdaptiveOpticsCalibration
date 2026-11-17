@inline _domain_rate_scale(::LiFTPhotonRate, ::Type{T}) where {T<:AbstractFloat} = one(T)
@inline _domain_rate_scale(domain::LiFTExpectedCounts, ::Type{T}) where {T<:AbstractFloat} =
    inv(T(domain.exposure_duration_s) * T(domain.quantum_efficiency))
@inline _domain_rate_scale(domain::LiFTNormalizedIntensity, ::Type{T}) where {T<:AbstractFloat} =
    T(domain.photon_rate_per_unit)

@inline _shot_variance_scale(domain::LiFTPhotonRate, ::Type{T}) where {T<:AbstractFloat} =
    inv(T(domain.noise_equivalent_exposure_s) * T(domain.quantum_efficiency))
@inline _shot_variance_scale(domain::LiFTExpectedCounts, ::Type{T}) where {T<:AbstractFloat} =
    inv(T(domain.exposure_duration_s) * T(domain.quantum_efficiency))
@inline _shot_variance_scale(domain::LiFTNormalizedIntensity, ::Type{T}) where {T<:AbstractFloat} =
    inv(T(domain.noise_equivalent_exposure_s) * T(domain.quantum_efficiency))

function _validate_domain(domain::LiFTPhotonRate, ::Type{T}) where {T<:AbstractFloat}
    exposure = T(domain.noise_equivalent_exposure_s)
    qe = T(domain.quantum_efficiency)
    isfinite(exposure) && exposure > zero(T) || throw(ArgumentError("LiFT photon-rate noise_equivalent_exposure_s must be finite and positive"))
    isfinite(qe) && zero(T) < qe <= one(T) || throw(ArgumentError("LiFT photon-rate quantum_efficiency must be finite and in (0, 1]"))
end
function _validate_domain(domain::LiFTExpectedCounts, ::Type{T}) where {T<:AbstractFloat}
    exposure = T(domain.exposure_duration_s)
    qe = T(domain.quantum_efficiency)
    isfinite(exposure) && exposure > zero(T) || throw(ArgumentError("LiFT expected-count exposure_duration_s must be finite and positive"))
    isfinite(qe) && zero(T) < qe <= one(T) || throw(ArgumentError("LiFT expected-count quantum_efficiency must be finite and in (0, 1]"))
end
function _validate_domain(domain::LiFTNormalizedIntensity, ::Type{T}) where {T<:AbstractFloat}
    scale = T(domain.photon_rate_per_unit)
    isfinite(scale) && scale > zero(T) || throw(ArgumentError("LiFT normalized-intensity photon_rate_per_unit must be finite and positive"))
    exposure = T(domain.noise_equivalent_exposure_s)
    qe = T(domain.quantum_efficiency)
    isfinite(exposure) && exposure > zero(T) || throw(ArgumentError("LiFT normalized-intensity noise_equivalent_exposure_s must be finite and positive"))
    isfinite(qe) && zero(T) < qe <= one(T) || throw(ArgumentError("LiFT normalized-intensity quantum_efficiency must be finite and in (0, 1]"))
end

function _validate_damping(::LiFTDampingNone, ::Type{T}) where {T<:AbstractFloat}
    nothing
end
function _validate_damping(damping::LiFTLevenbergMarquardt, ::Type{T}) where {T<:AbstractFloat}
    lambda0, growth, rtol = T(damping.lambda0), T(damping.growth), T(damping.condition_rtol)
    isfinite(lambda0) && lambda0 >= zero(T) || throw(ArgumentError("LiFT lambda0 must be finite and nonnegative"))
    isfinite(growth) && growth > one(T) || throw(ArgumentError("LiFT damping growth must be finite and greater than one"))
    isfinite(rtol) && rtol > zero(T) || throw(ArgumentError("LiFT condition_rtol must be finite and positive"))
end
function _validate_damping(damping::LiFTAdaptiveLevenbergMarquardt, ::Type{T}) where {T<:AbstractFloat}
    _validate_damping(LiFTLevenbergMarquardt(
        lambda0=T(damping.lambda0), growth=T(damping.growth),
        condition_rtol=T(damping.condition_rtol),
    ), T)
    shrink, minimum = T(damping.shrink), T(damping.min_lambda)
    isfinite(shrink) && shrink > one(T) || throw(ArgumentError("LiFT damping shrink must be finite and greater than one"))
    isfinite(minimum) && minimum >= zero(T) || throw(ArgumentError("LiFT minimum damping must be finite and nonnegative"))
end

function _validate_weighting(weighting::LiFTVarianceMapWeighting, axes_::Tuple, ::Type{T}) where {T<:AbstractFloat}
    axes(weighting.variance) == axes_ || throw(DimensionMismatch("LiFT variance-map axes must match the model observation axes"))
    eltype(weighting.variance) === T || throw(ArgumentError("LiFT variance map must use the LiFT numeric type"))
    all(value -> isfinite(value) && value >= zero(T), weighting.variance) ||
        throw(ArgumentError("LiFT variance map must contain finite nonnegative values"))
end
_validate_weighting(::AbstractLiFTWeighting, ::Tuple, ::Type{<:AbstractFloat}) = nothing

_prepared_weighting(weighting::Union{LiFTInitialModelWeighting,LiFTIterativeModelWeighting,LiFTReadNoiseWeighting}) = weighting
_prepared_weighting(weighting::LiFTVarianceMapWeighting) = LiFTVarianceMapWeighting(copy(weighting.variance))

_validated_observation_axes(axes_::Tuple) = axes_
_validated_observation_axes(axes_) = throw(ArgumentError(
    "LiFT forward model observation_axes must return a tuple",
))

@inline function _all_finite_nonnegative(values::AbstractArray{T}) where {T<:AbstractFloat}
    @inbounds for value in values
        isfinite(value) && value >= zero(T) || return false
    end
    return true
end

@inline function _all_finite(values::AbstractArray{T}) where {T<:AbstractFloat}
    @inbounds for value in values
        isfinite(value) || return false
    end
    return true
end

function _validated_mode_indices(indices, count::Int)
    values = isnothing(indices) ? collect(1:count) : collect(indices)
    isempty(values) && throw(ArgumentError("LiFT mode_indices must not be empty"))
    all(index -> index > 0 && index <= count, values) ||
        throw(ArgumentError("LiFT mode_indices must select valid one-based coefficient indices"))
    allunique(values) || throw(ArgumentError("LiFT mode_indices must be unique"))
    return values
end

function prepare(method::LiFT, specification::LiFTSpecification{T}) where {T<:AbstractFloat}
    count = coefficient_count(specification.model)
    count > 0 || throw(ArgumentError("LiFT forward model coefficient_count must be positive"))
    axes_ = _validated_observation_axes(observation_axes(specification.model))
    output = allocate_photon_rate(specification.model)
    eltype(output) === T || throw(ArgumentError("LiFT forward model photon-rate output must use its declared numeric type"))
    axes(output) == axes_ || throw(DimensionMismatch("LiFT forward-model output axes must equal observation_axes(model)"))
    length(output) > 0 || throw(ArgumentError("LiFT forward model observation extent must be positive"))
    _validate_domain(specification.observation_domain, T)
    read_noise = specification.read_noise_std
    isfinite(read_noise) && read_noise >= zero(T) || throw(ArgumentError("LiFT read_noise_std must be finite and nonnegative"))
    _validate_damping(method.damping, T)
    weighting = _prepared_weighting(method.weighting)
    _validate_weighting(weighting, axes_, T)
    indices = _validated_mode_indices(method.mode_indices, count)
    length(indices) <= length(output) || throw(DimensionMismatch(
        "LiFT selected mode count must not exceed the observation count",
    ))
    observation_to_rate_scale = _domain_rate_scale(specification.observation_domain, T)
    shot_variance_rate_scale = _shot_variance_scale(specification.observation_domain, T)
    read_noise_rate = read_noise * observation_to_rate_scale
    isfinite(observation_to_rate_scale) && observation_to_rate_scale > zero(T) ||
        throw(ArgumentError("LiFT observation-to-rate scale must be finite and positive"))
    isfinite(shot_variance_rate_scale) && shot_variance_rate_scale > zero(T) ||
        throw(ArgumentError("LiFT shot-variance rate scale must be finite and positive"))
    isfinite(read_noise_rate) && read_noise_rate >= zero(T) ||
        throw(ArgumentError("LiFT read-noise rate must be finite and nonnegative"))
    return LiFTPlan{T,typeof(specification.model),typeof(axes_),typeof(method.jacobian_method),
        typeof(method.solve_mode),typeof(method.damping),typeof(indices),
        typeof(weighting),typeof(method.model_scaling)}(
        specification.model, count, axes_, length(output), indices, method.iterations,
        method.jacobian_method, method.solve_mode, method.damping, weighting,
        method.model_scaling, observation_to_rate_scale,
        shot_variance_rate_scale, read_noise_rate,
        method.check_convergence, T(method.relative_update_tolerance),
    )
end

function allocate_result(plan::LiFTPlan{T}) where {T<:AbstractFloat}
    n_modes = length(plan.mode_indices)
    diagnostics = LiFTDiagnostics{T}(zero(T), zero(T), zero(T), zero(T), zero(T), 0, false, false, false)
    coefficients = similar(allocate_photon_rate(plan.model), T, (n_modes,))
    fill!(coefficients, zero(T))
    return LiFTProduct{T,typeof(coefficients)}(coefficients, diagnostics)
end

function allocate_workspace(plan::LiFTPlan{T}) where {T<:AbstractFloat}
    allocate_rate() = allocate_photon_rate(plan.model)
    observation_rate, model_rate, plus_rate, minus_rate =
        allocate_rate(), allocate_rate(), allocate_rate(), allocate_rate()
    n_modes = length(plan.mode_indices)
    model_workspace = allocate_model_workspace(plan.model)
    residual = similar(observation_rate, T, (plan.observation_count,))
    sqrt_weights = similar(residual)
    jacobian = similar(observation_rate, T, (plan.observation_count, n_modes))
    jacobian_backup = similar(jacobian)
    normal = similar(observation_rate, T, (n_modes, n_modes))
    factor = similar(normal)
    rhs = similar(residual, T, (n_modes,))
    coefficients = similar(residual, T, (plan.coefficient_count,))
    fallback_work = similar(rhs)
    return LiFTWorkspace{T,typeof(model_workspace),typeof(observation_rate),typeof(residual),
        typeof(jacobian),typeof(normal),typeof(factor),typeof(rhs),typeof(coefficients),
        typeof(fallback_work)}(
        model_workspace, observation_rate, model_rate, plus_rate, minus_rate, residual,
        sqrt_weights, jacobian, jacobian_backup, normal, factor, rhs, coefficients, fallback_work,
    )
end

@inline function _workspace_arrays(workspace::LiFTWorkspace)
    return (
        workspace.observation_rate,
        workspace.model_rate,
        workspace.plus_rate,
        workspace.minus_rate,
        workspace.residual,
        workspace.sqrt_weights,
        workspace.jacobian,
        workspace.jacobian_backup,
        workspace.normal,
        workspace.factor,
        workspace.rhs,
        workspace.coefficients,
        workspace.fallback_work,
    )
end

@inline _plan_arrays(::AbstractLiFTWeighting) = ()
@inline _plan_arrays(weighting::LiFTVarianceMapWeighting) = (weighting.variance,)

function _validate_result_workspace_structure(plan::LiFTPlan{T}, result::LiFTProduct{T}, workspace::LiFTWorkspace{T}, inputs::LiFTInputs{T}) where {T<:AbstractFloat}
    axes(inputs.observation) == plan.observation_axes || throw(DimensionMismatch("LiFT observation axes must match the prepared forward model"))
    initial = inputs.initial_coefficients
    if !isnothing(initial)
        Base.require_one_based_indexing(initial)
        length(initial) == plan.coefficient_count || throw(DimensionMismatch("LiFT initial_coefficients must cover the complete forward-model basis"))
    end
    Base.require_one_based_indexing(
        result.coefficients,
        workspace.residual,
        workspace.sqrt_weights,
        workspace.jacobian,
        workspace.jacobian_backup,
        workspace.normal,
        workspace.factor,
        workspace.rhs,
        workspace.coefficients,
        workspace.fallback_work,
    )
    length(result.coefficients) == length(plan.mode_indices) || throw(DimensionMismatch("LiFT product coefficient storage has incompatible length"))
    length(workspace.coefficients) == plan.coefficient_count || throw(DimensionMismatch("LiFT workspace coefficient storage has incompatible length"))
    length(workspace.residual) == plan.observation_count || throw(DimensionMismatch("LiFT workspace residual storage has incompatible length"))
    length(workspace.sqrt_weights) == plan.observation_count || throw(DimensionMismatch("LiFT workspace weight storage has incompatible length"))
    axes(workspace.observation_rate) == plan.observation_axes || throw(DimensionMismatch("LiFT workspace observation-rate axes must match the model"))
    axes(workspace.model_rate) == plan.observation_axes || throw(DimensionMismatch("LiFT workspace model-rate axes must match the model"))
    axes(workspace.plus_rate) == plan.observation_axes || throw(DimensionMismatch("LiFT workspace positive-rate axes must match the model"))
    axes(workspace.minus_rate) == plan.observation_axes || throw(DimensionMismatch("LiFT workspace negative-rate axes must match the model"))
    size(workspace.jacobian) == (plan.observation_count, length(plan.mode_indices)) || throw(DimensionMismatch("LiFT workspace Jacobian has incompatible dimensions"))
    size(workspace.jacobian_backup) == size(workspace.jacobian) || throw(DimensionMismatch("LiFT workspace Jacobian backup has incompatible dimensions"))
    size(workspace.normal) == (length(plan.mode_indices), length(plan.mode_indices)) || throw(DimensionMismatch("LiFT workspace normal matrix has incompatible dimensions"))
    size(workspace.factor) == size(workspace.normal) || throw(DimensionMismatch("LiFT workspace factor matrix has incompatible dimensions"))
    length(workspace.rhs) == length(plan.mode_indices) || throw(DimensionMismatch("LiFT workspace right-hand side has incompatible length"))
    length(workspace.fallback_work) == length(plan.mode_indices) || throw(DimensionMismatch("LiFT workspace fallback storage has incompatible length"))
    Base.mightalias(result.coefficients, inputs.observation) && throw(ArgumentError("LiFT product coefficient and observation storage must not alias"))
    storages = _workspace_arrays(workspace)
    plan_arrays = _plan_arrays(plan.weighting)
    _mightalias_any(result.coefficients, plan_arrays) &&
        throw(ArgumentError("LiFT product storage must not alias plan-owned storage"))
    _any_cross_alias(storages, plan_arrays) &&
        throw(ArgumentError("LiFT workspace storage must not alias plan-owned storage"))
    _mightalias_any(result.coefficients, storages) &&
        throw(ArgumentError("LiFT product storage must not alias workspace storage"))
    _mightalias_any(inputs.observation, storages) &&
        throw(ArgumentError("LiFT observation storage must not alias workspace storage"))
    if !isnothing(initial)
        Base.mightalias(result.coefficients, initial) && throw(ArgumentError(
            "LiFT product and initial-coefficient storage must not alias",
        ))
        _mightalias_any(initial, storages) && throw(ArgumentError(
            "LiFT initial coefficients must not alias workspace storage",
        ))
    end
    _any_pair_alias(storages) &&
        throw(ArgumentError("LiFT workspace arrays must not alias"))
    return nothing
end

function _validate_result_workspace(plan::LiFTPlan{T}, result::LiFTProduct{T}, workspace::LiFTWorkspace{T}, inputs::LiFTInputs{T}) where {T<:AbstractFloat}
    _validate_result_workspace_structure(plan, result, workspace, inputs)
    _all_finite_nonnegative(inputs.observation) ||
        throw(ArgumentError("LiFT observation must contain only finite nonnegative values"))
    initial = inputs.initial_coefficients
    if !isnothing(initial)
        all(isfinite, initial) || throw(ArgumentError(
            "LiFT initial_coefficients must contain only finite values",
        ))
    end
    return nothing
end

@inline function _copy_observation_rate!(destination::AbstractArray{T}, observation::AbstractArray{T}, scale::T) where {T<:AbstractFloat}
    @inbounds @simd for index in eachindex(destination, observation)
        destination[index] = observation[index] * scale
    end
    return destination
end

@inline function _residual!(destination::AbstractVector{T}, observation::AbstractArray{T}, model::AbstractArray{T}) where {T<:AbstractFloat}
    row = firstindex(destination)
    @inbounds for index in eachindex(observation, model)
        destination[row] = observation[index] - model[index]
        row += 1
    end
    return destination
end

function _model_scale!(::LiFTTotalRateMatching, model::AbstractArray{T}, observation::AbstractArray{T}) where {T<:AbstractFloat}
    denominator = sum(model)
    scale = denominator > zero(T) ? sum(observation) / denominator : one(T)
    @. model *= scale
    return scale, max(abs(sum(model)), one(T))
end
function _model_scale!(::LiFTPeakRateMatching, model::AbstractArray{T}, observation::AbstractArray{T}) where {T<:AbstractFloat}
    denominator = maximum(model)
    scale = denominator > zero(T) ? maximum(observation) / denominator : one(T)
    @. model *= scale
    return scale, max(abs(sum(model)), one(T))
end
function _model_scale!(::LiFTPhysicalRatePreservation, model::AbstractArray{T}, ::AbstractArray{T}) where {T<:AbstractFloat}
    return one(T), max(abs(sum(model)), one(T))
end

function _sqrt_weights!(weights::AbstractVector)
    @inbounds @simd for index in eachindex(weights)
        weights[index] = sqrt(weights[index])
    end
    return weights
end

function _initialize_weights!(weights::AbstractVector{T}, ::LiFTReadNoiseWeighting,
    ::AbstractArray{T}, plan::LiFTPlan{T}) where {T<:AbstractFloat}
    fill!(weights, inv(max(plan.read_noise_rate^2, eps(T))))
    return _sqrt_weights!(weights)
end
function _initialize_weights!(weights::AbstractVector{T}, weighting::LiFTVarianceMapWeighting,
    ::AbstractArray{T}, plan::LiFTPlan{T}) where {T<:AbstractFloat}
    scale2 = plan.observation_to_rate_scale^2
    row = firstindex(weights)
    @inbounds for index in eachindex(weighting.variance)
        weights[row] = inv(max(weighting.variance[index] * scale2, eps(T)))
        row += 1
    end
    return _sqrt_weights!(weights)
end
function _initialize_weights!(weights::AbstractVector{T}, ::Union{LiFTInitialModelWeighting,LiFTIterativeModelWeighting},
    model::AbstractArray{T}, plan::LiFTPlan{T}) where {T<:AbstractFloat}
    row = firstindex(weights)
    @inbounds for index in eachindex(model)
        weights[row] = inv(max(model[index] * plan.shot_variance_rate_scale + plan.read_noise_rate^2, eps(T)))
        row += 1
    end
    return _sqrt_weights!(weights)
end

function _update_weights!(weights::AbstractVector, weighting::LiFTInitialModelWeighting,
    model::AbstractArray, plan, iteration::Int)
    iteration == 1 && _initialize_weights!(weights, weighting, model, plan)
    return weights
end
function _update_weights!(weights::AbstractVector, weighting::LiFTIterativeModelWeighting,
    model::AbstractArray, plan, ::Int)
    return _initialize_weights!(weights, weighting, model, plan)
end
_update_weights!(weights::AbstractVector, ::Union{LiFTReadNoiseWeighting,LiFTVarianceMapWeighting}, ::AbstractArray, plan, ::Int) = weights

function _numerical_jacobian!(jacobian::AbstractMatrix{T}, workspace::LiFTWorkspace{T}, plan::LiFTPlan{T}, step::T) where {T<:AbstractFloat}
    coefficients = workspace.coefficients
    @inbounds for column in axes(jacobian, 2)
        coefficient_index = plan.mode_indices[column]
        coefficients[coefficient_index] += step
        predict_photon_rate!(workspace.plus_rate, plan.model, workspace.model_workspace, coefficients)
        coefficients[coefficient_index] -= T(2) * step
        predict_photon_rate!(workspace.minus_rate, plan.model, workspace.model_workspace, coefficients)
        coefficients[coefficient_index] += step
        row = firstindex(jacobian, 1)
        for index in eachindex(workspace.plus_rate, workspace.minus_rate)
            jacobian[row, column] =
                (workspace.plus_rate[index] - workspace.minus_rate[index]) / (T(2) * step)
            row += 1
        end
    end
    return jacobian
end

function _jacobian!(::LiFTNumericalJacobian, jacobian, workspace, plan)
    return _numerical_jacobian!(jacobian, workspace, plan, eltype(jacobian)(plan.jacobian_method.step_m))
end
function _jacobian!(::LiFTAnalyticJacobian, jacobian, workspace, plan)
    return analytic_photon_rate_jacobian!(jacobian, plan.model, workspace.model_workspace,
        workspace.coefficients, plan.mode_indices)
end

function _apply_weights!(jacobian::AbstractMatrix{T}, residual::AbstractVector{T}, weights::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds for column in axes(jacobian, 2)
        @simd for row in axes(jacobian, 1)
            jacobian[row, column] *= weights[row]
        end
    end
    @inbounds @simd for index in eachindex(residual, weights)
        residual[index] *= weights[index]
    end
    return nothing
end

@inline function _normal_condition_ratio(normal::AbstractMatrix{T}) where {T<:AbstractFloat}
    maximum_diagonal, minimum_diagonal = zero(T), typemax(T)
    @inbounds for index in axes(normal, 1)
        value = abs(normal[index, index])
        maximum_diagonal = max(maximum_diagonal, value)
        minimum_diagonal = min(minimum_diagonal, value)
    end
    return iszero(maximum_diagonal) ? T(Inf) : maximum_diagonal / max(minimum_diagonal, eps(T))
end

@inline _regularization_load(normal::AbstractMatrix{T}) where {T<:AbstractFloat} = begin
    diagonal_sum = zero(T)
    @inbounds for index in axes(normal, 1)
        diagonal_sum += abs(normal[index, index])
    end
    max(sqrt(eps(T)) * max(diagonal_sum / T(size(normal, 1)), one(T)), eps(T))
end

@inline _damping_state(::LiFTDampingNone, ::Type{T}) where {T<:AbstractFloat} = zero(T)
@inline _damping_state(damping::LiFTLevenbergMarquardt, ::Type{T}) where {T<:AbstractFloat} = T(damping.lambda0)
@inline _damping_state(damping::LiFTAdaptiveLevenbergMarquardt, ::Type{T}) where {T<:AbstractFloat} = T(damping.lambda0)

@inline _effective_lambda(::LiFTDampingNone, ::T, ::AbstractMatrix{T}) where {T<:AbstractFloat} = zero(T)
@inline _effective_lambda(damping::AbstractLiFTDamping, state::T, normal::AbstractMatrix{T}) where {T<:AbstractFloat} =
    max(state * max(_regularization_load(normal), one(T)), _regularization_load(normal))

function _next_damping_state(damping::LiFTAdaptiveLevenbergMarquardt, state::T,
    previous::T, current::T) where {T<:AbstractFloat}
    minimum = T(damping.min_lambda)
    !isfinite(previous) && return max(T(damping.lambda0), minimum)
    current > previous * (one(T) + sqrt(eps(T))) && return max(state * T(damping.growth), minimum)
    return max(state / T(damping.shrink), minimum)
end
_next_damping_state(::AbstractLiFTDamping, state::T, ::T, ::T) where {T<:AbstractFloat} = state

function _fallback_svd!(workspace::LiFTWorkspace{T}, jacobian::AbstractMatrix{T}, λ::T) where {T<:AbstractFloat}
    factor = svd(jacobian; full=false)
    mul!(workspace.rhs, transpose(factor.U), workspace.residual)
    @inbounds @simd for index in eachindex(workspace.rhs, factor.S)
        denominator = factor.S[index]^2 + λ
        workspace.rhs[index] = iszero(denominator) ? zero(T) :
            factor.S[index] * workspace.rhs[index] / denominator
    end
    mul!(workspace.fallback_work, factor.V, workspace.rhs)
    copyto!(workspace.rhs, workspace.fallback_work)
    return λ
end

function _prepare_normal_factor!(workspace::LiFTWorkspace{T}, λ::T) where {T<:AbstractFloat}
    copyto!(workspace.factor, workspace.normal)
    @inbounds for index in axes(workspace.factor, 1)
        workspace.factor[index, index] += λ
    end
    return nothing
end

function _try_normal!(workspace::LiFTWorkspace{T}, λ::T) where {T<:Union{Float32,Float64}}
    _prepare_normal_factor!(workspace, λ)
    _, info = LAPACK.potrf!(Char(85), workspace.factor)
    info == 0 || return false
    LAPACK.potrs!(Char(85), workspace.factor, workspace.rhs)
    return true
end

function _try_normal!(workspace::LiFTWorkspace{T}, λ::T) where {T<:AbstractFloat}
    _prepare_normal_factor!(workspace, λ)
    chol = cholesky!(Hermitian(workspace.factor), check=false)
    issuccess(chol) || return false
    ldiv!(chol, workspace.rhs)
    return true
end

function _solve_normal!(workspace::LiFTWorkspace{T}, ::LiFTDampingNone, ::T,
    fallback_jacobian::AbstractMatrix{T}) where {T<:AbstractFloat}
    _try_normal!(workspace, zero(T)) && return zero(T), false
    λ = _regularization_load(workspace.normal)
    _try_normal!(workspace, λ) && return λ, false
    λ *= T(10)
    _try_normal!(workspace, λ) && return λ, false
    return _fallback_svd!(workspace, fallback_jacobian, λ), true
end

function _solve_normal!(workspace::LiFTWorkspace{T}, damping::D, state::T,
    fallback_jacobian::AbstractMatrix{T}) where {T<:AbstractFloat,D<:Union{LiFTLevenbergMarquardt,LiFTAdaptiveLevenbergMarquardt}}
    λ = _effective_lambda(damping, state, workspace.normal)
    while !_try_normal!(workspace, λ)
        λ = max(λ * T(damping.growth), _regularization_load(workspace.normal))
        λ > T(1e12) && return _fallback_svd!(workspace, fallback_jacobian, λ), true
    end
    return λ, false
end

@inline _qr_condition_limit(::LiFTDampingNone, ::Type{T}) where {T<:AbstractFloat} = inv(sqrt(eps(T)))
@inline _qr_condition_limit(damping::Union{LiFTLevenbergMarquardt,LiFTAdaptiveLevenbergMarquardt}, ::Type{T}) where {T<:AbstractFloat} =
    inv(max(T(damping.condition_rtol), eps(T)))

function _solve_qr!(workspace::LiFTWorkspace{T}, damping::AbstractLiFTDamping, state::T) where {T<:AbstractFloat}
    copyto!(workspace.jacobian_backup, workspace.jacobian)
    factor = qr!(workspace.jacobian)
    maximum_diagonal, minimum_diagonal = zero(T), typemax(T)
    @inbounds for index in axes(workspace.normal, 1)
        value = abs(factor.factors[index, index])
        maximum_diagonal = max(maximum_diagonal, value)
        minimum_diagonal = min(minimum_diagonal, value)
    end
    condition = iszero(maximum_diagonal) ? T(Inf) : maximum_diagonal / max(minimum_diagonal, eps(T))
    if condition > _qr_condition_limit(damping, T)
        λ, fallback = _solve_normal!(workspace, damping, state, workspace.jacobian_backup)
        return condition, λ, false, true
    end
    ldiv!(factor, workspace.residual)
    @inbounds @simd for index in eachindex(workspace.rhs)
        workspace.rhs[index] = workspace.residual[index]
    end
    return condition, zero(T), true, false
end

_solve!(::LiFTSolveAuto, workspace, damping, state) = _solve_qr!(workspace, damping, state)
_solve!(::LiFTSolveQR, workspace, damping, state) = _solve_qr!(workspace, damping, state)
function _solve!(::LiFTSolveNormalEquations, workspace::LiFTWorkspace{T}, damping, state) where {T<:AbstractFloat}
    regularization, fallback = _solve_normal!(workspace, damping, state, workspace.jacobian)
    return _normal_condition_ratio(workspace.normal), regularization, false, fallback
end

_initialize_static_weights!(::Union{LiFTInitialModelWeighting,LiFTIterativeModelWeighting}, workspace, plan) = nothing
function _initialize_static_weights!(weighting::Union{LiFTReadNoiseWeighting,LiFTVarianceMapWeighting}, workspace, plan)
    _initialize_weights!(workspace.sqrt_weights, weighting, workspace.observation_rate, plan)
    return nothing
end

function process!(result::LiFTProduct{T}, workspace::LiFTWorkspace{T}, plan::LiFTPlan{T}, inputs::LiFTInputs{T}) where {T<:AbstractFloat}
    _validate_result_workspace(plan, result, workspace, inputs)
    _copy_observation_rate!(workspace.observation_rate, inputs.observation, plan.observation_to_rate_scale)
    _all_finite_nonnegative(workspace.observation_rate) || throw(ArgumentError(
        "LiFT rate conversion must produce finite nonnegative observations",
    ))
    if isnothing(inputs.initial_coefficients)
        fill!(workspace.coefficients, zero(T))
    else
        copyto!(workspace.coefficients, inputs.initial_coefficients)
    end
    _initialize_static_weights!(plan.weighting, workspace, plan)
    damping_state = _damping_state(plan.damping, T)
    previous_weighted_norm = T(Inf)
    diagnostics = LiFTDiagnostics{T}(zero(T), zero(T), zero(T), zero(T), zero(T), 0, false, false, false)
    for iteration in 1:plan.iterations
        predict_photon_rate!(workspace.model_rate, plan.model, workspace.model_workspace, workspace.coefficients)
        _all_finite_nonnegative(workspace.model_rate) ||
            throw(ArgumentError("LiFT forward model must predict finite nonnegative photon rates"))
        scale, objective_scale = _model_scale!(plan.model_scaling, workspace.model_rate, workspace.observation_rate)
        isfinite(scale) && scale >= zero(T) || throw(ArgumentError(
            "LiFT model scaling must produce a finite nonnegative scale",
        ))
        isfinite(objective_scale) && objective_scale > zero(T) || throw(ArgumentError(
            "LiFT objective scale must be finite and positive",
        ))
        _all_finite_nonnegative(workspace.model_rate) || throw(ArgumentError(
            "LiFT model scaling must preserve finite nonnegative photon rates",
        ))
        _residual!(workspace.residual, workspace.observation_rate, workspace.model_rate)
        _all_finite(workspace.residual) || throw(ArgumentError(
            "LiFT residual formation must produce finite values",
        ))
        residual_norm = norm(workspace.residual)
        isfinite(residual_norm) || throw(ArgumentError("LiFT residual norm must be finite"))
        _update_weights!(workspace.sqrt_weights, plan.weighting, workspace.model_rate, plan, iteration)
        _all_finite_nonnegative(workspace.sqrt_weights) || throw(ArgumentError(
            "LiFT weighting must produce finite nonnegative square-root weights",
        ))
        _jacobian!(plan.jacobian_method, workspace.jacobian, workspace, plan)
        _all_finite(workspace.jacobian) ||
            throw(ArgumentError("LiFT forward-model Jacobian must contain only finite values"))
        @. workspace.jacobian *= scale / objective_scale
        _all_finite(workspace.jacobian) || throw(ArgumentError(
            "LiFT Jacobian scaling must produce finite values",
        ))
        _apply_weights!(workspace.jacobian, workspace.residual, workspace.sqrt_weights)
        _all_finite(workspace.jacobian) && _all_finite(workspace.residual) ||
            throw(ArgumentError("LiFT weighted system must contain only finite values"))
        weighted_norm = norm(workspace.residual)
        isfinite(weighted_norm) || throw(ArgumentError(
            "LiFT weighted residual norm must be finite",
        ))
        @. workspace.residual /= objective_scale
        _all_finite(workspace.residual) || throw(ArgumentError(
            "LiFT normalized residual must contain only finite values",
        ))
        mul!(workspace.normal, transpose(workspace.jacobian), workspace.jacobian)
        mul!(workspace.rhs, transpose(workspace.jacobian), workspace.residual)
        _all_finite(workspace.normal) && _all_finite(workspace.rhs) ||
            throw(ArgumentError("LiFT normal system must contain only finite values"))
        damping_state = _next_damping_state(plan.damping, damping_state, previous_weighted_norm, weighted_norm)
        isfinite(damping_state) && damping_state >= zero(T) || throw(ArgumentError(
            "LiFT damping state must be finite and nonnegative",
        ))
        condition, regularization, used_qr, used_fallback =
            _solve!(plan.solve_mode, workspace, plan.damping, damping_state)
        !isnan(condition) && condition >= zero(T) || throw(ArgumentError(
            "LiFT condition ratio must be nonnegative and not NaN",
        ))
        isfinite(regularization) && regularization >= zero(T) || throw(ArgumentError(
            "LiFT regularization must be finite and nonnegative",
        ))
        _all_finite(workspace.rhs) || throw(ArgumentError(
            "LiFT solve must produce a finite coefficient update",
        ))
        update_norm = norm(workspace.rhs)
        isfinite(update_norm) || throw(ArgumentError("LiFT update norm must be finite"))
        @inbounds for index in eachindex(workspace.rhs)
            workspace.coefficients[plan.mode_indices[index]] += workspace.rhs[index]
        end
        _all_finite(workspace.coefficients) || throw(ArgumentError(
            "LiFT coefficient update must produce finite values",
        ))
        coefficient_norm = norm(workspace.coefficients)
        isfinite(coefficient_norm) || throw(ArgumentError("LiFT coefficient norm must be finite"))
        converged = plan.check_convergence && update_norm / max(coefficient_norm, eps(T)) < plan.relative_update_tolerance
        diagnostics = LiFTDiagnostics{T}(residual_norm, weighted_norm, update_norm, condition,
            regularization, iteration, converged, used_qr, used_fallback)
        previous_weighted_norm = weighted_norm
        converged && break
    end
    @inbounds @simd for index in eachindex(result.coefficients)
        result.coefficients[index] = workspace.coefficients[plan.mode_indices[index]]
    end
    result.diagnostics = diagnostics
    return result
end
