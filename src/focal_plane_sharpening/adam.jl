# Kingma and Ba (2015), Adam. See docs/REFERENCES.md.
"""
    AdamStep(alpha, beta1, beta2, epsilon)

One minimization step with explicit positive α and ε and `0 ≤ β₁, β₂ < 1`.
Bias-correct both moments; ε is outside the square root. Coefficients have
homogeneous caller units; α has coefficient units, ε has gradient units.
No schedule, constraints, activation, or persistent history is held here.
"""
struct AdamStep <: AbstractCalibrationMethod
    alpha::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    function AdamStep(alpha::Real, beta1::Real, beta2::Real, epsilon::Real)
        a, e = _positive(alpha, "alpha"), _positive(epsilon, "epsilon")
        b1, b2 = _beta(beta1), _beta(beta2)
        return new(a, b1, b2, e)
    end
end

function _beta(beta::Real)
    b = Float64(beta)
    isfinite(b) && 0 <= b < 1 || throw(ArgumentError("beta must satisfy 0 ≤ beta < 1"))
    return b
end

"""Immutable Adam parameters and caller-declared coefficient count."""
struct AdamPlan <: AbstractCalibrationPlan
    method::AdamStep
    count::Int
end
prepare(method::AdamStep, spec::CoefficientSpecification) = AdamPlan(method, spec.count)

"""
    AdamState(first_moment, second_moment, iteration; beta1, beta2)

Explicit persistent snapshot, with moment vectors in the coefficient order.
The first moment has gradient units and the nonnegative second moment squared
gradient units. `iteration` is a nonnegative UInt64-representable count.
The supplied beta history tags must equal the prepared method's fixed betas.
Iteration zero requires both moments zero. To change betas, supply a fresh
zero-moment, zero-iteration state with the new tags. Arrays are borrowed inputs;
processing never mutates them. An accepted product's `next_state` is owned by
that product and remains subject to later product reuse by its caller.
"""
struct AdamState{M<:AbstractVector{<:Real},V<:AbstractVector{<:Real}}
    first_moment::M
    second_moment::V
    iteration::UInt64
    beta1::Float64
    beta2::Float64
    function AdamState(m::M, v::V, t::Integer; beta1::Real, beta2::Real) where
                     {M<:AbstractVector{<:Real},V<:AbstractVector{<:Real}}
        0 <= t <= typemax(UInt64) || throw(ArgumentError("iteration outside UInt64 range"))
        return new{M,V}(m, v, UInt64(t), _beta(beta1), _beta(beta2))
    end
end

"""Supply coefficients, gradient, and an explicit AdamState snapshot in identical order."""
struct AdamInputs{Q<:AbstractVector{<:Real},G<:AbstractVector{<:Real},S<:AdamState}
    coefficients::Q
    gradient::G
    state::S
end

"""
Caller-owned Float64 `candidate` and independent `next_state` moment storage.
`NumericalFailure` zeroes candidate and moments and resets the output count to
zero; it is not an accepted next state. Input state is never advanced implicitly.
"""
mutable struct AdamProduct
    status::SharpeningStatus
    candidate::Vector{Float64}
    next_state::AdamState{Vector{Float64},Vector{Float64}}
end

"""Replaceable candidate/moment scratch, fully overwritten for each supplied snapshot."""
struct AdamWorkspace
    candidate::Vector{Float64}
    first_moment::Vector{Float64}
    second_moment::Vector{Float64}
end
allocate_workspace(plan::AdamPlan) = AdamWorkspace(
    zeros(plan.count), zeros(plan.count), zeros(plan.count))
function allocate_result(plan::AdamPlan)
    state = AdamState(zeros(plan.count), zeros(plan.count), 0;
        beta1=plan.method.beta1, beta2=plan.method.beta2)
    return AdamProduct(NumericalFailure, zeros(plan.count), state)
end

# Stable for beta near one; beta zero must avoid log(0) and 0*Inf.
_bias(beta::Float64, t::UInt64) = beta == 0.0 ? 1.0 : -expm1(Float64(t) * log(beta))

function process!(result::AdamProduct, workspace::AdamWorkspace,
                  plan::AdamPlan, inputs::AdamInputs)
    q, g, state = inputs.coefficients, inputs.gradient, inputs.state
    m, v, method = state.first_moment, state.second_moment, plan.method
    _vector(q, plan.count, "coefficients")
    _vector(g, plan.count, "gradient")
    _vector(m, plan.count, "first moment")
    _vector(v, plan.count, "second moment")
    all(x -> x >= 0, v) || throw(ArgumentError("second moments must be nonnegative"))
    state.beta1 == method.beta1 && state.beta2 == method.beta2 ||
        throw(ArgumentError("Adam beta history differs; reset state explicitly"))
    state.iteration < typemax(UInt64) || throw(ArgumentError("iteration cannot advance"))
    if state.iteration == 0
        all(iszero, m) && all(iszero, v) ||
            throw(ArgumentError("iteration zero requires zero moments"))
    end
    outputs = (result.candidate, result.next_state.first_moment,
        result.next_state.second_moment, workspace.candidate,
        workspace.first_moment, workspace.second_moment)
    for a in outputs
        _storage(a, plan.count, "Adam output/workspace")
    end
    _noalias(outputs, (q, g, m, v))
    t = state.iteration + UInt64(1)
    bias1, bias2 = _bias(method.beta1, t), _bias(method.beta2, t)
    valid = true
    @inbounds for i in 1:plan.count
        gi = Float64(g[i])
        mi = method.beta1 * Float64(m[i]) + (1 - method.beta1) * gi
        vi = method.beta2 * Float64(v[i]) + (1 - method.beta2) * (gi * gi)
        mhat, vhat = mi / bias1, vi / bias2
        candidate = Float64(q[i]) - method.alpha * (mhat / (sqrt(vhat) + method.epsilon))
        workspace.first_moment[i] = mi
        workspace.second_moment[i] = vi
        workspace.candidate[i] = candidate
        valid &= isfinite(mi) && isfinite(vi) && isfinite(mhat) &&
            isfinite(vhat) && isfinite(candidate)
    end
    if valid
        copyto!(result.candidate, workspace.candidate)
        copyto!(result.next_state.first_moment, workspace.first_moment)
        copyto!(result.next_state.second_moment, workspace.second_moment)
        result.status = Valid
    else
        fill!(result.candidate, 0.0)
        fill!(result.next_state.first_moment, 0.0)
        fill!(result.next_state.second_moment, 0.0)
        result.status = NumericalFailure
    end
    result.next_state = AdamState(result.next_state.first_moment,
        result.next_state.second_moment, valid ? t : UInt64(0);
        beta1=method.beta1, beta2=method.beta2)
    return result
end
