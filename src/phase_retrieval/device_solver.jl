# Shared GPU LiFT numerical policy. Vendor extensions retain storage conversion
# and factorization dispatch; the scalar CPU solver keeps its separate path.
function _try_device_normal!(inverse, regularization)
    copyto!(inverse.factor, inverse.normal)
    if regularization > zero(regularization)
        @views inverse.factor[diagind(inverse.factor)] .+= regularization
    end
    factorization = cholesky!(Hermitian(inverse.factor), check=false)
    issuccess(factorization) || return false
    ldiv!(factorization, inverse.rhs)
    return true
end

function _device_svd_step!(inverse, factorization, regularization)
    mul!(inverse.rhs, transpose(factorization.U), inverse.residual)
    @. inverse.rhs = ifelse(
        iszero(factorization.S^2 + regularization),
        zero(eltype(inverse.rhs)),
        factorization.S * inverse.rhs /
        (factorization.S^2 + regularization),
    )
    mul!(inverse.fallback_work, factorization.V, inverse.rhs)
    copyto!(inverse.rhs, inverse.fallback_work)
    return regularization, true
end

function _solve_device_normal!(workspace, ::LiFTDampingNone,
    ::T, regularization_load::T) where {T<:AbstractFloat}
    inverse = workspace.inverse
    _try_device_normal!(inverse, zero(T)) && return zero(T), false
    _try_device_normal!(inverse, regularization_load) &&
        return regularization_load, false
    regularization = T(10) * regularization_load
    _try_device_normal!(inverse, regularization) &&
        return regularization, false
    return _device_normal_fallback!(workspace, regularization)
end

function _solve_device_normal!(workspace, damping::D,
    state::T, regularization_load::T) where {
    T<:AbstractFloat,
    D<:Union{LiFTLevenbergMarquardt,LiFTAdaptiveLevenbergMarquardt},
}
    inverse = workspace.inverse
    regularization = max(
        state * max(regularization_load, one(T)),
        regularization_load,
    )
    while !_try_device_normal!(inverse, regularization)
        regularization = max(
            regularization * T(damping.growth),
            regularization_load,
        )
        regularization > T(1e12) &&
            return _device_normal_fallback!(workspace, regularization)
    end
    return regularization, false
end

# Implemented only for supported vendor storage, without a host fallback.
function _device_normal_fallback! end
