"""Abstract method family for compact-SVD inverse construction."""
abstract type AbstractSVDInverse <: AbstractCalibrationMethod end

"""Select the exact compact-SVD pseudoinverse of an interaction matrix."""
struct ExactPseudoInverse <: AbstractSVDInverse end

"""
    TSVDInverse(; rtol=eps(Float64), atol=0, n_trunc=0)

Select a truncated-SVD inverse. Singular values must exceed
`max(atol, rtol * s₁)`. `n_trunc` then discards that many additional
accepted components from the low-singular-value end.
"""
struct TSVDInverse{T<:AbstractFloat} <: AbstractSVDInverse
    rtol::T
    atol::T
    n_trunc::Int
end

function TSVDInverse(
    ; rtol::Real=eps(Float64),
    atol::Real=0.0,
    n_trunc::Integer=0,
)
    T = promote_type(typeof(float(rtol)), typeof(float(atol)))
    return TSVDInverse{T}(T(rtol), T(atol), Int(n_trunc))
end

"""
    TikhonovInverse(lambda; rtol=eps(Float64), atol=0)

Select the compact-SVD Tikhonov inverse with spectral gains
`sᵢ / (sᵢ² + lambda²)`. The tolerances classify effective rank for
diagnostics; they do not truncate gains.
"""
struct TikhonovInverse{T<:AbstractFloat} <: AbstractSVDInverse
    lambda::T
    rtol::T
    atol::T
end

function TikhonovInverse(
    lambda::Real;
    rtol::Real=eps(Float64),
    atol::Real=0.0,
)
    T = promote_type(
        typeof(float(lambda)),
        typeof(float(rtol)),
        typeof(float(atol)),
    )
    return TikhonovInverse{T}(T(lambda), T(rtol), T(atol))
end

"""Unweighted interaction matrix in measurement-row, corrector-column order."""
struct SVDReconstructorInputs{D<:AbstractMatrix{<:Real}}
    interaction_matrix::D
end

"""Prepared dimensions, representation, and compact-SVD inverse method."""
struct SVDReconstructorPlan{
    T<:AbstractFloat,
    M<:AbstractSVDInverse,
} <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    singular_count::Int
    method::M
end

function _validated_svd_tolerance(::Type{T}, value, name) where {T}
    converted = T(value)
    isfinite(converted) && converted >= zero(T) ||
        throw(ArgumentError("$name must be finite and nonnegative"))
    return converted
end

function _svd_reconstructor_plan(method, specification::ReconstructorSpecification{T}) where {T}
    return SVDReconstructorPlan{T,typeof(method)}(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        min(specification.measurement_count, specification.corrector_coordinate_count),
        method,
    )
end

function prepare(
    method::ExactPseudoInverse,
    specification::ReconstructorSpecification{T},
) where {T}
    return _svd_reconstructor_plan(method, specification)
end

function prepare(
    method::TSVDInverse,
    specification::ReconstructorSpecification{T},
) where {T}
    method.n_trunc >= 0 || throw(ArgumentError("n_trunc must be nonnegative"))
    prepared = TSVDInverse{T}(
        _validated_svd_tolerance(T, method.rtol, "rtol"),
        _validated_svd_tolerance(T, method.atol, "atol"),
        method.n_trunc,
    )
    return _svd_reconstructor_plan(prepared, specification)
end

function prepare(
    method::TikhonovInverse,
    specification::ReconstructorSpecification{T},
) where {T}
    lambda = T(method.lambda)
    isfinite(lambda) && lambda >= zero(T) ||
        throw(ArgumentError("lambda must be finite and nonnegative"))
    prepared = TikhonovInverse{T}(
        lambda,
        _validated_svd_tolerance(T, method.rtol, "rtol"),
        _validated_svd_tolerance(T, method.atol, "atol"),
    )
    return _svd_reconstructor_plan(prepared, specification)
end

"""
Numerical diagnostics for one compact-SVD inverse.

Exact and TSVD condition numbers are `s₁ / sᵣ`, where `r` is effective
rank, or infinity when no component is effective. The Tikhonov condition
number preserves the source convention `s₁ / max(sₙ, λ)`, or infinity for
a zero denominator.
"""
struct SVDReconstructorDiagnostics{T<:AbstractFloat}
    condition_number::T
    effective_rank::Int
    truncation_count::Int
end

"""
Caller-owned compact-SVD factors, reconstructor, singular values, gains, and
diagnostics. `measurement_singular_vectors` contains the left singular vectors
in measurement coordinates and `corrector_singular_vectors` contains the right
singular vectors in corrector coordinates. The reconstructor matrix is in
corrector-row, measurement-column order, and its coefficient unit is the
reciprocal of the interaction-matrix coefficient unit. No unit conversion is
performed.
"""
mutable struct SVDReconstructorProduct{T<:AbstractFloat}
    matrix::Matrix{T}
    measurement_singular_vectors::Matrix{T}
    corrector_singular_vectors::Matrix{T}
    singular_values::Vector{T}
    spectral_gains::Vector{T}
    diagnostics::SVDReconstructorDiagnostics{T}
end

"""Return the caller-owned corrector-by-measurement reconstructor matrix."""
reconstructor(result::SVDReconstructorProduct) = result.matrix

"""Return the compact-SVD left singular vectors in measurement coordinates."""
measurement_singular_vectors(result::SVDReconstructorProduct) =
    result.measurement_singular_vectors

"""Return the compact-SVD right singular vectors in corrector coordinates."""
corrector_singular_vectors(result::SVDReconstructorProduct) =
    result.corrector_singular_vectors

"""Return the compact-SVD singular values in descending order."""
singular_values(result::SVDReconstructorProduct) = result.singular_values

"""Return the applied compact-SVD inverse gain for every singular component."""
spectral_gains(result::SVDReconstructorProduct) = result.spectral_gains

"""Return the method-specific effective condition number."""
condition_number(result::SVDReconstructorProduct) = result.diagnostics.condition_number

"""Return the number of singular components classified as effective."""
effective_rank(result::SVDReconstructorProduct) = result.diagnostics.effective_rank

"""Return the total number of singular components discarded by TSVD."""
truncation_count(result::SVDReconstructorProduct) = result.diagnostics.truncation_count

function allocate_result(plan::SVDReconstructorPlan{T}) where {T}
    return SVDReconstructorProduct(
        zeros(T, plan.corrector_coordinate_count, plan.measurement_count),
        zeros(T, plan.measurement_count, plan.singular_count),
        zeros(T, plan.corrector_coordinate_count, plan.singular_count),
        zeros(T, plan.singular_count),
        zeros(T, plan.singular_count),
        SVDReconstructorDiagnostics{T}(T(Inf), 0, 0),
    )
end

allocate_workspace(::SVDReconstructorPlan) = nothing

function _validate_svd_product(plan, result)
    size(result.matrix) ==
        (plan.corrector_coordinate_count, plan.measurement_count) ||
        throw(DimensionMismatch(
            "SVD reconstructor result has an incompatible matrix extent",
        ))
    size(result.measurement_singular_vectors) ==
        (plan.measurement_count, plan.singular_count) ||
        throw(DimensionMismatch(
            "measurement_singular_vectors has an incompatible extent",
        ))
    size(result.corrector_singular_vectors) ==
        (plan.corrector_coordinate_count, plan.singular_count) ||
        throw(DimensionMismatch(
            "corrector_singular_vectors has an incompatible extent",
        ))
    length(result.singular_values) == plan.singular_count ||
        throw(DimensionMismatch("singular_values has an incompatible extent"))
    length(result.spectral_gains) == plan.singular_count ||
        throw(DimensionMismatch("spectral_gains has an incompatible extent"))
    products = (
        result.matrix,
        result.measurement_singular_vectors,
        result.corrector_singular_vectors,
        result.singular_values,
        result.spectral_gains,
    )
    for first_index in eachindex(products)
        for second_index in (first_index + 1):length(products)
            Base.mightalias(products[first_index], products[second_index]) &&
                throw(ArgumentError("SVD reconstructor products must not alias"))
        end
    end
    return nothing
end

function _spectral_inverse(
    ::ExactPseudoInverse,
    singular_values::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(singular_values)
    gains = Vector{T}(undef, length(singular_values))
    effective_rank = 0
    for index in eachindex(singular_values)
        value = singular_values[index]
        if iszero(value)
            gains[index] = zero(T)
        else
            gains[index] = inv(value)
            effective_rank += 1
        end
    end
    condition = iszero(effective_rank) ? T(Inf) :
        first(singular_values) / singular_values[effective_rank]
    return gains, SVDReconstructorDiagnostics(condition, effective_rank, 0)
end

function _spectral_inverse(
    method::TSVDInverse{T},
    singular_values::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(singular_values)
    cutoff = max(method.atol, method.rtol * first(singular_values))
    rank_by_tolerance = count(>(cutoff), singular_values)
    effective_rank = max(rank_by_tolerance - method.n_trunc, 0)
    gains = zeros(T, length(singular_values))
    for index in 1:effective_rank
        gains[index] = inv(singular_values[index])
    end
    condition = iszero(effective_rank) ? T(Inf) :
        first(singular_values) / singular_values[effective_rank]
    diagnostics = SVDReconstructorDiagnostics(
        condition,
        effective_rank,
        length(singular_values) - effective_rank,
    )
    return gains, diagnostics
end

function _spectral_inverse(
    method::TikhonovInverse{T},
    singular_values::AbstractVector{T},
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(singular_values)
    gains = [_tikhonov_gain(value, method.lambda) for value in singular_values]
    cutoff = max(method.atol, method.rtol * first(singular_values))
    effective_rank = count(>(cutoff), singular_values)
    denominator = max(last(singular_values), method.lambda)
    condition = iszero(denominator) ? T(Inf) : first(singular_values) / denominator
    return gains, SVDReconstructorDiagnostics(condition, effective_rank, 0)
end

"""
Prepare one complete unweighted compact-SVD inverse into caller-owned storage.

The finite input is copied into the plan's numeric representation before
factorization, so noncontiguous views and result/input overlap are supported.
Invalid input or non-finite calculations throw before the existing product is
changed. This is a cold host-side preparation operation and may allocate.
"""
function process!(
    result::SVDReconstructorProduct{T},
    ::Nothing,
    plan::SVDReconstructorPlan{T},
    input::SVDReconstructorInputs,
) where {T}
    _validate_svd_product(plan, result)
    interaction_matrix = _validated_matrix(
        T,
        input.interaction_matrix,
        (plan.measurement_count, plan.corrector_coordinate_count),
        "interaction_matrix",
    )
    factors = svd(interaction_matrix; full=false)
    values = Vector{T}(factors.S)
    all(isfinite, values) || throw(ArgumentError(
        "interaction-matrix SVD produced non-finite singular values",
    ))
    gains, diagnostics = _spectral_inverse(plan.method, values)
    matrix = factors.V * Diagonal(gains) * factors.U'
    all(isfinite, gains) && all(isfinite, matrix) &&
        all(isfinite, factors.U) && all(isfinite, factors.V) || throw(ArgumentError(
        "SVD reconstructor preparation produced non-finite values",
    ))
    copyto!(result.matrix, matrix)
    copyto!(result.measurement_singular_vectors, factors.U)
    copyto!(result.corrector_singular_vectors, factors.V)
    copyto!(result.singular_values, values)
    copyto!(result.spectral_gains, gains)
    result.diagnostics = diagnostics
    return result
end
