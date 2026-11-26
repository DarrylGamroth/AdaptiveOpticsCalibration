# Ellerbroek (1994) supplies the AO covariance-weighted reconstruction context. See docs/REFERENCES.md.
"""Dimensions and floating-point representation of a reconstructor result."""
struct ReconstructorSpecification{T<:AbstractFloat}
    measurement_count::Int
    corrector_coordinate_count::Int

    function ReconstructorSpecification{T}(
        measurement_count::Integer,
        corrector_coordinate_count::Integer,
    ) where {T<:AbstractFloat}
        measurements = Int(measurement_count)
        correctors = Int(corrector_coordinate_count)
        measurements > 0 || throw(ArgumentError(
            "measurement_count must be positive",
        ))
        correctors > 0 || throw(ArgumentError(
            "corrector_coordinate_count must be positive",
        ))
        _checked_extent("reconstructor", measurements, correctors)
        return new{T}(measurements, correctors)
    end
end

ReconstructorSpecification(
    measurement_count::Integer,
    corrector_coordinate_count::Integer,
    ::Type{T}=Float64,
) where {T<:AbstractFloat} = ReconstructorSpecification{T}(
    measurement_count,
    corrector_coordinate_count,
)

"""
    MAPReconstructor(; prior_weight=1, rtol=nothing, atol=0,
                     symmetry_rtol=nothing)

Select the zero-mean linear-Gaussian MAP estimator
`(D'Cn⁻¹D + αCc⁻¹)⁻¹D'Cn⁻¹`. `prior_weight` is `α` and must be positive.
Rank tolerances classify the whitened interaction matrix; MAP does not
truncate singular components. The default `rtol` is `sqrt(eps(T))`.
"""
struct MAPReconstructor{T<:AbstractFloat} <: AbstractCalibrationMethod
    prior_weight::T
    rank_rtol::Union{Nothing,T}
    rank_atol::T
    symmetry_rtol::Union{Nothing,T}
end

function MAPReconstructor(
    ; prior_weight::Real=1.0,
    rtol::Union{Nothing,Real}=nothing,
    atol::Real=0.0,
    symmetry_rtol::Union{Nothing,Real}=nothing,
)
    supplied = filter(!isnothing, (prior_weight, rtol, atol, symmetry_rtol))
    T = float(promote_type(map(typeof, supplied)...))
    return MAPReconstructor{T}(
        T(prior_weight),
        isnothing(rtol) ? nothing : T(rtol),
        T(atol),
        isnothing(symmetry_rtol) ? nothing : T(symmetry_rtol),
    )
end

"""
    StrokeWeightedTikhonov(retained_rank; tikhonov_scale=0,
                           rtol=nothing, atol=0, symmetry_rtol=nothing)

Select a retained-rank reconstructor regularized by the corrector stroke
metric `Wc`: `(D'Cn⁻¹D + λ²Wc)⁻¹D'Cn⁻¹` when all components are retained.
The default rank cutoff is `max(measurement_count,
corrector_coordinate_count) * eps(T) * maximum(singular_values)`, matching
the FGA preparation contract.
"""
struct StrokeWeightedTikhonov{T<:AbstractFloat} <: AbstractCalibrationMethod
    retained_rank::Int
    tikhonov_scale::T
    rank_rtol::Union{Nothing,T}
    rank_atol::T
    symmetry_rtol::Union{Nothing,T}
end

function StrokeWeightedTikhonov(
    retained_rank::Integer;
    tikhonov_scale::Real=0.0,
    rtol::Union{Nothing,Real}=nothing,
    atol::Real=0.0,
    symmetry_rtol::Union{Nothing,Real}=nothing,
)
    rank = Int(retained_rank)
    rank > 0 || throw(ArgumentError("retained_rank must be positive"))
    supplied = filter(!isnothing, (tikhonov_scale, rtol, atol, symmetry_rtol))
    T = float(promote_type(map(typeof, supplied)...))
    return StrokeWeightedTikhonov{T}(
        rank,
        T(tikhonov_scale),
        isnothing(rtol) ? nothing : T(rtol),
        T(atol),
        isnothing(symmetry_rtol) ? nothing : T(symmetry_rtol),
    )
end

"""MAP inputs in measurement-row and corrector-column order."""
struct MAPReconstructorInputs{
    D<:AbstractMatrix{<:Real},
    N<:AbstractMatrix{<:Real},
    C<:AbstractMatrix{<:Real},
}
    interaction_matrix::D
    measurement_noise_covariance::N
    command_prior_covariance::C
end

"""Stroke-weighted inputs in measurement-row and corrector-column order."""
struct StrokeWeightedReconstructorInputs{
    D<:AbstractMatrix{<:Real},
    N<:AbstractMatrix{<:Real},
    W<:AbstractMatrix{<:Real},
}
    interaction_matrix::D
    measurement_noise_covariance::N
    corrector_stroke_metric::W
end

abstract type AbstractRegularization{T<:AbstractFloat} end

struct PriorCovarianceRegularization{T<:AbstractFloat} <: AbstractRegularization{T}
    prior_weight::T
end

struct StrokeMetricRegularization{T<:AbstractFloat} <: AbstractRegularization{T}
    tikhonov_scale::T
end

"""Prepared dimensions, tolerances, and regularization semantics."""
struct ReconstructorPlan{
    T<:AbstractFloat,
    R<:AbstractRegularization{T},
} <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    singular_count::Int
    retained_rank::Int
    rank_rtol::T
    rank_atol::T
    symmetry_rtol::T
    regularization::R
end

function _validated_tolerances(
    ::Type{T},
    rtol,
    default_rtol,
    atol,
    symmetry_rtol,
) where {T}
    relative = isnothing(rtol) ? T(default_rtol) : T(rtol)
    absolute = T(atol)
    symmetry = isnothing(symmetry_rtol) ? sqrt(eps(T)) : T(symmetry_rtol)
    isfinite(relative) && relative >= zero(T) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(absolute) && absolute >= zero(T) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(symmetry) && zero(T) <= symmetry < one(T) ||
        throw(ArgumentError("symmetry_rtol must be finite and in [0,1)"))
    return relative, absolute, symmetry
end

function prepare(
    method::MAPReconstructor,
    specification::ReconstructorSpecification{T},
) where {T}
    α = T(method.prior_weight)
    isfinite(α) && α > zero(T) ||
        throw(ArgumentError("prior_weight must be finite and positive"))
    tolerances = _validated_tolerances(
        T,
        method.rank_rtol,
        sqrt(eps(T)),
        method.rank_atol,
        method.symmetry_rtol,
    )
    singular_count = min(
        specification.measurement_count,
        specification.corrector_coordinate_count,
    )
    return ReconstructorPlan(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        singular_count,
        singular_count,
        tolerances...,
        PriorCovarianceRegularization(α),
    )
end

function prepare(
    method::StrokeWeightedTikhonov,
    specification::ReconstructorSpecification{T},
) where {T}
    method.retained_rank > 0 || throw(ArgumentError("retained_rank must be positive"))
    λ = T(method.tikhonov_scale)
    isfinite(λ) && λ >= zero(T) ||
        throw(ArgumentError("tikhonov_scale must be finite and nonnegative"))
    tolerances = _validated_tolerances(
        T,
        method.rank_rtol,
        T(max(
            specification.measurement_count,
            specification.corrector_coordinate_count,
        )) * eps(T),
        method.rank_atol,
        method.symmetry_rtol,
    )
    singular_count = min(
        specification.measurement_count,
        specification.corrector_coordinate_count,
    )
    method.retained_rank <= singular_count || throw(DimensionMismatch(
        "retained_rank exceeds the interaction-matrix singular extent",
    ))
    return ReconstructorPlan(
        specification.measurement_count,
        specification.corrector_coordinate_count,
        singular_count,
        method.retained_rank,
        tolerances...,
        StrokeMetricRegularization(λ),
    )
end

"""Numerical diagnostics for one weighted reconstructor."""
struct ReconstructorDiagnostics{T<:AbstractFloat}
    numerical_rank::Int
    retained_rank::Int
    whitened_interaction_matrix_condition_number::T
    retained_condition_number::T
    whitened_interaction_matrix_residual::T
    full_reconstructor_residual::T
    regularization_metric_noise_amplification::T
    noise_covariance_symmetry_error::T
    regularization_matrix_symmetry_error::T
end

"""Caller-owned reconstructor matrix, spectral factors, and diagnostics."""
mutable struct ReconstructorProduct{T<:AbstractFloat}
    matrix::Matrix{T}
    singular_values::Vector{T}
    tikhonov_gains::Vector{T}
    diagnostics::ReconstructorDiagnostics{T}
end

"""Return the caller-owned corrector-by-measurement reconstructor matrix."""
reconstructor(result::ReconstructorProduct) = result.matrix

"""Return the singular values of the weighted interaction matrix."""
singular_values(result::ReconstructorProduct) = result.singular_values

"""Return all Tikhonov gains, including components omitted by rank truncation."""
tikhonov_gains(result::ReconstructorProduct) = result.tikhonov_gains

function _empty_diagnostics(::Type{T}, retained_rank) where {T}
    return ReconstructorDiagnostics{T}(
        0,
        retained_rank,
        T(Inf),
        T(Inf),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
    )
end

function allocate_result(plan::ReconstructorPlan{T}) where {T}
    return ReconstructorProduct(
        zeros(T, plan.corrector_coordinate_count, plan.measurement_count),
        zeros(T, plan.singular_count),
        zeros(T, plan.singular_count),
        _empty_diagnostics(T, plan.retained_rank),
    )
end

allocate_workspace(::ReconstructorPlan) = nothing

function _validated_matrix(::Type{T}, matrix, shape, name) where {T}
    size(matrix) == shape || throw(DimensionMismatch("$name must have shape $shape"))
    result = Matrix{T}(matrix)
    all(isfinite, result) || throw(ArgumentError("$name must contain only finite values"))
    return result
end

function _covariance_factor(matrix::Matrix{T}, tolerance::T, name) where {T}
    scale = norm(matrix)
    symmetry_error = iszero(scale) ? zero(T) : T(norm(matrix - matrix') / scale)
    symmetry_error <= tolerance || throw(ArgumentError(
        "$name must be symmetric within symmetry_rtol",
    ))
    symmetric = (matrix + matrix') / T(2)
    factor = try
        cholesky(Symmetric(symmetric))
    catch error
        error isa PosDefException || rethrow()
        throw(ArgumentError("$name must be positive definite"))
    end
    return factor, symmetry_error
end

@inline function _tikhonov_gain(σ::T, λ::T) where {T}
    iszero(λ) && return iszero(σ) ? zero(T) : inv(σ)
    if σ >= λ
        ratio = λ / σ
        return inv(σ * (one(T) + ratio * ratio))
    end
    ratio = σ / λ
    return ratio / (λ * (one(T) + ratio * ratio))
end

function _relative_residual(reference, approximation, ::Type{T}) where {T}
    denominator = norm(reference)
    return iszero(denominator) ? T(norm(approximation)) :
        T(norm(reference - approximation) / denominator)
end

function _validate_product(plan, result)
    size(result.matrix) ==
        (plan.corrector_coordinate_count, plan.measurement_count) ||
        throw(DimensionMismatch("reconstructor result has an incompatible matrix extent"))
    length(result.singular_values) == plan.singular_count ||
        throw(DimensionMismatch("singular_values has an incompatible extent"))
    length(result.tikhonov_gains) == plan.singular_count ||
        throw(DimensionMismatch("tikhonov_gains has an incompatible extent"))
    return nothing
end

function _prepare_weighting(plan::ReconstructorPlan{T}, input) where {T}
    D = _validated_matrix(
        T,
        input.interaction_matrix,
        (plan.measurement_count, plan.corrector_coordinate_count),
        "interaction_matrix",
    )
    Cn = _validated_matrix(
        T,
        input.measurement_noise_covariance,
        (plan.measurement_count, plan.measurement_count),
        "measurement_noise_covariance",
    )
    regularization_matrix, regularization_name = _regularization_matrix(plan, input)
    Cn_factor, noise_error = _covariance_factor(
        Cn,
        plan.symmetry_rtol,
        "measurement_noise_covariance",
    )
    regularization_factor, regularization_error = _covariance_factor(
        regularization_matrix,
        plan.symmetry_rtol,
        regularization_name,
    )
    return D, Cn_factor, regularization_factor, noise_error, regularization_error
end

function _regularization_matrix(
    plan::ReconstructorPlan{T,<:PriorCovarianceRegularization},
    input::MAPReconstructorInputs,
) where {T}
    return _validated_matrix(
        T,
        input.command_prior_covariance,
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count),
        "command_prior_covariance",
    ), "command_prior_covariance"
end

function _regularization_matrix(
    plan::ReconstructorPlan{T,<:StrokeMetricRegularization},
    input::StrokeWeightedReconstructorInputs,
) where {T}
    return _validated_matrix(
        T,
        input.corrector_stroke_metric,
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count),
        "corrector_stroke_metric",
    ), "corrector_stroke_metric"
end

function _weighted_system(
    plan::ReconstructorPlan{T,<:PriorCovarianceRegularization},
    D,
    noise_factor,
    prior_factor,
) where {T}
    return noise_factor.L \ (D * prior_factor.L), sqrt(plan.regularization.prior_weight)
end

function _weighted_system(
    plan::ReconstructorPlan{T,<:StrokeMetricRegularization},
    D,
    noise_factor,
    stroke_factor,
) where {T}
    return noise_factor.L \ D / stroke_factor.U,
        plan.regularization.tikhonov_scale
end

function _restore_command_coordinates(
    ::ReconstructorPlan{T,<:PriorCovarianceRegularization},
    weighted_reconstructor,
    prior_factor,
) where {T}
    return prior_factor.L * weighted_reconstructor
end

function _restore_command_coordinates(
    ::ReconstructorPlan{T,<:StrokeMetricRegularization},
    weighted_reconstructor,
    stroke_factor,
) where {T}
    return stroke_factor.U \ weighted_reconstructor
end

function _process_reconstructor!(result, plan::ReconstructorPlan{T}, input) where {T}
    _validate_product(plan, result)
    D, noise_factor, regularization_factor, noise_error, regularization_error =
        _prepare_weighting(plan, input)
    whitened, λ = _weighted_system(
        plan,
        D,
        noise_factor,
        regularization_factor,
    )
    factors = svd(whitened; full=false)
    σ = Vector{T}(factors.S)
    all(isfinite, σ) || throw(ArgumentError(
        "weighted interaction-matrix SVD produced non-finite singular values",
    ))
    gains = [_tikhonov_gain(value, λ) for value in σ]
    cutoff = isempty(σ) ? plan.rank_atol :
        max(plan.rank_atol, plan.rank_rtol * first(σ))
    numerical_rank = count(>(cutoff), σ)
    retained = 1:plan.retained_rank
    retained_response = factors.U[:, retained] * Diagonal(σ[retained]) *
        factors.Vt[retained, :]
    weighted_matrix = factors.V[:, retained] * Diagonal(gains[retained]) *
        factors.U[:, retained]' / noise_factor.L
    weighted_full_matrix = factors.V * Diagonal(gains) * factors.U' / noise_factor.L
    matrix = _restore_command_coordinates(
        plan,
        weighted_matrix,
        regularization_factor,
    )
    full_matrix = _restore_command_coordinates(
        plan,
        weighted_full_matrix,
        regularization_factor,
    )
    metric_noise_amplification = plan.regularization isa PriorCovarianceRegularization ?
        T(norm(Diagonal(gains))) :
        T(norm(regularization_factor.U * matrix * noise_factor.L))
    condition_number = numerical_rank == length(σ) && numerical_rank > 0 ?
        T(first(σ) / last(σ)) : T(Inf)
    retained_condition = plan.retained_rank <= numerical_rank ?
        T(first(σ) / σ[plan.retained_rank]) : T(Inf)
    diagnostics = ReconstructorDiagnostics{T}(
        numerical_rank,
        plan.retained_rank,
        condition_number,
        retained_condition,
        _relative_residual(whitened, retained_response, T),
        _relative_residual(full_matrix, matrix, T),
        metric_noise_amplification,
        noise_error,
        regularization_error,
    )
    all(isfinite, matrix) && all(isfinite, gains) || throw(ArgumentError(
        "reconstructor preparation produced non-finite values",
    ))
    for (name, value) in (
        ("whitened_interaction_matrix_residual",
            diagnostics.whitened_interaction_matrix_residual),
        ("full_reconstructor_residual", diagnostics.full_reconstructor_residual),
        ("regularization_metric_noise_amplification",
            diagnostics.regularization_metric_noise_amplification),
        ("noise_covariance_symmetry_error",
            diagnostics.noise_covariance_symmetry_error),
        ("regularization_matrix_symmetry_error",
            diagnostics.regularization_matrix_symmetry_error),
    )
        isfinite(value) || throw(ArgumentError(
            "reconstructor preparation produced non-finite $name",
        ))
    end
    copyto!(result.matrix, matrix)
    copyto!(result.singular_values, σ)
    copyto!(result.tikhonov_gains, gains)
    result.diagnostics = diagnostics
    return result
end

"""Prepare a complete MAP reconstructor result from covariance inputs."""
function process!(
    result::ReconstructorProduct{T},
    ::Nothing,
    plan::ReconstructorPlan{T,<:PriorCovarianceRegularization},
    input::MAPReconstructorInputs,
) where {T}
    return _process_reconstructor!(result, plan, input)
end

"""Prepare a retained-rank, stroke-weighted Tikhonov reconstructor result."""
function process!(
    result::ReconstructorProduct{T},
    ::Nothing,
    plan::ReconstructorPlan{T,<:StrokeMetricRegularization},
    input::StrokeWeightedReconstructorInputs,
) where {T}
    return _process_reconstructor!(result, plan, input)
end
