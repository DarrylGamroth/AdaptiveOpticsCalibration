# This basis diagonalizes the sampled influence Gram matrix. See docs/REFERENCES.md.
"""Abstract method family for construction of a modal basis."""
abstract type AbstractModalBasisMethod <: AbstractCalibrationMethod end

"""
    InfluenceFunctionEigenbasis(; remove_piston=true)

Select construction of a sampled influence-function eigenbasis. Preparation
forms the unnormalized corrector Gram matrix `G = MᵀM`, orders its symmetric
eigenpairs by descending eigenvalue, and publishes the retained corrector
eigenvectors and sampled modes. This is not an atmospheric Karhunen–Loève
basis: no atmospheric covariance or sample-count normalization is present.

When `remove_piston` is true, the arithmetic mean of every sampled mode over
the explicit pupil support is subtracted from every sample. No weighting,
masking, sign convention, or further normalization is applied.

Repeated eigenvalues define a subspace, not portable individual eigenvectors.
A retained mode count that splits a repeated-eigenvalue cluster likewise has
no portable retained-subspace identity.
"""
struct InfluenceFunctionEigenbasis <: AbstractModalBasisMethod
    remove_piston::Bool
end

InfluenceFunctionEigenbasis(; remove_piston::Bool=true) =
    InfluenceFunctionEigenbasis(remove_piston)

"""
    SampledInfluenceBasisSpecification(sample_count, corrector_coordinate_count,
                                       mode_count, pupil_support, T=Float64)

Declare a sampled-influence matrix with samples in rows and corrector
coordinates in columns. `pupil_support` has one Boolean per flattened sample
in Julia column-major order and must contain at least one `true` element.
`T` is the host floating-point representation required at execution.
"""
struct SampledInfluenceBasisSpecification{T<:AbstractFloat}
    sample_count::Int
    corrector_coordinate_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}

    function SampledInfluenceBasisSpecification{T}(
        sample_count::Integer,
        corrector_coordinate_count::Integer,
        mode_count::Integer,
        pupil_support::AbstractVector{Bool},
    ) where {T<:AbstractFloat}
        T === Float32 || T === Float64 || throw(ArgumentError(
            "sampled influence-function eigensolves support Float32 and Float64",
        ))
        samples = Int(sample_count)
        correctors = Int(corrector_coordinate_count)
        modes = Int(mode_count)
        samples > 0 || throw(ArgumentError("sample_count must be positive"))
        correctors > 0 || throw(ArgumentError("corrector_coordinate_count must be positive"))
        modes > 0 || throw(ArgumentError("mode_count must be positive"))
        modes <= correctors || throw(DimensionMismatch(
            "mode_count must not exceed corrector_coordinate_count",
        ))
        Base.require_one_based_indexing(pupil_support)
        length(pupil_support) == samples || throw(DimensionMismatch(
            "pupil_support must have one entry per sampled pupil coordinate",
        ))
        any(pupil_support) || throw(ArgumentError("pupil_support must be nonempty"))
        _checked_extent("sampled influence matrix", samples, correctors)
        _checked_extent("sampled modal basis", samples, modes)
        _checked_extent("modal-to-command matrix", correctors, modes)
        _checked_extent("sampled influence Gram matrix", correctors, correctors)
        return new{T}(samples, correctors, modes, Vector{Bool}(pupil_support))
    end
end

function SampledInfluenceBasisSpecification(
    sample_count::Integer,
    corrector_coordinate_count::Integer,
    mode_count::Integer,
    pupil_support::AbstractVector{Bool},
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return SampledInfluenceBasisSpecification{T}(
        sample_count,
        corrector_coordinate_count,
        mode_count,
        pupil_support,
    )
end

"""Prepared dimensions, pupil support, and sampled-basis method."""
struct InfluenceFunctionEigenbasisPlan{
    T<:AbstractFloat,
    M<:AbstractModalBasisMethod,
} <: AbstractCalibrationPlan
    sample_count::Int
    corrector_coordinate_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}
    supported_sample_count::Int
    method::M
end

function prepare(
    method::InfluenceFunctionEigenbasis,
    specification::SampledInfluenceBasisSpecification{T},
) where {T}
    support = copy(specification.pupil_support)
    return InfluenceFunctionEigenbasisPlan{T,typeof(method)}(
        specification.sample_count,
        specification.corrector_coordinate_count,
        specification.mode_count,
        support,
        count(support),
        method,
    )
end

"""
Caller-owned sampled influence-function eigenbasis product. `modal_to_command`
contains dimensionless, Euclidean-orthonormal corrector-coordinate
eigenvectors. `sampled_modes` is sample-by-mode in the unchanged sampled
influence coefficient unit. Gram eigenvalues have its squared unit.
"""
mutable struct SampledInfluenceEigenbasisProduct{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    C<:AbstractMatrix{T},
    S<:AbstractMatrix{T},
}
    gram_eigenvalues::V
    modal_to_command::C
    sampled_modes::S
end

"""Return the retained Gram eigenvalues in descending order."""
gram_eigenvalues(result::SampledInfluenceEigenbasisProduct) = result.gram_eigenvalues

"""Return the caller-owned corrector-coordinate modal-to-command matrix."""
modal_to_command(result::SampledInfluenceEigenbasisProduct) = result.modal_to_command

"""Return the caller-owned sample-by-mode matrix."""
sampled_modes(result::SampledInfluenceEigenbasisProduct) = result.sampled_modes

"""
Replaceable dense host scratch for sampled influence-function eigensolves.
The Gram matrix must be strided host storage with unit stride in its first
dimension, as required by the LAPACK symmetric eigensolver.
"""
struct SampledInfluenceEigenbasisWorkspace{
    T<:AbstractFloat,
    G<:StridedMatrix{T},
    V<:AbstractVector{T},
    C<:AbstractMatrix{T},
    S<:AbstractMatrix{T},
}
    gram::G
    gram_eigenvalues::V
    modal_to_command::C
    sampled_modes::S
end

function allocate_result(plan::InfluenceFunctionEigenbasisPlan{T}) where {T}
    return SampledInfluenceEigenbasisProduct(
        zeros(T, plan.mode_count),
        zeros(T, plan.corrector_coordinate_count, plan.mode_count),
        zeros(T, plan.sample_count, plan.mode_count),
    )
end

function allocate_workspace(plan::InfluenceFunctionEigenbasisPlan{T}) where {T}
    return SampledInfluenceEigenbasisWorkspace(
        zeros(T, plan.corrector_coordinate_count, plan.corrector_coordinate_count),
        zeros(T, plan.corrector_coordinate_count),
        zeros(T, plan.corrector_coordinate_count, plan.mode_count),
        zeros(T, plan.sample_count, plan.mode_count),
    )
end

function _validate_eigenbasis_product(
    result::SampledInfluenceEigenbasisProduct,
    workspace::SampledInfluenceEigenbasisWorkspace,
    plan,
    influence_functions,
)
    Base.require_one_based_indexing(
        result.gram_eigenvalues,
        result.modal_to_command,
        result.sampled_modes,
        workspace.gram,
        workspace.gram_eigenvalues,
        workspace.modal_to_command,
        workspace.sampled_modes,
        influence_functions,
    )
    length(result.gram_eigenvalues) == plan.mode_count || throw(DimensionMismatch(
        "Gram-eigenvalue result has an incompatible extent",
    ))
    size(result.modal_to_command) ==
        (plan.corrector_coordinate_count, plan.mode_count) || throw(DimensionMismatch(
        "modal-to-command result has an incompatible extent",
    ))
    size(result.sampled_modes) == (plan.sample_count, plan.mode_count) ||
        throw(DimensionMismatch("sampled-mode result has an incompatible extent"))
    size(workspace.gram) ==
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count) ||
        throw(DimensionMismatch("sampled-basis Gram workspace has an incompatible extent"))
    stride(workspace.gram, 1) == 1 || throw(ArgumentError(
        "sampled-basis Gram workspace must have unit first-dimension stride",
    ))
    length(workspace.gram_eigenvalues) == plan.corrector_coordinate_count ||
        throw(DimensionMismatch("sampled-basis eigenvalue workspace has an incompatible extent"))
    size(workspace.modal_to_command) ==
        (plan.corrector_coordinate_count, plan.mode_count) ||
        throw(DimensionMismatch("modal-to-command workspace has an incompatible extent"))
    size(workspace.sampled_modes) == (plan.sample_count, plan.mode_count) ||
        throw(DimensionMismatch("sampled-mode workspace has an incompatible extent"))
    size(influence_functions) ==
        (plan.sample_count, plan.corrector_coordinate_count) || throw(DimensionMismatch(
        "sampled influence functions must have shape $((plan.sample_count, plan.corrector_coordinate_count))",
    ))
    storage = (
        result.gram_eigenvalues,
        result.modal_to_command,
        result.sampled_modes,
        workspace.gram,
        workspace.gram_eigenvalues,
        workspace.modal_to_command,
        workspace.sampled_modes,
        influence_functions,
    )
    _any_pair_alias(storage) && throw(ArgumentError(
        "sampled influence-function products, workspace, and input must not alias",
    ))
    return nothing
end

function _copy_sampled_influence_gram!(
    gram::StridedMatrix{T},
    influence_functions::AbstractMatrix{T},
    plan,
) where {T<:AbstractFloat}
    all(isfinite, influence_functions) || throw(ArgumentError(
        "sampled influence functions must contain only finite values",
    ))
    mul!(gram, transpose(influence_functions), influence_functions)
    all(isfinite, gram) || throw(ArgumentError(
        "sampled influence-function Gram matrix must be finite",
    ))
    return gram
end

function _sampled_modes!(
    destination::AbstractMatrix{T},
    influence_functions::AbstractMatrix{T},
    modal_to_command::AbstractMatrix{T},
    plan::InfluenceFunctionEigenbasisPlan{T},
) where {T}
    for mode in 1:plan.mode_count
        for sample in 1:plan.sample_count
            accumulation = zero(T)
            for corrector in 1:plan.corrector_coordinate_count
                accumulation += influence_functions[sample, corrector] *
                                modal_to_command[corrector, mode]
            end
            isfinite(accumulation) || throw(ArgumentError(
                "sampled influence-function eigenbasis produced non-finite modes",
            ))
            destination[sample, mode] = accumulation
        end
        if plan.method.remove_piston
            mean_value = zero(T)
            for sample in 1:plan.sample_count
                plan.pupil_support[sample] && (mean_value += destination[sample, mode])
            end
            mean_value /= T(plan.supported_sample_count)
            isfinite(mean_value) || throw(ArgumentError(
                "sampled influence-function piston removal produced a non-finite mean",
            ))
            for sample in 1:plan.sample_count
                value = destination[sample, mode] - mean_value
                isfinite(value) || throw(ArgumentError(
                    "sampled influence-function piston removal produced non-finite modes",
                ))
                destination[sample, mode] = value
            end
        end
    end
    return destination
end

"""
    process!(result, workspace, plan, influence_functions)

Construct one sampled influence-function eigenbasis. `influence_functions` is
sample-by-corrector and must have the plan's exact floating-point element type.
All input, result, and workspace arrays must use one-based axes. This cold host
CPU operation supports `Float32` and `Float64` and may allocate during its
dense symmetric eigensolve. It publishes only after all validation and
numerical work succeed.
"""
function process!(
    result::SampledInfluenceEigenbasisProduct{T},
    workspace::SampledInfluenceEigenbasisWorkspace{T},
    plan::InfluenceFunctionEigenbasisPlan{T},
    influence_functions::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _validate_eigenbasis_product(result, workspace, plan, influence_functions)
    _copy_sampled_influence_gram!(workspace.gram, influence_functions, plan)
    factorization = eigen!(Symmetric(workspace.gram))
    all(isfinite, factorization.values) && all(isfinite, factorization.vectors) ||
        throw(ArgumentError("sampled influence-function eigensolve produced non-finite values"))
    for mode in 1:plan.mode_count
        source = plan.corrector_coordinate_count - mode + 1
        workspace.gram_eigenvalues[mode] = factorization.values[source]
        for corrector in 1:plan.corrector_coordinate_count
            workspace.modal_to_command[corrector, mode] =
                factorization.vectors[corrector, source]
        end
    end
    _sampled_modes!(
        workspace.sampled_modes,
        influence_functions,
        workspace.modal_to_command,
        plan,
    )
    copyto!(result.gram_eigenvalues, @view workspace.gram_eigenvalues[1:plan.mode_count])
    copyto!(result.modal_to_command, workspace.modal_to_command)
    copyto!(result.sampled_modes, workspace.sampled_modes)
    return result
end
