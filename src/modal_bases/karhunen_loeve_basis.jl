"""
    KarhunenLoeveBasis(; remove_piston=true, relative_rank_tolerance=nothing)

Select construction of a finite-corrector Karhunen–Loève basis. Given a
sample-by-corrector influence matrix `M`, pupil Gram matrix `G = MᴾM`, and
projected atmospheric phase covariance `C`, processing solves
`C v = λ G v`. The published sampled modes are `M v`, are orthonormal over
the declared pupil support, and have diagonal atmospheric covariance.

When `remove_piston` is true, processing restricts the corrector span to modes
whose arithmetic mean over the pupil support is zero. This is a constraint in
the eigensolve; the sampled modes are never modified independently of their
modal-to-command vectors.

`relative_rank_tolerance` determines the retained numerical rank of the pupil
Gram matrix. The default is `sqrt(eps(T))` for the specification's floating-
point type. Repeated atmospheric variances define a subspace rather than
portable individual mode identities.
"""
struct KarhunenLoeveBasis{T<:AbstractFloat} <: AbstractModalBasisMethod
    remove_piston::Bool
    relative_rank_tolerance::Union{Nothing,T}
end

function KarhunenLoeveBasis(;
    remove_piston::Bool=true,
    relative_rank_tolerance=nothing,
)
    if relative_rank_tolerance === nothing
        return KarhunenLoeveBasis{Float64}(remove_piston, nothing)
    end
    relative_rank_tolerance isa AbstractFloat || throw(ArgumentError(
        "relative_rank_tolerance must be an AbstractFloat or nothing",
    ))
    tolerance = relative_rank_tolerance
    T = typeof(tolerance)
    isfinite(tolerance) && tolerance > zero(T) ||
        throw(ArgumentError("relative_rank_tolerance must be positive and finite"))
    return KarhunenLoeveBasis{T}(remove_piston, tolerance)
end

"""
    KarhunenLoeveBasisSpecification(sample_count, corrector_coordinate_count,
                                    mode_count, pupil_support, T=Float64)

Declare the dimensions, pupil support, and host floating-point representation
of a finite-corrector atmospheric Karhunen–Loève construction. Samples are rows
and corrector coordinates are columns. `pupil_support` contains one Boolean per
sample in Julia column-major flattened order and must be nonempty.
"""
struct KarhunenLoeveBasisSpecification{T<:AbstractFloat}
    sample_count::Int
    corrector_coordinate_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}

    function KarhunenLoeveBasisSpecification{T}(
        sample_count::Integer,
        corrector_coordinate_count::Integer,
        mode_count::Integer,
        pupil_support::AbstractVector{Bool},
    ) where {T<:AbstractFloat}
        T === Float32 || T === Float64 || throw(ArgumentError(
            "Karhunen–Loève eigensolves support Float32 and Float64",
        ))
        samples = Int(sample_count)
        correctors = Int(corrector_coordinate_count)
        modes = Int(mode_count)
        samples > 0 || throw(ArgumentError("sample_count must be positive"))
        correctors > 0 || throw(ArgumentError(
            "corrector_coordinate_count must be positive",
        ))
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
        _checked_extent("projected atmospheric covariance", correctors, correctors)
        _checked_extent("Karhunen–Loève sampled modes", samples, modes)
        _checked_extent("Karhunen–Loève modal-to-command matrix", correctors, modes)
        return new{T}(samples, correctors, modes, Vector{Bool}(pupil_support))
    end
end

function KarhunenLoeveBasisSpecification(
    sample_count::Integer,
    corrector_coordinate_count::Integer,
    mode_count::Integer,
    pupil_support::AbstractVector{Bool},
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return KarhunenLoeveBasisSpecification{T}(
        sample_count,
        corrector_coordinate_count,
        mode_count,
        pupil_support,
    )
end

"""Prepared dimensions, pupil support, rank policy, and piston policy."""
struct KarhunenLoeveBasisPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    sample_count::Int
    corrector_coordinate_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}
    supported_sample_count::Int
    remove_piston::Bool
    relative_rank_tolerance::T
end

function prepare(
    method::KarhunenLoeveBasis,
    specification::KarhunenLoeveBasisSpecification{T},
) where {T<:AbstractFloat}
    tolerance = method.relative_rank_tolerance === nothing ?
        sqrt(eps(T)) : T(method.relative_rank_tolerance)
    isfinite(tolerance) && tolerance > zero(T) || throw(ArgumentError(
        "relative_rank_tolerance must convert to a positive finite specification value",
    ))
    support = copy(specification.pupil_support)
    return KarhunenLoeveBasisPlan{T}(
        specification.sample_count,
        specification.corrector_coordinate_count,
        specification.mode_count,
        support,
        count(support),
        method.remove_piston,
        tolerance,
    )
end

"""
Inputs for one Karhunen–Loève construction. `sampled_influences` is
sample-by-corrector. `projected_atmospheric_covariance[i,j]` is
`mᵢᵀ W Cₒₚᴅ W mⱼ`, where `W` is the normalized pupil-sample measure
and `Cₒₚᴅ` is sampled atmospheric OPD covariance in square metres. Both
arrays must have the plan's exact floating-point element type.
"""
struct KarhunenLoeveBasisInputs{
    T<:AbstractFloat,
    M<:AbstractMatrix{T},
    C<:AbstractMatrix{T},
}
    sampled_influences::M
    projected_atmospheric_covariance::C
end

"""
Caller-owned Karhunen–Loève product. `atmospheric_variances` are ordered from
largest to smallest, `modal_to_command` contains corrector-coordinate modes,
and `sampled_modes` is the exact sample-by-mode product of the supplied
influences and `modal_to_command`.
"""
mutable struct KarhunenLoeveBasisProduct{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    C<:AbstractMatrix{T},
    S<:AbstractMatrix{T},
}
    atmospheric_variances::V
    modal_to_command::C
    sampled_modes::S
end

"""Return retained atmospheric modal variances in descending order."""
atmospheric_variances(result::KarhunenLoeveBasisProduct) = result.atmospheric_variances

"""Return the caller-owned corrector-coordinate modal-to-command matrix."""
modal_to_command(result::KarhunenLoeveBasisProduct) = result.modal_to_command

"""Return the caller-owned sample-by-mode Karhunen–Loève matrix."""
sampled_modes(result::KarhunenLoeveBasisProduct) = result.sampled_modes

"""Replaceable dense host scratch for a Karhunen–Loève eigensolve."""
struct KarhunenLoeveBasisWorkspace{
    T<:AbstractFloat,
    G<:StridedMatrix{T},
    C<:StridedMatrix{T},
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    S<:AbstractMatrix{T},
}
    pupil_gram::G
    projected_atmospheric_covariance::C
    piston_coupling::V
    atmospheric_variances::V
    modal_to_command::M
    sampled_modes::S
end

function allocate_result(plan::KarhunenLoeveBasisPlan{T}) where {T}
    return KarhunenLoeveBasisProduct(
        zeros(T, plan.mode_count),
        zeros(T, plan.corrector_coordinate_count, plan.mode_count),
        zeros(T, plan.sample_count, plan.mode_count),
    )
end

function allocate_workspace(plan::KarhunenLoeveBasisPlan{T}) where {T}
    correctors = plan.corrector_coordinate_count
    return KarhunenLoeveBasisWorkspace(
        zeros(T, correctors, correctors),
        zeros(T, correctors, correctors),
        zeros(T, correctors),
        zeros(T, plan.mode_count),
        zeros(T, correctors, plan.mode_count),
        zeros(T, plan.sample_count, plan.mode_count),
    )
end

function _validate_karhunen_loeve_storage(
    result::KarhunenLoeveBasisProduct,
    workspace::KarhunenLoeveBasisWorkspace,
    plan::KarhunenLoeveBasisPlan,
    inputs::KarhunenLoeveBasisInputs,
)
    arrays = (
        result.atmospheric_variances,
        result.modal_to_command,
        result.sampled_modes,
        workspace.pupil_gram,
        workspace.projected_atmospheric_covariance,
        workspace.piston_coupling,
        workspace.atmospheric_variances,
        workspace.modal_to_command,
        workspace.sampled_modes,
        inputs.sampled_influences,
        inputs.projected_atmospheric_covariance,
    )
    Base.require_one_based_indexing(arrays...)
    length(result.atmospheric_variances) == plan.mode_count ||
        throw(DimensionMismatch("atmospheric-variance result has an incompatible extent"))
    size(result.modal_to_command) ==
        (plan.corrector_coordinate_count, plan.mode_count) || throw(DimensionMismatch(
        "Karhunen–Loève modal-to-command result has an incompatible extent",
    ))
    size(result.sampled_modes) == (plan.sample_count, plan.mode_count) ||
        throw(DimensionMismatch("Karhunen–Loève sampled-mode result has an incompatible extent"))
    size(workspace.pupil_gram) ==
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count) ||
        throw(DimensionMismatch("pupil Gram workspace has an incompatible extent"))
    stride(workspace.pupil_gram, 1) == 1 || throw(ArgumentError(
        "pupil Gram workspace must have unit first-dimension stride",
    ))
    size(workspace.projected_atmospheric_covariance) == size(workspace.pupil_gram) ||
        throw(DimensionMismatch("atmospheric covariance workspace has an incompatible extent"))
    stride(workspace.projected_atmospheric_covariance, 1) == 1 || throw(ArgumentError(
        "atmospheric covariance workspace must have unit first-dimension stride",
    ))
    length(workspace.piston_coupling) == plan.corrector_coordinate_count ||
        throw(DimensionMismatch("piston-coupling workspace has an incompatible extent"))
    length(workspace.atmospheric_variances) == plan.mode_count ||
        throw(DimensionMismatch("atmospheric-variance workspace has an incompatible extent"))
    size(workspace.modal_to_command) == size(result.modal_to_command) ||
        throw(DimensionMismatch("modal-to-command workspace has an incompatible extent"))
    size(workspace.sampled_modes) == size(result.sampled_modes) ||
        throw(DimensionMismatch("sampled-mode workspace has an incompatible extent"))
    size(inputs.sampled_influences) ==
        (plan.sample_count, plan.corrector_coordinate_count) || throw(DimensionMismatch(
        "sampled influences have an incompatible extent",
    ))
    size(inputs.projected_atmospheric_covariance) ==
        (plan.corrector_coordinate_count, plan.corrector_coordinate_count) ||
        throw(DimensionMismatch("projected atmospheric covariance has an incompatible extent"))
    _any_pair_alias(arrays) && throw(ArgumentError(
        "Karhunen–Loève products, workspace, and inputs must not alias",
    ))
    return nothing
end

function _pupil_gram_and_piston!(
    gram::AbstractMatrix{T},
    piston_coupling::AbstractVector{T},
    sampled_influences::AbstractMatrix{T},
    plan::KarhunenLoeveBasisPlan{T},
) where {T<:AbstractFloat}
    fill!(gram, zero(T))
    fill!(piston_coupling, zero(T))
    @inbounds for sample in 1:plan.sample_count
        plan.pupil_support[sample] || continue
        for first_corrector in 1:plan.corrector_coordinate_count
            first_value = sampled_influences[sample, first_corrector]
            piston_coupling[first_corrector] += first_value
            for second_corrector in first_corrector:plan.corrector_coordinate_count
                gram[first_corrector, second_corrector] +=
                    first_value * sampled_influences[sample, second_corrector]
            end
        end
    end
    @inbounds for column in 1:plan.corrector_coordinate_count
        for row in (column + 1):plan.corrector_coordinate_count
            gram[row, column] = gram[column, row]
        end
    end
    gram ./= T(plan.supported_sample_count)
    return gram
end

function _symmetric_covariance_copy!(
    destination::AbstractMatrix{T},
    source::AbstractMatrix{T},
    relative_tolerance::T,
) where {T<:AbstractFloat}
    scale = max(maximum(abs, source), floatmin(T))
    maximum_asymmetry = zero(T)
    @inbounds for column in axes(source, 2), row in 1:column
        maximum_asymmetry = max(
            maximum_asymmetry,
            abs(source[row, column] - source[column, row]),
        )
    end
    maximum_asymmetry <= relative_tolerance * scale || throw(ArgumentError(
        "projected atmospheric covariance must be numerically symmetric",
    ))
    @inbounds for column in axes(source, 2)
        for row in 1:column
            value = (source[row, column] + source[column, row]) / T(2)
            destination[row, column] = value
            destination[column, row] = value
        end
    end
    return destination
end

"""
    process!(result, workspace, plan, inputs)

Construct one finite-corrector atmospheric Karhunen–Loève basis. This is a
cold host operation over `Float32` or `Float64` arrays and may allocate during
rank reduction and symmetric eigensolves. It validates all inputs and completes
the numerical construction in workspace before publishing `result`.
"""
function process!(
    result::KarhunenLoeveBasisProduct{T},
    workspace::KarhunenLoeveBasisWorkspace{T},
    plan::KarhunenLoeveBasisPlan{T},
    inputs::KarhunenLoeveBasisInputs{T},
) where {T<:AbstractFloat}
    _validate_karhunen_loeve_storage(result, workspace, plan, inputs)
    all(isfinite, inputs.sampled_influences) || throw(ArgumentError(
        "sampled influences must contain only finite values",
    ))
    all(isfinite, inputs.projected_atmospheric_covariance) || throw(ArgumentError(
        "projected atmospheric covariance must contain only finite values",
    ))

    _pupil_gram_and_piston!(
        workspace.pupil_gram,
        workspace.piston_coupling,
        inputs.sampled_influences,
        plan,
    )
    _symmetric_covariance_copy!(
        workspace.projected_atmospheric_covariance,
        inputs.projected_atmospheric_covariance,
        plan.relative_rank_tolerance,
    )
    gram_factorization = eigen(Symmetric(workspace.pupil_gram))
    all(isfinite, gram_factorization.values) &&
        all(isfinite, gram_factorization.vectors) || throw(ArgumentError(
        "pupil Gram eigensolve produced non-finite values",
    ))
    largest_gram_value = maximum(gram_factorization.values)
    largest_gram_value > zero(T) || throw(ArgumentError(
        "sampled corrector span has zero pupil norm",
    ))
    gram_threshold = plan.relative_rank_tolerance * largest_gram_value
    retained_gram_indices = findall(value -> value > gram_threshold,
        gram_factorization.values)
    length(retained_gram_indices) >= plan.mode_count || throw(DimensionMismatch(
        "mode_count exceeds the numerical rank of the piston-constrained corrector span",
    ))
    gram_vectors = gram_factorization.vectors[:, retained_gram_indices]
    inverse_sqrt_gram = Diagonal(
        inv.(sqrt.(gram_factorization.values[retained_gram_indices])),
    )
    whitener = gram_vectors * inverse_sqrt_gram

    if plan.remove_piston
        # Resolve the piston constraint only after rank reduction and pupil-
        # metric whitening. Each column of `whitener` then produces a sampled
        # mode with unit pupil RMS, so this coupling is a dimensionless pupil
        # mean. A threshold in these normalized coordinates cannot hide a
        # small raw coupling that is amplified by an ill-scaled corrector.
        normalized_piston_coupling =
            transpose(whitener) * workspace.piston_coupling /
            T(plan.supported_sample_count)
        piston_tolerance = sqrt(eps(T)) *
                           sqrt(T(length(normalized_piston_coupling)))
        if norm(normalized_piston_coupling) > piston_tolerance
            piston_free_coordinates = nullspace(
                reshape(normalized_piston_coupling, 1, :);
                atol=zero(T),
                rtol=zero(T),
            )
            size(piston_free_coordinates, 2) > 0 || throw(DimensionMismatch(
                "piston removal leaves no corrector-coordinate subspace",
            ))
            whitener = whitener * piston_free_coordinates
        end
    end
    size(whitener, 2) >= plan.mode_count || throw(DimensionMismatch(
        "mode_count exceeds the numerical rank of the piston-constrained corrector span",
    ))

    whitened_covariance = Symmetric(
        transpose(whitener) * workspace.projected_atmospheric_covariance * whitener,
    )
    covariance_factorization = eigen(whitened_covariance)
    all(isfinite, covariance_factorization.values) &&
        all(isfinite, covariance_factorization.vectors) || throw(ArgumentError(
        "atmospheric covariance eigensolve produced non-finite values",
    ))
    covariance_scale = maximum(abs, covariance_factorization.values)
    covariance_tolerance = plan.relative_rank_tolerance *
                           max(covariance_scale, floatmin(T))
    minimum(covariance_factorization.values) >= -covariance_tolerance ||
        throw(ArgumentError("projected atmospheric covariance is not positive semidefinite"))

    @inbounds for mode in 1:plan.mode_count
        source = length(covariance_factorization.values) - mode + 1
        workspace.atmospheric_variances[mode] =
            max(covariance_factorization.values[source], zero(T))
        for corrector in 1:plan.corrector_coordinate_count
            value = zero(T)
            for whitened_coordinate in axes(whitener, 2)
                value += whitener[corrector, whitened_coordinate] *
                         covariance_factorization.vectors[whitened_coordinate, source]
            end
            workspace.modal_to_command[corrector, mode] = value
        end
    end
    mul!(workspace.sampled_modes, inputs.sampled_influences,
        workspace.modal_to_command)
    all(isfinite, workspace.atmospheric_variances) &&
        all(isfinite, workspace.modal_to_command) &&
        all(isfinite, workspace.sampled_modes) || throw(ArgumentError(
        "Karhunen–Loève construction produced non-finite values",
    ))

    copyto!(result.atmospheric_variances, workspace.atmospheric_variances)
    copyto!(result.modal_to_command, workspace.modal_to_command)
    copyto!(result.sampled_modes, workspace.sampled_modes)
    return result
end
