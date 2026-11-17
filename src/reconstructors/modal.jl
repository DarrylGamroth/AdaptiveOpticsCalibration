"""
    ModalReconstructor(owned_modes, fitted_modes; recovery_modes=Colon(), rtol)

Fit a truncated SVD to the specified interaction-matrix columns and retain
only the owned rows in the full modal reconstructor. Compute recovery against
the owned-mode selector. Evaluate its error on the square submatrix selected
by `recovery_modes` (all coordinates by default). The caller chooses the
modal partition, cutoff and acceptance thresholds.
"""
struct ModalReconstructor{O,F,R,T<:AbstractFloat} <: AbstractCalibrationMethod
    owned_modes::O
    fitted_modes::F
    recovery_modes::R
    rtol::T
end

ModalReconstructor(owned_modes, fitted_modes; recovery_modes=Colon(), rtol::AbstractFloat) =
    ModalReconstructor(owned_modes, fitted_modes, recovery_modes, rtol)

struct ModalReconstructorPlan{P,R} <: AbstractCalibrationPlan
    inverse::P
    measurements::Int
    modes::Int
    owned::Vector{Int}
    fitted::Vector{Int}
    unowned::Vector{Int}
    recovery_modes::R
end

function mode_indices(indices, modes)
    values = collect(Int, indices)
    !isempty(values) && all(index -> 1 <= index <= modes, values) &&
        allunique(values) || throw(ArgumentError("invalid mode indices"))
    return values
end

recovery_indices(::Colon, modes) = Colon()
recovery_indices(indices, modes) = mode_indices(indices, modes)

function prepare(method::ModalReconstructor, spec::ReconstructorSpecification{T}) where {T}
    measurements, modes = spec.measurement_count, spec.corrector_coordinate_count
    owned = mode_indices(method.owned_modes, modes)
    fitted = mode_indices(method.fitted_modes, modes)
    issubset(owned, fitted) || throw(ArgumentError("every owned mode must be fitted"))
    inverse = prepare(
        TSVDInverse(rtol=T(method.rtol)),
        ReconstructorSpecification(measurements, length(fitted), T),
    )
    return ModalReconstructorPlan(
        inverse, measurements, modes, owned, fitted, setdiff(1:modes, owned),
        recovery_indices(method.recovery_modes, modes),
    )
end

function process(
    plan::ModalReconstructorPlan,
    (interaction, half)::Tuple{AbstractMatrix{T},AbstractMatrix{T}},
) where {T<:AbstractFloat}
    size(interaction) == size(half) == (plan.measurements, plan.modes) ||
        throw(DimensionMismatch("full- and half-poke matrices must match the modal plan"))
    inverse = process(plan.inverse, SVDReconstructorInputs(interaction[:, plan.fitted]))
    matrix = zeros(T, plan.modes, plan.measurements)
    matrix[plan.fitted, :] .= reconstructor(inverse)
    matrix[plan.unowned, :] .= 0
    selection = zeros(T, plan.modes, plan.modes)
    for mode in plan.owned
        selection[mode, mode] = 1
    end
    recovery = matrix * interaction
    indices = plan.recovery_modes
    recovery_error = norm(recovery[indices, indices] - selection[indices, indices]) /
        sqrt(length(plan.owned))
    derivative_error = norm(interaction - half) / max(norm(half), eps(T))
    cross_talk = isempty(plan.unowned) ? 0.0 :
        norm(recovery[plan.owned, plan.unowned]) / sqrt(length(plan.owned))
    return (;
        reconstructor=matrix, selection, recovery, recovery_error, derivative_error, cross_talk,
        retained_rank=effective_rank(inverse), singular_values=singular_values(inverse),
    )
end

"""
    MeasurementNulling(owned_modes; rtol)

Subtract owned-mode command components from unowned probe columns to null
their measured response. Use a full-row-rank TSVD of the owned response and
normalize every resulting command column by its maximum absolute value.
"""
struct MeasurementNulling{O,T<:AbstractFloat} <: AbstractCalibrationMethod
    owned_modes::O
    rtol::T
end

MeasurementNulling(owned_modes; rtol::AbstractFloat) = MeasurementNulling(owned_modes, rtol)

struct MeasurementNullingPlan{P} <: AbstractCalibrationPlan
    inverse::P
    measurements::Int
    modes::Int
    owned::Vector{Int}
    unowned::Vector{Int}
end

function prepare(method::MeasurementNulling, spec::ReconstructorSpecification{T}) where {T}
    measurements, modes = spec.measurement_count, spec.corrector_coordinate_count
    owned = mode_indices(method.owned_modes, modes)
    unowned = setdiff(1:modes, owned)
    isempty(unowned) &&
        throw(ArgumentError("measurement nulling requires unowned command modes"))
    inverse = prepare(
        TSVDInverse(rtol=T(method.rtol)),
        ReconstructorSpecification(measurements, length(owned), T),
    )
    return MeasurementNullingPlan(inverse, measurements, modes, owned, unowned)
end

function process(
    plan::MeasurementNullingPlan,
    (commands, interaction)::Tuple{AbstractMatrix,AbstractMatrix{T}},
) where {T<:AbstractFloat}
    size(commands, 2) == plan.modes &&
        size(interaction) == (plan.measurements, plan.modes) ||
        throw(DimensionMismatch("command columns and response must match the nulling plan"))
    projection = Matrix{T}(commands)
    owned_response = interaction[:, plan.owned]
    unowned_response = interaction[:, plan.unowned]
    inverse = process(plan.inverse, SVDReconstructorInputs(owned_response))
    rank = effective_rank(inverse)
    rank == plan.measurements ||
        throw(ArgumentError("owned response must retain full row rank for measurement nulling"))
    coefficients = Matrix(reconstructor(inverse)) * unowned_response
    projection[:, plan.unowned] .-= projection[:, plan.owned] * coefficients
    scales = [maximum(abs, column) for column in eachcol(projection)]
    all(scale -> isfinite(scale) && scale > 0, scales) && all(isfinite, projection) ||
        throw(ArgumentError("measurement nulling produced null or nonfinite command columns"))
    projection ./= permutedims(scales)
    residual = norm(unowned_response - owned_response * coefficients)
    response_norm = norm(unowned_response)
    relative_null_residual =
        iszero(response_norm) ? (iszero(residual) ? 0.0 : Inf) : residual / response_norm
    return (;
        projection, coefficients, scales, relative_null_residual,
        owned_response, unowned_response,
        retained_rank=rank, owned_modes=copy(plan.owned), unowned_modes=copy(plan.unowned),
    )
end
