abstract type AbstractStructuredPushPull <: AbstractCalibrationMethod end

"""
    StructuredInteractionMatrixStatus

Classify one complete structured interaction-matrix estimation.
`StructuredNotProcessed` identifies newly allocated product storage;
`StructuredValid` identifies a published interaction matrix; the non-finite
statuses identify expected numerical failures that preserve the caller-owned
matrix. Invalid response shape, result/workspace storage, or storage aliasing
remains an exception.
"""
@enum StructuredInteractionMatrixStatus::UInt8 begin
    StructuredNotProcessed
    StructuredValid
    StructuredNonFiniteResponse
    StructuredNonFiniteCalculation
end

@doc "The product has not yet received an interaction-matrix estimate."
StructuredNotProcessed
@doc "The interaction matrix was estimated and published." StructuredValid
@doc "A response sample is non-finite; the interaction matrix was not modified."
StructuredNonFiniteResponse
@doc "Finite responses produced a non-finite calculation; the matrix was not modified."
StructuredNonFiniteCalculation

function _amplitude_vector(
    amplitudes::AbstractVector{T},
    operation::AbstractString,
) where {T<:Real}
    Base.require_one_based_indexing(amplitudes)
    isempty(amplitudes) && throw(ArgumentError(
        "$operation requires at least one corrector coordinate",
    ))
    F = _float_type(T, operation)
    values = Vector{F}(amplitudes)
    for amplitude in values
        isfinite(amplitude) && amplitude > zero(F) || throw(ArgumentError(
            "$operation amplitudes must be finite and positive",
        ))
    end
    return values
end

"""
    ZonalPushPull(amplitudes)

Select complete-cycle zonal push-pull interaction-matrix identification.

`amplitudes[i]` is the positive finite command amplitude for corrector
coordinate `i`. Preparation copies the amplitudes into the immutable plan.
"""
struct ZonalPushPull{T<:AbstractFloat} <: AbstractStructuredPushPull
    amplitudes::Vector{T}

    function ZonalPushPull{T}(amplitudes::Vector{T}) where {T<:AbstractFloat}
        values = _amplitude_vector(amplitudes, "zonal push-pull interaction-matrix identification")
        return new{T}(values)
    end
end

function ZonalPushPull(amplitudes::AbstractVector{T}) where {T<:Real}
    values = _amplitude_vector(amplitudes, "zonal push-pull interaction-matrix identification")
    return ZonalPushPull{eltype(values)}(values)
end

"""
    HadamardPushPull(amplitudes)

Select complete-cycle balanced Sylvester-Hadamard interaction-matrix identification.

The plan uses the smallest Sylvester order strictly greater than the number of
corrector coordinates and excludes the all-ones DC column.
"""
struct HadamardPushPull{T<:AbstractFloat} <: AbstractStructuredPushPull
    amplitudes::Vector{T}

    function HadamardPushPull{T}(amplitudes::Vector{T}) where {T<:AbstractFloat}
        values = _amplitude_vector(amplitudes, "Hadamard push-pull interaction-matrix identification")
        return new{T}(values)
    end
end

function HadamardPushPull(amplitudes::AbstractVector{T}) where {T<:Real}
    values = _amplitude_vector(amplitudes, "Hadamard push-pull interaction-matrix identification")
    return HadamardPushPull{eltype(values)}(values)
end

"""
    ResponseSpecification(measurement_count)

Describe the measurement-coordinate extent of one complete response cycle.
Response rows are ordered probe exposures and response columns are ordered
measurement coordinates.
"""
struct ResponseSpecification
    measurement_count::Int

    function ResponseSpecification(measurement_count::Integer)
        count = Int(measurement_count)
        count > 0 || throw(ArgumentError(
            "measurement_count must be positive",
        ))
        return new(count)
    end
end

"""Prepared zonal push-pull interaction-matrix plan."""
struct ZonalPushPullPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    response_row_count::Int
    amplitudes::Vector{T}
    reciprocal_scales::Vector{T}
end

"""Prepared balanced Sylvester-Hadamard push-pull interaction-matrix plan."""
struct HadamardPushPullPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int
    response_row_count::Int
    order::Int
    amplitudes::Vector{T}
    reciprocal_scales::Vector{T}
end

function _reciprocal_scales(amplitudes::AbstractVector{T}, normalization::T) where {T}
    return map(amplitudes) do amplitude
        scale = inv(normalization * amplitude)
        isfinite(scale) && scale > zero(T) || throw(ArgumentError(
            "amplitude reciprocal scales must be finite and positive",
        ))
        scale
    end
end

function prepare(
    method::ZonalPushPull{T},
    specification::ResponseSpecification,
) where {T}
    measurements = specification.measurement_count
    amplitudes = copy(method.amplitudes)
    correctors = length(amplitudes)
    response_rows = _checked_extent("zonal response", 2, correctors)
    _checked_extent("zonal response cycle", response_rows, measurements)
    _checked_extent("interaction matrix", measurements, correctors)
    return ZonalPushPullPlan(
        measurements,
        correctors,
        response_rows,
        amplitudes,
        _reciprocal_scales(amplitudes, T(2)),
    )
end

function prepare(
    method::HadamardPushPull{T},
    specification::ResponseSpecification,
) where {T}
    measurements = specification.measurement_count
    amplitudes = copy(method.amplitudes)
    correctors = length(amplitudes)
    order = _hadamard_order(correctors)
    order_scalar = T(order)
    order_scalar == order || throw(ArgumentError(
        "Hadamard order is not represented exactly by $T",
    ))
    response_rows = _checked_extent("Hadamard response", 2, order)
    _checked_extent("Hadamard response cycle", response_rows, measurements)
    _checked_extent("interaction matrix", measurements, correctors)
    return HadamardPushPullPlan(
        measurements,
        correctors,
        response_rows,
        order,
        amplitudes,
        _reciprocal_scales(amplitudes, T(2) * order_scalar),
    )
end

"""
    InteractionMatrixProduct(matrix)

Wrap caller-owned measurement-by-corrector interaction-matrix storage.
"""
mutable struct InteractionMatrixProduct{T<:AbstractFloat,M<:AbstractMatrix{T}}
    status::StructuredInteractionMatrixStatus
    matrix::M
end

InteractionMatrixProduct(matrix::M) where {T<:AbstractFloat,M<:AbstractMatrix{T}} =
    InteractionMatrixProduct{T,M}(StructuredNotProcessed, matrix)

"""Return the caller-owned matrix in an [`InteractionMatrixProduct`](@ref)."""
interaction_matrix(result::InteractionMatrixProduct) = result.matrix

"""Replaceable staged storage for atomic interaction-matrix estimation."""
struct StructuredWorkspace{T<:AbstractFloat,M<:AbstractMatrix{T}}
    staged_matrix::M
end

function allocate_result(
    plan::Union{ZonalPushPullPlan{T},HadamardPushPullPlan{T}},
) where {T}
    return InteractionMatrixProduct(
        zeros(T, plan.measurement_count, plan.corrector_coordinate_count),
    )
end

function allocate_workspace(
    plan::Union{ZonalPushPullPlan{T},HadamardPushPullPlan{T}},
) where {T}
    return StructuredWorkspace(
        zeros(T, plan.measurement_count, plan.corrector_coordinate_count),
    )
end

function _validate_structured_storage(
    output::AbstractMatrix{T},
    workspace::StructuredWorkspace{T},
    plan,
    responses::AbstractMatrix{T},
) where {T}
    Base.require_one_based_indexing(output, workspace.staged_matrix)
    expected_output = (plan.measurement_count, plan.corrector_coordinate_count)
    size(output) == expected_output || throw(DimensionMismatch(
        "interaction-matrix result must have shape $expected_output",
    ))
    size(workspace.staged_matrix) == expected_output || throw(DimensionMismatch(
        "structured workspace must have shape $expected_output",
    ))
    expected_responses = (plan.response_row_count, plan.measurement_count)
    size(responses) == expected_responses || throw(DimensionMismatch(
        "responses must have shape $expected_responses",
    ))
    Base.mightalias(output, workspace.staged_matrix) && throw(ArgumentError(
        "result matrix must not alias structured workspace",
    ))
    Base.mightalias(responses, workspace.staged_matrix) && throw(ArgumentError(
        "responses must not alias structured workspace",
    ))
    return nothing
end

function _structured_response_status(responses)
    for response in responses
        isfinite(response) || return StructuredNonFiniteResponse
    end
    return StructuredValid
end

function _publish!(destination::AbstractMatrix, source::AbstractMatrix)
    for (destination_column, source_column) in zip(axes(destination, 2), axes(source, 2))
        for (destination_row, source_row) in zip(axes(destination, 1), axes(source, 1))
            destination[destination_row, destination_column] =
                source[source_row, source_column]
        end
    end
    return destination
end

"""
    process!(output, workspace, plan::ZonalPushPullPlan, responses)

Estimate an interaction matrix from one complete zonal push-pull response cycle.

For response matrix `Y` and amplitudes `a`, the result matrix is
`D[j,i] = (Y[2i-1,j] - Y[2i,j]) / (2a[i])`. Response rows enumerate
associated exposures, response columns enumerate measurements, and result
columns enumerate corrector coordinates. The direct-output method returns
`StructuredValid` after publication or an expected non-finite status without
changing `output`. It does not update an `InteractionMatrixProduct` that also
refers to `output`; use the product overload when persistent product status is
required. Invalid validation contracts throw without changing the output.
"""
function process!(
    output::AbstractMatrix{T},
    workspace::StructuredWorkspace{T},
    plan::ZonalPushPullPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    _validate_structured_storage(output, workspace, plan, responses)
    response_status = _structured_response_status(responses)
    response_status === StructuredValid || return response_status
    response_rows = axes(responses, 1)
    response_measurements = axes(responses, 2)
    first_response_row = first(response_rows)
    for corrector_coordinate in 1:plan.corrector_coordinate_count
        positive_row = first_response_row + 2 * (corrector_coordinate - 1)
        negative_row = positive_row + 1
        scale = plan.reciprocal_scales[corrector_coordinate]
        for (measurement, response_measurement) in enumerate(response_measurements)
            difference = responses[positive_row, response_measurement] -
                         responses[negative_row, response_measurement]
            coefficient = difference * scale
            isfinite(difference) && isfinite(coefficient) ||
                return StructuredNonFiniteCalculation
            workspace.staged_matrix[measurement, corrector_coordinate] = coefficient
        end
    end
    _publish!(output, workspace.staged_matrix)
    return StructuredValid
end

function process!(
    result::InteractionMatrixProduct{T},
    workspace::StructuredWorkspace{T},
    plan::ZonalPushPullPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    result.status = process!(result.matrix, workspace, plan, responses)
    return result
end

"""
    process!(output, workspace, plan::HadamardPushPullPlan, responses)

Estimate an interaction matrix from one complete balanced Sylvester-Hadamard
push-pull response cycle.

For order `q`, response matrix `Y`, Sylvester signs `H`, and amplitudes `a`,
the result is
`D[j,i] = sum(H[r,i] * (Y[2r-1,j] - Y[2r,j]), r=1:q)/(2q*a[i])`.
The all-ones DC column is excluded. The direct-output method returns
`StructuredValid` after publication or an expected non-finite status without
changing `output`. It does not update an `InteractionMatrixProduct` that also
refers to `output`; use the product overload when persistent product status is
required. Invalid validation contracts throw without changing the output.
"""
function process!(
    output::AbstractMatrix{T},
    workspace::StructuredWorkspace{T},
    plan::HadamardPushPullPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    _validate_structured_storage(output, workspace, plan, responses)
    response_status = _structured_response_status(responses)
    response_status === StructuredValid || return response_status
    fill!(workspace.staged_matrix, zero(T))
    response_measurements = axes(responses, 2)
    first_response_row = first(axes(responses, 1))
    for row in 0:(plan.order - 1)
        positive_row = first_response_row + 2 * row
        negative_row = positive_row + 1
        for (measurement, response_measurement) in enumerate(response_measurements)
            difference = responses[positive_row, response_measurement] -
                         responses[negative_row, response_measurement]
            isfinite(difference) || return StructuredNonFiniteCalculation
            for corrector_coordinate in 1:plan.corrector_coordinate_count
                signed_difference = iseven(count_ones(row & corrector_coordinate)) ?
                    difference : -difference
                accumulation =
                    workspace.staged_matrix[measurement, corrector_coordinate] +
                    signed_difference
                isfinite(accumulation) || return StructuredNonFiniteCalculation
                workspace.staged_matrix[measurement, corrector_coordinate] = accumulation
            end
        end
    end
    for corrector_coordinate in 1:plan.corrector_coordinate_count
        scale = plan.reciprocal_scales[corrector_coordinate]
        for measurement in 1:plan.measurement_count
            coefficient = workspace.staged_matrix[measurement, corrector_coordinate] * scale
            isfinite(coefficient) || return StructuredNonFiniteCalculation
            workspace.staged_matrix[measurement, corrector_coordinate] = coefficient
        end
    end
    _publish!(output, workspace.staged_matrix)
    return StructuredValid
end

function process!(
    result::InteractionMatrixProduct{T},
    workspace::StructuredWorkspace{T},
    plan::HadamardPushPullPlan{T},
    responses::AbstractMatrix{T},
) where {T}
    result.status = process!(result.matrix, workspace, plan, responses)
    return result
end
