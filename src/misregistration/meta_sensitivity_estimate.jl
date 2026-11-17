"""Select complete-data meta-sensitivity offset estimation."""
struct MetaSensitivityEstimate <: AbstractCalibrationMethod end

"""
    MetaSensitivityEstimateSpecification(reference_response, meta_sensitivity,
                                          field_order, field_units)

Describe a complete meta-sensitivity estimate. `reference_response` is D₀ in
measurement-row, command-column order. `meta_sensitivity` is J with
`vec(reference_response)` row order and one column for each ordered parameter.
`field_order` and `field_units` are copied metadata; this module does not
interpret or convert them.
"""
struct MetaSensitivityEstimateSpecification{
    T<:AbstractFloat,
    D<:AbstractMatrix{T},
    J<:AbstractMatrix{T},
    O<:Tuple{Vararg{Symbol}},
    U<:Tuple{Vararg{String}},
}
    reference_response::D
    meta_sensitivity::J
    field_order::O
    field_units::U
end

function MetaSensitivityEstimateSpecification(
    reference_response::D,
    meta_sensitivity::J,
    field_order::AbstractVector{Symbol},
    field_units::AbstractVector{<:AbstractString},
) where {T<:AbstractFloat,D<:AbstractMatrix{T},J<:AbstractMatrix{T}}
    return MetaSensitivityEstimateSpecification(
        reference_response,
        meta_sensitivity,
        Tuple(field_order),
        Tuple(String.(field_units)),
    )
end

"""Bundle one observed response matrix D without copying it."""
struct MetaSensitivityEstimateInputs{T<:AbstractFloat,M<:AbstractMatrix{T}}
    observed_response::M
end

"""Rank and conditioning diagnostics of the prepared TSVD inverse."""
struct MetaSensitivityEstimateDiagnostics{T<:AbstractFloat}
    condition_number::T
    effective_rank::Int
end

"""
Prepared complete-data meta-sensitivity contract.

The plan owns copies of D₀, J, and J⁺. `field_order` and `field_units` retain
their supplied order without a physical interpretation or unit conversion.
"""
struct MetaSensitivityEstimatePlan{
    T<:AbstractFloat,
    D<:AbstractMatrix{T},
    J<:AbstractMatrix{T},
    I<:AbstractMatrix{T},
    M<:Reconstructors.TSVDInverse,
    O<:Tuple{Vararg{Symbol}},
    U<:Tuple{Vararg{String}},
} <: AbstractCalibrationPlan
    response_rows::Int
    response_columns::Int
    response_count::Int
    field_count::Int
    reference_response::D
    meta_sensitivity::J
    inverse::I
    inverse_method::M
    field_order::O
    field_units::U
    diagnostics::MetaSensitivityEstimateDiagnostics{T}
end

"""
Caller-owned raw ordered parameter offsets and inverse diagnostics.

`parameter_offsets` is δ from `J⁺ * vec(D - D₀)`. It is not rounded, scaled,
added to a physical zero point, or interpreted as a physical registration.
"""
mutable struct MetaSensitivityEstimateProduct{T<:AbstractFloat,V<:AbstractVector{T}}
    parameter_offsets::V
    diagnostics::MetaSensitivityEstimateDiagnostics{T}
end

"""Return the caller-owned ordered raw parameter offsets δ."""
parameter_offsets(result::MetaSensitivityEstimateProduct) =
    result.parameter_offsets

"""Return the prepared TSVD condition number for a meta-sensitivity product."""
condition_number(result::MetaSensitivityEstimateProduct) =
    result.diagnostics.condition_number

"""Return the prepared TSVD effective rank for a meta-sensitivity product."""
effective_rank(result::MetaSensitivityEstimateProduct) =
    result.diagnostics.effective_rank

"""Replaceable scratch for response differencing and candidate offsets."""
struct MetaSensitivityEstimateWorkspace{T<:AbstractFloat,V<:AbstractVector{T}}
    response_difference::V
    candidate_offsets::V
end

function _validate_meta_sensitivity_specification(specification)
    Base.require_one_based_indexing(
        specification.reference_response,
        specification.meta_sensitivity,
    )
    response_rows, response_columns = size(specification.reference_response)
    response_rows > 0 || throw(ArgumentError("reference_response must have positive row count"))
    response_columns > 0 || throw(ArgumentError("reference_response must have positive column count"))
    response_count = _checked_extent(
        "meta-sensitivity response",
        response_rows,
        response_columns,
    )
    field_count = size(specification.meta_sensitivity, 2)
    field_count > 0 || throw(ArgumentError("meta_sensitivity must have at least one field column"))
    size(specification.meta_sensitivity, 1) == response_count || throw(DimensionMismatch(
        "meta_sensitivity row count must equal length(reference_response)",
    ))
    length(specification.field_order) == field_count || throw(DimensionMismatch(
        "field_order length must equal meta_sensitivity column count",
    ))
    length(specification.field_units) == field_count || throw(DimensionMismatch(
        "field_units length must equal meta_sensitivity column count",
    ))
    all(isfinite, specification.reference_response) || throw(ArgumentError(
        "reference_response must contain only finite values",
    ))
    all(isfinite, specification.meta_sensitivity) || throw(ArgumentError(
        "meta_sensitivity must contain only finite values",
    ))
    return response_rows, response_columns, response_count, field_count
end

"""
    prepare(MetaSensitivityEstimate(), specification)

Copy D₀ and J, then prepare the source-parity TSVD inverse J⁺ with
`rtol = √eps(T)`, zero absolute tolerance, and no additional truncation.
Preparation is CPU-only and may allocate.
"""
function prepare(
    ::MetaSensitivityEstimate,
    specification::MetaSensitivityEstimateSpecification{T},
) where {T<:AbstractFloat}
    response_rows, response_columns, response_count, field_count =
        _validate_meta_sensitivity_specification(specification)
    reference_response = Matrix{T}(specification.reference_response)
    meta_sensitivity = Matrix{T}(specification.meta_sensitivity)
    inverse_method = Reconstructors.TSVDInverse(
        rtol=sqrt(eps(T)),
        atol=zero(T),
        n_trunc=0,
    )
    inverse_plan = prepare(
        inverse_method,
        Reconstructors.ReconstructorSpecification(response_count, field_count, T),
    )
    inverse_product = process(
        inverse_plan,
        Reconstructors.SVDReconstructorInputs(meta_sensitivity),
    )
    inverse = copy(Reconstructors.reconstructor(inverse_product))
    diagnostics = MetaSensitivityEstimateDiagnostics(
        Reconstructors.condition_number(inverse_product),
        Reconstructors.effective_rank(inverse_product),
    )
    all(isfinite, inverse) || throw(ArgumentError(
        "meta-sensitivity inverse produced non-finite values",
    ))
    return MetaSensitivityEstimatePlan(
        response_rows,
        response_columns,
        response_count,
        field_count,
        reference_response,
        meta_sensitivity,
        inverse,
        inverse_method,
        specification.field_order,
        specification.field_units,
        diagnostics,
    )
end

function allocate_result(plan::MetaSensitivityEstimatePlan{T}) where {T}
    return MetaSensitivityEstimateProduct(
        zeros(T, plan.field_count),
        plan.diagnostics,
    )
end

function allocate_workspace(plan::MetaSensitivityEstimatePlan{T}) where {T}
    return MetaSensitivityEstimateWorkspace(
        zeros(T, plan.response_count),
        zeros(T, plan.field_count),
    )
end

function _reject_meta_sensitivity_aliases(result, workspace, plan, input)
    offsets = result.parameter_offsets
    difference = workspace.response_difference
    candidate = workspace.candidate_offsets
    observed = input.observed_response
    reference = plan.reference_response
    sensitivity = plan.meta_sensitivity
    inverse = plan.inverse
    _reject_meta_sensitivity_alias(offsets, difference)
    _reject_meta_sensitivity_alias(offsets, candidate)
    _reject_meta_sensitivity_alias(offsets, observed)
    _reject_meta_sensitivity_alias(offsets, reference)
    _reject_meta_sensitivity_alias(offsets, sensitivity)
    _reject_meta_sensitivity_alias(offsets, inverse)
    _reject_meta_sensitivity_alias(difference, candidate)
    _reject_meta_sensitivity_alias(difference, observed)
    _reject_meta_sensitivity_alias(difference, reference)
    _reject_meta_sensitivity_alias(difference, sensitivity)
    _reject_meta_sensitivity_alias(difference, inverse)
    _reject_meta_sensitivity_alias(candidate, observed)
    _reject_meta_sensitivity_alias(candidate, reference)
    _reject_meta_sensitivity_alias(candidate, sensitivity)
    _reject_meta_sensitivity_alias(candidate, inverse)
    _reject_meta_sensitivity_alias(observed, reference)
    _reject_meta_sensitivity_alias(observed, sensitivity)
    _reject_meta_sensitivity_alias(observed, inverse)
    return nothing
end

@inline function _reject_meta_sensitivity_alias(first, second)
    Base.mightalias(first, second) && throw(ArgumentError(
        "meta-sensitivity product, workspace, input, and plan storage must not alias",
    ))
    return nothing
end

function _validate_meta_sensitivity_storage(result, workspace, plan, input)
    Base.require_one_based_indexing(
        result.parameter_offsets,
        workspace.response_difference,
        workspace.candidate_offsets,
        input.observed_response,
    )
    length(result.parameter_offsets) == plan.field_count || throw(DimensionMismatch(
        "parameter_offsets has an incompatible extent",
    ))
    length(workspace.response_difference) == plan.response_count || throw(DimensionMismatch(
        "response_difference workspace has an incompatible extent",
    ))
    length(workspace.candidate_offsets) == plan.field_count || throw(DimensionMismatch(
        "candidate_offsets workspace has an incompatible extent",
    ))
    size(input.observed_response) ==
        (plan.response_rows, plan.response_columns) || throw(DimensionMismatch(
            "observed_response must match the prepared reference_response shape",
        ))
    _reject_meta_sensitivity_aliases(result, workspace, plan, input)
    return nothing
end

"""
    process!(result, workspace, plan, inputs)

Compute and publish the raw ordered offset δ = J⁺ vec(D - D₀). The explicit
workspace CPU path is inferred and performs zero Julia heap allocations after
warmup for ordinary dense arrays and documented strided views. Invalid input
or a non-finite computed offset throws before publishing `result`.
"""
function process!(
    result::MetaSensitivityEstimateProduct{T},
    workspace::MetaSensitivityEstimateWorkspace{T},
    plan::MetaSensitivityEstimatePlan{T},
    input::MetaSensitivityEstimateInputs{T},
) where {T<:AbstractFloat}
    _validate_meta_sensitivity_storage(result, workspace, plan, input)
    all(isfinite, input.observed_response) || throw(ArgumentError(
        "observed_response must contain only finite values",
    ))
    @inbounds for index in eachindex(plan.reference_response)
        workspace.response_difference[index] =
            input.observed_response[index] - plan.reference_response[index]
    end
    all(isfinite, workspace.response_difference) || throw(ArgumentError(
        "meta-sensitivity response difference produced non-finite values",
    ))
    mul!(workspace.candidate_offsets, plan.inverse,
        workspace.response_difference)
    all(isfinite, workspace.candidate_offsets) || throw(ArgumentError(
        "meta-sensitivity estimate produced non-finite parameter offsets",
    ))
    copyto!(result.parameter_offsets, workspace.candidate_offsets)
    result.diagnostics = plan.diagnostics
    return result
end
