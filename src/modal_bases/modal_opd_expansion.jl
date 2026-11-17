"""Select pure expansion of an explicit sampled modal basis into OPD."""
struct ModalOPDExpansion <: AbstractCalibrationMethod end

"""
    ModalOPDExpansionSpecification(rows, columns, mode_count, pupil_support,
                                    T=Float64)

Declare the two-dimensional sampled OPD axes, exact modal count, and Boolean
pupil support for a modal OPD expansion. The basis is indexed as
`basis[row, column, mode]`, and its first two axes use Julia column-major
sample order. `pupil_support` is copied during construction and must be
nonempty. `T` is the floating-point representation shared by the execution
arrays.
"""
struct ModalOPDExpansionSpecification{T<:AbstractFloat}
    rows::Int
    columns::Int
    sample_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}

    function ModalOPDExpansionSpecification{T}(
        rows::Integer,
        columns::Integer,
        mode_count::Integer,
        pupil_support::AbstractMatrix{Bool},
    ) where {T<:AbstractFloat}
        T === Float32 || T === Float64 || throw(ArgumentError(
            "modal OPD expansion supports Float32 and Float64",
        ))
        row_count = Int(rows)
        column_count = Int(columns)
        modes = Int(mode_count)
        row_count > 0 || throw(ArgumentError("rows must be positive"))
        column_count > 0 || throw(ArgumentError("columns must be positive"))
        modes > 0 || throw(ArgumentError("mode_count must be positive"))
        Base.require_one_based_indexing(pupil_support)
        size(pupil_support) == (row_count, column_count) || throw(DimensionMismatch(
            "pupil_support must have shape $((row_count, column_count))",
        ))
        any(pupil_support) || throw(ArgumentError("pupil_support must be nonempty"))
        samples = _checked_extent("modal OPD", row_count, column_count)
        _checked_extent("modal OPD basis", row_count, column_count, modes)
        return new{T}(
            row_count,
            column_count,
            samples,
            modes,
            vec(copy(pupil_support)),
        )
    end
end

function ModalOPDExpansionSpecification(
    rows::Integer,
    columns::Integer,
    mode_count::Integer,
    pupil_support::AbstractMatrix{Bool},
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return ModalOPDExpansionSpecification{T}(rows, columns, mode_count, pupil_support)
end

"""Prepared dimensions and copied pupil support for one modal OPD expansion."""
struct ModalOPDExpansionPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    sample_count::Int
    mode_count::Int
    pupil_support::Vector{Bool}
end

function prepare(
    ::ModalOPDExpansion,
    specification::ModalOPDExpansionSpecification{T},
) where {T<:AbstractFloat}
    return ModalOPDExpansionPlan{T}(
        specification.rows,
        specification.columns,
        specification.sample_count,
        specification.mode_count,
        copy(specification.pupil_support),
    )
end

"""
    ModalOPDExpansionInputs(basis, coefficients)

Bundle one finite sampled modal basis and its modal coefficients. `basis` has
shape `(rows, columns, mode_count)` and `coefficients` has exactly
`mode_count` entries. This operation performs no unit conversion: each output
OPD sample has the basis coefficient unit multiplied by the corresponding
modal-coefficient unit.
"""
struct ModalOPDExpansionInputs{
    T<:AbstractFloat,
    B<:AbstractArray{T,3},
    C<:AbstractVector{T},
}
    basis::B
    coefficients::C
end

"""Caller-owned OPD product for one explicit modal expansion."""
struct ModalOPDExpansionProduct{T<:AbstractFloat,M<:AbstractMatrix{T}}
    opd::M
end

"""Return the caller-owned OPD matrix in a modal expansion product."""
opd(result::ModalOPDExpansionProduct) = result.opd

function allocate_result(plan::ModalOPDExpansionPlan{T}) where {T<:AbstractFloat}
    return ModalOPDExpansionProduct(zeros(T, plan.rows, plan.columns))
end

allocate_workspace(::ModalOPDExpansionPlan) = nothing

function _validate_modal_opd_expansion(
    result::ModalOPDExpansionProduct,
    plan::ModalOPDExpansionPlan,
    input::ModalOPDExpansionInputs,
)
    Base.require_one_based_indexing(result.opd, input.basis, input.coefficients)
    size(result.opd) == (plan.rows, plan.columns) || throw(DimensionMismatch(
        "modal OPD result must have shape $((plan.rows, plan.columns))",
    ))
    size(input.basis) == (plan.rows, plan.columns, plan.mode_count) || throw(
        DimensionMismatch(
            "modal OPD basis must have shape $((plan.rows, plan.columns, plan.mode_count))",
        ),
    )
    length(input.coefficients) == plan.mode_count || throw(DimensionMismatch(
        "modal OPD coefficients must have length $(plan.mode_count)",
    ))
    Base.mightalias(result.opd, input.basis) && throw(ArgumentError(
        "modal OPD result and basis must not alias",
    ))
    Base.mightalias(result.opd, input.coefficients) && throw(ArgumentError(
        "modal OPD result and coefficients must not alias",
    ))
    all(isfinite, input.basis) || throw(ArgumentError(
        "modal OPD basis must contain only finite values",
    ))
    all(isfinite, input.coefficients) || throw(ArgumentError(
        "modal OPD coefficients must contain only finite values",
    ))
    return nothing
end

@inline function _modal_opd_value(
    plan::ModalOPDExpansionPlan{T},
    input::ModalOPDExpansionInputs{T},
    row::Int,
    column::Int,
) where {T<:AbstractFloat}
    sample = row + (column - 1) * plan.rows
    plan.pupil_support[sample] || return zero(T)
    value = zero(T)
    for mode in 1:plan.mode_count
        value += input.basis[row, column, mode] * input.coefficients[mode]
    end
    return value
end

"""
    process!(result, nothing, plan, inputs)

Evaluate `opd[row, column] = Σₖ basis[row, column, k] * coefficients[k]` on
the declared pupil support and write zero outside it. The modal count is exact;
the operation never silently truncates a basis or coefficient vector. The
explicit-workspace scalar CPU call is inferred and performs zero Julia heap
allocations after warmup for ordinary dense arrays. It uses scalar indexing and
does not claim accelerator support.
"""
function process!(
    result::ModalOPDExpansionProduct{T},
    ::Nothing,
    plan::ModalOPDExpansionPlan{T},
    input::ModalOPDExpansionInputs{T},
) where {T<:AbstractFloat}
    _validate_modal_opd_expansion(result, plan, input)
    for column in 1:plan.columns
        for row in 1:plan.rows
            value = _modal_opd_value(plan, input, row, column)
            isfinite(value) || throw(ArgumentError(
                "modal OPD expansion produced non-finite values",
            ))
        end
    end
    for column in 1:plan.columns
        for row in 1:plan.rows
            result.opd[row, column] = _modal_opd_value(plan, input, row, column)
        end
    end
    return result
end
