"""Select the pure sampled modal fitting diagnostic."""
struct ModalFitting <: AbstractCalibrationMethod end

"""
    ModalFittingSpecification(rows, columns, mode_count, T=Float64)

Declare two-dimensional sampled OPD axes and an explicit modal count. The
flattened sample order is Julia column-major order. `T` is the floating-point
representation shared by all execution arrays.
"""
struct ModalFittingSpecification{T<:AbstractFloat}
    rows::Int
    columns::Int
    sample_count::Int
    mode_count::Int

    function ModalFittingSpecification{T}(
        rows::Integer,
        columns::Integer,
        mode_count::Integer,
    ) where {T<:AbstractFloat}
        row_count = Int(rows)
        column_count = Int(columns)
        modes = Int(mode_count)
        row_count > 0 || throw(ArgumentError("rows must be positive"))
        column_count > 0 || throw(ArgumentError("columns must be positive"))
        modes > 0 || throw(ArgumentError("mode_count must be positive"))
        samples = _checked_extent("modal fitting OPD", row_count, column_count)
        _checked_extent("modal fitting projector", modes, samples)
        _checked_extent("modal fitting sampled modes", samples, modes)
        return new{T}(row_count, column_count, samples, modes)
    end
end

function ModalFittingSpecification(
    rows::Integer,
    columns::Integer,
    mode_count::Integer,
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return ModalFittingSpecification{T}(rows, columns, mode_count)
end

"""Prepared dimensions for one pure modal fitting diagnostic."""
struct ModalFittingPlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    sample_count::Int
    mode_count::Int
end

function prepare(method::ModalFitting, specification::ModalFittingSpecification{T}) where {T}
    return ModalFittingPlan{T}(
        specification.rows,
        specification.columns,
        specification.sample_count,
        specification.mode_count,
    )
end

"""
    ModalFittingInputs(input_opd, projector, sampled_modes)

Bundle one finite sampled OPD, a mode-by-sample projector, and a sample-by-mode
basis. All arrays have the same `AbstractFloat` element type. This operation
does not apply a pupil mask, weighting, reference subtraction, or unit
conversion.
"""
struct ModalFittingInputs{
    T<:AbstractFloat,
    O<:AbstractMatrix{T},
    P<:AbstractMatrix{T},
    B<:AbstractMatrix{T},
}
    input_opd::O
    projector::P
    sampled_modes::B
end

"""
Caller-owned OPD products for a pure modal fit. Every field retains the input
OPD unit and original two-dimensional sample axes. `input_opd` is a copy of
the supplied input; it is intentionally not relabeled as turbulence.
"""
mutable struct ModalFittingProduct{
    T<:AbstractFloat,
    I<:AbstractMatrix{T},
    F<:AbstractMatrix{T},
    R<:AbstractMatrix{T},
}
    input_opd::I
    fitted_opd::F
    residual_opd::R
end

"""Return the copied input OPD in a fitting product."""
input_opd(result::ModalFittingProduct) = result.input_opd

"""Return the fitted OPD in a fitting product."""
fitted_opd(result::ModalFittingProduct) = result.fitted_opd

"""Return the residual OPD in a fitting product."""
residual_opd(result::ModalFittingProduct) = result.residual_opd

"""Replaceable modal-coefficient scratch for a pure modal fit."""
struct ModalFittingWorkspace{T<:AbstractFloat,V<:AbstractVector{T}}
    modal_coefficients::V
end

function allocate_result(plan::ModalFittingPlan{T}) where {T}
    return ModalFittingProduct(
        zeros(T, plan.rows, plan.columns),
        zeros(T, plan.rows, plan.columns),
        zeros(T, plan.rows, plan.columns),
    )
end

function allocate_workspace(plan::ModalFittingPlan{T}) where {T}
    return ModalFittingWorkspace(zeros(T, plan.mode_count))
end

function _validate_modal_fitting_storage(
    result::ModalFittingProduct,
    workspace::ModalFittingWorkspace,
    plan,
    input::ModalFittingInputs,
)
    Base.require_one_based_indexing(
        result.input_opd,
        result.fitted_opd,
        result.residual_opd,
        workspace.modal_coefficients,
        input.input_opd,
        input.projector,
        input.sampled_modes,
    )
    expected_opd = (plan.rows, plan.columns)
    size(result.input_opd) == expected_opd || throw(DimensionMismatch(
        "input OPD result must have shape $expected_opd",
    ))
    size(result.fitted_opd) == expected_opd || throw(DimensionMismatch(
        "fitted OPD result must have shape $expected_opd",
    ))
    size(result.residual_opd) == expected_opd || throw(DimensionMismatch(
        "residual OPD result must have shape $expected_opd",
    ))
    length(workspace.modal_coefficients) == plan.mode_count || throw(DimensionMismatch(
        "modal-coefficient workspace has an incompatible extent",
    ))
    size(input.input_opd) == expected_opd || throw(DimensionMismatch(
        "input OPD must have shape $expected_opd",
    ))
    size(input.projector) == (plan.mode_count, plan.sample_count) ||
        throw(DimensionMismatch("modal projector has an incompatible extent"))
    size(input.sampled_modes) == (plan.sample_count, plan.mode_count) ||
        throw(DimensionMismatch("sampled modal basis has an incompatible extent"))
    _reject_modal_fitting_aliases(
        result.input_opd,
        result.fitted_opd,
        result.residual_opd,
        workspace.modal_coefficients,
        input.input_opd,
        input.projector,
        input.sampled_modes,
    )
    return nothing
end

function _reject_modal_fitting_aliases(input_opd, fitted_opd, residual_opd, coefficients,
    supplied_opd, projector, sampled_modes)
    inputs = (supplied_opd, projector, sampled_modes)
    (_mightalias_any(input_opd, (fitted_opd, residual_opd, coefficients, inputs...)) ||
        _mightalias_any(fitted_opd, (residual_opd, coefficients, inputs...)) ||
        _mightalias_any(residual_opd, (coefficients, inputs...)) ||
        _mightalias_any(coefficients, inputs)) && throw(ArgumentError(
        "modal fitting products, workspace, and inputs must not alias",
    ))
    return nothing
end

function _validate_modal_fitting_finite(input::ModalFittingInputs)
    all(isfinite, input.input_opd) || throw(ArgumentError("input OPD must contain only finite values"))
    all(isfinite, input.projector) || throw(ArgumentError("modal projector must contain only finite values"))
    all(isfinite, input.sampled_modes) || throw(ArgumentError("sampled modal basis must contain only finite values"))
    return nothing
end

"""
    process!(result, workspace, plan, input)

Compute `a = Pφ`, `φ̂ = B a`, and `φ - φ̂` for one finite sampled OPD. The
explicit-workspace CPU call is inferred and performs zero Julia heap
allocations after warmup for ordinary dense matrices and their documented
strided views. All arrays must use one-based axes. The call validates all
input, shape, axis, and alias contracts before publishing any product matrix.
"""
function process!(
    result::ModalFittingProduct{T},
    workspace::ModalFittingWorkspace{T},
    plan::ModalFittingPlan{T},
    input::ModalFittingInputs{T},
) where {T<:AbstractFloat}
    _validate_modal_fitting_storage(result, workspace, plan, input)
    _validate_modal_fitting_finite(input)
    for mode in 1:plan.mode_count
        accumulation = zero(T)
        for sample in 1:plan.sample_count
            accumulation += input.projector[mode, sample] * input.input_opd[sample]
        end
        isfinite(accumulation) || throw(ArgumentError(
            "modal fitting produced non-finite coefficients",
        ))
        workspace.modal_coefficients[mode] = accumulation
    end
    for column in 1:plan.columns
        for row in 1:plan.rows
            sample = row + (column - 1) * plan.rows
            fitted = zero(T)
            for mode in 1:plan.mode_count
                fitted += input.sampled_modes[sample, mode] *
                          workspace.modal_coefficients[mode]
            end
            residual = input.input_opd[row, column] - fitted
            isfinite(fitted) && isfinite(residual) || throw(ArgumentError(
                "modal fitting produced non-finite OPD products",
            ))
        end
    end
    for column in 1:plan.columns
        for row in 1:plan.rows
            sample = row + (column - 1) * plan.rows
            fitted = zero(T)
            for mode in 1:plan.mode_count
                fitted += input.sampled_modes[sample, mode] *
                          workspace.modal_coefficients[mode]
            end
            opd = input.input_opd[row, column]
            result.input_opd[row, column] = opd
            result.fitted_opd[row, column] = fitted
            result.residual_opd[row, column] = opd - fitted
        end
    end
    return result
end
