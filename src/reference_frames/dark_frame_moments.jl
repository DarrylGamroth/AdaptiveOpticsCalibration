"""Select Float64 Welford mean and unbiased variance of a complete dark batch."""
struct DarkFrameMoments <: AbstractCalibrationMethod end

"""
    DarkFrameSpecification(rows, columns, sample_count)

Declare positive detector dimensions and an exact exposure count of at least
two. Inputs are one-based unsigned integer arrays of shape
`(rows, columns, sample_count)`, indexed as row × column × exposure. Samples
have raw ADC-count units; no normalization or detector model is inferred.
"""
struct DarkFrameSpecification
    rows::Int
    columns::Int
    sample_count::Int

    function DarkFrameSpecification(rows::Integer, columns::Integer,
                                    sample_count::Integer)
        r, c, n = Int(rows), Int(columns), Int(sample_count)
        r > 0 || throw(ArgumentError("rows must be positive"))
        c > 0 || throw(ArgumentError("columns must be positive"))
        n >= 2 || throw(ArgumentError("dark sample_count must be at least two"))
        _checked_extent("dark frame batch", r, c, n)
        return new(r, c, n)
    end
end

"""Run-immutable dimensions for one complete dark-frame calculation."""
struct DarkFramePlan <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    sample_count::Int
end

prepare(::DarkFrameMoments, spec::DarkFrameSpecification) =
    DarkFramePlan(spec.rows, spec.columns, spec.sample_count)

"""
    DarkFrameProduct

Caller-owned Float64 `mean` in ADC counts and unbiased `variance` in squared
ADC counts, indexed row × column. `status` is `Valid` for finite moments and
`Invalid` for a non-finite calculation; invalid products contain zero arrays.
"""
mutable struct DarkFrameProduct
    status::ReferenceStatus
    mean::Matrix{Float64}
    variance::Matrix{Float64}
end

"""Replaceable Welford scratch; both arrays are reset for every complete batch."""
struct DarkFrameWorkspace
    mean::Matrix{Float64}
    sum_squared_deviations::Matrix{Float64}
end

allocate_result(plan::DarkFramePlan) = DarkFrameProduct(
    Invalid, zeros(plan.rows, plan.columns), zeros(plan.rows, plan.columns),
)
allocate_workspace(plan::DarkFramePlan) = DarkFrameWorkspace(
    zeros(plan.rows, plan.columns), zeros(plan.rows, plan.columns),
)

function _validate_dark(result, workspace, plan, raw)
    Base.require_one_based_indexing(raw)
    size(raw) == (plan.rows, plan.columns, plan.sample_count) || throw(
        DimensionMismatch("raw dark batch shape differs from its specification"),
    )
    expected = (plan.rows, plan.columns)
    for array in (result.mean, result.variance, workspace.mean,
                  workspace.sum_squared_deviations)
        size(array) == expected || throw(DimensionMismatch(
            "dark product and workspace dimensions must match the plan",
        ))
        Base.mightalias(array, raw) && throw(ArgumentError(
            "dark output and scratch must not alias the input",
        ))
    end
    arrays = (result.mean, result.variance, workspace.mean,
              workspace.sum_squared_deviations)
    _any_pair_alias(arrays) && throw(ArgumentError(
        "dark product and workspace arrays must not alias",
    ))
    return nothing
end

"""
    process!(result, workspace, plan::DarkFramePlan, raw)

Compute per-pixel Float64 Welford moments over all declared exposures:
`δ = xₙ − μₙ₋₁`, `μₙ = μₙ₋₁ + δ/n`,
`M₂ₙ = M₂ₙ₋₁ + δ(xₙ − μₙ)`, and `variance = M₂ₙ/(N − 1)`.
Unsigned input values are converted to Float64 individually. The workspace is
reset each call and retains no history. Structural failures throw before
changing the product; a non-finite calculation returns a zero `Invalid`
product. Ordinary dense unsigned arrays perform zero Julia heap allocations
after warmup with caller-owned output and workspace. Scalar CPU indexing is
the only supported execution path.
"""
function process!(result::DarkFrameProduct, workspace::DarkFrameWorkspace,
                  plan::DarkFramePlan, raw::AbstractArray{<:Unsigned,3})
    _validate_dark(result, workspace, plan, raw)
    fill!(workspace.mean, 0.0)
    fill!(workspace.sum_squared_deviations, 0.0)
    finite = _update_dark!(workspace.mean, workspace.sum_squared_deviations,
        raw, 0, plan.sample_count, plan.rows, plan.columns)
    return _dark_product!(result, workspace.mean,
        workspace.sum_squared_deviations, plan.sample_count, finite)
end

# All callers validate one-based axes, dimensions and aliases before this kernel.
function _update_dark!(mean, sum_squared_deviations, raw, count, samples, rows, columns)
    finite = true
    for exposure in 1:samples
        n = count + exposure
        for column in 1:columns, row in 1:rows
            value = Float64(raw[row, column, exposure])
            delta = value - mean[row, column]
            next_mean = mean[row, column] + delta / n
            m2 = sum_squared_deviations[row, column] + delta * (value - next_mean)
            mean[row, column] = next_mean
            sum_squared_deviations[row, column] = m2
            finite &= isfinite(next_mean) && isfinite(m2)
        end
    end
    return finite
end

function _dark_product!(result, mean, sum_squared_deviations, count, finite)
    for column in axes(mean, 2), row in axes(mean, 1)
        variance = sum_squared_deviations[row, column] / (count - 1)
        result.variance[row, column] = variance
        finite &= isfinite(variance)
    end
    if finite
        copyto!(result.mean, mean)
        result.status = Valid
    else
        fill!(result.mean, 0.0)
        fill!(result.variance, 0.0)
        result.status = Invalid
    end
    return result
end

"""
    DarkFrameState(plan::DarkFramePlan)

Own an initially empty dark-frame numerical history bound to `plan` dimensions
and exact total exposure count. `count` exposures have Float64 `mean` in ADC
counts and Welford `sum_squared_deviations` (M₂) in squared ADC counts.
`finite` is a sticky numerical-validity diagnostic. No samples are retained.
State arrays are owned dense CPU storage; do not modify the bound plan or
history between updates. Construct a fresh state to reset. Independent state
merging is not supported.
"""
mutable struct DarkFrameState
    const plan::DarkFramePlan
    count::Int
    const mean::Matrix{Float64}
    const sum_squared_deviations::Matrix{Float64}
    finite::Bool
    function DarkFrameState(plan::DarkFramePlan)
        return new(plan, 0, zeros(plan.rows, plan.columns),
            zeros(plan.rows, plan.columns), true)
    end
end

function _validate_dark_state(state::DarkFrameState, plan::DarkFramePlan)
    (state.plan.rows, state.plan.columns, state.plan.sample_count) ==
        (plan.rows, plan.columns, plan.sample_count) || throw(ArgumentError(
        "dark state was prepared for a different plan",
    ))
    0 <= state.count <= plan.sample_count || throw(ArgumentError(
        "dark state count is outside its prepared exposure count",
    ))
    return nothing
end

"""
    process!(state::DarkFrameState, plan::DarkFramePlan, raw)

Reduce an unsigned row × column detector frame or nonempty row × column ×
exposure chunk into explicit state, using the same Float64 Welford recurrence
and global exposure index as complete-batch processing. One-based arrays and
views are supported. Shape, plan mismatch, alias and exposure-overrun errors
throw before mutation. Numerical failure is sticky; exposures still count.
This completes one numerical state update and retains no input binding.
"""
function process!(state::DarkFrameState, plan::DarkFramePlan,
                  raw::Union{AbstractMatrix{<:Unsigned},AbstractArray{<:Unsigned,3}})
    _validate_dark_state(state, plan)
    Base.require_one_based_indexing(raw)
    samples = size(raw, 3)
    size(raw, 1) == plan.rows && size(raw, 2) == plan.columns || throw(
        DimensionMismatch("dark chunk detector shape differs from its specification"),
    )
    0 < samples <= plan.sample_count - state.count || throw(ArgumentError(
        "dark chunk must be nonempty and fit the remaining exposure count",
    ))
    _any_cross_alias((state.mean, state.sum_squared_deviations), (raw,)) &&
        throw(ArgumentError("dark state must not alias the input"))
    finite = _update_dark!(state.mean, state.sum_squared_deviations, raw,
        state.count, samples, plan.rows, plan.columns)
    state.finite &= finite
    state.count += samples
    return state
end

"""
    process!(result::DarkFrameProduct, workspace, plan::DarkFramePlan, state)

Extract a complete dark product from `DarkFrameState` without changing history.
The count must equal the plan's exact exposure count; incomplete, mismatched
or aliased calls throw before writes. Units, unbiased variance and numerical
failure behavior match complete-batch processing. Workspace remains replaceable.
"""
function process!(result::DarkFrameProduct, workspace::DarkFrameWorkspace,
                  plan::DarkFramePlan, state::DarkFrameState)
    _validate_dark_state(state, plan)
    state.count == plan.sample_count || throw(ArgumentError(
        "dark state must contain the complete declared exposure count",
    ))
    expected = (plan.rows, plan.columns)
    outputs = (result.mean, result.variance, workspace.mean,
        workspace.sum_squared_deviations)
    for array in outputs
        size(array) == expected || throw(DimensionMismatch(
            "dark product and workspace dimensions must match the plan",
        ))
    end
    _any_pair_alias(outputs) && throw(ArgumentError(
        "dark product and workspace arrays must not alias",
    ))
    _any_cross_alias(outputs, (state.mean, state.sum_squared_deviations)) &&
        throw(ArgumentError("dark product and workspace must not alias state"))
    return _dark_product!(result, state.mean, state.sum_squared_deviations,
        state.count, state.finite)
end
