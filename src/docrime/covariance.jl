"""Select Float64 online centered covariance reduction for DO-CRIME."""
struct CenteredCovariance <: AbstractCalibrationMethod end

"""Prepared measurement and corrector-coordinate dimensions for covariance reduction."""
struct CovariancePlan <: AbstractCalibrationPlan
    measurement_count::Int
    corrector_coordinate_count::Int

    function CovariancePlan(specification::CovarianceSpecification)
        return new(specification.measurement_count, specification.corrector_coordinate_count)
    end
end

prepare(::CenteredCovariance, specification::CovarianceSpecification) =
    CovariancePlan(specification)

"""
    CovarianceSample(measurements, corrector_sample)

One aligned pair of real, one-based CPU vectors in fixed measurement and
corrector-coordinate order. Supply the exposure-associated corrector vector for
open loop or the complete exposure-associated command increment `Δc` for
closed loop. Keep each coordinate's basis and units consistent across samples.
"""
struct CovarianceSample{M<:AbstractVector{<:Real},C<:AbstractVector{<:Real}}
    measurements::M
    corrector_sample::C
end

"""
    CovarianceState(plan::CovariancePlan)

Caller-owned, initially empty DO-CRIME covariance history. After `sample_count`
accepted pairs, the state represents the unnormalized centered sums
`Σ (mₖ − mean(m))(cₖ − mean(c))ᵀ` and `Σ (cₖ − mean(c))(cₖ − mean(c))ᵀ`.
Entries have the product of their row and column coordinate units and preserve
the supplied coordinate order. The solve normalizes by `sample_count - 1`.

Means are represented as offsets from the first sample. In shifted
coordinates, deviations `δm` and `δc` from the preceding means update the
means by `δ/n` and the sums by `(n−1)/n * δm * δcᵀ` and
`(n−1)/n * δc * δcᵀ`. The corrector sum is exactly symmetric. All state arrays
are owned Float64 storage and should only be changed through `process!` or
`empty!`. Storage is independent of the number of samples.
"""
mutable struct CovarianceState
    const plan::CovariancePlan
    sample_count::Int
    const measurement_origin::Vector{Float64}
    const corrector_origin::Vector{Float64}
    const measurement_mean_offset::Vector{Float64}
    const corrector_mean_offset::Vector{Float64}
    const measurement_corrector_cross_covariance_sum::Matrix{Float64}
    const corrector_covariance_sum::Matrix{Float64}

    function CovarianceState(plan::CovariancePlan)
        m, c = plan.measurement_count, plan.corrector_coordinate_count
        return new(plan, 0, zeros(m), zeros(c), zeros(m), zeros(c),
            zeros(m, c), zeros(c, c))
    end
end

allocate_result(plan::CovariancePlan) = CovarianceState(plan)

_covariance_state_arrays(state::CovarianceState) = (
    state.measurement_origin, state.corrector_origin,
    state.measurement_mean_offset, state.corrector_mean_offset,
    state.measurement_corrector_cross_covariance_sum, state.corrector_covariance_sum,
)

"""Reset a covariance state to zero accepted pairs, reusing its storage."""
function Base.empty!(state::CovarianceState)
    for array in _covariance_state_arrays(state)
        fill!(array, 0.0)
    end
    state.sample_count = 0
    return state
end

"""
    CovarianceSums(state::CovarianceState)

Copy the count and both centered covariance sums into an independent
interaction-matrix solve input. Subsequent updates or resets of the state
leave this snapshot unchanged. Empty and single-sample snapshots have zero
sums and produce `InsufficientSamples` in the solve.
"""
CovarianceSums(state::CovarianceState) = CovarianceSums(
    state.sample_count,
    copy(state.measurement_corrector_cross_covariance_sum),
    copy(state.corrector_covariance_sum),
)

"""Replaceable Float64 scratch for one sample's measurement and corrector deviations."""
struct CovarianceWorkspace
    measurement_deviation::Vector{Float64}
    corrector_deviation::Vector{Float64}

    function CovarianceWorkspace(plan::CovariancePlan)
        return new(Vector{Float64}(undef, plan.measurement_count),
            Vector{Float64}(undef, plan.corrector_coordinate_count))
    end
end

allocate_workspace(plan::CovariancePlan) = CovarianceWorkspace(plan)

function _stage_covariance_deviations!(deviation, values, origin, mean_offset, n)
    for i in eachindex(deviation)
        value = Float64(values[i])
        δ = n == 1 ? value : (value - origin[i]) - mean_offset[i]
        isfinite(δ) && isfinite(mean_offset[i] + δ / n) || throw(ArgumentError(
            "DO-CRIME covariance sample or centered mean is non-finite in Float64",
        ))
        deviation[i] = δ
    end
    return nothing
end

function _update_covariance_sums!(state, workspace, weight, ::Val{write}) where {write}
    cross = state.measurement_corrector_cross_covariance_sum
    corrector = state.corrector_covariance_sum
    δm = workspace.measurement_deviation
    δc = workspace.corrector_deviation
    @inbounds for j in axes(corrector, 2)
        weighted_δc = weight * δc[j]
        for i in axes(cross, 1)
            updated = fma(δm[i], weighted_δc, cross[i, j])
            if write
                cross[i, j] = updated
            elseif !isfinite(updated)
                return false
            end
        end
        for i in j:size(corrector, 1)
            updated = fma(δc[i], weighted_δc, corrector[i, j])
            if write
                corrector[i, j] = corrector[j, i] = updated
            elseif !isfinite(updated)
                return false
            end
        end
    end
    return true
end

"""
    process!(state::CovarianceState, [workspace,] plan::CovariancePlan, sample::CovarianceSample)

Accumulate one aligned sample pair. Finite samples that fit Float64 and keep
the centered recurrence finite advance the count by one. Invalid dimensions,
axes, aliases, count or non-finite arithmetic throw without advancing or
changing the state. The maximum count is `min(typemax(Int), 2^53)`.

One-based CPU vectors and strided views are supported. Inputs may share
storage with each other and must be independent of the state and workspace.
Reusing the explicit workspace enables allocation-free warmed updates with
dense vectors and strided views of Float32 or Float64 data. Use independent
state and workspace for concurrent updates.
"""
function process!(state::CovarianceState, workspace::CovarianceWorkspace,
                  plan::CovariancePlan, sample::CovarianceSample)
    state.plan === plan || throw(ArgumentError("DO-CRIME covariance state requires its plan"))
    count = state.sample_count
    0 <= count < min(typemax(Int), Int64(1) << 53) || throw(ArgumentError(
        "DO-CRIME covariance sample count cannot advance in Float64",
    ))
    m, c = plan.measurement_count, plan.corrector_coordinate_count
    measurements, corrector_sample = sample.measurements, sample.corrector_sample
    Base.require_one_based_indexing(measurements, corrector_sample)
    length(measurements) == m && length(corrector_sample) == c || throw(
        DimensionMismatch("DO-CRIME covariance sample dimensions must match the plan"),
    )
    δm, δc = workspace.measurement_deviation, workspace.corrector_deviation
    length(δm) == m && length(δc) == c || throw(DimensionMismatch(
        "DO-CRIME covariance workspace dimensions must match the plan",
    ))
    _any_cross_alias((_covariance_state_arrays(state)..., δm, δc),
        (measurements, corrector_sample)) && throw(ArgumentError(
        "DO-CRIME covariance inputs must not alias state or workspace",
    ))

    next_count = count + 1
    n = Float64(next_count)
    mean_m, mean_c = state.measurement_mean_offset, state.corrector_mean_offset
    _stage_covariance_deviations!(δm, measurements, state.measurement_origin, mean_m, n)
    _stage_covariance_deviations!(δc, corrector_sample, state.corrector_origin, mean_c, n)
    if count == 0
        copyto!(state.measurement_origin, δm)
        copyto!(state.corrector_origin, δc)
    else
        weight = Float64(count) / n
        # Check every sum before writing, using only vector-sized scratch.
        _update_covariance_sums!(state, workspace, weight, Val(false)) || throw(
            ArgumentError("DO-CRIME centered covariance sum overflows Float64"),
        )
        _update_covariance_sums!(state, workspace, weight, Val(true))
        @. mean_m += δm / n
        @. mean_c += δc / n
    end
    state.sample_count = next_count
    return state
end
