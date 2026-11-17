"""Select Float64 moments of complete, equally weighted repeated response vectors."""
struct RepeatedResponseMoments <: AbstractCalibrationMethod end

"""Numerical outcome of one complete repeated-response calculation."""
@enum ResponseMomentsStatus::UInt8 begin
    ResponseMomentsNotProcessed
    ResponseMomentsValid
    ResponseMomentsNonFiniteInput
    ResponseMomentsNonFiniteCalculation
end

@doc "The product has not received a repeated-response calculation." ResponseMomentsNotProcessed
@doc "All repeated-response moments and factor entries are finite." ResponseMomentsValid
@doc "A response input is non-finite; every product array is zero." ResponseMomentsNonFiniteInput
@doc "Finite inputs produced a non-finite Float64 calculation; every product array is zero." ResponseMomentsNonFiniteCalculation

"""
    RepeatedResponseSpecification(measurement_count, sample_count)

Declare positive measurement extent and at least two equally weighted repeats.
Input shape is measurement × repeat, with one-based axes, including ordinary
views. A column is one complete response vector, possibly an already averaged
batch or an estimated central difference. `sample_count` counts these columns,
not the underlying detector exposures. The caller supplies their association,
common coordinates, homogeneous units, and sampling assumptions.
"""
struct RepeatedResponseSpecification
    measurement_count::Int
    sample_count::Int

    function RepeatedResponseSpecification(measurement_count::Integer,
                                           sample_count::Integer)
        measurements, samples = Int(measurement_count), Int(sample_count)
        measurements > 0 || throw(ArgumentError("measurement_count must be positive"))
        samples >= 2 || throw(ArgumentError("sample_count must be at least two"))
        _checked_extent("repeated response", measurements, samples)
        return new(measurements, samples)
    end
end

"""Run-immutable dimensions of a Float64 repeated-response calculation."""
struct RepeatedResponseMomentsPlan <: AbstractCalibrationPlan
    measurement_count::Int
    sample_count::Int

    function RepeatedResponseMomentsPlan(measurement_count::Integer,
                                        sample_count::Integer)
        spec = RepeatedResponseSpecification(measurement_count, sample_count)
        return new(spec.measurement_count, spec.sample_count)
    end
end

prepare(::RepeatedResponseMoments, spec::RepeatedResponseSpecification) =
    RepeatedResponseMomentsPlan(spec.measurement_count, spec.sample_count)

"""
    RepeatedResponseMomentsProduct

Caller-owned Float64 `mean`, unbiased `sample_variance`,
`mean_standard_error = sqrt(sample_variance / sample_count)`, and
`covariance_factor[:,r] = (offset[:,r] - offset_mean) / sqrt(sample_count - 1)`,
where offsets are relative to the first response column. The returned mean is
the rounded Float64 sum of that origin and the computed offset mean. Factor
centering uses the offset mean, preserving small represented differences even
when the returned absolute mean rounds. It does not re-subtract that rounded
absolute mean from the inputs.
Vectors retain measurement order; factor columns retain repeat order.
Mean, standard error and factor entries have the input unit; variance has its
square. Standard error estimates mean uncertainty only under independent
repeats with a common mean and covariance. It is not a confidence interval.
The mathematical sample covariance is `covariance_factor * covariance_factor'`
and has rank at most `sample_count - 1`; rounded centering is not a numerical
rank certificate. Non-finite outcomes publish zero arrays. No acceptance or
independence classification is performed.
"""
mutable struct RepeatedResponseMomentsProduct{
    A<:AbstractVector{Float64}, B<:AbstractVector{Float64},
    C<:AbstractVector{Float64}, F<:AbstractMatrix{Float64},
}
    status::ResponseMomentsStatus
    mean::A
    sample_variance::B
    mean_standard_error::C
    covariance_factor::F
end

"""Replaceable Float64 scratch; prior contents never affect a new calculation."""
struct RepeatedResponseMomentsWorkspace{
    A<:AbstractVector{Float64}, B<:AbstractVector{Float64},
    F<:AbstractMatrix{Float64},
}
    mean::A
    sum_squared_deviations::B
    covariance_factor::F
end

allocate_result(plan::RepeatedResponseMomentsPlan) = RepeatedResponseMomentsProduct(
    ResponseMomentsNotProcessed, zeros(plan.measurement_count),
    zeros(plan.measurement_count), zeros(plan.measurement_count),
    zeros(plan.measurement_count, plan.sample_count),
)

allocate_workspace(plan::RepeatedResponseMomentsPlan) = RepeatedResponseMomentsWorkspace(
    zeros(plan.measurement_count), zeros(plan.measurement_count),
    zeros(plan.measurement_count, plan.sample_count),
)

function _validate_response_moments(result, workspace, plan, responses)
    vectors = (result.mean, result.sample_variance, result.mean_standard_error,
        workspace.mean, workspace.sum_squared_deviations)
    factors = (result.covariance_factor, workspace.covariance_factor)
    arrays = (vectors..., factors...)
    Base.require_one_based_indexing(responses, arrays...)
    expected = (plan.measurement_count, plan.sample_count)
    size(responses) == expected || throw(DimensionMismatch(
        "repeated responses must have shape $expected",
    ))
    foreach(vectors) do vector
        length(vector) == plan.measurement_count || throw(DimensionMismatch(
            "repeated-response vectors must match measurement_count",
        ))
    end
    foreach(factors) do factor
        size(factor) == expected || throw(DimensionMismatch(
            "repeated-response factors must have shape $expected",
        ))
    end
    _any_cross_alias(arrays, (responses,)) && throw(ArgumentError(
        "repeated-response product and scratch must not alias input",
    ))
    _any_pair_alias(arrays) && throw(ArgumentError(
        "repeated-response product and scratch arrays must not alias",
    ))
    return nothing
end

function _zero_response_moments!(result, status::ResponseMomentsStatus)
    fill!(result.mean, 0.0)
    fill!(result.sample_variance, 0.0)
    fill!(result.mean_standard_error, 0.0)
    fill!(result.covariance_factor, 0.0)
    result.status = status
    return result
end

"""
    process!(result, workspace, plan::RepeatedResponseMomentsPlan, responses)

Compute complete equally weighted moments by scalar CPU indexing. Each original
real input must be finite and is individually converted to Float64. Set the
fixed origin to the first column and compute offsets `z_r = x_r - origin`.
Initialize the offset mean to zero, then use Welford updates
`δ = z_r - ν_(r-1)`, `ν_r = ν_(r-1) + δ/r`,
`M₂_r = M₂_(r-1) + δ(z_r - ν_r)` and variance `M₂/(R-1)`.
Center the covariance factor in offset coordinates before rounding the
returned absolute mean `origin + ν_R`.
All seven product/scratch arrays and the input must have one-based axes and
must not overlap. Shape, axis and alias failures throw without changing the
product. Non-finite input or Float64 conversion/arithmetic returns its typed
status and zeros every product array. Scratch carries no history. There is no
streaming, covariance inversion, temporal-correlation correction, frame-count
scaling, confidence interval, or acceptance policy.
"""
function process!(result::RepeatedResponseMomentsProduct,
                  workspace::RepeatedResponseMomentsWorkspace,
                  plan::RepeatedResponseMomentsPlan,
                  responses::AbstractMatrix{<:Real})
    _validate_response_moments(result, workspace, plan, responses)
    all(isfinite, responses) || return _zero_response_moments!(
        result, ResponseMomentsNonFiniteInput,
    )
    staged = workspace.covariance_factor
    @inbounds for sample in 1:plan.sample_count, measurement in 1:plan.measurement_count
        value = Float64(responses[measurement, sample])
        isfinite(value) || return _zero_response_moments!(
            result, ResponseMomentsNonFiniteCalculation,
        )
        staged[measurement, sample] = value
    end
    fill!(workspace.sum_squared_deviations, 0.0)
    fill!(workspace.mean, 0.0)
    # Keep the fixed origin in column one until that factor row is finalized.
    @inbounds for sample in 2:plan.sample_count, measurement in 1:plan.measurement_count
        value = staged[measurement, sample] - staged[measurement, 1]
        delta = value - workspace.mean[measurement]
        mean = workspace.mean[measurement] + delta / sample
        m2 = workspace.sum_squared_deviations[measurement] + delta * (value - mean)
        isfinite(value) && isfinite(mean) && isfinite(m2) || return _zero_response_moments!(
            result, ResponseMomentsNonFiniteCalculation,
        )
        staged[measurement, sample] = value
        workspace.mean[measurement] = mean
        workspace.sum_squared_deviations[measurement] = m2
    end
    denominator = plan.sample_count - 1
    factor_scale = sqrt(Float64(denominator))
    @inbounds for measurement in 1:plan.measurement_count
        variance = workspace.sum_squared_deviations[measurement] / denominator
        isfinite(variance) && variance >= 0.0 || return _zero_response_moments!(
            result, ResponseMomentsNonFiniteCalculation,
        )
        workspace.sum_squared_deviations[measurement] = variance
        offset_mean = workspace.mean[measurement]
        mean = staged[measurement, 1] + offset_mean
        isfinite(mean) || return _zero_response_moments!(
            result, ResponseMomentsNonFiniteCalculation,
        )
        for sample in 2:plan.sample_count
            centered = (staged[measurement, sample] - offset_mean) / factor_scale
            isfinite(centered) || return _zero_response_moments!(
                result, ResponseMomentsNonFiniteCalculation,
            )
            staged[measurement, sample] = centered
        end
        staged[measurement, 1] = -offset_mean / factor_scale
        workspace.mean[measurement] = mean
    end
    copyto!(result.mean, workspace.mean)
    copyto!(result.sample_variance, workspace.sum_squared_deviations)
    copyto!(result.covariance_factor, staged)
    @inbounds for measurement in 1:plan.measurement_count
        result.mean_standard_error[measurement] =
            sqrt(workspace.sum_squared_deviations[measurement] / plan.sample_count)
    end
    result.status = ResponseMomentsValid
    return result
end
