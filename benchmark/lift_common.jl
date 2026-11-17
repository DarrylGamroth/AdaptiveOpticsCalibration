using AdaptiveOpticsCalibration
using LinearAlgebra

import AdaptiveOpticsCalibration: allocate_result, allocate_workspace

const PR = AdaptiveOpticsCalibration.PhaseRetrieval

struct BenchmarkLiFTModel{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
} <: PR.AbstractLiFTForwardModel{T}
    reference_rate::V
    jacobian::M
end

PR.coefficient_count(model::BenchmarkLiFTModel) = size(model.jacobian, 2)
PR.observation_axes(model::BenchmarkLiFTModel) = axes(model.reference_rate)
PR.allocate_model_workspace(::BenchmarkLiFTModel) = nothing
PR.allocate_photon_rate(model::BenchmarkLiFTModel) = similar(model.reference_rate)

function PR.predict_photon_rate!(
    output,
    model::BenchmarkLiFTModel,
    ::Nothing,
    coefficients,
)
    mul!(output, model.jacobian, coefficients)
    output .+= model.reference_rate
    return output
end

function PR.analytic_photon_rate_jacobian!(
    output,
    model::BenchmarkLiFTModel,
    ::Nothing,
    coefficients,
    mode_indices,
)
    length(mode_indices) == size(model.jacobian, 2) || throw(
        DimensionMismatch("the benchmark selects the complete modal basis"),
    )
    @inbounds for index in eachindex(mode_indices)
        mode_indices[index] == index || throw(ArgumentError(
            "the benchmark requires canonical complete-basis mode order",
        ))
    end
    copyto!(output, model.jacobian)
    return output
end

function benchmark_problem(rows::Int, columns::Int)
    reference_rate = fill(100f0, rows)
    jacobian = Matrix{Float32}(undef, rows, columns)
    scale = inv(sqrt(Float32(rows)))
    @inbounds for column in axes(jacobian, 2), row in axes(jacobian, 1)
        parity = isodd(count_ones((row - 1) & (column - 1)))
        jacobian[row, column] = parity ? -scale : scale
    end
    truth = Float32[0.02 * sin(0.31 * index) for index in 1:columns]
    observation = reference_rate + jacobian * truth
    return reference_rate, jacobian, truth, observation
end

function quantile_sorted(values::Vector{UInt64}, probability::Float64)
    index = clamp(ceil(Int, probability * length(values)), 1, length(values))
    return values[index]
end
