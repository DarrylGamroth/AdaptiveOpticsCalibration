module InteractionMatrixStabilityTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.Diagnostics
using LinearAlgebra
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "interaction_matrix_stability.toml"))

function row_major_matrix(values, rows, columns)
    return permutedims(reshape(values, columns, rows))
end

@testset "interaction-matrix stability fixture" begin
    @test FIXTURE["schema"] ==
        "test.julia-filter-graph/interaction-matrix-stability/1"
    @test FIXTURE["source_repository"] == "JuliaFilterGraph.jl"
    @test FIXTURE["source_revision"] ==
        "7e9587b3ea059fe2271973a92b182c31b211c25b"
    @test FIXTURE["source_path"] ==
        "fixtures/algorithms/interaction_matrix_stability.toml"
    measurements = FIXTURE["measurement_count"]
    correctors = FIXTURE["corrector_coordinate_count"]
    first_window = row_major_matrix(
        FIXTURE["first_window"], measurements, correctors)
    second_window = row_major_matrix(
        FIXTURE["second_window"], measurements, correctors)
    input = InteractionMatrixPair(first_window, second_window)
    method = InteractionMatrixStability(
        FIXTURE["minimum_column_norm"],
        FIXTURE["maximum_relative_column_difference"],
        FIXTURE["maximum_relative_frobenius_difference"],
    )
    plan = prepare(
        method,
        InteractionMatrixStabilitySpecification(measurements, correctors),
    )
    result = allocate_result(plan)

    @test process!(result, nothing, plan, input) === result
    @test FIXTURE["expected_status"] ==
        "excessive_relative_column_difference"
    @test result.status === ExcessiveRelativeColumnDifference
    @test result.status_corrector_coordinate ==
        FIXTURE["expected_status_corrector_coordinate_zero_based"] + 1
    @test result.minimum_first_window_column_norm ≈
        FIXTURE["expected_minimum_first_window_column_norm"] atol=
        FIXTURE["absolute_tolerance_f64"]
    @test result.minimum_second_window_column_norm ≈
        FIXTURE["expected_minimum_second_window_column_norm"] atol=
        FIXTURE["absolute_tolerance_f64"]
    @test result.maximum_relative_column_difference ≈
        FIXTURE["expected_maximum_relative_column_difference"] atol=
        FIXTURE["absolute_tolerance_f64"]
    @test result.maximum_relative_column_difference_coordinate ==
        FIXTURE["expected_maximum_relative_column_difference_coordinate_zero_based"] + 1
    @test result.relative_frobenius_difference ≈
        FIXTURE["expected_relative_frobenius_difference"] atol=
        FIXTURE["absolute_tolerance_f64"]
end

@testset "stability specification and input-derived preparation agree" begin
    first_window = Matrix{Float64}(I, 2, 2)
    pair = InteractionMatrixPair(first_window, copy(first_window))
    method = InteractionMatrixStability(0.1, 0.5, 0.5)
    explicit = @inferred prepare(
        method,
        InteractionMatrixStabilitySpecification(2, 2),
    )
    input_derived = @inferred prepare(method, pair)
    @test explicit.measurement_count == input_derived.measurement_count
    @test explicit.corrector_coordinate_count == input_derived.corrector_coordinate_count
    @test explicit.criteria == input_derived.criteria
    @test_throws ArgumentError InteractionMatrixStabilitySpecification(0, 2)
    @test_throws ArgumentError InteractionMatrixStabilitySpecification(2, 0)
    @test_throws ArgumentError InteractionMatrixStabilitySpecification(-1, 2)
    @test_throws ArgumentError InteractionMatrixStabilitySpecification(2, -1)
    @test_throws DimensionMismatch prepare(
        method,
        InteractionMatrixPair(zeros(0, 2), zeros(0, 2)),
    )
end

@testset "stability precedence, ties, and views" begin
    storage_a = zeros(4, 4)
    storage_b = zeros(4, 4)
    first_window = @view storage_a[1:2:4, 1:2:4]
    second_window = @view storage_b[1:2:4, 1:2:4]
    first_window .= [1.0 0.0; 0.0 1.0]
    second_window .= [2.0 0.0; 0.0 2.0]
    input = InteractionMatrixPair(first_window, second_window)
    plan = prepare(InteractionMatrixStability(0.1, 0.5, 0.5), input)
    result = allocate_result(plan)
    process!(result, nothing, plan, input)
    @test result.maximum_relative_column_difference_coordinate == 1

    insufficient_input = InteractionMatrixPair(
        [0.01 1.0; 0.0 0.0],
        [0.02 3.0; 0.0 0.0],
    )
    insufficient_plan = prepare(
        InteractionMatrixStability(0.1, 0.1, 0.1),
        insufficient_input,
    )
    insufficient = allocate_result(insufficient_plan)
    process!(insufficient, nothing, insufficient_plan, insufficient_input)
    @test insufficient.status === InsufficientColumnNorm
    @test insufficient.status_corrector_coordinate == 1
end

@testset "stability failures preserve result" begin
    input = InteractionMatrixPair(Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2))
    plan = prepare(InteractionMatrixStability(0.1, 0.5, 0.5), input)
    result = allocate_result(plan)
    process!(result, nothing, plan, input)
    before = Tuple(getfield(result, field) for field in fieldnames(typeof(result)))

    invalid = InteractionMatrixPair([NaN 0.0; 0.0 1.0], Matrix{Float64}(I, 2, 2))
    @test_throws ArgumentError process!(result, nothing, plan, invalid)
    @test Tuple(getfield(result, field) for field in fieldnames(typeof(result))) == before
    @test_throws DimensionMismatch process!(
        result,
        nothing,
        plan,
        InteractionMatrixPair(ones(1, 2), ones(1, 2)),
    )
    @test Tuple(getfield(result, field) for field in fieldnames(typeof(result))) == before
end

@testset "non-finite stability calculation is a typed outcome" begin
    first_window = Float32[floatmax(Float32) 0; 0 1]
    second_window = Float32[-floatmax(Float32) 0; 0 2]
    input = InteractionMatrixPair(first_window, second_window)
    plan = prepare(
        InteractionMatrixStability(0.1f0, 0.5f0, 0.5f0),
        InteractionMatrixStabilitySpecification(2, 2),
    )
    result = allocate_result(plan)
    @test @inferred(process!(result, nothing, plan, input)) === result
    @test result.status === NonFiniteStabilityCalculation
    @test result.status_corrector_coordinate == 0
    @test result.minimum_first_window_column_norm == 0
    @test result.minimum_second_window_column_norm == 0
    @test result.maximum_relative_column_difference == 0
    @test result.maximum_relative_column_difference_coordinate == 0
    @test result.relative_frobenius_difference == 0
    process!(result, nothing, plan, input)
    @test @allocated(process!(result, nothing, plan, input)) == 0
end

@testset "stability criteria validation" begin
    for value in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError InteractionMatrixStability(value, 0.5, 0.5)
    end
    for value in (-0.1, 2.1, NaN, Inf)
        @test_throws ArgumentError InteractionMatrixStability(0.1, value, 0.5)
        @test_throws ArgumentError InteractionMatrixStability(0.1, 0.5, value)
    end
end

end
