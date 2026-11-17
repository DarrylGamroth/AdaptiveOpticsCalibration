module StructuredInteractionMatrixTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.InteractionMatrices
using TOML
using Test

const STRUCTURED_INTERACTION_MATRIX_FIXTURE = TOML.parsefile(
    joinpath(@__DIR__, "fixtures", "structured_interaction_matrix.toml"),
)

function row_major_matrix(values, rows, columns, ::Type{T}=Float64) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

const ZONAL_FIXTURE = STRUCTURED_INTERACTION_MATRIX_FIXTURE["zonal"]
const HADAMARD_FIXTURE = STRUCTURED_INTERACTION_MATRIX_FIXTURE["hadamard"]

function fixture_matrix(section, key, rows, columns)
    return row_major_matrix(section[key], rows, columns)
end

function estimate_interaction_matrix(method, measurement_count, responses)
    plan = prepare(method, ResponseSpecification(measurement_count))
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    process!(result, workspace, plan, responses)
    return result, plan
end

@testset "structured interaction-matrix fixtures" begin
    @test STRUCTURED_INTERACTION_MATRIX_FIXTURE["schema"] ==
        "test.julia-filter-graph/structured-interaction-matrix/1"
    @test STRUCTURED_INTERACTION_MATRIX_FIXTURE["source_repository"] ==
        "JuliaFilterGraph.jl"
    @test STRUCTURED_INTERACTION_MATRIX_FIXTURE["source_revision"] ==
        "7e9587b3ea059fe2271973a92b182c31b211c25b"
    @test STRUCTURED_INTERACTION_MATRIX_FIXTURE["source_path"] ==
        "fixtures/algorithms/structured_interaction_matrix.toml"

    zonal_measurement_count = Int(ZONAL_FIXTURE["measurement_count"])
    zonal_corrector_count = Int(ZONAL_FIXTURE["corrector_coordinate_count"])
    zonal_response_count =
        length(ZONAL_FIXTURE["responses"]) ÷ zonal_measurement_count
    zonal_responses = fixture_matrix(
        ZONAL_FIXTURE,
        "responses",
        zonal_response_count,
        zonal_measurement_count,
    )
    zonal_expected = fixture_matrix(
        ZONAL_FIXTURE,
        "expected_interaction_matrix",
        zonal_measurement_count,
        zonal_corrector_count,
    )
    zonal_product, zonal_plan = estimate_interaction_matrix(
        ZonalPushPull(ZONAL_FIXTURE["amplitudes"]),
        zonal_measurement_count,
        zonal_responses,
    )
    @test interaction_matrix(zonal_product) ≈ zonal_expected atol=
        STRUCTURED_INTERACTION_MATRIX_FIXTURE["absolute_tolerance_f64"]
    @test zonal_plan.response_row_count == zonal_response_count

    hadamard_measurement_count = Int(HADAMARD_FIXTURE["measurement_count"])
    hadamard_corrector_count = Int(HADAMARD_FIXTURE["corrector_coordinate_count"])
    hadamard_response_count =
        length(HADAMARD_FIXTURE["responses"]) ÷ hadamard_measurement_count
    hadamard_responses = fixture_matrix(
        HADAMARD_FIXTURE,
        "responses",
        hadamard_response_count,
        hadamard_measurement_count,
    )
    hadamard_expected = fixture_matrix(
        HADAMARD_FIXTURE,
        "expected_interaction_matrix",
        hadamard_measurement_count,
        hadamard_corrector_count,
    )
    hadamard_product, hadamard_plan = estimate_interaction_matrix(
        HadamardPushPull(HADAMARD_FIXTURE["amplitudes"]),
        hadamard_measurement_count,
        hadamard_responses,
    )
    @test interaction_matrix(hadamard_product) ≈ hadamard_expected atol=
        STRUCTURED_INTERACTION_MATRIX_FIXTURE["absolute_tolerance_f64"]
    @test hadamard_plan.order == Int(HADAMARD_FIXTURE["order"])
    @test hadamard_plan.response_row_count == hadamard_response_count
end

@testset "structured interaction-matrix plans preserve floating representation" begin
    for T in (Float16, Float32, Float64, BigFloat)
        zonal_product, _ = estimate_interaction_matrix(
            ZonalPushPull(T[1]),
            1,
            reshape(T[3, -1], 2, 1),
        )
        @test interaction_matrix(zonal_product) == fill(T(2), 1, 1)
        @test eltype(interaction_matrix(zonal_product)) === T

        hadamard_product, _ = estimate_interaction_matrix(
            HadamardPushPull(T[1]),
            1,
            reshape(T[2, -2, -2, 2], 4, 1),
        )
        @test interaction_matrix(hadamard_product) == fill(T(2), 1, 1)
        @test eltype(interaction_matrix(hadamard_product)) === T
    end
end

@testset "structured interaction-matrix plan views and atomic failure" begin
    responses = fixture_matrix(
        HADAMARD_FIXTURE,
        "responses",
        length(HADAMARD_FIXTURE["responses"]) ÷
        Int(HADAMARD_FIXTURE["measurement_count"]),
        Int(HADAMARD_FIXTURE["measurement_count"]),
    )
    response_storage = zeros(16, 4)
    response_view = @view response_storage[1:2:16, 1:2:4]
    response_view .= responses
    output_storage = fill(99.0, 4, 6)
    output_view = @view output_storage[1:2:4, 1:2:6]
    result = InteractionMatrixProduct(output_view)
    plan = prepare(
        HadamardPushPull(HADAMARD_FIXTURE["amplitudes"]),
        ResponseSpecification(Int(HADAMARD_FIXTURE["measurement_count"])),
    )
    workspace = allocate_workspace(plan)
    @test process!(result, workspace, plan, response_view) === result
    @test result.status === StructuredValid
    @test output_view ≈ fixture_matrix(
        HADAMARD_FIXTURE,
        "expected_interaction_matrix",
        Int(HADAMARD_FIXTURE["measurement_count"]),
        Int(HADAMARD_FIXTURE["corrector_coordinate_count"]),
    ) atol=STRUCTURED_INTERACTION_MATRIX_FIXTURE["absolute_tolerance_f64"]
    @test @inferred(process!(output_view, workspace, plan, response_view)) ===
        StructuredValid
    @test @allocated(process!(output_view, workspace, plan, response_view)) == 0

    fill!(output_view, 17.0)
    response_view .= responses
    response_view[1, 1] = NaN
    @test @inferred(process!(output_view, workspace, plan, response_view)) ===
        StructuredNonFiniteResponse
    @test output_view == fill(17.0, size(output_view))
    @test @allocated(process!(output_view, workspace, plan, response_view)) == 0

    response_view .= responses
    response_view[1, 1] = floatmax(Float64)
    response_view[2, 1] = -floatmax(Float64)
    @test @inferred(process!(output_view, workspace, plan, response_view)) ===
        StructuredNonFiniteCalculation
    @test output_view == fill(17.0, size(output_view))
    @test @allocated(process!(output_view, workspace, plan, response_view)) == 0

    zonal = prepare(ZonalPushPull([1.0]), ResponseSpecification(1))
    zonal_product = allocate_result(zonal)
    zonal_workspace = allocate_workspace(zonal)
    fill!(zonal_product.matrix, 17.0)
    @test process!(
        zonal_product,
        zonal_workspace,
        zonal,
        reshape([NaN, 0.0], 2, 1),
    ) === zonal_product
    @test zonal_product.status === StructuredNonFiniteResponse
    @test zonal_product.matrix == fill(17.0, 1, 1)
    @test process!(
        zonal_product,
        zonal_workspace,
        zonal,
        reshape([floatmax(Float64), -floatmax(Float64)], 2, 1),
    ) === zonal_product
    @test zonal_product.status === StructuredNonFiniteCalculation
    @test zonal_product.matrix == fill(17.0, 1, 1)
    @test_throws DimensionMismatch process!(
        zonal_product,
        zonal_workspace,
        zonal,
        zeros(1, 1),
    )
    @test zonal_product.status === StructuredNonFiniteCalculation
end

function assert_structured_expected_failures!(
    method,
    valid_responses,
    nonfinite_responses,
    overflow_responses,
)
    plan = prepare(method, ResponseSpecification(1))
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    @test result.status === StructuredNotProcessed

    @test @inferred(process!(result, workspace, plan, valid_responses)) === result
    @test result.status === StructuredValid
    @test @allocated(process!(result, workspace, plan, valid_responses)) == 0
    @test @inferred(process!(result.matrix, workspace, plan, valid_responses)) ===
        StructuredValid
    @test @allocated(process!(result.matrix, workspace, plan, valid_responses)) == 0

    fill!(result.matrix, 17f0)
    @test @inferred(process!(result, workspace, plan, nonfinite_responses)) === result
    @test result.status === StructuredNonFiniteResponse
    @test result.matrix == fill(17f0, size(result.matrix))
    @test @allocated(process!(result, workspace, plan, nonfinite_responses)) == 0
    @test @inferred(process!(result.matrix, workspace, plan, nonfinite_responses)) ===
        StructuredNonFiniteResponse
    @test @allocated(process!(result.matrix, workspace, plan, nonfinite_responses)) == 0

    fill!(result.matrix, 17f0)
    @test @inferred(process!(result, workspace, plan, overflow_responses)) === result
    @test result.status === StructuredNonFiniteCalculation
    @test result.matrix == fill(17f0, size(result.matrix))
    @test @allocated(process!(result, workspace, plan, overflow_responses)) == 0
    @test @inferred(process!(result.matrix, workspace, plan, overflow_responses)) ===
        StructuredNonFiniteCalculation
    @test @allocated(process!(result.matrix, workspace, plan, overflow_responses)) == 0

    @test process!(result, workspace, plan, valid_responses) === result
    @test result.status === StructuredValid
    return nothing
end

@testset "structured interaction-matrix expected failures are typed and allocation-free" begin
    assert_structured_expected_failures!(
        ZonalPushPull(Float32[1]),
        reshape(Float32[3, -1], 2, 1),
        reshape(Float32[NaN, 0], 2, 1),
        reshape(Float32[floatmax(Float32), -floatmax(Float32)], 2, 1),
    )
    assert_structured_expected_failures!(
        HadamardPushPull(Float32[1]),
        reshape(Float32[2, -2, -2, 2], 4, 1),
        reshape(Float32[NaN, 0, 0, 0], 4, 1),
        reshape(Float32[floatmax(Float32), -floatmax(Float32), 0, 0], 4, 1),
    )
end

@testset "structured preparation rejects invalid contracts" begin
    @test_throws ArgumentError ResponseSpecification(0)
    for constructor in (ZonalPushPull, HadamardPushPull)
        @test_throws ArgumentError constructor(Float64[])
        for amplitude in (0.0, -1.0, NaN, Inf)
            @test_throws ArgumentError constructor([amplitude])
        end
    end
    @test_throws ArgumentError prepare(
        ZonalPushPull([nextfloat(0.0f0)]),
        ResponseSpecification(1),
    )
end

end
