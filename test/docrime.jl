module DOCRIMETests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.DOCRIME
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "docrime_interaction_matrix.toml"))

function row_major_matrix(values, rows, columns)
    return permutedims(reshape(values, columns, rows))
end

const CRITERIA = ConditioningCriteria(
    FIXTURE["conditioning_criteria"]["minimum_corrector_variance"],
    FIXTURE["conditioning_criteria"]["maximum_absolute_corrector_correlation"],
    FIXTURE["conditioning_criteria"]["minimum_factorization_pivot_ratio"],
)
const CROSS_SUM = row_major_matrix(
    FIXTURE["measurement_corrector_cross_covariance_sum"],
    FIXTURE["measurement_count"],
    FIXTURE["corrector_coordinate_count"],
)
const CORRECTOR_SUM = row_major_matrix(
    FIXTURE["corrector_covariance_sum"],
    FIXTURE["corrector_coordinate_count"],
    FIXTURE["corrector_coordinate_count"],
)
const EXPECTED_OPEN = row_major_matrix(
    FIXTURE["expected_open_loop_interaction_matrix"],
    FIXTURE["measurement_count"],
    FIXTURE["corrector_coordinate_count"],
)
const EXPECTED_CLOSED = row_major_matrix(
    FIXTURE["expected_closed_loop_interaction_matrix"],
    FIXTURE["measurement_count"],
    FIXTURE["corrector_coordinate_count"],
)
const F64_ATOL = FIXTURE["absolute_tolerance_f64"]

struct ZeroBasedMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::A
end

Base.size(matrix::ZeroBasedMatrix) = size(matrix.parent)
Base.axes(matrix::ZeroBasedMatrix) = (
    0:(size(matrix, 1) - 1),
    0:(size(matrix, 2) - 1),
)
Base.getindex(matrix::ZeroBasedMatrix, row::Int, column::Int) =
    matrix.parent[row + 1, column + 1]

function steady_state_allocation(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

function direct_steady_state_allocation(output, workspace, plan, input)
    process!(output, workspace, plan, input)
    return @allocated process!(output, workspace, plan, input)
end

function test_steady_state_process!(result, workspace, plan, input, expected_status)
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test result.diagnostics.status === expected_status
    expected_status === Valid || @test iszero(result.matrix)
    @test steady_state_allocation(result, workspace, plan, input) == 0
    return nothing
end

@testset "DO-CRIME open- and closed-loop FGA fixture" begin
    @test FIXTURE["schema"] == "test.julia-filter-graph/docrime-interaction-matrix/1"
    @test FIXTURE["source_repository"] == "JuliaFilterGraph.jl"
    @test FIXTURE["source_revision"] ==
        "7e9587b3ea059fe2271973a92b182c31b211c25b"
    @test FIXTURE["source_path"] ==
        "fixtures/algorithms/docrime_interaction_matrix.toml"
    specification = CovarianceSpecification(
        FIXTURE["measurement_count"],
        FIXTURE["corrector_coordinate_count"],
    )
    input = CovarianceSums(FIXTURE["sample_count"], CROSS_SUM, CORRECTOR_SUM)

    open_plan = prepare(OpenLoop(CRITERIA), specification)
    open_product = allocate_result(open_plan)
    open_workspace = allocate_workspace(open_plan)
    @test process!(open_product, open_workspace, open_plan, input) === open_product
    @test open_product.matrix ≈ EXPECTED_OPEN atol=F64_ATOL
    @test open_product.diagnostics.status === Valid
    @test open_product.diagnostics.sample_count == FIXTURE["sample_count"]
    @test open_product.diagnostics.minimum_corrector_variance ≈
        FIXTURE["expected_diagnostics"]["minimum_corrector_variance"] atol=
        F64_ATOL
    @test open_product.diagnostics.maximum_corrector_variance ≈
        FIXTURE["expected_diagnostics"]["maximum_corrector_variance"] atol=
        F64_ATOL
    @test open_product.diagnostics.maximum_absolute_corrector_correlation ≈
        FIXTURE["expected_diagnostics"]["maximum_absolute_corrector_correlation"] atol=
        F64_ATOL
    @test open_product.diagnostics.minimum_factorization_pivot_ratio ≈
        FIXTURE["expected_diagnostics"]["minimum_factorization_pivot_ratio"] atol=
        F64_ATOL

    closed_plan = prepare(
        ClosedLoop(FIXTURE["closed_loop_gain"], CRITERIA),
        specification,
    )
    closed_product = allocate_result(closed_plan)
    process!(closed_product, allocate_workspace(closed_plan), closed_plan, input)
    @test closed_product.matrix ≈ EXPECTED_CLOSED atol=F64_ATOL
    @test closed_product.diagnostics.status === Valid
end

@testset "DO-CRIME expected conditioning statuses" begin
    function diagnostics_for(corrector_sum; criteria=CRITERIA, sample_count=3)
        plan = prepare(OpenLoop(criteria), CovarianceSpecification(2, size(corrector_sum, 1)))
        result = allocate_result(plan)
        input = CovarianceSums(
            sample_count,
            zeros(2, size(corrector_sum, 1)),
            corrector_sum,
        )
        process!(result, allocate_workspace(plan), plan, input)
        @test iszero(result.matrix)
        return result.diagnostics
    end

    @test diagnostics_for(CORRECTOR_SUM; sample_count=1).status === InsufficientSamples

    insufficient = diagnostics_for([0.1 0.0; 0.0 4.0])
    @test insufficient.status === InsufficientCorrectorVariance
    @test insufficient.first_corrector_coordinate == 1

    correlated = diagnostics_for([4.0 3.8; 3.8 4.0])
    @test correlated.status === ExcessiveCorrectorCorrelation
    @test (correlated.first_corrector_coordinate,
        correlated.second_corrector_coordinate) == (1, 2)

    indefinite = diagnostics_for(
        [2.0 -1.2 -1.2; -1.2 2.0 -1.2; -1.2 -1.2 2.0];
        criteria=ConditioningCriteria(0.1, 0.9, 0.01),
    )
    @test indefinite.status === NonPositiveDefinite

    weak_pivot = diagnostics_for(
        [4.0 3.2; 3.2 4.0];
        criteria=ConditioningCriteria(0.1, 0.9, 0.5),
    )
    @test weak_pivot.status === InsufficientFactorizationPivot
    @test weak_pivot.minimum_factorization_pivot_ratio ≈ 0.36
end

@testset "DO-CRIME views, Float32, and malformed input atomicity" begin
    cross_storage = zeros(Float32, 4, 4)
    covariance_storage = zeros(Float32, 4, 4)
    cross = @view cross_storage[1:2:4, 1:2:4]
    covariance = @view covariance_storage[1:2:4, 1:2:4]
    cross .= Float32.(CROSS_SUM)
    covariance .= Float32.(CORRECTOR_SUM)
    criteria = ConditioningCriteria(0.1f0, 0.9f0, 0.01f0)
    plan = prepare(OpenLoop(criteria), CovarianceSpecification(2, 2, Float32))
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    process!(result, workspace, plan, CovarianceSums(3, cross, covariance))
    @test eltype(result.matrix) === Float32
    @test result.matrix ≈ Float32.(EXPECTED_OPEN) atol=2f-6
    view_input = CovarianceSums(3, cross, covariance)
    @test @inferred(process!(result, workspace, plan, view_input)) === result
    @test steady_state_allocation(result, workspace, plan, view_input) == 0

    before = (copy(result.matrix), result.diagnostics)
    invalid = CovarianceSums(3, fill(Float32(NaN), 2, 2), covariance)
    @test_throws ArgumentError process!(result, workspace, plan, invalid)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    aliased_cross = CovarianceSums(
        3,
        @view(workspace.staged_interaction_matrix[:, :]),
        covariance,
    )
    @test_throws ArgumentError process!(result, workspace, plan, aliased_cross)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    aliased_corrector = CovarianceSums(
        3,
        cross,
        transpose(workspace.corrector_covariance),
    )
    @test_throws ArgumentError process!(result, workspace, plan, aliased_corrector)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    zero_based = CovarianceSums(3, ZeroBasedMatrix(Float32.(CROSS_SUM)), covariance)
    @test_throws ArgumentError process!(result, workspace, plan, zero_based)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    # Product/input overlap is safe: both inputs are staged before publication.
    result.matrix .= Float32.(CROSS_SUM)
    process!(result, workspace, plan, CovarianceSums(3, result.matrix, covariance))
    @test result.matrix ≈ Float32.(EXPECTED_OPEN) atol=2f-6

    before = (copy(result.matrix), result.diagnostics)
    asymmetric = CovarianceSums(3, cross, Float32[2 1; 0 2])
    @test_throws ArgumentError process!(result, workspace, plan, asymmetric)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    # Symmetry is defined in the plan representation, after per-element
    # conversion. These unequal Float64 values both convert to 1.0f0.
    converted_symmetric = Float64[2 1 + eps(Float32) / 4; 1 2]
    process!(
        result,
        workspace,
        plan,
        CovarianceSums(3, Float64.(CROSS_SUM), converted_symmetric),
    )
    @test result.diagnostics.status === Valid
    converted_input = CovarianceSums(3, Float64.(CROSS_SUM), converted_symmetric)
    @test @inferred(process!(result, workspace, plan, converted_input)) === result
    @test steady_state_allocation(result, workspace, plan, converted_input) == 0

    converted_nonfinite = CovarianceSums(
        3,
        fill(big(10)^100, 2, 2),
        covariance,
    )
    before = (copy(result.matrix), result.diagnostics)
    @test_throws ArgumentError process!(result, workspace, plan, converted_nonfinite)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    invalid_cross_shape = CovarianceSums(3, zeros(Float32, 1, 2), covariance)
    @test_throws DimensionMismatch process!(result, workspace, plan, invalid_cross_shape)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]

    invalid_corrector_shape = CovarianceSums(3, cross, zeros(Float32, 1, 2))
    @test_throws DimensionMismatch process!(result, workspace, plan, invalid_corrector_shape)
    @test result.matrix == before[1]
    @test result.diagnostics === before[2]
end

@testset "DO-CRIME explicit-workspace steady-state allocation contract" begin
    specification = CovarianceSpecification(2, 2)
    input = CovarianceSums(3, CROSS_SUM, CORRECTOR_SUM)

    open_plan = prepare(OpenLoop(CRITERIA), specification)
    open_result = allocate_result(open_plan)
    open_workspace = allocate_workspace(open_plan)
    test_steady_state_process!(open_result, open_workspace, open_plan, input, Valid)
    @test open_result.matrix ≈ EXPECTED_OPEN atol=F64_ATOL

    closed_plan = prepare(
        ClosedLoop(FIXTURE["closed_loop_gain"], CRITERIA),
        specification,
    )
    closed_result = allocate_result(closed_plan)
    closed_workspace = allocate_workspace(closed_plan)
    test_steady_state_process!(closed_result, closed_workspace, closed_plan, input, Valid)
    @test closed_result.matrix ≈ EXPECTED_CLOSED atol=F64_ATOL

    function status_fixture(corrector_sum; criteria=CRITERIA, sample_count=3)
        plan = prepare(OpenLoop(criteria), CovarianceSpecification(2, size(corrector_sum, 1)))
        result = allocate_result(plan)
        workspace = allocate_workspace(plan)
        input = CovarianceSums(
            sample_count,
            zeros(2, size(corrector_sum, 1)),
            corrector_sum,
        )
        return result, workspace, plan, input
    end

    insufficient_samples = status_fixture(CORRECTOR_SUM; sample_count=1)
    test_steady_state_process!(insufficient_samples..., InsufficientSamples)

    insufficient_variance = status_fixture([0.1 0.0; 0.0 4.0])
    test_steady_state_process!(insufficient_variance..., InsufficientCorrectorVariance)

    excessive_correlation = status_fixture([4.0 3.8; 3.8 4.0])
    test_steady_state_process!(excessive_correlation..., ExcessiveCorrectorCorrelation)

    non_positive_definite = status_fixture(
        [2.0 -1.2 -1.2; -1.2 2.0 -1.2; -1.2 -1.2 2.0];
        criteria=ConditioningCriteria(0.1, 0.9, 0.01),
    )
    test_steady_state_process!(non_positive_definite..., NonPositiveDefinite)

    insufficient_pivot = status_fixture(
        [4.0 3.2; 3.2 4.0];
        criteria=ConditioningCriteria(0.1, 0.9, 0.5),
    )
    test_steady_state_process!(insufficient_pivot..., InsufficientFactorizationPivot)
end

@testset "DO-CRIME direct output reports typed failures atomically" begin
    criteria = ConditioningCriteria(0.1f0, 0.9f0, 0.01f0)
    plan = prepare(OpenLoop(criteria), CovarianceSpecification(2, 2, Float32))
    workspace = allocate_workspace(plan)
    output = fill(7.0f0, 2, 2)
    valid = CovarianceSums(
        UInt64(3),
        Float32[6 0; 8 13],
        Float32[4 2; 2 4],
    )

    @test @inferred(
        Union{Nothing,InteractionMatrixFailure},
        process!(output, workspace, plan, valid),
    ) === nothing
    @test output ≈ Float32.(EXPECTED_OPEN) atol=2f-6
    @test workspace.diagnostics.status === Valid
    @test direct_steady_state_allocation(output, workspace, plan, valid) == 0

    retained = copy(output)
    failures = (
        (
            CovarianceSums(UInt64(3), Float32[NaN 0; 0 0], valid.corrector_covariance_sum),
            NonFiniteCrossCovariance,
        ),
        (
            CovarianceSums(UInt64(3), valid.measurement_corrector_cross_covariance_sum,
                Float32[4 Inf; Inf 4]),
            NonFiniteCorrectorCovariance,
        ),
        (
            CovarianceSums(UInt64(3), valid.measurement_corrector_cross_covariance_sum,
                Float32[4 1; 2 4]),
            NonSymmetricCorrectorCovariance,
        ),
        (
            CovarianceSums(UInt64(16_777_218),
                valid.measurement_corrector_cross_covariance_sum,
                valid.corrector_covariance_sum),
            SampleCountNotRepresentable,
        ),
        (
            CovarianceSums(typemax(UInt64),
                valid.measurement_corrector_cross_covariance_sum,
                valid.corrector_covariance_sum),
            SampleCountOutOfRange,
        ),
    )
    for (input, expected) in failures
        @test @inferred(
            Union{Nothing,InteractionMatrixFailure},
            process!(output, workspace, plan, input),
        ) === expected
        @test output == retained
        @test direct_steady_state_allocation(output, workspace, plan, input) == 0
    end

    closed_plan = prepare(
        ClosedLoop(2.0f0, criteria),
        CovarianceSpecification(2, 2, Float32),
    )
    closed_workspace = allocate_workspace(closed_plan)
    overflowing = CovarianceSums(
        UInt64(2),
        fill(floatmax(Float32), 2, 2),
        Float32[1 0; 0 1],
    )
    @test @inferred(
        Union{Nothing,InteractionMatrixFailure},
        process!(output, closed_workspace, closed_plan, overflowing),
    ) === NonFiniteSolve
    @test output == retained
    @test direct_steady_state_allocation(
        output,
        closed_workspace,
        closed_plan,
        overflowing,
    ) == 0
end

@testset "DO-CRIME option validation" begin
    @test_throws ArgumentError ConditioningCriteria(0.0, 0.9, 0.01)
    @test_throws ArgumentError ConditioningCriteria(0.1, 1.0, 0.01)
    @test_throws ArgumentError ConditioningCriteria(0.1, 0.9, 0.0)
    @test_throws ArgumentError ClosedLoop(0.0, CRITERIA)
    @test_throws ArgumentError CovarianceSums(-1, CROSS_SUM, CORRECTOR_SUM)
    @test_throws ArgumentError CovarianceSpecification(0, 2)
    @test_throws ArgumentError CovarianceSpecification(2, 0)
    @test_throws ArgumentError CovarianceSpecification(-1, 2)
    @test_throws ArgumentError CovarianceSpecification(2, -1)
end

end
