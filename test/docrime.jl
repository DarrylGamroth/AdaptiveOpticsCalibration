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

struct ZeroBasedVector{T,A<:AbstractVector{T}} <: AbstractVector{T}
    parent::A
end

Base.size(vector::ZeroBasedVector) = size(vector.parent)
Base.axes(vector::ZeroBasedVector) = (0:(length(vector) - 1),)
Base.getindex(vector::ZeroBasedVector, i::Int) = vector.parent[i + 1]

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

@testset "DO-CRIME incremental covariance agrees with the FGA fixture" begin
    # JuliaFilterGraph.jl at 47fe941e568c6ab160672d362cf61d780c35b11f,
    # julia/FilterGraphAlgorithms/test/data/algorithms/docrime_covariance.toml.
    # The FGA fixture records Rust FGN DocrimeCovarianceF32 at
    # 3d49237b3c9f4423f120b52bd2761cd5a90f5e94 as its source.
    samples = (
        (Float32[10, 4.5], Float32[1, 2], zeros(2, 2), zeros(2, 2)),
        (Float32[16, -0.5], Float32[3, 0], [6 -6; -5 5], [2 -2; -2 2]),
        (Float32[16, 12.5], Float32[5, 4], [12 0; 16 26], [8 4; 4 8]),
    )
    specification = CovarianceSpecification(2, 2, Float32)
    plan = prepare(CenteredCovariance(), specification)
    state = CovarianceState(plan)
    solve = prepare(OpenLoop(CRITERIA), specification)
    empty_sums = CovarianceSums(state)
    @test empty_sums.sample_count == 0
    @test process(solve, empty_sums).diagnostics.status === InsufficientSamples
    for (i, (m, c, cross, corrector)) in enumerate(samples)
        # Each call gets new scratch, exercising the three-argument interface.
        @test process!(state, plan, CovarianceSample(m, c)) === state
        sums = CovarianceSums(state)
        @test sums.sample_count == i
        @test eltype(sums.corrector_covariance_sum) === Float64
        @test sums.measurement_corrector_cross_covariance_sum ≈ cross rtol=8eps(Float64)
        @test sums.corrector_covariance_sum ≈ corrector rtol=8eps(Float64)
        i == 1 && @test process(solve, sums).diagnostics.status === InsufficientSamples
    end
    sums = CovarianceSums(state)
    saved_cross = copy(sums.measurement_corrector_cross_covariance_sum)
    saved_corrector = copy(sums.corrector_covariance_sum)
    @test empty!(state) === state
    process!(state, plan, CovarianceSample([8, 9], [4, 1]))
    @test sums.sample_count == 3
    @test sums.measurement_corrector_cross_covariance_sum == saved_cross
    @test sums.corrector_covariance_sum == saved_corrector
    @test CovarianceSums(state).sample_count == 1
    @test iszero(CovarianceSums(state).measurement_corrector_cross_covariance_sum)
    @test iszero(CovarianceSums(state).corrector_covariance_sum)

    first = CovarianceSample(samples[1][1], samples[1][2])
    @test process(plan, first).sample_count == 1
end

@testset "DO-CRIME centered covariance with large offsets" begin
    excitation = [-1.5 -0.75 0.5 0.25 1.0 0.0 0.75;
                  -0.5  0.0  1.75 -1.5 0.25 1.0 -1.25]
    response = [2.0 -1.0; 0.5 1.5; -0.75 2.5]
    correctors = excitation .+ [2.0^42, -2.0^43]
    measurements = response * excitation .+ [-2.0^43, 2.0^42, 2.0^44]
    specification = CovarianceSpecification(3, 2)
    plan = prepare(CenteredCovariance(), specification)
    workspace = allocate_workspace(plan)
    state = CovarianceState(plan)
    for k in axes(excitation, 2)
        process!(state, workspace, plan,
            CovarianceSample(view(measurements, :, k), view(correctors, :, k)))
    end

    n = size(excitation, 2)
    sums = CovarianceSums(state)
    @test sums.sample_count == n
    m, c = BigFloat.(measurements), BigFloat.(correctors)
    centered_m = m .- sum(m; dims=2) ./ n
    centered_c = c .- sum(c; dims=2) ./ n
    expected_cross = Float64.(centered_m * centered_c')
    expected_corrector = Float64.(centered_c * centered_c')
    @test sums.measurement_corrector_cross_covariance_sum ≈ expected_cross rtol=16eps(Float64)
    @test sums.corrector_covariance_sum ≈ expected_corrector rtol=16eps(Float64)
    @test isequal(sums.corrector_covariance_sum, transpose(sums.corrector_covariance_sum))

    gain = 0.4
    open = process(prepare(OpenLoop(CRITERIA), specification), sums)
    closed = process(prepare(ClosedLoop(gain, CRITERIA), specification), sums)
    @test open.diagnostics.status === Valid
    @test open.matrix ≈ response rtol=16eps(Float64)
    @test closed.diagnostics.status === Valid
    @test closed.matrix ≈ gain * response rtol=16eps(Float64)
end

@testset "DO-CRIME incremental covariance rejects malformed samples atomically" begin
    plan = prepare(CenteredCovariance(), CovarianceSpecification(2, 2))
    state = CovarianceState(plan)
    workspace = allocate_workspace(plan)
    input = CovarianceSample([2.0, -1.0], [1.0, 0.0])
    process!(state, workspace, plan, input)
    process!(state, workspace, plan, CovarianceSample([3.0, -2.0], [-2.0, 1.0]))
    reference_state = deepcopy(state)
    covariance_history(s) = (s.sample_count, s.measurement_origin, s.corrector_origin,
        s.measurement_mean_offset, s.corrector_mean_offset,
        s.measurement_corrector_cross_covariance_sum, s.corrector_covariance_sum)

    @test_throws DimensionMismatch process!(state, workspace, plan,
        CovarianceSample([0.0], input.corrector_sample))
    @test_throws DimensionMismatch process!(state, workspace, plan,
        CovarianceSample(input.measurements, [0.0, 1.0, 2.0]))
    other = prepare(CenteredCovariance(), CovarianceSpecification(2, 1))
    @test_throws ArgumentError process!(state, workspace, other, input)
    @test_throws DimensionMismatch process!(state, allocate_workspace(other), plan, input)

    for malformed in (
        CovarianceSample([0.0, NaN], input.corrector_sample),
        CovarianceSample(input.measurements, [0.0, Inf]),
        CovarianceSample(input.measurements, [0, big(10)^400]),
        CovarianceSample(ZeroBasedVector(input.measurements), input.corrector_sample),
        CovarianceSample(input.measurements, view(workspace.corrector_deviation, :)),
        CovarianceSample(view(state.measurement_origin, :), input.corrector_sample),
        CovarianceSample(input.measurements, [0.0, 1e200]),
    )
        @test_throws ArgumentError process!(state, workspace, plan, malformed)
        @test covariance_history(state) == covariance_history(reference_state)
    end

    process!(state, workspace, plan, input)
    process!(reference_state, allocate_workspace(plan), plan, input)
    @test covariance_history(state) == covariance_history(reference_state)

    overflow_state = CovarianceState(plan)
    shared = zeros(2)
    process!(overflow_state, workspace, plan, CovarianceSample(shared, shared))
    process!(overflow_state, workspace, plan, CovarianceSample(shared, [0, 1e154]))
    # The new rank-one term is finite; adding it to the preceding sum overflows.
    before = deepcopy(overflow_state)
    overflowing = CovarianceSample(shared, [0, -1e154])
    @test_throws ArgumentError process!(overflow_state, workspace, plan, overflowing)
    @test covariance_history(overflow_state) == covariance_history(before)

    guard_state = CovarianceState(plan)
    guard_state.sample_count = min(typemax(Int), Int64(1) << 53)
    @test_throws ArgumentError process!(guard_state, workspace, plan, input)
    @test guard_state.sample_count == min(typemax(Int), Int64(1) << 53)
end

@testset "DO-CRIME incremental covariance explicit-workspace allocation" begin
    plan = prepare(CenteredCovariance(), CovarianceSpecification(2, 2))
    for T in (Float32, Float64)
        workspace = allocate_workspace(plan)
        state = CovarianceState(plan)
        measurement_storage = T[1, 0, 2, 0]
        corrector_storage = T[4, 0, 5, 0]
        sample = CovarianceSample(view(measurement_storage, 1:2:3),
            view(corrector_storage, 1:2:3))
        @test @inferred(process!(state, workspace, plan, sample)) === state
        measurement_storage[1] = -1
        corrector_storage[3] = 3
        @test steady_state_allocation(state, workspace, plan, sample) == 0
        @test CovarianceSums(state).sample_count == 3
    end
end

end
