module ReconstructorTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.Reconstructors
using LinearAlgebra
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "reconstructor_preparation.toml"))

function row_major_matrix(values, rows, columns)
    return permutedims(reshape(values, columns, rows))
end

const D = row_major_matrix(FIXTURE["interaction_matrix"],
    FIXTURE["measurement_count"], FIXTURE["corrector_coordinate_count"])
const CN = row_major_matrix(FIXTURE["measurement_noise_covariance"],
    FIXTURE["measurement_count"], FIXTURE["measurement_count"])
const WC = row_major_matrix(FIXTURE["corrector_stroke_metric"],
    FIXTURE["corrector_coordinate_count"], FIXTURE["corrector_coordinate_count"])
const EXPECTED_SINGULAR_VALUES = FIXTURE["expected_singular_values"]
const EXPECTED_GAINS = FIXTURE["expected_tikhonov_gains"]
const EXPECTED_RECONSTRUCTOR = row_major_matrix(FIXTURE["expected_reconstructor"],
    FIXTURE["corrector_coordinate_count"], FIXTURE["measurement_count"])
const EXPECTED_TRUNCATED_RECONSTRUCTOR = row_major_matrix(
    FIXTURE["expected_truncated_reconstructor"],
    FIXTURE["corrector_coordinate_count"], FIXTURE["measurement_count"])
const F64_ATOL = FIXTURE["absolute_tolerance_f64"]

const AOS_MAP_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_map_control_matrix.toml"))
const AOS_SVD_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_svd_inverse.toml"))
const SVD_D = row_major_matrix(
    AOS_SVD_FIXTURE["interaction_matrix"],
    AOS_SVD_FIXTURE["measurement_count"],
    AOS_SVD_FIXTURE["corrector_coordinate_count"],
)

function assert_svd_fixture_case(method, fixture_case)
    specification = ReconstructorSpecification(
        AOS_SVD_FIXTURE["measurement_count"],
        AOS_SVD_FIXTURE["corrector_coordinate_count"],
    )
    plan = @inferred prepare(method, specification)
    result = allocate_result(plan)
    input = SVDReconstructorInputs(SVD_D)
    @test @inferred(process!(result, nothing, plan, input)) === result
    tolerance = AOS_SVD_FIXTURE["absolute_tolerance_f64"]
    expected_matrix = row_major_matrix(
        fixture_case["expected_reconstructor"],
        specification.corrector_coordinate_count,
        specification.measurement_count,
    )
    @test reconstructor(result) === result.matrix
    @test measurement_singular_vectors(result) ===
        result.measurement_singular_vectors
    @test corrector_singular_vectors(result) ===
        result.corrector_singular_vectors
    @test singular_values(result) === result.singular_values
    @test spectral_gains(result) === result.spectral_gains
    @test result.singular_values ≈
        AOS_SVD_FIXTURE["expected_singular_values"] atol=tolerance rtol=tolerance
    @test result.spectral_gains ≈
        fixture_case["expected_spectral_gains"] atol=tolerance rtol=tolerance
    @test result.matrix ≈ expected_matrix atol=tolerance rtol=tolerance
    @test result.matrix ≈
        result.corrector_singular_vectors *
        Diagonal(result.spectral_gains) *
        result.measurement_singular_vectors' atol=tolerance rtol=tolerance
    @test effective_rank(result) == fixture_case["expected_effective_rank"]
    @test condition_number(result) ≈
        fixture_case["expected_condition_number"] atol=tolerance rtol=tolerance
    @test truncation_count(result) == fixture_case["expected_truncation_count"]
    return result
end

@testset "stroke-weighted reconstructor FGA fixture" begin
    @test FIXTURE["schema"] == "test.julia-filter-graph/reconstructor-preparation/1"
    @test FIXTURE["source_repository"] == "JuliaFilterGraph.jl"
    @test FIXTURE["source_revision"] == "c3ddba2d8c9f4ccebb3e089a4b3933ecde4b2f1e"
    @test FIXTURE["source_path"] == "fixtures/algorithms/reconstructor_preparation.toml"
    specification = ReconstructorSpecification(
        FIXTURE["measurement_count"], FIXTURE["corrector_coordinate_count"])
    input = StrokeWeightedReconstructorInputs(D, CN, WC)
    plan = prepare(StrokeWeightedTikhonov(FIXTURE["retained_rank"];
        tikhonov_scale=FIXTURE["tikhonov_scale"]), specification)
    result = allocate_result(plan)

    @test plan.rank_rtol == 3eps(Float64)
    @test process!(result, nothing, plan, input) === result
    @test result.singular_values ≈ EXPECTED_SINGULAR_VALUES atol=F64_ATOL
    @test result.tikhonov_gains ≈ EXPECTED_GAINS atol=F64_ATOL
    @test result.matrix ≈ EXPECTED_RECONSTRUCTOR atol=F64_ATOL
    direct = (D' * (CN \ D) + FIXTURE["tikhonov_scale"]^2 * WC) \
        (D' / CN)
    @test result.matrix ≈ direct atol=F64_ATOL
    @test result.diagnostics.whitened_interaction_matrix_condition_number ≈
        FIXTURE["expected_condition_number"] atol=F64_ATOL
    @test result.diagnostics.whitened_interaction_matrix_residual ≈
        FIXTURE["expected_full_response_residual"] atol=F64_ATOL
    @test result.diagnostics.full_reconstructor_residual ≈
        FIXTURE["expected_full_reconstructor_residual"] atol=F64_ATOL
    @test result.diagnostics.regularization_metric_noise_amplification ≈
        FIXTURE["expected_full_stroke_noise_amplification"] atol=F64_ATOL

    truncated_plan = prepare(
        StrokeWeightedTikhonov(FIXTURE["truncated_rank"];
            tikhonov_scale=FIXTURE["tikhonov_scale"]),
        specification,
    )
    truncated = allocate_result(truncated_plan)
    process!(truncated, nothing, truncated_plan, input)
    @test truncated.matrix ≈ EXPECTED_TRUNCATED_RECONSTRUCTOR atol=F64_ATOL
    @test truncated.diagnostics.whitened_interaction_matrix_residual ≈
        FIXTURE["expected_truncated_response_residual"] atol=F64_ATOL
    @test truncated.diagnostics.full_reconstructor_residual ≈
        FIXTURE["expected_truncated_reconstructor_residual"] atol=F64_ATOL
    @test truncated.diagnostics.regularization_metric_noise_amplification ≈
        FIXTURE["expected_truncated_stroke_noise_amplification"] atol=F64_ATOL
end

@testset "MAP covariance equation and parameterization equivalence" begin
    prior = inv(WC)
    α = FIXTURE["tikhonov_scale"]^2
    specification = ReconstructorSpecification(
        FIXTURE["measurement_count"], FIXTURE["corrector_coordinate_count"])
    input = MAPReconstructorInputs(D, CN, prior)
    plan = prepare(MAPReconstructor(prior_weight=α), specification)
    result = allocate_result(plan)
    process!(result, nothing, plan, input)

    @test plan.rank_rtol == sqrt(eps(Float64))
    direct = (D' * (CN \ D) + α * inv(prior)) \ (D' / CN)
    @test result.matrix ≈ direct atol=1e-12
    @test result.matrix ≈ EXPECTED_RECONSTRUCTOR atol=F64_ATOL
    @test result.diagnostics.retained_rank == FIXTURE["retained_rank"]
    @test result.diagnostics.regularization_metric_noise_amplification ≈
        norm(result.tikhonov_gains) atol=F64_ATOL
end

@testset "MAP reconstructor AOS snapshot fixture" begin
    @test AOS_MAP_FIXTURE["schema"] == "test.adaptive-optics-sim/map-control-matrix/1"
    @test AOS_MAP_FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test AOS_MAP_FIXTURE["base_revision"] ==
        "d30db3f78d7fad77d0d5e9212b3eee1a34d86244"
    @test AOS_MAP_FIXTURE["source_path"] == "src/calibration/map_control_matrix.jl"
    @test AOS_MAP_FIXTURE["source_sha256"] ==
        "66eff29e70b881cdcf90df8a564d1a50474fef2bbcee1b2fc5f73879955a58af"
    @test AOS_MAP_FIXTURE["source_git_object"] ==
        "c8c0c4c77b56396c295688c1bcaf8aaeba7b61ce"
    @test AOS_MAP_FIXTURE["test_path"] == "test/testsets/control_reconstruction.jl"
    @test AOS_MAP_FIXTURE["test_sha256"] ==
        "f97efa8eeed554b48d0eef80bf7e086b3c168f48a7b88df183d72d4880074903"

    measurements = AOS_MAP_FIXTURE["measurement_count"]
    correctors = AOS_MAP_FIXTURE["corrector_coordinate_count"]
    interaction = row_major_matrix(
        AOS_MAP_FIXTURE["interaction_matrix"], measurements, correctors)
    noise_covariance = row_major_matrix(
        AOS_MAP_FIXTURE["measurement_noise_covariance"], measurements, measurements)
    prior_covariance = row_major_matrix(
        AOS_MAP_FIXTURE["command_prior_covariance"], correctors, correctors)
    expected_matrix = row_major_matrix(
        AOS_MAP_FIXTURE["expected_reconstructor"], correctors, measurements)
    expected_singular_values = AOS_MAP_FIXTURE["expected_singular_values"]
    expected_tikhonov_gains = AOS_MAP_FIXTURE["expected_tikhonov_gains"]
    tolerance = AOS_MAP_FIXTURE["absolute_tolerance_f64"]

    specification = ReconstructorSpecification(measurements, correctors)
    plan = prepare(
        MAPReconstructor(prior_weight=AOS_MAP_FIXTURE["prior_weight"]),
        specification,
    )
    result = allocate_result(plan)
    process!(result, nothing, plan, MAPReconstructorInputs(
        interaction,
        noise_covariance,
        prior_covariance,
    ))

    @test result.matrix ≈ expected_matrix atol=tolerance rtol=tolerance
    @test result.singular_values ≈
        expected_singular_values atol=tolerance rtol=tolerance
    @test result.tikhonov_gains ≈
        expected_tikhonov_gains atol=tolerance rtol=tolerance
    @test result.diagnostics.numerical_rank ==
        AOS_MAP_FIXTURE["expected_numerical_rank"]
    @test result.diagnostics.whitened_interaction_matrix_condition_number ≈
        AOS_MAP_FIXTURE["expected_condition_number"] atol=tolerance rtol=tolerance
    @test result.diagnostics.regularization_metric_noise_amplification ≈
        AOS_MAP_FIXTURE["expected_noise_amplification"] atol=tolerance rtol=tolerance
    @test result.diagnostics.noise_covariance_symmetry_error ==
        AOS_MAP_FIXTURE["expected_noise_covariance_symmetry_error"]
    @test result.diagnostics.regularization_matrix_symmetry_error ==
        AOS_MAP_FIXTURE["expected_prior_covariance_symmetry_error"]

    posterior_covariance_form = prior_covariance * interaction' /
        (interaction * prior_covariance * interaction' +
         AOS_MAP_FIXTURE["prior_weight"] * noise_covariance)
    @test result.matrix ≈ posterior_covariance_form atol=tolerance rtol=tolerance

    command_coordinate_scale = Diagonal(AOS_MAP_FIXTURE["command_coordinate_scale"])
    scaled = allocate_result(plan)
    process!(scaled, nothing, plan, MAPReconstructorInputs(
        interaction * command_coordinate_scale,
        noise_covariance,
        command_coordinate_scale \ prior_covariance / command_coordinate_scale',
    ))
    scaled_matrix = command_coordinate_scale * scaled.matrix
    @test scaled_matrix ≈ result.matrix atol=tolerance rtol=tolerance
    @test scaled.diagnostics.regularization_metric_noise_amplification ≈
        result.diagnostics.regularization_metric_noise_amplification atol=tolerance rtol=tolerance
end

@testset "rank classification and zero-scale pseudoinverse" begin
    matrix = [1.0 0.0; 0.0 0.0]
    input = StrokeWeightedReconstructorInputs(matrix, Matrix{Float64}(I, 2, 2),
        Matrix{Float64}(I, 2, 2))
    plan = prepare(
        StrokeWeightedTikhonov(2; tikhonov_scale=0, rtol=1e-12),
        ReconstructorSpecification(2, 2),
    )
    result = allocate_result(plan)
    process!(result, nothing, plan, input)
    @test result.tikhonov_gains == [1.0, 0.0]
    @test result.matrix == matrix
    @test result.diagnostics.numerical_rank == 1
    @test isinf(result.diagnostics.retained_condition_number)
end

@testset "views, Float32, validation, and atomic failure" begin
    @test_throws ArgumentError ReconstructorSpecification(0, 2)
    @test_throws ArgumentError ReconstructorSpecification(-1, 2)
    @test_throws ArgumentError ReconstructorSpecification(2, 0)
    @test_throws ArgumentError ReconstructorSpecification(2, -1)

    storage = zeros(Float32, 6, 4)
    interaction = @view storage[1:2:6, 1:2:4]
    interaction .= Float32.(D)
    input = MAPReconstructorInputs(
        interaction,
        Float32.(CN),
        Float32.(inv(WC)),
    )
    plan = prepare(
        MAPReconstructor(prior_weight=Float32(0.35^2)),
        ReconstructorSpecification(3, 2, Float32),
    )
    result = allocate_result(plan)
    process!(result, nothing, plan, input)
    @test eltype(result.matrix) === Float32
    @test result.matrix ≈ Float32.(EXPECTED_RECONSTRUCTOR) atol=
        FIXTURE["absolute_tolerance_f32"]

    before = (
        copy(result.matrix),
        copy(result.singular_values),
        copy(result.tikhonov_gains),
        result.diagnostics,
    )
    invalid = MAPReconstructorInputs(fill(Float32(NaN), 3, 2), Float32.(CN),
        Float32.(inv(WC)))
    @test_throws ArgumentError process!(result, nothing, plan, invalid)
    @test result.matrix == before[1]
    @test result.singular_values == before[2]
    @test result.tikhonov_gains == before[3]
    @test result.diagnostics === before[4]

    tiny_asymmetric = Float32[1f-20 1f-20; 0 1f-20]
    bad_covariance = MAPReconstructorInputs(Float32.(D), Float32.(CN), tiny_asymmetric)
    @test_throws ArgumentError process!(result, nothing, plan, bad_covariance)
    @test_throws DimensionMismatch prepare(
        StrokeWeightedTikhonov(3),
        ReconstructorSpecification(3, 2),
    )
end

@testset "AOS compact-SVD inverse fixture" begin
    @test ExactPseudoInverse() isa AbstractSVDInverse
    @test TSVDInverse() isa AbstractSVDInverse
    @test TikhonovInverse(0) isa AbstractSVDInverse
    @test AOS_SVD_FIXTURE["schema"] ==
        "test.adaptive-optics-sim/svd-inverse/1"
    @test AOS_SVD_FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test AOS_SVD_FIXTURE["source_revision"] ==
        "378ee3dfb6bb45812a240ac58f3d3f204a9fc4fa"
    @test AOS_SVD_FIXTURE["source_paths"] == [
        "src/calibration/inverse_policies.jl",
        "src/calibration/control_matrix.jl",
    ]
    @test AOS_SVD_FIXTURE["source_test_path"] ==
        "test/testsets/control_reconstruction.jl"
    @test AOS_SVD_FIXTURE["matrix_order"] == "row-major"
    @test AOS_SVD_FIXTURE["spectral_gains_provenance"] ==
        "derived from the source equations and recorded source singular values"

    exact = assert_svd_fixture_case(
        ExactPseudoInverse(),
        AOS_SVD_FIXTURE["exact"],
    )
    @test exact.diagnostics isa SVDReconstructorDiagnostics{Float64}

    tolerance_case = AOS_SVD_FIXTURE["tsvd_tolerance"]
    tolerance_plan = @inferred prepare(
        TSVDInverse(
            rtol=tolerance_case["rtol"],
            atol=tolerance_case["atol"],
            n_trunc=tolerance_case["n_trunc"],
        ),
        ReconstructorSpecification(4, 4),
    )
    @test tolerance_plan isa SVDReconstructorPlan
    assert_svd_fixture_case(tolerance_plan.method, tolerance_case)

    explicit_case = AOS_SVD_FIXTURE["tsvd_explicit"]
    assert_svd_fixture_case(
        TSVDInverse(
            rtol=explicit_case["rtol"],
            atol=explicit_case["atol"],
            n_trunc=explicit_case["n_trunc"],
        ),
        explicit_case,
    )

    tikhonov_case = AOS_SVD_FIXTURE["tikhonov"]
    assert_svd_fixture_case(
        TikhonovInverse(
            tikhonov_case["lambda"];
            rtol=tikhonov_case["rtol"],
            atol=tikhonov_case["atol"],
        ),
        tikhonov_case,
    )
end

@testset "rank-deficient zero-regularization correction" begin
    divergence = AOS_SVD_FIXTURE["zero_lambda_divergence"]
    @test divergence["source_behavior"] ==
        "all-NaN reconstructor from a NaN null-space gain"
    @test divergence["target_behavior"] ==
        "finite pseudoinverse with a zero null-space gain"
    result = assert_svd_fixture_case(
        TikhonovInverse(
            divergence["lambda"];
            rtol=divergence["rtol"],
            atol=divergence["atol"],
        ),
        divergence,
    )
    @test all(isfinite, result.matrix)
    @test all(isfinite, result.spectral_gains)
    @test iszero(last(result.spectral_gains))
    @test isinf(condition_number(result))
end

@testset "compact-SVD views, Float32, and atomic failure" begin
    storage = zeros(Float32, 8, 8)
    interaction = @view storage[1:2:8, 1:2:8]
    interaction .= Float32.(SVD_D)
    plan = @inferred prepare(
        TSVDInverse(rtol=1.0f-3, atol=0.0f0),
        ReconstructorSpecification(4, 4, Float32),
    )
    result = allocate_result(plan)
    input = SVDReconstructorInputs(interaction)
    @test @inferred(process!(result, nothing, plan, input)) === result
    @test eltype(result.matrix) === Float32
    @test measurement_singular_vectors(result) ===
        result.measurement_singular_vectors
    @test corrector_singular_vectors(result) ===
        result.corrector_singular_vectors
    @test size(measurement_singular_vectors(result)) == (4, 4)
    @test size(corrector_singular_vectors(result)) == (4, 4)
    @test result.matrix ≈
        corrector_singular_vectors(result) *
        Diagonal(result.spectral_gains) *
        measurement_singular_vectors(result)' atol=3.0f-5 rtol=3.0f-5
    expected = row_major_matrix(
        AOS_SVD_FIXTURE["tsvd_tolerance"]["expected_reconstructor"],
        4,
        4,
    )
    @test result.matrix ≈ Float32.(expected) atol=2.0f-4 rtol=2.0f-5

    before = (
        copy(result.matrix),
        copy(result.measurement_singular_vectors),
        copy(result.corrector_singular_vectors),
        copy(result.singular_values),
        copy(result.spectral_gains),
        result.diagnostics,
    )
    @test_throws ArgumentError process!(
        result,
        nothing,
        plan,
        SVDReconstructorInputs(fill(Float32(NaN), 4, 4)),
    )
    @test result.matrix == before[1]
    @test result.measurement_singular_vectors == before[2]
    @test result.corrector_singular_vectors == before[3]
    @test result.singular_values == before[4]
    @test result.spectral_gains == before[5]
    @test result.diagnostics === before[6]
    @test_throws DimensionMismatch process!(
        result,
        nothing,
        plan,
        SVDReconstructorInputs(zeros(Float32, 3, 4)),
    )
    @test result.matrix == before[1]

    incompatible = SVDReconstructorProduct(
        zeros(Float32, 3, 4),
        zeros(Float32, 4, 4),
        zeros(Float32, 4, 4),
        zeros(Float32, 4),
        zeros(Float32, 4),
        SVDReconstructorDiagnostics{Float32}(Inf32, 0, 0),
    )
    @test_throws DimensionMismatch process!(incompatible, nothing, plan, input)
    shared = zeros(Float32, 4)
    aliased = SVDReconstructorProduct(
        zeros(Float32, 4, 4),
        zeros(Float32, 4, 4),
        zeros(Float32, 4, 4),
        shared,
        shared,
        SVDReconstructorDiagnostics{Float32}(Inf32, 0, 0),
    )
    @test_throws ArgumentError process!(aliased, nothing, plan, input)
end

@testset "compact-SVD method validation" begin
    specification = ReconstructorSpecification(4, 4, Float32)
    @test_throws ArgumentError prepare(TSVDInverse(rtol=-1), specification)
    @test_throws ArgumentError prepare(TSVDInverse(atol=-1), specification)
    @test_throws ArgumentError prepare(TSVDInverse(rtol=Inf), specification)
    @test_throws ArgumentError prepare(TSVDInverse(atol=NaN), specification)
    @test_throws ArgumentError prepare(TSVDInverse(n_trunc=-1), specification)
    @test_throws ArgumentError prepare(TikhonovInverse(-1), specification)
    @test_throws ArgumentError prepare(TikhonovInverse(Inf), specification)
    @test_throws ArgumentError prepare(TikhonovInverse(0; rtol=-1), specification)
    @test_throws ArgumentError prepare(TikhonovInverse(0; atol=NaN), specification)

    all_truncated = prepare(
        TSVDInverse(rtol=0, atol=0, n_trunc=typemax(Int)),
        specification,
    )
    result = process(all_truncated, SVDReconstructorInputs(Float32.(SVD_D)))
    @test iszero(result.matrix)
    @test effective_rank(result) == 0
    @test truncation_count(result) == 4
    @test isinf(condition_number(result))
end

@testset "compact-SVD rectangular factor orientation" begin
    for interaction in (
        Float64[1 2; 3 5; 7 11],
        Float64[1 2 4; 3 5 8],
    )
        measurement_count, corrector_count = size(interaction)
        singular_count = min(measurement_count, corrector_count)
        plan = prepare(
            ExactPseudoInverse(),
            ReconstructorSpecification(
                measurement_count,
                corrector_count,
                Float64,
            ),
        )
        result = process(plan, SVDReconstructorInputs(interaction))
        measurement_vectors = measurement_singular_vectors(result)
        corrector_vectors = corrector_singular_vectors(result)
        @test size(measurement_vectors) ==
            (measurement_count, singular_count)
        @test size(corrector_vectors) ==
            (corrector_count, singular_count)
        @test interaction ≈
            measurement_vectors * Diagonal(singular_values(result)) *
            corrector_vectors' atol=1e-12 rtol=1e-12
        @test measurement_vectors' * measurement_vectors ≈
            Matrix{Float64}(I, singular_count, singular_count) atol=1e-12
        @test corrector_vectors' * corrector_vectors ≈
            Matrix{Float64}(I, singular_count, singular_count) atol=1e-12
        probe = collect(range(-0.4, 0.6; length=measurement_count))
        factorized = corrector_vectors *
            (spectral_gains(result) .* (measurement_vectors' * probe))
        @test factorized ≈ reconstructor(result) * probe atol=1e-12 rtol=1e-12
    end
end

end
