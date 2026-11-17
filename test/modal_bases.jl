module ModalBasesTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ModalBases
using LinearAlgebra
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_modal_basis_and_fitting.toml"))

row_major_matrix(values, rows, columns) = permutedims(reshape(values, columns, rows))

struct NonOneBasedTestVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end
Base.size(array::NonOneBasedTestVector) = size(array.parent)
Base.axes(array::NonOneBasedTestVector) = (0:(length(array.parent) - 1),)
Base.IndexStyle(::Type{<:NonOneBasedTestVector}) = IndexLinear()
Base.getindex(array::NonOneBasedTestVector, index::Int) = array.parent[index + 1]

struct NonOneBasedTestMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
end
Base.size(array::NonOneBasedTestMatrix) = size(array.parent)
Base.axes(array::NonOneBasedTestMatrix) =
    (0:(size(array.parent, 1) - 1), 0:(size(array.parent, 2) - 1))
Base.IndexStyle(::Type{<:NonOneBasedTestMatrix}) = IndexCartesian()
Base.getindex(array::NonOneBasedTestMatrix, row::Int, column::Int) =
    array.parent[row + 1, column + 1]

const EIGEN_FIXTURE = FIXTURE["sampled_influence_eigenbasis"]
const FITTING_FIXTURE = FIXTURE["modal_fitting"]
const INFLUENCE_FUNCTIONS = row_major_matrix(
    EIGEN_FIXTURE["sampled_influence_matrix"],
    EIGEN_FIXTURE["sample_count"],
    EIGEN_FIXTURE["corrector_coordinate_count"],
)
const EXPECTED_V = row_major_matrix(
    EIGEN_FIXTURE["expected_modal_to_command"],
    EIGEN_FIXTURE["corrector_coordinate_count"],
    EIGEN_FIXTURE["mode_count"],
)
const EXPECTED_SAMPLED_MODES = row_major_matrix(
    EIGEN_FIXTURE["expected_sampled_modes"],
    EIGEN_FIXTURE["sample_count"],
    EIGEN_FIXTURE["mode_count"],
)
const INPUT_OPD = row_major_matrix(
    FITTING_FIXTURE["input_opd"], FITTING_FIXTURE["rows"], FITTING_FIXTURE["columns"])
const PROJECTOR = row_major_matrix(
    FITTING_FIXTURE["projector"], FITTING_FIXTURE["mode_count"],
    FITTING_FIXTURE["rows"] * FITTING_FIXTURE["columns"])
const FITTING_SAMPLED_MODES = row_major_matrix(
    FITTING_FIXTURE["sampled_modes"], FITTING_FIXTURE["rows"] * FITTING_FIXTURE["columns"],
    FITTING_FIXTURE["mode_count"])
const EXPECTED_FITTED = row_major_matrix(
    FITTING_FIXTURE["expected_fitted_opd"], FITTING_FIXTURE["rows"], FITTING_FIXTURE["columns"])
const EXPECTED_RESIDUAL = row_major_matrix(
    FITTING_FIXTURE["expected_residual_opd"], FITTING_FIXTURE["rows"], FITTING_FIXTURE["columns"])

function modal_plan(::Type{T}=Float64) where {T<:AbstractFloat}
    return prepare(
        InfluenceFunctionEigenbasis(remove_piston=EIGEN_FIXTURE["remove_piston"]),
        SampledInfluenceBasisSpecification(
            EIGEN_FIXTURE["sample_count"],
            EIGEN_FIXTURE["corrector_coordinate_count"],
            EIGEN_FIXTURE["mode_count"],
            EIGEN_FIXTURE["pupil_support"],
            T,
        ),
    )
end

function fitting_plan(::Type{T}=Float64) where {T<:AbstractFloat}
    return prepare(
        ModalFitting(),
        ModalFittingSpecification(
            FITTING_FIXTURE["rows"], FITTING_FIXTURE["columns"], FITTING_FIXTURE["mode_count"], T),
    )
end

function fitting_input(::Type{T}=Float64) where {T<:AbstractFloat}
    return ModalFittingInputs(T.(INPUT_OPD), T.(PROJECTOR), T.(FITTING_SAMPLED_MODES))
end

@testset "AOS sampled influence-function eigenbasis fixture" begin
    @test FIXTURE["schema"] == "test.adaptive-optics-sim/modal-basis-and-fitting/1"
    @test FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test FIXTURE["source_revision"] == "32ebf7e57079c4590bc1ccb2a38d39b5fc1b1d85"
    @test FIXTURE["source_paths"] == [
        "src/calibration/modal_basis.jl",
        "src/calibration/fitting_error.jl",
    ]
    @test FIXTURE["source_test_path"] ==
        "test/testsets/calibration_workflows.jl"
    @test FIXTURE["matrix_order"] == "row-major"

    method = InfluenceFunctionEigenbasis(remove_piston=true)
    @test method isa AbstractModalBasisMethod
    plan = modal_plan()
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    @test process!(result, workspace, plan, INFLUENCE_FUNCTIONS) === result
    tolerance = FIXTURE["absolute_tolerance_f64"]
    @test gram_eigenvalues(result) === result.gram_eigenvalues
    @test modal_to_command(result) === result.modal_to_command
    @test sampled_modes(result) === result.sampled_modes
    @test result.gram_eigenvalues ≈ EIGEN_FIXTURE["expected_gram_eigenvalues"] atol=tolerance rtol=tolerance
    signs = [dot(@view(result.modal_to_command[:, mode]), @view(EXPECTED_V[:, mode])) < 0 ?
        -1.0 : 1.0 for mode in axes(EXPECTED_V, 2)]
    @test result.modal_to_command ≈
        EXPECTED_V .* reshape(signs, 1, :) atol=tolerance rtol=tolerance
    @test result.sampled_modes ≈
        EXPECTED_SAMPLED_MODES .* reshape(signs, 1, :) atol=tolerance rtol=tolerance
    @test result.modal_to_command' * result.modal_to_command ≈
        Matrix{Float64}(I, EIGEN_FIXTURE["mode_count"], EIGEN_FIXTURE["mode_count"]) atol=tolerance
    @test result.sampled_modes ≈ INFLUENCE_FUNCTIONS * result.modal_to_command .-
        reshape([
            sum((INFLUENCE_FUNCTIONS * result.modal_to_command)[
                EIGEN_FIXTURE["pupil_support"], mode]) /
            count(EIGEN_FIXTURE["pupil_support"])
            for mode in 1:EIGEN_FIXTURE["mode_count"]
        ], 1, :) atol=tolerance rtol=tolerance
end

@testset "sampled influence basis options and degenerate subspace" begin
    no_piston_plan = prepare(
        InfluenceFunctionEigenbasis(remove_piston=false),
        SampledInfluenceBasisSpecification(
            EIGEN_FIXTURE["sample_count"],
            EIGEN_FIXTURE["corrector_coordinate_count"],
            2,
            EIGEN_FIXTURE["pupil_support"],
        ),
    )
    no_piston = process(no_piston_plan, INFLUENCE_FUNCTIONS)
    tolerance = FIXTURE["absolute_tolerance_f64"]
    @test no_piston.gram_eigenvalues ≈
        EIGEN_FIXTURE["expected_gram_eigenvalues"][1:2] atol=tolerance rtol=tolerance
    no_piston_signs = [
        dot(@view(no_piston.modal_to_command[:, mode]), @view(EXPECTED_V[:, mode])) < 0 ?
            -1.0 : 1.0 for mode in 1:2
    ]
    @test no_piston.modal_to_command ≈
        EXPECTED_V[:, 1:2] .* reshape(no_piston_signs, 1, :) atol=tolerance rtol=tolerance
    @test no_piston.sampled_modes ≈
        INFLUENCE_FUNCTIONS * no_piston.modal_to_command atol=tolerance rtol=tolerance

    repeated_plan = prepare(
        InfluenceFunctionEigenbasis(remove_piston=false),
        SampledInfluenceBasisSpecification(3, 3, 3, trues(3)),
    )
    repeated = process(repeated_plan, Matrix{Float64}(I, 3, 3))
    @test repeated.gram_eigenvalues == ones(3)
    @test repeated.modal_to_command * repeated.modal_to_command' ≈
        Matrix{Float64}(I, 3, 3)
    @test repeated.sampled_modes * repeated.sampled_modes' ≈
        Matrix{Float64}(I, 3, 3)
end

@testset "sampled influence basis boundaries and failures" begin
    @test_throws ArgumentError SampledInfluenceBasisSpecification(0, 1, 1, Bool[true])
    @test_throws ArgumentError SampledInfluenceBasisSpecification(1, 0, 1, Bool[true])
    @test_throws ArgumentError SampledInfluenceBasisSpecification(1, 1, 0, Bool[true])
    @test_throws DimensionMismatch SampledInfluenceBasisSpecification(1, 1, 2, Bool[true])
    @test_throws DimensionMismatch SampledInfluenceBasisSpecification(2, 1, 1, Bool[true])
    @test_throws ArgumentError SampledInfluenceBasisSpecification(1, 1, 1, Bool[false])
    @test_throws ArgumentError SampledInfluenceBasisSpecification(
        1, 1, 1, NonOneBasedTestVector(Bool[true]),
    )
    @test_throws ArgumentError SampledInfluenceBasisSpecification(1, 1, 1, Bool[true], Float16)
    @test_throws ArgumentError SampledInfluenceBasisSpecification(1, 1, 1, Bool[true], BigFloat)
    @test_throws DimensionMismatch SampledInfluenceBasisSpecification{Float64}(
        2, 2, 3, Bool[true, true],
    )

    storage = zeros(Float32, 32, 6)
    influence_view = @view storage[1:2:32, 1:2:6]
    influence_view .= Float32.(INFLUENCE_FUNCTIONS)
    plan = @inferred modal_plan(Float32)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    @test process!(result, workspace, plan, influence_view) === result
    @test eltype(result.modal_to_command) === Float32
    @test result.gram_eigenvalues ≈ Float32.(EIGEN_FIXTURE["expected_gram_eigenvalues"]) atol=3f-5 rtol=3f-5

    eigenvalue_product_storage = zeros(Float32, 5)
    command_product_storage = zeros(Float32, 5, 5)
    sampled_product_storage = zeros(Float32, 18, 5)
    view_result = SampledInfluenceEigenbasisProduct(
        @view(eigenvalue_product_storage[2:4]),
        @view(command_product_storage[2:4, 2:4]),
        @view(sampled_product_storage[2:17, 2:4]),
    )
    gram_storage = zeros(Float32, 5, 5)
    eigenvalue_workspace_storage = zeros(Float32, 5)
    command_workspace_storage = zeros(Float32, 5, 5)
    sampled_workspace_storage = zeros(Float32, 18, 5)
    view_workspace = SampledInfluenceEigenbasisWorkspace(
        @view(gram_storage[2:4, 2:4]),
        @view(eigenvalue_workspace_storage[2:4]),
        @view(command_workspace_storage[2:4, 2:4]),
        @view(sampled_workspace_storage[2:17, 2:4]),
    )
    @test process!(view_result, view_workspace, plan, influence_view) === view_result
    @test view_result.modal_to_command ≈ result.modal_to_command atol=3f-5 rtol=3f-5
    @test view_result.sampled_modes ≈ result.sampled_modes atol=3f-5 rtol=3f-5

    before = (copy(result.gram_eigenvalues), copy(result.modal_to_command), copy(result.sampled_modes))
    bad = copy(influence_view)
    bad[1, 1] = NaN32
    @test_throws ArgumentError process!(result, workspace, plan, bad)
    @test (result.gram_eigenvalues, result.modal_to_command, result.sampled_modes) == before
    @test_throws DimensionMismatch process!(result, workspace, plan, zeros(Float32, 1, 3))
    @test (result.gram_eigenvalues, result.modal_to_command, result.sampled_modes) == before
    overflowing = fill(floatmax(Float32), 16, 3)
    @test_throws ArgumentError process!(result, workspace, plan, overflowing)
    @test (result.gram_eigenvalues, result.modal_to_command, result.sampled_modes) == before

    shared = zeros(Float32, 3, 3)
    aliased = SampledInfluenceEigenbasisProduct(zeros(Float32, 3), shared, shared)
    small_plan = prepare(
        InfluenceFunctionEigenbasis(),
        SampledInfluenceBasisSpecification(3, 3, 3, Bool[true, true, true], Float32),
    )
    @test_throws ArgumentError process!(aliased, allocate_workspace(small_plan), small_plan,
        Matrix{Float32}(I, 3, 3))

    strided_gram_storage = zeros(Float32, 6, 3)
    invalid_workspace = SampledInfluenceEigenbasisWorkspace(
        @view(strided_gram_storage[1:2:5, :]),
        zeros(Float32, 3),
        zeros(Float32, 3, 3),
        zeros(Float32, 3, 3),
    )
    @test_throws ArgumentError process!(
        allocate_result(small_plan),
        invalid_workspace,
        small_plan,
        Matrix{Float32}(I, 3, 3),
    )
end

@testset "finite-corrector atmospheric Karhunen–Loève basis" begin
    influences = [
        1.0 0.0 1.0
        0.0 1.0 1.0
        -1.0 0.0 1.0
        0.0 -1.0 1.0
        7.0 -3.0 2.0
    ]
    support = Bool[true, true, true, true, false]
    projected_covariance = [
        2.0 0.2 0.0
        0.2 0.5 0.0
        0.0 0.0 3.0
    ]
    method = KarhunenLoeveBasis(remove_piston=true)
    @test method isa AbstractModalBasisMethod
    @test method.relative_rank_tolerance === nothing
    plan = @inferred prepare(
        method,
        KarhunenLoeveBasisSpecification(5, 3, 2, support),
    )
    result = @inferred process(plan, KarhunenLoeveBasisInputs(
        influences,
        projected_covariance,
    ))

    sampled = sampled_modes(result)
    command = modal_to_command(result)
    variances = atmospheric_variances(result)
    @test sampled === result.sampled_modes
    @test command === result.modal_to_command
    @test variances === result.atmospheric_variances
    @test sampled ≈ influences * command atol=2e-14
    @test vec(sum(sampled[support, :]; dims=1)) ≈ zeros(2) atol=2e-14
    @test sampled[support, :]' * sampled[support, :] / count(support) ≈
        Matrix{Float64}(I, 2, 2) atol=2e-14
    @test command' * projected_covariance * command ≈
        Diagonal(variances) atol=2e-14
    @test issorted(variances; rev=true)
    @test all(>=(0), variances)

    coordinate_scaling = Diagonal([10.0, 0.5, 2.0])
    scaled_influences = influences * coordinate_scaling
    scaled_covariance = coordinate_scaling' * projected_covariance * coordinate_scaling
    scaled_result = process(plan, KarhunenLoeveBasisInputs(
        scaled_influences,
        scaled_covariance,
    ))
    for mode in 1:2
        sign = dot(sampled[support, mode], scaled_result.sampled_modes[support, mode]) < 0 ?
            -1.0 : 1.0
        @test scaled_result.sampled_modes[support, mode] ≈
            sign .* sampled[support, mode] atol=3e-14
    end
    @test scaled_result.atmospheric_variances ≈ variances atol=3e-14

    repeated_covariance = influences' * Diagonal(Float64.(support)) * influences /
                          count(support)
    repeated_result = process(plan, KarhunenLoeveBasisInputs(
        influences,
        repeated_covariance,
    ))
    repeated_scaled_result = process(plan, KarhunenLoeveBasisInputs(
        scaled_influences,
        coordinate_scaling' * repeated_covariance * coordinate_scaling,
    ))
    @test repeated_result.atmospheric_variances ≈ ones(2) atol=2e-14
    @test repeated_scaled_result.atmospheric_variances ≈ ones(2) atol=3e-14
    @test repeated_scaled_result.sampled_modes[support, :] *
          repeated_scaled_result.sampled_modes[support, :]' ≈
        repeated_result.sampled_modes[support, :] *
        repeated_result.sampled_modes[support, :]' atol=5e-14

    with_piston_plan = prepare(
        KarhunenLoeveBasis(remove_piston=false, relative_rank_tolerance=1e-12),
        KarhunenLoeveBasisSpecification(5, 3, 3, support),
    )
    with_piston = process(with_piston_plan, KarhunenLoeveBasisInputs(
        influences,
        projected_covariance,
    ))
    @test with_piston.sampled_modes[support, :]' *
          with_piston.sampled_modes[support, :] / count(support) ≈
        Matrix{Float64}(I, 3, 3) atol=2e-14
    @test with_piston.modal_to_command' * projected_covariance *
          with_piston.modal_to_command ≈
        Diagonal(with_piston.atmospheric_variances) atol=2e-14
end

@testset "Karhunen–Loève rank and covariance failures are atomic" begin
    support = trues(3)
    plan = prepare(
        KarhunenLoeveBasis(remove_piston=false),
        KarhunenLoeveBasisSpecification(3, 2, 2, support, Float32),
    )
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    good_inputs = KarhunenLoeveBasisInputs(
        Float32[1 0; 0 1; 1 -1],
        Float32[2 0; 0 1],
    )
    @test @inferred(process!(result, workspace, plan, good_inputs)) === result
    @test eltype(result.sampled_modes) === Float32
    before = (
        copy(result.atmospheric_variances),
        copy(result.modal_to_command),
        copy(result.sampled_modes),
    )

    rank_deficient = KarhunenLoeveBasisInputs(
        Float32[1 1; 2 2; 3 3],
        Float32[1 1; 1 1],
    )
    @test_throws DimensionMismatch process!(result, workspace, plan, rank_deficient)
    @test (result.atmospheric_variances, result.modal_to_command,
        result.sampled_modes) == before

    indefinite = KarhunenLoeveBasisInputs(
        good_inputs.sampled_influences,
        Float32[1 0; 0 -1],
    )
    @test_throws ArgumentError process!(result, workspace, plan, indefinite)
    @test (result.atmospheric_variances, result.modal_to_command,
        result.sampled_modes) == before

    asymmetric = KarhunenLoeveBasisInputs(
        good_inputs.sampled_influences,
        Float32[1 0.1; 0 1],
    )
    @test_throws ArgumentError process!(result, workspace, plan, asymmetric)
    @test (result.atmospheric_variances, result.modal_to_command,
        result.sampled_modes) == before

    @test_throws ArgumentError KarhunenLoeveBasis(relative_rank_tolerance=0.0)
    @test_throws ArgumentError KarhunenLoeveBasis(relative_rank_tolerance=Inf)
    @test_throws ArgumentError KarhunenLoeveBasis(relative_rank_tolerance=1)
    @test_throws ArgumentError KarhunenLoeveBasis(relative_rank_tolerance="small")
    @test_throws ArgumentError KarhunenLoeveBasisSpecification(3, 2, 2, support, Float16)

    weak_raw_piston_influences = Float32[
        1 5e-5
        -1 5e-5
        0 0.02005
        0 -0.01995
    ]
    weak_raw_piston_covariance =
        weak_raw_piston_influences' * weak_raw_piston_influences / 4
    overfilled_piston_plan = prepare(
        KarhunenLoeveBasis(remove_piston=true),
        KarhunenLoeveBasisSpecification(4, 2, 2, trues(4), Float32),
    )
    @test_throws DimensionMismatch process(
        overfilled_piston_plan,
        KarhunenLoeveBasisInputs(
            weak_raw_piston_influences,
            weak_raw_piston_covariance,
        ),
    )
    constrained_piston_plan = prepare(
        KarhunenLoeveBasis(remove_piston=true),
        KarhunenLoeveBasisSpecification(4, 2, 1, trues(4), Float32),
    )
    constrained_piston_result = process(
        constrained_piston_plan,
        KarhunenLoeveBasisInputs(
            weak_raw_piston_influences,
            weak_raw_piston_covariance,
        ),
    )
    @test abs(sum(constrained_piston_result.sampled_modes) / 4) <= 5f-6
end

@testset "AOS pure modal fitting fixture" begin
    plan = @inferred fitting_plan()
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = fitting_input()
    @test @inferred(process!(result, workspace, plan, input)) === result
    tolerance = FIXTURE["absolute_tolerance_f64"]
    @test input_opd(result) === result.input_opd
    @test fitted_opd(result) === result.fitted_opd
    @test residual_opd(result) === result.residual_opd
    @test workspace.modal_coefficients ≈ FITTING_FIXTURE["expected_coefficients"] atol=tolerance rtol=tolerance
    @test result.input_opd == INPUT_OPD
    @test result.fitted_opd ≈ EXPECTED_FITTED atol=tolerance rtol=tolerance
    @test result.residual_opd ≈ EXPECTED_RESIDUAL atol=tolerance rtol=tolerance
    @test result.fitted_opd + result.residual_opd ≈ result.input_opd atol=tolerance rtol=tolerance
end

function fitting_allocation(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

@testset "modal fitting writable aliases and shared readonly inputs" begin
    plan = fitting_plan()
    shapes = ((2, 2), (2, 2), (2, 2), (2,), (2, 2), (2, 4), (4, 2))
    # Four writable arrays require 6 pairwise and 12 input checks. Each pair
    # overlaps through views with different starting addresses.
    for first in 1:4, second in (first + 1):7
        arrays = map(shape -> fill(7.0, shape), shapes)
        storage = fill(11.0, 9)
        first_array = reshape(@view(storage[2:(1 + prod(shapes[first]))]), shapes[first])
        second_array = reshape(@view(storage[1:prod(shapes[second])]), shapes[second])
        arrays = Base.setindex(Base.setindex(arrays, first_array, first), second_array, second)
        result = ModalFittingProduct(arrays[1:3]...)
        workspace = ModalFittingWorkspace(arrays[4])
        input = ModalFittingInputs(arrays[5:7]...)
        before = map(copy, arrays)
        @test_throws ArgumentError process!(result, workspace, plan, input)
        @test arrays == before
    end

    # All three readonly pairs may share storage without changing the result.
    storage = collect(1.0:8.0)
    input = ModalFittingInputs(reshape(@view(storage[1:4]), 2, 2),
        reshape(storage, 2, 4), reshape(storage, 4, 2))
    expected = process(plan, ModalFittingInputs(copy(input.input_opd),
        copy(input.projector), copy(input.sampled_modes)))
    result, workspace = allocate_result(plan), allocate_workspace(plan)
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test result.input_opd == expected.input_opd
    @test result.fitted_opd == expected.fitted_opd
    @test result.residual_opd == expected.residual_opd
end

@testset "modal fitting storage, Float32, inference, and allocation" begin
    @test_throws ArgumentError ModalFittingSpecification(0, 1, 1)
    @test_throws ArgumentError ModalFittingSpecification(1, 0, 1)
    @test_throws ArgumentError ModalFittingSpecification(1, 1, 0)
    @test_throws MethodError ModalFittingSpecification{Float64}(1, 1, 2, 1)

    plan = @inferred fitting_plan(Float32)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    opd_storage = zeros(Float32, 4, 4)
    projector_storage = zeros(Float32, 4, 8)
    modes_storage = zeros(Float32, 8, 4)
    opd = @view opd_storage[1:2:4, 1:2:4]
    projector = @view projector_storage[1:2:4, 1:2:8]
    modes = @view modes_storage[1:2:8, 1:2:4]
    opd .= Float32.(INPUT_OPD)
    projector .= Float32.(PROJECTOR)
    modes .= Float32.(FITTING_SAMPLED_MODES)
    input = ModalFittingInputs(opd, projector, modes)
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test result.fitted_opd ≈ Float32.(EXPECTED_FITTED) atol=2f-6 rtol=2f-6
    @test result.residual_opd ≈ Float32.(EXPECTED_RESIDUAL) atol=2f-6 rtol=2f-6
    @test fitting_allocation(result, workspace, plan, input) == 0

    product_storage = zeros(Float32, 6, 6)
    view_result = ModalFittingProduct(
        @view(product_storage[1:2, 1:2]),
        @view(product_storage[3:4, 1:2]),
        @view(product_storage[5:6, 1:2]),
    )
    coefficient_storage = zeros(Float32, 4)
    view_workspace = ModalFittingWorkspace(@view coefficient_storage[2:3])
    @test @inferred(process!(view_result, view_workspace, plan, input)) === view_result
    @test view_result.fitted_opd ≈ Float32.(EXPECTED_FITTED) atol=2f-6 rtol=2f-6
    @test fitting_allocation(view_result, view_workspace, plan, input) == 0

    before = (
        copy(result.input_opd), copy(result.fitted_opd), copy(result.residual_opd),
        copy(workspace.modal_coefficients),
    )
    invalid = ModalFittingInputs(fill(NaN32, 2, 2), projector, modes)
    @test_throws ArgumentError process!(result, workspace, plan, invalid)
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before
    invalid_projector = ModalFittingInputs(opd, fill(NaN32, 2, 4), modes)
    @test_throws ArgumentError process!(result, workspace, plan, invalid_projector)
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before
    invalid_modes = ModalFittingInputs(opd, projector, fill(NaN32, 4, 2))
    @test_throws ArgumentError process!(result, workspace, plan, invalid_modes)
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before
    non_one_based_input = ModalFittingInputs(
        NonOneBasedTestMatrix(Float32.(INPUT_OPD)), projector, modes,
    )
    @test_throws ArgumentError process!(result, workspace, plan, non_one_based_input)
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before
    @test_throws DimensionMismatch process!(
        result, workspace, plan, ModalFittingInputs(zeros(Float32, 1, 2), projector, modes))
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before

    aliased_input = ModalFittingInputs(result.input_opd, projector, modes)
    @test_throws ArgumentError process!(result, workspace, plan, aliased_input)
    @test (result.input_opd, result.fitted_opd, result.residual_opd,
        workspace.modal_coefficients) == before

    shared = zeros(Float32, 2, 2)
    aliased_result = ModalFittingProduct(shared, shared, zeros(Float32, 2, 2))
    @test_throws ArgumentError process!(aliased_result, workspace, plan, input)

    orthonormal_modes = Float32[
        1 0
        0 1
        0 0
        0 0
    ]
    shared_readonly_input = ModalFittingInputs(
        Float32.(INPUT_OPD),
        transpose(orthonormal_modes),
        orthonormal_modes,
    )
    @test process!(result, workspace, plan, shared_readonly_input) === result

    valid_before_overflow = (
        copy(result.input_opd), copy(result.fitted_opd), copy(result.residual_opd),
    )
    finite_overflow = ModalFittingInputs(
        fill(floatmax(Float32), 2, 2),
        fill(Float32(2), 2, 4),
        zeros(Float32, 4, 2),
    )
    @test_throws ArgumentError process!(result, workspace, plan, finite_overflow)
    @test (result.input_opd, result.fitted_opd, result.residual_opd) ==
        valid_before_overflow
end

end
