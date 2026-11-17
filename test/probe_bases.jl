module ProbeBasisTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ProbeBases
using LinearAlgebra
using Test
using TOML

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "probe_bases.toml"))

function row_major_matrix(values, rows, columns, ::Type{T}=Float32) where {T}
    return permutedims(reshape(T.(values), columns, rows))
end

function commands(method)
    plan = prepare(method, nothing)
    result = allocate_result(plan)
    process!(result, allocate_workspace(plan), plan, nothing)
    return probe_commands(result), plan
end

@testset "modal and zonal push-pull command bases" begin
    @test FIXTURE["schema"] == "test.adaptive-optics-calibration/probe-bases/1"
    @test FIXTURE["source_repository"] == "JuliaFilterGraph.jl"
    @test FIXTURE["source_revision"] ==
        "7e9587b3ea059fe2271973a92b182c31b211c25b"
    @test FIXTURE["source_path"] == "fixtures/algorithms/calibration_probe.toml"
    @test FIXTURE["source_sha256"] ==
        "b0fdaf1f5650c6c8c12e20a6413213ab00f607b91dd8ef33d36d7fa61c80da8b"
    @test FIXTURE["upstream_fgn_revision"] ==
        "3d49237b3c9f4423f120b52bd2761cd5a90f5e94"

    modal_fixture = FIXTURE["modal"]
    modal_positive = row_major_matrix(
        modal_fixture["positive_commands"],
        modal_fixture["pattern_count"],
        modal_fixture["corrector_coordinate_count"],
    )
    modal, modal_plan = commands(ModalPushPull(modal_positive))
    modal_expected = row_major_matrix(
        modal_fixture["expected_commands"],
        2 * modal_fixture["pattern_count"],
        modal_fixture["corrector_coordinate_count"],
    )
    @test modal == modal_expected
    @test collect(reinterpret(UInt32, vec(permutedims(modal)))) ==
        UInt32.(modal_fixture["expected_bits"])
    @test modal_plan.pattern_count == modal_fixture["pattern_count"]

    zonal_fixture = FIXTURE["zonal"]
    zonal, zonal_plan = commands(ZonalPushPull(
        Float32.(zonal_fixture["amplitudes"]),
    ))
    zonal_expected = row_major_matrix(
        zonal_fixture["expected_commands"],
        2 * zonal_fixture["pattern_count"],
        zonal_fixture["corrector_coordinate_count"],
    )
    @test zonal == zonal_expected
    @test collect(reinterpret(UInt32, vec(permutedims(zonal)))) ==
        UInt32.(zonal_fixture["expected_bits"])
    @test zonal_plan.pattern_count == zonal_fixture["pattern_count"]
end

@testset "balanced Hadamard command basis" begin
    fixture = FIXTURE["hadamard"]
    hadamard, plan = commands(HadamardPushPull(Float32.(fixture["amplitudes"])))
    expected = row_major_matrix(
        fixture["expected_commands"],
        2 * fixture["pattern_count"],
        fixture["corrector_coordinate_count"],
    )
    @test hadamard == expected
    @test collect(reinterpret(UInt32, vec(permutedims(hadamard)))) ==
        UInt32.(fixture["expected_bits"])
    positive = @view hadamard[1:2:end, :]
    amplitudes = Float32.(fixture["amplitudes"])
    @test positive' * positive ≈
        Float32(fixture["pattern_count"]) .* Diagonal(amplitudes .^ 2)
    @test plan.pattern_count == fixture["pattern_count"]
end

@testset "probe bases recover a known interaction matrix" begin
    interaction = [1.0 -3.0 2.0; 0.5 4.0 -2.0]
    offset = [100.0, -30.0]
    commands_matrix, _ = commands(HadamardPushPull([0.25, 2.0, 0.5]))
    responses = commands_matrix * interaction'
    responses .+= offset'

    interaction_plan = prepare(
        AdaptiveOpticsCalibration.InteractionMatrices.HadamardPushPull(
            [0.25, 2.0, 0.5],
        ),
        AdaptiveOpticsCalibration.InteractionMatrices.ResponseSpecification(2),
    )
    result = allocate_result(interaction_plan)
    process!(result, allocate_workspace(interaction_plan), interaction_plan, responses)
    @test result.matrix ≈ interaction atol=1e-12
end

@testset "probe-basis validation" begin
    @test_throws ArgumentError prepare(ZonalPushPull(Float64[]), nothing)
    @test_throws ArgumentError prepare(HadamardPushPull([0.0]), nothing)
    @test_throws ArgumentError prepare(ModalPushPull(zeros(1, 2)), nothing)
    @test_throws ArgumentError prepare(ModalPushPull([1.0 NaN]), nothing)

    plan = prepare(ZonalPushPull([1.0, 2.0]), nothing)
    result = ProbeCommandBasisProduct(zeros(3, 2))
    @test_throws DimensionMismatch process!(result, nothing, plan, nothing)
end

end
