module TomographyTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.Tomography
import KernelAbstractions
using LinearAlgebra
using SpecialFunctions
using TOML
using Test

const AOS_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_tomography_covariance.toml"))
const AOS_MULTISOURCE_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_von_karman_multisource.toml"))

function fixture_complex_coordinates(table)
    shape = Tuple(Int.(table["shape"]))
    return reshape(complex.(table["real"], table["imag"]), shape)
end

row_major_matrix(values, rows, columns) = permutedims(reshape(values, columns, rows))

struct ZeroBasedMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    storage::A
end

struct ZeroBasedCoordinates{T,A<:AbstractArray{T,3}} <: AbstractArray{T,3}
    storage::A
end

Base.size(coordinates::ZeroBasedCoordinates) = size(coordinates.storage)
Base.axes(coordinates::ZeroBasedCoordinates) = (
    0:(size(coordinates, 1) - 1),
    0:(size(coordinates, 2) - 1),
    0:(size(coordinates, 3) - 1),
)
Base.getindex(coordinates::ZeroBasedCoordinates, sample::Int, source::Int, layer::Int) =
    coordinates.storage[sample + 1, source + 1, layer + 1]

const VON_KARMAN_SOURCE = (
    repository="AdaptiveOpticsSim.jl",
    revision="21ebf111d9b9b3ff37531cf67a7c8deaf19d62a5",
    path="src/tomography/reconstructors.jl",
    equation="von Kármán phase covariance using gamma and besselk(5/6, ·)",
)

function von_karman_coordinates(::Type{T}=Float64) where {T<:AbstractFloat}
    guide = Array{Complex{T}}(undef, 2, 2, 2)
    guide[:, :, 1] .= Complex{T}[0 0 + 0.2im; 0.1 0.1 + 0.2im]
    guide[:, :, 2] .= Complex{T}[0.03 + 0.01im -0.02 + 0.07im;
                                  0.08 + 0.01im 0.02 + 0.07im]
    fit = Array{Complex{T}}(undef, 3, 2, 2)
    fit[:, :, 1] .= Complex{T}[0 0.04 + 0.12im;
                                0.06 + 0.03im 0.10 + 0.12im;
                                0.12 - 0.02im 0.16 + 0.12im]
    fit[:, :, 2] .= Complex{T}[0.025 + 0.015im -0.01 + 0.055im;
                                0.065 + 0.025im 0.045 + 0.055im;
                                0.115 - 0.015im 0.095 + 0.055im]
    return guide, fit
end

function direct_von_karman_constants(r0_m::T, L0_m::T) where {T<:AbstractFloat}
    base = (T(24) * gamma(T(6) / T(5)) / T(5))^(T(5) / T(6))
    scale = (L0_m / r0_m)^(T(5) / T(3))
    covariance_constant = base * gamma(T(11) / T(6)) /
        (T(2)^(T(5) / T(6)) * T(π)^(T(8) / T(3))) * scale
    zero_separation_covariance = base * gamma(T(11) / T(6)) * gamma(T(5) / T(6)) /
        (T(2) * T(π)^(T(8) / T(3))) * scale
    return covariance_constant, zero_separation_covariance, inv(L0_m)
end

function direct_von_karman_layer(first_coordinate::Complex{T}, second_coordinate::Complex{T},
    covariance_constant::T, zero_separation_covariance::T, inverse_outer_scale_m::T,
    fractional_cn2::T) where {T<:AbstractFloat}
    separation_m = abs(first_coordinate - second_coordinate)
    iszero(separation_m) && return zero_separation_covariance * fractional_cn2
    order = T(5) / T(6)
    argument = T(2π) * separation_m * inverse_outer_scale_m
    return covariance_constant * T(argument^order * besselk(order, argument)) *
        fractional_cn2
end

function direct_auto_covariance(guide, r0_m, L0_m, fractional_cn2)
    guide_sample_count, guide_source_count, layer_count = size(guide)
    out = zeros(eltype(fractional_cn2), guide_sample_count * guide_source_count,
        guide_sample_count * guide_source_count)
    constants = direct_von_karman_constants(r0_m, L0_m)
    for second_source in 1:guide_source_count, second_sample in 1:guide_sample_count,
        first_source in 1:guide_source_count, first_sample in 1:guide_sample_count,
        layer in 1:layer_count
        row = first_sample + (first_source - 1) * guide_sample_count
        column = second_sample + (second_source - 1) * guide_sample_count
        out[row, column] += direct_von_karman_layer(
            guide[first_sample, first_source, layer],
            guide[second_sample, second_source, layer],
            constants...,
            fractional_cn2[layer],
        )
    end
    return out
end

function direct_cross_covariance(guide, fit, r0_m, L0_m, fractional_cn2)
    guide_sample_count, guide_source_count, layer_count = size(guide)
    fit_sample_count, fit_source_count, _ = size(fit)
    out = zeros(eltype(fractional_cn2), fit_source_count, fit_sample_count,
        guide_sample_count * guide_source_count)
    constants = direct_von_karman_constants(r0_m, L0_m)
    for guide_source in 1:guide_source_count, guide_sample in 1:guide_sample_count,
        fit_source in 1:fit_source_count, fit_sample in 1:fit_sample_count,
        layer in 1:layer_count
        flattened_guide_sample = guide_sample +
            (guide_source - 1) * guide_sample_count
        out[fit_source, fit_sample, flattened_guide_sample] += direct_von_karman_layer(
            fit[fit_sample, fit_source, layer],
            guide[guide_sample, guide_source, layer],
            constants...,
            fractional_cn2[layer],
        )
    end
    return out
end

Base.size(matrix::ZeroBasedMatrix) = size(matrix.storage)
Base.axes(matrix::ZeroBasedMatrix) = (
    0:(size(matrix, 1) - 1),
    0:(size(matrix, 2) - 1),
)

function tomography_input(::Type{T}=Float64) where {T<:AbstractFloat}
    projection = T[1 0 -1; 0.5 2 1]
    phase_covariance = T[2 0.25 0; 0.25 1.5 0.1; 0 0.1 1]
    fit_phase_covariance = T[1 0.5 -0.25; -0.5 0.75 2]
    measurement_noise_covariance = T[0.4 0.1; 0.1 0.6]
    return CovarianceReconstructorInputs(
        projection,
        phase_covariance,
        fit_phase_covariance,
        measurement_noise_covariance,
    )
end

function direct_reconstructor(input)
    P = input.projection
    return input.fit_phase_covariance * transpose(P) /
        (P * input.phase_covariance * transpose(P) +
         input.measurement_noise_covariance)
end

@testset "explicit covariance tomography equation" begin
    input = tomography_input()
    specification = CovarianceReconstructorSpecification(3, 2, 2)
    direct = direct_reconstructor(input)

    plan = @inferred prepare(CovarianceReconstructor(), specification)
    result = allocate_result(plan)
    @test allocate_workspace(plan) === nothing
    @test @inferred(process!(result, nothing, plan, input)) === result
    @test reconstructor(result) === result.matrix
    @test result.matrix ≈ direct atol=1e-13 rtol=1e-13
    @test !result.diagnostics.used_lu_fallback
    @test result.diagnostics.system_matrix_norm ≈
        norm(input.projection * input.phase_covariance * input.projection' +
             input.measurement_noise_covariance)
    @test result.diagnostics.reconstructor_norm ≈ norm(direct)
    @test process(plan, input).matrix ≈ direct atol=1e-13 rtol=1e-13

    float32_input = tomography_input(Float32)
    float32_plan = prepare(
        CovarianceReconstructor(),
        CovarianceReconstructorSpecification(3, 2, 2, Float32),
    )
    float32_result = process(float32_plan, float32_input)
    @test eltype(float32_result.matrix) === Float32
    @test float32_result.matrix ≈ direct_reconstructor(float32_input) atol=3f-5 rtol=3f-5
end

@testset "AOS von Kármán covariance equation and order" begin
    @test VON_KARMAN_SOURCE.repository == "AdaptiveOpticsSim.jl"
    @test VON_KARMAN_SOURCE.revision == "21ebf111d9b9b3ff37531cf67a7c8deaf19d62a5"
    @test VON_KARMAN_SOURCE.path == "src/tomography/reconstructors.jl"
    @test occursin("besselk", VON_KARMAN_SOURCE.equation)

    guide, fit = von_karman_coordinates()
    r0_m = 0.16
    L0_m = 25.0
    fractional_cn2 = [0.35, 0.65]
    direct_cxx = direct_auto_covariance(guide, r0_m, L0_m, fractional_cn2)
    direct_cox = direct_cross_covariance(guide, fit, r0_m, L0_m, fractional_cn2)

    auto_specification = VonKarmanAutoCovarianceSpecification(
        guide, r0_m, L0_m, fractional_cn2)
    auto_plan = @inferred prepare(VonKarmanAutoCovariance(), auto_specification)
    auto_result = allocate_result(auto_plan)
    @test allocate_workspace(auto_plan) === nothing
    @test @inferred(process!(auto_result, nothing, auto_plan, nothing)) === auto_result
    @test auto_covariance(auto_result) === auto_result.Cxx
    @test auto_result.Cxx ≈ direct_cxx rtol=2e-13 atol=2e-13
    @test auto_result.Cxx ≈ transpose(auto_result.Cxx) rtol=2e-13 atol=2e-13
    @test process(auto_plan, nothing).Cxx ≈ direct_cxx rtol=2e-13 atol=2e-13
    @test !Base.mightalias(auto_plan.guide_coordinates, guide)

    cross_specification = VonKarmanCrossCovarianceSpecification(
        guide, fit, r0_m, L0_m, fractional_cn2)
    cross_plan = @inferred prepare(VonKarmanCrossCovariance(), cross_specification)
    cross_result = allocate_result(cross_plan)
    @test allocate_workspace(cross_plan) === nothing
    @test @inferred(process!(cross_result, nothing, cross_plan, nothing)) === cross_result
    @test cross_covariance(cross_result) === cross_result.Cox
    @test size(cross_result.Cox) == (2, 3, 4)
    @test cross_result.Cox ≈ direct_cox rtol=2e-13 atol=2e-13
    @test process(cross_plan, nothing).Cox ≈ direct_cox rtol=2e-13 atol=2e-13
    @test !Base.mightalias(cross_plan.guide_coordinates, guide)
    @test !Base.mightalias(cross_plan.fit_coordinates, fit)

    # This explicit transformed-coordinate case has two guide sources, two
    # layers, and two fit sources. The third Cox axis is guide-source blocks.
    @test cross_result.Cox[2, 3, 1] == direct_cox[2, 3, 1]
    @test cross_result.Cox[2, 3, 3] == direct_cox[2, 3, 3]
    @test cross_result.Cox[1, 1, 1] > cross_result.Cox[2, 3, 4]

    guide32, fit32 = von_karman_coordinates(Float32)
    auto32 = process(prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide32, 0.16f0, 25f0,
            Float32[0.35, 0.65])), nothing)
    cross32 = process(prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(guide32, fit32, 0.16f0, 25f0,
            Float32[0.35, 0.65])), nothing)
    @test eltype(auto32.Cxx) === Float32
    @test eltype(cross32.Cox) === Float32
    @test auto32.Cxx ≈ Float32.(direct_cxx) rtol=5f-5 atol=5f-5
    @test cross32.Cox ≈ Float32.(direct_cox) rtol=5f-5 atol=5f-5
end

@testset "AOS von Kármán covariance fixture anchor" begin
    # The 11-point masked AOS grid has 4 m spacing across its 40 m support.
    # These entries anchor this direct-coordinate target to the frozen AOS
    # `aos_s6_tomography_cpu.toml` Cxx product without an AOS adapter.
    guide = reshape(ComplexF64[0, 4], 2, 1, 1)
    plan = prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.2, 25.0, [1.0]))
    Cxx = process(plan, nothing).Cxx
    @test Cxx[1, 1] ≈ 269.7322809744831 rtol=1e-12
    @test Cxx[2, 1] ≈ 144.37670563082148 rtol=1e-12
end

@testset "Frozen AOS multi-source von Kármán covariance" begin
    fixture = AOS_MULTISOURCE_FIXTURE
    @test fixture["schema"] == "test.adaptive-optics-calibration/aos-von-karman-multisource/1"
    @test fixture["source_revision"] == VON_KARMAN_SOURCE.revision
    @test fixture["physical_source"]["fit_src_height_m"] < Inf
    guide = fixture_complex_coordinates(fixture["guide_coordinates"])
    fit = fixture_complex_coordinates(fixture["fit_coordinates"])
    @test size(guide) == (2, 2, 2)
    @test size(fit) == (2, 4, 2)
    @test guide[:, :, 1] != guide[:, :, 2]
    @test fit[:, :, 1] != fit[:, :, 2]
    r0_m = fixture["r0_m"]
    L0_m = fixture["L0_m"]
    fractional_cn2 = fixture["fractional_cn2"]
    Cxx = process(prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            guide, r0_m, L0_m, fractional_cn2)), nothing).Cxx
    Cox = process(prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(
            guide, fit, r0_m, L0_m, fractional_cn2)), nothing).Cox
    expected_cxx = reshape(fixture["cxx"]["values"],
        Tuple(Int.(fixture["cxx"]["shape"])))
    expected_cox = reshape(fixture["cox"]["values"],
        Tuple(Int.(fixture["cox"]["shape"])))
    @test Cxx ≈ expected_cxx rtol=2e-12 atol=2e-12
    @test Cox ≈ expected_cox rtol=2e-12 atol=2e-12
end

@testset "von Kármán zero-separation rank" begin
    repeated = reshape(ComplexF64[0, 0], 2, 1, 1)
    Cxx = process(prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            repeated, 0.2, 25.0, [1.0])), nothing).Cxx
    @test Cxx == fill(Cxx[1, 1], 2, 2)
    @test rank(Cxx) == 1
end

@testset "von Kármán covariance validation" begin
    guide, fit = von_karman_coordinates()
    valid_fractional_cn2 = [0.35, 0.65]
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.0, 25.0, valid_fractional_cn2))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, Inf, valid_fractional_cn2))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            reshape(Complex{BigFloat}[0], 1, 1, 1),
            big"0.2", big"25.0", BigFloat[1]))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            reshape(ComplexF64[0], 1, 1, 1),
            nextfloat(0.0), nextfloat(0.0), [1.0]))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, [0.4, 0.5]))
    approximate_weights = [0.4, 0.6000001]
    approximate_auto = prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            guide, 0.16, 25.0, approximate_weights))
    approximate_cross = prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(
            guide, fit, 0.16, 25.0, approximate_weights))
    @test approximate_auto.fractional_cn2 == approximate_weights
    @test approximate_cross.fractional_cn2 == approximate_weights
    @test auto_covariance(process(approximate_auto, nothing)) ≈
        direct_auto_covariance(guide, 0.16, 25.0, approximate_weights)
    @test cross_covariance(process(approximate_cross, nothing)) ≈
        direct_cross_covariance(guide, fit, 0.16, 25.0, approximate_weights)
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, [0.4, 0.600003]))
    @test_throws ArgumentError prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(
            guide, fit, 0.16, 25.0, [0.4, 0.600003]))
    guide32 = ComplexF32.(guide)
    fit32 = ComplexF32.(fit)
    roundoff_weights32 = Float32[0.4, 0.6001]
    @test prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            guide32, 0.16f0, 25f0, roundoff_weights32)).fractional_cn2 ==
        roundoff_weights32
    @test prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(
            guide32, fit32, 0.16f0, 25f0, roundoff_weights32)).fractional_cn2 ==
        roundoff_weights32
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(
            guide32, 0.16f0, 25f0, Float32[0.4, 0.601]))
    @test_throws ArgumentError prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(
            guide32, fit32, 0.16f0, 25f0, Float32[0.4, 0.601]))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, [1.2, -0.2]))
    @test_throws DimensionMismatch prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, [1.0]))
    @test_throws ArgumentError prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(ZeroBasedCoordinates(guide), 0.16, 25.0,
            valid_fractional_cn2))

    @test_throws DimensionMismatch prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(guide, fit[:, :, 1:1], 0.16, 25.0,
            valid_fractional_cn2))
    nonfinite_fit = copy(fit)
    nonfinite_fit[1] = Complex(NaN, 0.0)
    @test_throws ArgumentError prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(guide, nonfinite_fit, 0.16, 25.0,
            valid_fractional_cn2))

    auto_plan = prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, valid_fractional_cn2))
    @test_throws DimensionMismatch process!(VonKarmanAutoCovarianceProduct(zeros(3, 3)),
        nothing, auto_plan, nothing)
    cross_plan = prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(guide, fit, 0.16, 25.0, valid_fractional_cn2))
    @test_throws DimensionMismatch process!(VonKarmanCrossCovarianceProduct(zeros(2, 3, 3)),
        nothing, cross_plan, nothing)

    far_apart = reshape(ComplexF64[-floatmax(Float64), floatmax(Float64)], 2, 1, 1)
    far_plan = prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(far_apart, 0.2, 25.0, [1.0]))
    previous = VonKarmanAutoCovarianceProduct(fill(-42.0, 2, 2))
    @test_throws ArgumentError process!(previous, nothing, far_plan, nothing)
    @test all(==(-42.0), previous.Cxx)
end

@testset "scalar covariance products cannot alias prepared geometry" begin
    guide = fill(0.0 + 0.0im, 1, 1, 1)
    fit = fill(0.01 + 0.02im, 1, 1, 1)
    auto_plan = prepare(VonKarmanAutoCovariance(),
        VonKarmanAutoCovarianceSpecification(guide, 0.16, 25.0, [1.0]))
    cross_plan = prepare(VonKarmanCrossCovariance(),
        VonKarmanCrossCovarianceSpecification(guide, fit, 0.16, 25.0, [1.0]))
    auto_expected = process(auto_plan, nothing).Cxx[1]
    cross_expected = process(cross_plan, nothing).Cox[1]

    for (plan, product, expected) in (
        (auto_plan, VonKarmanAutoCovarianceProduct(
            reshape(auto_plan.fractional_cn2, 1, 1)), auto_expected),
        (cross_plan, VonKarmanCrossCovarianceProduct(
            reshape(cross_plan.fractional_cn2, 1, 1, 1)), cross_expected),
    )
        previous_weights = copy(plan.fractional_cn2)
        @test_throws ArgumentError process!(product, nothing, plan, nothing)
        @test plan.fractional_cn2 == previous_weights
        normal = allocate_result(plan)
        @test process!(normal, nothing, plan, nothing) === normal
        actual = plan isa VonKarmanAutoCovariancePlan ? normal.Cxx[1] : normal.Cox[1]
        @test actual == expected
    end
end

@testset "AOS covariance tomography fixture" begin
    @test AOS_FIXTURE["schema"] == "test.adaptive-optics-sim/tomography-covariance/1"
    @test AOS_FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test AOS_FIXTURE["source_revision"] ==
        "d30db3f78d7fad77d0d5e9212b3eee1a34d86244"
    @test AOS_FIXTURE["source_path"] == "src/tomography/reconstructors.jl"
    @test AOS_FIXTURE["source_test_path"] == "test/tomography.jl"
    @test AOS_FIXTURE["source_generation"] ==
        "Executed the AOS compact model case and recorded its computed operators"
    @test AOS_FIXTURE["source_product"] == "model.operators.recstat"
    @test AOS_FIXTURE["excluded_model_scale"] == "asterism.wavelength_m / 2"

    phase = AOS_FIXTURE["phase_count"]
    measurements = AOS_FIXTURE["measurement_count"]
    fit = AOS_FIXTURE["fit_count"]
    input = CovarianceReconstructorInputs(
        row_major_matrix(AOS_FIXTURE["projection"], measurements, phase),
        row_major_matrix(AOS_FIXTURE["phase_covariance"], phase, phase),
        row_major_matrix(AOS_FIXTURE["fit_phase_covariance"], fit, phase),
        row_major_matrix(AOS_FIXTURE["measurement_noise_covariance"], measurements, measurements),
    )
    expected = row_major_matrix(
        AOS_FIXTURE["expected_recstat"], fit, measurements)
    plan = prepare(
        CovarianceReconstructor(),
        CovarianceReconstructorSpecification(phase, measurements, fit),
    )
    result = process(plan, input)
    tolerance = AOS_FIXTURE["absolute_tolerance_f64"]
    @test result.matrix ≈ expected atol=tolerance rtol=tolerance
    @test result.matrix ≈ direct_reconstructor(input) atol=tolerance rtol=tolerance
    @test !result.diagnostics.used_lu_fallback
end

@testset "covariance tomography fallback and validation" begin
    fallback_input = CovarianceReconstructorInputs(
        Matrix{Float64}(I, 2, 2),
        [1.0 0.0; 0.0 -2.0],
        [2.0 -1.0],
        zeros(2, 2),
    )
    plan = prepare(
        CovarianceReconstructor(),
        CovarianceReconstructorSpecification(2, 2, 1),
    )
    fallback_result = process(plan, fallback_input)
    @test fallback_result.diagnostics.used_lu_fallback
    @test fallback_result.matrix ≈ direct_reconstructor(fallback_input)

    singular_input = CovarianceReconstructorInputs(
        Matrix{Float64}(I, 2, 2),
        zeros(2, 2),
        [1.0 -1.0],
        zeros(2, 2),
    )
    preserved = allocate_result(plan)
    fill!(preserved.matrix, 17.0)
    @test_throws SingularException process!(preserved, nothing, plan, singular_input)
    @test all(==(17.0), preserved.matrix)

    asymmetric_input = CovarianceReconstructorInputs(
        Matrix{Float64}(I, 2, 2),
        [2.0 0.0; 1.0 2.0],
        [0.0 1.0],
        zeros(2, 2),
    )
    preserved.diagnostics = CovarianceReconstructorDiagnostics(false, 5.0, 7.0)
    @test_throws ArgumentError process!(preserved, nothing, plan, asymmetric_input)
    @test all(==(17.0), preserved.matrix)
    @test preserved.diagnostics.system_matrix_norm == 5.0
    @test preserved.diagnostics.reconstructor_norm == 7.0

    nearly_symmetric_input = CovarianceReconstructorInputs(
        [0.0 1.0 0.0; 0.0 0.0 1.0],
        [1.0 0.0 0.0; 0.0 1e-10 0.0; 0.0 1e-9 1e-10],
        [0.0 0.0 1.0],
        zeros(2, 2),
    )
    nearly_symmetric_plan = prepare(
        CovarianceReconstructor(),
        CovarianceReconstructorSpecification(3, 2, 1),
    )
    nearly_symmetric_result = process(nearly_symmetric_plan, nearly_symmetric_input)
    symmetrized_phase = [1.0 0.0 0.0; 0.0 1e-10 5e-10; 0.0 5e-10 1e-10]
    expected_symmetrized = nearly_symmetric_input.fit_phase_covariance *
        transpose(nearly_symmetric_input.projection) /
        (nearly_symmetric_input.projection * symmetrized_phase *
         transpose(nearly_symmetric_input.projection))
    @test nearly_symmetric_result.matrix ≈ expected_symmetrized rtol=1e-12
    @test nearly_symmetric_result.diagnostics.used_lu_fallback

    valid = tomography_input()
    valid_plan = prepare(
        CovarianceReconstructor(),
        CovarianceReconstructorSpecification(3, 2, 2),
    )
    @test_throws DimensionMismatch process!(valid_plan |> allocate_result, nothing, valid_plan,
        CovarianceReconstructorInputs(valid.projection[:, 1:2], valid.phase_covariance,
            valid.fit_phase_covariance, valid.measurement_noise_covariance))
    @test_throws ArgumentError process!(valid_plan |> allocate_result, nothing, valid_plan,
        CovarianceReconstructorInputs(fill(NaN, 2, 3), valid.phase_covariance,
            valid.fit_phase_covariance, valid.measurement_noise_covariance))
    zero_based = ZeroBasedMatrix(valid.projection)
    @test_throws ArgumentError process!(valid_plan |> allocate_result, nothing, valid_plan,
        CovarianceReconstructorInputs(zero_based, valid.phase_covariance,
            valid.fit_phase_covariance, valid.measurement_noise_covariance))
    @test_throws ArgumentError CovarianceReconstructorSpecification(0, 2, 2)
    @test_throws ArgumentError CovarianceReconstructorSpecification(2, 0, 2)
    @test_throws ArgumentError CovarianceReconstructorSpecification(2, 2, 0)
end

@testset "Portable covariance reconstructor kernels" begin
    ext = Base.get_extension(AdaptiveOpticsCalibration, :AdaptiveOpticsCalibrationKernelAbstractionsExt)
    plan = ext.CovarianceReconstructorKernelPlan(KernelExecution(
        CovarianceReconstructorSpecification(3, 2, 1), KernelAbstractions.CPU(); workgroup_size=4))
    projection = [1.0 0.0 0.5; 0.0 1.0 0.0]
    phase = [1.0 0.2 0.3; 0.2 1.0 0.1; 0.3 0.1 0.9]
    cross = [0.2 0.4 0.7]
    # The second invertible system is indefinite and requires the LU fallback.
    for (noise, use_lu) in (([0.2 0.0; 0.0 0.1], false), ([0.2 0.0; 0.0 -2.0], true))
        input = CovarianceReconstructorInputs(projection, phase, cross, noise)
        result = process(plan, input)
        system = projection * phase * projection' + noise
        @test result.matrix ≈ (cross * projection') / system
        @test result.diagnostics.used_lu_fallback == use_lu
        previous = copy(result.matrix)
        bad = CovarianceReconstructorInputs(projection, phase, cross, [1.0 0.1; 0.5 1.0])
        @test_throws ArgumentError process!(result, plan, bad)
        @test result.matrix == previous
    end
end

end
