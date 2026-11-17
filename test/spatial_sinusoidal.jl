module SpatialSinusoidalTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ProbeBases
using Test

struct ShiftedCoordinates{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
end
Base.size(array::ShiftedCoordinates) = size(array.parent)
Base.axes(array::ShiftedCoordinates) =
    (0:(size(array, 1) - 1), Base.OneTo(size(array, 2)))
Base.IndexStyle(::Type{<:ShiftedCoordinates}) = IndexCartesian()
Base.getindex(array::ShiftedCoordinates, row::Int, column::Int) =
    array.parent[row + 1, column]

function spatial_fixture(::Type{T}=Float64) where {T<:AbstractFloat}
    positions = T[0 0; 0.25 0; 0.5 0; 0.75 0; 0 0.25; 0 0.5; 0 0.75]
    frequencies = T[1 0; 0 1]
    amplitudes = T[2, 3, 4, 5]
    return positions, frequencies, amplitudes
end

function spatial_plan(positions, frequencies, amplitudes;
                      normalization=:none, minimum_sampled_peak=1e-8)
    return prepare(SpatialSinusoidalPushPull(amplitudes;
        normalization, minimum_sampled_peak),
        SpatialSinusoidalSpecification(positions, frequencies))
end

@testset "spatial sine/cosine equations, actuator and command row order" begin
    positions, frequencies, amplitudes = spatial_fixture()
    plan = @inferred spatial_plan(positions, frequencies, amplitudes)
    expected = [0.0 1 0 -1 0 0 0; 1.0 0 -1 0 1 1 1;
                0.0 0 0 0 1 0 -1; 1.0 1 1 1 0 -1 0]
    @test plan.basis == expected
    @test plan.sampled_peak == ones(4)
    @test plan.sampled_rms ≈ sqrt.([2, 5, 2, 5] ./ 7) rtol=2e-15
    @test plan.normalization === :none
    @test plan.modal_plan.pattern_count == 4
    @test plan.modal_plan.corrector_coordinate_count == 7
    @test plan.modal_plan.positive_commands == expected .* amplitudes
    product = @inferred process(plan, nothing)
    commands = probe_commands(product)
    @test size(commands) == (8, 7)
    @test commands[1:2:end, :] == expected .* amplitudes
    @test commands[2:2:end, :] == -expected .* amplitudes
    @test commands[1, :] == amplitudes[1] .* expected[1, :]
    @test commands[3, :] == amplitudes[2] .* expected[2, :]
    @test commands[5, :] == amplitudes[3] .* expected[3, :]
    @test commands[7, :] == amplitudes[4] .* expected[4, :]
    @test allocate_workspace(plan) === nothing
    @test @inferred(process!(allocate_result(plan), nothing, plan, nothing)).commands == commands
    @test process!(allocate_result(plan), plan, nothing).commands == commands
    # Reordering physical positions reorders only command columns.
    order = [7, 3, 1, 5, 2, 4, 6]
    reordered = spatial_plan(positions[order, :], frequencies, amplitudes)
    @test reordered.basis == expected[:, order]
    # A prepared plan owns its arrays, independently of cold input mutation.
    positions .= 100
    frequencies .= 0
    amplitudes .= 0
    @test process(plan, nothing).commands == commands
end

@testset "explicit sampled peak normalization and amplitude envelope" begin
    positions = [0.125 0.0; 0.375 0.0; 0.625 0.0; 0.875 0.0]
    frequencies = [1.0 0.0]
    amplitudes = [2.0, 3.0]
    unnormalized = spatial_plan(positions, frequencies, amplitudes; normalization=:none)
    normalized = spatial_plan(positions, frequencies, amplitudes; normalization=:peak)
    @test unnormalized.sampled_peak ≈ fill(sqrt(0.5), 2) rtol=2e-15
    @test unnormalized.sampled_rms ≈ fill(sqrt(0.5), 2) rtol=2e-15
    @test normalized.sampled_peak == unnormalized.sampled_peak
    @test normalized.sampled_rms == unnormalized.sampled_rms
    @test normalized.basis == unnormalized.basis ./ unnormalized.sampled_peak
    raw_commands = process(unnormalized, nothing).commands
    peak_commands = process(normalized, nothing).commands
    @test vec(maximum(abs, raw_commands[1:2:end, :]; dims=2)) ≈
        amplitudes .* unnormalized.sampled_peak rtol=2e-15
    @test vec(maximum(abs, peak_commands[1:2:end, :]; dims=2)) == amplitudes
    @test all(abs.(raw_commands[1:2:end, :]) .<= amplitudes)
    @test_throws ArgumentError spatial_plan(positions, frequencies, amplitudes;
        minimum_sampled_peak=0.8)
    # The caller-declared guard is strict: equality is rejected as well.
    @test_throws ArgumentError spatial_plan(positions, frequencies, amplitudes;
        minimum_sampled_peak=minimum(unnormalized.sampled_peak))
end

@testset "spatial phase under coordinate translation and unit rescaling" begin
    positions, frequencies, amplitudes = spatial_fixture()
    original = spatial_plan(positions, frequencies, amplitudes)
    translation = [0.123, -0.2]
    shifted = spatial_plan(positions .+ translation', frequencies, amplitudes)
    for frequency in axes(frequencies, 1)
        delta = 2π * sum(frequencies[frequency, :] .* translation)
        s, c = 2 * frequency - 1, 2 * frequency
        @test shifted.basis[s, :] ≈ original.basis[s, :] .* cos(delta) .+
            original.basis[c, :] .* sin(delta) atol=1e-14 rtol=1e-14
        @test shifted.basis[c, :] ≈ original.basis[c, :] .* cos(delta) .-
            original.basis[s, :] .* sin(delta) atol=1e-14 rtol=1e-14
    end
    # Metres/cycles-per-metre and millimetres/cycles-per-millimetre agree.
    rescaled = spatial_plan(1000 .* positions, frequencies ./ 1000, amplitudes)
    @test rescaled.basis == original.basis
    # Oblique frequencies use both physical coordinates with positive phase.
    oblique_positions = [0.02 0.04; 0.13 -0.2; -0.25 0.31]
    wavevector = [2.0 -0.5]
    oblique = spatial_plan(oblique_positions, wavevector, [1.0, 1.0])
    phase = 2π .* (oblique_positions * vec(wavevector))
    @test oblique.basis[1, :] ≈ sin.(phase) atol=2e-15 rtol=2e-15
    @test oblique.basis[2, :] ≈ cos.(phase) atol=2e-15 rtol=2e-15
end

@testset "Float32, views, exact selected 277-actuator order" begin
    positions, frequencies, amplitudes = spatial_fixture(Float32)
    plan = @inferred spatial_plan(@view(positions[:, :]),
        @view(frequencies[:, :]), @view(amplitudes[:]))
    @test eltype(plan.basis) === Float32
    @test eltype(plan.sampled_rms) === Float64
    @test eltype(process(plan, nothing).commands) === Float32
    @test plan.sampled_rms ≈ sqrt.(Float32[2, 5, 2, 5] ./ 7) rtol=2eps(Float32)
    correctors = 277
    positions_277 = hcat((0:(correctors - 1)) ./ correctors,
        reverse((0:(correctors - 1)) ./ correctors))
    large = spatial_plan(positions_277, [1.0 0.0], [0.1, 0.2]; normalization=:peak)
    @test size(large.basis) == (2, 277)
    @test size(process(large, nothing).commands) == (4, 277)
    @test large.basis[1, :] ≈ sinpi.(2 .* positions_277[:, 1]) ./ large.sampled_peak[1]
end

@testset "RMS retains low-precision sample contributions" begin
    positions = repeat(Float16[0.125 0], 4096, 1)
    plan = spatial_plan(positions, Float16[1 0], Float16[1, 1])
    expected_rms = vec(sqrt.(sum(abs2, Float64.(plan.basis); dims=2) ./ 4096))
    @test plan.sampled_rms ≈ expected_rms rtol=2e-15 atol=0
    @test eltype(plan.sampled_rms) === Float64
    @test eltype(plan.sampled_peak) === Float16
    @test eltype(plan.basis) === Float16
    @test eltype(process(plan, nothing).commands) === Float16
    setprecision(BigFloat, 128) do
        positions = BigFloat[0.125 0; 0.375 0; 0.625 0; 0.875 0]
        high_precision = spatial_plan(positions, BigFloat[1 0], BigFloat[1, 1])
        @test eltype(high_precision.sampled_rms) === BigFloat
        @test high_precision.sampled_rms ≈ fill(sqrt(BigFloat(0.5)), 2) rtol=4eps(BigFloat)
    end
end

@testset "spatial policy, finite inputs, shape, and sampled degeneracy" begin
    positions, frequencies, amplitudes = spatial_fixture()
    @test_throws UndefKeywordError SpatialSinusoidalPushPull(amplitudes)
    @test_throws UndefKeywordError SpatialSinusoidalPushPull(amplitudes; normalization=:none)
    @test_throws UndefKeywordError SpatialSinusoidalPushPull(amplitudes;
        minimum_sampled_peak=1e-8)
    @test_throws ArgumentError SpatialSinusoidalPushPull(amplitudes;
        normalization=:rms, minimum_sampled_peak=1e-8)
    for guard in (-1, 0, 1, 2, NaN, Inf)
        @test_throws ArgumentError SpatialSinusoidalPushPull(amplitudes;
            normalization=:none, minimum_sampled_peak=guard)
    end
    for bad_amplitudes in (Float64[], [0.0, 1.0], [-1.0, 1.0], [NaN, 1.0], [Inf, 1.0])
        @test_throws ArgumentError SpatialSinusoidalPushPull(bad_amplitudes;
            normalization=:none, minimum_sampled_peak=1e-8)
    end
    @test_throws DimensionMismatch spatial_plan(positions, frequencies, [1.0])
    @test_throws DimensionMismatch spatial_plan(zeros(3, 3), frequencies, amplitudes)
    @test_throws DimensionMismatch spatial_plan(positions, zeros(2, 1), amplitudes)
    @test_throws ArgumentError spatial_plan(zeros(0, 2), frequencies, amplitudes)
    @test_throws ArgumentError spatial_plan(positions, zeros(0, 2), Float64[])
    @test_throws ArgumentError spatial_plan(ShiftedCoordinates(positions), frequencies, amplitudes)
    @test_throws ArgumentError spatial_plan(positions, ShiftedCoordinates(frequencies), amplitudes)
    @test_throws ArgumentError SpatialSinusoidalPushPull(AbstractFloat[1.0, 1.0];
        normalization=:none, minimum_sampled_peak=1e-8)
    @test_throws ArgumentError spatial_plan(Matrix{AbstractFloat}(positions), frequencies, amplitudes)
    @test_throws ArgumentError spatial_plan(positions, Matrix{AbstractFloat}(frequencies), amplitudes)
    for bad in (NaN, Inf, -Inf)
        invalid_positions = copy(positions)
        invalid_positions[1, 1] = bad
        @test_throws ArgumentError spatial_plan(invalid_positions, frequencies, amplitudes)
        invalid_frequencies = copy(frequencies)
        invalid_frequencies[1, 1] = bad
        @test_throws ArgumentError spatial_plan(positions, invalid_frequencies, amplitudes)
    end
    @test_throws ArgumentError spatial_plan([1e308 0.0], [2.0 0.0], [1.0, 1.0])
    @test_throws ArgumentError spatial_plan(positions, [0.0 0.0], [1.0, 1.0])
    @test_throws ArgumentError spatial_plan([0.0 0.0; 0.5 0.0], [1.0 0.0], [1.0, 1.0])
    @test_throws ArgumentError spatial_plan([0.25 0.0; 0.75 0.0], [1.0 0.0], [1.0, 1.0])
    @test_throws ArgumentError spatial_plan([1e-12 0.0; 2e-12 0.0], [1.0 0.0], [1.0, 1.0])
    # Redundant frequencies are accepted: the guard does not claim global rank.
    duplicate = spatial_plan(positions, [1.0 0.0; 1.0 0.0], amplitudes)
    @test duplicate.basis[1:2, :] == duplicate.basis[3:4, :]
    method = SpatialSinusoidalPushPull(amplitudes;
        normalization=:none, minimum_sampled_peak=1e-8)
    method.amplitudes[1] = 0
    @test_throws ArgumentError prepare(method, SpatialSinusoidalSpecification(positions, frequencies))
    plan = spatial_plan(positions, frequencies, amplitudes)
    @test_throws DimensionMismatch process!(ProbeCommandBasisProduct(zeros(7, 7)),
        nothing, plan, nothing)
end

end
