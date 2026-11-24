using Test
import AdaptiveOpticsCalibration as AOC
import AdaptiveOpticsCalibration.OpticalGains as OG

@testset "Bounded optical-gain correction from modal dither" begin
    α = 0.5
    method = OG.OpticalGainUpdate(; integrator_gain=α, response_floor=0.125,
        reference_floor=0.01, phase_tolerance=0.25)
    phases = [0.4, -0.3, 0.6, 0.2, 0.7, -0.3, 0.1, 0.05, 0.1]
    reference = 2cis.(phases)
    reference[1] *= -1  # The reconstruction sign is part of the reference.
    normalized = ComplexF64[0.8, 1.5, 0.5, 2.0, 0.05, -1, cis(0.6), 1, 0]
    mode_count = length(reference)
    lower = [0.4, 0.4, 0.4, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4]
    upper = [2.2, 2.2, 1.25, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]
    specification = OG.OpticalGainUpdateSpecification(reference, lower, upper)
    plan = AOC.prepare(method, specification)

    time = (0:63) ./ 128
    frequencies = collect(2:2:2mode_count)
    command_phases = 2π .* time .* transpose(frequencies)
    command_amplitude = ones(mode_count)
    command_amplitude[8] = 1e-9
    command = cos.(command_phases) .* transpose(command_amplitude)
    response = real.(cis.(command_phases) .* transpose(command_amplitude .* reference .* normalized))
    dither_method = OG.ModalDitherResponse(; command_amplitude_floor=1e-6)
    dither_plan = AOC.prepare(dither_method,
        OG.ModalDitherResponseSpecification(frequencies, time; sample_rate=128))
    dither = AOC.process(dither_plan, OG.ModalDitherResponseInputs(command, response))
    previous = ones(mode_count)
    inputs = OG.OpticalGainUpdateInputs(dither, previous)
    result = AOC.process(plan, inputs)
    @test result.valid == [true, true, true, true, false, false, false, false, false]
    @test result.correction ≈ [1.125, 5 / 6, 1.25, 0.9, 1, 1, 1, 1, 1] atol=1e-14
    @test result.optical_gains ≈ [real.(normalized[1:7]); 0; 0] atol=1e-14
    @test previous == ones(mode_count)
    @test abs(1 - result.correction[1] * 0.8) < abs(1 - previous[1] * 0.8)
    @test abs(1 - result.correction[2] * 1.5) < abs(1 - previous[2] * 1.5)

    dither_valid = BitVector(dither.valid)
    direct = AOC.process(plan, OG.OpticalGainUpdateInputs(dither.response_ratios, dither_valid, previous))
    @test direct.correction == result.correction
    @test direct.optical_gains == result.optical_gains
    @test direct.valid == result.valid
    dither_valid[1] = false
    held = AOC.process(plan, OG.OpticalGainUpdateInputs(dither.response_ratios, dither_valid, previous))
    @test !held.valid[1] && held.correction[1] == previous[1] && iszero(held.optical_gains[1])
    @test_throws DimensionMismatch AOC.process(plan,
        OG.OpticalGainUpdateInputs(dither.response_ratios, [true], previous))

    # A rolling caller can use two disjoint buffers and hold invalid modes.
    other = AOC.allocate_result(plan)
    for _ in 1:28
        @test AOC.process!(other, nothing, plan,
            OG.OpticalGainUpdateInputs(dither, result.correction)) === other
        result, other = other, result
    end
    @test result.correction ≈ [1.25, 2 / 3, 1.25, 0.9, 1, 1, 1, 1, 1] atol=1e-9

    saved = copy(result.correction)
    bad_previous = copy(previous)
    bad_previous[end] = NaN
    @test_throws ArgumentError AOC.process!(result, plan,
        OG.OpticalGainUpdateInputs(dither, bad_previous))
    @test result.correction == saved
    @test_throws ArgumentError AOC.process!(result, plan,
        OG.OpticalGainUpdateInputs(dither, result.correction))
    @test_throws DimensionMismatch AOC.process(plan, OG.OpticalGainUpdateInputs(dither, [1.0]))

    reference[1] = 0
    lower[4] = 0.4
    upper[3] = 2.2
    @test AOC.process(plan, inputs).correction ≈ [1.125, 5 / 6, 1.25, 0.9, 1, 1, 1, 1, 1]
    @test_throws ArgumentError AOC.prepare(method, specification)
    @test_throws ArgumentError AOC.prepare(method,
        OG.OpticalGainUpdateSpecification([0.01], [0.5], [2.0]))
    @test_throws ArgumentError AOC.prepare(method,
        OG.OpticalGainUpdateSpecification([1.0], [0.0], [2.0]))
    @test_throws ArgumentError AOC.prepare(method,
        OG.OpticalGainUpdateSpecification([1.0], [2.0], [1.0]))
    @test_throws DimensionMismatch AOC.prepare(method,
        OG.OpticalGainUpdateSpecification([1.0], [0.5, 1.5], [2.0]))
    @test_throws ArgumentError OG.OpticalGainUpdate(; integrator_gain=0, response_floor=0,
        reference_floor=0, phase_tolerance=0)
    @test_throws ArgumentError OG.OpticalGainUpdate(; integrator_gain=1, response_floor=0,
        reference_floor=0, phase_tolerance=π / 2)

    # Exact boundary and corrupt-telemetry checks use an unrounded fitted ratio.
    reference = ones(mode_count)
    plan = AOC.prepare(method,
        OG.OpticalGainUpdateSpecification(reference, fill(0.4, mode_count), fill(2.2, mode_count)))
    dither.response_ratios[1:2] .= [0.125, Inf + 0im]
    exact = AOC.process(plan, inputs)
    @test exact.valid[1:2] == [false, false]
    @test exact.optical_gains[1:2] == [0.125, 0]
    @test exact.correction[1:2] == [1, 1]
end
