using AdaptiveOpticsCalibration.ModalGainOptimization
using LinearAlgebra: eigvals

function psd_gain_plan(samples;
    modes=1, sample_rate=128, delay_samples=1, gains=[0.0],
    minimum_gain_margin=1, segments=1,
)
    return prepare(PSDGainOptimization(), PSDGainOptimizationSpecification(;
        samples, modes, sample_rate, delay_samples, gains, minimum_gain_margin, segments))
end

@testset "One-sided modal PSD normalization" begin
    for samples in (8, 9), segments in (1, 3)
        # The highest bin is Nyquist only in the even case. A strided view and
        # unequal modes check both record dimensions and mode ordering.
        t = 0:samples*segments-1
        k = fld(samples, 2)
        signal = hcat(2^45 .+ sinpi.(2 .* t ./ samples),
            -17 .+ 2 .* cospi.(2 .* k .* t ./ samples))
        padded = zeros(2size(signal, 1), 2)
        padded[1:2:end, :] .= signal
        record = @view padded[1:2:end, :]
        plan = psd_gain_plan(length(t); modes=2, sample_rate=72, segments)
        result = process(plan, PSDGainOptimizationInputs(record, zeros(size(record))))
        bin_width = 72 / samples

        @test result.frequencies ≈ (0:k) .* bin_width
        for mode in 1:2
            # Direct centered DFT oracle. Center in extended precision so
            # the large constant does not erase a fractional sample mean.
            values = BigFloat.(signal[:, mode])
            centered = Float64.(values .- sum(values) / length(t))
            expected = zeros(k + 1)
            for segment in 0:segments-1, bin in 0:k
                coefficient = sum(centered[segment*samples + j + 1] *
                    cis(-2π * j * bin / samples) for j in 0:samples-1) / samples
                weight = bin == 0 || 2bin == samples ? 1 : 2
                expected[bin + 1] += weight * abs2(coefficient) / (bin_width * segments)
            end
            variance = sum(abs2, centered) / length(t)
            @test result.disturbance_psd[:, mode] ≈ expected rtol=3e-14 atol=1e-15
            @test sum(result.disturbance_psd[:, mode]) * bin_width ≈ variance rtol=3e-14
            @test result.residual_variance[mode] ≈ variance rtol=3e-14
        end
        @test iszero(result.noise_psd)
        @test result.gains == [0, 0]
        @test result.gain_margin == [Inf, Inf]
    end

    # A full-record mean is removed once, so the segment DC bin retains
    # fluctuations in each segment mean. Subtraction follows averaging:
    # open-loop Nyquist bin powers are 4 and 0, noise powers are 0 and 2.25.
    plan = psd_gain_plan(8; segments=2, sample_rate=12)
    open_loop = reshape([15, 11, 15, 11, 7, 7, 7, 7], :, 1)
    noise = reshape([20.0, 20, 20, 20, 21.5, 18.5, 21.5, 18.5], :, 1)
    result = process(plan, PSDGainOptimizationInputs(open_loop, noise))
    @test vec(result.disturbance_psd) ≈ [9, 0, (4 - 2.25) / 2] / 3
    @test vec(result.noise_psd) ≈ [0, 0, 2.25 / 2] / 3
    @test only(result.residual_variance) ≈ 9 + (4 - 2.25) / 2

    # An independent measurement may exceed the open-loop periodogram in a bin.
    result = process(plan, PSDGainOptimizationInputs(fill(1.0, 8, 1), noise))
    @test iszero(result.disturbance_psd)
    @test only(result.residual_variance) == 0
end

@testset "Integrator delay, gain margin, and modal residual variance" begin
    for delay in (1, 2, 3, 7)
        critical = 2sin(π / (4delay - 2))
        stable = 0.999critical
        unstable = 1.001critical
        plan = psd_gain_plan(32; delay_samples=delay,
            gains=[unstable, stable, 0.0])
        @test plan.gains == [0.0, stable]
        @test last(plan.gain_margin) ≈ critical / stable
        for (gain, expected_stable) in ((stable, true), (unstable, false))
            # State [c[t-1], c[t-2], ...] for the homogeneous recurrence.
            transition = zeros(delay, delay)
            transition[1, 1] = 1
            transition[1, delay] -= gain
            for row in 2:delay
                transition[row, row - 1] = 1
            end
            @test (maximum(abs, eigvals(transition)) < 1) == expected_stable
        end
    end
    for (delay, limit) in ((1, 2.0), (2, 1.0))
        plan = psd_gain_plan(32; delay_samples=delay, gains=[limit, prevfloat(limit)])
        @test plan.gains == [prevfloat(limit)]
    end

    # A known Nyquist disturbance and an unrelated noise-only bin. At Nyquist
    # the disturbance denominator is 2-g for d=1 and 2+g for d=2.
    t = 0:7
    disturbance = (-1.0) .^ t
    noise = 2 .* sinpi.(t ./ 2)
    inputs = PSDGainOptimizationInputs(reshape(16 .+ disturbance .+ noise, :, 1),
        reshape(10 .+ 2 .* cospi.(t ./ 2), :, 1))
    for (delay, sign) in ((1, -1), (2, 1))
        gain = 0.5
        plan = psd_gain_plan(8; delay_samples=delay, gains=[gain], minimum_gain_margin=2)
        result = process(plan, inputs)
        expected = 4 / (2 + sign * gain)^2 + 2gain^2 / (1 + (1 - gain)^2)
        @test only(result.residual_variance) ≈ expected rtol=5e-15
        @test only(result.gain_margin) == (delay == 1 ? 4 : 2)
    end
    grid = [0.8, 0.25, 0.5, 1.0, 0.0, 0.25]
    plan = psd_gain_plan(8; delay_samples=2, gains=grid, minimum_gain_margin=2)
    @test plan.gains == [0.0, 0.25, 0.5]
    fill!(grid, 0.0)
    @test plan.gains == [0.0, 0.25, 0.5]
    zero_inputs = PSDGainOptimizationInputs(zeros(8, 1), zeros(8, 1))
    @test only(process(plan, zero_inputs).gains) == 0
end

function replay_modal_variance(disturbance, noise, gain, delay)
    samples = length(disturbance)
    warmup = 16samples
    command = zeros(warmup + samples + delay)
    squared_residual = 0.0
    for t in 1:warmup+samples
        sample = mod1(t, samples)
        # Only the noisy WFS sample is fed back. Score true residual samples
        # over a full held-out period after transients have decayed.
        residual = disturbance[sample] - command[t]
        command[t + delay] = command[t + delay - 1] + gain * (residual + noise[sample])
        if t > warmup
            squared_residual += abs2(residual)
        end
    end
    return squared_residual / samples
end

@testset "Held-out time-domain gain selection" begin
    samples = 128
    t = 0:samples-1
    disturbance = cospi.(4 .* t ./ samples)
    noise = 1.7 .* cospi.(32 .* t ./ samples)
    noise_only = 1.7 .* sinpi.(32 .* t ./ samples)
    inputs = PSDGainOptimizationInputs(reshape(disturbance + noise, :, 1),
        reshape(noise_only, :, 1))
    holdout_disturbance = sinpi.(4 .* t ./ samples .+ 0.32)
    holdout_noise = 1.7 .* cospi.(32 .* t ./ samples .+ 0.63)
    grid = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.2]
    for (delay, expected_gain) in ((1, 0.2), (3, 0.15))
        plan = psd_gain_plan(samples; delay_samples=delay, gains=grid, minimum_gain_margin=1.25)
        result = process(plan, inputs)
        held_out = [replay_modal_variance(holdout_disturbance, holdout_noise, gain, delay)
            for gain in plan.gains]
        @test only(result.gains) == expected_gain == plan.gains[argmin(held_out)]
        @test only(result.residual_variance) ≈ minimum(held_out) rtol=2e-14
        @test minimum(held_out) < 0.6first(held_out)
        @test only(result.gain_margin) > 1.25
    end

    # Signal-free open-loop record with the same noise spectrum and distinct
    # phase. Closing the loop would inject noise without disturbance rejection.
    plan = psd_gain_plan(samples; gains=grid)
    noise_inputs = PSDGainOptimizationInputs(reshape(noise, :, 1), reshape(noise_only, :, 1))
    @test only(process(plan, noise_inputs).gains) == 0
end

@testset "PSD gain workspace reuse and input validation" begin
    @test_throws ArgumentError psd_gain_plan(1)
    @test_throws ArgumentError psd_gain_plan(4; modes=0)
    @test_throws ArgumentError psd_gain_plan(4; delay_samples=0)
    @test_throws ArgumentError psd_gain_plan(4; sample_rate=Inf)
    @test_throws ArgumentError psd_gain_plan(4; minimum_gain_margin=0.9)
    @test_throws ArgumentError psd_gain_plan(4; segments=0)
    @test_throws ArgumentError psd_gain_plan(8; segments=3)
    @test_throws ArgumentError psd_gain_plan(4; segments=4)
    @test_throws ArgumentError psd_gain_plan(4; gains=[-0.1, 0.0])
    @test_throws ArgumentError psd_gain_plan(4; gains=[NaN, 0.0])
    @test_throws ArgumentError psd_gain_plan(4; delay_samples=2, gains=[1.0, 1.1])

    plan = psd_gain_plan(2; gains=[0.0, 0.2], modes=2)
    workspace = allocate_workspace(plan)
    product = allocate_result(plan)
    inputs = PSDGainOptimizationInputs([0.0 2.0; 1.0 4.0], zeros(2, 2))
    @test process!(product, workspace, plan, inputs) === product
    expected = [0.25, 1.0]
    @test product.residual_variance ≈ expected
    bad_inputs = PSDGainOptimizationInputs([2.0 0.0; 1.0 NaN], zeros(2, 2))
    @test_throws ArgumentError process!(product, workspace, plan, bad_inputs)
    @test product.residual_variance ≈ expected
    @test_throws DimensionMismatch process!(product, workspace, plan,
        PSDGainOptimizationInputs(ones(2, 1), ones(2, 1)))
    @test_throws ArgumentError process!(product, workspace, plan,
        PSDGainOptimizationInputs(product.disturbance_psd, zeros(2, 2)))
    @test_throws DimensionMismatch process!(product, allocate_workspace(psd_gain_plan(3)),
        plan, inputs)
    @test process!(product, workspace, plan, inputs) === product
    @test product.residual_variance ≈ expected
    @test process!(product, plan, inputs).residual_variance ≈ expected
end
