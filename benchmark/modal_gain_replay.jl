# julia --startup-file=no --project=. benchmark/modal_gain_replay.jl
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.ModalGainOptimization
using LinearAlgebra: eigvals
using Printf: @printf
using Random: Xoshiro, randn
using Statistics: var

# Independent simulated modal OPD telemetry in µm. Each mode combines a
# stationary AR(1) disturbance, a narrow vibration, and white measurement noise.
function modal_telemetry(seed, samples, sample_rate)
    rng = Xoshiro(seed)
    pole = (0.987, 0.93, 0.0)
    disturbance_rms = (0.12, 0.065, 0.15)
    frequency = (8.3, 36.7, 0.0)
    vibration_peak = (0.05, 0.035, 0.0)
    noise_rms = (0.025, 0.10, 0.10)
    truth = zeros(samples, length(pole))
    noise = similar(truth)
    for m in eachindex(pole)
        turbulent = disturbance_rms[m] * randn(rng)
        phase = 2π * rand(rng)
        innovation = disturbance_rms[m] * sqrt(1 - pole[m]^2)
        for t in 1:samples
            turbulent = pole[m] * turbulent + innovation * randn(rng)
            truth[t, m] = turbulent +
                vibration_peak[m] * sinpi(2 * frequency[m] * t / sample_rate + phase / π)
            noise[t, m] = noise_rms[m] * randn(rng)
        end
    end
    return truth, noise
end

# Advance the causal time-domain controller on held-out data. Its command is
# first visible to the WFS exactly delay samples after the command update.
function replay_variance(truth, noise, delay, gain, burnin)
    commands = zeros(length(truth))
    residual = similar(commands)
    for t in eachindex(truth)
        applied = t > delay ? commands[t - delay] : 0.0
        previous = t > 1 ? commands[t - 1] : 0.0
        residual[t] = truth[t] - applied
        commands[t] = previous + gain * (residual[t] + noise[t])
    end
    return var(view(residual, burnin+1:length(residual)); corrected=false)
end

# Check stability in the time-domain state transition independently of the
# optimizer's frequency-domain gain-margin calculation.
function pole_radius(delay, gain)
    A = zeros(delay, delay)
    A[1, 1] = 1
    A[1, delay] -= gain
    for j in 2:delay
        A[j, j - 1] = 1
    end
    return maximum(abs, eigvals(A))
end

function main()
    samples, segments, modes, fs, delay = 131072, 128, 3, 1000.0, 3
    gains = collect(0.0:0.025:0.6)
    minimum_gain_margin = 1.5
    disturbance, noise = modal_telemetry(7231, samples, fs)
    _, noise_only = modal_telemetry(4601, samples, fs)
    spec = PSDGainOptimizationSpecification(;
        samples, segments, modes, sample_rate=fs, delay_samples=delay,
        gains, minimum_gain_margin)
    product = process(prepare(PSDGainOptimization(), spec),
                      PSDGainOptimizationInputs(disturbance + noise, noise_only))

    burnin = 2048
    truth, measurement_noise = modal_telemetry(9157, samples + burnin, fs)
    println("mode  gain   margin  predicted_µm²  measured_µm²  zero_µm²     measured/best")
    for m in 1:modes
        g = product.gains[m]
        replay(h) = replay_variance(view(truth, :, m), view(measurement_noise, :, m),
                                    delay, h, burnin)
        # Compare with the best eligible grid point on the held-out series
        # only after scoring the frozen selection. Zero gain bypasses control.
        best = minimum(replay(h) for h in gains if iszero(h) ||
                       pole_radius(delay, h * minimum_gain_margin) < 1)
        measured = replay(g)
        margin = product.gain_margin[m]
        if !iszero(g)
            @assert pole_radius(delay, g) < 1
            @assert pole_radius(delay, 0.999 * margin * g) < 1
            @assert pole_radius(delay, 1.001 * margin * g) > 1
        end
        predicted = product.residual_variance[m]
        @printf("%4d  %5.3f  %6.3f  %13.7g  %12.7g  %10.7g  %12.5f\n",
                m, g, margin, predicted, measured, replay(0.0), measured / best)
        @assert abs(predicted / measured - 1) < 0.12
        @assert measured <= 1.02 * best
    end
end

main()
