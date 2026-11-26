# Welch (1967) periodograms and Gendron and Léna (1994) modal gain optimization. See docs/REFERENCES.md.
"""
    ModalGainOptimization

Select temporal controller gains from complete modal power spectra.
"""
module ModalGainOptimization

using FFTW: FFTW, plan_rfft
using LinearAlgebra: mul!

import ..AdaptiveOpticsCalibration:
    AbstractCalibrationMethod,
    AbstractCalibrationPlan,
    _any_cross_alias,
    _any_pair_alias,
    _checked_extent,
    allocate_result,
    allocate_workspace,
    prepare,
    process!

export PSDGainOptimization, PSDGainOptimizationSpecification, PSDGainOptimizationInputs
export PSDGainOptimizationPlan, PSDGainOptimizationWorkspace, PSDGainOptimizationProduct

"""
    PSDGainOptimization()

Select the minimum predicted residual variance over an explicit gain grid
for each independent mode of a unity-response sampled integrator. For gain g
and integer delay d, the command and true residual are
`c[t] = c[t-1] + g*(r[t] + n[t])` and `r[t] = x[t] - c[t-d]`, with
disturbance x and additive measurement noise n in common modal units. Zero
gain disables correction.

For positive gain the loop transfer is `L = g*z^(-d)/(1 - z^(-1))`. The
disturbance and noise power weights are `abs2(1/(1+L))` and
`abs2(L/(1+L))`, including their limits 0 and 1 at DC. The integration
assumes disturbance and noise are uncorrelated. The calculation uses Float64
and a single-threaded CPU real FFT.
"""
struct PSDGainOptimization <: AbstractCalibrationMethod end

"""
    PSDGainOptimizationSpecification(; samples, modes, sample_rate, delay_samples,
        gains, minimum_gain_margin, segments=1)

Declare at least two samples, at least one mode, a finite positive sampling
rate in Hz, and a positive integer delay in samples. A command computed from
sample t affects the residual first at sample t + d. All modes share this
delay and have unit response in the supplied coordinates.

Divide the complete record into `segments` equal, consecutive,
nonoverlapping segments of at least two samples each.

Supply a one-based strided CPU vector or range of finite nonnegative
dimensionless gains and a finite linear gain margin of at least one.
The stability boundary is `gcrit = 2*sinpi(1/(4*d - 2))`. Preparation
retains only gains strictly below the boundary with `gcrit/g ≥ minimum_gain_margin`.
Disabled correction has infinite gain margin by convention. At least one
gain must qualify. Ties in predicted residual power favor the smallest gain.
"""
struct PSDGainOptimizationSpecification{G<:AbstractVector{<:Real}}
    samples::Int
    modes::Int
    sample_rate::Float64
    delay_samples::Int
    segments::Int
    gains::G
    minimum_gain_margin::Float64

    function PSDGainOptimizationSpecification(;
        samples::Integer,
        modes::Integer,
        sample_rate::Real,
        delay_samples::Integer,
        gains::G,
        minimum_gain_margin::Real,
        segments::Integer=1,
    ) where {G<:AbstractVector{<:Real}}
        return new{G}(Int(samples), Int(modes), Float64(sample_rate),
            Int(delay_samples), Int(segments), gains, Float64(minimum_gain_margin))
    end
end

"""
    PSDGainOptimizationInputs(open_loop, noise)

Bundle two complete, regularly sampled sample × mode records. `open_loop`
is the open-loop WFS estimate of the disturbance plus measurement noise.
`noise` is independently measured WFS noise alone under the same sampling and
noise conditions. Both records use the same ordered modal basis, residual
units, sign, and reference. Each must be a one-based strided CPU matrix of
finite real samples. A closed-loop residual requires model-dependent
conversion outside this operation.
"""
struct PSDGainOptimizationInputs{O<:AbstractMatrix{<:Real},N<:AbstractMatrix{<:Real}}
    open_loop::O
    noise::N
end

"""
Prepared dimensions, frequency bins, eligible gains, gain margins, and
disturbance and measurement-noise power weights. The plan is reentrant with
independent result and workspace storage.
"""
struct PSDGainOptimizationPlan <: AbstractCalibrationPlan
    samples::Int
    modes::Int
    segment_samples::Int
    bin_width::Float64
    frequencies::Vector{Float64}
    gains::Vector{Float64}
    gain_margin::Vector{Float64}
    disturbance_transfer_power::Matrix{Float64}
    noise_transfer_power::Matrix{Float64}
end

"""
One-sided `disturbance_psd` and `noise_psd` in bin × mode order, with
`frequencies` in Hz from DC to the highest nonnegative Fourier bin. Remove the
mean of each complete modal record, then average the rectangular-window
periodograms of its equal segments. With N samples per segment,
bin spacing Δf = sample_rate/N, and unscaled segment DFT coefficient X[k], the
bin density is `w[k]*abs2(X[k]/N)/Δf` in squared modal units per Hz.
The weight w is 1 at DC and at even-length Nyquist and 2 elsewhere.
Summing the averaged density times Δf gives the complete record's population
variance. The DC bin retains variation between segment means. Static offsets
are outside the prediction.

The noise PSD is the averaged noise-only periodogram. The disturbance PSD is
the averaged open-loop periodogram minus the noise PSD, floored at zero per bin.
Finite-window spectral scatter and clipping can bias this estimate.

Each selected dimensionless gain is in `gains` in input mode order, with
`residual_variance` in squared modal units and linear `gain_margin`. The
residual prediction describes the true residual, with measurement noise
entering through correction.
"""
struct PSDGainOptimizationProduct
    frequencies::Vector{Float64}
    disturbance_psd::Matrix{Float64}
    noise_psd::Matrix{Float64}
    gains::Vector{Float64}
    residual_variance::Vector{Float64}
    gain_margin::Vector{Float64}
end

"""
CPU real FFT scratch and staging for repeatable complete-window optimization.
Obtain matching scratch with `allocate_workspace(plan)`.
"""
struct PSDGainOptimizationWorkspace{F}
    samples::Vector{Float64}
    spectrum::Vector{ComplexF64}
    fft_plan::F
    staged::PSDGainOptimizationProduct
end

function prepare(::PSDGainOptimization, specification::PSDGainOptimizationSpecification)
    samples = specification.samples
    modes = specification.modes
    delay = specification.delay_samples
    segments = specification.segments
    sample_rate = specification.sample_rate
    minimum_margin = specification.minimum_gain_margin
    samples >= 2 || throw(ArgumentError("samples must be at least two"))
    modes > 0 || throw(ArgumentError("modes must be positive"))
    delay > 0 || throw(ArgumentError("delay_samples must be positive"))
    isfinite(sample_rate) && sample_rate > 0 ||
        throw(ArgumentError("sample_rate must be finite and positive"))
    isfinite(minimum_margin) && minimum_margin >= 1 ||
        throw(ArgumentError("minimum_gain_margin must be finite and at least one"))
    1 <= segments <= fld(samples, 2) && iszero(rem(samples, segments)) ||
        throw(ArgumentError("segments must divide the record into equal windows of at least two samples"))

    segment_samples = div(samples, segments)
    bin_width = sample_rate / segment_samples
    bin_width > 0 || throw(ArgumentError("frequency bin width underflows Float64"))
    bins = fld(segment_samples, 2) + 1
    grid = specification.gains
    Base.require_one_based_indexing(grid)
    grid isa Union{StridedVector,AbstractRange} ||
        throw(ArgumentError("gains must be a strided CPU vector or range"))
    gains = collect(Float64, grid)
    all(gain -> isfinite(gain) && gain >= 0, gains) ||
        throw(ArgumentError("gains must be finite and nonnegative"))

    # A marginal pole satisfies z^(d-1)*(z-1) = -g. The first crossing
    # occurs at θ = π/(2d-1), giving the positive stable interval (0, gcrit).
    critical_gain = 2 * sinpi(inv(4.0 * delay - 2))
    filter!(gain -> gain < critical_gain &&
        (iszero(gain) || critical_gain / gain >= minimum_margin), gains)
    sort!(unique!(gains))
    isempty(gains) && throw(ArgumentError("gain grid has no eligible gain"))
    gain_margin = [iszero(gain) ? Inf : critical_gain / gain for gain in gains]

    _checked_extent("modal window", samples, modes)
    _checked_extent("modal transfer matrix", bins, length(gains))
    frequencies = [k * bin_width for k in 0:bins-1]
    disturbance_transfer_power = Matrix{Float64}(undef, bins, length(gains))
    noise_transfer_power = similar(disturbance_transfer_power)
    for bin in 1:bins
        k = bin - 1
        amplitude = 2 * sinpi(k / segment_samples)
        # Rotate the characteristic denominator by z^d. The phase of
        # z^d (1 - z^-1) is π/2 + π(2d - 1)k/N. Reduce its integer
        # phase numerator before converting to Float64.
        phase = mod((2 * widen(delay) - 1) * k, 2 * widen(segment_samples)) / segment_samples
        sine, cosine = sincospi(phase)
        for (index, gain) in pairs(gains)
            if iszero(gain)
                disturbance_transfer_power[bin, index] = 1
                noise_transfer_power[bin, index] = 0
            else
                denominator = hypot(muladd(-amplitude, sine, gain), amplitude * cosine)
                disturbance_transfer_power[bin, index] = abs2(amplitude / denominator)
                noise_transfer_power[bin, index] = abs2(gain / denominator)
            end
        end
    end

    return PSDGainOptimizationPlan(samples, modes, segment_samples, bin_width, frequencies,
        gains, gain_margin, disturbance_transfer_power, noise_transfer_power)
end

function allocate_result(plan::PSDGainOptimizationPlan)
    bins = length(plan.frequencies)
    modes = plan.modes
    return PSDGainOptimizationProduct(
        zeros(bins), zeros(bins, modes), zeros(bins, modes),
        zeros(modes), zeros(modes), zeros(modes),
    )
end

function allocate_workspace(plan::PSDGainOptimizationPlan)
    samples = Vector{Float64}(undef, plan.segment_samples)
    spectrum = Vector{ComplexF64}(undef, length(plan.frequencies))
    fft_plan = plan_rfft(samples; flags=FFTW.ESTIMATE | FFTW.UNALIGNED, num_threads=1)
    return PSDGainOptimizationWorkspace(samples, spectrum, fft_plan, allocate_result(plan))
end

_psd_gain_arrays(product::PSDGainOptimizationProduct) = (
    product.disturbance_psd, product.noise_psd, product.gains,
    product.residual_variance, product.gain_margin,
)

function _check_psd_gain_storage(product, workspace, plan, inputs)
    open_loop = inputs.open_loop
    noise = inputs.noise
    Base.require_one_based_indexing(open_loop, noise)
    open_loop isa StridedMatrix && noise isa StridedMatrix ||
        throw(ArgumentError("modal records must be strided CPU matrices"))
    size(open_loop) == size(noise) == (plan.samples, plan.modes) || throw(DimensionMismatch(
        "modal records must have the prepared sample × mode shape",
    ))
    bins = length(plan.frequencies)
    for storage in (product, workspace.staged)
        length(storage.frequencies) == bins &&
            size(storage.disturbance_psd) == size(storage.noise_psd) == (bins, plan.modes) &&
            length(storage.gains) == length(storage.residual_variance) ==
            length(storage.gain_margin) == plan.modes ||
            throw(DimensionMismatch("PSD gain product must match the prepared shape"))
    end
    length(workspace.samples) == plan.segment_samples && length(workspace.spectrum) == bins ||
        throw(DimensionMismatch("PSD gain workspace must match the prepared shape"))
    buffers = (product.frequencies, _psd_gain_arrays(product)...,
        workspace.staged.frequencies, _psd_gain_arrays(workspace.staged)...,
        workspace.samples, workspace.spectrum)
    sources = (open_loop, noise, plan.frequencies, plan.gains, plan.gain_margin,
        plan.disturbance_transfer_power, plan.noise_transfer_power)
    (_any_pair_alias(buffers) || _any_cross_alias(buffers, sources)) &&
        throw(ArgumentError("PSD gain product and workspace must own disjoint storage"))
    return nothing
end

function _periodogram_power!(power, mode, workspace, record)
    samples = workspace.samples
    spectrum = workspace.spectrum
    segment_samples = length(samples)
    sample_count = size(record, 1)
    segments = div(sample_count, segment_samples)
    origin = Float64(record[1, mode])
    mean_offset = 0.0
    for t in axes(record, 1)
        sample = Float64(record[t, mode]) - origin
        isfinite(sample) || throw(ArgumentError("modal records must contain finite Float64 samples"))
        # Keep a small mean offset separate from the possibly large reference.
        mean_offset += (sample - mean_offset) / t
    end
    inverse_count = inv(segment_samples)
    inverse_segments = inv(segments)
    nyquist = iseven(segment_samples) ? length(spectrum) : 0
    for bin in eachindex(spectrum)
        power[bin, mode] = 0.0
    end
    for start in 0:segment_samples:sample_count-1
        for t in eachindex(samples)
            centered = (Float64(record[start + t, mode]) - origin) - mean_offset
            samples[t] = centered * inverse_count
        end
        mul!(spectrum, workspace.fft_plan, samples)
        for bin in eachindex(spectrum)
            weight = bin == 1 || bin == nyquist ? inverse_segments : 2 * inverse_segments
            power[bin, mode] = muladd(weight, abs2(spectrum[bin]), power[bin, mode])
        end
    end
    return nothing
end

function process!(
    product::PSDGainOptimizationProduct,
    workspace::PSDGainOptimizationWorkspace,
    plan::PSDGainOptimizationPlan,
    inputs::PSDGainOptimizationInputs,
)
    _check_psd_gain_storage(product, workspace, plan, inputs)
    staged = workspace.staged
    disturbance = staged.disturbance_psd
    noise = staged.noise_psd
    for mode in 1:plan.modes
        _periodogram_power!(disturbance, mode, workspace, inputs.open_loop)
        _periodogram_power!(noise, mode, workspace, inputs.noise)
        for bin in axes(disturbance, 1)
            open_loop_power = disturbance[bin, mode]
            noise_power = noise[bin, mode]
            isfinite(open_loop_power) && isfinite(noise_power) ||
                throw(ArgumentError("modal periodogram power must be finite"))
            disturbance[bin, mode] = max(open_loop_power - noise_power, 0.0)
        end

        best = 0
        best_power = Inf
        for index in eachindex(plan.gains)
            power = 0.0
            for bin in axes(disturbance, 1)
                power = muladd(disturbance[bin, mode], plan.disturbance_transfer_power[bin, index],
                    power)
                power = muladd(noise[bin, mode], plan.noise_transfer_power[bin, index], power)
            end
            if power < best_power
                best = index
                best_power = power
            end
        end
        best > 0 || throw(ArgumentError("no eligible gain has finite predicted residual power"))
        staged.gains[mode] = plan.gains[best]
        staged.residual_variance[mode] = best_power
        staged.gain_margin[mode] = plan.gain_margin[best]
    end
    disturbance ./= plan.bin_width
    noise ./= plan.bin_width
    all(isfinite, disturbance) && all(isfinite, noise) ||
        throw(ArgumentError("modal PSD must be finite"))
    copyto!(product.frequencies, plan.frequencies)
    foreach(copyto!, _psd_gain_arrays(product), _psd_gain_arrays(staged))
    return product
end

end
