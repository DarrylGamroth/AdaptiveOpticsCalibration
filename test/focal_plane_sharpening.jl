using AdaptiveOpticsCalibration.FocalPlaneSharpening

# Independently written equation references. The legacy equipment script is
# never included/executed; see docs/DESIGN_SHARPENING.md for path/hash attribution.
function sharpening_metric_reference(image, radius)
    rows, columns = size(image)
    filtered = zeros(rows, columns)
    for column in 1:columns, row in 1:rows
        samples = [Float64(image[clamp(row + dr, 1, rows),
            clamp(column + dc, 1, columns)]) for dr in -1:1 for dc in -1:1]
        filtered[row, column] = sort(samples)[5]
    end
    peak, center = findmax(filtered)
    included = [index for index in CartesianIndices(filtered)
        if sqrt(sum(abs2, Tuple(index - center))) < radius]
    objective = -sum(filtered[index] /
        (1 + sqrt(sum(abs2, Tuple(index - center))))^2 for index in included)
    return (; objective, center, peak, count=length(included), filtered)
end

struct SharpeningZeroVector <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(v::SharpeningZeroVector) = size(v.values)
Base.axes(v::SharpeningZeroVector) = (0:(length(v.values) - 1),)
Base.getindex(v::SharpeningZeroVector, i::Int) = v.values[i + 1]

function modal_scan_allocations(result, plan, metrics)
    process!(result, nothing, plan, metrics)
    return @allocated process!(result, nothing, plan, metrics)
end

@testset "modal amplitude scan" begin
    malformed = ModalScanPlan([-1., 0., 1.], [-1., 0., 1.], zeros(0, 0),
        0., 1., -1., 1., 1.)
    @test_throws DimensionMismatch process(malformed, [1., 3., 1.])
    amplitudes = [2., -2., 0., 1., 0., -1.]
    metrics = 7 .- 2 .* (amplitudes .- 0.25) .^ 2
    plan = prepare(ModalScan(), ModalScanSpecification(view(amplitudes, :)))
    fill!(amplitudes, NaN)
    result = process(plan, view(metrics, :))
    @test result.status == FocalPlaneSharpening.Valid
    @test result.best_sample_index == 3
    @test result.best_sample_amplitude == 0
    @test result.best_sample_metric == 6.875
    @test result.peak_amplitude ≈ 0.25 atol=5e-15
    @test result.residual_rms ≈ 0 atol=1e-13
    @test allocate_workspace(plan) === nothing
    @test modal_scan_allocations(result, plan, metrics) == 0
    @test modal_scan_allocations(result, plan, view(metrics, :)) == 0

    # Opposite perturbations at the two repetitions leave the fit invariant.
    noisy = copy(metrics)
    noisy[3] += 0.5
    noisy[5] -= 0.5
    @test process!(result, plan, noisy) === result
    @test result.peak_amplitude ≈ 0.25 atol=5e-15
    @test result.residual_rms ≈ sqrt(2 * 0.5^2 / length(metrics)) atol=1e-13
    @test result.best_sample_metric == 7.375
    shifted = process(plan, noisy .+ 2.0^40)
    @test shifted.status == result.status
    @test shifted.peak_amplitude == result.peak_amplitude
    @test shifted.residual_rms == result.residual_rms
    @test shifted.best_sample_metric - 2.0^40 == result.best_sample_metric

    # A tie retains the first sample, even when amplitudes are out of order.
    short = prepare(ModalScan(), ModalScanSpecification([1, -1, 0]))
    tied = process(short, [1, -1, 1])
    @test tied.status == FocalPlaneSharpening.Valid
    @test tied.peak_amplitude ≈ 0.5 atol=5e-15
    @test tied.best_sample_index == 1
    offset = 2.0^45
    translated = prepare(ModalScan(), ModalScanSpecification(offset .+ [1, -1, 0]))
    @test process(translated, [1, -1, 1]).peak_amplitude == offset + 0.5
    tiny = nextfloat(0.0)
    tiny_plan = prepare(ModalScan(), ModalScanSpecification([0, tiny, 2tiny]))
    @test process(tiny_plan, [1, 3, 1]).peak_amplitude == tiny

    for (values, expected) in (([1, 1, 1], UnresolvedCurvature),
            ([1, -1, 0], UnresolvedCurvature),
            ([1., -1., 8eps(Float64)], UnresolvedCurvature),
            ([nextfloat(1.), 1., 1.], UnresolvedCurvature),
            ([1, 1, 0], NonconcaveFit), ([3, -5, 0], UnbracketedPeak),
            ([0, -4, -1], UnbracketedPeak))
        process!(tied, short, values)
        @test tied.status == expected
        @test tied.peak_amplitude === nothing
        @test tied.residual_rms ≈ 0 atol=1e-13
    end
    extreme = floatmax(Float64)
    @test process!(tied, short, [extreme, -extreme, 0.]).status == NumericalFailure
    @test tied.best_sample_metric == extreme
    @test tied.peak_amplitude === tied.residual_rms === nothing
    @test process!(tied, short, [0, 0, 1]).status == FocalPlaneSharpening.Valid

    for invalid in ([1., 2.], [1., 1., 1.], [1., 2., 1.], [0., eps()/8, 1.],
                    [0., 1., Inf], [0., 1., NaN], SharpeningZeroVector([0., 1., 2.]))
        @test_throws ArgumentError prepare(ModalScan(), ModalScanSpecification(invalid))
    end
    @test_throws DimensionMismatch process!(tied, short, [1, 2])
    @test_throws ArgumentError process!(tied, short, [1, 2, NaN])
    @test_throws ArgumentError process!(tied, short, [1, 2, Inf])
    @test_throws ArgumentError process!(tied, short, SharpeningZeroVector([1., 2., 3.]))
    @test tied.status == FocalPlaneSharpening.Valid
    @test tied.best_sample_amplitude == 0
end

@testset "weighted core intensity" begin
    @test_throws ArgumentError WeightedCoreIntensity(0)
    @test_throws ArgumentError WeightedCoreIntensity(Inf)
    @test_throws ArgumentError WeightedCoreIntensity(NaN)
    @test_throws ArgumentError ImageSpecification(0, 2)
    @test_throws ArgumentError ImageSpecification(typemax(Int), 2)
    plan = prepare(WeightedCoreIntensity(1), ImageSpecification(2, 2))
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    image = [1 5; 3 7]
    process!(result, workspace, plan, image)
    @test workspace.filtered == [3 5; 3 5]
    @test result.status == FocalPlaneSharpening.Valid
    @test result.center == CartesianIndex(1, 2)
    @test result.peak_value == 5
    @test result.objective == -5
    @test result.aperture_count == 1 # strict radius, adjacent pixels excluded
    @test !result.cropped
    @test image == [1 5; 3 7]
    @test_throws DimensionMismatch process!(result, workspace, plan, ones(3, 2))
    @test_throws ArgumentError process!(result, workspace, plan, [1.0 NaN; 2 3])
    @test_throws ArgumentError process!(result, workspace, plan, workspace.filtered)
    @test_throws DimensionMismatch process!(result, WeightedCoreWorkspace(zeros(1, 2)), plan, image)
    @test result.objective == -5 # structural errors precede writes

    for image in (reshape([3.0], 1, 1), reshape([1.0, 5, 2, 4], 1, 4),
                  reshape([1.0, 5, 2, 4], 4, 1), zeros(3, 4),
                  fill(-3.0, 3, 4), reshape(Float32.(1:20), 4, 5),
                  [3.0 -2 7; 1 4 8; 9 0 6])
        for radius in (0.5, 1.0, 2.1, 7.0)
            p = prepare(WeightedCoreIntensity(radius), ImageSpecification(size(image)...))
            answer = process(p, image)
            reference = sharpening_metric_reference(image, radius)
            @test answer.status == FocalPlaneSharpening.Valid
            @test answer.objective ≈ reference.objective rtol=4eps(Float64) atol=0
            @test answer.center == reference.center
            @test answer.peak_value == reference.peak
            @test answer.aperture_count == reference.count
            @test answer.cropped == (min(reference.center[1], reference.center[2],
                size(image, 1) - reference.center[1] + 1,
                size(image, 2) - reference.center[2] + 1) < radius)
        end
    end
    zero = process(prepare(WeightedCoreIntensity(7), ImageSpecification(3, 3)), zeros(3, 3))
    @test isequal(zero.objective, -0.0)
    @test zero.center == CartesianIndex(1, 1)
    hot = zeros(3, 3); hot[2, 2] = 1e9
    @test process(prepare(WeightedCoreIntensity(7), ImageSpecification(3, 3)), hot).objective == 0
    strict = process(prepare(WeightedCoreIntensity(7), ImageSpecification(1, 9)), ones(1, 9))
    @test strict.aperture_count == 7
    tie = process(prepare(WeightedCoreIntensity(0.5), ImageSpecification(2, 2)), [0 7; 7 0])
    @test tie.center == CartesianIndex(2, 1) # column-major tie, not row-major
    p = prepare(WeightedCoreIntensity(2.1), ImageSpecification(3, 4))
    raw = reshape(Float64.(1:12), 3, 4)
    @test process(p, 13 .* raw).objective ≈ 13 * process(p, raw).objective
    @test process(p, view(raw, :, :)).objective == process(p, raw).objective
    interior = zeros(15, 15); interior[6:10, 6:10] .= 1
    contained = process(prepare(WeightedCoreIntensity(3), ImageSpecification(15, 15)), interior)
    @test !contained.cropped && contained.aperture_count == 25
    # The median and recentering are nonlinear: image averaging is caller-owned.
    a = [0.0 9 0]
    b = [9.0 0 0]
    nonlinear = prepare(WeightedCoreIntensity(7), ImageSpecification(1, 3))
    @test process(nonlinear, (a + b) / 2).objective !=
        (process(nonlinear, a).objective + process(nonlinear, b).objective) / 2
    overflow_plan = prepare(WeightedCoreIntensity(7), ImageSpecification(2, 2))
    answer = process(overflow_plan, fill(floatmax(Float64), 2, 2))
    @test answer.status == NumericalFailure
    @test answer.objective == 0 && answer.aperture_count == 0
    process!(answer, overflow_plan, ones(2, 2))
    @test answer.status == FocalPlaneSharpening.Valid
    fill!(workspace.filtered, NaN)
    process!(result, workspace, plan, image)
    @test workspace.filtered == [3 5; 3 5]
end

@testset "supplied SPSA probes and gradient" begin
    @test_throws ArgumentError CoefficientSpecification(0)
    spec = CoefficientSpecification(2)
    probes = prepare(SPSAProbes(), spec)
    gradients = prepare(SPSAGradient(), spec)
    q, direction = [2.0, -3], [1, -1]
    inputs = SPSAProbeInputs(q, 0.25, direction)
    result = process(probes, inputs)
    @test result.status == FocalPlaneSharpening.Valid
    @test result.plus == [2.25, -3.25]
    @test result.minus == [1.75, -2.75]
    @test q == [2, -3] && direction == [1, -1]
    @test process(probes, SPSAProbeInputs(view(q, :), .25, view(direction, :))).plus == result.plus
    quadratic(x) = sum(x .^ 2)
    metrics = SPSAGradientInputs(direction, .25, quadratic(result.plus), quadratic(result.minus))
    gradient = process(gradients, metrics)
    @test gradient.status == FocalPlaneSharpening.Valid
    @test gradient.gradient == [10, -10] # dot(2q, Δ)*Δ, not [4,-6]
    reversed = process(gradients, SPSAGradientInputs(direction, .25,
        metrics.minus_metric, metrics.plus_metric))
    @test reversed.gradient == -gradient.gradient
    flipped = process(gradients, SPSAGradientInputs(-direction, .25,
        metrics.minus_metric, metrics.plus_metric))
    @test flipped.gradient == gradient.gradient
    @test iszero(process(gradients, SPSAGradientInputs(direction, .25, 3, 3)).gradient)
    one = CoefficientSpecification(1)
    pair = process(prepare(SPSAProbes(), one), SPSAProbeInputs([3.0], .125, [-1]))
    @test process(prepare(SPSAGradient(), one), SPSAGradientInputs([-1], .125,
        quadratic(pair.plus), quadratic(pair.minus))).gradient == [6]
    for bad in ([true, false], [1, 0], [1, 2], [1, NaN])
        @test_throws ArgumentError process(probes, SPSAProbeInputs(q, .25, bad))
        @test_throws ArgumentError process(gradients, SPSAGradientInputs(bad, .25, 1, 2))
    end
    for c in (0, -1, Inf, NaN)
        @test_throws ArgumentError process(probes, SPSAProbeInputs(q, c, direction))
        @test_throws ArgumentError process(gradients, SPSAGradientInputs(direction, c, 1, 2))
    end
    @test_throws ArgumentError process(probes, SPSAProbeInputs([Inf, 1.0], .1, direction))
    @test_throws ArgumentError process(gradients, SPSAGradientInputs(direction, .1, NaN, 2))
    @test_throws DimensionMismatch process(probes, SPSAProbeInputs([1.0], .1, direction))
    @test_throws DimensionMismatch process(probes, SPSAProbeInputs(q, .1, [1]))
    @test_throws ArgumentError process(probes, SPSAProbeInputs(SharpeningZeroVector(q), .1, direction))
    @test_throws ArgumentError process!(SPSAProbeProduct(NumericalFailure, q, zeros(2)),
        nothing, probes, inputs)
    shared = zeros(2)
    @test_throws ArgumentError process!(SPSAProbeProduct(NumericalFailure, shared, shared),
        nothing, probes, inputs)
    d64 = Float64.(direction)
    @test_throws ArgumentError process!(SPSAGradientProduct(NumericalFailure, d64),
        nothing, gradients, SPSAGradientInputs(d64,.25,1,2))
    @test q == [2, -3]
    oneprobe = prepare(SPSAProbes(), one)
    for (coefficients, c) in (([1.0], eps(Float64)/16),
            (Float32[1], Float64(eps(Float32))/16),
            (BigFloat[BigFloat(1) + eps(BigFloat)], eps(Float64)/16))
        collapsed = process(oneprobe, SPSAProbeInputs(coefficients, c, [1]))
        @test collapsed.status == LostPerturbation
        @test iszero(collapsed.plus) && iszero(collapsed.minus)
    end
    @test process(oneprobe, SPSAProbeInputs(Float32[1], eps(Float32), [1])).status == FocalPlaneSharpening.Valid
    @test process(oneprobe, SPSAProbeInputs(BigFloat[1], .25, [1])).plus == [1.25]
    @test process(oneprobe, SPSAProbeInputs([floatmax(Float64)], floatmax(Float64), [1])).status == NumericalFailure
    @test process(oneprobe, SPSAProbeInputs(Float32[floatmax(Float32)], Float64(floatmax(Float32)), [1])).status == NumericalFailure
    onegradient = prepare(SPSAGradient(), one)
    @test process(onegradient, SPSAGradientInputs([1], 1., floatmax(Float64), -floatmax(Float64))).status == NumericalFailure
    tiny = nextfloat(0.0)
    @test process(onegradient, SPSAGradientInputs([1], tiny, tiny, 0.)).gradient == [.5]
    @test process(onegradient, SPSAGradientInputs([1], floatmax(Float64), floatmax(Float64), 0.)).gradient == [.5]
    @test process(onegradient, SPSAGradientInputs([1], tiny, 1., 0.)).status == NumericalFailure
    fill!(result.plus, NaN); fill!(result.minus, Inf)
    @test process!(result, nothing, probes, inputs).plus == [2.25, -3.25]
end

function sharpening_adam_reference(q, g, m, v, t, alpha, beta1, beta2, epsilon)
    # Independent BigFloat bias correction supplies a near-one reference too.
    mm = beta1 .* m .+ (1 - beta1) .* g
    vv = beta2 .* v .+ (1 - beta2) .* (g .^ 2)
    mhat = mm ./ Float64(1 - BigFloat(beta1)^(t + 1))
    vhat = vv ./ Float64(1 - BigFloat(beta2)^(t + 1))
    return (; candidate=q .- alpha .* mhat ./ (sqrt.(vhat) .+ epsilon), m=mm, v=vv)
end

@testset "explicit-state Adam" begin
    for args in ((0,.9,.99,1e-8), (.1,-.1,.99,1e-8), (.1,.9,1,1e-8),
                 (.1,.9,.99,0), (Inf,.9,.99,1e-8), (.1,NaN,.99,1e-8))
        @test_throws ArgumentError AdamStep(args...)
    end
    @test_throws ArgumentError AdamState([0.], [0.], -1; beta1=.9, beta2=.99)
    method = AdamStep(.1, .9, .99, .01)
    plan = prepare(method, CoefficientSpecification(2))
    q, g, m, v = [1., 2.], [2., -4.], zeros(2), zeros(2)
    state = AdamState(m, v, 0; beta1=.9, beta2=.99)
    inputs = AdamInputs(q, g, state)
    result = process(plan, inputs)
    @test result.status == FocalPlaneSharpening.Valid
    @test result.candidate ≈ q - .1 .* g ./ (abs.(g) .+ .01)
    @test result.next_state.first_moment ≈ .1 .* g
    @test result.next_state.second_moment ≈ .01 .* g.^2
    @test result.next_state.iteration == 1
    @test q == [1,2] && g == [2,-4] && iszero(m) && iszero(v)
    @test state.iteration == 0
    q2 = copy(result.candidate)
    m2 = copy(result.next_state.first_moment)
    v2 = copy(result.next_state.second_moment)
    second = process(plan, AdamInputs(q2, [0.,3.], result.next_state))
    reference = sharpening_adam_reference(q2, [0.,3.], m2, v2, 1, .1,.9,.99,.01)
    @test second.candidate ≈ reference.candidate rtol=4eps(Float64)
    @test second.next_state.first_moment ≈ reference.m
    @test second.next_state.second_moment ≈ reference.v
    @test second.next_state.iteration == 2
    @test second.candidate[1] != q2[1] # prior momentum persists for zero gradient
    @test result.next_state.first_moment == m2 && result.next_state.second_moment == v2
    for beta in (0., prevfloat(1.))
        p = prepare(AdamStep(.1, beta, beta, .01), CoefficientSpecification(2))
        s = AdamState(zeros(2), zeros(2), 0; beta1=beta, beta2=beta)
        answer = process(p, AdamInputs(q, g, s))
        r = sharpening_adam_reference(q, g, zeros(2), zeros(2), 0, .1,beta,beta,.01)
        @test answer.status == FocalPlaneSharpening.Valid
        @test answer.candidate ≈ r.candidate rtol=4eps(Float64)
    end
    for badstate in (AdamState([1.,0], zeros(2), 0; beta1=.9,beta2=.99),
            AdamState(zeros(2), [-1.,0], 1; beta1=.9,beta2=.99),
            AdamState([NaN,0], zeros(2), 1; beta1=.9,beta2=.99),
            AdamState(zeros(2), zeros(2), 1; beta1=.8,beta2=.99),
            AdamState(zeros(2), zeros(2), typemax(UInt64); beta1=.9,beta2=.99))
        @test_throws ArgumentError process(plan, AdamInputs(q, g, badstate))
    end
    @test_throws ArgumentError process(plan, AdamInputs([Inf,1.], g, state))
    @test_throws DimensionMismatch process(plan, AdamInputs(q, [1.], state))
    @test_throws ArgumentError process(plan, AdamInputs(SharpeningZeroVector(q), g, state))
    owned = allocate_result(plan)
    scratch = allocate_workspace(plan)
    @test_throws ArgumentError process!(AdamProduct(NumericalFailure, q, owned.next_state), scratch, plan, inputs)
    @test_throws ArgumentError process!(owned, AdamWorkspace(owned.candidate, zeros(2), zeros(2)), plan, inputs)
    alias_state = AdamState(m, zeros(2), 0; beta1=.9,beta2=.99)
    @test_throws ArgumentError process!(AdamProduct(NumericalFailure, zeros(2), alias_state), scratch, plan, inputs)
    @test q == [1,2] && iszero(m)
    # Check every scratch/output slot against every borrowed input, and every
    # pair of writable slots. All failures must precede any input writes.
    for output_index in 1:6, input in (q, g, m, v)
        arrays = [zeros(2) for _ in 1:6]
        arrays[output_index] = input
        outstate = AdamState(arrays[2], arrays[3], 0; beta1=.9,beta2=.99)
        out = AdamProduct(NumericalFailure, arrays[1], outstate)
        work = AdamWorkspace(arrays[4],arrays[5],arrays[6])
        @test_throws ArgumentError process!(out, work, plan, inputs)
    end
    for first in 1:6, last in (first + 1):6
        arrays = [zeros(2) for _ in 1:6]
        arrays[last] = arrays[first]
        outstate = AdamState(arrays[2], arrays[3], 0; beta1=.9,beta2=.99)
        out = AdamProduct(NumericalFailure, arrays[1], outstate)
        work = AdamWorkspace(arrays[4],arrays[5],arrays[6])
        @test_throws ArgumentError process!(out, work, plan, inputs)
    end
    @test q == [1,2] && g == [2,-4] && iszero(m) && iszero(v)
    counter_edge = AdamState(zeros(2), zeros(2), typemax(UInt64)-1; beta1=.9,beta2=.99)
    @test process(plan, AdamInputs(q, g, counter_edge)).next_state.iteration == typemax(UInt64)
    failed = process(plan, AdamInputs(q, [floatmax(Float64),1.], state))
    @test failed.status == NumericalFailure
    @test iszero(failed.candidate) && iszero(failed.next_state.first_moment)
    @test iszero(failed.next_state.second_moment) && failed.next_state.iteration == 0
    @test process!(failed, plan, inputs).candidate == result.candidate
    fill!(scratch.candidate, NaN); fill!(scratch.first_moment, Inf)
    fill!(scratch.second_moment, -1.)
    repeated = process!(owned, scratch, plan, inputs)
    @test repeated.candidate == result.candidate
    @test repeated.next_state.first_moment == result.next_state.first_moment
    @test repeated.next_state.second_moment == result.next_state.second_moment
    @test process(plan, AdamInputs(view(q,:), view(g,:),
        AdamState(view(m,:),view(v,:),0;beta1=.9,beta2=.99))).candidate == result.candidate
end
