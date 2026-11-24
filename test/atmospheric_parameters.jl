using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.AtmosphericParameters
using Test

@testset "Atmospheric parameters from modal variances" begin
    wavelength = 500e-9
    to_opd_variance = (wavelength / (2π))^2
    reference_r0 = 0.15
    candidates = [8.0, 24.0, 64.0, Inf]
    templates = [75.0 180.0 330.0 460.0;
                 18.0 25.0 30.0 32.0;
                  7.0  9.0 10.0 10.5]
    weights = [2.0, 4.0, 1.0]
    specification = ModalVarianceSpecification(candidates, templates;
        reference_r0_m=reference_r0, reference_wavelength_m=wavelength, weights)
    plan = prepare(ModalVarianceFit(), specification)
    truth = 0.164
    turbulence = templates[:, 2] .* (to_opd_variance * (reference_r0 / truth)^(5 / 3))
    noise = [9.0, 2.0, 1.0] .* to_opd_variance
    observed = turbulence + noise
    inputs = ModalVarianceInputs(view(observed, :), noise)

    # Preparation retained physical copies and processing respects subarray inputs.
    candidates[2] = 2.0
    templates[:, 2] .= 0
    weights[2] = 0
    result = process(plan, inputs)
    @test result.r0_m ≈ truth rtol=5e-15
    @test result.L0_m == 24.0
    @test result.L0_support_m == (24.0, 24.0)
    @test !result.L0_at_grid_boundary
    @test result.fitted_variance_m2 ≈ turbulence rtol=5e-15
    @test result.relative_rms ≤ 5e-15
    @test all(>(0), result.relative_rms_by_L0[[1, 3, 4]])
    @test result.r0_by_L0_m[2] == result.r0_m

    workspace = AdaptiveOpticsCalibration.allocate_workspace(plan)
    @test process!(result, workspace, plan, inputs) === result
    @test process!(result, plan, inputs) === result
    @test result.r0_m ≈ truth rtol=5e-15

    mode_weights = [2.0, 4.0, 1.0]
    # Direct weighted rectangular least squares supplies an independent profile.
    y = turbulence .* [1.1, 0.9, 1.2]
    biased = process(plan, ModalVarianceInputs(y + noise, noise))
    for column in eachindex(plan.L0_candidates_m)
        q = sqrt.(mode_weights) .* plan.reference_variance_m2[:, column] ./ y
        b = sqrt.(mode_weights)
        amplitude = q \ b
        @test biased.r0_by_L0_m[column] ≈ reference_r0 * amplitude^(-3 / 5)
        @test biased.relative_rms_by_L0[column] ≈ sqrt(sum(abs2, b - q * amplitude) / sum(mode_weights))
    end

    # Identical shape up to amplitude cannot identify the outer scale.
    proportional = ModalVarianceSpecification([12.0, 48.0, Inf], [2 4 8; 1 2 4];
        reference_r0_m=0.15, reference_wavelength_m=2π)
    unresolved = process(prepare(ModalVarianceFit(), proportional),
        ModalVarianceInputs([4.0, 2.0], [0.0, 0.0]))
    @test unresolved.L0_support_m == (12.0, Inf)
    @test unresolved.L0_at_grid_boundary
    @test unresolved.relative_rms_by_L0 == zeros(3)
    @test allunique(unresolved.r0_by_L0_m)

    fixed = ModalVarianceSpecification([Inf], [5.0;;];
        reference_r0_m=0.7, reference_wavelength_m=2π)
    known_L0 = process(prepare(ModalVarianceFit(), fixed), ModalVarianceInputs([6], [1]))
    @test known_L0.r0_m ≈ 0.7
    @test known_L0.L0_m == Inf
    @test known_L0.L0_support_m == (Inf, Inf)
    @test !known_L0.L0_at_grid_boundary

    @test_throws ArgumentError ModalVarianceFit(relative_rms_tolerance=NaN)
    @test_throws ArgumentError prepare(ModalVarianceFit(), specification)
    @test_throws DimensionMismatch prepare(ModalVarianceFit(), ModalVarianceSpecification([1, 2], [3;;];
        reference_r0_m=0.7, reference_wavelength_m=2π))
    @test_throws ArgumentError prepare(ModalVarianceFit(), ModalVarianceSpecification([1, 2], [3 4];
        reference_r0_m=0.7, reference_wavelength_m=2π))
    @test_throws ArgumentError prepare(ModalVarianceFit(), ModalVarianceSpecification([Inf], [3;;];
        reference_r0_m=1.0, reference_wavelength_m=0))
    for (variance, noise_variance) in (([5, 2, 1], [-1, 0, 0]),
                                      ([5, 2, 1], [5, 0, 0]),
                                      ([5, Inf, 1], [0, 0, 0]))
        @test_throws ArgumentError process!(result, workspace, plan,
            ModalVarianceInputs(variance, noise_variance))
    end
    @test_throws DimensionMismatch process(plan, ModalVarianceInputs(observed, [0, 0]))
    @test_throws ArgumentError process!(result, workspace, plan,
        ModalVarianceInputs(result.fitted_variance_m2, noise))
    @test result.r0_m ≈ truth rtol=5e-15
    @test result.fitted_variance_m2 ≈ turbulence rtol=5e-15
end
