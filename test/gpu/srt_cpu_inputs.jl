function test_srt_cpu_inputs(to_device)
    @testset "CPU WFS and turbulence fits reject device storage" begin
        sh = AdaptiveOpticsCalibration.ShackHartmann
        atm = AdaptiveOpticsCalibration.AtmosphericParameters
        threshold = prepare(sh.SubapertureThreshold(0.2), sh.SubapertureThresholdSpecification(2))
        weights = prepare(sh.WeightedCentroidMap(), sh.WeightedCentroidMapSpecification(2, 2, 1))
        @test_throws ArgumentError process(threshold,
            sh.SubapertureThresholdInputs(to_device([1.0, 3.0]), 2))
        @test_throws ArgumentError process(weights,
            sh.WeightedCentroidMapInputs(to_device(ones(2, 2, 1)), 1))
        spec(template) = atm.ModalVarianceSpecification([10.0], template;
            reference_r0_m=0.15, reference_wavelength_m=500e-9)
        @test_throws ArgumentError prepare(atm.ModalVarianceFit(), spec(to_device(ones(2, 1))))
        plan = prepare(atm.ModalVarianceFit(), spec(ones(2, 1)))
        @test_throws ArgumentError process(plan,
            atm.ModalVarianceInputs(to_device([3e-14, 2e-14]), zeros(2)))
    end
end
