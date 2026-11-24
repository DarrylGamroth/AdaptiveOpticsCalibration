using AdaptiveOpticsCalibration.ShackHartmann

@testset "Active subaperture peak-median threshold" begin
    spec = SubapertureThresholdSpecification(6)
    plan = prepare(SubapertureThreshold(0.25), spec)
    peaks = [32.0, 4.0, 20.0, 12.0, NaN, Inf]
    inputs = SubapertureThresholdInputs(peaks, 4)
    result = process(plan, inputs)
    @test result.valid
    @test result.active_count == 4
    @test result.median_peak == 16.0
    @test result.threshold == 4.0
    @test isequal(peaks, [32.0, 4.0, 20.0, 12.0, NaN, Inf])

    workspace = allocate_workspace(plan)
    strided_peaks = view(repeat(peaks; inner=2), 1:2:12)
    @test process!(result, workspace, plan, SubapertureThresholdInputs(strided_peaks, 3)) === result
    @test result.active_count == 3
    @test result.median_peak == 20.0
    @test result.threshold == 5.0

    @test process!(result, plan, SubapertureThresholdInputs(peaks, 5)) === result
    @test !result.valid
    @test isnan(result.median_peak) && isnan(result.threshold)
    @test result.active_count == 5

    @test process!(result, workspace, plan, SubapertureThresholdInputs(peaks, 0)) === result
    @test !result.valid
    @test result.active_count == 0
    @test isnan(result.median_peak) && isnan(result.threshold)

    @test !process(plan, SubapertureThresholdInputs([big(10)^400, 0, 0, 0, 0, 0], 1)).valid
    for value in (floatmax(Float64), nextfloat(0.0), -8.0)
        equal_peaks = SubapertureThresholdInputs(fill(value, 6), 2)
        @test process(plan, equal_peaks).median_peak == value
    end
    overflow_plan = prepare(SubapertureThreshold(2), spec)
    overflow = process(overflow_plan, SubapertureThresholdInputs(fill(floatmax(Float64), 6), 2))
    @test overflow.median_peak == floatmax(Float64)
    @test !overflow.valid && isnan(overflow.threshold)

    @test_throws ArgumentError SubapertureThreshold(-0.5)
    @test_throws ArgumentError SubapertureThreshold(Inf)
    @test_throws ArgumentError SubapertureThreshold(NaN)
    @test_throws ArgumentError SubapertureThresholdSpecification(0)
    @test_throws ArgumentError process(plan, SubapertureThresholdInputs(peaks, -1))
    @test_throws ArgumentError process(plan, SubapertureThresholdInputs(peaks, 7))
    @test_throws DimensionMismatch process(plan, SubapertureThresholdInputs(peaks[1:4], 3))
    @test_throws ArgumentError process!(result, workspace, plan,
        SubapertureThresholdInputs(workspace.peak_pixels, 4))
    shifted_axes = view(zeros(7), Base.IdentityUnitRange(2:7))
    @test_throws ArgumentError process(plan, SubapertureThresholdInputs(shifted_axes, 4))
end

@testset "Weighted centroid map in supplied subaperture order" begin
    spec = WeightedCentroidMapSpecification(2, 3, 5)
    plan = prepare(WeightedCentroidMap(), spec)
    pixels = zeros(2, 3, 5)
    pixels[:, :, 1] = [4 -2 0; 0 1 2]
    pixels[:, :, 2] .= -8
    pixels[:, :, 3] .= 16
    pixels[2, 3, 3] = NaN
    pixels[:, :, 4] = [2 4 0; 8 0 2]
    pixels[:, :, 5] .= Inf
    inputs = WeightedCentroidMapInputs(pixels, 4)
    result = process(plan, inputs)
    @test result.active_count == 4
    @test result.finite == [true, true, false, true, false]
    @test result.positive_peak == [true, false, false, true, false]
    @test isequal(result.peak, [4.0, 0.0, NaN, 8.0, 0.0])
    @test result.weights[:, :, 1] == [1 0 0; 0 0.25 0.5]
    @test all(==(1.0), view(result.weights, :, :, 2))
    @test all(iszero, view(result.weights, :, :, 3))
    @test result.weights[:, :, 4] == [0.25 0.5 0; 1 0 0.25]
    @test all(iszero, view(result.weights, :, :, 5))

    @test isnothing(allocate_workspace(plan))
    @test process!(result, plan, WeightedCentroidMapInputs(pixels, 0)) === result
    @test all(iszero, result.weights)
    @test all(iszero, result.peak)
    @test all(!, result.finite) && all(!, result.positive_peak)
    @test result.active_count == 0

    @test_throws ArgumentError WeightedCentroidMap(; gaussian_fwhm=-1)
    @test_throws ArgumentError WeightedCentroidMap(; gaussian_fwhm=NaN)
    @test_throws ArgumentError WeightedCentroidMap(; gaussian_fwhm=Inf)
    @test_throws ArgumentError WeightedCentroidMapSpecification(2, 0, 3)
    @test_throws ArgumentError WeightedCentroidMapSpecification(2, 3, 0)
    @test_throws DimensionMismatch process(plan, WeightedCentroidMapInputs(zeros(3, 2, 5), 4))
    @test_throws ArgumentError process(plan, WeightedCentroidMapInputs(pixels, 6))
    @test_throws ArgumentError process!(result, plan, WeightedCentroidMapInputs(result.weights, 4))
    shifted_axes = view(zeros(3, 3, 5), Base.IdentityUnitRange(2:3), :, :)
    @test_throws ArgumentError process(plan, WeightedCentroidMapInputs(shifted_axes, 4))
end

@testset "Gaussian smoothing precedes clipping and preserves zero lag" begin
    spec = WeightedCentroidMapSpecification(2, 3, 2)
    plan = prepare(WeightedCentroidMap(; gaussian_fwhm=2.0), spec)
    pixels = zeros(2, 3, 2)
    pixels[:, :, 1] = [4 -2 0; 0 1 2]
    pixels[2, 2, 2] = 1
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    strided_pixels = view(repeat(pixels; inner=(2, 1, 1)), 1:2:4, :, :)
    inputs = WeightedCentroidMapInputs(strided_pixels, 2)
    @test process!(result, workspace, plan, inputs) === result
    # At FWHM 2 the adjacent coefficient is 1/2 in both directions.
    @test result.peak ≈ [3.25, 1.0] rtol=1e-14
    @test result.weights[:, :, 1] ≈ [3.25 1 0.25; 2 2 2] / 3.25 rtol=1e-14
    @test result.weights[:, :, 2] ≈ [0.25 0.5 0.25; 0.5 1 0.5] rtol=1e-14
    @test all(result.finite) && all(result.positive_peak)
    @test pixels[1, 2, 1] == -2
    @test_throws ArgumentError process!(result, nothing, plan, inputs)
    @test_throws DimensionMismatch process!(result, WeightedCentroidMapWorkspace(zeros(3, 2)),
        plan, inputs)

    pixels[:, :, 1] .= floatmax(Float64)
    pixels[:, :, 2] .= -1
    process!(result, workspace, plan, WeightedCentroidMapInputs(pixels, 2))
    @test isequal(result.peak, [NaN, 0.0])
    @test result.finite == [false, true]
    @test result.positive_peak == [false, false]
    @test all(iszero, view(result.weights, :, :, 1))
    @test all(==(1.0), view(result.weights, :, :, 2))
end

@testset "Weighted-centroid response to independently shifted spots" begin
    x, y = -15:15, -12:12
    # Independent test spots have equal widths and a known 4-column, -2-row shift.
    # The product of two equal-width Gaussians is centered halfway between them.
    spot(dx, dy) = [exp(-0.5 * (((column - dx) / 2.3)^2 + ((row - dy) / 1.2)^2))
        for row in y, column in x]
    training_spot = spot(0, 0)
    held_out_spot = spot(4, -2)
    spec = WeightedCentroidMapSpecification(length(y), length(x), 1)
    plan = prepare(WeightedCentroidMap(), spec)
    result = process(plan, WeightedCentroidMapInputs(reshape(training_spot, length(y), length(x), 1), 1))
    weighted = result.weights[:, :, 1] .* held_out_spot
    centroid(pixels) = (sum(pixels .* transpose(x)) / sum(pixels), sum(pixels .* y) / sum(pixels))
    @test collect(centroid(held_out_spot)) ≈ [4.0, -2.0] atol=5e-5 rtol=0
    @test collect(centroid(weighted)) ≈ [2.0, -1.0] atol=1e-10 rtol=0
end
