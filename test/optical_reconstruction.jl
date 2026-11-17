using AdaptiveOpticsCalibration.Diagnostics: HeldOutReconstruction, PupilResidualRatio
using AdaptiveOpticsCalibration.InteractionMatrices: CentralDifference
using AdaptiveOpticsCalibration.Reconstructors:
    MeasurementNulling, ModalReconstructor, ReconstructorSpecification
using LinearAlgebra: I, norm

@testset "signed central differences preserve the divisor" begin
    plus = Float32[0.125, -2, 0]
    minus = Float32[-0.375, -1, 0]
    for amplitude in (0.031f0, 0.031f0 / 2)
        plan = prepare(CentralDifference(amplitude), 3)
        expected = (plus .- minus) ./ (2amplitude)
        @test process(plan, (plus, minus)) == expected
        storage = zeros(Float32, 3, 2)
        output = view(storage, :, 2)
        @test process!(output, plan, (plus, minus)) === output
        @test storage[:, 2] == expected
        @test_throws DimensionMismatch process(plan, (plus, minus[1:2]))
    end
    @test_throws ArgumentError CentralDifference(0.0f0)
    @test_throws ArgumentError prepare(CentralDifference(1.0f0), 0)
end

@testset "modal reconstruction retains only owned rows" begin
    D = Float32[1 0 0.5; 0 2 -0.25]
    half = 0.9f0 .* D
    spec = ReconstructorSpecification(size(D)..., Float32)
    fit = ModalReconstructor(1:2, 1:2; recovery_modes=1:2, rtol=1f-5)
    plan = prepare(fit, spec)
    result = process(plan, (D, half))
    @test result.reconstructor ≈ Float32[1 0; 0 0.5; 0 0]
    @test result.selection == Float32[1 0 0; 0 1 0; 0 0 0]
    @test result.recovery == result.reconstructor * D
    @test result.recovery_error ≈ 0 atol=eps(Float32)
    @test result.derivative_error == norm(D - half) / norm(half)
    @test result.cross_talk == norm(result.recovery[1:2, 3:3]) / sqrt(2)
    @test result.retained_rank == 2
    @test process(plan, (D, zero(D))).derivative_error == norm(D) / eps(Float32)
    @test process(plan, (zero(D), zero(D))).retained_rank == 0
    @test_throws ArgumentError prepare(ModalReconstructor([1], [2]; rtol=1f-5), spec)
    @test_throws DimensionMismatch process(plan, (D, D[:, 1:2]))
end

@testset "nulling retains full row rank and bounds command columns" begin
    D = Float32[1 0 0.5; 0 2 -0.25]
    commands = Matrix{Float32}(I, 3, 3)
    plan = prepare(MeasurementNulling(1:2; rtol=1f-5),
        ReconstructorSpecification(size(D)..., Float32))
    result = process(plan, (commands, D))
    @test result.coefficients ≈ reshape(Float32[0.5, -0.125], 2, 1)
    @test result.scales == ones(Float32, 3)
    @test result.relative_null_residual ≈ 0 atol=eps(Float32)
    @test D * result.projection[:, 3] ≈ zeros(Float32, 2) atol=eps(Float32)
    @test commands == Matrix{Float32}(I, 3, 3)
    @test_throws ArgumentError process(plan, (commands, zero(D)))
end

@testset "held-out reconstruction and piston-removed pupil residual" begin
    plan = prepare(HeldOutReconstruction(), 2)
    matrices = (reshape(Float32[0.5, 0], 2, 1), reshape(Float32[0, 1], 2, 1))
    residuals = (Float32[-2], Float32[3])
    command = Float64[-1, 2]
    result = process(plan, (matrices, residuals, command))
    @test result.estimate == Float32[-1, 3]
    @test eltype(result.estimate) == Float32
    @test result.reconstruction_error == 1 / norm(command)
    @test_throws ArgumentError process(plan, (matrices, residuals, zeros(2)))
    pupil = prepare(PupilResidualRatio(), 2)
    @test process(pupil, ([2.0, 4.0], [8.0, 9.0])) == 0.5
    @test process(pupil, ([2.0, 2.0], [3.0, 3.0])) == 0
    @test_throws DimensionMismatch process(pupil, ([1.0], [1.0]))
end
