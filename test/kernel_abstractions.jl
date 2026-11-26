module KernelAbstractionsTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using KernelAbstractions
using LinearAlgebra
using Test
using ..PhaseRetrievalTests: FrozenLinearizedLiFTModel

const IM = AdaptiveOpticsCalibration.InteractionMatrices
const PB = AdaptiveOpticsCalibration.ProbeBases
const DG = AdaptiveOpticsCalibration.Diagnostics
const TM = AdaptiveOpticsCalibration.Tomography
const PR = AdaptiveOpticsCalibration.PhaseRetrieval
const KAE = Base.get_extension(AdaptiveOpticsCalibration,
    :AdaptiveOpticsCalibrationKernelAbstractionsExt)

@testset "von Kármán sampled scalar approximation error" begin
    extension_module = Base.get_extension(AdaptiveOpticsCalibration,
        :AdaptiveOpticsCalibrationKernelAbstractionsExt)
    for T in (Float32, Float64)
        coefficients = extension_module._von_karman_series_coefficients(T)
        arguments = vcat(T[
            eps(T), T(1e-6), T(0.01), prevfloat(T(0.1)), T(0.1),
            nextfloat(T(0.1)), T(0.5), T(1), T(5),
            prevfloat(T(20)), T(20), nextfloat(T(20)), T(40), T(60), T(80),
        ], [T(0.05 + 0.005 * i) for i in 0:20],
           [T(18 + 0.1 * i) for i in 0:40])
        reference = [x^(T(5) / T(6)) * T(TM.besselk(T(5) / T(6), x))
            for x in arguments]
        approximate = [extension_module._scaled_kv56_device(x, coefficients)
            for x in arguments]
        absolute_errors = abs.(approximate .- reference)
        relative_errors = absolute_errors ./ abs.(reference)
        max_absolute_error = maximum(absolute_errors)
        max_relative_error = maximum(relative_errors)
        @info "von Kármán sampled scalar approximation error" precision=T max_absolute_error max_relative_error
        @test max_absolute_error <= (T === Float32 ? 2f-5 : 2e-7)
        @test max_relative_error <= (T === Float32 ? 2f-5 : 2e-6)
    end
end

function run_operation(method, specification, inputs)
    plan = prepare(method, specification)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    process!(result, workspace, plan, inputs)
    return result, plan, workspace
end

@testset "von Kármán KA CPU covariance" begin
    extension_module = Base.get_extension(AdaptiveOpticsCalibration,
        :AdaptiveOpticsCalibrationKernelAbstractionsExt)
    for T in (Float32, Float64)
        guide = reshape(Complex{T}[
            0, T(0.1), T(0.3), T(0.4),
            T(0.02), T(0.12), T(0.28), T(0.43),
        ], 2, 2, 2)
        fit = reshape(Complex{T}[
            T(0.01), T(0.08), T(0.15),
            T(0.25), T(0.31), T(0.38),
            T(0.03), T(0.11), T(0.18),
            T(0.27), T(0.32), T(0.4),
        ], 3, 2, 2)
        weights = T[T(0.3), T(0.7)]
        specifications = (
            (TM.VonKarmanAutoCovariance(),
                TM.VonKarmanAutoCovarianceSpecification(guide, T(0.16), T(25), weights)),
            (TM.VonKarmanCrossCovariance(),
                TM.VonKarmanCrossCovarianceSpecification(
                    guide, fit, T(0.16), T(25), weights)),
        )
        for (method, specification) in specifications
            reference = process(prepare(method, specification), nothing)
            plan = prepare(method, KernelExecution(specification, CPU(); workgroup_size=4))
            result = allocate_result(plan)
            workspace = allocate_workspace(plan)
            @test KernelAbstractions.get_backend(plan.guide_coordinates) == CPU()
            @test KernelAbstractions.get_backend(plan.fractional_cn2) == CPU()
            @test KernelAbstractions.get_backend(workspace.candidate) == CPU()
            @test @inferred(allocate_result(plan)) isa
                Union{TM.VonKarmanAutoCovarianceProduct,
                      TM.VonKarmanCrossCovarianceProduct}
            @test @inferred(allocate_workspace(plan)) isa
                extension_module.VonKarmanKernelWorkspace
            @test (@inferred process!(result, workspace, plan, nothing)) === result
            actual = method isa TM.VonKarmanAutoCovariance ? result.Cxx : result.Cox
            expected = method isa TM.VonKarmanAutoCovariance ? reference.Cxx : reference.Cox
            @test actual ≈ expected rtol=(T === Float32 ? 2f-4 : 2e-6)
            @test size(actual) == size(expected)
            fill!(actual, T(-42))
            bad = T === Float32 ? T(3e38) : T(1e308)
            fill!(plan.guide_coordinates, complex(bad, bad))
            plan.guide_coordinates[2, 1, 1] = complex(-bad, -bad)
            @test_throws ArgumentError process!(result, workspace, plan, nothing)
            @test all(==(T(-42)), actual)
            copyto!(plan.guide_coordinates, guide)
            @test process!(result, workspace, plan, nothing) === result
            @test actual ≈ expected rtol=(T === Float32 ? 2f-4 : 2e-6)
        end

        repeated = fill(complex(zero(T), zero(T)), 2, 2, 1)
        repeated_spec = TM.VonKarmanAutoCovarianceSpecification(
            repeated, T(0.16), T(25), T[1])
        repeated_plan = prepare(TM.VonKarmanAutoCovariance(),
            KernelExecution(repeated_spec, CPU()))
        repeated_result = process(repeated_plan, nothing)
        @test all(==(repeated_result.Cxx[1]), repeated_result.Cxx)
        one_spec = TM.VonKarmanAutoCovarianceSpecification(
            fill(complex(zero(T), zero(T)), 1, 1, 1), T(0.16), T(25), T[1])
        one_plan = prepare(TM.VonKarmanAutoCovariance(),
            KernelExecution(one_spec, CPU()))
        one_result = allocate_result(one_plan)
        one_workspace = allocate_workspace(one_plan)
        aliased_workspace = typeof(one_workspace)(one_workspace.candidate,
            one_workspace.invalid_first, one_workspace.invalid_first,
            one_workspace.host_invalid)
        @test_throws ArgumentError process!(one_result, aliased_workspace,
            one_plan, nothing)
        if T === Float64
            fill!(one_result.Cxx, -42.0)
            mixed_workspace = extension_module.VonKarmanKernelWorkspace(
                zeros(Float32, 1, 1), one_workspace.invalid_first,
                one_workspace.invalid_second, one_workspace.host_invalid)
            @test_throws ArgumentError process!(one_result, mixed_workspace,
                one_plan, nothing)
            @test one_result.Cxx == fill(-42.0, 1, 1)

            huge_spec = TM.VonKarmanAutoCovarianceSpecification(
                fill(0.0 + 0.0im, 1, 1, 1), 1e-24, 25.0, [1.0])
            huge_plan = prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(huge_spec, CPU()))
            huge_result = TM.VonKarmanAutoCovarianceProduct(fill(-42.0, 1, 1))
            huge_workspace = allocate_workspace(huge_plan)
            huge_mixed = extension_module.VonKarmanKernelWorkspace(
                zeros(Float32, 1, 1), huge_workspace.invalid_first,
                huge_workspace.invalid_second, huge_workspace.host_invalid)
            @test process(prepare(TM.VonKarmanAutoCovariance(), huge_spec),
                nothing).Cxx[1] > floatmax(Float32)
            @test_throws ArgumentError process!(huge_result, huge_mixed,
                huge_plan, nothing)
            @test huge_result.Cxx == fill(-42.0, 1, 1)
        end
        previous_weight = copy(one_plan.fractional_cn2)
        weight_result = TM.VonKarmanAutoCovarianceProduct(
            reshape(one_plan.fractional_cn2, 1, 1))
        @test_throws ArgumentError process!(weight_result, one_workspace,
            one_plan, nothing)
        @test one_plan.fractional_cn2 == previous_weight
        candidate_alias = extension_module.VonKarmanKernelWorkspace(
            reshape(one_plan.fractional_cn2, 1, 1),
            one_workspace.invalid_first, one_workspace.invalid_second,
            one_workspace.host_invalid)
        fill!(one_result.Cxx, T(-42))
        @test_throws ArgumentError process!(one_result, candidate_alias,
            one_plan, nothing)
        @test one_plan.fractional_cn2 == previous_weight
        @test one_result.Cxx == fill(T(-42), 1, 1)
        if T === Float32
            host_alias = extension_module.VonKarmanKernelWorkspace(
                reshape(reinterpret(Float32, one_workspace.host_invalid), 1, 1),
                one_workspace.invalid_first, one_workspace.invalid_second,
                one_workspace.host_invalid)
            @test_throws ArgumentError process!(one_result, host_alias,
                one_plan, nothing)
            @test one_result.Cxx == fill(T(-42), 1, 1)
            @test process!(one_result, one_workspace, one_plan, nothing) ===
                one_result
            @test one_result.Cxx[1] > zero(T)
        end
        @test_throws ArgumentError prepare(TM.VonKarmanAutoCovariance(),
            KernelExecution(TM.VonKarmanAutoCovarianceSpecification(
                guide, T(0.16), T(25), T[T(0.2), T(0.2)]), CPU()))
        bad_result = TM.VonKarmanAutoCovarianceProduct(zeros(T, 3, 3))
        @test_throws DimensionMismatch process!(bad_result,
            allocate_workspace(repeated_plan), repeated_plan, nothing)
    end
end

@testset "KernelExecution contract" begin
    @test Base.get_extension(
        AdaptiveOpticsCalibration,
        :AdaptiveOpticsCalibrationKernelAbstractionsExt,
    ) !== nothing
    @test_throws ArgumentError KernelExecution(nothing, CPU(); workgroup_size=0)

    gains = AdaptiveOpticsCalibration.OpticalGains
    gain_specification = gains.GainSensingSpecification(
        ones(ComplexF64, 2, 2), ones(2, 2, 1), ones(2, 2))
    @test_throws "requires an accelerator backend" prepare(
        gains.GainSensing(), KernelExecution(gain_specification, CPU()))

    method = AdaptiveOpticsCalibration.Reconstructors.MAPReconstructor()
    specification = AdaptiveOpticsCalibration.Reconstructors.ReconstructorSpecification(2, 2)
    @test_throws MethodError prepare(
        method,
        KernelExecution(specification, CPU()),
    )
end

@testset "LiFT KA CPU photon-rate linearization" begin
    reference = [2.0, 4.0, 6.0]
    jacobian = [1.0 2.0; 2.0 -1.0; 0.0 1.0]
    model = FrozenLinearizedLiFTModel(reference, jacobian)
    counts = [6.0, 8.0, 10.0]
    expected_rate = 2 .* counts
    coefficients = [0.25, -0.5]
    model_rate = reference + jacobian * coefficients
    specification = PR.LiFTSpecification(model,
        PR.LiFTExpectedCounts(2; quantum_efficiency=0.25); read_noise_std=0.2)
    execution = KernelExecution(specification, CPU(); workgroup_size=2)
    @test_throws ArgumentError prepare(PR.LiFT(solve_mode=PR.LiFTSolveQR()), execution)
    plan = prepare(PR.LiFT(mode_indices=(2, 1)), execution)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    inverse = workspace.inverse
    kernels = plan.execution.kernels
    @test plan.execution.mode_indices == [2, 1]

    fill!(result.coefficients, 17)
    @test_throws "finite nonnegative values" process!(result, workspace, plan,
        PR.LiFTInputs([6.0, 8.0, NaN]))
    @test_throws "initial_coefficients must contain only finite values" process!(
        result, workspace, plan, PR.LiFTInputs(counts; initial_coefficients=[Inf, 0.0]))
    @test result.coefficients == [17.0, 17.0]

    kernels.copy_rate(inverse.observation_rate, counts,
        plan.plan.observation_to_rate_scale, length(counts); ndrange=length(counts))
    @test inverse.observation_rate == expected_rate
    copyto!(inverse.coefficients, coefficients)
    for jacobian_method in (PR.LiFTAnalyticJacobian(), PR.LiFTNumericalJacobian(0x1p-12))
        @test KAE._lift_jacobian!(workspace, plan, jacobian_method) == jacobian[:, [2, 1]]
        @test inverse.coefficients == coefficients
    end

    for (scaling, expected_scale) in (
        (PR.LiFTTotalRateMatching(), sum(expected_rate) / sum(model_rate)),
        (PR.LiFTPeakRateMatching(), maximum(expected_rate) / maximum(model_rate)),
        (PR.LiFTPhysicalRatePreservation(), 1.0),
    )
        copyto!(inverse.model_rate, model_rate)
        scale, objective = KAE._lift_model_scale!(workspace, plan, scaling)
        @test scale ≈ expected_scale rtol=4eps(Float64)
        @test objective ≈ sum(expected_scale .* model_rate) rtol=4eps(Float64)
        @test inverse.model_rate ≈ expected_scale .* model_rate rtol=4eps(Float64)
    end

    kernels.residual(inverse.residual, inverse.observation_rate, inverse.model_rate,
        length(counts); ndrange=length(counts))
    KAE._lift_static_weights!(workspace, plan, PR.LiFTReadNoiseWeighting())
    kernels.apply_weights(inverse.jacobian, inverse.residual, inverse.sqrt_weights,
        size(jacobian)...; ndrange=size(jacobian))
    @test inverse.jacobian ≈ jacobian[:, [2, 1]] ./ 0.4 rtol=4eps(Float64)
    @test inverse.residual ≈ (expected_rate - model_rate) ./ 0.4 rtol=4eps(Float64)
    @test KAE._lift_norm!(workspace, plan, [3e200, 0.0, -4e200]) ≈ 5e200
    @test KAE._lift_norm!(workspace, plan, zeros(3)) == 0

    update = [0.125, 0.25]
    kernels.scatter_update(inverse.coefficients, update, plan.execution.mode_indices,
        length(update); ndrange=length(update))
    kernels.gather(result.coefficients, inverse.coefficients, plan.execution.mode_indices,
        length(update); ndrange=length(update))
    @test result.coefficients == coefficients[[2, 1]] + update
end

@testset "LiFT KA CPU count-domain variance and frozen weights" begin
    model_rate = [2.0, 4.0, 6.0]
    model = FrozenLinearizedLiFTModel(model_rate, ones(3, 1))
    specification = PR.LiFTSpecification(model,
        PR.LiFTExpectedCounts(2; quantum_efficiency=0.25); read_noise_std=0.2)
    variance = [0.0, 0.5, 4.0]
    first_model_variance = 2 .* model_rate .+ 0.4^2
    for (weighting, first_variance, second_variance) in (
        (PR.LiFTVarianceMapWeighting(variance), 4 .* variance, 4 .* variance),
        (PR.LiFTInitialModelWeighting(), first_model_variance, first_model_variance),
        (PR.LiFTIterativeModelWeighting(), first_model_variance, 4 .* model_rate .+ 0.4^2),
    )
        plan = prepare(PR.LiFT(weighting=weighting),
            KernelExecution(specification, CPU(); workgroup_size=2))
        workspace = allocate_workspace(plan)
        copyto!(workspace.inverse.model_rate, model_rate)
        KAE._lift_static_weights!(workspace, plan, plan.plan.weighting)
        KAE._lift_update_weights!(workspace, plan, plan.plan.weighting, 1)
        @test workspace.inverse.sqrt_weights ≈
            1 ./ sqrt.(max.(first_variance, eps(Float64))) rtol=4eps(Float64)
        workspace.inverse.model_rate .*= 2
        KAE._lift_update_weights!(workspace, plan, plan.plan.weighting, 2)
        @test workspace.inverse.sqrt_weights ≈
            1 ./ sqrt.(max.(second_variance, eps(Float64))) rtol=4eps(Float64)
    end
end

@testset "LiFT normal-equation and SVD steps on host storage" begin
    jacobian = [1.0 2.0; 0.0 0.0; 0.0 0.0]
    residual = [1.0, 2.0, 3.0]
    model = FrozenLinearizedLiFTModel(fill(10.0, 3), jacobian)
    plan = prepare(PR.LiFT(),
        KernelExecution(PR.LiFTSpecification(model, PR.LiFTPhotonRate()), CPU()))
    workspace = allocate_workspace(plan)
    inverse = workspace.inverse
    copyto!(inverse.normal, jacobian' * jacobian)
    copyto!(inverse.jacobian, jacobian)
    copyto!(inverse.residual, residual)
    condition, load = KAE._lift_normal_metrics!(workspace, plan)
    @test condition == 4.0
    @test load == 2.5 * sqrt(eps(Float64))

    regularization = 0.25
    rhs = jacobian' * residual
    expected = (jacobian' * jacobian + regularization * I) \ rhs
    for damping in (PR.LiFTDampingNone(), PR.LiFTLevenbergMarquardt())
        copyto!(inverse.rhs, rhs)
        @test PR._solve_device_normal!(workspace, damping, regularization, regularization) ==
            (regularization, false)
        @test inverse.rhs ≈ expected rtol=2e-14
    end
    @test PR._device_svd_step!(inverse, svd(jacobian; full=false), regularization) ==
        (regularization, true)
    @test inverse.rhs ≈ expected rtol=2e-14
end

@testset "structured KA CPU kernels match scalar CPU" begin
    zonal_responses = Float32[
        10.5 -6.75
        9.5 -7.25
        14 -11
        6 -3
        8.5 -1
        11.5 -13
    ]
    hadamard_responses = Float32[
        95.25 -22.875
        104.75 -37.125
        92.75 -21.125
        107.25 -38.875
        105.25 -36.875
        94.75 -23.125
        106.75 -39.125
        93.25 -20.875
    ]
    specification = IM.ResponseSpecification(2)
    for (method, responses) in (
        (IM.ZonalPushPull(Float32[0.5, 2, 1.5]), zonal_responses),
        (IM.HadamardPushPull(Float32[0.25, 2, 0.5]), hadamard_responses),
    )
        scalar, _, _ = run_operation(method, specification, responses)
        kernel, plan, workspace = run_operation(
            method,
            KernelExecution(specification, CPU(); workgroup_size=4),
            responses,
        )
        @test kernel.matrix ≈ scalar.matrix atol=2f-6
        @test KernelAbstractions.get_backend(kernel.matrix) == CPU()
        @test KernelAbstractions.get_backend(workspace.staged_matrix) == CPU()
        @test KernelAbstractions.get_backend(plan.reciprocal_scales) == CPU()
        @test KernelAbstractions.get_backend(workspace.invalid_first) == CPU()
        @test KernelAbstractions.get_backend(workspace.invalid_second) == CPU()

        fill!(kernel.matrix, 17f0)
        invalid = copy(responses)
        invalid[begin] = Float32(NaN)
        @test process!(kernel, workspace, plan, invalid) === kernel
        @test kernel.status === IM.StructuredNonFiniteResponse
        @test kernel.matrix == fill(17f0, size(kernel.matrix))
        process!(kernel, workspace, plan, responses)
        @test kernel.matrix ≈ scalar.matrix atol=2f-6
    end

    method = IM.ZonalPushPull(Float32[2])
    specification = IM.ResponseSpecification(1)
    result, plan, workspace = run_operation(
        method,
        KernelExecution(specification, CPU(); workgroup_size=4),
        reshape(Float32[5, 1], 2, 1),
    )
    @test result.matrix == ones(Float32, 1, 1)
    @test length(workspace.invalid_first) == 1
    @test process!(result, workspace, plan, reshape(Float32[5, 1], 2, 1)) === result
    @test result.status === IM.StructuredValid
    @test @inferred(process!(
        result.matrix,
        workspace,
        plan,
        reshape(Float32[5, 1], 2, 1),
    )) === IM.StructuredValid

    fill!(result.matrix, 17f0)
    overflow = reshape(Float32[floatmax(Float32), -floatmax(Float32)], 2, 1)
    @test process!(result, workspace, plan, overflow) === result
    @test result.status === IM.StructuredNonFiniteCalculation
    @test result.matrix == fill(17f0, 1, 1)
    @test @inferred(process!(result.matrix, workspace, plan, overflow)) ===
        IM.StructuredNonFiniteCalculation
    @test process!(result, workspace, plan, reshape(Float32[5, 1], 2, 1)) === result
    @test result.status === IM.StructuredValid
end

@testset "probe-basis KA CPU kernels match scalar CPU" begin
    methods = (
        PB.ModalPushPull(Float32[0.5 -1 0; -0.25 0.5 2]),
        PB.ZonalPushPull(Float32[1, 2]),
        PB.HadamardPushPull(Float32[1, 2, 3]),
    )
    for method in methods
        scalar, _, _ = run_operation(method, nothing, nothing)
        kernel, _, workspace = run_operation(
            method,
            KernelExecution(nothing, CPU(); workgroup_size=4),
            nothing,
        )
        @test workspace === nothing
        @test kernel.commands == scalar.commands
        @test KernelAbstractions.get_backend(kernel.commands) == CPU()
    end
end

@testset "stability KA CPU kernel matches scalar CPU" begin
    first_storage = zeros(Float64, 4, 4)
    second_storage = zeros(Float64, 4, 4)
    first_window = @view first_storage[1:2:4, 1:2:4]
    second_window = @view second_storage[1:2:4, 1:2:4]
    first_window .= Matrix{Float64}(I, 2, 2)
    second_window .= 2 .* first_window
    input = DG.InteractionMatrixPair(first_window, second_window)
    method = DG.InteractionMatrixStability(0.1, 0.5, 0.5)
    scalar, _, _ = run_operation(method, input, input)
    kernel, plan, workspace = run_operation(
        method,
        KernelExecution(input, CPU(); workgroup_size=2),
        input,
    )
    for field in fieldnames(typeof(scalar))
        @test getfield(kernel, field) == getfield(scalar, field)
    end
    @test KernelAbstractions.get_backend(workspace.first_column_squares) == CPU()

    snapshot = Tuple(getfield(kernel, field) for field in fieldnames(typeof(kernel)))
    invalid = DG.InteractionMatrixPair(
        [NaN 0.0; 0.0 1.0],
        Matrix{Float64}(I, 2, 2),
    )
    @test_throws ArgumentError process!(kernel, workspace, plan, invalid)
    @test Tuple(getfield(kernel, field) for field in fieldnames(typeof(kernel))) == snapshot
end

end
