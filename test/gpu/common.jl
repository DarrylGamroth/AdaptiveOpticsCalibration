using TOML

struct AcceleratorLinearLiFTModel{T<:AbstractFloat,V<:AbstractVector{T},M<:AbstractMatrix{T}} <:
       AdaptiveOpticsCalibration.PhaseRetrieval.AbstractLiFTForwardModel{T}
    reference_rate::V
    jacobian::M
end

const PR = AdaptiveOpticsCalibration.PhaseRetrieval
const GAIN_SENSING_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "..", "fixtures", "aos_s6_gsc_cpu.toml"))

function gain_sensing_fixture_array(section, ::Type{T}) where {T<:AbstractFloat}
    return reshape(T.(section["values"]), Tuple(Int.(section["shape"])))
end

function gain_sensing_fixture_mask(::Type{T}) where {T<:AbstractFloat}
    real_part = gain_sensing_fixture_array(GAIN_SENSING_FIXTURE["mask"]["real"], T)
    imaginary_part = gain_sensing_fixture_array(GAIN_SENSING_FIXTURE["mask"]["imag"], T)
    return complex.(real_part, imaginary_part)
end

function covariance_workspace_with(
    workspace;
    phase_covariance=workspace.phase_covariance,
    invalid_first=workspace.invalid_first,
    metric_first=workspace.metric_first,
    pivot=workspace.pivot,
)
    return typeof(workspace)(
        phase_covariance,
        workspace.measurement_noise_covariance,
        workspace.multiplication,
        workspace.system,
        workspace.factor,
        workspace.right_hand_side,
        workspace.solution,
        workspace.candidate,
        invalid_first,
        workspace.invalid_second,
        pivot,
        metric_first,
        workspace.metric_second,
        workspace.host_invalid,
        workspace.host_metric,
    )
end

function gain_sensing_extended_inputs(::Type{T}) where {T<:AbstractFloat}
    side = 48
    mask = [complex(
        T(0.8) + T(0.02) * T(mod(13 * row + 7 * column +
            3 * row * column, 19)),
        T(0.05) + T(0.015) * T(mod(5 * row + 11 * column +
            row * column, 17)),
    ) for row in 1:side, column in 1:side]
    basis = Array{T}(undef, side, side, 3)
    for mode in 1:3, column in 1:side, row in 1:side
        basis[row, column, mode] = T(mod((row + 2 * mode) *
            (3 * column + 5 * mode), 23) - 11) / T(11)
    end
    reference = [T(20 + mod(7 * row + 11 * column +
        3 * row * column, 29)) for row in 1:side, column in 1:side]
    current = [reference[row, column] * (one(T) + T(0.1) *
        T(mod(17 * row + 5 * column + 2 * row * column, 31) - 15) /
        T(15)) for row in 1:side, column in 1:side]
    return mask, basis, reference, current
end

PR.coefficient_count(model::AcceleratorLinearLiFTModel) = size(model.jacobian, 2)
PR.observation_axes(model::AcceleratorLinearLiFTModel) = axes(model.reference_rate)
PR.allocate_model_workspace(::AcceleratorLinearLiFTModel) = nothing
PR.allocate_photon_rate(model::AcceleratorLinearLiFTModel) = similar(model.reference_rate)
function PR.predict_photon_rate!(out, model::AcceleratorLinearLiFTModel, ::Nothing, coefficients)
    mul!(out, model.jacobian, coefficients)
    out .+= model.reference_rate
    return out
end
function PR.analytic_photon_rate_jacobian!(
    jacobian,
    model::AcceleratorLinearLiFTModel,
    ::Nothing,
    coefficients,
    mode_indices,
)
    for column in axes(jacobian, 2)
        copyto!(
            @view(jacobian[:, column]),
            @view(model.jacobian[:, mode_indices[column]]),
        )
    end
    return jacobian
end


function test_accelerator_tomography_covariance(backend, to_device, device_array_type)
    TM = AdaptiveOpticsCalibration.Tomography
    VK_EXT = Base.get_extension(AdaptiveOpticsCalibration,
        :AdaptiveOpticsCalibrationKernelAbstractionsExt)
    @testset "von Kármán covariance on accelerator" begin
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
            for (method, specification) in (
                (TM.VonKarmanAutoCovariance(),
                    TM.VonKarmanAutoCovarianceSpecification(
                        guide, T(0.16), T(25), weights)),
                (TM.VonKarmanCrossCovariance(),
                    TM.VonKarmanCrossCovarianceSpecification(
                        guide, fit, T(0.16), T(25), weights)),
            )
                reference = process(prepare(method, specification), nothing)
                plan = prepare(method,
                    KernelExecution(specification, backend; workgroup_size=64))
                result = @inferred allocate_result(plan)
                workspace = @inferred allocate_workspace(plan)
                @test plan.guide_coordinates isa device_array_type
                @test plan.fractional_cn2 isa device_array_type
                @test workspace.candidate isa device_array_type
                @test workspace.invalid_first isa device_array_type
                @test workspace.invalid_second isa device_array_type
                if method isa TM.VonKarmanCrossCovariance
                    @test plan.fit_coordinates isa device_array_type
                end
                @test (@inferred process!(result, workspace, plan, nothing)) === result
                actual = method isa TM.VonKarmanAutoCovariance ? result.Cxx : result.Cox
                expected = method isa TM.VonKarmanAutoCovariance ? reference.Cxx : reference.Cox
                @test actual isa device_array_type
                @test Array(actual) ≈ expected rtol=(T === Float32 ? 2f-4 : 2e-6)
                fill!(actual, T(-42))
                host_output = fill(T(-42), size(actual))
                host_result = method isa TM.VonKarmanAutoCovariance ?
                    TM.VonKarmanAutoCovarianceProduct(host_output) :
                    TM.VonKarmanCrossCovarianceProduct(host_output)
                @test_throws ArgumentError process!(host_result, workspace,
                    plan, nothing)
                @test host_output == fill(T(-42), size(actual))
                host_candidate = VK_EXT.VonKarmanKernelWorkspace(
                    zeros(T, size(workspace.candidate)), workspace.invalid_first,
                    workspace.invalid_second, workspace.host_invalid)
                @test_throws ArgumentError process!(result, host_candidate,
                    plan, nothing)
                @test Array(actual) == fill(T(-42), size(actual))
                other_precision = T === Float32 ? Float64 : Float32
                wrong_candidate = KernelAbstractions.allocate(backend,
                    other_precision, size(workspace.candidate))
                mixed_workspace = VK_EXT.VonKarmanKernelWorkspace(wrong_candidate,
                    workspace.invalid_first, workspace.invalid_second,
                    workspace.host_invalid)
                @test_throws ArgumentError process!(result, mixed_workspace,
                    plan, nothing)
                @test all(==(T(-42)), Array(actual))
                bad = T === Float32 ? T(3e38) : T(1e308)
                far_guide = fill(complex(bad, bad), size(guide))
                far_guide[2] = complex(-bad, -bad)
                copyto!(plan.guide_coordinates, far_guide)
                @test_throws ArgumentError process!(result, workspace, plan, nothing)
                @test all(==(T(-42)), Array(actual))
                copyto!(plan.guide_coordinates, to_device(guide))
                @test process!(result, workspace, plan, nothing) === result
                @test Array(actual) ≈ expected rtol=(T === Float32 ? 2f-4 : 2e-6)
            end
            bad_guide = to_device(guide)
            bad_weights = to_device(weights)
            @test_throws ArgumentError prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(TM.VonKarmanAutoCovarianceSpecification(
                    bad_guide, T(0.16), T(25), weights), backend))
            @test_throws ArgumentError prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(TM.VonKarmanAutoCovarianceSpecification(
                    guide, T(0.16), T(25), bad_weights), backend))
            @test_throws ArgumentError prepare(TM.VonKarmanCrossCovariance(),
                KernelExecution(TM.VonKarmanCrossCovarianceSpecification(
                    guide, to_device(fit), T(0.16), T(25), weights), backend))

            one_spec = TM.VonKarmanAutoCovarianceSpecification(
                fill(complex(zero(T), zero(T)), 1, 1, 1), T(0.16), T(25), T[1])
            one_plan = prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(one_spec, backend))
            one_workspace = allocate_workspace(one_plan)
            previous_weight = Array(one_plan.fractional_cn2)
            weight_result = TM.VonKarmanAutoCovarianceProduct(
                reshape(one_plan.fractional_cn2, 1, 1))
            @test_throws ArgumentError process!(weight_result, one_workspace,
                one_plan, nothing)
            @test Array(one_plan.fractional_cn2) == previous_weight
            candidate_alias = VK_EXT.VonKarmanKernelWorkspace(
                reshape(one_plan.fractional_cn2, 1, 1),
                one_workspace.invalid_first, one_workspace.invalid_second,
                one_workspace.host_invalid)
            one_result = allocate_result(one_plan)
            fill!(one_result.Cxx, T(-42))
            @test_throws ArgumentError process!(one_result, candidate_alias,
                one_plan, nothing)
            @test Array(one_plan.fractional_cn2) == previous_weight
            @test Array(one_result.Cxx) == fill(T(-42), 1, 1)
        end

        for T in (Float32, Float64)
            guide_weighted = reshape(Complex{T}[
                0, T(0.1), T(0.02), T(0.12),
            ], 2, 1, 2)
            weights = T[0, one(T) + T(1e-7)]
            specification = TM.VonKarmanAutoCovarianceSpecification(
                guide_weighted, T(0.16), T(25), weights)
            normalized = TM.VonKarmanAutoCovarianceSpecification(
                guide_weighted, T(0.16), T(25), T[0, 1])
            reference = process(prepare(TM.VonKarmanAutoCovariance(),
                specification), nothing).Cxx
            normalized_reference = process(prepare(TM.VonKarmanAutoCovariance(),
                normalized), nothing).Cxx
            device = process(prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(specification, backend)), nothing)
            @test Array(device.Cxx) ≈ reference rtol=(T === Float32 ? 2f-4 : 2e-6)
            @test reference != normalized_reference
        end

        # n=8 pupil sampling with two guide sources, four fit sources,
        # and two independent layers exercises the source-major layout.
        T = Float32
        guide_n8 = [complex(
            T(mod(sample - 1, 8) + 2 * (source - 1)) / T(80),
            T((sample - 1) ÷ 8 + layer - 1) / T(80),
        ) for sample in 1:64, source in 1:2, layer in 1:2]
        fit_n8 = [complex(
            T(mod(sample - 1, 8) + source - 1) / T(80),
            T((sample - 1) ÷ 8 + 2 * (layer - 1)) / T(80),
        ) for sample in 1:64, source in 1:4, layer in 1:2]
        for (method, specification) in (
            (TM.VonKarmanAutoCovariance(),
                TM.VonKarmanAutoCovarianceSpecification(
                    guide_n8, T(0.16), T(25), T[T(0.3), T(0.7)])),
            (TM.VonKarmanCrossCovariance(),
                TM.VonKarmanCrossCovarianceSpecification(
                    guide_n8, fit_n8, T(0.16), T(25), T[T(0.3), T(0.7)])),
        )
            reference = process(prepare(method, specification), nothing)
            device = process(prepare(method,
                KernelExecution(specification, backend; workgroup_size=64)), nothing)
            expected = method isa TM.VonKarmanAutoCovariance ? reference.Cxx : reference.Cox
            actual = method isa TM.VonKarmanAutoCovariance ? device.Cxx : device.Cox
            @test actual isa device_array_type
            @test Array(actual) ≈ expected rtol=2f-4
        end

        # Both approximation branch transitions are exercised by actual-device
        # separations, including the u ≥ 20 asymptotic tail.
        for T in (Float32, Float64)
            outer_scale_m = T(25)
            arguments = T[0, T(0.1), T(20), T(25)]
            branch_guide = reshape(complex.(arguments .* outer_scale_m ./ T(2π),
                zero(T)), 4, 1, 1)
            branch_spec = TM.VonKarmanAutoCovarianceSpecification(
                branch_guide, T(0.16), outer_scale_m, T[1])
            reference = process(prepare(TM.VonKarmanAutoCovariance(), branch_spec),
                nothing).Cxx
            result = process(prepare(TM.VonKarmanAutoCovariance(),
                KernelExecution(branch_spec, backend)), nothing)
            @test all(isapprox.(Array(result.Cxx), reference;
                rtol=(T === Float32 ? 2f-4 : 2e-6), atol=T(1e-12)))
        end
    end
end


function test_accelerator_covariance_reconstructor(backend, to_device, device_array_type)
    TM = AdaptiveOpticsCalibration.Tomography
    VK_EXT = Base.get_extension(AdaptiveOpticsCalibration,
        :AdaptiveOpticsCalibrationKernelAbstractionsExt)
    function covariance_alias_workspace(workspace, phase_covariance, ::Type{T}) where {T}
        noise_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        multiplication_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        system_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        factor_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        rhs_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        solution_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        candidate_parent = KernelAbstractions.allocate(backend, T, 2, 3)
        return VK_EXT.CovarianceReconstructorKernelWorkspace(
            phase_covariance,
            @view(noise_parent[:, 1:2]),
            @view(multiplication_parent[:, 1:2]),
            @view(system_parent[:, 1:2]),
            @view(factor_parent[:, 1:2]),
            @view(rhs_parent[:, 1:2]),
            @view(solution_parent[:, 1:2]),
            @view(candidate_parent[:, 1:2]),
            workspace.invalid_first,
            workspace.invalid_second,
            workspace.pivot,
            workspace.metric_first,
            workspace.metric_second,
            workspace.host_invalid,
            workspace.host_metric,
        )
    end
    @testset "covariance reconstructor on accelerator" begin
        for T in (Float32, Float64)
            tolerance = T === Float32 ? 3f-5 : 3e-12
            projection = T[1 0 -1; T(0.5) 2 1]
            phase_covariance = T[2 T(0.25) 0;
                                 T(0.25) T(1.5) T(0.1);
                                 0 T(0.1) 1]
            fit_phase_covariance = T[1 T(0.5) T(-0.25);
                                     T(-0.5) T(0.75) 2]
            measurement_noise_covariance = T[T(0.4) T(0.1);
                                             T(0.1) T(0.6)]
            host_input = TM.CovarianceReconstructorInputs(
                projection, phase_covariance, fit_phase_covariance,
                measurement_noise_covariance)
            specification = TM.CovarianceReconstructorSpecification(3, 2, 2, T)
            host_result = process(prepare(TM.CovarianceReconstructor(), specification),
                host_input)
            input = TM.CovarianceReconstructorInputs(
                to_device(projection), to_device(phase_covariance),
                to_device(fit_phase_covariance), to_device(measurement_noise_covariance))
            plan = prepare(TM.CovarianceReconstructor(),
                KernelExecution(specification, backend; workgroup_size=64))
            result = allocate_result(plan)
            workspace = allocate_workspace(plan)
            @test result.matrix isa device_array_type
            @test workspace.phase_covariance isa device_array_type
            @test workspace.measurement_noise_covariance isa device_array_type
            @test workspace.system isa device_array_type
            @test workspace.candidate isa device_array_type
            @test input.projection isa device_array_type
            @test @inferred(process!(result, workspace, plan, input)) === result
            @test Array(result.matrix) ≈ host_result.matrix rtol=tolerance atol=tolerance
            @test result.diagnostics.used_lu_fallback == host_result.diagnostics.used_lu_fallback
            @test result.diagnostics.system_matrix_norm ≈
                host_result.diagnostics.system_matrix_norm rtol=tolerance
            @test result.diagnostics.reconstructor_norm ≈
                host_result.diagnostics.reconstructor_norm rtol=tolerance

            for (phase_covariance_value, fit_phase_covariance_value) in (
                (T === Float32 ? T(1e20) : T(1e200), one(T)),
                (one(T), T === Float32 ? T(1e-30) : T(1e-200)),
            )
                extreme_specification = TM.CovarianceReconstructorSpecification(1, 1, 1, T)
                extreme_host = TM.CovarianceReconstructorInputs(
                    reshape(T[1], 1, 1), reshape(T[phase_covariance_value], 1, 1),
                    reshape(T[fit_phase_covariance_value], 1, 1), zeros(T, 1, 1))
                extreme = TM.CovarianceReconstructorInputs(
                    to_device(extreme_host.projection),
                    to_device(extreme_host.phase_covariance),
                    to_device(extreme_host.fit_phase_covariance),
                    to_device(extreme_host.measurement_noise_covariance))
                extreme_host_result = process(prepare(TM.CovarianceReconstructor(),
                    extreme_specification), extreme_host)
                extreme_result = process(prepare(TM.CovarianceReconstructor(),
                    KernelExecution(extreme_specification, backend; workgroup_size=64)), extreme)
                @test Array(extreme_result.matrix) ≈ extreme_host_result.matrix rtol=tolerance
                @test extreme_result.diagnostics.system_matrix_norm ≈
                    extreme_host_result.diagnostics.system_matrix_norm rtol=tolerance
                @test extreme_result.diagnostics.reconstructor_norm ≈
                    extreme_host_result.diagnostics.reconstructor_norm rtol=tolerance
            end

            malformed_result = allocate_result(plan)
            fill!(malformed_result.matrix, T(-42))
            malformed_result.diagnostics = TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))
            bad_phase_workspace = covariance_workspace_with(workspace;
                phase_covariance=KernelAbstractions.allocate(backend, T, 1, 1))
            @test_throws DimensionMismatch process!(malformed_result, bad_phase_workspace,
                plan, input)
            bad_flag_workspace = covariance_workspace_with(workspace;
                invalid_first=KernelAbstractions.allocate(backend, UInt32, 1))
            @test_throws DimensionMismatch process!(malformed_result, bad_flag_workspace,
                plan, input)
            bad_metric_workspace = covariance_workspace_with(workspace;
                metric_first=KernelAbstractions.allocate(backend, T, 1))
            @test_throws DimensionMismatch process!(malformed_result, bad_metric_workspace,
                plan, input)
            bad_pivot_workspace = covariance_workspace_with(workspace;
                pivot=KernelAbstractions.allocate(backend, Cint, 1))
            @test_throws DimensionMismatch process!(malformed_result, bad_pivot_workspace,
                plan, input)
            @test Array(malformed_result.matrix) == fill(T(-42), 2, 2)
            @test malformed_result.diagnostics ==
                TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))

            alias_specification = TM.CovarianceReconstructorSpecification(2, 2, 2, T)
            alias_plan = prepare(TM.CovarianceReconstructor(),
                KernelExecution(alias_specification, backend; workgroup_size=64))
            alias_workspace_base = allocate_workspace(alias_plan)
            alias_input = TM.CovarianceReconstructorInputs(
                to_device(Matrix{T}(I, 2, 2)), to_device(Matrix{T}(I, 2, 2)),
                to_device(Matrix{T}(I, 2, 2)), to_device(Matrix{T}(I, 2, 2)))
            for transpose_result in (false, true)
                storage = KernelAbstractions.allocate(backend, T, 2, 3)
                fill!(storage, one(T))
                shared_phase_covariance = @view(storage[:, 1:2])
                result_view = @view(storage[:, 2:3])
                result_matrix = transpose_result ? transpose(result_view) : result_view
                shared_result = TM.CovarianceReconstructorProduct(
                    result_matrix, TM.CovarianceReconstructorDiagnostics(false, T(5), T(7)))
                shared_workspace = covariance_alias_workspace(
                    alias_workspace_base, shared_phase_covariance, T)
                shared_before = Array(storage)
                shared_failure = try
                    process!(shared_result, shared_workspace, alias_plan, alias_input)
                    nothing
                catch error
                    error
                end
                @test shared_failure isa ArgumentError
                @test occursin("must not alias", sprint(showerror, shared_failure))
                @test Array(storage) == shared_before
                @test shared_result.diagnostics ==
                    TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))
            end

            fallback_host = TM.CovarianceReconstructorInputs(
                Matrix{T}(I, 2, 2), T[1 0; 0 -2], T[2 -1], zeros(T, 2, 2))
            fallback = TM.CovarianceReconstructorInputs(
                to_device(fallback_host.projection),
                to_device(fallback_host.phase_covariance),
                to_device(fallback_host.fit_phase_covariance),
                to_device(fallback_host.measurement_noise_covariance))
            fallback_specification = TM.CovarianceReconstructorSpecification(2, 2, 1, T)
            fallback_plan = prepare(TM.CovarianceReconstructor(),
                KernelExecution(fallback_specification, backend; workgroup_size=64))
            fallback_result = process(fallback_plan, fallback)
            host_fallback = process(prepare(TM.CovarianceReconstructor(),
                fallback_specification), fallback_host)
            @test fallback_result.matrix isa device_array_type
            @test fallback_result.diagnostics.used_lu_fallback
            @test Array(fallback_result.matrix) ≈ host_fallback.matrix rtol=tolerance atol=tolerance

            preserved = allocate_result(fallback_plan)
            fill!(preserved.matrix, T(-42))
            preserved.diagnostics = TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))
            singular_host = TM.CovarianceReconstructorInputs(
                Matrix{T}(I, 2, 2), zeros(T, 2, 2), T[1 -1], zeros(T, 2, 2))
            singular = TM.CovarianceReconstructorInputs(
                to_device(singular_host.projection), to_device(singular_host.phase_covariance),
                to_device(singular_host.fit_phase_covariance),
                to_device(singular_host.measurement_noise_covariance))
            @test_throws SingularException process!(preserved, allocate_workspace(fallback_plan),
                fallback_plan, singular)
            @test Array(preserved.matrix) == fill(T(-42), 1, 2)
            @test preserved.diagnostics == TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))

            asymmetric_host = TM.CovarianceReconstructorInputs(
                Matrix{T}(I, 2, 2), T[2 0; 1 2], T[0 1], zeros(T, 2, 2))
            asymmetric = TM.CovarianceReconstructorInputs(
                to_device(asymmetric_host.projection), to_device(asymmetric_host.phase_covariance),
                to_device(asymmetric_host.fit_phase_covariance),
                to_device(asymmetric_host.measurement_noise_covariance))
            @test_throws ArgumentError process!(preserved, allocate_workspace(fallback_plan),
                fallback_plan, asymmetric)
            @test Array(preserved.matrix) == fill(T(-42), 1, 2)
            @test preserved.diagnostics == TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))

            near_difference = sqrt(eps(T)) / T(4)
            nearly_symmetric_host = TM.CovarianceReconstructorInputs(
                Matrix{T}(I, 2, 2), T[2 near_difference; -near_difference 1],
                T[1 2], zeros(T, 2, 2))
            nearly_symmetric = TM.CovarianceReconstructorInputs(
                to_device(nearly_symmetric_host.projection),
                to_device(nearly_symmetric_host.phase_covariance),
                to_device(nearly_symmetric_host.fit_phase_covariance),
                to_device(nearly_symmetric_host.measurement_noise_covariance))
            near_result = process(fallback_plan, nearly_symmetric)
            host_near = process(prepare(TM.CovarianceReconstructor(),
                fallback_specification), nearly_symmetric_host)
            @test Array(near_result.matrix) ≈ host_near.matrix rtol=tolerance atol=tolerance
            @test near_result.diagnostics.used_lu_fallback ==
                host_near.diagnostics.used_lu_fallback

            invalid_projection = copy(projection)
            invalid_projection[1] = T(NaN)
            invalid = TM.CovarianceReconstructorInputs(
                to_device(invalid_projection), to_device(phase_covariance),
                to_device(fit_phase_covariance), to_device(measurement_noise_covariance))
            output = allocate_result(plan)
            fill!(output.matrix, T(-42))
            output.diagnostics = TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))
            @test_throws ArgumentError process!(output, allocate_workspace(plan), plan, invalid)
            @test Array(output.matrix) == fill(T(-42), 2, 2)
            @test output.diagnostics == TM.CovarianceReconstructorDiagnostics(false, T(5), T(7))
        end
    end
end


function test_accelerator_gain_sensing_frozen(backend, to_device, device_array_type)
    OG = AdaptiveOpticsCalibration.OpticalGains
    @testset "gain sensing on frozen AOS image" begin
        @test GAIN_SENSING_FIXTURE["schema"] ==
            "test.adaptive-optics-sim/gain-sensing-camera-cpu/1"
        for T in (Float32, Float64)
            mask = gain_sensing_fixture_mask(T)
            basis = gain_sensing_fixture_array(GAIN_SENSING_FIXTURE["basis"], T)
            reference = gain_sensing_fixture_array(
                GAIN_SENSING_FIXTURE["frames"]["reference"], T)
            current = gain_sensing_fixture_array(
                GAIN_SENSING_FIXTURE["frames"]["current"], T)
            specification = OG.GainSensingSpecification(mask, basis, reference)
            plan = prepare(
                OG.GainSensing(T(GAIN_SENSING_FIXTURE["sensitivity_floor"])),
                KernelExecution(specification, backend; workgroup_size=64),
            )
            product = allocate_result(plan)
            workspace = allocate_workspace(plan)
            input = OG.GainSensingInputs(to_device(current))
            @test (@inferred process!(product, workspace, plan, input)) === product
            @test product.optical_gains isa device_array_type
            @test product.current_sensitivities isa device_array_type
            @test product.weak_mode_mask isa device_array_type
            @test plan.mask isa device_array_type
            @test plan.basis_products isa device_array_type
            @test workspace.buffer isa device_array_type
            gain_tolerance = T === Float32 ? 2f-4 : 1e-9
            sensitivity_tolerance = T === Float32 ? 1f-7 : 1e-12
            expected_reference_ir = gain_sensing_fixture_array(
                GAIN_SENSING_FIXTURE["impulse_responses"]["reference"], T)
            @test Array(plan.reference_impulse_response) ≈
                expected_reference_ir atol=(T === Float32 ? 1f-7 : 1e-12)
            expected_reference_sensitivity =
                GAIN_SENSING_FIXTURE["sensitivities"]["reference"]
            @test Array(plan.reference_sensitivities) ≈ complex.(
                T.(expected_reference_sensitivity["real"]),
                T.(expected_reference_sensitivity["imag"]),
            ) atol=sensitivity_tolerance
            @test Array(product.optical_gains) ≈
                T.(GAIN_SENSING_FIXTURE["optical_gains"]) rtol=gain_tolerance
            expected_sensitivity = GAIN_SENSING_FIXTURE["sensitivities"]["current"]
            @test Array(product.current_sensitivities) ≈ complex.(
                T.(expected_sensitivity["real"]),
                T.(expected_sensitivity["imag"]),
            ) atol=sensitivity_tolerance
            @test Array(product.weak_mode_mask) ==
                Bool.(GAIN_SENSING_FIXTURE["weak_mode_mask"])
            @test product.diagnostics.current_total_flux ≈
                T(GAIN_SENSING_FIXTURE["frames"]["current"]["total_flux"]) rtol=gain_tolerance

            before = (copy(Array(product.optical_gains)),
                copy(Array(product.current_sensitivities)),
                copy(Array(product.weak_mode_mask)), product.diagnostics)
            @test_throws ArgumentError process!(product, workspace, plan,
                OG.GainSensingInputs(to_device(fill(T(NaN), size(current)))))
            @test_throws ArgumentError process!(product, workspace, plan,
                OG.GainSensingInputs(to_device(zeros(T, size(current)))))
            @test (Array(product.optical_gains),
                Array(product.current_sensitivities),
                Array(product.weak_mode_mask), product.diagnostics) == before

            offset_sensitivities = view(vec(plan.basis_products),
                2:(plan.mode_count + 1))
            aliased_product = OG.GainSensingProduct(product.optical_gains,
                offset_sensitivities, product.weak_mode_mask, product.diagnostics)
            basis_before = Array(plan.basis_products)
            @test_throws ArgumentError process!(aliased_product, workspace,
                plan, input)
            @test Array(plan.basis_products) == basis_before

            negative_gain_probe = zeros(T, size(current))
            negative_gain_probe[1, 1] = one(T)
            process!(product, workspace, plan,
                OG.GainSensingInputs(to_device(negative_gain_probe)))
            @test Array(product.optical_gains) ≈ T[-4.147768508503635,
                -9.572789888214944, 1] rtol=gain_tolerance
            @test Array(product.weak_mode_mask) == [false, false, true]
        end
    end
end


function test_accelerator_gain_sensing_parity(backend, to_device, device_array_type)
    OG = AdaptiveOpticsCalibration.OpticalGains
    @testset "gain sensing 48×48 CPU and device parity" begin
        for T in (Float32, Float64)
            mask, basis, reference, current = gain_sensing_extended_inputs(T)
            specification = OG.GainSensingSpecification(mask, basis, reference)
            method = OG.GainSensing(T(1e-9))
            cpu_plan = prepare(method, specification)
            @test !any(cpu_plan.weak_mode_mask)
            cpu_product = process(cpu_plan, OG.GainSensingInputs(current))
            gpu_plan = prepare(method,
                KernelExecution(specification, backend; workgroup_size=64))
            gpu_product = allocate_result(gpu_plan)
            gpu_workspace = allocate_workspace(gpu_plan)
            gpu_inputs = OG.GainSensingInputs(to_device(current))
            @test (@inferred process!(gpu_product, gpu_workspace,
                gpu_plan, gpu_inputs)) === gpu_product
            @test gpu_plan.basis_products isa device_array_type
            @test gpu_product.optical_gains isa device_array_type
            @test gpu_product.current_sensitivities isa device_array_type
            @test gpu_product.weak_mode_mask isa device_array_type
            rtol = T === Float32 ? 5f-3 : 2e-8
            ir_atol = T === Float32 ? 2f-6 : 1e-10
            sensitivity_atol = T === Float32 ? 1f-9 : 1e-13
            @test Array(gpu_plan.reference_impulse_response) ≈
                cpu_plan.reference_impulse_response rtol=rtol atol=ir_atol
            @test Array(gpu_plan.reference_sensitivities) ≈
                cpu_plan.reference_sensitivities rtol=rtol atol=sensitivity_atol
            @test Array(gpu_product.current_sensitivities) ≈
                cpu_product.current_sensitivities rtol=rtol atol=sensitivity_atol
            @test Array(gpu_product.optical_gains) ≈
                cpu_product.optical_gains rtol=rtol
            @test Array(gpu_product.weak_mode_mask) == cpu_product.weak_mode_mask
            @test gpu_product.diagnostics.current_total_flux ≈
                cpu_product.diagnostics.current_total_flux rtol=rtol
        end
    end
end


function test_accelerator_structured_interaction_matrices(backend, to_device, device_array_type)
    IM = AdaptiveOpticsCalibration.InteractionMatrices
    @testset "structured interaction matrices" begin
        specification = IM.ResponseSpecification(2)
        fixtures = (
            (
                IM.ZonalPushPull(Float32[0.5, 2, 1.5]),
                Float32[
                    10.5 -6.75
                    9.5 -7.25
                    14 -11
                    6 -3
                    8.5 -1
                    11.5 -13
                ],
                Float32[1 2 -1; 0.5 -2 4],
            ),
            (
                IM.HadamardPushPull(Float32[0.25, 2, 0.5]),
                Float32[
                    95.25 -22.875
                    104.75 -37.125
                    92.75 -21.125
                    107.25 -38.875
                    105.25 -36.875
                    94.75 -23.125
                    106.75 -39.125
                    93.25 -20.875
                ],
                Float32[1 -3 2; 0.5 4 -2],
            ),
        )
        for (method, host_responses, expected) in fixtures
            plan = prepare(
                method,
                KernelExecution(specification, backend; workgroup_size=64),
            )
            result = allocate_result(plan)
            workspace = allocate_workspace(plan)
            responses = to_device(host_responses)
            process!(result, workspace, plan, responses)
            @test result.matrix isa device_array_type
            @test Array(result.matrix) ≈ expected atol=2f-6
            @test plan.reciprocal_scales isa device_array_type
            @test workspace.invalid_first isa device_array_type
            @test workspace.invalid_second isa device_array_type

            fill!(result.matrix, 17f0)
            invalid_host = copy(host_responses)
            invalid_host[begin] = Float32(NaN)
            invalid_host[3, 1] = floatmax(Float32)
            invalid_host[4, 1] = -floatmax(Float32)
            invalid = to_device(invalid_host)
            @test process!(result, workspace, plan, invalid) === result
            @test result.status === IM.StructuredNonFiniteResponse
            @test Array(result.matrix) == fill(17f0, size(expected))
            @test process!(result.matrix, workspace, plan, invalid) ===
                IM.StructuredNonFiniteResponse
            @test process!(result, workspace, plan, responses) === result
            @test result.status === IM.StructuredValid
            @test Array(result.matrix) ≈ expected atol=2f-6
        end
        method = IM.ZonalPushPull(Float32[2])
        specification = IM.ResponseSpecification(1)
        plan = prepare(
            method,
            KernelExecution(specification, backend; workgroup_size=64),
        )
        result = allocate_result(plan)
        workspace = allocate_workspace(plan)
        responses = to_device(reshape(Float32[5, 1], 2, 1))
        process!(result, workspace, plan, responses)
        @test Array(result.matrix) == ones(Float32, 1, 1)
        @test length(workspace.invalid_first) == 1
    end
end


function test_accelerator_probe_bases(backend, to_device, device_array_type)
    PB = AdaptiveOpticsCalibration.ProbeBases
    @testset "probe command bases" begin
        fixtures = (
            (
                PB.ModalPushPull(Float32[0.5 -1 0; -0.25 0.5 2]),
                Float32[
                    0.5 -1 0
                    -0.5 1 -0.0
                    -0.25 0.5 2
                    0.25 -0.5 -2
                ],
            ),
            (PB.ZonalPushPull(Float32[1, 2]), Float32[1 0; -1 0; 0 2; 0 -2]),
            (
                PB.HadamardPushPull(Float32[1, 2, 3]),
                Float32[
                    1 2 3
                    -1 -2 -3
                    -1 2 -3
                    1 -2 3
                    1 -2 -3
                    -1 2 3
                    -1 -2 3
                    1 2 -3
                ],
            ),
        )
        for (method, expected) in fixtures
            plan = prepare(
                method,
                KernelExecution(nothing, backend; workgroup_size=64),
            )
            result = allocate_result(plan)
            process!(result, allocate_workspace(plan), plan, nothing)
            @test result.commands isa device_array_type
            @test Array(result.commands) == expected
        end
    end
end


function test_accelerator_interaction_matrix_stability(backend, to_device, device_array_type)
    DG = AdaptiveOpticsCalibration.Diagnostics
    @testset "interaction-matrix stability" begin
        first_window = to_device(Matrix{Float32}(I, 2, 2))
        second_window = to_device(2f0 .* Matrix{Float32}(I, 2, 2))
        input = DG.InteractionMatrixPair(first_window, second_window)
        method = DG.InteractionMatrixStability(0.1f0, 0.5f0, 0.5f0)
        plan = prepare(
            method,
            KernelExecution(input, backend; workgroup_size=64),
        )
        result = allocate_result(plan)
        workspace = allocate_workspace(plan)
        process!(result, workspace, plan, input)
        @test workspace.first_column_squares isa device_array_type
        @test result.status === DG.ExcessiveRelativeColumnDifference
        @test result.maximum_relative_column_difference ≈ 2f0 / 3f0 atol=2f-6
        @test result.relative_frobenius_difference ≈ 2f0 / 3f0 atol=2f-6
    end
end


include("lift_fallback.jl")

function test_accelerator_lift(backend, to_device, device_array_type)
    test_accelerator_lift_fallback(backend, to_device, device_array_type)
    PR = AdaptiveOpticsCalibration.PhaseRetrieval
    @testset "LiFT phase retrieval" begin
        reference_rate = Float32[10, 12, 8, 15]
        jacobian = Float32[1 0; 0 1; 1 1; 2 -1]
        truth = Float32[0.1, -0.2]
        observation_rate = reference_rate + jacobian * truth
        model = AcceleratorLinearLiFTModel(
            to_device(reference_rate),
            to_device(jacobian),
        )
        for (domain, observation) in (
            (PR.LiFTPhotonRate(), observation_rate),
            (PR.LiFTExpectedCounts(0.5f0; quantum_efficiency=0.8f0),
                observation_rate .* 0.5f0 .* 0.8f0),
            (PR.LiFTNormalizedIntensity(20f0), observation_rate ./ 20f0),
        ), jacobian_method in (
            PR.LiFTAnalyticJacobian(),
            PR.LiFTNumericalJacobian(1f-3),
        )
            method = PR.LiFT(
                iterations=2,
                jacobian_method=jacobian_method,
                solve_mode=PR.LiFTSolveAuto(),
                weighting=PR.LiFTReadNoiseWeighting(),
                model_scaling=PR.LiFTPhysicalRatePreservation(),
                check_convergence=false,
            )
            specification = PR.LiFTSpecification(
                model,
                domain;
                read_noise_std=0.1f0,
            )
            plan = prepare(method, KernelExecution(
                specification,
                backend;
                workgroup_size=64,
            ))
            result = allocate_result(plan)
            workspace = allocate_workspace(plan)
            inputs = PR.LiFTInputs(to_device(observation))
            @test (@inferred process!(result, workspace, plan, inputs)) ===
                result
            @test result.coefficients isa device_array_type
            @test Array(result.coefficients) ≈ truth atol=4f-4
            @test !result.diagnostics.used_qr
            @test !result.diagnostics.used_fallback
            @test (@inferred process!(result, workspace, plan, inputs)) ===
                result
            @test Array(result.coefficients) ≈ truth atol=4f-4
            @test Array(inputs.observation) == observation
        end


        host_model = AcceleratorLinearLiFTModel(reference_rate, jacobian)
        for (host_weighting, device_weighting) in (
            (PR.LiFTReadNoiseWeighting(), PR.LiFTReadNoiseWeighting()),
            (PR.LiFTInitialModelWeighting(), PR.LiFTInitialModelWeighting()),
            (PR.LiFTIterativeModelWeighting(), PR.LiFTIterativeModelWeighting()),
            (
                PR.LiFTVarianceMapWeighting(fill(0.25f0, 4)),
                PR.LiFTVarianceMapWeighting(to_device(fill(0.25f0, 4))),
            ),
        )
            common = (
                iterations=2,
                solve_mode=PR.LiFTSolveNormalEquations(),
                model_scaling=PR.LiFTPhysicalRatePreservation(),
                check_convergence=false,
            )
            host_plan = prepare(
                PR.LiFT(; common..., weighting=host_weighting),
                PR.LiFTSpecification(
                    host_model,
                    PR.LiFTPhotonRate();
                    read_noise_std=0.1f0,
                ),
            )
            device_plan = prepare(
                PR.LiFT(; common..., weighting=device_weighting),
                KernelExecution(
                    PR.LiFTSpecification(
                        model,
                        PR.LiFTPhotonRate();
                        read_noise_std=0.1f0,
                    ),
                    backend;
                    workgroup_size=64,
                ),
            )
            expected = process(host_plan, PR.LiFTInputs(observation_rate))
            actual = process(
                device_plan,
                PR.LiFTInputs(to_device(observation_rate)),
            )
            @test Array(actual.coefficients) ≈ expected.coefficients atol=2f-5
            @test actual.diagnostics.iterations == expected.diagnostics.iterations
        end


        for scaling in (
            PR.LiFTTotalRateMatching(),
            PR.LiFTPeakRateMatching(),
            PR.LiFTPhysicalRatePreservation(),
        ), damping in (
            PR.LiFTDampingNone(),
            PR.LiFTLevenbergMarquardt(
                lambda0=1f-6,
                growth=10f0,
                condition_rtol=1f-3,
            ),
            PR.LiFTAdaptiveLevenbergMarquardt(
                lambda0=1f-6,
                growth=10f0,
                shrink=2f0,
                min_lambda=1f-10,
                condition_rtol=1f-3,
            ),
        )
            method = PR.LiFT(
                iterations=2,
                solve_mode=PR.LiFTSolveNormalEquations(),
                damping=damping,
                weighting=PR.LiFTReadNoiseWeighting(),
                model_scaling=scaling,
                check_convergence=false,
            )
            host_plan = prepare(method, PR.LiFTSpecification(
                host_model,
                PR.LiFTPhotonRate();
                read_noise_std=0.1f0,
            ))
            device_plan = prepare(method, KernelExecution(
                PR.LiFTSpecification(
                    model,
                    PR.LiFTPhotonRate();
                    read_noise_std=0.1f0,
                ),
                backend;
                workgroup_size=64,
            ))
            expected = process(host_plan, PR.LiFTInputs(observation_rate))
            actual = process(
                device_plan,
                PR.LiFTInputs(to_device(observation_rate)),
            )
            @test Array(actual.coefficients) ≈ expected.coefficients atol=3f-4
            @test actual.diagnostics.used_fallback ==
                expected.diagnostics.used_fallback
        end


        ordered_jacobian = Float32[
            1 0 2
            0 1 1
            2 1 0
            1 2 1
        ]
        ordered_reference = fill(20f0, 4)
        ordered_truth = Float32[0.5, 0.25, -0.75]
        ordered_observation = ordered_reference +
            ordered_jacobian * ordered_truth
        ordered_model = AcceleratorLinearLiFTModel(
            to_device(ordered_reference),
            to_device(ordered_jacobian),
        )
        ordered_plan = prepare(
            PR.LiFT(
                iterations=1,
                mode_indices=(3, 1),
                solve_mode=PR.LiFTSolveNormalEquations(),
                weighting=PR.LiFTReadNoiseWeighting(),
                model_scaling=PR.LiFTPhysicalRatePreservation(),
                check_convergence=false,
            ),
            KernelExecution(
                PR.LiFTSpecification(
                    ordered_model,
                    PR.LiFTPhotonRate();
                    read_noise_std=1f0,
                ),
                backend;
                workgroup_size=64,
            ),
        )
        ordered_result = process(
            ordered_plan,
            PR.LiFTInputs(
                to_device(ordered_observation);
                initial_coefficients=to_device(Float32[0, 0.25, 0]),
            ),
        )
        @test Array(ordered_result.coefficients) ≈
            ordered_truth[[3, 1]] atol=2f-5


        dependent_model = AcceleratorLinearLiFTModel(
            to_device(fill(10f0, 4)),
            to_device(Float32[1 1; 2 2; 3 3; 4 4]),
        )
        dependent_plan = prepare(
            PR.LiFT(
                iterations=1,
                solve_mode=PR.LiFTSolveNormalEquations(),
                weighting=PR.LiFTReadNoiseWeighting(),
                model_scaling=PR.LiFTPhysicalRatePreservation(),
                check_convergence=false,
            ),
            KernelExecution(
                PR.LiFTSpecification(
                    dependent_model,
                    PR.LiFTPhotonRate();
                    read_noise_std=1f0,
                ),
                backend;
                workgroup_size=64,
            ),
        )
        dependent_result = process(
            dependent_plan,
            PR.LiFTInputs(to_device(fill(11f0, 4))),
        )
        # The accelerator path executes the explicitly selected normal
        # equations directly. A positive diagonal load makes this
        # rank-deficient system solvable, so it must not report the CPU
        # Auto/QR-to-normal fallback diagnostic.
        @test !dependent_result.diagnostics.used_fallback
        @test all(isfinite, Array(dependent_result.coefficients))


        large_reference = ones(Float32, 2)
        large_jacobian = ones(Float32, 2, 1)
        large_observation = fill(1f20, 2)
        large_host_model = AcceleratorLinearLiFTModel(
            large_reference,
            large_jacobian,
        )
        large_device_model = AcceleratorLinearLiFTModel(
            to_device(large_reference),
            to_device(large_jacobian),
        )
        large_method = PR.LiFT(
            iterations=1,
            solve_mode=PR.LiFTSolveNormalEquations(),
            weighting=PR.LiFTReadNoiseWeighting(),
            model_scaling=PR.LiFTPhysicalRatePreservation(),
            check_convergence=false,
        )
        large_expected = process(
            prepare(
                large_method,
                PR.LiFTSpecification(
                    large_host_model,
                    PR.LiFTPhotonRate();
                    read_noise_std=1f0,
                ),
            ),
            PR.LiFTInputs(large_observation),
        )
        large_actual = process(
            prepare(
                large_method,
                KernelExecution(
                    PR.LiFTSpecification(
                        large_device_model,
                        PR.LiFTPhotonRate();
                        read_noise_std=1f0,
                    ),
                    backend,
                ),
            ),
            PR.LiFTInputs(to_device(large_observation)),
        )
        @test isfinite(large_actual.diagnostics.residual_norm)
        @test large_actual.diagnostics.residual_norm ≈
            large_expected.diagnostics.residual_norm rtol=2f-6
        @test Array(large_actual.coefficients) ≈
            large_expected.coefficients rtol=2f-6


        atomic_plan = prepare(
            PR.LiFT(
                iterations=1,
                solve_mode=PR.LiFTSolveNormalEquations(),
                model_scaling=PR.LiFTPhysicalRatePreservation(),
            ),
            KernelExecution(
                PR.LiFTSpecification(model, PR.LiFTPhotonRate()),
                backend;
                workgroup_size=64,
            ),
        )
        atomic_result = allocate_result(atomic_plan)
        fill!(atomic_result.coefficients, 7f0)
        invalid_observation = copy(observation_rate)
        invalid_observation[1] = Float32(NaN)
        @test_throws ArgumentError process!(
            atomic_result,
            allocate_workspace(atomic_plan),
            atomic_plan,
            PR.LiFTInputs(to_device(invalid_observation)),
        )
        @test Array(atomic_result.coefficients) == fill(7f0, 2)
        @test_throws ArgumentError process!(
            atomic_result,
            allocate_workspace(atomic_plan),
            atomic_plan,
            PR.LiFTInputs(observation_rate),
        )
        @test Array(atomic_result.coefficients) == fill(7f0, 2)
        atomic_workspace = allocate_workspace(atomic_plan)
        aliased_reduction_workspace = PR.LiFTKernelWorkspace(
            atomic_workspace.inverse,
            atomic_workspace.reduction_first,
            atomic_workspace.reduction_first,
            atomic_workspace.invalid_first,
            atomic_workspace.invalid_second,
            atomic_workspace.host_scalar,
            atomic_workspace.host_invalid,
        )
        @test_throws ArgumentError process!(
            atomic_result,
            aliased_reduction_workspace,
            atomic_plan,
            PR.LiFTInputs(to_device(observation_rate)),
        )
        device_host_scalar_workspace = PR.LiFTKernelWorkspace(
            atomic_workspace.inverse,
            atomic_workspace.reduction_first,
            atomic_workspace.reduction_second,
            atomic_workspace.invalid_first,
            atomic_workspace.invalid_second,
            to_device(zeros(Float32, 1)),
            atomic_workspace.host_invalid,
        )
        @test_throws ArgumentError process!(
            atomic_result,
            device_host_scalar_workspace,
            atomic_plan,
            PR.LiFTInputs(to_device(observation_rate)),
        )
        @test Array(atomic_result.coefficients) == fill(7f0, 2)
        invalid_variance = fill(0.25f0, 4)
        invalid_variance[2] = Float32(NaN)
        @test_throws ArgumentError prepare(
            PR.LiFT(weighting=PR.LiFTVarianceMapWeighting(
                to_device(invalid_variance),
            )),
            KernelExecution(
                PR.LiFTSpecification(model, PR.LiFTPhotonRate()),
                backend,
            ),
        )
        @test_throws ArgumentError prepare(
            PR.LiFT(solve_mode=PR.LiFTSolveQR()),
            KernelExecution(
                PR.LiFTSpecification(model, PR.LiFTPhotonRate()),
                backend,
            ),
        )
    end
end


include("srt_cpu_inputs.jl")

const ACCELERATOR_TEST_DOMAINS = (
    "tomography", "gain", "structured", "probes", "stability", "lift",
)

function run_accelerator_tests(
    backend, to_device, device_array_type; phase_retrieval=false, domain=nothing,
)
    if domain === nothing
        test_srt_cpu_inputs(to_device)
        test_accelerator_tomography_covariance(backend, to_device, device_array_type)
        test_accelerator_covariance_reconstructor(backend, to_device, device_array_type)
        test_accelerator_gain_sensing_frozen(backend, to_device, device_array_type)
        test_accelerator_gain_sensing_parity(backend, to_device, device_array_type)
        test_accelerator_structured_interaction_matrices(backend, to_device, device_array_type)
        test_accelerator_probe_bases(backend, to_device, device_array_type)
        test_accelerator_interaction_matrix_stability(backend, to_device, device_array_type)
        phase_retrieval && test_accelerator_lift(backend, to_device, device_array_type)
        return nothing
    end
    domain in ACCELERATOR_TEST_DOMAINS || error(
        "Unknown accelerator test domain '$domain'. Choose one of: " *
        join(ACCELERATOR_TEST_DOMAINS, ", "))
    if domain == "tomography"
        test_accelerator_tomography_covariance(backend, to_device, device_array_type)
        test_accelerator_covariance_reconstructor(backend, to_device, device_array_type)
    elseif domain == "gain"
        test_accelerator_gain_sensing_frozen(backend, to_device, device_array_type)
        test_accelerator_gain_sensing_parity(backend, to_device, device_array_type)
    elseif domain == "structured"
        test_accelerator_structured_interaction_matrices(backend, to_device, device_array_type)
    elseif domain == "probes"
        test_accelerator_probe_bases(backend, to_device, device_array_type)
    elseif domain == "stability"
        test_accelerator_interaction_matrix_stability(backend, to_device, device_array_type)
    elseif domain == "lift" && phase_retrieval
        test_accelerator_lift(backend, to_device, device_array_type)
    end
    return nothing
end
