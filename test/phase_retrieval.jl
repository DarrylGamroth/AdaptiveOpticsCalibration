module PhaseRetrievalTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.PhaseRetrieval
import AdaptiveOpticsCalibration.PhaseRetrieval: predict_photon_rate!
using LinearAlgebra
using Test
using TOML

const AOS_LIFT_FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_lift_inverse.toml"))

struct NonOneBasedMutableVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end
Base.size(array::NonOneBasedMutableVector) = size(array.parent)
Base.axes(array::NonOneBasedMutableVector) = (0:(length(array.parent) - 1),)
Base.IndexStyle(::Type{<:NonOneBasedMutableVector}) = IndexCartesian()
Base.getindex(array::NonOneBasedMutableVector, index::Int) = array.parent[index + 1]
Base.setindex!(array::NonOneBasedMutableVector, value, index::Int) =
    setindex!(array.parent, value, index + 1)
Base.similar(array::NonOneBasedMutableVector) =
    NonOneBasedMutableVector(similar(array.parent))

"""Frozen affine evaluation of a recorded physical LiFT model and Jacobian."""
struct FrozenLinearizedLiFTModel{
    T<:AbstractFloat,
    R<:AbstractArray{T},
    J<:AbstractMatrix{T},
} <: AbstractLiFTForwardModel{T}
    reference_rate::R
    jacobian::J
end

PhaseRetrieval.coefficient_count(model::FrozenLinearizedLiFTModel) =
    size(model.jacobian, 2)
PhaseRetrieval.observation_axes(model::FrozenLinearizedLiFTModel) =
    axes(model.reference_rate)
PhaseRetrieval.allocate_model_workspace(::FrozenLinearizedLiFTModel) = nothing
PhaseRetrieval.allocate_photon_rate(model::FrozenLinearizedLiFTModel) =
    similar(model.reference_rate)
function PhaseRetrieval.predict_photon_rate!(out, model::FrozenLinearizedLiFTModel,
    ::Nothing, coefficients)
    row = firstindex(model.jacobian, 1)
    @inbounds for index in eachindex(out, model.reference_rate)
        value = model.reference_rate[index]
        for column in axes(model.jacobian, 2)
            value += model.jacobian[row, column] * coefficients[column]
        end
        out[index] = value
        row += 1
    end
    return out
end
function PhaseRetrieval.analytic_photon_rate_jacobian!(jacobian,
    model::FrozenLinearizedLiFTModel, ::Nothing, ::AbstractVector, mode_indices)
    @inbounds for column in axes(jacobian, 2), row in axes(jacobian, 1)
        jacobian[row, column] = model.jacobian[row, mode_indices[column]]
    end
    return jacobian
end

"""Small typed photon-rate model used to test the package-neutral LiFT seam."""
struct PolynomialModel{
    T<:AbstractFloat,
    B<:AbstractMatrix{T},
    O<:AbstractArray{T},
} <: AbstractLiFTForwardModel{T}
    basis::B
    offset::O
end

PhaseRetrieval.coefficient_count(model::PolynomialModel) = size(model.basis, 2)
PhaseRetrieval.observation_axes(model::PolynomialModel) = axes(model.offset)
PhaseRetrieval.allocate_model_workspace(::PolynomialModel) = nothing
PhaseRetrieval.allocate_photon_rate(model::PolynomialModel{T}) where {T} = similar(model.offset)
function PhaseRetrieval.predict_photon_rate!(out, model::PolynomialModel, ::Nothing, coefficients)
    T = eltype(out)
    row = 1
    for index in eachindex(IndexLinear(), out, model.offset)
        linear = model.offset[index]
        @inbounds for coefficient_index in eachindex(coefficients)
            linear += model.basis[row, coefficient_index] *
                coefficients[coefficient_index]
        end
        out[index] = linear^2 + one(T)
        row += 1
    end
    return out
end
function PhaseRetrieval.analytic_photon_rate_jacobian!(jacobian, model::PolynomialModel,
    ::Nothing, coefficients, mode_indices)
    for column in axes(jacobian, 2), row in axes(jacobian, 1)
        linear = model.offset[row] + sum(model.basis[row, index] * coefficients[index] for index in axes(model.basis, 2))
        jacobian[row, column] = 2 * linear * model.basis[row, mode_indices[column]]
    end
    return jacobian
end

function model_and_observation(::Type{T}=Float64) where {T<:AbstractFloat}
    basis = T[1 0; 0 1; 1 1; 2 -1]
    model = PolynomialModel(basis, T[2, 3, 1, 4])
    truth = T[0.1, -0.2]
    observation = similar(model.offset)
    predict_photon_rate!(observation, model, nothing, truth)
    return model, truth, observation
end

function replace_lift_workspace(
    workspace::LiFTWorkspace;
    residual=workspace.residual,
    sqrt_weights=workspace.sqrt_weights,
    normal=workspace.normal,
    coefficients=workspace.coefficients,
)
    return LiFTWorkspace(
        workspace.model_workspace,
        workspace.observation_rate,
        workspace.model_rate,
        workspace.plus_rate,
        workspace.minus_rate,
        residual,
        sqrt_weights,
        workspace.jacobian,
        workspace.jacobian_backup,
        normal,
        workspace.factor,
        workspace.rhs,
        coefficients,
        workspace.fallback_work,
    )
end

@testset "LiFT frozen AOS migration oracle" begin
    @test AOS_LIFT_FIXTURE["schema"] == "test.adaptive-optics-sim/lift-inverse/1"
    @test AOS_LIFT_FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test AOS_LIFT_FIXTURE["source_revision"] ==
        "df044b6d44e3de6a4cc1aedc09cecb03fe194499"
    @test AOS_LIFT_FIXTURE["numeric_type"] == "Float64"
    @test AOS_LIFT_FIXTURE["coefficient_unit"] == "metre OPD"
    @test AOS_LIFT_FIXTURE["basis_order"] == "basis[:,:,mode_index], one-based"
    @test AOS_LIFT_FIXTURE["observation_linear_order"] == "Julia column-major"

    resolution = AOS_LIFT_FIXTURE["focal_resolution"]
    basis = reshape(AOS_LIFT_FIXTURE["basis"],
        AOS_LIFT_FIXTURE["pupil_resolution"], AOS_LIFT_FIXTURE["pupil_resolution"], 2)
    observation = reshape(AOS_LIFT_FIXTURE["observation_photon_rate"], resolution, resolution)
    @test size(basis) == (4, 4, 2)
    @test size(observation) == (4, 4)
    @test length(AOS_LIFT_FIXTURE["truth_coefficients"]) == size(basis, 3)
    for mode in ("analytic", "numerical")
        reference = AOS_LIFT_FIXTURE[mode]
        jacobian = reshape(reference["initial_jacobian"], length(observation), 2)
        @test size(jacobian) == (16, 2)
        @test all(isfinite, jacobian)
        @test reference["expected_coefficients"] ≈ AOS_LIFT_FIXTURE["truth_coefficients"] atol=3e-20
        @test isfinite(reference["residual_norm"])
        @test isfinite(reference["weighted_residual_norm"])
        @test !reference["used_qr"]
        @test !reference["used_fallback"]
    end
    @test AOS_LIFT_FIXTURE["numerical"]["difference_step_m"] == 1e-9

    analytic = AOS_LIFT_FIXTURE["analytic"]
    initial_rate = reshape(AOS_LIFT_FIXTURE["initial_model_photon_rate"],
        resolution, resolution)
    initial_jacobian = reshape(analytic["initial_jacobian"], length(observation), 2)
    frozen_model = FrozenLinearizedLiFTModel(initial_rate, initial_jacobian)
    frozen_plan = prepare(
        LiFT(
            iterations=1,
            solve_mode=LiFTSolveNormalEquations(),
            damping=LiFTDampingNone(),
            weighting=LiFTReadNoiseWeighting(),
            model_scaling=LiFTPhysicalRatePreservation(),
            check_convergence=false,
        ),
        LiFTSpecification(frozen_model, LiFTPhotonRate()),
    )
    frozen_result = process(frozen_plan, LiFTInputs(observation))
    @test frozen_result.coefficients ≈ analytic["one_iteration_expected_coefficients"] rtol=2e-15
    @test frozen_result.diagnostics.residual_norm ≈ analytic["one_iteration_residual_norm"] rtol=2e-15
    @test frozen_result.diagnostics.weighted_residual_norm ≈
        analytic["one_iteration_weighted_residual_norm"] rtol=2e-15
    @test frozen_result.diagnostics.update_norm ≈ analytic["one_iteration_update_norm"] rtol=2e-15
    @test frozen_result.diagnostics.condition_ratio ≈
        analytic["one_iteration_condition_ratio"] rtol=2e-15
    @test frozen_result.diagnostics.regularization == analytic["one_iteration_regularization"]
    @test frozen_result.diagnostics.used_qr == analytic["one_iteration_used_qr"]
    @test frozen_result.diagnostics.used_fallback == analytic["one_iteration_used_fallback"]
end

@testset "LiFT direct equations and typed forward seam" begin
    model, truth, observation = model_and_observation()
    specification = LiFTSpecification(model, LiFTPhotonRate(); read_noise_std=0.2)
    plan = @inferred prepare(LiFT(iterations=8, solve_mode=LiFTSolveNormalEquations(),
        damping=LiFTLevenbergMarquardt(), weighting=LiFTReadNoiseWeighting(),
        model_scaling=LiFTPhysicalRatePreservation()), specification)
    result, workspace = allocate_result(plan), allocate_workspace(plan)
    @test all(isconcretetype, fieldtypes(typeof(plan)))
    @test all(isconcretetype, fieldtypes(typeof(workspace)))
    @test all(isconcretetype, fieldtypes(typeof(result)))
    @test @inferred(process!(result, workspace, plan, LiFTInputs(observation))) === result
    @test lift_coefficients(result) ≈ truth atol=1e-8
    @test result.diagnostics.iterations <= 8
    @test !result.diagnostics.used_qr
    @test lift_diagnostics(result) === result.diagnostics
end

@testset "LiFT observation, weighting, and Jacobian modes" begin
    model, truth, rate = model_and_observation()
    counts = rate .* 2 .* 0.5
    normalized = rate ./ 10
    for (domain, observation) in ((LiFTPhotonRate(), rate),
        (LiFTExpectedCounts(2; quantum_efficiency=0.5), counts),
        (LiFTNormalizedIntensity(10), normalized))
        for jacobian in (LiFTAnalyticJacobian(), LiFTNumericalJacobian(1e-6))
            for weighting in (LiFTInitialModelWeighting(), LiFTIterativeModelWeighting(),
                LiFTReadNoiseWeighting(), LiFTVarianceMapWeighting(fill(eltype(rate)(0.1), length(rate))))
                plan = prepare(LiFT(iterations=8, jacobian_method=jacobian,
                    solve_mode=LiFTSolveNormalEquations(), weighting=weighting,
                    model_scaling=LiFTPhysicalRatePreservation()),
                    LiFTSpecification(model, domain; read_noise_std=0.1))
                result = process(plan, LiFTInputs(observation))
                @test result.coefficients ≈ truth atol=3e-5
            end
        end
    end
end

@testset "LiFT Float32, reuse, validation, and allocation" begin
    model, truth, observation = model_and_observation(Float32)
    plan = prepare(LiFT(iterations=5, jacobian_method=LiFTAnalyticJacobian(),
        solve_mode=LiFTSolveNormalEquations(), weighting=LiFTReadNoiseWeighting(),
        model_scaling=LiFTPhysicalRatePreservation(), check_convergence=false),
        LiFTSpecification(model, LiFTPhotonRate(); read_noise_std=Float32(0.1)))
    result, workspace, inputs = allocate_result(plan), allocate_workspace(plan), LiFTInputs(observation)
    @test @inferred(process!(result, workspace, plan, inputs)) === result
    first = copy(result.coefficients)
    process!(result, workspace, plan, inputs)
    @test result.coefficients == first
    @test eltype(result.coefficients) === Float32
    @test @allocated(process!(result, workspace, plan, inputs)) == 0
    observation_storage = zeros(Float32, 7)
    noncontiguous_observation = @view observation_storage[1:2:7]
    copyto!(noncontiguous_observation, observation)
    view_result = process(plan, LiFTInputs(noncontiguous_observation))
    @test view_result.coefficients ≈ truth atol=3f-5

    matrix_model = PolynomialModel(model.basis, reshape(copy(model.offset), 2, 2))
    matrix_observation = similar(matrix_model.offset)
    predict_photon_rate!(matrix_observation, matrix_model, nothing, truth)
    matrix_plan = prepare(
        LiFT(
            iterations=5,
            solve_mode=LiFTSolveNormalEquations(),
            weighting=LiFTReadNoiseWeighting(),
            model_scaling=LiFTPhysicalRatePreservation(),
            check_convergence=false,
        ),
        LiFTSpecification(
            matrix_model,
            LiFTPhotonRate();
            read_noise_std=Float32(0.1),
        ),
    )
    @test process(matrix_plan, LiFTInputs(matrix_observation)).coefficients ≈
        truth atol=3f-5

    offset_model = FrozenLinearizedLiFTModel(
        NonOneBasedMutableVector(Float32[10, 20]),
        reshape(Float32[1, 2], 2, 1),
    )
    offset_truth = Float32[0.5]
    offset_observation = similar(offset_model.reference_rate)
    predict_photon_rate!(offset_observation, offset_model, nothing, offset_truth)
    for (jacobian_method, weighting) in (
        (
            LiFTAnalyticJacobian(),
            LiFTVarianceMapWeighting(NonOneBasedMutableVector(Float32[1, 4])),
        ),
        (LiFTNumericalJacobian(1f-2), LiFTIterativeModelWeighting()),
    )
        offset_plan = prepare(
            LiFT(
                iterations=1,
                jacobian_method=jacobian_method,
                solve_mode=LiFTSolveNormalEquations(),
                weighting=weighting,
                model_scaling=LiFTPhysicalRatePreservation(),
                check_convergence=false,
            ),
            LiFTSpecification(offset_model, LiFTPhotonRate()),
        )
        @test process(offset_plan, LiFTInputs(offset_observation)).coefficients ≈
            offset_truth atol=3f-4
    end
    before = copy(result.coefficients)
    @test_throws DimensionMismatch process!(result, workspace, plan, LiFTInputs(zeros(Float32, 3)))
    @test result.coefficients == before
    @test_throws ArgumentError prepare(LiFT(), LiFTSpecification(model, LiFTExpectedCounts(-1)))
    @test_throws ArgumentError LiFT(iterations=0)
end

@testset "LiFT solver policies, scaling, and exceptional numerical paths" begin
    model, truth, observation = model_and_observation()
    specification = LiFTSpecification(model, LiFTPhotonRate(); read_noise_std=0.1)
    analytic_plan = prepare(LiFT(jacobian_method=LiFTAnalyticJacobian(),
        solve_mode=LiFTSolveNormalEquations(), model_scaling=LiFTPhysicalRatePreservation()), specification)
    numerical_plan = prepare(LiFT(jacobian_method=LiFTNumericalJacobian(1e-6),
        solve_mode=LiFTSolveNormalEquations(), model_scaling=LiFTPhysicalRatePreservation()), specification)
    analytic_workspace, numerical_workspace = allocate_workspace(analytic_plan), allocate_workspace(numerical_plan)
    fill!(analytic_workspace.coefficients, 0)
    fill!(numerical_workspace.coefficients, 0)
    PhaseRetrieval._jacobian!(analytic_plan.jacobian_method, analytic_workspace.jacobian,
        analytic_workspace, analytic_plan)
    PhaseRetrieval._jacobian!(numerical_plan.jacobian_method, numerical_workspace.jacobian,
        numerical_workspace, numerical_plan)
    @test numerical_workspace.jacobian ≈ analytic_workspace.jacobian rtol=1e-6 atol=1e-6

    for (scaling, expected_scale, expected_objective) in (
        (LiFTTotalRateMatching(), 3.0, 18.0),
        (LiFTPeakRateMatching(), 3.0, 18.0),
        (LiFTPhysicalRatePreservation(), 1.0, 6.0),
    )
        rates = [2.0, 4.0]
        scale, objective = PhaseRetrieval._model_scale!(scaling, rates, [6.0, 12.0])
        @test scale == expected_scale
        @test objective == expected_objective
    end

    inputs = LiFTInputs(observation)
    for solve_mode in (LiFTSolveAuto(), LiFTSolveQR(), LiFTSolveNormalEquations())
        for damping in (LiFTDampingNone(), LiFTLevenbergMarquardt(),
            LiFTAdaptiveLevenbergMarquardt())
            plan = prepare(LiFT(iterations=4, solve_mode=solve_mode, damping=damping,
                weighting=LiFTReadNoiseWeighting(),
                model_scaling=LiFTPhysicalRatePreservation()), specification)
            result = process(plan, inputs)
            @test result.coefficients ≈ truth atol=1e-6
            @test isfinite(result.diagnostics.regularization)
        end
    end

    capped = process(prepare(LiFT(iterations=1, check_convergence=false,
        solve_mode=LiFTSolveNormalEquations(), model_scaling=LiFTPhysicalRatePreservation()),
        specification), inputs)
    @test capped.diagnostics.iterations == 1
    @test !capped.diagnostics.converged
    converged = process(prepare(LiFT(iterations=5, relative_update_tolerance=1e9,
        solve_mode=LiFTSolveNormalEquations(), model_scaling=LiFTPhysicalRatePreservation()),
        specification), inputs)
    @test converged.diagnostics.iterations == 1
    @test converged.diagnostics.converged

    variance = fill(0.1, length(observation))
    variance_plan = prepare(LiFT(weighting=LiFTVarianceMapWeighting(variance)), specification)
    variance[1] = 99
    @test variance_plan.weighting.variance[1] == 0.1

    weighted_model = FrozenLinearizedLiFTModel(fill(10.0, 3), ones(3, 1))
    weighted_observation = [11.0, 12.0, 14.0]
    variances = [1.0, 4.0, 9.0]
    weighted_plan = prepare(LiFT(
        iterations=1,
        solve_mode=LiFTSolveNormalEquations(),
        weighting=LiFTVarianceMapWeighting(variances),
        model_scaling=LiFTPhysicalRatePreservation(),
        check_convergence=false,
    ), LiFTSpecification(weighted_model, LiFTPhotonRate()))
    weighted_result = process(weighted_plan, LiFTInputs(weighted_observation))
    inverse_variances = inv.(variances)
    expected_weighted_update =
        sum(inverse_variances .* (weighted_observation .- 10)) / sum(inverse_variances)
    @test only(weighted_result.coefficients) ≈ expected_weighted_update rtol=1e-14

    ordered_model = FrozenLinearizedLiFTModel(fill(20.0, 4), [
        1.0 0.0 2.0
        0.0 1.0 1.0
        2.0 1.0 0.0
        1.0 2.0 1.0
    ])
    ordered_truth = [0.5, 0.25, -0.75]
    ordered_observation = similar(ordered_model.reference_rate)
    predict_photon_rate!(ordered_observation, ordered_model, nothing, ordered_truth)
    ordered_plan = prepare(LiFT(
        iterations=1,
        mode_indices=(3, 1),
        solve_mode=LiFTSolveNormalEquations(),
        weighting=LiFTReadNoiseWeighting(),
        model_scaling=LiFTPhysicalRatePreservation(),
        check_convergence=false,
    ), LiFTSpecification(ordered_model, LiFTPhotonRate(); read_noise_std=1.0))
    ordered_result = process(ordered_plan, LiFTInputs(
        ordered_observation; initial_coefficients=[0.0, ordered_truth[2], 0.0]))
    @test ordered_result.coefficients ≈ ordered_truth[[3, 1]] rtol=1e-14 atol=1e-14

    singular_model = PolynomialModel(zeros(4, 2), ones(4))
    singular_plan = prepare(LiFT(solve_mode=LiFTSolveQR(),
        model_scaling=LiFTPhysicalRatePreservation()),
        LiFTSpecification(singular_model, LiFTPhotonRate()))
    singular_result = process(singular_plan, LiFTInputs(ones(4)))
    @test singular_result.diagnostics.used_fallback
    @test all(isfinite, singular_result.coefficients)
    singular_workspace = allocate_workspace(singular_plan)
    fill!(singular_workspace.jacobian, 0)
    fill!(singular_workspace.normal, 0)
    fill!(singular_workspace.residual, 1)
    fill!(singular_workspace.rhs, 0)
    @test PhaseRetrieval._fallback_svd!(singular_workspace, singular_workspace.jacobian, 0.0) == 0.0
    @test singular_workspace.rhs == zeros(2)

    regularized_jacobian = [1.0 2.0; 0.5 -1.0; 2.0 0.25; -1.0 0.5]
    regularized_residual = [0.5, -1.0, 2.0, 0.25]
    copyto!(singular_workspace.jacobian, regularized_jacobian)
    copyto!(singular_workspace.residual, regularized_residual)
    λ = 0.25
    expected_regularized =
        (regularized_jacobian' * regularized_jacobian + λ * I) \
        (regularized_jacobian' * regularized_residual)
    @test PhaseRetrieval._fallback_svd!(
        singular_workspace, singular_workspace.jacobian, λ) == λ
    @test singular_workspace.rhs ≈ expected_regularized rtol=1e-14 atol=1e-14

    dependent_model = PolynomialModel([1.0 1.0; 2.0 2.0; 3.0 3.0; 4.0 4.0],
        ones(4))
    dependent_plan = prepare(LiFT(
        solve_mode=LiFTSolveQR(),
        model_scaling=LiFTPhysicalRatePreservation(),
    ), LiFTSpecification(dependent_model, LiFTPhotonRate()))
    dependent_result = process(dependent_plan, LiFTInputs(fill(4.0, 4)))
    @test dependent_result.diagnostics.used_fallback
    @test all(isfinite, dependent_result.coefficients)
end

@testset "LiFT invalid model and storage contracts are atomic" begin
    model, _, observation = model_and_observation()
    plan = prepare(LiFT(solve_mode=LiFTSolveNormalEquations()),
        LiFTSpecification(model, LiFTPhotonRate()))
    result, workspace = allocate_result(plan), allocate_workspace(plan)
    process!(result, workspace, plan, LiFTInputs(observation))
    before = copy(result.coefficients)
    @test_throws ArgumentError process!(result, workspace, plan, LiFTInputs(-observation))
    @test result.coefficients == before
    aliased_product_workspace = replace_lift_workspace(
        workspace;
        coefficients=result.coefficients,
    )
    @test_throws ArgumentError process!(
        result,
        aliased_product_workspace,
        plan,
        LiFTInputs(observation),
    )
    malformed_workspace = replace_lift_workspace(workspace; normal=zeros(1, 1))
    @test_throws DimensionMismatch process!(
        result,
        malformed_workspace,
        plan,
        LiFTInputs(observation),
    )
    internally_aliased_workspace = replace_lift_workspace(
        workspace;
        sqrt_weights=workspace.residual,
    )
    @test_throws ArgumentError process!(
        result,
        internally_aliased_workspace,
        plan,
        LiFTInputs(observation),
    )

    zero_based_workspace = replace_lift_workspace(
        workspace;
        coefficients=NonOneBasedMutableVector(workspace.coefficients),
    )
    @test_throws ArgumentError process!(
        result,
        zero_based_workspace,
        plan,
        LiFTInputs(observation),
    )

    variance_plan = prepare(
        LiFT(
            solve_mode=LiFTSolveNormalEquations(),
            weighting=LiFTVarianceMapWeighting(fill(0.25, length(observation))),
        ),
        LiFTSpecification(model, LiFTPhotonRate()),
    )
    variance_result = allocate_result(variance_plan)
    aliased_plan_result = LiFTProduct(
        @view(variance_plan.weighting.variance[1:length(variance_result.coefficients)]),
        variance_result.diagnostics,
    )
    @test_throws ArgumentError process!(
        aliased_plan_result,
        allocate_workspace(variance_plan),
        variance_plan,
        LiFTInputs(observation),
    )
    aliased_plan_workspace = replace_lift_workspace(
        allocate_workspace(variance_plan);
        residual=variance_plan.weighting.variance,
    )
    @test_throws ArgumentError process!(
        allocate_result(variance_plan),
        aliased_plan_workspace,
        variance_plan,
        LiFTInputs(observation),
    )

    underdetermined_model = PolynomialModel(ones(1, 2), ones(1))
    @test_throws DimensionMismatch prepare(
        LiFT(), LiFTSpecification(underdetermined_model, LiFTPhotonRate()))

    extreme_model = PolynomialModel(reshape([5e199], 1, 1), ones(1))
    extreme_plan = prepare(
        LiFT(
            iterations=1,
            solve_mode=LiFTSolveNormalEquations(),
            weighting=LiFTReadNoiseWeighting(),
            model_scaling=LiFTPhysicalRatePreservation(),
            check_convergence=false,
        ),
        LiFTSpecification(extreme_model, LiFTPhotonRate()),
    )
    extreme_result = allocate_result(extreme_plan)
    extreme_result.coefficients .= 7.0
    @test_throws ArgumentError process!(
        extreme_result,
        allocate_workspace(extreme_plan),
        extreme_plan,
        LiFTInputs([1e300]),
    )
    @test extreme_result.coefficients == [7.0]

    @test_throws ArgumentError prepare(
        LiFT(),
        LiFTSpecification(
            PolynomialModel(ones(1, 1), ones(1)),
            LiFTExpectedCounts(floatmin(Float64); quantum_efficiency=floatmin(Float64)),
        ),
    )
end

end
