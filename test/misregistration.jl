module MisregistrationTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.Misregistration
using LinearAlgebra
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_s6_misregistration_cpu.toml"))

struct NonOneBasedMisregistrationMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
end

Base.size(array::NonOneBasedMisregistrationMatrix) = size(array.parent)
Base.axes(array::NonOneBasedMisregistrationMatrix) = (
    0:(size(array.parent, 1) - 1),
    0:(size(array.parent, 2) - 1),
)
Base.IndexStyle(::Type{<:NonOneBasedMisregistrationMatrix}) = IndexCartesian()
Base.getindex(array::NonOneBasedMisregistrationMatrix, row::Int, column::Int) =
    array.parent[row + 1, column + 1]

function fixture_matrix(name)
    matrix = FIXTURE["matrices"][name]
    return reshape(Float64.(matrix["values"]), Tuple(Int.(matrix["shape"])))
end

function fixture_specification(sensitivity)
    field_order = Symbol.(FIXTURE["field_order"])
    field_units = [FIXTURE["field_units"][String(field)] for field in field_order]
    return MetaSensitivityEstimateSpecification(
        fixture_matrix("D0"),
        fixture_matrix("J_$(sensitivity)"),
        field_order,
        field_units,
    )
end

function misregistration_allocation(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

@testset "AOS S6 meta-sensitivity source fixture" begin
    @test FIXTURE["schema"] == "test.adaptive-optics-sim/misregistration-cpu/1"
    @test FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test FIXTURE["source_revision"] ==
        "7071c4eb61bf691ef845721fec141a38d0906873"
    @test FIXTURE["source_paths"] == [
        "src/calibration/misregistration_identification.jl",
        "src/calibration/ad_sensitivities.jl",
    ]
    @test FIXTURE["source_test_path"] ==
        "test/testsets/calibration_workflows.jl"
    @test FIXTURE["storage_order"] == "column-major"
    @test FIXTURE["wfs_response_unit"] == "m OPD"
    @test FIXTURE["matrices"]["D0"]["unit"] == "m OPD"
    @test FIXTURE["matrices"]["J_ad"]["unit"] == "m OPD per field unit"
    @test FIXTURE["matrices"]["J_finite_difference"]["unit"] ==
        "m OPD per field unit"
    @test FIXTURE["matrices"]["D_observed"]["unit"] == "m OPD"

    for sensitivity in ("ad", "finite_difference")
        specification = fixture_specification(sensitivity)
        plan = @inferred prepare(MetaSensitivityEstimate(), specification)
        result = allocate_result(plan)
        workspace = allocate_workspace(plan)
        input = MetaSensitivityEstimateInputs(fixture_matrix("D_observed"))
        @test @inferred(process!(result, workspace, plan, input)) === result

        inverse = FIXTURE["inverse"]
        diagnostics = FIXTURE["inverse_diagnostics"][sensitivity]
        @test plan.field_order == Tuple(Symbol.(FIXTURE["field_order"]))
        @test plan.field_units == Tuple([
            FIXTURE["field_units"][field] for field in FIXTURE["field_order"]
        ])
        @test plan.reference_response !== specification.reference_response
        @test plan.meta_sensitivity !== specification.meta_sensitivity
        @test !Base.mightalias(plan.reference_response,
            specification.reference_response)
        @test !Base.mightalias(plan.meta_sensitivity,
            specification.meta_sensitivity)
        @test !Base.mightalias(plan.inverse, plan.meta_sensitivity)
        @test nameof(typeof(plan.inverse_method)) == Symbol(inverse["method"])
        @test plan.inverse_method.rtol == inverse["rtol"]
        @test plan.inverse_method.atol == inverse["atol"]
        @test plan.inverse_method.n_trunc == inverse["n_trunc"]
        @test parameter_offsets(result) === result.parameter_offsets
        @test result.parameter_offsets ≈
            FIXTURE["unrounded_delta"][sensitivity]["values"] rtol=1e-12 atol=1e-15
        @test effective_rank(result) == diagnostics["effective_rank"]
        @test condition_number(result) ≈ diagnostics["condition_number"] rtol=1e-12
        @test misregistration_allocation(result, workspace, plan, input) == 0
    end
end

@testset "rank-deficient meta-sensitivity minimum-norm estimate" begin
    reference = zeros(2, 1)
    sensitivity = [1.0 1.0; 0.0 0.0]
    specification = MetaSensitivityEstimateSpecification(
        reference,
        sensitivity,
        [:first, :second],
        ["dimensionless", "dimensionless"],
    )
    plan = prepare(MetaSensitivityEstimate(), specification)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = MetaSensitivityEstimateInputs(reshape([2.0, 0.0], 2, 1))
    @test process!(result, workspace, plan, input) === result
    @test effective_rank(result) == 1 < plan.field_count
    @test condition_number(result) == 1.0
    @test result.parameter_offsets ≈ [1.0, 1.0] atol=1e-14
    @test sensitivity * result.parameter_offsets ≈
        vec(input.observed_response) atol=1e-14
    @test result.parameter_offsets ≈ fill(1.0, 2) atol=1e-14
end

@testset "meta-sensitivity validation, aliases, and publication" begin
    reference = zeros(2, 1)
    sensitivity = [1.0 1.0; 0.0 0.0]
    fields = [:first, :second]
    units = ["dimensionless", "dimensionless"]
    specification = MetaSensitivityEstimateSpecification(
        reference, sensitivity, fields, units)
    plan = prepare(MetaSensitivityEstimate(), specification)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = MetaSensitivityEstimateInputs(reshape([2.0, 0.0], 2, 1))
    process!(result, workspace, plan, input)
    before = (copy(result.parameter_offsets), result.diagnostics)

    @test_throws ArgumentError prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            fill(NaN, 2, 1), sensitivity, fields, units),
    )
    @test_throws ArgumentError prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            reference, fill(NaN, 2, 2), fields, units),
    )
    @test_throws DimensionMismatch prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(reference, ones(1, 2), fields, units),
    )
    @test_throws DimensionMismatch prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(reference, sensitivity, [:first], units),
    )
    @test_throws DimensionMismatch prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(reference, sensitivity, fields, ["dimensionless"]),
    )
    @test_throws MethodError MetaSensitivityEstimateSpecification(
        reference, sensitivity, (:first, :second), ("dimensionless", 1))
    @test_throws MethodError MetaSensitivityEstimateSpecification(
        reference, sensitivity, (:first, 2), ("dimensionless", "dimensionless"))
    @test_throws ArgumentError prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            NonOneBasedMisregistrationMatrix(reference), sensitivity, fields, units),
    )

    @test_throws ArgumentError process!(
        result,
        workspace,
        plan,
        MetaSensitivityEstimateInputs(fill(NaN, 2, 1)),
    )
    @test (result.parameter_offsets, result.diagnostics) == before
    overflow_reference = reshape([floatmax(Float64), 0.0], 2, 1)
    overflow_plan = prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            overflow_reference, sensitivity, fields, units),
    )
    overflow_result = allocate_result(overflow_plan)
    overflow_workspace = allocate_workspace(overflow_plan)
    process!(overflow_result, overflow_workspace, overflow_plan,
        MetaSensitivityEstimateInputs(copy(overflow_reference)))
    overflow_before = (copy(overflow_result.parameter_offsets),
        overflow_result.diagnostics)
    @test_throws ArgumentError process!(
        overflow_result,
        overflow_workspace,
        overflow_plan,
        MetaSensitivityEstimateInputs(reshape([-floatmax(Float64), 0.0], 2, 1)),
    )
    @test (overflow_result.parameter_offsets, overflow_result.diagnostics) ==
        overflow_before

    candidate_overflow_plan = prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            zeros(2, 1),
            [1e-200 0.0; 0.0 1e-200],
            fields,
            units,
        ),
    )
    candidate_overflow_result = allocate_result(candidate_overflow_plan)
    candidate_overflow_workspace = allocate_workspace(candidate_overflow_plan)
    process!(candidate_overflow_result, candidate_overflow_workspace,
        candidate_overflow_plan, MetaSensitivityEstimateInputs(zeros(2, 1)))
    candidate_overflow_before = (copy(candidate_overflow_result.parameter_offsets),
        candidate_overflow_result.diagnostics)
    @test_throws ArgumentError process!(
        candidate_overflow_result,
        candidate_overflow_workspace,
        candidate_overflow_plan,
        MetaSensitivityEstimateInputs(reshape([floatmax(Float64), 0.0], 2, 1)),
    )
    @test (candidate_overflow_result.parameter_offsets,
        candidate_overflow_result.diagnostics) == candidate_overflow_before

    @test_throws DimensionMismatch process!(
        result,
        workspace,
        plan,
        MetaSensitivityEstimateInputs(ones(1, 2)),
    )
    @test (result.parameter_offsets, result.diagnostics) == before
    @test_throws ArgumentError process!(
        result,
        workspace,
        plan,
        MetaSensitivityEstimateInputs(NonOneBasedMisregistrationMatrix(zeros(2, 1))),
    )

    aliased_product = MetaSensitivityEstimateProduct(
        workspace.candidate_offsets, result.diagnostics)
    @test_throws ArgumentError process!(aliased_product, workspace, plan, input)
    aliased_workspace = MetaSensitivityEstimateWorkspace(
        workspace.response_difference, workspace.response_difference)
    @test_throws ArgumentError process!(result, aliased_workspace, plan, input)
    shared_observation = reshape([2.0, 0.0], 2, 1)
    aliased_input_product = MetaSensitivityEstimateProduct(
        vec(shared_observation), result.diagnostics)
    @test_throws ArgumentError process!(
        aliased_input_product,
        allocate_workspace(plan),
        plan,
        MetaSensitivityEstimateInputs(shared_observation),
    )
end

@testset "meta-sensitivity views, Float32, and allocation" begin
    reference_storage = zeros(Float32, 4, 2)
    sensitivity_storage = zeros(Float32, 4, 4)
    reference = @view reference_storage[1:2:4, 1:1]
    sensitivity = @view sensitivity_storage[1:2:4, 1:2]
    reference .= Float32[0; 0]
    sensitivity .= Float32[1 1; 0 0]
    plan = @inferred prepare(
        MetaSensitivityEstimate(),
        MetaSensitivityEstimateSpecification(
            reference,
            sensitivity,
            [:first, :second],
            ["dimensionless", "dimensionless"],
        ),
    )
    result_storage = zeros(Float32, 4)
    workspace_storage = zeros(Float32, 8)
    result = MetaSensitivityEstimateProduct(
        @view(result_storage[2:3]),
        MetaSensitivityEstimateDiagnostics(Float32(Inf), 0),
    )
    workspace = MetaSensitivityEstimateWorkspace(
        @view(workspace_storage[1:2]),
        @view(workspace_storage[5:6]),
    )
    observed_storage = zeros(Float32, 4, 2)
    observed_response = @view observed_storage[1:2:4, 1:1]
    observed_response .= Float32[2; 0]
    input = MetaSensitivityEstimateInputs(observed_response)
    @test stride(input.observed_response, 1) != 1
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test result.parameter_offsets ≈ Float32[1, 1] atol=2f-7
    @test misregistration_allocation(result, workspace, plan, input) == 0
end

end
