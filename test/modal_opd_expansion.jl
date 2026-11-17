module ModalOPDExpansionTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ModalBases
using Test

struct NonOneBasedExpansionMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
end
Base.size(array::NonOneBasedExpansionMatrix) = size(array.parent)
Base.axes(array::NonOneBasedExpansionMatrix) =
    (0:(size(array.parent, 1) - 1), 0:(size(array.parent, 2) - 1))
Base.IndexStyle(::Type{<:NonOneBasedExpansionMatrix}) = IndexCartesian()
Base.getindex(array::NonOneBasedExpansionMatrix, row::Int, column::Int) =
    array.parent[row + 1, column + 1]

struct NonOneBasedExpansionBasis{T,A<:AbstractArray{T,3}} <: AbstractArray{T,3}
    parent::A
end
Base.size(array::NonOneBasedExpansionBasis) = size(array.parent)
Base.axes(array::NonOneBasedExpansionBasis) = (
    0:(size(array.parent, 1) - 1),
    0:(size(array.parent, 2) - 1),
    0:(size(array.parent, 3) - 1),
)
Base.IndexStyle(::Type{<:NonOneBasedExpansionBasis}) = IndexCartesian()
Base.getindex(array::NonOneBasedExpansionBasis, row::Int, column::Int, mode::Int) =
    array.parent[row + 1, column + 1, mode + 1]

function expansion_plan(::Type{T}=Float64) where {T<:AbstractFloat}
    support = Bool[true false; true true]
    return prepare(
        ModalOPDExpansion(),
        ModalOPDExpansionSpecification(2, 2, 2, support, T),
    )
end

function expansion_input(::Type{T}=Float64) where {T<:AbstractFloat}
    basis = T[
        1 2
        3 4;;;
        -1 0.5
        2 -2
    ]
    return ModalOPDExpansionInputs(basis, T[2, -0.5])
end

function expansion_allocation(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

function aos_zernike_expansion_fixture()
    support = Bool[
        false true true false
        true true true true
        true true true true
        false true true false
    ]
    basis = zeros(Float64, 4, 4, 3)
    for column in axes(support, 2), row in axes(support, 1)
        if support[row, column]
            basis[row, column, 1] = 1.0
            basis[row, column, 2] = (column - 2.5) / 2
            basis[row, column, 3] = (row - 2.5) / 2
        end
    end
    expected = [
        0.0 -1.7500000000000004e-9 -1.25e-9 0.0
        -1.2500000000000004e-9 -7.500000000000001e-10 -2.4999999999999996e-10 2.4999999999999986e-10
        -2.5000000000000007e-10 2.500000000000001e-10 7.500000000000001e-10 1.25e-9
        0.0 1.2500000000000002e-9 1.7500000000000002e-9 0.0
    ]
    return support, basis, Float64[0.0, 1e-9, 2e-9], expected
end

@testset "AOS Zernike modal OPD expansion fixture" begin
    # Frozen from AOS 6fb8617's `ZernikeBasis` and `combine_basis!` conventions:
    # resolution 4, diameter 8 m, no central obstruction, Noll modes 1:3.
    support, basis, coefficients, expected = aos_zernike_expansion_fixture()
    plan = prepare(
        ModalOPDExpansion(), ModalOPDExpansionSpecification(4, 4, 3, support),
    )
    result = process(plan, ModalOPDExpansionInputs(basis, coefficients))
    @test result.opd ≈ expected atol=1e-24 rtol=1e-14
end

@testset "explicit modal OPD expansion" begin
    plan = @inferred expansion_plan()
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = expansion_input()

    @test plan.pupil_support == Bool[true, true, false, true]
    @test workspace === nothing
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test opd(result) === result.opd
    @test result.opd == [2.5 0.0; 5.0 9.0]
    @test process(plan, input).opd == result.opd
    @test process!(allocate_result(plan), plan, input).opd == result.opd

    support = Bool[true false; true true]
    specification = ModalOPDExpansionSpecification(2, 2, 2, support)
    support .= false
    copied_plan = prepare(ModalOPDExpansion(), specification)
    specification.pupil_support .= false
    @test copied_plan.pupil_support == Bool[true, true, false, true]
end

@testset "modal OPD expansion Float32, views, inference, and allocation" begin
    plan = @inferred expansion_plan(Float32)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    basis_storage = zeros(Float32, 4, 4, 4)
    coefficients_storage = zeros(Float32, 4)
    basis = @view basis_storage[1:2, 1:2, 1:2]
    basis .= expansion_input(Float32).basis
    coefficients = @view coefficients_storage[2:3]
    coefficients .= Float32[2, -0.5]
    input = ModalOPDExpansionInputs(basis, coefficients)
    @test @inferred(process!(result, workspace, plan, input)) === result
    @test result.opd == Float32[2.5 0; 5 9]
    @test expansion_allocation(result, workspace, plan, input) == 0

    product_storage = zeros(Float32, 4, 4)
    view_result = ModalOPDExpansionProduct(@view product_storage[1:2, 2:3])
    @test @inferred(process!(view_result, workspace, plan, input)) === view_result
    @test view_result.opd == result.opd
    @test expansion_allocation(view_result, workspace, plan, input) == 0
end

@testset "modal OPD expansion validation and atomicity" begin
    support = Bool[true false; true true]
    @test_throws ArgumentError ModalOPDExpansionSpecification(0, 2, 2, support)
    @test_throws ArgumentError ModalOPDExpansionSpecification(2, 0, 2, support)
    @test_throws ArgumentError ModalOPDExpansionSpecification(2, 2, 0, support)
    @test_throws DimensionMismatch ModalOPDExpansionSpecification(2, 2, 2, trues(1, 2))
    @test_throws ArgumentError ModalOPDExpansionSpecification(2, 2, 2, falses(2, 2))
    @test_throws ArgumentError ModalOPDExpansionSpecification(
        2, 2, 2, support, Float16,
    )
    @test_throws ArgumentError ModalOPDExpansionSpecification(
        2, 2, 2, support, BigFloat,
    )
    @test_throws ArgumentError ModalOPDExpansionSpecification(
        2, 2, 2, NonOneBasedExpansionMatrix(support),
    )

    plan = expansion_plan(Float32)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = expansion_input(Float32)
    process!(result, workspace, plan, input)
    before = copy(result.opd)

    @test_throws DimensionMismatch process!(
        result, workspace, plan, ModalOPDExpansionInputs(zeros(Float32, 2, 2, 1), Float32[1]),
    )
    @test result.opd == before
    @test_throws DimensionMismatch process!(
        result, workspace, plan, ModalOPDExpansionInputs(zeros(Float32, 2, 2, 2), Float32[1]),
    )
    @test result.opd == before
    @test_throws ArgumentError process!(
        result, workspace, plan, ModalOPDExpansionInputs(fill(NaN32, 2, 2, 2), Float32[1, 2]),
    )
    @test result.opd == before
    @test_throws ArgumentError process!(
        result, workspace, plan, ModalOPDExpansionInputs(zeros(Float32, 2, 2, 2), Float32[NaN, 2]),
    )
    @test result.opd == before
    @test_throws ArgumentError process!(
        result,
        workspace,
        plan,
        ModalOPDExpansionInputs(NonOneBasedExpansionBasis(input.basis), input.coefficients),
    )
    @test result.opd == before
    @test_throws ArgumentError process!(
        ModalOPDExpansionProduct(NonOneBasedExpansionMatrix(zeros(Float32, 2, 2))),
        workspace,
        plan,
        input,
    )
    @test result.opd == before
    @test_throws ArgumentError process!(
        result,
        workspace,
        plan,
        ModalOPDExpansionInputs(fill(floatmax(Float32), 2, 2, 2), Float32[2, 2]),
    )
    @test result.opd == before

    aliased_basis = reshape(result.opd, 2, 2, 1)
    one_mode_plan = prepare(
        ModalOPDExpansion(), ModalOPDExpansionSpecification(2, 2, 1, support, Float32),
    )
    @test_throws ArgumentError process!(
        result,
        nothing,
        one_mode_plan,
        ModalOPDExpansionInputs(aliased_basis, Float32[1]),
    )
    @test result.opd == before
end

end
