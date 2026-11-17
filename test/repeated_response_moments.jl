module RepeatedResponseMomentsTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.Diagnostics
using LinearAlgebra
using Test

struct ShiftedMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::A
end
Base.size(array::ShiftedMatrix) = size(array.parent)
Base.axes(array::ShiftedMatrix) =
    (0:(size(array, 1) - 1), Base.OneTo(size(array, 2)))
Base.IndexStyle(::Type{<:ShiftedMatrix}) = IndexCartesian()
Base.getindex(array::ShiftedMatrix, row::Int, column::Int) = array.parent[row + 1, column]

struct ShiftedVector{A<:AbstractVector{Float64}} <: AbstractVector{Float64}
    parent::A
end
Base.size(array::ShiftedVector) = size(array.parent)
Base.axes(array::ShiftedVector) = (0:(length(array.parent) - 1),)
Base.IndexStyle(::Type{<:ShiftedVector}) = IndexCartesian()
Base.getindex(array::ShiftedVector, index::Int) = array.parent[index + 1]

function product_arrays(result)
    return (result.mean, result.sample_variance, result.mean_standard_error,
        result.covariance_factor)
end

function repeated_allocations(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

@testset "known repeated-response moments and covariance" begin
    # Independently specified scalar moments: deviations are -3,-1,1,3.
    responses = [1.0 3 5 7; 2 4 6 8; 9 9 9 9]
    plan = @inferred prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(3, 4))
    result = allocate_result(plan)
    @test result.status === ResponseMomentsNotProcessed
    @test all(iszero, product_arrays(result))
    result = @inferred process(plan, responses)
    @test result.status === ResponseMomentsValid
    @test result.mean == [4.0, 5.0, 9.0]
    @test result.sample_variance ≈ [20 / 3, 20 / 3, 0.0]
    @test result.mean_standard_error ≈ [sqrt(5 / 3), sqrt(5 / 3), 0.0]
    expected_factor = [-3.0 -1 1 3; -3 -1 1 3; 0 0 0 0] / sqrt(3)
    @test result.covariance_factor ≈ expected_factor
    @test result.covariance_factor * result.covariance_factor' ≈
        [20 / 3 20 / 3 0; 20 / 3 20 / 3 0; 0 0 0]
    # Four rounded factor terms are summed in each coordinate.
    @test vec(sum(result.covariance_factor; dims=2)) ≈ zeros(3) atol=4eps(Float64)
    @test rank(result.covariance_factor) == 1
    @test all(array -> eltype(array) === Float64, product_arrays(result))
    @test responses == [1.0 3 5 7; 2 4 6 8; 9 9 9 9]

    workspace = allocate_workspace(plan)
    @test @inferred(process!(result, workspace, plan, responses)) === result
    @test repeated_allocations(result, workspace, plan, responses) == 0
    process!(result, workspace, plan, fill(Float32(7), 3, 4))
    @test result.mean == fill(7.0, 3)
    @test iszero(result.sample_variance)
    @test iszero(result.mean_standard_error)
    @test iszero(result.covariance_factor)
    @test process!(result, plan, responses).mean == [4.0, 5.0, 9.0]
    @test process!(result, allocate_workspace(plan), plan, responses).covariance_factor ≈ expected_factor

    two = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(1, 2))
    result_two = process(two, reshape(Int16[0, 4], 1, 2))
    @test result_two.mean == [2.0]
    @test result_two.sample_variance == [8.0]
    @test result_two.mean_standard_error == [2.0]
    @test result_two.covariance_factor == reshape([-2.0, 2.0], 1, 2)
end

@testset "precision and measurement/repeat order" begin
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 4))
    # A naive second-moment subtraction loses this variance at a large offset.
    high_offset = [1e12 1e12+2 1e12+4 1e12+6; -1e12 -1e12+2 -1e12+4 -1e12+6]
    result = process(plan, high_offset)
    @test result.status === ResponseMomentsValid
    @test result.mean == [1e12 + 3, -1e12 + 3]
    @test result.sample_variance ≈ fill(20 / 3, 2)
    @test result.covariance_factor ≈ [-3.0 -1 1 3; -3 -1 1 3] / sqrt(3)
    extreme_constant = process(plan, fill(floatmax(Float64), 2, 4))
    @test extreme_constant.status === ResponseMomentsValid
    @test extreme_constant.mean == fill(floatmax(Float64), 2)
    @test iszero(extreme_constant.sample_variance)
    @test iszero(extreme_constant.covariance_factor)
    float16_constant = process(plan, fill(Float16(4096), 2, 4))
    @test float16_constant.mean == fill(4096.0, 2)
    @test iszero(float16_constant.sample_variance)

    order_plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(4, 2))
    responses = [1.0 3; 10 14; 100 100; -6 -2]
    ordered = process(order_plan, responses)
    @test ordered.mean == [2.0, 12.0, 100.0, -4.0]
    @test ordered.sample_variance == [2.0, 8.0, 0.0, 8.0]
    reversed = process(order_plan, responses[:, end:-1:1])
    @test reversed.mean == ordered.mean
    @test reversed.sample_variance == ordered.sample_variance
    @test reversed.covariance_factor == ordered.covariance_factor[:, end:-1:1]
end

@testset "near-ULP represented differences retain variance across permutations" begin
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(1, 4))
    values = [1e16 + 2, 1e16, 1e16, 1e16]
    oracle_mean, oracle_variance = setprecision(BigFloat, 256) do
        exact = BigFloat.(values)
        mean = sum(exact) / 4
        variance = sum((value - mean)^2 for value in exact) / 3
        mean, variance
    end
    @test oracle_mean == BigFloat(1e16) + BigFloat(0.5)
    @test oracle_variance == 1
    for shift in 0:3
        result = process(plan, reshape(circshift(values, shift), 1, 4))
        @test result.status === ResponseMomentsValid
        @test only(result.mean) == Float64(oracle_mean)
        @test only(result.sample_variance) ≈ Float64(oracle_variance) rtol=4eps(Float64)
        @test only(result.mean_standard_error) ≈ 0.5 rtol=4eps(Float64)
        @test sum(abs2, result.covariance_factor) ≈ Float64(oracle_variance) rtol=4eps(Float64)
        @test result.covariance_factor * result.covariance_factor' ≈
            reshape(result.sample_variance, 1, 1) rtol=4eps(Float64)
        @test only(sum(result.covariance_factor; dims=2)) ≈ 0.0 atol=4eps(Float64)
    end
end

@testset "ordinary strided views and explicit one-based axes" begin
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 4))
    parent = zeros(Float32, 4, 8)
    responses = @view parent[1:2:4, 1:2:8]
    responses .= [1 3 5 7; 2 4 6 8]
    vectors = ntuple(_ -> @view(zeros(4)[1:2:4]), 5)
    factors = ntuple(_ -> @view(zeros(4, 8)[1:2:4, 1:2:8]), 2)
    result = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed,
        vectors[1], vectors[2], vectors[3], factors[1])
    workspace = RepeatedResponseMomentsWorkspace(vectors[4], vectors[5], factors[2])
    @test process!(result, workspace, plan, responses).mean == [4.0, 5.0]
    @test result.sample_variance ≈ fill(20 / 3, 2)
    before = map(copy, product_arrays(result))
    @test_throws ArgumentError process!(result, workspace, plan, ShiftedMatrix(responses))
    @test product_arrays(result) == before
    shifted = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed,
        ShiftedVector(zeros(2)), zeros(2), zeros(2), zeros(2, 4))
    @test_throws ArgumentError process!(shifted, workspace, plan, responses)
    shifted_factor = RepeatedResponseMomentsWorkspace(zeros(2), zeros(2), ShiftedMatrix(zeros(2, 4)))
    @test_throws ArgumentError process!(result, shifted_factor, plan, responses)
end

@testset "specification and structural failures preserve products" begin
    for measurements in (-1, 0)
        @test_throws ArgumentError RepeatedResponseSpecification(measurements, 2)
    end
    for samples in (-1, 0, 1)
        @test_throws ArgumentError RepeatedResponseSpecification(1, samples)
    end
    @test_throws ArgumentError RepeatedResponseSpecification(typemax(Int), 2)
    # Direct construction must preserve the bounds proof for scalar indexing.
    @test_throws ArgumentError RepeatedResponseMomentsPlan(0, 2)
    @test_throws ArgumentError RepeatedResponseMomentsPlan(2, 1)
    @test_throws ArgumentError RepeatedResponseMomentsPlan(typemax(Int), 2)
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 2))
    responses = [1.0 3; 10 14]
    result, workspace = process(plan, responses), allocate_workspace(plan)
    before = map(copy, product_arrays(result))
    for invalid in (zeros(1, 2), zeros(2, 1), zeros(2, 3))
        @test_throws DimensionMismatch process!(result, workspace, plan, invalid)
    end
    short_vector = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed,
        zeros(1), zeros(2), zeros(2), zeros(2, 2))
    @test_throws DimensionMismatch process!(short_vector, workspace, plan, responses)
    short_factor = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed,
        zeros(2), zeros(2), zeros(2), zeros(2, 1))
    @test_throws DimensionMismatch process!(short_factor, workspace, plan, responses)
    bad_workspace = RepeatedResponseMomentsWorkspace(zeros(2), zeros(1), zeros(2, 2))
    @test_throws DimensionMismatch process!(result, bad_workspace, plan, responses)
    @test result.status === ResponseMomentsValid
    @test product_arrays(result) == before
end

@testset "input and all product/scratch alias guards" begin
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 2))
    responses = [1.0 3; 10 14]
    # All 21 pairs of distinct writable arrays must be independent.
    for first in 1:7, second in (first + 1):7
        arrays = (zeros(2), zeros(2), zeros(2), zeros(2, 2), zeros(2), zeros(2), zeros(2, 2))
        storage = zeros(4)
        first_array = first in (4, 7) ? reshape(storage, 2, 2) : @view(storage[1:2])
        second_array = second in (4, 7) ? reshape(storage, 2, 2) : @view(storage[1:2])
        arrays = Base.setindex(Base.setindex(arrays, first_array, first), second_array, second)
        result = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed, arrays[1:4]...)
        workspace = RepeatedResponseMomentsWorkspace(arrays[5:7]...)
        @test_throws ArgumentError process!(result, workspace, plan, responses)
        @test result.status === ResponseMomentsNotProcessed
        @test all(iszero, product_arrays(result))
    end
    for index in 1:7
        arrays = (zeros(2), zeros(2), zeros(2), zeros(2, 2), zeros(2), zeros(2), zeros(2, 2))
        input_alias = index in (4, 7) ? responses : @view(responses[:, 1])
        arrays = Base.setindex(arrays, input_alias, index)
        result = RepeatedResponseMomentsProduct(ResponseMomentsNotProcessed, arrays[1:4]...)
        workspace = RepeatedResponseMomentsWorkspace(arrays[5:7]...)
        @test_throws ArgumentError process!(result, workspace, plan, responses)
        @test responses == [1.0 3; 10 14]
    end
end

@testset "typed non-finite outcomes zero every product and permit reuse" begin
    plan = prepare(RepeatedResponseMoments(), RepeatedResponseSpecification(2, 2))
    responses = [1.0 3; 10 14]
    result, workspace = process(plan, responses), allocate_workspace(plan)
    for value in (NaN, Inf, -Inf)
        invalid = copy(responses)
        invalid[2, 2] = value
        @test process!(result, workspace, plan, invalid) === result
        @test result.status === ResponseMomentsNonFiniteInput
        @test all(iszero, product_arrays(result))
        @test process!(result, workspace, plan, responses).status === ResponseMomentsValid
    end
    for invalid in ([floatmax(Float64) -floatmax(Float64); 1 2],
                    [0.0 1e200; 1 2],
                    [big"1e1000" big"1e1000"; 1 2])
        @test all(isfinite, invalid)
        @test process!(result, workspace, plan, invalid).status === ResponseMomentsNonFiniteCalculation
        @test all(iszero, product_arrays(result))
        @test process!(result, workspace, plan, responses).mean == [2.0, 12.0]
    end
end

@testset "public repeated-response documentation" begin
    for name in (:RepeatedResponseMoments, :RepeatedResponseSpecification,
                 :RepeatedResponseMomentsPlan, :RepeatedResponseMomentsProduct,
                 :RepeatedResponseMomentsWorkspace, :ResponseMomentsStatus,
                 :ResponseMomentsNotProcessed, :ResponseMomentsValid,
                 :ResponseMomentsNonFiniteInput, :ResponseMomentsNonFiniteCalculation)
        @test Base.isexported(Diagnostics, name)
        @test Base.Docs.hasdoc(Diagnostics, name)
    end
end

end
