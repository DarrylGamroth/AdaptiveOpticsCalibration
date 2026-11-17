module ReferenceFrameTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ReferenceFrames
using Test

struct ShiftedCube{T,A<:AbstractArray{T,3}} <: AbstractArray{T,3}
    parent::A
end
Base.size(array::ShiftedCube) = size(array.parent)
Base.axes(array::ShiftedCube) = (0:(size(array, 1) - 1),
    Base.OneTo(size(array, 2)), Base.OneTo(size(array, 3)))
Base.IndexStyle(::Type{<:ShiftedCube}) = IndexCartesian()
Base.getindex(array::ShiftedCube, row::Int, col::Int, sample::Int) =
    array.parent[row + 1, col, sample]

function repeated_allocations(result, workspace, plan, input)
    process!(result, workspace, plan, input)
    return @allocated process!(result, workspace, plan, input)
end

@testset "dark Welford moments and repeat calls" begin
    raw = UInt16[1 10; 20 100;;; 3 12; 22 102;;; 5 14; 24 104]
    plan = @inferred prepare(DarkFrameMoments(), DarkFrameSpecification(2, 2, 3))
    product = @inferred process(plan, raw)
    @test product.status === Valid
    @test product.mean == [3.0 12.0; 22.0 102.0]
    @test product.variance == fill(4.0, 2, 2) # N - 1 denominator, not N.
    @test eltype(product.mean) === Float64
    @test eltype(product.variance) === Float64
    workspace = allocate_workspace(plan)
    @test @inferred(process!(product, workspace, plan, raw)) === product
    @test repeated_allocations(product, workspace, plan, raw) == 0
    process!(product, workspace, plan, fill(UInt8(7), 2, 2, 3))
    @test product.mean == fill(7.0, 2, 2)
    @test iszero(product.variance)
    @test process!(allocate_result(plan), plan, raw).mean == [3.0 12.0; 22.0 102.0]
    @test process!(product, allocate_workspace(plan), plan, raw).variance == fill(4.0, 2, 2)
    two = prepare(DarkFrameMoments(), DarkFrameSpecification(1, 1, 2))
    @test process(two, reshape(UInt32[0, 4], 1, 1, 2)).variance == fill(8.0, 1, 1)
    widest = process(two, reshape(UInt128[0, typemax(UInt128)], 1, 1, 2))
    @test widest.status === Valid
    @test all(isfinite, widest.mean)
    @test all(isfinite, widest.variance)

    for n in (-1, 0, 1)
        @test_throws ArgumentError DarkFrameSpecification(2, 2, n)
    end
    @test_throws ArgumentError DarkFrameSpecification(0, 2, 2)
    @test_throws ArgumentError DarkFrameSpecification(2, 0, 2)
    @test_throws ArgumentError DarkFrameSpecification(typemax(Int), 2, 2)
    before = copy(product.mean)
    @test_throws DimensionMismatch process!(product, workspace, plan, zeros(UInt16, 1, 2, 3))
    @test_throws DimensionMismatch process!(product, workspace, plan, zeros(UInt16, 2, 2, 2))
    @test_throws ArgumentError process!(product, workspace, plan, ShiftedCube(raw))
    @test_throws ArgumentError process!(product,
        DarkFrameWorkspace(product.mean, zeros(2, 2)), plan, raw)
    @test product.mean == before
    storage = zeros(UInt16, 4, 4, 6)
    view_raw = @view storage[1:2:4, 1:2:4, 1:2:6]
    view_raw .= raw
    @test process(plan, view_raw).mean == before
end

function lamp_fixture(::Type{T}=Float64) where {T<:AbstractFloat}
    centroids = T[1 3 5; 10 14 18; 100 110 120; 20 22 24; -4 -2 0; 40 44 48]
    flux = T[5 6 7; 20 21 22; 5 4 6]
    valid = fill(true, 3, 3)
    valid[2, 2] = false
    return LampReferenceInputs(centroids, flux, valid, fill(UInt16(42), 2, 2, 3);
        centroids_are_absolute=true)
end

@testset "lamp per-ROI criteria, row ordering, and repeat calls" begin
    input = lamp_fixture()
    method = LampReference([5.0, 10.0, 4.0], UInt16(100))
    plan = @inferred prepare(method, LampReferenceSpecification(2, 2, 3, 3))
    product = @inferred process(plan, input)
    @test product.status === Valid
    @test product.eligible == [true, false, true]
    @test product.reference_centroids == [3.0, 14.0, 0.0, 0.0, -2.0, 44.0]
    @test product.minimum_flux == [5.0, 20.0, 4.0]
    @test product.all_training_valid == [true, false, true]
    @test product.adc_maximum == 42.0
    @test eltype(product.reference_centroids) === Float64
    workspace = allocate_workspace(plan)
    @test @inferred(process!(product, workspace, plan, input)) === product
    @test repeated_allocations(product, workspace, plan, input) == 0
    @test repeated_allocations(product, workspace, plan, lamp_fixture(Float32)) == 0
    @test process!(allocate_result(plan), plan, input).reference_centroids ==
        product.reference_centroids
    method.minimum_flux .= 1000.0
    @test plan.minimum_flux == [5.0, 10.0, 4.0]
    input.intrinsic_valid .= true
    input.centroids .= 8.0
    process!(product, workspace, plan, input)
    @test product.reference_centroids == fill(8.0, 6)
    @test product.eligible == [true, true, true]
    @test process(plan, input).reference_centroids == product.reference_centroids
    view_input = LampReferenceInputs(@view(input.centroids[:, :]),
        @view(input.flux[:, :]), @view(input.intrinsic_valid[:, :]),
        @view(input.raw_adc[:, :, :]); centroids_are_absolute=true)
    @test process(plan, view_input).reference_centroids == fill(8.0, 6)
    unweighted_plan = prepare(LampReference([1.0], 100),
        LampReferenceSpecification(1, 1, 1, 2))
    individual = LampReferenceInputs([0.0 10.0; 2.0 6.0], [1.0 99.0],
        fill(true, 1, 2), fill(UInt8(1), 1, 1, 2); centroids_are_absolute=true)
    # Averaging deployed centroids differs from dividing accumulated moments
    # by accumulated flux: x = 5 pixels here, not 9.9 pixels.
    @test process(unweighted_plan, individual).reference_centroids == [5.0, 4.0]
end

@testset "188 ROI geometry independent of eligible count" begin
    plan = prepare(LampReference(fill(3.0, 188), 65535),
        LampReferenceSpecification(1, 1, 188, 2))
    centroids = repeat(reshape(Float64.(1:376), 376, 1), 1, 2)
    flux = fill(3.0, 188, 2)
    valid = fill(true, 188, 2)
    valid[1:4, 1] .= false
    input = LampReferenceInputs(centroids, flux, valid, fill(UInt16(1), 1, 1, 2);
        centroids_are_absolute=true)
    product = process(plan, input)
    @test product.status === Valid
    @test length(product.eligible) == 188
    @test length(product.minimum_flux) == 188
    @test count(product.eligible) == 184
    @test product.reference_centroids[1:8] == zeros(8)
    @test product.reference_centroids[9:end] == Float64.(9:376)
    valid[5, 2] = false
    @test count(process(plan, input).eligible) == 183
    valid .= true
    @test count(process(plan, input).eligible) == 188
end

@testset "lamp ADC rail, finite inputs, and no eligible ROI" begin
    plan = prepare(LampReference([5.0, 10.0, 4.0], 100),
        LampReferenceSpecification(2, 2, 3, 3))
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = lamp_fixture()
    input.raw_adc[2, 2, 3] = 99
    @test process!(product, workspace, plan, input).status === Valid
    input.raw_adc[2, 2, 3] = 100
    @test process!(product, workspace, plan, input).status === Invalid
    @test !any(product.eligible)
    @test iszero(product.reference_centroids)
    @test product.adc_maximum == 100.0
    @test product.minimum_flux == [5.0, 20.0, 4.0]
    @test product.all_training_valid == [true, false, true]
    input.raw_adc[2, 2, 3] = 101
    @test process!(product, workspace, plan, input).status === Invalid
    input.raw_adc .= 42
    input.intrinsic_valid .= false
    @test process!(product, workspace, plan, input).status === NoEligible
    @test iszero(product.reference_centroids)
    @test !any(product.eligible)
    input.intrinsic_valid .= true
    input.flux .= 0
    @test process!(product, workspace, plan, input).status === NoEligible
    for bad in (NaN, Inf, -Inf)
        input = lamp_fixture()
        input.flux[2, 2] = bad
        @test process!(product, workspace, plan, input).status === Invalid
        # Infinite positive flux is itself invalid even though its min is finite.
        @test !product.all_training_valid[2]
        @test product.minimum_flux[[1, 3]] == [5.0, 4.0]
        @test !any(product.eligible)
        input = lamp_fixture()
        input.centroids[3, 2] = bad
        @test process!(product, workspace, plan, input).status === Invalid
        @test iszero(product.reference_centroids)
    end
    input = lamp_fixture()
    input.centroids[1, :] .= [floatmax(Float64), -floatmax(Float64), 1.0]
    @test process!(product, workspace, plan, input).status === Invalid
    @test process!(product, workspace, plan, lamp_fixture()).status === Valid
    absolute = lamp_fixture()
    known_reference = Float64[1, 2, 3, 4, 5, 6]
    residuals = absolute.centroids .- known_reference
    reconstructed = LampReferenceInputs(residuals .+ known_reference,
        absolute.flux, absolute.intrinsic_valid, absolute.raw_adc;
        centroids_are_absolute=true)
    @test process(plan, reconstructed).reference_centroids ==
        process(plan, absolute).reference_centroids
end

@testset "lamp shape contracts and policy validation" begin
    for thresholds in (Float64[], [-1.0], [NaN], [Inf])
        @test_throws ArgumentError LampReference(thresholds, 100)
    end
    for rail in (0, -1, NaN, Inf)
        @test_throws ArgumentError LampReference([1.0], rail)
    end
    for dims in ((0, 2, 3, 3), (2, 0, 3, 3), (2, 2, 0, 3), (2, 2, 3, 0))
        @test_throws ArgumentError LampReferenceSpecification(dims...)
    end
    @test_throws ArgumentError LampReferenceSpecification(typemax(Int), 2, 3, 3)
    @test_throws ArgumentError LampReferenceSpecification(1, 1, typemax(Int), 1)
    spec = LampReferenceSpecification(2, 2, 3, 3)
    @test_throws DimensionMismatch prepare(LampReference([1.0], 100), spec)
    plan = prepare(LampReference([5.0, 10.0, 4.0], 100), spec)
    input = lamp_fixture()
    @test_throws UndefKeywordError LampReferenceInputs(input.centroids, input.flux,
        input.intrinsic_valid, input.raw_adc)
    falsely_declared = LampReferenceInputs(input.centroids, input.flux,
        input.intrinsic_valid, input.raw_adc; centroids_are_absolute=false)
    product = process(plan, input)
    workspace = allocate_workspace(plan)
    before = copy(product.reference_centroids)
    @test_throws ArgumentError process!(product, workspace, plan, falsely_declared)
    @test_throws DimensionMismatch process!(product, workspace, plan,
        LampReferenceInputs(zeros(5, 3), input.flux, input.intrinsic_valid,
            input.raw_adc; centroids_are_absolute=true))
    @test_throws DimensionMismatch process!(product, workspace, plan,
        LampReferenceInputs(input.centroids, zeros(2, 3), input.intrinsic_valid,
            input.raw_adc; centroids_are_absolute=true))
    @test_throws DimensionMismatch process!(product, workspace, plan,
        LampReferenceInputs(input.centroids, input.flux, fill(true, 2, 3),
            input.raw_adc; centroids_are_absolute=true))
    @test_throws DimensionMismatch process!(product, workspace, plan,
        LampReferenceInputs(input.centroids, input.flux, input.intrinsic_valid,
            zeros(UInt16, 2, 2, 2); centroids_are_absolute=true))
    @test_throws ArgumentError process!(product, workspace, plan,
        LampReferenceInputs(input.centroids, input.flux, input.intrinsic_valid,
            ShiftedCube(input.raw_adc); centroids_are_absolute=true))
    @test_throws ArgumentError process!(product,
        LampReferenceWorkspace(product.reference_centroids), plan, input)
    @test product.reference_centroids == before
    single = prepare(LampReference([0.0], 10), LampReferenceSpecification(1, 1, 1, 1))
    one = LampReferenceInputs(reshape([2.0, -3.0], 2, 1), zeros(1, 1),
        fill(true, 1, 1), zeros(UInt8, 1, 1, 1); centroids_are_absolute=true)
    @test process(single, one).reference_centroids == [2.0, -3.0]
end

end
