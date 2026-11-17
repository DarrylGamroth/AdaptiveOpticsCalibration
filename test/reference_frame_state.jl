module ReferenceFrameStateTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.ReferenceFrames
using Test
import ..ReferenceFrameTests: ShiftedCube, lamp_fixture

snapshot(state::DarkFrameState) = deepcopy((state.count, state.mean,
    state.sum_squared_deviations, state.finite))
snapshot(state::LampReferenceState) = deepcopy((state.count, state.reference_centroids,
    state.minimum_flux, state.all_training_valid, state.adc_maximum,
    state.finite, state.below_rail))
product_snapshot(product::DarkFrameProduct) = deepcopy((product.status,
    product.mean, product.variance))
product_snapshot(product::LampReferenceProduct) = deepcopy((product.status,
    product.reference_centroids, product.eligible, product.minimum_flux,
    product.all_training_valid, product.adc_maximum))
function chunk(input, exposures)
    return LampReferenceInputs(@view(input.centroids[:, exposures]),
        @view(input.flux[:, exposures]), @view(input.intrinsic_valid[:, exposures]),
        @view(input.raw_adc[:, :, exposures]); centroids_are_absolute=true)
end

@testset "dark ordered state and independent analytic moments" begin
    raw = UInt16[1 10; 20 100;;; 3 12; 22 102;;; 5 14; 24 104;;; 7 16; 26 106]
    plan = prepare(DarkFrameMoments(), DarkFrameSpecification(2, 2, 4))
    batch = process(plan, raw)
    for partitions in ((1:4,), (1:1, 2:4), (1:3, 4:4), (1:1, 2:2, 3:3, 4:4))
        state = @inferred DarkFrameState(plan)
        @test state.count == 0
        for samples in partitions
            @test @inferred(process!(state, plan, @view raw[:, :, samples])) === state
        end
        before = snapshot(state)
        product = @inferred process(plan, state)
        @test product.mean == [4.0 13.0; 23.0 103.0]
        @test product.variance == fill(20 / 3, 2, 2)
        @test product_snapshot(product) == product_snapshot(batch)
        @test snapshot(state) == before
        product.mean .= 0.0
        @test snapshot(state) == before
        @test process!(allocate_result(plan), allocate_workspace(plan), plan, state).mean == batch.mean
    end
    state = DarkFrameState(plan)
    frame = copy(raw[:, :, 1])
    process!(state, plan, frame)
    frame .= 0
    @test state.mean == Float64.(raw[:, :, 1])
    for exposure in 2:4
        process!(state, plan, @view raw[:, :, exposure])
    end
    @test process(plan, state).mean == batch.mean
    @test DarkFrameState(plan).count == 0
    @test iszero(DarkFrameState(plan).mean)
    for T in (UInt8, UInt16, UInt32, UInt64, UInt128)
        constant_raw = fill(typemax(T), 2, 2, 4)
        state = DarkFrameState(plan)
        for exposure in 1:4
            process!(state, plan, @view constant_raw[:, :, exposure])
        end
        product = process(plan, state)
        @test product.mean == fill(Float64(typemax(T)), 2, 2)
        @test iszero(product.variance)
        @test product.status === Valid
    end
end

@testset "dark rejection before writes and owned extraction" begin
    plan = prepare(DarkFrameMoments(), DarkFrameSpecification(2, 2, 3))
    state = DarkFrameState(plan)
    before = snapshot(state)
    for raw in (zeros(UInt16, 1, 2, 1), zeros(UInt16, 2, 1, 1))
        @test_throws DimensionMismatch process!(state, plan, raw)
        @test snapshot(state) == before
    end
    for raw in (zeros(UInt16, 2, 2, 0), zeros(UInt16, 2, 2, 4))
        @test_throws ArgumentError process!(state, plan, raw)
        @test snapshot(state) == before
    end
    @test_throws ArgumentError process!(state, plan, ShiftedCube(zeros(UInt16, 2, 2, 1)))
    @test_throws ArgumentError process!(state,
        prepare(DarkFrameMoments(), DarkFrameSpecification(2, 2, 4)), zeros(UInt16, 2, 2, 1))
    aliased = reshape(reinterpret(UInt64, vec(state.mean)), 2, 2, 1)
    @test_throws ArgumentError process!(state, plan, aliased)
    @test snapshot(state) == before
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    product.mean .= 42
    output_before = product_snapshot(product)
    @test_throws ArgumentError process!(product, workspace, plan, state)
    @test product_snapshot(product) == output_before
    process!(state, plan, fill(UInt16(7), 2, 2, 3))
    complete = snapshot(state)
    @test_throws ArgumentError process!(state, plan, fill(UInt16(8), 2, 2, 1))
    @test snapshot(state) == complete
    @test_throws DimensionMismatch process!(DarkFrameProduct(Invalid, zeros(1, 2), zeros(2, 2)),
        workspace, plan, state)
    @test_throws ArgumentError process!(DarkFrameProduct(Invalid, state.mean, zeros(2, 2)),
        workspace, plan, state)
    @test_throws ArgumentError process!(product,
        DarkFrameWorkspace(state.mean, zeros(2, 2)), plan, state)
    @test snapshot(state) == complete
    @test product_snapshot(product) == output_before
    @test process(plan, state).mean == fill(7.0, 2, 2)
end

@testset "lamp ordered state preserves deployed sample means and diagnostics" begin
    input = lamp_fixture()
    plan = prepare(LampReference([5.0, 10.0, 4.0], UInt16(100)),
        LampReferenceSpecification(2, 2, 3, 3))
    batch = process(plan, input)
    for partitions in ((1:3,), (1:1, 2:3), (1:2, 3:3), (1:1, 2:2, 3:3))
        state = @inferred LampReferenceState(plan)
        for samples in partitions
            @test @inferred(process!(state, plan, chunk(input, samples))) === state
        end
        before = snapshot(state)
        product = @inferred process(plan, state)
        @test product_snapshot(product) == product_snapshot(batch)
        @test product.reference_centroids == [3.0, 14.0, 0.0, 0.0, -2.0, 44.0]
        @test product.minimum_flux == [5.0, 20.0, 4.0]
        @test product.all_training_valid == [true, false, true]
        @test state.reference_centroids == [3.0, 14.0, 110.0, 22.0, -2.0, 44.0]
        @test snapshot(state) == before
        product.reference_centroids .= 999
        product.minimum_flux .= 999
        product.all_training_valid .= false
        @test snapshot(state) == before
    end
    unweighted_plan = prepare(LampReference([1.0], 100),
        LampReferenceSpecification(1, 1, 1, 2))
    unequal = LampReferenceInputs([0.0 10.0; 2.0 6.0], [1.0 99.0],
        fill(true, 1, 2), fill(UInt8(1), 1, 1, 2); centroids_are_absolute=true)
    state = LampReferenceState(unweighted_plan)
    process!(state, unweighted_plan, chunk(unequal, 1:1))
    process!(state, unweighted_plan, chunk(unequal, 2:2))
    @test process(unweighted_plan, state).reference_centroids == [5.0, 4.0]
    single_plan = prepare(LampReference([0.0], 10), LampReferenceSpecification(1, 1, 1, 1))
    state = LampReferenceState(single_plan)
    process!(state, single_plan, chunk(unequal, 1:1))
    @test process(single_plan, state).reference_centroids == [0.0, 2.0]
    @test LampReferenceState(single_plan).count == 0
end

@testset "lamp failures remain sticky across partitions with full diagnostics" begin
    plan = prepare(LampReference([5.0, 10.0, 4.0], UInt16(100)),
        LampReferenceSpecification(2, 2, 3, 3))
    cases = Symbol[:rail_equal, :rail_above, :nan_flux, :inf_flux, :negative_inf_flux,
        :nan_centroid, :inf_centroid, :overflow_centroid, :none_valid, :below_flux]
    for case in cases
        input = lamp_fixture()
        case == :rail_equal && (input.raw_adc[1, 1, 1] = 100)
        case == :rail_above && (input.raw_adc[1, 1, 1] = 101)
        case == :nan_flux && (input.flux[1, 1] = NaN)
        case == :inf_flux && (input.flux[1, 1] = Inf)
        case == :negative_inf_flux && (input.flux[1, 1] = -Inf)
        case == :nan_centroid && (input.centroids[1, 1] = NaN)
        case == :inf_centroid && (input.centroids[1, 1] = Inf)
        case == :overflow_centroid && (input.centroids[1, :] .= [floatmax(Float64), -floatmax(Float64), 1.0])
        case == :none_valid && (input.intrinsic_valid .= false)
        case == :below_flux && (input.flux .= 0)
        state = LampReferenceState(plan)
        for exposure in 1:3
            process!(state, plan, chunk(input, exposure:exposure))
        end
        actual = process(plan, state)
        @test isequal(product_snapshot(actual), product_snapshot(process(plan, input)))
        @test actual.status === (case in (:none_valid, :below_flux) ? NoEligible : Invalid)
        @test iszero(actual.reference_centroids)
        @test !any(actual.eligible)
        @test state.count == 3
        @test state.minimum_flux[3] == (case == :below_flux ? 0.0 : 4.0)
    end
    # Training invalidity masks only its ROI and never removes a whole sample.
    input = lamp_fixture()
    input.intrinsic_valid .= true
    input.intrinsic_valid[1, 1] = false
    state = LampReferenceState(plan)
    process!(state, plan, chunk(input, 1:1))
    process!(state, plan, chunk(input, 2:3))
    @test process(plan, state).eligible == [false, true, true]
    @test process(plan, state).reference_centroids == [0.0, 0.0, 110.0, 22.0, -2.0, 44.0]
end

@testset "188 ROI state retains full measurement ordering" begin
    plan = prepare(LampReference(fill(3.0, 188), 65535),
        LampReferenceSpecification(1, 1, 188, 2))
    input = LampReferenceInputs(repeat(reshape(Float32.(1:376), 376, 1), 1, 2),
        fill(3.0f0, 188, 2), fill(true, 188, 2), fill(UInt16(1), 1, 1, 2);
        centroids_are_absolute=true)
    input.intrinsic_valid[1:4, 1] .= false
    state = LampReferenceState(plan)
    process!(state, plan, chunk(input, 1:1))
    process!(state, plan, chunk(input, 2:2))
    product = process(plan, state)
    @test length(product.eligible) == 188
    @test count(product.eligible) == 184
    @test product.reference_centroids[1:8] == zeros(8)
    @test product.reference_centroids[9:end] == Float64.(9:376)
    @test product_snapshot(product) == product_snapshot(process(plan, input))
end

@testset "lamp state structural rejection, configuration and storage ownership" begin
    plan = prepare(LampReference([5.0, 10.0, 4.0], UInt16(100)),
        LampReferenceSpecification(2, 2, 3, 3))
    input = lamp_fixture()
    state = LampReferenceState(plan)
    before = snapshot(state)
    malformed = (
        LampReferenceInputs(zeros(5, 1), zeros(3, 1), fill(true, 3, 1), zeros(UInt16, 2, 2, 1); centroids_are_absolute=true),
        LampReferenceInputs(zeros(6, 1), zeros(2, 1), fill(true, 3, 1), zeros(UInt16, 2, 2, 1); centroids_are_absolute=true),
        LampReferenceInputs(zeros(6, 1), zeros(3, 1), fill(true, 2, 1), zeros(UInt16, 2, 2, 1); centroids_are_absolute=true),
        LampReferenceInputs(zeros(6, 1), zeros(3, 1), fill(true, 3, 1), zeros(UInt16, 1, 2, 1); centroids_are_absolute=true),
    )
    for bad in malformed
        @test_throws DimensionMismatch process!(state, plan, bad)
        @test snapshot(state) == before
    end
    @test_throws ArgumentError process!(state, plan, chunk(input, 1:0))
    @test_throws ArgumentError process!(state, plan, LampReferenceInputs(input.centroids,
        input.flux, input.intrinsic_valid, input.raw_adc; centroids_are_absolute=false))
    @test_throws ArgumentError process!(state, plan, LampReferenceInputs(input.centroids,
        input.flux, input.intrinsic_valid, ShiftedCube(input.raw_adc); centroids_are_absolute=true))
    for storage in (:centroid, :flux, :validity)
        bad = LampReferenceInputs(
            storage == :centroid ? reshape(state.reference_centroids, 6, 1) : zeros(6, 1),
            storage == :flux ? reshape(state.minimum_flux, 3, 1) : zeros(3, 1),
            storage == :validity ? reshape(state.all_training_valid, 3, 1) : fill(true, 3, 1),
            zeros(UInt16, 2, 2, 1); centroids_are_absolute=true)
        @test_throws ArgumentError process!(state, plan, bad)
        @test snapshot(state) == before
    end
    other = prepare(LampReference([5.0, 10.0, 4.0], UInt16(101)),
        LampReferenceSpecification(2, 2, 3, 3))
    @test_throws ArgumentError process!(state, other, chunk(input, 1:1))
    other = prepare(LampReference([5.0, 10.0, 4.0], UInt16(100)),
        LampReferenceSpecification(2, 2, 3, 4))
    @test_throws ArgumentError process!(state, other, chunk(input, 1:1))
    plan.minimum_flux[1] = 6.0
    @test state.plan.minimum_flux == [5.0, 10.0, 4.0]
    @test_throws ArgumentError process!(state, plan, chunk(input, 1:1))
    @test snapshot(state) == before
    plan.minimum_flux[1] = 5.0
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    output_before = product_snapshot(product)
    @test_throws ArgumentError process!(product, workspace, plan, state)
    @test product_snapshot(product) == output_before
    process!(state, plan, chunk(input, 1:1))
    first = snapshot(state)
    input.centroids[:, 1] .= 999
    input.flux[:, 1] .= 999
    input.intrinsic_valid[:, 1] .= false
    @test snapshot(state) == first
    process!(state, plan, chunk(input, 2:3))
    complete = snapshot(state)
    @test_throws ArgumentError process!(state, plan, chunk(input, 1:1))
    @test snapshot(state) == complete
    @test_throws ArgumentError process!(product,
        LampReferenceWorkspace(state.reference_centroids), plan, state)
    aliased = LampReferenceProduct(Invalid, zeros(6), fill(false, 3), state.minimum_flux,
        fill(false, 3), 0.0)
    @test_throws ArgumentError process!(aliased, workspace, plan, state)
    @test_throws DimensionMismatch process!(LampReferenceProduct(Invalid, zeros(5),
        fill(false, 3), zeros(3), fill(false, 3), 0.0), workspace, plan, state)
    @test snapshot(state) == complete
    @test product_snapshot(product) == output_before
    @test process(plan, state).reference_centroids == [3.0, 14.0, 0.0, 0.0, -2.0, 44.0]
    @test !Base.mightalias(state.reference_centroids, LampReferenceState(plan).reference_centroids)
end

function update_allocations(state, plan, input)
    process!(state, plan, input)
    return @allocated process!(state, plan, input)
end
function extraction_allocations(product, workspace, plan, state)
    process!(product, workspace, plan, state)
    return @allocated process!(product, workspace, plan, state)
end

@testset "bounded state update and extraction resource checks" begin
    plan = prepare(DarkFrameMoments(), DarkFrameSpecification(2, 2, 3))
    for raw in (fill(UInt16(7), 2, 2), fill(UInt16(7), 2, 2, 1))
        state = DarkFrameState(plan)
        @test update_allocations(state, plan, raw) == 0
        process!(state, plan, raw)
        @test extraction_allocations(allocate_result(plan), allocate_workspace(plan), plan, state) == 0
    end
    lamp_plan = prepare(LampReference([5.0, 10.0, 4.0], UInt16(100)),
        LampReferenceSpecification(2, 2, 3, 3))
    for T in (Float32, Float64)
        state = LampReferenceState(lamp_plan)
        input = chunk(lamp_fixture(T), 1:1)
        @test update_allocations(state, lamp_plan, input) == 0
        process!(state, lamp_plan, input)
        @test extraction_allocations(allocate_result(lamp_plan),
            allocate_workspace(lamp_plan), lamp_plan, state) == 0
    end
end

end
