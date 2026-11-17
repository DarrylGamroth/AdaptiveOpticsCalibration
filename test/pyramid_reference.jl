module PyramidReferenceTests

using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.ReferenceFrames
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using Test
import ..ReferenceFrameTests: ShiftedCube

function fixture()
    base = Float32[1 2 3; 4 5 6; 7 8 9; 10 11 12]
    pixels = cat(base .- 2, base .+ 1, base .+ 4; dims=3)
    raw = reshape(UInt16.(1:18), 2, 3, 3)
    R = [1.0 10 100 1000 0 0 0 0 0 0 0 0;
         0.0 0 0 0 0 0 3 0 0 0 0 -2]
    spec = PyramidReferenceSpecification(2, 3, 3, 3, R; row_units=["m", "rad"])
    return prepare(PyramidReference(UInt16(100)), spec),
        PyramidReferenceInputs(pixels, [1.0, 10.0, 1000.0], raw)
end

chunk(input, exposures) = PyramidReferenceInputs(
    @view(input.reconstruction_pixels[:, :, exposures]),
    @view(input.mean_pupil_intensity[exposures]),
    @view(input.raw_frames[:, :, exposures]))
snapshot(p::PyramidReferenceProduct) = deepcopy((p.status, p.reference_pixels,
    p.modal_reference, p.adc_maximum, p.minimum_mean_pupil_intensity))
snapshot(s::PyramidReferenceState) = deepcopy((s.count, s.reference_pixels,
    s.adc_maximum, s.minimum_mean_pupil_intensity, s.finite, s.positive_mean, s.below_rail))

@testset "pyramid independent pupil-major mean, sign and declared units" begin
    plan, input = fixture()
    raw_before = copy(input.raw_frames)
    pixel_before = copy(input.reconstruction_pixels)
    mean_before = copy(input.mean_pupil_intensity)
    product = @inferred process(plan, input)
    # Hand-calculated means: each entry averages b-2, b+1, b+4 to b+1.
    expected = [2.0 3 4; 5 6 7; 8 9 10; 11 12 13]
    @test product.status === Valid
    @test product.reference_pixels == expected
    # Row 1: -(2 + 10*3 + 100*4 + 1000*5); row 2: -(3*8 - 2*13).
    @test product.modal_reference == [-5432.0, 2.0]
    @test plan.row_units == ("m", "rad")
    @test product.adc_maximum == 18.0
    @test product.minimum_mean_pupil_intensity == 1.0
    @test input.raw_frames == raw_before
    @test input.reconstruction_pixels == pixel_before
    @test input.mean_pupil_intensity == mean_before
    @test eltype(product.reference_pixels) === Float64
    @test eltype(product.modal_reference) === Float64
    # Permuting selected positions without changing R changes the scientific result.
    reordered = PyramidReferenceInputs(input.reconstruction_pixels[:, [3, 2, 1], :],
        input.mean_pupil_intensity, input.raw_frames)
    @test process(plan, reordered).modal_reference[1] == -7234.0
    # A different declared unit label does not convert numerical coordinates.
    relabeled = prepare(PyramidReference(100), PyramidReferenceSpecification(
        2, 3, 3, 3, plan.reconstructor; row_units=["nm", "dimensionless"]))
    @test process(relabeled, input).modal_reference == product.modal_reference
end

@testset "pyramid batch, chunk and single-exposure equivalence" begin
    plan, input = fixture()
    expected = snapshot(process(plan, input))
    for partitions in ((1:3,), (1:1, 2:3), (1:2, 3:3), (1:1, 2:2, 3:3))
        state = @inferred PyramidReferenceState(plan)
        @test state.plan === plan
        @test state.plan.reconstructor === plan.reconstructor
        @test state.count == 0
        @test iszero(state.reference_pixels)
        for exposures in partitions
            @test @inferred(process!(state, plan, chunk(input, exposures))) === state
        end
        before = snapshot(state)
        product = @inferred process(plan, state)
        @test snapshot(product) == expected
        @test snapshot(state) == before
        product.reference_pixels .= 999
        product.modal_reference .= 999
        @test snapshot(state) == before
        workspace, result = allocate_workspace(plan), allocate_result(plan)
        @test process!(result, workspace, plan, state) === result
        @test snapshot(result) == expected
        @test snapshot(process!(result, plan, state)) == expected
    end
    # Noncontiguous views retain logical axes and exposure ordering.
    padded = zeros(Float32, 8, 6, 6)
    padded[1:2:8, 1:2:6, 1:2:6] .= input.reconstruction_pixels
    viewed = PyramidReferenceInputs(@view(padded[1:2:8, 1:2:6, 1:2:6]),
        input.mean_pupil_intensity, input.raw_frames)
    @test snapshot(process(plan, viewed)) == expected
    reversed = PyramidReferenceInputs(input.reconstruction_pixels[:, :, 3:-1:1],
        input.mean_pupil_intensity[3:-1:1], input.raw_frames[:, :, 3:-1:1])
    @test snapshot(process(plan, reversed)) == expected
    # Smallest valid contract; zero normalized pixels are not a brightness policy.
    single = prepare(PyramidReference(2), PyramidReferenceSpecification(
        1, 1, 1, 1, ones(1, 4); row_units=["m"]))
    zeros_input = PyramidReferenceInputs(zeros(Float32, 4, 1, 1), [1.0], zeros(UInt8, 1, 1, 1))
    @test process(single, zeros_input).status === Valid
    @test iszero(process(single, zeros_input).modal_reference)
    nonzero = PyramidReferenceInputs(reshape(Float32[1, 2, 3, 4], 4, 1, 1),
        [1.0], zeros(UInt8, 1, 1, 1))
    @test process(single, nonzero).modal_reference == [-10.0]
end

@testset "pyramid sticky scientific violations and complete diagnostics" begin
    for case in (:nan_pixel, :inf_pixel, :negative_inf_pixel, :overflow_mean,
                 :nan_mean, :inf_mean, :negative_inf_mean, :zero_mean,
                 :negative_mean, :rail_equal, :rail_above)
        plan, input = fixture()
        pixels = Float64.(input.reconstruction_pixels)
        means, raw = copy(input.mean_pupil_intensity), copy(input.raw_frames)
        case == :nan_pixel && (pixels[1, 1, 1] = NaN)
        case == :inf_pixel && (pixels[1, 1, 1] = Inf)
        case == :negative_inf_pixel && (pixels[1, 1, 1] = -Inf)
        case == :overflow_mean && (pixels[1, 1, :] .= [floatmax(Float64), -floatmax(Float64), 1.0])
        case == :nan_mean && (means[1] = NaN)
        case == :inf_mean && (means[1] = Inf)
        case == :negative_inf_mean && (means[1] = -Inf)
        case == :zero_mean && (means[1] = 0.0)
        case == :negative_mean && (means[1] = -1.0)
        case == :rail_equal && (raw[1, 1, 1] = 100)
        case == :rail_above && (raw[1, 1, 1] = 101)
        bad = PyramidReferenceInputs(pixels, means, raw)
        batch = process(plan, bad)
        @test batch.status === Invalid
        @test iszero(batch.reference_pixels)
        @test iszero(batch.modal_reference)
        @test batch.adc_maximum == (case == :rail_equal ? 100.0 : case == :rail_above ? 101.0 : 18.0)
        @test isequal(batch.minimum_mean_pupil_intensity, minimum(means))
        for partitions in ((1:1, 2:3), (1:2, 3:3), (1:1, 2:2, 3:3))
            state = PyramidReferenceState(plan)
            for exposures in partitions
                process!(state, plan, chunk(bad, exposures))
            end
            @test state.count == 3
            @test isequal(snapshot(process(plan, state)), snapshot(batch))
        end
        # Reused workspace resets after every invalid complete batch.
        result, workspace = allocate_result(plan), allocate_workspace(plan)
        process!(result, workspace, plan, bad)
        process!(result, workspace, plan, input)
        @test snapshot(result) == snapshot(process(plan, input))
    end
    # A finite R/input may overflow during the modal product, which invalidates all arrays.
    plan = prepare(PyramidReference(2), PyramidReferenceSpecification(
        1, 1, 1, 1, fill(floatmax(Float64), 1, 4); row_units=["m"]))
    input = PyramidReferenceInputs(fill(2.0, 4, 1, 1), [1.0], zeros(UInt8, 1, 1, 1))
    @test process(plan, input).status === Invalid
    state = PyramidReferenceState(plan)
    process!(state, plan, input)
    @test process(plan, state).status === Invalid
    # UInt64 values differ above Float64's consecutive-integer range: compare raw ADC exactly.
    rail = UInt64(2)^60 + 1
    exact = prepare(PyramidReference(rail), PyramidReferenceSpecification(
        1, 1, 1, 1, ones(1, 4); row_units=["m"]))
    for (raw, status) in ((rail-1, Valid), (rail, Invalid), (rail+1, Invalid))
        @test process(exact, PyramidReferenceInputs(ones(4, 1, 1), [1.0],
            fill(raw, 1, 1, 1))).status === status
    end
end

@testset "pyramid preparation and coefficient ownership" begin
    for rail in (0, -1, NaN, Inf)
        @test_throws ArgumentError PyramidReference(rail)
    end
    for dims in ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0))
        @test_throws ArgumentError PyramidReferenceSpecification(dims..., ones(1, 4); row_units=["m"])
    end
    @test_throws ArgumentError PyramidReferenceSpecification(typemax(Int), 2, 1, 1,
        ones(1, 4); row_units=["m"])
    for R in (ones(1, 3), zeros(0, 4))
        @test_throws DimensionMismatch PyramidReferenceSpecification(1, 1, 1, 1, R; row_units=["m"])
    end
    @test_throws DimensionMismatch PyramidReferenceSpecification(1, 1, 1, 1,
        ones(1, 4); row_units=String[])
    @test_throws ArgumentError PyramidReferenceSpecification(1, 1, 1, 1,
        ones(1, 4); row_units=[" "])
    for value in (NaN, Inf, -Inf, big(10)^1000)
        @test_throws ArgumentError prepare(PyramidReference(2),
            PyramidReferenceSpecification(1, 1, 1, 1, fill(value, 1, 4); row_units=["m"]))
    end
    R, units = ones(1, 4), ["m"]
    spec = PyramidReferenceSpecification(1, 1, 1, 1, R; row_units=units)
    units[1] = "rad"
    plan = prepare(PyramidReference(2), spec)
    R .= 999
    @test plan.reconstructor == ones(1, 4)
    @test plan.row_units == ("m",)
    @test plan.reconstructor !== spec.reconstructor
    state = PyramidReferenceState(plan)
    input = PyramidReferenceInputs(ones(4, 1, 1), [1.0], zeros(UInt8, 1, 1, 1))
    process!(state, plan, input)
    before = snapshot(state)
    input.reconstruction_pixels .= 999
    input.mean_pupil_intensity .= 999
    input.raw_frames .= 255
    @test snapshot(state) == before
    @test process(plan, state).modal_reference == [-4.0]
    @test PyramidReferenceState(plan).count == 0
end

@testset "pyramid rejection before writes and state boundaries" begin
    plan, input = fixture()
    state = PyramidReferenceState(plan)
    before = snapshot(state)
    malformed = (
        PyramidReferenceInputs(zeros(3, 3, 1), ones(1), zeros(UInt16, 2, 3, 1)),
        PyramidReferenceInputs(zeros(4, 2, 1), ones(1), zeros(UInt16, 2, 3, 1)),
        PyramidReferenceInputs(zeros(4, 3, 2), ones(1), zeros(UInt16, 2, 3, 1)),
        PyramidReferenceInputs(zeros(4, 3, 1), ones(2), zeros(UInt16, 2, 3, 1)),
        PyramidReferenceInputs(zeros(4, 3, 1), ones(1), zeros(UInt16, 3, 2, 1)),
    )
    for bad in malformed
        @test_throws DimensionMismatch process!(state, plan, bad)
        @test snapshot(state) == before
    end
    @test_throws ArgumentError process!(state, plan, chunk(input, 1:0))
    @test_throws ArgumentError process!(state, plan, PyramidReferenceInputs(
        input.reconstruction_pixels, input.mean_pupil_intensity, ShiftedCube(input.raw_frames)))
    other, _ = fixture()
    @test_throws ArgumentError process!(state, other, chunk(input, 1:1))
    aliased = PyramidReferenceInputs(reshape(state.reference_pixels, 4, 3, 1),
        [1.0], zeros(UInt16, 2, 3, 1))
    @test_throws ArgumentError process!(state, plan, aliased)
    # A separate compatible geometry exercises unsigned reinterpretation alias detection.
    alias_plan = prepare(PyramidReference(100), PyramidReferenceSpecification(
        2, 6, 3, 1, plan.reconstructor; row_units=["m", "rad"]))
    alias_state = PyramidReferenceState(alias_plan)
    raw_alias = reshape(reinterpret(UInt64, vec(alias_state.reference_pixels)), 2, 6, 1)
    @test_throws ArgumentError process!(alias_state, alias_plan,
        PyramidReferenceInputs(zeros(4, 3, 1), [1.0], raw_alias))
    @test snapshot(state) == before
    result, workspace = allocate_result(plan), allocate_workspace(plan)
    result.reference_pixels .= 42
    result_before = snapshot(result)
    @test_throws ArgumentError process!(result, workspace, plan, state)
    @test_throws DimensionMismatch process!(result, workspace, plan, chunk(input, 1:1))
    @test snapshot(result) == result_before
    process!(state, plan, input)
    complete = snapshot(state)
    @test_throws ArgumentError process!(state, plan, chunk(input, 1:1))
    @test snapshot(state) == complete
    @test_throws ArgumentError process!(result, PyramidReferenceWorkspace(state.reference_pixels), plan, state)
    @test_throws ArgumentError process!(PyramidReferenceProduct(Invalid,
        state.reference_pixels, zeros(2), 0, 0), workspace, plan, state)
    @test_throws DimensionMismatch process!(PyramidReferenceProduct(Invalid,
        zeros(4, 2), zeros(2), 0, 0), workspace, plan, state)
    @test_throws ArgumentError process!(result, PyramidReferenceWorkspace(result.reference_pixels), plan, input)
    # A prepared coefficient matrix and input pixels cannot become mutable scratch.
    alias_plan = prepare(PyramidReference(100), PyramidReferenceSpecification(
        2, 3, 3, 1, ones(1, 12); row_units=["m"]))
    alias_input = chunk(input, 1:1)
    alias_product = PyramidReferenceProduct(Invalid,
        reshape(alias_plan.reconstructor, 4, 3), zeros(1), 0, 0)
    @test_throws ArgumentError process!(alias_product, allocate_workspace(alias_plan),
        alias_plan, alias_input)
    pixel_alias_product = PyramidReferenceProduct(Invalid,
        reshape(Float64.(alias_input.reconstruction_pixels), 4, 3), zeros(1), 0, 0)
    pixel_alias_input = PyramidReferenceInputs(reshape(pixel_alias_product.reference_pixels, 4, 3, 1),
        [1.0], zeros(UInt16, 2, 3, 1))
    @test_throws ArgumentError process!(pixel_alias_product, allocate_workspace(alias_plan),
        alias_plan, pixel_alias_input)
    @test snapshot(state) == complete
    @test snapshot(result) == result_before
end

end
