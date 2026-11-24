module OpticalGainTests

using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using AdaptiveOpticsCalibration.OpticalGains
using TOML
using Test

const FIXTURE = TOML.parsefile(joinpath(
    @__DIR__, "fixtures", "aos_s6_gsc_cpu.toml"))

struct NonOneBasedGainSensingMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    parent::M
end

Base.size(array::NonOneBasedGainSensingMatrix) = size(array.parent)
Base.axes(array::NonOneBasedGainSensingMatrix) = (
    0:(size(array.parent, 1) - 1),
    0:(size(array.parent, 2) - 1),
)
Base.IndexStyle(::Type{<:NonOneBasedGainSensingMatrix}) = IndexCartesian()
Base.getindex(array::NonOneBasedGainSensingMatrix, row::Int, column::Int) =
    array.parent[row + 1, column + 1]

function fixture_array(section)
    return reshape(Float64.(section["values"]), Tuple(Int.(section["shape"])))
end

function fixture_complex_mask()
    real = fixture_array(FIXTURE["mask"]["real"])
    imag = fixture_array(FIXTURE["mask"]["imag"])
    return complex.(real, imag)
end

function fixture_sensitivities(name)
    values = FIXTURE["sensitivities"][name]
    return complex.(Float64.(values["real"]), Float64.(values["imag"]))
end

function fixture_specification()
    return GainSensingSpecification(
        fixture_complex_mask(),
        fixture_array(FIXTURE["basis"]),
        fixture_array(FIXTURE["frames"]["reference"]),
    )
end

function fixture_process_allocation(product, workspace, plan, input)
    process!(product, workspace, plan, input)
    return @allocated process!(product, workspace, plan, input)
end

@testset "AOS S6 gain-sensing source fixture" begin
    @test FIXTURE["schema"] == "test.adaptive-optics-sim/gain-sensing-camera-cpu/1"
    @test FIXTURE["source_repository"] == "AdaptiveOpticsSim.jl"
    @test FIXTURE["source_revision"] ==
        "4dec9c2469cd6f6c3d0a6dfe275ca777901c51c5"
    @test FIXTURE["source_paths"] == ["src/calibration/gain_sensing_camera.jl"]
    @test FIXTURE["source_test_path"] == "test/testsets/optical_analysis.jl"
    @test FIXTURE["array_backend"] == "CPU"
    @test FIXTURE["storage_order"] == "column-major"
    @test FIXTURE["fft_centering"] == "fftshift(fft(fftshift(x))) / N"
    @test FIXTURE["optical_gains_unit"] == "dimensionless signed ratio"
    @test FIXTURE["physical_branch_reference"]["current_image_shape"] == [48, 48]

    specification = fixture_specification()
    method = GainSensing(FIXTURE["sensitivity_floor"])
    plan = @inferred prepare(method, specification)
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    input = GainSensingInputs(fixture_array(FIXTURE["frames"]["current"]))

    @test plan.mask !== specification.mask
    @test !hasproperty(plan, :normalized_reference_frame)
    @test !Base.mightalias(plan.mask, specification.mask)
    @test plan.side == 8
    @test plan.mode_count == 3
    @test plan.sample_count == 64
    @test plan.sensitivity_floor == FIXTURE["sensitivity_floor"]
    @test plan.reference_total_flux ≈ FIXTURE["frames"]["reference"]["total_flux"]
    @test plan.normalized_reference_total_flux ≈
        FIXTURE["frames"]["reference"]["normalized_total"] atol=1e-14
    @test plan.reference_impulse_response ≈
        fixture_array(FIXTURE["impulse_responses"]["reference"]) rtol=1e-12 atol=1e-22
    @test plan.reference_sensitivities ≈ fixture_sensitivities("reference") rtol=1e-12 atol=1e-20
    @test plan.weak_mode_mask == Bool.(FIXTURE["weak_mode_mask"])
    @test any(.!plan.weak_mode_mask)
    @test any(plan.weak_mode_mask)

    @test @inferred(process(plan, input)) isa GainSensingProduct
    @test @inferred(process!(product, workspace, plan, input)) === product
    @test optical_gains(product) === product.optical_gains
    @test current_sensitivities(product) === product.current_sensitivities
    @test weak_mode_mask(product) === product.weak_mode_mask
    @test flux_diagnostics(product) === product.diagnostics
    @test product.current_sensitivities ≈ fixture_sensitivities("current") rtol=1e-12 atol=1e-20
    @test product.optical_gains ≈ Float64.(FIXTURE["optical_gains"]) rtol=1e-12 atol=1e-14
    @test product.weak_mode_mask == Bool.(FIXTURE["weak_mode_mask"])
    @test product.diagnostics.current_total_flux ≈
        FIXTURE["frames"]["current"]["total_flux"]
    @test product.diagnostics.normalized_current_total_flux ≈
        FIXTURE["frames"]["current"]["normalized_total"] atol=1e-14
    @test fixture_process_allocation(product, workspace, plan, input) == 0
end

@testset "gain-sensing validation and transactional publication" begin
    specification = fixture_specification()
    plan = prepare(GainSensing(FIXTURE["sensitivity_floor"]), specification)
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    current = fixture_array(FIXTURE["frames"]["current"])
    process!(product, workspace, plan, GainSensingInputs(current))
    before = (
        copy(product.optical_gains),
        copy(product.current_sensitivities),
        copy(product.weak_mode_mask),
        product.diagnostics,
    )

    @test_throws ArgumentError process!(product, workspace, plan,
        GainSensingInputs(fill(NaN, size(current))))
    @test_throws ArgumentError process!(product, workspace, plan,
        GainSensingInputs(zeros(size(current))))
    @test_throws ArgumentError process!(product, workspace, plan,
        GainSensingInputs(fill(-one(eltype(current)), size(current))))
    @test (product.optical_gains, product.current_sensitivities,
        product.weak_mode_mask, product.diagnostics) == before

    negative_sample = copy(current)
    negative_sample[1] = -one(eltype(negative_sample))
    @test process!(product, workspace, plan,
        GainSensingInputs(negative_sample)) === product
    @test product.diagnostics.current_total_flux > 0

    # Independent AOS-equation calculation for a unit detector sample at (1, 1):
    # the two active source sensitivities have signed ratios -4.147768508503635
    # and -9.572789888214944. The zero modal basis remains weak and returns one.
    negative_gain_probe = zeros(size(current))
    negative_gain_probe[1, 1] = 1.0
    @test process!(product, workspace, plan,
        GainSensingInputs(negative_gain_probe)) === product
    @test product.optical_gains ≈
        [-4.147768508503635, -9.572789888214944, 1.0] rtol=1e-12 atol=1e-14
    @test product.weak_mode_mask == [false, false, true]

    @test_throws ArgumentError prepare(GainSensing(), GainSensingSpecification(
        fixture_complex_mask(), fixture_array(FIXTURE["basis"]),
        zeros(size(current)),
    ))
    @test_throws ArgumentError prepare(GainSensing(), GainSensingSpecification(
        NonOneBasedGainSensingMatrix(fixture_complex_mask()),
        fixture_array(FIXTURE["basis"]), fixture_array(FIXTURE["frames"]["reference"]),
    ))
    @test_throws DimensionMismatch process!(product, workspace, plan,
        GainSensingInputs(ones(2, 2)))

    detached_buffer_workspace = GainSensingWorkspace(
        workspace.buffer,
        copy(workspace.buffer_vector),
        workspace.forward,
        workspace.inverse,
        workspace.normalized_current_frame,
        workspace.current_impulse_response,
        workspace.current_sensitivities,
        workspace.candidate_gains,
        workspace.candidate_weak_mode_mask,
        workspace.fft_plan,
        workspace.ifft_plan,
    )
    @test_throws ArgumentError process!(product, detached_buffer_workspace, plan,
        GainSensingInputs(current))

    short_buffer_workspace = GainSensingWorkspace(
        workspace.buffer,
        zeros(eltype(workspace.buffer_vector), plan.sample_count - 1),
        workspace.forward,
        workspace.inverse,
        workspace.normalized_current_frame,
        workspace.current_impulse_response,
        workspace.current_sensitivities,
        workspace.candidate_gains,
        workspace.candidate_weak_mode_mask,
        workspace.fft_plan,
        workspace.ifft_plan,
    )
    @test_throws DimensionMismatch process!(product, short_buffer_workspace, plan,
        GainSensingInputs(current))
end

@testset "gain-sensing one-based CPU views and Float32" begin
    mask_storage = fixture_complex_mask()
    basis_storage = fixture_array(FIXTURE["basis"])
    reference_storage = fixture_array(FIXTURE["frames"]["reference"])
    current_storage = fixture_array(FIXTURE["frames"]["current"])
    specification = GainSensingSpecification(
        @view(mask_storage[:, :]),
        @view(basis_storage[:, :, :]),
        @view(reference_storage[:, :]),
    )
    plan = @inferred prepare(GainSensing(FIXTURE["sensitivity_floor"]), specification)
    product = allocate_result(plan)
    workspace = allocate_workspace(plan)
    view_input = GainSensingInputs(@view(current_storage[:, :]))
    @test process!(product, workspace, plan, view_input) === product
    @test product.optical_gains ≈ Float64.(FIXTURE["optical_gains"]) rtol=1e-12
    @test fixture_process_allocation(product, workspace, plan, view_input) == 0

    mask32 = ComplexF32.(mask_storage)
    basis32 = Float32.(basis_storage)
    reference32 = Float32.(reference_storage)
    current32 = Float32.(current_storage)
    default_plan32 = @inferred prepare(GainSensing(), GainSensingSpecification(
        mask32, basis32, reference32,
    ))
    default_product32 = process(default_plan32, GainSensingInputs(current32))
    @test all(default_product32.weak_mode_mask)
    @test default_product32.optical_gains == ones(Float32, 3)

    plan32 = @inferred prepare(
        GainSensing(Float32(FIXTURE["sensitivity_floor"])),
        GainSensingSpecification(
            mask32, basis32, reference32,
        ),
    )
    product32 = allocate_result(plan32)
    workspace32 = allocate_workspace(plan32)
    input32 = GainSensingInputs(current32)
    @test @inferred(process!(product32, workspace32, plan32, input32)) === product32
    @test eltype(product32.optical_gains) === Float32
    @test all(isfinite, product32.optical_gains)
    @test product32.optical_gains ≈ Float32.(FIXTURE["optical_gains"]) rtol=2e-5
    @test fixture_process_allocation(product32, workspace32, plan32, input32) == 0
end

@testset "gain-sensing owner separation" begin
    plan = prepare(GainSensing(FIXTURE["sensitivity_floor"]), fixture_specification())
    workspace = allocate_workspace(plan)
    input = GainSensingInputs(fixture_array(FIXTURE["frames"]["current"]))
    product = allocate_result(plan)

    aliased_reference = GainSensingProduct(
        product.optical_gains, plan.reference_sensitivities,
        product.weak_mode_mask, product.diagnostics)
    reference_before = copy(plan.reference_sensitivities)
    @test_throws ArgumentError process!(aliased_reference, workspace, plan, input)
    @test plan.reference_sensitivities == reference_before

    aliased_weak_mask = GainSensingProduct(
        product.optical_gains, product.current_sensitivities,
        plan.weak_mode_mask, product.diagnostics)
    @test_throws ArgumentError process!(aliased_weak_mask, workspace, plan, input)

    aliased_plan_workspace = GainSensingWorkspace(
        workspace.buffer, workspace.buffer_vector, plan.mask_fft,
        workspace.inverse, workspace.normalized_current_frame,
        workspace.current_impulse_response, workspace.current_sensitivities,
        workspace.candidate_gains, workspace.candidate_weak_mode_mask,
        workspace.fft_plan, workspace.ifft_plan)
    @test_throws ArgumentError process!(product, aliased_plan_workspace, plan, input)
    @test_throws ArgumentError process!(product, workspace, plan,
        GainSensingInputs(@view workspace.normalized_current_frame[:, :]))

    gain_view = @view workspace.normalized_current_frame[1:plan.mode_count]
    fill!(gain_view, 23.0)
    aliased_workspace = GainSensingProduct(
        gain_view, product.current_sensitivities,
        product.weak_mode_mask, product.diagnostics)
    tiny_frame = zeros(plan.side, plan.side)
    tiny_frame[1, 1] = nextfloat(0.0)
    @test_throws ArgumentError process!(aliased_workspace, workspace, plan,
        GainSensingInputs(tiny_frame))
    @test all(==(23.0), gain_view)
end

@testset "modal dither response over fractional cycles with DC offsets" begin
    sample_rate = 128.0
    samples = 0:46
    frequencies = [5.2, 9.7]
    frame_times = 4096 .+ samples ./ sample_rate
    command_phases = [0.37, -0.82]
    command_amplitudes = [0.75, 1.25]
    gains = [1.5, 0.35]
    phase_lags = [0.41, -0.79]
    angles = [2π * (f / sample_rate) * n for n in samples, f in frequencies]
    commands = [(-1)^i * 2.0^25 + command_amplitudes[i] * cos(angles[n,i] - command_phases[i])
        for n in axes(angles, 1), i in eachindex(frequencies)]
    responses = [(-1)^(i+1) * 2.0^24 + gains[i] * command_amplitudes[i] *
        cos(angles[n,i] - command_phases[i] - phase_lags[i])
        for n in axes(angles, 1), i in eachindex(frequencies)]
    expected_ratios = gains .* cis.(-phase_lags)

    # Direct mixing has finite-window DC leakage at these two carrier frequencies.
    direct_ratios = [sum(responses[:, i] .* cis.(-angles[:, i])) /
        sum(commands[:, i] .* cis.(-angles[:, i])) for i in eachindex(frequencies)]
    @test all(abs.(direct_ratios .- expected_ratios) .> 0.2)

    method = ModalDitherResponse(; command_amplitude_floor=0.05)
    spec = ModalDitherResponseSpecification(frequencies, frame_times; sample_rate)
    plan = @inferred prepare(method, spec)
    inputs = ModalDitherResponseInputs(commands, responses)
    result = @inferred process(plan, inputs)
    # Quantization at the large offsets limits the accuracy of the input samples.
    @test result.command_amplitudes ≈ command_amplitudes atol=3e-9
    @test result.response_amplitudes ≈ gains .* command_amplitudes atol=3e-9
    @test result.response_ratios ≈ expected_ratios atol=3e-9
    @test result.gains ≈ gains atol=3e-9
    @test result.phase_lags ≈ phase_lags atol=3e-9
    @test result.valid == [true, true]

    command_storage = zeros(2 * size(commands, 1), size(commands, 2))
    response_storage = similar(command_storage)
    command_view = @view command_storage[1:2:end, :]
    response_view = @view response_storage[2:2:end, :]
    command_view .= commands
    response_view .= responses
    view_inputs = ModalDitherResponseInputs(command_view, response_view)
    reused = allocate_result(plan)
    workspace = allocate_workspace(plan)
    @test @inferred(process!(reused, workspace, plan, view_inputs)) === reused
    @test reused.response_ratios == result.response_ratios
    @test process!(reused, plan, view_inputs).phase_lags == result.phase_lags
    @test fixture_process_allocation(reused, workspace, plan, inputs) == 0
    @test fixture_process_allocation(reused, workspace, plan, view_inputs) == 0
end

@testset "modal dither FGA phase and amplitude convention" begin
    samples = 0:23
    periods = [8, 12]
    sample_rate = 96.0
    angles = [2π * (n % period) / period for n in samples, period in periods]
    command_phases = [2.8, -2.8]
    phase_lags = [0.7, -0.7]
    commands = [0.5 + cos(angles[n,i] - command_phases[i])
        for n in axes(angles, 1), i in eachindex(periods)]
    responses = [1.0 + 0.75cos(angles[n,i] - command_phases[i] - phase_lags[i])
        for n in axes(angles, 1), i in eachindex(periods)]

    # FGA estimator.jl quadratures and subtraction sign for a unity
    # calibration response and complete integer-period windows.
    scale = 2 / length(samples)
    fga_gains = Float64[]
    fga_lags = Float64[]
    for i in eachindex(periods)
        command_cosine = scale * sum(commands[:, i] .* cos.(angles[:, i]))
        command_sine = scale * sum(commands[:, i] .* sin.(angles[:, i]))
        response_cosine = scale * sum(responses[:, i] .* cos.(angles[:, i]))
        response_sine = scale * sum(responses[:, i] .* sin.(angles[:, i]))
        push!(fga_gains, hypot(response_cosine, response_sine) /
            hypot(command_cosine, command_sine))
        phase = atan(response_sine, response_cosine) - atan(command_sine, command_cosine)
        push!(fga_lags, rem2pi(phase, RoundNearest))
    end
    plan = prepare(
        ModalDitherResponse(; command_amplitude_floor=0),
        ModalDitherResponseSpecification(sample_rate ./ periods,
            range(0; step=inv(sample_rate), length=length(samples)); sample_rate),
    )
    result = process(plan, ModalDitherResponseInputs(commands, responses))
    @test result.gains ≈ fga_gains atol=1e-14
    @test result.phase_lags ≈ fga_lags atol=1e-14
    @test result.phase_lags ≈ phase_lags atol=1e-14
    @test result.response_ratios ≈ fga_gains .* cis.(-fga_lags) atol=1e-14
end

@testset "modal dither window and observability checks" begin
    method = ModalDitherResponse(; command_amplitude_floor=0.1)
    sample_rate = 16.0
    frame_times = collect(0:7) ./ sample_rate
    spec(frequencies, times=frame_times; sample_rate=sample_rate) =
        ModalDitherResponseSpecification(frequencies, times; sample_rate)

    @test_throws ArgumentError ModalDitherResponse(; command_amplitude_floor=NaN)
    @test_throws ArgumentError ModalDitherResponse(; command_amplitude_floor=-0.1)
    @test_throws ArgumentError ModalDitherResponse(;
        command_amplitude_floor=0.1, relative_rank_tolerance=0)
    @test_throws ArgumentError prepare(method, spec([2.0], frame_times[1:2]))
    @test_throws ArgumentError prepare(method, spec(Float64[]))
    @test_throws ArgumentError prepare(method, spec([2.0]; sample_rate=Inf))
    @test_throws ArgumentError prepare(method, spec([2.0]; sample_rate=17.0))
    @test_throws ArgumentError prepare(method, spec([2.0], frame_times[[1, 2, 2, 4]]))
    @test_throws ArgumentError prepare(method, spec([2.0], [NaN; frame_times[2:end]]))
    rounded_times = Float32.(collect(0:63) ./ 1000)
    rounded_plan = prepare(method, spec([70.0], rounded_times; sample_rate=1000))
    exact_plan = prepare(method, spec([70.0], collect(0:63) ./ 1000; sample_rate=1000))
    @test rounded_plan.projection == exact_plan.projection
    rounded_times[16] += 0.00025f0
    @test_throws ArgumentError prepare(method, spec([70.0], rounded_times; sample_rate=1000))
    for frequency in (-1.0, 0.0, 1e-6, prevfloat(sample_rate / 2), sample_rate / 2, 10.0, NaN)
        @test_throws ArgumentError prepare(method, spec([frequency]))
    end

    # The same carrier is admissible for isolated modal coordinates.
    plan = prepare(method, spec(fill(2.0, 4)))
    carrier = cospi.((0:7) ./ 4)
    commands = hcat(carrier, fill(2.0^50, 8), 0.0625 .* carrier, carrier)
    responses = hcat(2 .* carrier, carrier, carrier, ones(8))
    inputs = ModalDitherResponseInputs(commands, responses)
    product = process(plan, inputs)
    workspace = allocate_workspace(plan)
    @test product.valid == [true, false, false, false]
    @test product.command_amplitudes ≈ [1.0, 0.0, 0.0625, 1.0] atol=1e-14
    @test product.response_amplitudes ≈ [2.0, 1.0, 1.0, 0.0] atol=1e-14
    @test product.response_ratios[2:end] == zeros(ComplexF64, 3)
    @test product.gains[2:end] == zeros(3)
    @test product.phase_lags[2:end] == zeros(3)

    @test_throws DimensionMismatch process!(product, workspace, plan,
        ModalDitherResponseInputs(commands[1:7, :], responses))
    @test_throws ArgumentError process!(product, workspace, plan,
        ModalDitherResponseInputs(NonOneBasedGainSensingMatrix(commands), responses))
    @test_throws ArgumentError process!(product, product, plan, inputs)
    invalid_responses = copy(responses)
    invalid_responses[end, end] = Inf
    @test_throws ArgumentError process!(product, workspace, plan,
        ModalDitherResponseInputs(2 .* commands, invalid_responses))
    @test product.command_amplitudes ≈ [1.0, 0.0, 0.0625, 1.0] atol=1e-14
    @test product.valid == [true, false, false, false]
end

end
