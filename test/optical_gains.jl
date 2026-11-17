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

end
