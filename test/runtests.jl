using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using Test

include("structured_interaction_matrices.jl")
include("probe_bases.jl")
include("spatial_sinusoidal.jl")
include("modal_bases.jl")
include("modal_opd_expansion.jl")
include("interaction_matrix_stability.jl")
include("repeated_response_moments.jl")
include("reconstructors.jl")
include("optical_reconstruction.jl")
include("misregistration.jl")
include("tomography.jl")
include("phase_retrieval.jl")
include("docrime.jl")
include("optical_gains.jl")
include("optical_gain_update.jl")
include("shack_hartmann.jl")
include("atmospheric_parameters.jl")
include("modal_gain_optimization.jl")
include("reference_frames.jl")
include("reference_frame_state.jl")
include("pyramid_reference.jl")
include("focal_plane_sharpening.jl")
include("kernel_abstractions.jl")

struct ScaleMethod{T<:Real} <: AbstractCalibrationMethod
    factor::T
end

struct ScaleSpecification
    length::Int
end

struct ScalePlan{T<:Real} <: AbstractCalibrationPlan
    factor::T
    length::Int
end

struct ScaleProduct{A<:AbstractVector}
    values::A
end

function AdaptiveOpticsCalibration.prepare(
    method::ScaleMethod,
    specification::ScaleSpecification,
)
    specification.length > 0 || throw(ArgumentError("length must be positive"))
    return ScalePlan(method.factor, specification.length)
end

function AdaptiveOpticsCalibration.allocate_workspace(plan::ScalePlan{T}) where {T}
    return Vector{T}(undef, plan.length)
end

function AdaptiveOpticsCalibration.allocate_result(plan::ScalePlan{T}) where {T}
    return ScaleProduct(Vector{T}(undef, plan.length))
end

function AdaptiveOpticsCalibration.process!(
    result::ScaleProduct,
    workspace::AbstractVector,
    plan::ScalePlan,
    inputs::AbstractVector,
)
    length(inputs) == plan.length || throw(DimensionMismatch("input length"))
    length(workspace) == plan.length || throw(DimensionMismatch("workspace length"))
    length(result.values) == plan.length || throw(DimensionMismatch("result length"))
    copyto!(workspace, inputs)
    @. result.values = plan.factor * workspace
    return result
end

@testset "prepared calibration interface" begin
    method = ScaleMethod(2.0)
    specification = ScaleSpecification(3)
    plan = prepare(method, specification)

    @test plan isa AbstractCalibrationPlan
    @test plan.factor == 2.0
    @test plan.length == 3

    result = process(plan, [1.0, -2.0, 3.5])

    @test result isa ScaleProduct
    @test result.values == [2.0, -4.0, 7.0]

    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    returned = process!(result, workspace, plan, [1.0, -2.0, 3.5])

    @test returned === result
    @test result.values == [2.0, -4.0, 7.0]

    second_product = allocate_result(plan)
    process!(second_product, plan, [3.0, 2.0, 1.0])
    @test second_product.values == [6.0, 4.0, 2.0]
    @test result.values == [2.0, -4.0, 7.0]
end

@testset "preparation and processing failures" begin
    @test_throws ArgumentError prepare(ScaleMethod(1.0), ScaleSpecification(0))

    plan = prepare(ScaleMethod(1.0), ScaleSpecification(2))
    workspace = allocate_workspace(plan)
    result = allocate_result(plan)

    @test_throws DimensionMismatch process!(result, workspace, plan, [1.0])
    @test_throws DimensionMismatch process!(result, zeros(1), plan, [1.0, 2.0])
    @test_throws DimensionMismatch process!(ScaleProduct(zeros(1)), workspace, plan,
        [1.0, 2.0])
end

@testset "public documentation" begin
    @test Base.isexported(AdaptiveOpticsCalibration, :prepare)
    @test Base.isexported(AdaptiveOpticsCalibration, :process)
    @test Base.isexported(AdaptiveOpticsCalibration, :process!)
    @test Base.ispublic(AdaptiveOpticsCalibration, :allocate_result)
    @test Base.ispublic(AdaptiveOpticsCalibration, :allocate_workspace)
    @test !Base.isexported(AdaptiveOpticsCalibration, :allocate_result)
    @test !Base.isexported(AdaptiveOpticsCalibration, :allocate_workspace)

    for module_ in (
        AdaptiveOpticsCalibration,
        AdaptiveOpticsCalibration.InteractionMatrices,
        AdaptiveOpticsCalibration.ProbeBases,
        AdaptiveOpticsCalibration.ModalBases,
        AdaptiveOpticsCalibration.Diagnostics,
        AdaptiveOpticsCalibration.Reconstructors,
        AdaptiveOpticsCalibration.Misregistration,
        AdaptiveOpticsCalibration.Tomography,
        AdaptiveOpticsCalibration.PhaseRetrieval,
        AdaptiveOpticsCalibration.DOCRIME,
        AdaptiveOpticsCalibration.OpticalGains,
        AdaptiveOpticsCalibration.ModalGainOptimization,
        AdaptiveOpticsCalibration.ReferenceFrames,
        AdaptiveOpticsCalibration.ShackHartmann,
        AdaptiveOpticsCalibration.AtmosphericParameters,
        AdaptiveOpticsCalibration.FocalPlaneSharpening,
    )
        for name in names(module_)
            @test Base.Docs.hasdoc(module_, name)
        end
    end
end

@testset "method ambiguities" begin
    @test isempty(Test.detect_ambiguities(AdaptiveOpticsCalibration; recursive=true))
end
