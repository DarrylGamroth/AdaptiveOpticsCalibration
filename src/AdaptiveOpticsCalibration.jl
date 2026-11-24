"""
    AdaptiveOpticsCalibration

Reusable numerical algorithms for adaptive-optics calibration.

The package owns scientific calibration methods, their run-immutable plans,
replaceable workspaces, caller-visible products, and numerical diagnostics. It
does not own plant propagation, real-time graph execution, equipment control,
transport, artifact persistence, acceptance policy, or activation.
"""
module AdaptiveOpticsCalibration

include("interface.jl")
include("utilities.jl")
include("interaction_matrices/interaction_matrices.jl")
include("probe_bases/probe_bases.jl")
include("modal_bases/modal_bases.jl")
include("diagnostics/diagnostics.jl")
include("reconstructors/reconstructors.jl")
include("misregistration/misregistration.jl")
include("tomography/tomography.jl")
include("phase_retrieval/phase_retrieval.jl")
include("docrime/docrime.jl")
include("optical_gains/optical_gains.jl")
include("modal_gain_optimization/modal_gain_optimization.jl")
include("reference_frames/reference_frames.jl")
include("shack_hartmann/shack_hartmann.jl")
include("atmospheric_parameters/atmospheric_parameters.jl")
include("focal_plane_sharpening/focal_plane_sharpening.jl")

export AbstractCalibrationMethod, AbstractCalibrationPlan, KernelExecution
export prepare, process, process!
public allocate_result, allocate_workspace
export InteractionMatrices, ProbeBases, ModalBases, Diagnostics
export Reconstructors, Misregistration, PhaseRetrieval
export Tomography
export DOCRIME
export OpticalGains
export ModalGainOptimization
export ReferenceFrames
export ShackHartmann
export AtmosphericParameters
export FocalPlaneSharpening

end
