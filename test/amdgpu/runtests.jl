using AMDGPU
using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using KernelAbstractions
using LinearAlgebra
using Test

AMDGPU.functional() || error("AMDGPU is not functional on this host")
AMDGPU.allowscalar(false)

include(joinpath(@__DIR__, "..", "gpu", "common.jl"))

selected_domain = isempty(ARGS) ? nothing : length(ARGS) == 1 ? ARGS[1] :
    error("Select at most one accelerator test domain: " *
        join(ACCELERATOR_TEST_DOMAINS, ", "))

run_accelerator_tests(
    AMDGPU.ROCBackend(),
    AMDGPU.ROCArray,
    AMDGPU.ROCArray,
    phase_retrieval=true,
    domain=selected_domain,
)
