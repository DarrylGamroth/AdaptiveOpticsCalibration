using AdaptiveOpticsCalibration
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using CUDA
using KernelAbstractions
using LinearAlgebra
using Test

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "..", "gpu", "common.jl"))

selected_domain = isempty(ARGS) ? nothing : length(ARGS) == 1 ? ARGS[1] :
    error("Select at most one accelerator test domain: " *
        join(ACCELERATOR_TEST_DOMAINS, ", "))

run_accelerator_tests(
    CUDA.CUDABackend(),
    CUDA.CuArray,
    CUDA.CuArray,
    phase_retrieval=true,
    domain=selected_domain,
)
