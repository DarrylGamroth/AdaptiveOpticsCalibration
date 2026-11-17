using CUDA

include("lift_accelerator_common.jl")

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

function cuda_device_allocated_bytes(f)
    return CUDA.@allocated f()
end

function cuda_profile_region(f)
    CUDA.@profile external=true f()
    return nothing
end

run_lift_accelerator_benchmark(
    "cuda",
    CUDA.CUDABackend(),
    CUDA.CuArray,
    CUDA.synchronize,
    cuda_device_allocated_bytes,
    CUDA.used_memory,
    cuda_profile_region,
)
