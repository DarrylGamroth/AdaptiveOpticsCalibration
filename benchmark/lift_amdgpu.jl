using AMDGPU

include("lift_accelerator_common.jl")

AMDGPU.functional() || error("AMDGPU is not functional on this host")
AMDGPU.allowscalar(false)

function amdgpu_device_allocated_bytes(f)
    before = AMDGPU.alloc_stats.alloc_bytes
    f()
    return AMDGPU.alloc_stats.alloc_bytes - before
end

function amdgpu_profile_region(f)
    resume_status = ccall(
        (:roctxProfilerResume, "librocprofiler-sdk-roctx.so"),
        Cint,
        (UInt64,),
        0,
    )
    iszero(resume_status) || error(
        "rocprofiler rejected the LiFT capture-region resume request",
    )
    try
        f()
    finally
        AMDGPU.synchronize()
        pause_status = ccall(
            (:roctxProfilerPause, "librocprofiler-sdk-roctx.so"),
            Cint,
            (UInt64,),
            0,
        )
        iszero(pause_status) || error(
            "rocprofiler rejected the LiFT capture-region pause request",
        )
    end
    return nothing
end

run_lift_accelerator_benchmark(
    "amdgpu",
    AMDGPU.ROCBackend(),
    AMDGPU.ROCArray,
    AMDGPU.synchronize,
    amdgpu_device_allocated_bytes,
    AMDGPU.used_memory,
    amdgpu_profile_region,
)
