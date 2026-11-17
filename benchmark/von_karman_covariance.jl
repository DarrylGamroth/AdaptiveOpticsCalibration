#=
Opt-in shape-equivalent covariance assembly comparison. Run one backend at a time:

    julia --startup-file=no --project=. benchmark/von_karman_covariance.jl cpu
    julia --startup-file=no --project=test/cuda benchmark/von_karman_covariance.jl cuda
    julia --startup-file=no --project=test/amdgpu benchmark/von_karman_covariance.jl amdgpu

`AOC_COV_SAMPLES` (default 3) and `AOC_COV_WARMUPS` (default 1) control the
closed-loop repeated-call samples. The synthetic 17×17 selected coordinates
are dimension-equivalent to the target AOS product shapes, not the AOS
physical fixture. The CPU reference uses SpecialFunctions;
device execution uses the package's KA approximation. This is a cold
calibration-operation comparison, not a fixed-arrival-rate, full AOS builder,
or RTC benchmark. `cpu_process_plus_output_upload` includes the full output
copy and synchronization, and excludes both CPU and device plan preparation.
=#
using AdaptiveOpticsCalibration
using AdaptiveOpticsCalibration.Tomography
import AdaptiveOpticsCalibration: allocate_result, allocate_workspace
using LinearAlgebra

const MODE = only(ARGS)
if MODE == "cuda"
    @eval using CUDA
    @eval using KernelAbstractions
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
elseif MODE == "amdgpu"
    @eval using AMDGPU
    @eval using KernelAbstractions
    AMDGPU.functional() || error("AMDGPU is not functional")
    AMDGPU.allowscalar(false)
elseif MODE != "cpu"
    error("mode must be cpu, cuda, or amdgpu")
end

function backend_for_mode()
    MODE == "cuda" && return CUDA.CUDABackend()
    MODE == "amdgpu" && return AMDGPU.ROCBackend()
    return nothing
end

function covariance_problem(::Type{T}) where {T<:AbstractFloat}
    # 17 × 17 pupil samples give 289 points per source: Cxx 578 × 578,
    # Cox 4 × 289 × 578, with two independently transformed layers.
    guide = [complex(
        T(mod(sample - 1, 17) + 2 * (source - 1)) / T(32),
        T((sample - 1) ÷ 17 + layer - 1) / T(32),
    ) for sample in 1:289, source in 1:2, layer in 1:2]
    fit = [complex(
        T(mod(sample - 1, 17) + source - 1) / T(32),
        T((sample - 1) ÷ 17 + 2 * (layer - 1)) / T(32),
    ) for sample in 1:289, source in 1:4, layer in 1:2]
    weights = T[T(0.3), T(0.7)]
    return (
        (VonKarmanAutoCovariance(),
            VonKarmanAutoCovarianceSpecification(guide, T(0.16), T(25), weights)),
        (VonKarmanCrossCovariance(),
            VonKarmanCrossCovarianceSpecification(
                guide, fit, T(0.16), T(25), weights)),
    )
end

@noinline function run_covariance!(result, workspace, plan)
    process!(result, workspace, plan, nothing)
    return result
end

function timed_samples(operation, count)
    times = Vector{UInt64}(undef, count)
    for sample in eachindex(times)
        start = time_ns()
        operation()
        times[sample] = time_ns() - start
    end
    sort!(times)
    return times
end

function report_times(prefix, times)
    println(prefix, "_samples = ", length(times))
    println(prefix, "_min_ns = ", first(times))
    middle = cld(length(times), 2)
    median_ns = isodd(length(times)) ? times[middle] :
        (Float64(times[middle]) + Float64(times[middle + 1])) / 2
    println(prefix, "_median_ns = ", median_ns)
    println(prefix, "_max_ns = ", last(times))
end

function benchmark_covariance(method, specification, backend, warmups, samples)
    name = method isa VonKarmanAutoCovariance ? "Cxx" : "Cox"
    cpu_start = time_ns()
    cpu_plan = prepare(method, specification)
    cpu_prepare_ns = time_ns() - cpu_start
    cpu_result = allocate_result(cpu_plan)
    cpu_workspace = allocate_workspace(cpu_plan)
    run_covariance!(cpu_result, cpu_workspace, cpu_plan)
    for _ in 1:warmups
        run_covariance!(cpu_result, cpu_workspace, cpu_plan)
    end
    cpu_host_bytes = @allocated run_covariance!(cpu_result, cpu_workspace, cpu_plan)
    cpu_times = timed_samples(() -> run_covariance!(cpu_result, cpu_workspace, cpu_plan), samples)
    cpu_output = method isa VonKarmanAutoCovariance ? cpu_result.Cxx : cpu_result.Cox

    println("product = ", repr(name))
    println("shape = ", repr(size(cpu_output)))
    println("cpu_prepare_ns = ", cpu_prepare_ns)
    println("cpu_process_host_allocated_bytes = ", cpu_host_bytes)
    report_times("cpu_process", cpu_times)
    backend === nothing && return nothing

    execution = KernelExecution(specification, backend; workgroup_size=256)
    device_start = time_ns()
    device_plan = prepare(method, execution)
    device_prepare_ns = time_ns() - device_start
    device_result = allocate_result(device_plan)
    device_workspace = allocate_workspace(device_plan)
    first_start = time_ns()
    run_covariance!(device_result, device_workspace, device_plan)
    first_process_ns = time_ns() - first_start
    for _ in 1:warmups
        run_covariance!(device_result, device_workspace, device_plan)
    end
    device_host_bytes = @allocated run_covariance!(device_result, device_workspace, device_plan)
    device_times = timed_samples(() -> run_covariance!(device_result,
        device_workspace, device_plan), samples)
    device_output = method isa VonKarmanAutoCovariance ?
        device_result.Cxx : device_result.Cox
    recovered = Array(device_output)
    maximum_absolute_error = maximum(abs, recovered - cpu_output)
    maximum_relative_error = maximum(abs.(recovered .- cpu_output) ./ abs.(cpu_output))
    isapprox(recovered, cpu_output; rtol=2f-4) || error(
        "$name device result failed CPU reference parity")

    upload_target = similar(device_output)
    upload!() = (copyto!(upload_target, cpu_output);
        KernelAbstractions.synchronize(backend))
    upload!()
    upload_host_bytes = @allocated upload!()
    upload_times = timed_samples(upload!, samples)
    baseline!() = (run_covariance!(cpu_result, cpu_workspace, cpu_plan); upload!())
    baseline_times = timed_samples(baseline!, samples)

    println("device_prepare_ns = ", device_prepare_ns)
    println("device_first_process_ns = ", first_process_ns)
    println("device_process_host_allocated_bytes = ", device_host_bytes)
    report_times("device_process", device_times)
    println("output_upload_host_allocated_bytes = ", upload_host_bytes)
    report_times("output_upload", upload_times)
    report_times("cpu_process_plus_output_upload", baseline_times)
    println("maximum_absolute_error = ", maximum_absolute_error)
    println("maximum_relative_error = ", maximum_relative_error)
    return nothing
end

function main()
    BLAS.set_num_threads(1)
    samples = parse(Int, get(ENV, "AOC_COV_SAMPLES", "3"))
    warmups = parse(Int, get(ENV, "AOC_COV_WARMUPS", "1"))
    samples > 0 && warmups >= 0 || error("samples must be positive and warmups nonnegative")
    backend = backend_for_mode()
    println("backend = ", repr(MODE))
    println("julia_version = ", repr(string(VERSION)))
    println("julia_threads = ", Threads.nthreads())
    println("blas_threads = ", BLAS.get_num_threads())
    println("head = ", repr(readchomp(`git rev-parse HEAD`)))
    println("dirty = ", !isempty(readchomp(`git status --porcelain`)))
    println("coordinate_corpus = \"synthetic_17x17_shape_equivalent\"")
    if MODE == "cuda"
        println("backend_version = ", repr(string(Base.pkgversion(CUDA))))
        println("device = ", repr(CUDA.name(CUDA.device())))
    elseif MODE == "amdgpu"
        println("backend_version = ", repr(string(Base.pkgversion(AMDGPU))))
        println("device = ", repr(AMDGPU.HIP.name(AMDGPU.device())))
    end
    println("hsa_override_gfx_version = ", repr(get(ENV, "HSA_OVERRIDE_GFX_VERSION", "")))
    println("warmups = ", warmups)
    println("samples = ", samples)
    for (method, specification) in covariance_problem(Float32)
        benchmark_covariance(method, specification, backend, warmups, samples)
    end
    return nothing
end

main()
