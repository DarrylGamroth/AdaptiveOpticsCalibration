using KernelAbstractions

include("lift_common.jl")

@noinline function synchronized_process!(synchronize, result, workspace, plan, inputs)
    process!(result, workspace, plan, inputs)
    synchronize()
    return result
end

function run_lift_accelerator_benchmark(
    backend_name::String,
    backend,
    to_device,
    synchronize,
    device_allocated_bytes,
    used_device_memory,
    profile_region,
)
    rows = parse(Int, get(ENV, "AOC_LIFT_ROWS", "4096"))
    columns = parse(Int, get(ENV, "AOC_LIFT_COLUMNS", "64"))
    iterations = parse(Int, get(ENV, "AOC_LIFT_ITERATIONS", "1"))
    warmups = parse(Int, get(ENV, "AOC_LIFT_WARMUPS", "10"))
    samples = parse(Int, get(ENV, "AOC_LIFT_SAMPLES", "50"))
    profile_calls = parse(Int, get(ENV, "AOC_LIFT_PROFILE_CALLS", "20"))

    reference_rate, jacobian, truth, observation =
        benchmark_problem(rows, columns)
    model = BenchmarkLiFTModel(
        to_device(reference_rate),
        to_device(jacobian),
    )
    plan = prepare(
        PR.LiFT(
            iterations=iterations,
            solve_mode=PR.LiFTSolveNormalEquations(),
            damping=PR.LiFTDampingNone(),
            weighting=PR.LiFTReadNoiseWeighting(),
            model_scaling=PR.LiFTPhysicalRatePreservation(),
            check_convergence=false,
        ),
        KernelExecution(
            PR.LiFTSpecification(
                model,
                PR.LiFTPhotonRate();
                read_noise_std=0.5f0,
            ),
            backend;
            workgroup_size=256,
        ),
    )
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    inputs = PR.LiFTInputs(to_device(observation))

    for _ in 1:warmups
        synchronized_process!(synchronize, result, workspace, plan, inputs)
    end
    recovered = Array(result.coefficients)
    maximum_error = maximum(abs, recovered - truth)
    maximum_error < 2f-4 || error(
        "LiFT accelerator benchmark failed its CPU-constructed coefficient oracle: " *
        "maximum error $(maximum_error)",
    )
    Array(inputs.observation) == observation || error(
        "LiFT accelerator execution mutated its caller-owned observation",
    )

    host_bytes = @allocated synchronized_process!(
        synchronize,
        result,
        workspace,
        plan,
        inputs,
    )
    device_bytes = device_allocated_bytes() do
        synchronized_process!(synchronize, result, workspace, plan, inputs)
    end
    memory_before = used_device_memory()
    synchronized_process!(synchronize, result, workspace, plan, inputs)
    memory_after = used_device_memory()

    elapsed_ns = Vector{UInt64}(undef, samples)
    for sample in eachindex(elapsed_ns)
        start = time_ns()
        synchronized_process!(synchronize, result, workspace, plan, inputs)
        elapsed_ns[sample] = time_ns() - start
    end
    sort!(elapsed_ns)

    if get(ENV, "AOC_LIFT_PROFILE", "0") == "1"
        profile_region() do
            for _ in 1:profile_calls
                process!(result, workspace, plan, inputs)
            end
            synchronize()
        end
    end

    recovered = Array(result.coefficients)
    post_profile_error = maximum(abs, recovered - truth)
    post_profile_error < 2f-4 || error(
        "LiFT accelerator post-profile oracle failed: maximum error " *
        "$(post_profile_error)",
    )

    println("backend = ", repr(backend_name))
    println("rows = ", rows)
    println("columns = ", columns)
    println("iterations = ", iterations)
    println("warmups = ", warmups)
    println("samples = ", samples)
    println("profile_calls = ", profile_calls)
    println("julia_threads = ", Threads.nthreads())
    println("maximum_error = ", maximum_error)
    println("post_profile_error = ", post_profile_error)
    println("host_allocated_bytes = ", host_bytes)
    println("device_allocated_bytes = ", device_bytes)
    println("used_device_memory_before = ", memory_before)
    println("used_device_memory_after = ", memory_after)
    println("p50_ns = ", quantile_sorted(elapsed_ns, 0.50))
    println("p99_ns = ", quantile_sorted(elapsed_ns, 0.99))
    println("minimum_ns = ", first(elapsed_ns))
    println("maximum_ns = ", last(elapsed_ns))
    return nothing
end
