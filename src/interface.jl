"""
    AbstractCalibrationMethod

Supertype for a cold selection of one calibration method and its scientific
options.

A method describes which equations and policies to use. It does not own
prepared arrays, mutable execution state, workspace scratch, caller-visible
products, scheduling, transport, or artifact lifecycle. Concrete methods
participate in [`prepare`](@ref) through multiple dispatch.
"""
abstract type AbstractCalibrationMethod end

"""
    AbstractCalibrationPlan

Supertype for a run-immutable prepared calibration contract.

A concrete plan may retain validated dimensions, layouts, coordinate
conventions, coefficients, factorizations, and logically immutable prepared
arrays. It must not retain persistent mutable scientific state, replaceable
workspace scratch, a caller-visible result, or transient input and output
bindings.

A plan may be shared by concurrent calls only when each call has independent
workspace and result storage and the concrete plan documents that its retained
operators are reentrant.
"""
abstract type AbstractCalibrationPlan end

"""
    KernelExecution(specification, backend; workgroup_size=256)

Bind an ordinary scientific `specification` to an explicit
KernelAbstractions backend for preparation.

The KernelAbstractions extension supplies methods only for supported
calibration families and `KernelAbstractions.Backend` values. Its prepared
plans keep array-valued scientific storage and prepared coefficients on
`backend`. Small host mirrors may be used for validity decisions and compact
diagnostics. [`process!`](@ref) synchronizes before it returns. This target
does not request or imply CUDA Graph or HIP Graph capture.
"""
struct KernelExecution{S,B}
    specification::S
    backend::B
    workgroup_size::Int

    # Julia 1.12/1.13 can discard pointer stores in an intermediate boxed
    # specification while retaining its GC root (JuliaLang/julia#62524).
    # Keep this cold binding opaque so the specification is fully initialized.
    @noinline function KernelExecution(
        specification::S,
        backend::B;
        workgroup_size::Integer=256,
    ) where {S,B}
        size = Int(workgroup_size)
        size > 0 || throw(ArgumentError("workgroup_size must be positive"))
        return new{S,B}(specification, backend, size)
    end
end

"""
    prepare(method, specification) -> AbstractCalibrationPlan

Prepare a run-immutable numerical contract for `method` and `specification`.

Preparation is cold and may allocate, validate structural invariants, select an
implementation through dispatch, and construct reusable factorizations. It
must not perform equipment I/O, publish or activate an artifact, schedule a
workflow, or retain mutable execution state merely for later convenience.

Concrete packages extend this generic for their method and specification
types. Array and accelerator implementations should use ordinary dispatch on
the concrete specification, arrays, or an explicitly defined execution target;
the core interface does not define a backend registry.
"""
function prepare end

"""
    allocate_result(plan::AbstractCalibrationPlan)

Allocate caller-visible scientific result storage compatible with `plan`.

A result may contain several arrays and typed numerical diagnostics; it need
not be a single matrix. The result belongs to the caller and must not be
retained by the plan or workspace.

This advanced allocation seam is public but not exported. Callers normally use
[`process`](@ref), or construct compatible result storage when the concrete
algorithm documents that contract.
"""
function allocate_result end

"""
    allocate_workspace(plan::AbstractCalibrationPlan)

Allocate replaceable scratch storage compatible with `plan`.

Recreating a workspace must not change the deterministic scientific result.
An operation that needs no scratch may return `nothing`. Persistent values that
can affect a later result are state, not workspace, and require an
algorithm-specific owner outside this minimal interface.

This advanced allocation seam is public but not exported. Callers needing
repeatable scratch should retain and reuse the returned workspace with the
four-argument [`process!`](@ref) form.
"""
function allocate_workspace end

"""
    process(plan::AbstractCalibrationPlan, inputs) -> result

Allocate compatible result and workspace storage, then process one complete
calibration operation. This allocating convenience composes
[`allocate_result`](@ref), [`allocate_workspace`](@ref), and the explicit
workspace [`process!`](@ref) form.
"""
function process(plan::AbstractCalibrationPlan, inputs)
    result = allocate_result(plan)
    workspace = allocate_workspace(plan)
    return process!(result, workspace, plan, inputs)
end

"""
    process!(result, plan::AbstractCalibrationPlan, inputs) -> result

Process one complete calibration operation into caller-owned `result`.

This mutating convenience allocates workspace and then composes the explicit
workspace [`process!`](@ref) form. It may therefore allocate scratch. Callers
that need repeatable scratch, allocation control, or KernelAbstractions
execution should use `process!(result, workspace, plan, inputs)`.
"""
function process!(result, plan::AbstractCalibrationPlan, inputs)
    workspace = allocate_workspace(plan)
    return process!(result, workspace, plan, inputs)
end

"""
    process!(result, workspace, plan::AbstractCalibrationPlan, inputs) -> result

Process one complete calibration operation into caller-owned `result`.

An implementation may mutate `result` and replaceable `workspace`. It must not
mutate `plan` or retain the transient `inputs` or `result` after the call.
Expected numerical outcomes belong in the typed calibration product or its
diagnostics; exceptions indicate invalid input, a violated prepared invariant,
or an unexpected implementation failure.

The operation is complete when the function returns. The output, workspace,
plan, and input roles match the complete-operation form of
`FilterGraphAlgorithms.process!`, but this interface does not imply FGA's
real-time boundedness, zero-allocation, progressive-publication, transactional,
or graph-capture contracts. It also does not imply artifact activation. A
concrete method may document and test stricter resource guarantees.
"""
function process! end
