"""
    CovarianceReconstructor()

Select explicit-array tomographic covariance reconstruction. `projection`
may be a model phase-to-measurement operator or a measured interaction matrix;
both use the same equation. The caller declares its coordinate order and units.
`prepare(method, specification)` is the host scalar path. The optional
`prepare(method, KernelExecution(specification, backend))` path prepares a cold
CUDA or ROCm operation through the KernelAbstractions extension; it makes no
graph-capture or zero-allocation execution claim.
"""
struct CovarianceReconstructor <: AbstractCalibrationMethod end

function _covariance_reconstructor_lu_solve! end

function _covariance_reconstructor_alias_root(array::AbstractArray)
    parent_array = try
        parent(array)
    catch error
        error isa MethodError || rethrow()
        return array
    end
    parent_array === array && return array
    return _covariance_reconstructor_alias_root(parent_array)
end

_covariance_reconstructor_alias_root(array) = array
_covariance_reconstructor_root_mightalias(_, first, second) = Base.mightalias(first, second)

"""
    CovarianceReconstructorSpecification(phase_count, measurement_count,
                                         fit_count, T=Float64)

Dimensions and floating-point representation for the explicit covariance
equation `R = Cox Pᵀ / (P Cxx Pᵀ + Cnz)`. The specification deliberately
contains no geometric, wavelength, command-scale, or ordering policy.
Host preparation accepts any `AbstractFloat`; accelerator preparation accepts
`Float32` and `Float64` and requires all inputs, workspace, and product storage
to reside on its selected backend.
"""
struct CovarianceReconstructorSpecification{T<:AbstractFloat}
    phase_count::Int
    measurement_count::Int
    fit_count::Int

    function CovarianceReconstructorSpecification{T}(
        phase_count::Integer,
        measurement_count::Integer,
        fit_count::Integer,
    ) where {T<:AbstractFloat}
        phase = Int(phase_count)
        measurements = Int(measurement_count)
        fit = Int(fit_count)
        phase > 0 || throw(ArgumentError("phase_count must be positive"))
        measurements > 0 || throw(ArgumentError("measurement_count must be positive"))
        fit > 0 || throw(ArgumentError("fit_count must be positive"))
        _checked_extent("tomographic covariance", phase, measurements, fit)
        return new{T}(phase, measurements, fit)
    end
end

CovarianceReconstructorSpecification(
    phase_count::Integer,
    measurement_count::Integer,
    fit_count::Integer,
    ::Type{T}=Float64,
) where {T<:AbstractFloat} = CovarianceReconstructorSpecification{T}(
    phase_count,
    measurement_count,
    fit_count,
)

"""
    CovarianceReconstructorInputs(projection, phase_covariance,
                                  fit_phase_covariance,
                                  measurement_noise_covariance)

Explicit arrays for `R = Cox Pᵀ / (P Cxx Pᵀ + Cnz)`: `projection = P` has
measurement rows and phase columns; `phase_covariance = Cxx` is phase by
phase; `fit_phase_covariance = Cox` is fit by phase; and
`measurement_noise_covariance = Cnz` is measurement by measurement. All four
arrays use the same floating-point element type and one-based axes. Their
scientific units, coordinate order, sign, and normalization pass through
unchanged. `Cxx` and `Cnz` must be symmetric within `√eps(T)` times their
largest absolute entry. Accepted off-diagonal pairs are averaged before the
solve. They need not be positive definite because the source solve has an LU
fallback.

For accelerator preparation, each array must be a selected-backend `Float32`
or `Float64` matrix. The operation keeps full arrays on that backend, transfers
only compact validation and diagnostic scalars, synchronizes before returning,
and publishes the matrix and diagnostics only after the complete solve succeeds.
"""
struct CovarianceReconstructorInputs{
    T<:AbstractFloat,
    P<:AbstractMatrix{T},
    X<:AbstractMatrix{T},
    O<:AbstractMatrix{T},
    N<:AbstractMatrix{T},
}
    projection::P
    phase_covariance::X
    fit_phase_covariance::O
    measurement_noise_covariance::N
end

"""Prepared dimensions and floating-point representation for the raw solve."""
struct CovarianceReconstructorPlan{
    T<:AbstractFloat,
} <: AbstractCalibrationPlan
    phase_count::Int
    measurement_count::Int
    fit_count::Int
end

function prepare(
    ::CovarianceReconstructor,
    specification::CovarianceReconstructorSpecification{T},
) where {T<:AbstractFloat}
    return CovarianceReconstructorPlan{T}(
        specification.phase_count,
        specification.measurement_count,
        specification.fit_count,
    )
end

"""Numerical information recorded for one covariance reconstructor solve."""
struct CovarianceReconstructorDiagnostics{T<:AbstractFloat}
    used_lu_fallback::Bool
    system_matrix_norm::T
    reconstructor_norm::T
end

"""Caller-owned fit-coordinate-by-measurement reconstructor and diagnostics.

The matrix is a host `Matrix` for scalar preparation and selected-backend matrix
storage for accelerator preparation.
"""
mutable struct CovarianceReconstructorProduct{T<:AbstractFloat,M<:AbstractMatrix{T}}
    matrix::M
    diagnostics::CovarianceReconstructorDiagnostics{T}
end

"""Return the caller-owned fit-coordinate-by-measurement reconstructor matrix."""
reconstructor(result::CovarianceReconstructorProduct) = result.matrix

function _initial_diagnostics(::Type{T}) where {T<:AbstractFloat}
    return CovarianceReconstructorDiagnostics{T}(false, zero(T), zero(T))
end

function allocate_result(plan::CovarianceReconstructorPlan{T}) where {T<:AbstractFloat}
    return CovarianceReconstructorProduct(
        zeros(T, plan.fit_count, plan.measurement_count),
        _initial_diagnostics(T),
    )
end

allocate_workspace(::CovarianceReconstructorPlan) = nothing

function _validate_product(
    result::CovarianceReconstructorProduct,
    plan::CovarianceReconstructorPlan,
)
    Base.require_one_based_indexing(result.matrix)
    size(result.matrix) == (plan.fit_count, plan.measurement_count) || throw(
        DimensionMismatch("tomographic covariance result has an incompatible matrix extent"),
    )
    return nothing
end

function _validated_matrix(
    matrix::AbstractMatrix{T},
    shape::Tuple{Int,Int},
    name::AbstractString,
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(matrix)
    size(matrix) == shape || throw(DimensionMismatch("$name must have shape $shape"))
    all(isfinite, matrix) || throw(ArgumentError("$name must contain only finite values"))
    return Matrix{T}(matrix)
end

function _validated_covariance(
    matrix::AbstractMatrix{T},
    count::Int,
    name::AbstractString,
) where {T<:AbstractFloat}
    staged = _validated_matrix(matrix, (count, count), name)
    scale = maximum(abs, staged)
    tolerance = sqrt(eps(T)) * scale
    @inbounds for column in 1:count, row in 1:(column - 1)
        abs(staged[row, column] - staged[column, row]) <= tolerance || throw(
            ArgumentError("$name must be symmetric within numerical tolerance"),
        )
        average = staged[row, column] / T(2) + staged[column, row] / T(2)
        staged[row, column] = average
        staged[column, row] = average
    end
    return staged
end

function _staged_covariance_inputs(
    plan::CovarianceReconstructorPlan{T},
    input::CovarianceReconstructorInputs{T},
) where {T<:AbstractFloat}
    P = _validated_matrix(
        input.projection,
        (plan.measurement_count, plan.phase_count),
        "projection",
    )
    Cxx = _validated_covariance(
        input.phase_covariance,
        plan.phase_count,
        "phase_covariance",
    )
    Cox = _validated_matrix(
        input.fit_phase_covariance,
        (plan.fit_count, plan.phase_count),
        "fit_phase_covariance",
    )
    Cnz = _validated_covariance(
        input.measurement_noise_covariance,
        plan.measurement_count,
        "measurement_noise_covariance",
    )
    return P, Cxx, Cox, Cnz
end

function _stable_right_division(rhs::Matrix{T}, system::Matrix{T}) where {T<:AbstractFloat}
    factor = cholesky(Hermitian(system), check=false)
    if issuccess(factor)
        return transpose(factor \ transpose(rhs)), false
    end
    return transpose(lu(copy(transpose(system))) \ transpose(rhs)), true
end

"""
    process!(result, nothing, plan, inputs) -> result

Prepare the complete CPU covariance reconstructor
`Cox Pᵀ / (P Cxx Pᵀ + Cnz)`. The solve first uses
`cholesky(Hermitian(system), check=false)` and uses LU only when that factor is
not successful, matching the extracted AOS cold algebra. Inputs and temporary
products are validated before publication, so an invalid or singular solve does
not change `result`. This is cold preparation: it makes no GPU, per-frame, or
zero-allocation execution claim.
"""
function process!(
    result::CovarianceReconstructorProduct{T},
    ::Nothing,
    plan::CovarianceReconstructorPlan{T},
    input::CovarianceReconstructorInputs{T},
) where {T<:AbstractFloat}
    _validate_product(result, plan)
    P, Cxx, Cox, Cnz = _staged_covariance_inputs(plan, input)
    system = _validated_covariance(
        P * Cxx * transpose(P) + Cnz,
        plan.measurement_count,
        "tomographic covariance system",
    )
    rhs = Cox * transpose(P)
    all(isfinite, system) || throw(ArgumentError(
        "tomographic covariance system contains non-finite values",
    ))
    all(isfinite, rhs) || throw(ArgumentError(
        "tomographic covariance right-hand side contains non-finite values",
    ))
    matrix, used_lu_fallback = _stable_right_division(rhs, system)
    all(isfinite, matrix) || throw(ArgumentError(
        "tomographic covariance reconstructor contains non-finite values",
    ))
    system_norm = T(norm(system))
    reconstructor_norm = T(norm(matrix))
    isfinite(system_norm) || throw(ArgumentError(
        "tomographic covariance system norm is non-finite",
    ))
    isfinite(reconstructor_norm) || throw(ArgumentError(
        "tomographic covariance reconstructor norm is non-finite",
    ))
    diagnostics = CovarianceReconstructorDiagnostics(
        used_lu_fallback,
        system_norm,
        reconstructor_norm,
    )
    copyto!(result.matrix, matrix)
    result.diagnostics = diagnostics
    return result
end
