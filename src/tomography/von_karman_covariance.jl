"""
    VonKarmanAutoCovariance()

Select assembly of the guide-sample von Kármán phase covariance `Cxx` from
already transformed pupil coordinates. Plain preparation uses the scalar CPU
reference; `KernelExecution(specification, backend)` selects KA execution.
"""
struct VonKarmanAutoCovariance <: AbstractCalibrationMethod end

"""
    VonKarmanCrossCovariance()

Select assembly of the fit-by-guide von Kármán phase covariance `Cox` from
already transformed pupil coordinates. Plain preparation uses the scalar CPU
reference; `KernelExecution(specification, backend)` selects KA execution.
"""
struct VonKarmanCrossCovariance <: AbstractCalibrationMethod end

"""
    VonKarmanAutoCovarianceSpecification(guide_coordinates, r0_m, L0_m,
                                          fractional_cn2)

Specify a guide-sample covariance. `guide_coordinates` has axes
`(guide_sample, guide_source, layer)` and contains already transformed complex
pupil coordinates in m. Guide samples and sources retain this supplied order;
the output's guide axis is source-major, with guide sample varying fastest.
`r0_m` and `L0_m` are positive Fried and outer scales in m. `fractional_cn2`
has one nonnegative, finite normalized weight per layer and must sum to one
within `√eps(T)` or absolute and relative tolerances of `1e-6`. Accepted
weights are used without renormalization. Only `Float32` and `Float64` are supported. All input
arrays require one-based axes; `prepare` copies them into owned CPU storage.
Preparation throws `ArgumentError` for invalid values or unrepresentable
constants and `DimensionMismatch` for incompatible layer counts.

For KA execution, coordinates and weights must reside on the host and are
copied into owned backend storage. KA products also reside on that backend.
`process!` synchronizes before return and publishes only finite candidates.
Plain scalar CPU products require `Matrix{T}` storage for `Cxx`; KA products
use storage on the selected backend.

`Cxx` has phase-radian² units at the wavelength that defines `r0_m`; no
wavelength conversion is performed because this explicit-array contract has no
wavelength input. The copied coordinates and atmospheric scalars become the
plan's run-immutable numerical geometry, not transient process input.

This specification deliberately does not derive source directions, finite
height scaling, pupil geometry, or sample selection. Those physical transforms
belong to the caller, normally AOS.
"""
struct VonKarmanAutoCovarianceSpecification{
    T<:AbstractFloat,
    G<:AbstractArray{Complex{T},3},
    F<:AbstractVector{T},
}
    guide_coordinates::G
    r0_m::T
    L0_m::T
    fractional_cn2::F
end

"""
    VonKarmanCrossCovarianceSpecification(guide_coordinates, fit_coordinates,
                                           r0_m, L0_m, fractional_cn2)

Specify a fit-by-guide covariance. `guide_coordinates` has axes
`(guide_sample, guide_source, layer)` and `fit_coordinates` has axes
`(fit_sample, fit_source, layer)`; both contain already transformed complex
pupil coordinates in m. The source and sample order is retained. `r0_m`,
`L0_m`, and `fractional_cn2` have the same meaning and validation as for
[`VonKarmanAutoCovarianceSpecification`](@ref), including supported scalar
types and one-based axes.

The resulting `Cox` has axes `(fit_source, fit_sample, guide_sample)` where
the last axis is the source-major flattened guide axis: guide sample varies
fastest within each guide source. Thus it matches AOS
`cross_correlation` orientation: fit by guide, rather than the guide-by-fit
matrix evaluated before AOS transposes it.

KA execution accepts host coordinates and weights only, and copies them into
owned backend storage. `Cox` then resides on that backend.
Plain scalar CPU products require `Array{T,3}` storage for `Cox`; KA products
use storage on the selected backend.

`Cox` has phase-radian² units at the wavelength that defines `r0_m`; this
method does not convert wavelengths. Preparation copies its coordinates and
atmospheric scalars into run-immutable numerical geometry.
"""
struct VonKarmanCrossCovarianceSpecification{
    T<:AbstractFloat,
    G<:AbstractArray{Complex{T},3},
    F<:AbstractArray{Complex{T},3},
    W<:AbstractVector{T},
}
    guide_coordinates::G
    fit_coordinates::F
    r0_m::T
    L0_m::T
    fractional_cn2::W
end

"""Prepared scalar CPU constants and copied guide coordinates for `Cxx`."""
struct VonKarmanAutoCovariancePlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    guide_coordinates::Array{Complex{T},3}
    fractional_cn2::Vector{T}
    covariance_constant::T
    zero_separation_covariance::T
    inverse_outer_scale_m::T
end

"""Prepared scalar CPU constants and copied guide/fit coordinates for `Cox`."""
struct VonKarmanCrossCovariancePlan{T<:AbstractFloat} <: AbstractCalibrationPlan
    guide_coordinates::Array{Complex{T},3}
    fit_coordinates::Array{Complex{T},3}
    fractional_cn2::Vector{T}
    covariance_constant::T
    zero_separation_covariance::T
    inverse_outer_scale_m::T
end

"""Caller-owned source-major flattened guide covariance `Cxx`."""
mutable struct VonKarmanAutoCovarianceProduct{T<:AbstractFloat,A<:AbstractMatrix{T}}
    Cxx::A
end

"""Caller-owned fit-by-guide covariance `Cox` in AOS cross-correlation order."""
mutable struct VonKarmanCrossCovarianceProduct{T<:AbstractFloat,A<:AbstractArray{T,3}}
    Cox::A
end

"""Return the caller-owned guide-sample covariance matrix `Cxx`."""
auto_covariance(result::VonKarmanAutoCovarianceProduct) = result.Cxx

"""Return the caller-owned fit-by-guide covariance array `Cox`."""
cross_covariance(result::VonKarmanCrossCovarianceProduct) = result.Cox

function _validate_coordinates(coordinates, name::AbstractString)
    Base.require_one_based_indexing(coordinates)
    sample_count, source_count, layer_count = size(coordinates)
    sample_count > 0 || throw(ArgumentError("$name must have at least one sample"))
    source_count > 0 || throw(ArgumentError("$name must have at least one source"))
    layer_count > 0 || throw(ArgumentError("$name must have at least one layer"))
    _checked_extent("$name sample", sample_count, source_count)
    all(isfinite, coordinates) || throw(ArgumentError(
        "$name must contain only finite complex coordinates",
    ))
    return sample_count, source_count, layer_count
end

function _validate_von_karman_specification(
    guide_coordinates,
    r0_m::T,
    L0_m::T,
    fractional_cn2::AbstractVector{T},
) where {T<:AbstractFloat}
    T <: Union{Float32,Float64} || throw(ArgumentError(
        "von Kármán covariance supports Float32 and Float64 only",
    ))
    _, _, layer_count = _validate_coordinates(guide_coordinates, "guide_coordinates")
    Base.require_one_based_indexing(fractional_cn2)
    length(fractional_cn2) == layer_count || throw(DimensionMismatch(
        "fractional_cn2 length must equal the guide coordinate layer count",
    ))
    isfinite(r0_m) && r0_m > zero(T) || throw(ArgumentError("r0_m must be finite and positive"))
    isfinite(L0_m) && L0_m > zero(T) || throw(ArgumentError("L0_m must be finite and positive"))
    all(isfinite, fractional_cn2) || throw(ArgumentError(
        "fractional_cn2 must contain only finite values",
    ))
    all(>=(zero(T)), fractional_cn2) || throw(ArgumentError(
        "fractional_cn2 must be nonnegative",
    ))
    total_fraction = sum(fractional_cn2)
    isfinite(total_fraction) || throw(ArgumentError("fractional_cn2 sum must be finite"))
    normalized = isapprox(total_fraction, one(T); atol=1e-6, rtol=1e-6) ||
        abs(total_fraction - one(T)) <= sqrt(eps(T))
    normalized || throw(ArgumentError(
        "fractional_cn2 must sum to one within √eps(T) or 1e-6 absolute and relative tolerance",
    ))
    return layer_count
end

function _von_karman_constants(r0_m::T, L0_m::T) where {T<:AbstractFloat}
    gamma_6_5 = gamma(T(6) / T(5))
    gamma_11_6 = gamma(T(11) / T(6))
    gamma_5_6 = gamma(T(5) / T(6))
    base = (T(24) * gamma_6_5 / T(5))^(T(5) / T(6))
    scale = (L0_m / r0_m)^(T(5) / T(3))
    covariance_constant = base * (
        gamma_11_6 / (T(2)^(T(5) / T(6)) * T(π)^(T(8) / T(3)))
    ) * scale
    zero_separation_covariance = base * gamma_11_6 * gamma_5_6 /
        (T(2) * T(π)^(T(8) / T(3))) * scale
    inverse_outer_scale_m = inv(L0_m)
    all(isfinite, (covariance_constant, zero_separation_covariance,
                   inverse_outer_scale_m)) || throw(
        ArgumentError("von Kármán covariance constants must be finite"),
    )
    return covariance_constant, zero_separation_covariance, inverse_outer_scale_m
end

function prepare(
    ::VonKarmanAutoCovariance,
    specification::VonKarmanAutoCovarianceSpecification{T},
) where {T<:AbstractFloat}
    _validate_von_karman_specification(
        specification.guide_coordinates,
        specification.r0_m,
        specification.L0_m,
        specification.fractional_cn2,
    )
    covariance_constant, zero_separation_covariance, inverse_outer_scale_m =
        _von_karman_constants(specification.r0_m, specification.L0_m)
    return VonKarmanAutoCovariancePlan(
        Array{Complex{T},3}(specification.guide_coordinates),
        Vector{T}(specification.fractional_cn2),
        covariance_constant,
        zero_separation_covariance,
        inverse_outer_scale_m,
    )
end

function prepare(
    ::VonKarmanCrossCovariance,
    specification::VonKarmanCrossCovarianceSpecification{T},
) where {T<:AbstractFloat}
    layer_count = _validate_von_karman_specification(
        specification.guide_coordinates,
        specification.r0_m,
        specification.L0_m,
        specification.fractional_cn2,
    )
    _, _, fit_layer_count = _validate_coordinates(
        specification.fit_coordinates,
        "fit_coordinates",
    )
    fit_layer_count == layer_count || throw(DimensionMismatch(
        "fit_coordinates layer count must equal the guide coordinate layer count",
    ))
    covariance_constant, zero_separation_covariance, inverse_outer_scale_m =
        _von_karman_constants(specification.r0_m, specification.L0_m)
    return VonKarmanCrossCovariancePlan(
        Array{Complex{T},3}(specification.guide_coordinates),
        Array{Complex{T},3}(specification.fit_coordinates),
        Vector{T}(specification.fractional_cn2),
        covariance_constant,
        zero_separation_covariance,
        inverse_outer_scale_m,
    )
end

function allocate_result(plan::VonKarmanAutoCovariancePlan{T}) where {T<:AbstractFloat}
    guide_sample_count, guide_source_count, _ = size(plan.guide_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    return VonKarmanAutoCovarianceProduct(
        zeros(T, total_guide_samples, total_guide_samples),
    )
end

function allocate_result(plan::VonKarmanCrossCovariancePlan{T}) where {T<:AbstractFloat}
    guide_sample_count, guide_source_count, _ = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    _checked_extent("von Kármán cross covariance", fit_source_count,
        fit_sample_count, total_guide_samples)
    return VonKarmanCrossCovarianceProduct(
        zeros(T, fit_source_count, fit_sample_count, total_guide_samples),
    )
end

allocate_workspace(::Union{VonKarmanAutoCovariancePlan,VonKarmanCrossCovariancePlan}) =
    nothing

@inline function _von_karman_layer_covariance(
    first_coordinate::Complex{T},
    second_coordinate::Complex{T},
    covariance_constant::T,
    zero_separation_covariance::T,
    inverse_outer_scale_m::T,
    fractional_cn2::T,
) where {T<:AbstractFloat}
    separation_m = abs(first_coordinate - second_coordinate)
    isfinite(separation_m) || throw(ArgumentError(
        "von Kármán coordinate separation must be finite",
    ))
    if iszero(separation_m)
        return zero_separation_covariance * fractional_cn2
    end
    argument = T(2π) * separation_m * inverse_outer_scale_m
    isfinite(argument) && argument > zero(T) || throw(ArgumentError(
        "von Kármán scaled separation must be finite and positive",
    ))
    order = T(5) / T(6)
    return covariance_constant * T(argument^order * besselk(order, argument)) *
        fractional_cn2
end

function _validate_von_karman_cpu_product_disjoint(output, retained_arrays)
    for retained in retained_arrays
        Base.mightalias(output, retained) && throw(ArgumentError(
            "scalar CPU covariance product must not alias prepared coordinates or weights"))
    end
    return nothing
end

function _validate_product(
    result::VonKarmanAutoCovarianceProduct{T},
    plan::VonKarmanAutoCovariancePlan{T},
) where {T}
    result.Cxx isa Matrix{T} || throw(ArgumentError(
        "scalar CPU Cxx product must use Matrix storage"))
    Base.require_one_based_indexing(result.Cxx)
    guide_sample_count, guide_source_count, _ = size(plan.guide_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    size(result.Cxx) == (total_guide_samples, total_guide_samples) || throw(
        DimensionMismatch("Cxx has an incompatible guide-sample extent"),
    )
    _validate_von_karman_cpu_product_disjoint(result.Cxx,
        (plan.guide_coordinates, plan.fractional_cn2))
    return nothing
end

function _validate_product(
    result::VonKarmanCrossCovarianceProduct{T},
    plan::VonKarmanCrossCovariancePlan{T},
) where {T}
    result.Cox isa Array{T,3} || throw(ArgumentError(
        "scalar CPU Cox product must use Array storage"))
    Base.require_one_based_indexing(result.Cox)
    guide_sample_count, guide_source_count, _ = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    expected_size = (fit_source_count, fit_sample_count,
                     total_guide_samples)
    size(result.Cox) == expected_size || throw(
        DimensionMismatch("Cox has an incompatible fit-by-guide extent"),
    )
    _validate_von_karman_cpu_product_disjoint(result.Cox,
        (plan.guide_coordinates, plan.fit_coordinates, plan.fractional_cn2))
    return nothing
end

"""
    process!(result, nothing, plan::VonKarmanAutoCovariancePlan, nothing) -> result

Assemble `Cxx` with `Cxx[i, j] = Σₗ fₗ B(ρᵢₗ, ρⱼₗ)`, using the von Kármán
covariance from AOS `src/tomography/reconstructors.jl`. For separation
`d = |ρ₁ - ρ₂| > 0`, `B(d) = A u^(5/6) K₅/₆(u)` where
`u = 2πd/L₀`, `A = [24Γ(6/5)/5]^(5/6) Γ(11/6)
(L₀/r₀)^(5/3) / [2^(5/6) π^(8/3)]`, and `K₅/₆` is the modified
Bessel function of the second kind. At `d = 0`, the analytic limit is
`A Γ(5/6) 2^(-1/6)`. Products are phase-radian² at the wavelength
defining `r₀`. The retained
coordinates have already received all source-direction and finite-height
transforms. This cold scalar CPU operation makes no GPU, per-frame, or
zero-allocation claim. A malformed product or unrepresentable covariance
throws before publication, leaving the caller's previous product unchanged.
"""
function process!(
    result::VonKarmanAutoCovarianceProduct{T},
    ::Nothing,
    plan::VonKarmanAutoCovariancePlan{T},
    ::Nothing,
) where {T<:AbstractFloat}
    _validate_product(result, plan)
    guide_sample_count, guide_source_count, layer_count = size(plan.guide_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    candidate = zeros(T, total_guide_samples, total_guide_samples)
    @inbounds for second_source in 1:guide_source_count,
        second_sample in 1:guide_sample_count,
        first_source in 1:guide_source_count,
        first_sample in 1:guide_sample_count
        row = first_sample + (first_source - 1) * guide_sample_count
        column = second_sample + (second_source - 1) * guide_sample_count
        for layer in 1:layer_count
            candidate[row, column] += _von_karman_layer_covariance(
                plan.guide_coordinates[first_sample, first_source, layer],
                plan.guide_coordinates[second_sample, second_source, layer],
                plan.covariance_constant,
                plan.zero_separation_covariance,
                plan.inverse_outer_scale_m,
                plan.fractional_cn2[layer],
            )
        end
    end
    all(isfinite, candidate) || throw(ArgumentError("Cxx contains non-finite values"))
    copyto!(result.Cxx, candidate)
    return result
end

"""
    process!(result, nothing, plan::VonKarmanCrossCovariancePlan, nothing) -> result

Assemble `Cox` with `Cox[f, q, g] = Σₗ fₗ B(ρfit[f, q, l],
ρguide[g, l])`. Its axes are `(fit_source, fit_sample, guide_sample)`, and
the guide axis is source-major flattened guide coordinates with guide sample
varying fastest. This is the fit-by-guide orientation of AOS
`cross_correlation`, after its internal guide-by-fit covariance is transposed.
The operation is cold scalar CPU preparation only.
As with `Cxx`, an invalid product or unrepresentable covariance leaves the
caller-owned product unchanged.
"""
function process!(
    result::VonKarmanCrossCovarianceProduct{T},
    ::Nothing,
    plan::VonKarmanCrossCovariancePlan{T},
    ::Nothing,
) where {T<:AbstractFloat}
    _validate_product(result, plan)
    guide_sample_count, guide_source_count, layer_count = size(plan.guide_coordinates)
    fit_sample_count, fit_source_count, _ = size(plan.fit_coordinates)
    total_guide_samples = _checked_extent(
        "von Kármán guide sample", guide_sample_count, guide_source_count)
    candidate = zeros(T, fit_source_count, fit_sample_count,
                      total_guide_samples)
    @inbounds for guide_source in 1:guide_source_count,
        guide_sample in 1:guide_sample_count,
        fit_source in 1:fit_source_count,
        fit_sample in 1:fit_sample_count
        flattened_guide_sample = guide_sample +
            (guide_source - 1) * guide_sample_count
        for layer in 1:layer_count
            candidate[fit_source, fit_sample, flattened_guide_sample] +=
                _von_karman_layer_covariance(
                    plan.fit_coordinates[fit_sample, fit_source, layer],
                    plan.guide_coordinates[guide_sample, guide_source, layer],
                    plan.covariance_constant,
                    plan.zero_separation_covariance,
                    plan.inverse_outer_scale_m,
                    plan.fractional_cn2[layer],
                )
        end
    end
    all(isfinite, candidate) || throw(ArgumentError("Cox contains non-finite values"))
    copyto!(result.Cox, candidate)
    return result
end
