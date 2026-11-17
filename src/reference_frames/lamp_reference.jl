"""
    LampReference(minimum_flux, adc_upper_rail)

Declare one finite nonnegative minimum flux threshold for every ordered ROI
and a finite positive raw ADC upper rail. Threshold equality is eligible;
ADC rail equality invalidates the entire batch. Flux thresholds have the
deployed estimator's flux units. The threshold vector is copied; the rail's
numeric representation is preserved for comparison with raw unsigned samples.
"""
struct LampReference{R<:Real} <: AbstractCalibrationMethod
    minimum_flux::Vector{Float64}
    adc_upper_rail::R

    function LampReference(minimum_flux::AbstractVector{<:Real},
                           adc_upper_rail::R) where {R<:Real}
        Base.require_one_based_indexing(minimum_flux)
        isempty(minimum_flux) && throw(ArgumentError(
            "minimum_flux must declare at least one ROI",
        ))
        thresholds = Float64.(minimum_flux)
        all(x -> isfinite(x) && x >= 0.0, thresholds) || throw(ArgumentError(
            "minimum_flux thresholds must be finite and nonnegative",
        ))
        isfinite(adc_upper_rail) && adc_upper_rail > 0 || throw(ArgumentError(
            "adc_upper_rail must be finite and positive",
        ))
        return new{R}(thresholds, adc_upper_rail)
    end
end

"""
    LampReferenceSpecification(rows, columns, roi_count, sample_count)

Declare positive raw detector dimensions, ROI count, and exact exposure count.
Centroid rows are interleaved `(x₁, y₁, x₂, y₂, …)` in detector pixels, with
two rows for every ROI position. For a deployed 188-ROI estimator the centroid
shape is `(376, sample_count)`; the ROI count is never inferred from validity.
All inputs must use one-based indexing. No phase/radian conversion, pixel
geometry, or instrument-specific active-ROI count is imposed.
"""
struct LampReferenceSpecification
    rows::Int
    columns::Int
    roi_count::Int
    sample_count::Int

    function LampReferenceSpecification(rows::Integer, columns::Integer,
                                        roi_count::Integer, sample_count::Integer)
        r, c, k, n = Int(rows), Int(columns), Int(roi_count), Int(sample_count)
        r > 0 || throw(ArgumentError("rows must be positive"))
        c > 0 || throw(ArgumentError("columns must be positive"))
        k > 0 || throw(ArgumentError("roi_count must be positive"))
        n > 0 || throw(ArgumentError("lamp sample_count must be positive"))
        _checked_extent("lamp raw batch", r, c, n)
        _checked_extent("lamp centroids", 2, k, n)
        return new(r, c, k, n)
    end
end

"""Run-immutable lamp dimensions and copied per-ROI flux criteria."""
struct LampReferencePlan{R<:Real} <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    roi_count::Int
    sample_count::Int
    minimum_flux::Vector{Float64}
    adc_upper_rail::R
end

function prepare(method::LampReference, spec::LampReferenceSpecification)
    length(method.minimum_flux) == spec.roi_count || throw(DimensionMismatch(
        "minimum_flux must declare a threshold for each ROI position",
    ))
    return LampReferencePlan(spec.rows, spec.columns, spec.roi_count,
        spec.sample_count, copy(method.minimum_flux), method.adc_upper_rail)
end

"""
    LampReferenceInputs(centroids, flux, intrinsic_valid, raw_adc;
                        centroids_are_absolute)

Bundle complete deployed per-exposure centroids in pixels, estimator flux,
intrinsic ROI validity, and raw unsigned ADC images. Shapes are `(2K, N)`,
`(K, N)`, `(K, N)`, and `(rows, columns, N)`. The caller must explicitly assert
`centroids_are_absolute=true`: inputs must be actual detector-pixel
centroids. Acquire them with zero subtraction reference, or reconstruct them
by adding the explicitly known deployed reference to each residual sample.
Averaging residual slopes alone does not yield an absolute lamp reference.
This assertion cannot establish acquisition history; the composing application
owns the conversion and precondition. No hidden reference or optical truth is
inferred.
"""
struct LampReferenceInputs{
    C<:AbstractMatrix{<:AbstractFloat},
    F<:AbstractMatrix{<:Real},
    V<:AbstractMatrix{Bool},
    A<:AbstractArray{<:Unsigned,3},
}
    centroids::C
    flux::F
    intrinsic_valid::V
    raw_adc::A
    centroids_are_absolute::Bool
end

function LampReferenceInputs(centroids::AbstractMatrix{<:AbstractFloat},
                             flux::AbstractMatrix{<:Real},
                             intrinsic_valid::AbstractMatrix{Bool},
                             raw_adc::AbstractArray{<:Unsigned,3};
                             centroids_are_absolute::Bool)
    return LampReferenceInputs(centroids, flux, intrinsic_valid, raw_adc,
        centroids_are_absolute)
end

"""
    LampReferenceProduct

Caller-owned Float64 `reference_centroids` in interleaved detector-pixel
measurement order and one `eligible` Boolean per ROI. Inactive ROI reference
entries are explicitly zero. `minimum_flux` and `all_training_valid` retain
diagnostics for every ROI position, including inactive ones. The numerical
minimum is retained, with NaN propagating into its ROI minimum. `adc_maximum`
is the global maximum converted to Float64 for reporting; the rail comparison uses raw
samples and the declared rail directly. `Invalid` zeroes the whole reference
and eligibility mask. `NoEligible` denotes a finite below-rail batch with no
ROI satisfying both criteria. Acceptance remains caller-owned.
"""
mutable struct LampReferenceProduct
    status::ReferenceStatus
    reference_centroids::Vector{Float64}
    eligible::Vector{Bool}
    minimum_flux::Vector{Float64}
    all_training_valid::Vector{Bool}
    adc_maximum::Float64
end

"""Replaceable centroid-mean scratch, reset for each complete lamp batch."""
struct LampReferenceWorkspace
    reference_centroids::Vector{Float64}
end

allocate_result(plan::LampReferencePlan) = LampReferenceProduct(
    Invalid, zeros(2 * plan.roi_count), fill(false, plan.roi_count),
    zeros(plan.roi_count), fill(false, plan.roi_count), 0.0,
)
allocate_workspace(plan::LampReferencePlan) =
    LampReferenceWorkspace(zeros(2 * plan.roi_count))

function _validate_lamp(result, workspace, plan, input)
    input.centroids_are_absolute || throw(ArgumentError(
        "lamp inputs must be explicitly declared absolute detector-pixel centroids",
    ))
    Base.require_one_based_indexing(input.centroids, input.flux,
        input.intrinsic_valid, input.raw_adc)
    size(input.centroids) == (2 * plan.roi_count, plan.sample_count) || throw(
        DimensionMismatch("lamp centroid shape differs from its specification"),
    )
    expected = (plan.roi_count, plan.sample_count)
    size(input.flux) == expected || throw(DimensionMismatch(
        "lamp flux shape differs from its specification",
    ))
    size(input.intrinsic_valid) == expected || throw(DimensionMismatch(
        "lamp intrinsic-validity shape differs from its specification",
    ))
    size(input.raw_adc) == (plan.rows, plan.columns, plan.sample_count) || throw(
        DimensionMismatch("lamp raw ADC shape differs from its specification"),
    )
    length(result.reference_centroids) == 2 * plan.roi_count &&
        length(workspace.reference_centroids) == 2 * plan.roi_count || throw(
        DimensionMismatch("lamp reference and scratch dimensions must match the plan"),
    )
    length(result.eligible) == plan.roi_count &&
        length(result.minimum_flux) == plan.roi_count &&
        length(result.all_training_valid) == plan.roi_count || throw(
        DimensionMismatch("lamp diagnostics dimensions must match the plan"),
    )
    outputs = (result.reference_centroids, result.eligible, result.minimum_flux,
        result.all_training_valid, workspace.reference_centroids)
    inputs = (input.centroids, input.flux, input.intrinsic_valid, input.raw_adc,
        plan.minimum_flux)
    _any_cross_alias(outputs, inputs) && throw(ArgumentError(
        "lamp product and scratch must not alias inputs or prepared thresholds",
    ))
    _any_pair_alias(outputs) && throw(ArgumentError(
        "lamp product and workspace arrays must not alias",
    ))
    return nothing
end

"""
    process!(result, workspace, plan::LampReferencePlan, inputs)

Average individual deployed centroid samples in Float64, preserving their
interleaved x/y measurement order. For ROI `k`, eligibility is
`all(intrinsic_valid[k, :]) && minimum(flux[k, :]) ≥ minimum_flux[k]`.
Every raw ADC value must be strictly below the declared upper rail; a rail
failure invalidates the complete product without discarding exposures.
Non-finite flux, centroid input, or intermediate mean also yields `Invalid`.
All ROI diagnostics are computed even for invalid batches. Each call resets
scratch; no history, frame selection, or reference update is retained.
Ordinary dense Float32/Float64 centroids and flux with unsigned raw ADC arrays
and a machine integer or Float32/Float64 ADC rail perform zero Julia heap
allocations after warmup using caller-owned output and workspace. This is a
scalar CPU implementation with one-based indexing.
"""
function process!(result::LampReferenceProduct, workspace::LampReferenceWorkspace,
                  plan::LampReferencePlan, input::LampReferenceInputs)
    _validate_lamp(result, workspace, plan, input)
    fill!(workspace.reference_centroids, 0.0)
    fill!(result.minimum_flux, Inf)
    fill!(result.all_training_valid, true)
    finite, below_rail, adc_maximum = _update_lamp!(workspace.reference_centroids,
        result.minimum_flux, result.all_training_valid, plan, input,
        0, plan.sample_count, 0.0)
    return _lamp_product!(result, workspace.reference_centroids,
        result.minimum_flux, result.all_training_valid, plan, adc_maximum,
        finite, below_rail)
end

# Use the global sample index even when input arrives in separate chunks.
function _update_lamp!(reference_centroids, minimum_flux, all_training_valid,
                       plan, input, count, samples, adc_maximum)
    finite = true
    below_rail = true
    for exposure in 1:samples
        n = count + exposure
        for column in 1:plan.columns, row in 1:plan.rows
            value = input.raw_adc[row, column, exposure]
            adc_maximum = max(adc_maximum, Float64(value))
            below_rail &= value < plan.adc_upper_rail
        end
        for roi in 1:plan.roi_count
            flux = Float64(input.flux[roi, exposure])
            minimum_flux[roi] = min(minimum_flux[roi], flux)
            all_training_valid[roi] &= input.intrinsic_valid[roi, exposure]
            finite &= isfinite(flux)
        end
        for measurement in 1:(2 * plan.roi_count)
            value = Float64(input.centroids[measurement, exposure])
            mean = reference_centroids[measurement] +
                (value - reference_centroids[measurement]) / n
            reference_centroids[measurement] = mean
            finite &= isfinite(value) && isfinite(mean)
        end
    end
    return finite, below_rail, adc_maximum
end

function _lamp_product!(result, reference_centroids, minimum_flux,
                        all_training_valid, plan, adc_maximum, finite, below_rail)
    copyto!(result.minimum_flux, minimum_flux)
    copyto!(result.all_training_valid, all_training_valid)
    result.adc_maximum = adc_maximum
    if !finite || !below_rail || !isfinite(adc_maximum)
        fill!(result.reference_centroids, 0.0)
        fill!(result.eligible, false)
        result.status = Invalid
        return result
    end
    any_eligible = false
    @inbounds for roi in 1:plan.roi_count
        eligible = result.all_training_valid[roi] &&
            result.minimum_flux[roi] >= plan.minimum_flux[roi]
        result.eligible[roi] = eligible
        any_eligible |= eligible
        x, y = 2 * roi - 1, 2 * roi
        result.reference_centroids[x] = eligible ? reference_centroids[x] : 0.0
        result.reference_centroids[y] = eligible ? reference_centroids[y] : 0.0
    end
    result.status = any_eligible ? Valid : NoEligible
    return result
end

"""
    LampReferenceState(plan::LampReferencePlan)

Own an empty lamp-reference history with a copied prepared contract, including
thresholds and the ADC rail's numeric representation. `count` exposures have
Float64 mean `reference_centroids` in absolute interleaved detector-pixel x/y
order, per-ROI `minimum_flux` and `all_training_valid`, and global
`adc_maximum`. `finite` and `below_rail` are sticky diagnostics. No input arrays
are retained. State owns dense CPU arrays; do not modify its bound plan or
history between updates. Construct fresh state to reset; independent state
merging is not supported.
"""
mutable struct LampReferenceState{R<:Real}
    const plan::LampReferencePlan{R}
    count::Int
    const reference_centroids::Vector{Float64}
    const minimum_flux::Vector{Float64}
    const all_training_valid::Vector{Bool}
    adc_maximum::Float64
    finite::Bool
    below_rail::Bool
    function LampReferenceState(plan::LampReferencePlan{R}) where {R<:Real}
        bound = LampReferencePlan(plan.rows, plan.columns, plan.roi_count,
            plan.sample_count, copy(plan.minimum_flux), plan.adc_upper_rail)
        return new{R}(bound, 0, zeros(2 * plan.roi_count),
            fill(Inf, plan.roi_count), fill(true, plan.roi_count), 0.0, true, true)
    end
end

function _validate_lamp_state(state::LampReferenceState, plan::LampReferencePlan)
    bound = state.plan
    (bound.rows, bound.columns, bound.roi_count, bound.sample_count) ==
        (plan.rows, plan.columns, plan.roi_count, plan.sample_count) &&
        isequal(bound.minimum_flux, plan.minimum_flux) &&
        typeof(bound.adc_upper_rail) === typeof(plan.adc_upper_rail) &&
        isequal(bound.adc_upper_rail, plan.adc_upper_rail) || throw(ArgumentError(
        "lamp state was prepared for a different plan",
    ))
    0 <= state.count <= plan.sample_count || throw(ArgumentError(
        "lamp state count is outside its prepared exposure count",
    ))
    length(state.reference_centroids) == 2 * plan.roi_count &&
        length(state.minimum_flux) == plan.roi_count &&
        length(state.all_training_valid) == plan.roi_count || throw(
        DimensionMismatch("lamp state dimensions must match the plan"),
    )
    return nothing
end

"""
    process!(state::LampReferenceState, plan::LampReferencePlan, inputs)

Reduce a nonempty `LampReferenceInputs` chunk (one or more exposures) into
explicit state. Absolute detector-pixel centroids are mandatory. All exposures
contribute to the same means, minimum flux, intrinsic validity and raw unsigned
ADC rail checks as complete batches. One-based arrays and views are supported.
Structural, plan, alias and overrun errors throw before writes. Numerical/rail
failure stays sticky while subsequent exposures complete all diagnostics.
This completes one state update and retains no input binding.
"""
function process!(state::LampReferenceState, plan::LampReferencePlan,
                  input::LampReferenceInputs)
    _validate_lamp_state(state, plan)
    input.centroids_are_absolute || throw(ArgumentError(
        "lamp inputs must be explicitly declared absolute detector-pixel centroids",
    ))
    Base.require_one_based_indexing(input.centroids, input.flux,
        input.intrinsic_valid, input.raw_adc)
    samples = size(input.raw_adc, 3)
    size(input.raw_adc, 1) == plan.rows && size(input.raw_adc, 2) == plan.columns &&
        size(input.centroids) == (2 * plan.roi_count, samples) &&
        size(input.flux) == (plan.roi_count, samples) &&
        size(input.intrinsic_valid) == (plan.roi_count, samples) || throw(
        DimensionMismatch("lamp chunk shape differs from its specification"),
    )
    0 < samples <= plan.sample_count - state.count || throw(ArgumentError(
        "lamp chunk must be nonempty and fit the remaining exposure count",
    ))
    outputs = (state.reference_centroids, state.minimum_flux, state.all_training_valid)
    inputs = (input.centroids, input.flux, input.intrinsic_valid, input.raw_adc,
        plan.minimum_flux, state.plan.minimum_flux)
    _any_cross_alias(outputs, inputs) && throw(ArgumentError(
        "lamp state must not alias inputs or prepared thresholds",
    ))
    _any_pair_alias(outputs) && throw(ArgumentError("lamp state arrays must not alias"))
    finite, below_rail, adc_maximum = _update_lamp!(state.reference_centroids,
        state.minimum_flux, state.all_training_valid, plan, input,
        state.count, samples, state.adc_maximum)
    state.finite &= finite
    state.below_rail &= below_rail
    state.adc_maximum = adc_maximum
    state.count += samples
    return state
end

"""
    process!(result::LampReferenceProduct, workspace, plan::LampReferencePlan, state)

Extract a complete `LampReferenceState` into the existing lamp product without
changing history. The exact declared exposure count is required; incomplete,
mismatched and aliased calls throw before writes. Eligibility, zero inactive
rows, whole-product failure and diagnostics match complete-batch processing.
Workspace remains replaceable and contains no history.
"""
function process!(result::LampReferenceProduct, workspace::LampReferenceWorkspace,
                  plan::LampReferencePlan, state::LampReferenceState)
    _validate_lamp_state(state, plan)
    state.count == plan.sample_count || throw(ArgumentError(
        "lamp state must contain the complete declared exposure count",
    ))
    length(result.reference_centroids) == 2 * plan.roi_count &&
        length(workspace.reference_centroids) == 2 * plan.roi_count &&
        length(result.eligible) == plan.roi_count &&
        length(result.minimum_flux) == plan.roi_count &&
        length(result.all_training_valid) == plan.roi_count || throw(
        DimensionMismatch("lamp product and scratch dimensions must match the plan"),
    )
    outputs = (result.reference_centroids, result.eligible, result.minimum_flux,
        result.all_training_valid, workspace.reference_centroids)
    inputs = (state.reference_centroids, state.minimum_flux, state.all_training_valid,
        plan.minimum_flux, state.plan.minimum_flux)
    _any_cross_alias(outputs, inputs) && throw(ArgumentError(
        "lamp product and workspace must not alias state or prepared thresholds",
    ))
    _any_pair_alias(outputs) && throw(ArgumentError(
        "lamp product and workspace arrays must not alias",
    ))
    return _lamp_product!(result, state.reference_centroids, state.minimum_flux,
        state.all_training_valid, plan, state.adc_maximum, state.finite, state.below_rail)
end
