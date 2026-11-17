"""
    PyramidReference(adc_upper_rail)

Select an arithmetic mean of already-normalized pyramid reconstruction pixels
and its negative reconstructed modal reference. Declare a finite positive raw
ADC upper rail; equality invalidates the whole product. The rail's original
numeric representation is preserved for comparison with unsigned raw samples.
"""
struct PyramidReference{R<:Real} <: AbstractCalibrationMethod
    adc_upper_rail::R
    function PyramidReference(adc_upper_rail::R) where {R<:Real}
        isfinite(adc_upper_rail) && adc_upper_rail > 0 || throw(ArgumentError(
            "adc_upper_rail must be finite and positive",
        ))
        return new{R}(adc_upper_rail)
    end
end

"""
    PyramidReferenceSpecification(rows, columns, selected_pixel_count,
                                  exposure_count, reconstructor; row_units)

Declare positive detector dimensions, selected pixels per pupil and exact
retained exposure count. `reconstructor` has `4 * selected_pixel_count`
pupil-major columns and at least one row. `row_units` explicitly labels each
row's coordinate unit, with one nonempty string per row; no unit conversion is
performed. This cold specification may refer to the caller's matrix until
`prepare`, which validates and owns one finite Float64 copy. Unit labels are
copied to an immutable tuple. Inputs must use one-based indexing.
"""
struct PyramidReferenceSpecification{M<:AbstractMatrix{<:Real},U<:Tuple}
    rows::Int
    columns::Int
    selected_pixel_count::Int
    exposure_count::Int
    reconstructor::M
    row_units::U

    function PyramidReferenceSpecification(rows::Integer, columns::Integer,
            selected_pixel_count::Integer, exposure_count::Integer,
            reconstructor::M; row_units::AbstractVector{<:AbstractString}) where
            {M<:AbstractMatrix{<:Real}}
        Base.require_one_based_indexing(reconstructor, row_units)
        r, c, k, n = Int(rows), Int(columns), Int(selected_pixel_count), Int(exposure_count)
        r > 0 || throw(ArgumentError("rows must be positive"))
        c > 0 || throw(ArgumentError("columns must be positive"))
        k > 0 || throw(ArgumentError("selected_pixel_count must be positive"))
        n > 0 || throw(ArgumentError("exposure_count must be positive"))
        _checked_extent("pyramid raw batch", r, c, n)
        _checked_extent("pyramid reconstruction pixels", 4, k, n)
        size(reconstructor, 2) == 4 * k && size(reconstructor, 1) > 0 || throw(
            DimensionMismatch("reconstructor must have positive rows and 4K columns"),
        )
        length(row_units) == size(reconstructor, 1) || throw(DimensionMismatch(
            "row_units must label each reconstructor row",
        ))
        all(unit -> !isempty(strip(unit)), row_units) || throw(ArgumentError(
            "row_units must contain nonempty unit labels",
        ))
        units = Tuple(String.(row_units))
        return new{M,typeof(units)}(r, c, k, n, reconstructor, units)
    end
end

"""Run-immutable pyramid dimensions, owned Float64 reconstructor and row units."""
struct PyramidReferencePlan{R<:Real,U<:Tuple} <: AbstractCalibrationPlan
    rows::Int
    columns::Int
    selected_pixel_count::Int
    exposure_count::Int
    reconstructor::Matrix{Float64}
    row_units::U
    adc_upper_rail::R
end

function prepare(method::PyramidReference, spec::PyramidReferenceSpecification)
    size(spec.reconstructor, 2) == 4 * spec.selected_pixel_count &&
        size(spec.reconstructor, 1) == length(spec.row_units) || throw(
        DimensionMismatch("reconstructor shape changed after specification"),
    )
    reconstructor = Matrix{Float64}(spec.reconstructor)
    all(isfinite, reconstructor) || throw(ArgumentError(
        "reconstructor must be finite after Float64 conversion",
    ))
    return PyramidReferencePlan(spec.rows, spec.columns, spec.selected_pixel_count,
        spec.exposure_count, reconstructor, spec.row_units, method.adc_upper_rail)
end

"""
    PyramidReferenceInputs(reconstruction_pixels, mean_pupil_intensity, raw_frames)

Bundle dimensionless already-normalized `(4, K, N)` pixels, current calibrated
full-pupil mean intensities of length N, and unchanged unsigned raw ADC frames
of shape `(rows, columns, N)`. Pixels retain pupil and selected-mask ordering.
FGA normalizes with the PREVIOUS successful full-pupil mean, including masked
pixels; `mean_pupil_intensity[n]` describes the CURRENT frame. Orchestration must
discard priming exposures and retain successful-frame order. AOC cannot infer
these acquisition preconditions and never renormalizes pixels. Each mean must
be finite and strictly positive. Single-exposure chunks retain the N=1 axis.
"""
struct PyramidReferenceInputs{
    P<:AbstractArray{<:AbstractFloat,3},
    M<:AbstractVector{<:Real},
    A<:AbstractArray{<:Unsigned,3},
}
    reconstruction_pixels::P
    mean_pupil_intensity::M
    raw_frames::A
end

"""
    PyramidReferenceProduct

Caller-owned Float64 `reference_pixels` of shape `(4, K)` and `modal_reference`
with the plan's explicit row units. The latter equals `−R *
vec(permutedims(reference_pixels))`. `adc_maximum` reports the maximum raw
unsigned ADC converted to Float64, while rail comparisons preserve raw integer
precision. `minimum_mean_pupil_intensity` reports the minimum current calibrated
full-pupil mean; NaN propagates. Any nonfinite input/calculation, nonpositive
mean, or ADC rail violation yields `Invalid` and zeroes both reference arrays.
No exposure is removed and no acceptance threshold is inferred.
"""
mutable struct PyramidReferenceProduct
    status::ReferenceStatus
    reference_pixels::Matrix{Float64}
    modal_reference::Vector{Float64}
    adc_maximum::Float64
    minimum_mean_pupil_intensity::Float64
end

"""Replaceable Float64 pixel-mean scratch, reset for every complete batch."""
struct PyramidReferenceWorkspace
    reference_pixels::Matrix{Float64}
end

allocate_result(plan::PyramidReferencePlan) = PyramidReferenceProduct(
    Invalid, zeros(4, plan.selected_pixel_count), zeros(size(plan.reconstructor, 1)),
    0.0, Inf,
)
allocate_workspace(plan::PyramidReferencePlan) =
    PyramidReferenceWorkspace(zeros(4, plan.selected_pixel_count))

function _validate_pyramid_input(plan, input)
    Base.require_one_based_indexing(input.reconstruction_pixels,
        input.mean_pupil_intensity, input.raw_frames)
    samples = size(input.raw_frames, 3)
    size(input.raw_frames, 1) == plan.rows &&
        size(input.raw_frames, 2) == plan.columns &&
        size(input.reconstruction_pixels) == (4, plan.selected_pixel_count, samples) &&
        length(input.mean_pupil_intensity) == samples || throw(DimensionMismatch(
        "pyramid input shape differs from its specification",
    ))
    return samples
end

function _validate_pyramid_storage(result, workspace, plan, inputs)
    size(result.reference_pixels) == (4, plan.selected_pixel_count) &&
        size(workspace.reference_pixels) == (4, plan.selected_pixel_count) &&
        length(result.modal_reference) == size(plan.reconstructor, 1) || throw(
        DimensionMismatch("pyramid product and scratch dimensions must match the plan"),
    )
    outputs = (result.reference_pixels, result.modal_reference, workspace.reference_pixels)
    _any_cross_alias(outputs, (plan.reconstructor, inputs...)) && throw(ArgumentError(
        "pyramid product and scratch must not alias inputs or prepared reconstructor",
    ))
    _any_pair_alias(outputs) && throw(ArgumentError(
        "pyramid product and scratch arrays must not alias",
    ))
    return nothing
end

# The global exposure index makes every sequential partition use the same recurrence.
function _update_pyramid!(reference_pixels, plan, input, count, samples,
                          adc_maximum, minimum_mean_pupil_intensity)
    finite, positive_mean, below_rail = true, true, true
    for exposure in 1:samples
        n = count + exposure
        for column in 1:plan.columns, row in 1:plan.rows
            value = input.raw_frames[row, column, exposure]
            adc_maximum = max(adc_maximum, Float64(value))
            below_rail &= value < plan.adc_upper_rail
        end
        intensity = Float64(input.mean_pupil_intensity[exposure])
        minimum_mean_pupil_intensity = min(minimum_mean_pupil_intensity, intensity)
        finite &= isfinite(intensity)
        positive_mean &= intensity > 0.0
        for pixel in 1:plan.selected_pixel_count, pupil in 1:4
            value = Float64(input.reconstruction_pixels[pupil, pixel, exposure])
            mean = reference_pixels[pupil, pixel] +
                (value - reference_pixels[pupil, pixel]) / n
            reference_pixels[pupil, pixel] = mean
            finite &= isfinite(value) && isfinite(mean)
        end
    end
    return finite, positive_mean, below_rail, adc_maximum, minimum_mean_pupil_intensity
end

function _pyramid_product!(result, reference_pixels, plan, finite,
        positive_mean, below_rail, adc_maximum, minimum_mean_pupil_intensity)
    result.adc_maximum = adc_maximum
    result.minimum_mean_pupil_intensity = minimum_mean_pupil_intensity
    if finite && positive_mean && below_rail && isfinite(adc_maximum)
        copyto!(result.reference_pixels, reference_pixels)
        fill!(result.modal_reference, 0.0)
        for pupil in 1:4, pixel in 1:plan.selected_pixel_count
            column = (pupil - 1) * plan.selected_pixel_count + pixel
            reference = reference_pixels[pupil, pixel]
            for mode in eachindex(result.modal_reference)
                value = result.modal_reference[mode] - plan.reconstructor[mode, column] * reference
                result.modal_reference[mode] = value
                finite &= isfinite(value)
            end
        end
        if finite
            result.status = Valid
            return result
        end
    end
    fill!(result.reference_pixels, 0.0)
    fill!(result.modal_reference, 0.0)
    result.status = Invalid
    return result
end

"""
    process!(result, workspace, plan::PyramidReferencePlan, input)

Average all declared normalized exposures using a bounded Float64 online mean,
then apply the negative pupil-major reconstructor. All diagnostics are retained
on failure. Scratch is reset every call; no normalization history, acquisition,
frame selection, eligibility mask, unit conversion or activation is performed.
This deterministic scalar CPU implementation accepts one-based arrays/views.
"""
function process!(result::PyramidReferenceProduct, workspace::PyramidReferenceWorkspace,
                  plan::PyramidReferencePlan, input::PyramidReferenceInputs)
    samples = _validate_pyramid_input(plan, input)
    samples == plan.exposure_count || throw(DimensionMismatch(
        "pyramid batch must contain exactly the declared exposure count",
    ))
    _validate_pyramid_storage(result, workspace, plan,
        (input.reconstruction_pixels, input.mean_pupil_intensity, input.raw_frames))
    fill!(workspace.reference_pixels, 0.0)
    diagnostics = _update_pyramid!(workspace.reference_pixels, plan, input,
        0, samples, 0.0, Inf)
    return _pyramid_product!(result, workspace.reference_pixels, plan, diagnostics...)
end

"""
    PyramidReferenceState(plan::PyramidReferencePlan)

Own bounded Float64 `reference_pixels`, `count` and diagnostics for ordered
incremental reduction. `finite`, `positive_mean` and `below_rail` are sticky.
State binds the exact logically immutable plan, sharing its already-owned R
rather than copying a large matrix. Do not modify this plan or the history
between updates. No caller exposure arrays or raw history are retained.
Construct fresh state to reset; independent state merging is unsupported.
"""
mutable struct PyramidReferenceState{R<:Real,U<:Tuple}
    const plan::PyramidReferencePlan{R,U}
    count::Int
    const reference_pixels::Matrix{Float64}
    adc_maximum::Float64
    minimum_mean_pupil_intensity::Float64
    finite::Bool
    positive_mean::Bool
    below_rail::Bool
    function PyramidReferenceState(plan::PyramidReferencePlan{R,U}) where {R,U}
        return new{R,U}(plan, 0, zeros(4, plan.selected_pixel_count),
            0.0, Inf, true, true, true)
    end
end

function _validate_pyramid_state(state, plan)
    state.plan === plan || throw(ArgumentError(
        "pyramid state requires its exact bound prepared plan",
    ))
    0 <= state.count <= plan.exposure_count || throw(ArgumentError(
        "pyramid state count is outside its prepared exposure count",
    ))
    size(state.reference_pixels) == (4, plan.selected_pixel_count) || throw(
        DimensionMismatch("pyramid state dimensions must match the plan"),
    )
    return nothing
end

"""
    process!(state::PyramidReferenceState, plan::PyramidReferencePlan, input)

Reduce a nonempty one-or-more-exposure chunk into explicit state with the same
global-index recurrence as complete batches. Structural, alias, plan and count
errors throw before writes. Numerical/rail failures remain sticky, while all
subsequent exposures still contribute to diagnostics and count. Inputs are
never retained. Extraction requires the exact declared count.
"""
function process!(state::PyramidReferenceState, plan::PyramidReferencePlan,
                  input::PyramidReferenceInputs)
    _validate_pyramid_state(state, plan)
    samples = _validate_pyramid_input(plan, input)
    0 < samples <= plan.exposure_count - state.count || throw(ArgumentError(
        "pyramid chunk must be nonempty and fit the remaining exposure count",
    ))
    _any_cross_alias((state.reference_pixels,),
        (input.reconstruction_pixels, input.mean_pupil_intensity,
         input.raw_frames, plan.reconstructor)) && throw(ArgumentError(
        "pyramid state must not alias inputs or prepared reconstructor",
    ))
    finite, positive_mean, below_rail, maximum, minimum = _update_pyramid!(
        state.reference_pixels, plan, input, state.count, samples,
        state.adc_maximum, state.minimum_mean_pupil_intensity)
    state.finite &= finite
    state.positive_mean &= positive_mean
    state.below_rail &= below_rail
    state.adc_maximum = maximum
    state.minimum_mean_pupil_intensity = minimum
    state.count += samples
    return state
end

"""
    process!(result, workspace, plan::PyramidReferencePlan, state::PyramidReferenceState)

Extract a complete explicit-state product without advancing history. The exact
bound plan and declared count are required. Output and workspace must be
independent of state and coefficients. Whole-product failure, sign, order and
diagnostics match complete-batch processing; workspace carries no history.
"""
function process!(result::PyramidReferenceProduct, workspace::PyramidReferenceWorkspace,
                  plan::PyramidReferencePlan, state::PyramidReferenceState)
    _validate_pyramid_state(state, plan)
    state.count == plan.exposure_count || throw(ArgumentError(
        "pyramid state must contain the complete declared exposure count",
    ))
    _validate_pyramid_storage(result, workspace, plan, (state.reference_pixels,))
    return _pyramid_product!(result, state.reference_pixels, plan, state.finite,
        state.positive_mean, state.below_rail, state.adc_maximum,
        state.minimum_mean_pupil_intensity)
end
