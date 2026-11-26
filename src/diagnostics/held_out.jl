# Package-defined held-out residual and pupil RMS metrics. See docs/REFERENCES.md.
"""
    HeldOutReconstruction()

Evaluate recovery of a finite, nonzero held-out command in modal order.
Prepare with the number of modes and process a tuple of branch reconstructor
matrices, corresponding measurement residuals, and the commanded coefficients.
The estimate accumulates the branch corrections in their supplied order in the
reconstructor element type. The error is relative to the commanded norm.
"""
struct HeldOutReconstruction <: AbstractCalibrationMethod end

struct HeldOutReconstructionPlan <: AbstractCalibrationPlan
    modes::Int
end

function prepare(::HeldOutReconstruction, modes::Integer)
    count = _checked_extent("mode count", modes)
    count > 0 || throw(ArgumentError("mode count must be positive"))
    return HeldOutReconstructionPlan(count)
end

function process(
    plan::HeldOutReconstructionPlan,
    (reconstructors, residuals, command)::Tuple{Tuple,Tuple,AbstractVector},
)
    !isempty(reconstructors) && length(reconstructors) == length(residuals) ||
        throw(ArgumentError("each branch needs a reconstructor and measurement residual"))
    length(command) == plan.modes ||
        throw(DimensionMismatch("held-out command must match the prepared mode count"))
    magnitude = norm(command)
    isfinite(magnitude) && magnitude > 0 ||
        throw(ArgumentError("held-out mixture must have finite nonzero norm"))
    estimate = zeros(eltype(first(reconstructors)), plan.modes)
    for (matrix, residual) in zip(reconstructors, residuals)
        size(matrix) == (plan.modes, length(residual)) ||
            throw(DimensionMismatch("branch reconstructor does not match its residual"))
        estimate .+= matrix * residual
    end
    return (; estimate, reconstruction_error=norm(estimate - command) / magnitude)
end

"""
    PupilResidualRatio()

Measure the corrected-to-uncorrected piston-removed RMS ratio over the same
pupil support. Prepare with the number of supported pixels and process
`(before, after)` vectors of OPD relative to the same flat reference.
The before norm is floored at `eps(Float64)`, preserving the ratio convention of the original optical calibration.
"""
struct PupilResidualRatio <: AbstractCalibrationMethod end

struct PupilResidualRatioPlan <: AbstractCalibrationPlan
    pixels::Int
end

function prepare(::PupilResidualRatio, pixels::Integer)
    count = _checked_extent("supported pixel count", pixels)
    count > 0 || throw(ArgumentError("supported pixel count must be positive"))
    return PupilResidualRatioPlan(count)
end

function process(
    plan::PupilResidualRatioPlan,
    (before, after)::Tuple{AbstractVector,AbstractVector},
)
    length(before) == length(after) == plan.pixels ||
        throw(DimensionMismatch("residual vectors must match the prepared pupil support"))
    return piston_removed_norm(after) / max(piston_removed_norm(before), eps(Float64))
end

piston_removed_norm(residual) = norm(residual .- sum(residual) / length(residual))
