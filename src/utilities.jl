function _checked_extent(name::AbstractString, factors::Integer...)
    extent = 1
    try
        for factor in factors
            extent = Base.Checked.checked_mul(extent, Int(factor))
        end
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("$name extent overflows Int"))
    end
    return extent
end

function _hadamard_order(corrector_coordinate_count::Integer)
    count = Int(corrector_coordinate_count)
    count > 0 || throw(ArgumentError(
        "corrector_coordinate_count must be positive",
    ))
    target = try
        Base.Checked.checked_add(count, 1)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("Hadamard order overflows Int"))
    end
    order = 1
    while order < target
        order = try
            Base.Checked.checked_mul(order, 2)
        catch error
            error isa OverflowError || rethrow()
            throw(ArgumentError("Hadamard order overflows Int"))
        end
    end
    return order
end

function _float_type(::Type{T}, operation::AbstractString) where {T<:Real}
    F = float(T)
    F <: AbstractFloat || throw(ArgumentError(
        "$operation requires an AbstractFloat representation",
    ))
    return F
end

@inline _mightalias_any(::AbstractArray, ::Tuple{}) = false
@inline function _mightalias_any(array::AbstractArray, arrays::Tuple)
    return Base.mightalias(array, first(arrays)) ||
        _mightalias_any(array, Base.tail(arrays))
end

@inline _any_pair_alias(::Tuple{}) = false
@inline function _any_pair_alias(arrays::Tuple)
    remaining = Base.tail(arrays)
    return _mightalias_any(first(arrays), remaining) ||
        _any_pair_alias(remaining)
end

@inline _any_cross_alias(::Tuple{}, ::Tuple) = false
@inline function _any_cross_alias(arrays::Tuple, others::Tuple)
    return _mightalias_any(first(arrays), others) ||
        _any_cross_alias(Base.tail(arrays), others)
end
