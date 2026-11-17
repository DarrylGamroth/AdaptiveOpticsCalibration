function copy_to_backend(backend, source)
    destination = KernelAbstractions.allocate(
        backend,
        eltype(source),
        size(source),
    )
    copyto!(destination, source)
    return destination
end

function require_backend(backend, arrays...)
    for array in arrays
        KernelAbstractions.get_backend(array) == backend || throw(ArgumentError(
            "array backend does not match the prepared KernelExecution backend",
        ))
    end
    return nothing
end

function require_one_based(arrays...)
    Base.require_one_based_indexing(arrays...)
    return nothing
end
