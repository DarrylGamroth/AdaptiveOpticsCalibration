# LLM Agent Instructions for AdaptiveOpticsCalibration.jl

## Purpose

This package owns reusable scientific algorithms for adaptive-optics
calibration. Read `DESIGN.md` before changing the public interface and
`MIGRATION.md` before extracting an implementation from another repository.

The package is a numerical toolbox. It is not an RTC, plant simulator, graph
engine, equipment-control service, artifact store, or commissioning workflow.

## Public interface

Algorithms use the package-owned multiple-dispatch interface:

```julia
plan = prepare(method, specification)
result = process(plan, inputs)
process!(result, plan, inputs)
process!(result, workspace, plan, inputs)
```

`allocate_result` and `allocate_workspace` are advanced public, non-exported
extension seams. Routine examples should use `process` or the three-argument
`process!` form.

- Do not add algorithm-qualified entry points such as
  `prepare_map_reconstructor` or `measure_zonal_interaction_matrix!`.
- Put concrete scientific types in domain modules when namespacing is useful.
- Add a new generic function only when it represents a genuinely different
  semantic operation that cannot satisfy the existing contract.
- Keep public functions and types documented.

## Ownership

- A method is cold scientific configuration.
- A plan is a logically immutable prepared numerical contract.
- Persistent values that can affect later results are state.
- Workspace is replaceable scratch; recreating it cannot change a result.
- Products are caller-visible scientific results and diagnostics.
- Plans and workspaces must not retain transient caller input or product
  bindings unless a separate prepared owner explicitly documents that role.

Do not disguise state as workspace. Do not add `commit!`, `discard!`,
progressive publication, runtime properties, graph capture, or transport
metadata to the common interface. Such behavior belongs in FGA, an SRT, or a
separately justified algorithm-specific owner.

## Scientific contracts

- State equations, units, coordinate systems, bases, ordering, signs, and
  normalization in documentation and tests.
- Use the canonical AO term rather than a generic software synonym.
- Treat AOS, FGA, HEART, and SPECULA as sources to verify, not authorities to
  copy blindly.
- Record source revisions and parameter mappings when migrating algorithms.
- Preserve attribution and licensing for derived implementations and fixtures.

## Implementation

- Use multiple dispatch and parametric concrete types.
- Do not store `Any`, abstract element types, or uninstantiated parametric
  families in execution owners.
- Provide a deterministic scalar CPU implementation as the scientific
  reference.
- Add KernelAbstractions or vendor extensions only for a measured use case.
  Keep optional accelerator dependencies in package extensions.
- Avoid scalar indexing on accelerators and undeclared host fallbacks.
- Preallocate repeatable numerical work where practical, but do not claim
  universal zero allocation or hard real-time behavior.
- Keep core free of file formats and serialization policy.

## Validation

- Add focused CPU tests for normal, boundary, and invalid inputs.
- Use justified numerical tolerances and deterministic fixtures.
- Require cross-package parity before removing a source implementation.
- Test every claimed accelerator on actual supported hardware; CPU tests do not
  constitute accelerator validation.
- Run `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'` before
  handing off a change.

## Migration boundary

- AdaptiveOpticsSim retains physical plant propagation, detector simulation,
  truth products, and simulated interaction-matrix acquisition.
- FilterGraphAlgorithms retains bounded graph adapters, ports, schemas,
  transactional state, progressive accumulation, and runtime publication.
- The SRT retains equipment sequencing, telemetry association, artifact
  provenance, acceptance, persistence, and activation.
- This package owns only reusable scientific calibration calculations and
  diagnostics.
