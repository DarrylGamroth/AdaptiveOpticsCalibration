# Calibration package migration

AdaptiveOpticsCalibration owns reusable calibration mathematics. AdaptiveOpticsSim
owns the physical plant, detector simulation and simulated acquisition.
FilterGraphAlgorithms owns the graph adapters and real-time execution contracts.

Use `prepare(method, specification)` followed by `process(plan, inputs)`.
For repeated execution, use the in-place interface and its reusable workspace.
See [Design](DESIGN.md) for the ownership and numerical contracts, and
[Calibration catalogue](docs/CALIBRATION_CATALOGUE.md) for method provenance,
scientific references and fixture coverage.

The optical reconstruction methods originate from PipeWireAORTC at `13de882`,
in `ext/optical_calibration.jl` and `ext/optical_measurements.jl`. The initial
package split used AdaptiveOpticsSim at `d30db3f`, FilterGraphAlgorithms at
`6257cac`, HEART at `d5ba594` and SPECULA at `e52b12c` for source review
and comparison. The [sharpening design](docs/DESIGN_SHARPENING.md) records
the source signature and equations for the focal-plane methods.
