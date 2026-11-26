# Release provenance

`AdaptiveOpticsCalibration` uses UUID
`3c8b5851-926e-4ebb-af30-f8544b98d45f`. This repository publishes the Julia
package under the [MIT license](../LICENSE) and is the source for the active
Julia registry entries. The separate private development repository retains
the earlier integration history and historical release trees.

## v0.17.7

This release uses the implementation, accelerator extensions, and test fixtures
from the public MIT v0.17.6 tree `0e7977c0e00cb44788f7fe52eb4506652d0ebd22`.
The release adds attribution comments to the existing algorithm files and an
[algorithm reference index](REFERENCES.md). The MIT notice remains unchanged.
The package runs Aqua and tests the portable optical-gain and covariance
reconstruction kernels on the CPU through internal plan constructors. The
public backend checks and numerical operations remain as in v0.17.6.

[MIGRATION.md](../MIGRATION.md) records the implementation sources and numerical
validation history. References to papers credit scientific methods. Any future
source or fixture import needs its own redistribution and notice review.

The accelerator von Kármán covariance extension identifies Darryl Gamroth's
[`kv56.jl` in AdaptiveOpticsSim at `b4d2da2`](https://github.com/DarrylGamroth/AdaptiveOpticsSim.jl/blob/b4d2da2/src/core/kv56.jl)
as the source of its Bessel approximation. This release retains the adaptation
already distributed under the calibration package's MIT notice.

## Registered history

The active registry starts at v0.17.5. The public Git tags v0.17.5 and v0.17.6
match their original registered trees exactly. Earlier versions were retired
from the active registry when it moved to this repository.
