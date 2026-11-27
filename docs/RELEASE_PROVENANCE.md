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

### Imported code and parity fixtures

The package's MIT notice covers the distributed release. The historical
source snapshots below predate a top-level license notice in their respective
trees. The recorded source authorship and the maintainer's MIT release
confirmation provide the provenance for the adaptations identified here.

| Release material | Source evidence |
| --- | --- |
| Accelerator von Kármán Bessel approximation | The extension adapts [`kv56.jl` in AdaptiveOpticsSim at `b4d2da2`](https://github.com/DarrylGamroth/AdaptiveOpticsSim.jl/blob/b4d2da2/src/core/kv56.jl). Darryl Gamroth is the recorded author of the file at source commit `e5449f68`. |
| Optical reconstruction and calibration | [The migration record](../MIGRATION.md) identifies the PipeWireAORTC extraction at `13de882`. The recorded source changes to the optical calibration and measurement files include Darryl Gamroth's commit `b0b51d3`. |
| Numerical parity fixtures | The [TOML fixtures](../test/fixtures/) record their source revisions, upstream paths, and numerical layouts. AOS fixtures record model or calibration cases. JuliaFilterGraph fixtures retain the recorded upstream test inputs and expected numerical products. |

The public v0.17.7 tree contains the package source, documentation, and
numerical tests. Instrument optical prescriptions, vendor catalogs, and
measured calibration bundles belong to the separate instrument repository.

The MIT notice covers the package distribution. Julia installs dependencies
and binary artifacts separately under their own terms. For example, the
native FFTW library is distributed under the
[GNU GPL](https://www.fftw.org/faq/section1.html). An application bundle that
redistributes dependencies must account for their licenses.

## Registered history

The active registry starts at v0.17.5. The public Git tags v0.17.5 and v0.17.6
match their original registered trees exactly. Earlier versions were retired
from the active registry when it moved to this repository.
