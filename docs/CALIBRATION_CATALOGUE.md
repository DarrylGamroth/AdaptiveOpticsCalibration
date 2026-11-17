# Calibration algorithm inventory

Reviewed 2026-10-02 against AOC `875a7c2` plus `ReferenceFrames` (`3c11576`), spatial sinusoidal probes
(`d2c27a8`) and focal-plane sharpening (`ecd4fab`). This catalogue separates implemented numerical methods, external
references and selected additions. It is not instrument qualification evidence.

## Existing and selected AOC methods

| Method | AOC state | Operational meaning |
| --- | --- | --- |
| Zonal push–pull | Probe generation and interaction estimation implemented | Selected Classic procedure: ±0.02 µm OPD on each of 277 physical actuators; mean deployed WFS response difference divided by twice the amplitude. Precision acceptance is separate. |
| Hadamard push–pull | Probe generation and decoding implemented | The balanced implementation uses an order strictly greater than the coordinate count; 277 coordinates require 512 patterns and 1,024 signed commands. This is not automatically faster than 554 zonal commands. |
| Modal push–pull | Supplied modal command rows can generate probes | A supplied basis is not an inferred physical actuator geometry or a generic modal-response decoder. |
| Spatial sine/cosine modes | Scalar CPU probe construction implemented and package-tested | Explicit actuator positions, reciprocal-coordinate spatial frequencies, amplitudes and normalization. Distinct from temporal sine modulation/demodulation. Incomplete modal probes do not identify a full physical-coordinate matrix. |
| Dark/reference windows | `ReferenceFrames` implemented and package-tested | Float64 dark means/unbiased variances; mean individual absolute deployed centroids; declared per-ROI all-valid/minimum-flux and global ADC-rail diagnostics. Association and acceptance remain caller-owned. |
| Sensorless sharpening | Scalar CPU numerical operations implemented and package-tested | SPIDERS-compatible image merit, supplied SPSA probe pair/gradient and explicit-state Adam step. No live equipment callback or optimization campaign in the numerical package. |
| DOCRIME | Covariance-based numerical estimators implemented | Acquisition, excitation, delays and covariance accumulation need an instrument-specific declared procedure. Not currently selected by the deployed Classic campaign. |
| Optical gain | Complete-image gain sensing implemented | Supplied reference/current frames and modal basis; separate from temporal carrier demodulation and controller gain optimization. |
| Misregistration | Complete-data meta-sensitivity estimator implemented | Supplied response matrices and ordered sensitivities; RTC/plant acquisition and physical parameter interpretation remain outside AOC. |
| LiFT | Iterative focal-plane phase retrieval implemented | Requires a supplied forward model and conventions; distinct from optimizing an image-sharpness metric. |
| Reconstructors | SVD, Tikhonov, covariance-weighted MAP and stroke-weighted methods implemented | Require accepted input matrices, covariances, priors, coordinates and numerical policy. They do not validate an interaction acquisition by themselves. |

## External primary references

### pyRTC

Inspected revision `1c0a55e81433eb7499808cac9c2641d8d6b4cfd7`.
[Loop.py](https://github.com/jacotay7/pyRTC/blob/1c0a55e81433eb7499808cac9c2641d8d6b4cfd7/pyRTC/Loop.py)
implements modal push–pull with averaged positive/negative slopes and a
DO-CRIME covariance path with command-delay association. These support the
existing AOC method families rather than justify duplicating them. Its command
units and acquisition assumptions must be supplied explicitly in an AOC use
case. Review source license before reusing implementation text; this inventory
does not authorize importing GPL implementation into AOC.

### SPECULA

Inspected revision `e52b12cd6751568e6daad559c85c28f981ee58c2`.

- [Modal probe sequence](https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/e52b12cd6751568e6daad559c85c28f981ee58c2/specula/lib/modal_pushpull_signal.py)
  supports alternating signals and mode-order amplitude schedules. Such
  schedules are explicit scientific choices, not new defaults for AOC.
- [SPRINT estimator](https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/e52b12cd6751568e6daad559c85c28f981ee58c2/specula/processing_objects/base_sprint_estimator.py)
  and [demodulation](https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/e52b12cd6751568e6daad559c85c28f981ee58c2/specula/lib/demodulate_signal.py)
  are useful references for temporal carriers, optical gain and misregistration
  sensitivity. Their SH geometry and state updates require separate contracts.
- [MMSE reconstructor](https://github.com/ArcetriAdaptiveOptics/SPECULA/blob/e52b12cd6751568e6daad559c85c28f981ee58c2/specula/lib/mmse_reconstructor.py)
  provides a covariance/prior reference for AOC's existing MAP family.

SPECULA's MIT license permits attributed reuse subject to its terms. No code
was copied or instrument experiment run for this inventory.

### MagAO-X

Inspected repository revision `66a6087f7c4dbb7197468c43c9259d3e8b0350d3`.

- [DM mode generation](https://github.com/magao-x/MagAOX/blob/66a6087f7c4dbb7197468c43c9259d3e8b0350d3/magaox-python/magaox/deformable_mirror.py)
  includes Fourier and Gaussian-poke helpers. The named `XHadamardMirror`
  ends in an incomplete generator and commented parent constructor at this
  revision; it is not evidence of a usable Hadamard acquisition method.
  Instrument command normalization is separate from a transport-neutral basis
  definition.
- [Eye Doctor handbook](https://magao-x.org/docs/handbook/operating/software/utils/eyedoctor.html)
  describes mode-wise focal-plane scans and quadratic fitting. This is a
  documented workflow; no extracted implementation or hardware qualification
  is established here.
- [Modal PSD processing](https://github.com/magao-x/MagAOX/blob/66a6087f7c4dbb7197468c43c9259d3e8b0350d3/apps/modalGainOpt/modalPsdProcessor.hpp)
  offers noise-floor and disturbance-spectrum methods for future temporal gain
  optimization. PSD units, frequency axes and spectral sampling need their own
  contract; modal PSD statistics are not pixel dark statistics.

MagAO-X repository code is GPL-3.0; numerical equations and cited descriptions
can guide independent implementations, but copying code requires license
review. No instrument actions or gain changes are authorized by this inventory.

## SPIDERS sharpening reference

The local `spiders/scripts/focal-plane-sharpen.jl` has SHA-256
`fe2dc48de82539bfbd0c4a6a67ee5032e3c2374de00d7d364ce13663ea8361c5`.
A reproducible public revision/environment and the FITS basis units were not
established. The selected extraction independently implements the equations
reviewed in [the sharpening design](DESIGN_SHARPENING.md): median-filtered
weighted core intensity, SPSA and Adam. It excludes live Aeron, waits, implicit
frame association, file replacement and the script's clamp-then-mean command
construction. Removing the mean after clipping can exceed the earlier bounds.

## Priority and evidence limits

### RTC calibration method selection

The GPI calibration discussion on 2026-10-28 retains these method families as
toolkit options. Existing numerical implementations above should be reused;
their presence does not mean every method is exposed or qualified by an RTC
recipe. Keep the spatial excitation basis separate from its temporal schedule:
a Fourier sine/cosine mode can be measured with signed push–pull steps or,
with a separately implemented estimator, temporal sinusoidal modulation.

| Option | Intended use and remaining integration |
| --- | --- |
| Sequential push–pull | Retain the full-matrix baseline and compare amplitude sensitivity, repeatability and held-out loop behavior. |
| Hadamard push–pull | Reuse existing probe/decoding methods; compare equal precision and command constraints, not only pattern count. A multiplexed pattern must remain in the WFS linear regime. |
| Temporal sine probes | Future carrier generation and demodulation for response/dynamics or optical-gain tracking; specify sample rate, frequency, phase, settling, latency and averaging explicitly. |
| Model-assisted updates | Reuse applicable sensitivity estimators with a validated model and selected measured responses; qualify registration and optical-gain updates separately from full-matrix acquisition. |

Push–pull is a central response difference, not a guarantee of the most accurate
calibration. Noise, finite amplitude, drift and the pyramid operating point
affect accuracy. Relevant primary references are the
[pyramid optical-gain analysis](https://doi.org/10.1051/0004-6361/202037836),
[slope-oriented Hadamard study](https://doi.org/10.1364/OE.23.027134), and
[SPRINT](https://academic.oup.com/mnras/article/504/3/4274/6274481).
RTC sequencing and adoption remain outside this numerical package. A CPU
simulation's propagation time is not a prediction of instrument acquisition
time; GPU acquisition is a backend choice rather than a calibration method.

1. Spatial sine/cosine and the SPIDERS numerical extraction have scalar
   references, ordinary-array tests, attribution and independent review. See
   [the numerical review](CALIBRATION_METHODS_REVIEW.md) for corrected defects
   and software validation limits.
2. Qualify each chosen method through completion-driven commanded/adopted
   probes and actual detector observations. Do not replace the measured zonal
   baseline merely because another basis uses fewer or multiplexed patterns.
3. Consider Eye Doctor-style modal scanning as another separate optimizer once
   the image metric, command coordinates and noise policy are accepted.
4. Temporal lock-in/SPRINT and PSD gain optimization remain documented future
   candidates. User selection in this increment is spatial sine/cosine modes.

Scientific suitability, noise efficiency, convergence, throughput and safe
physical operation are not inferred from another project's source or docs.
