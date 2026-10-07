# yi-corr multi-sideband input (frinZ 5.4.0)

`frinZ --in joint.mbcor` reads yi-corr `YIMBCOR\0` version 1 directly. This
format retains the native COR headers and spectra for both bands. No RAW data
or temporary conversion files are needed. It also accepts `.mbcor.zst`.

```bash
frinZ --in YAMAGU32_YAMAGU34_2026280081000_joint.mbcor

# Thirty consecutive ten-sector windows; 10 sectors = 10 s at 1 Hz output.
frinZ --in YAMAGU32_YAMAGU34_2026280081000_joint.mbcor \
  --length 10 --loop 30 --search peak --add-plot --npz --cpu 6
```

The joint search uses 6600--7112 and 8192--8704 MHz as two separate frequency
grids. The 1080 MHz gap contributes no noise weight: occupied bandwidth is
1024 MHz, RF span is 2104 MHz. Zero bins are excluded. Spectral coordinates
are obtained from the embedded headers, rather than hard-coded C/X labels.
The two bands must have equal channel spacing and lie on one common frequency
grid. Integration sectors must be contiguous and uniformly spaced within each
window, with identical timestamps/durations in both bands. Metadata, complete
payload size, non-finite samples, and unsupported layouts are checked.

The common matched-filter model is

```text
V(f,t) = A(f) exp[2 pi i ((f - f_ref) delay + (f / f_ref) rate (t - t_mid))]
```

For this observation `f_ref = 7652 MHz`. The reported `rate_hz` is the fringe
rate at that RF, and `delay_rate_s_per_s = rate_hz / reference_hz`. Phase is
referenced to the window midpoint. All bands share one delay and delay rate.
The coarse search uses time FFTs and sparse-frequency delay FFTs, followed by
exact coherent sums to refine delay/rate. Padding gaps do not change visibility
normalization. No per-target IF-phase fitting is performed: transfer the
strong-calibrator solution using yi-corr before analyzing faint sources.

`peak`, `deep`, `deep2` and `coherent` select this same RF-aware joint search.
`--iter` controls exact refinement (default 5), and `--rate-padding` accepts
1, 2, 4, 8 (default 8). `--cpu` limits this search's Rayon pool. `--length`
counts correlator sectors; `--skip` is seconds and must align to a sector.
`--loop` windows must fit the input; no partial last window is silently used.
With no `--length`, one window spans the remaining file.

Delay corrections and `--drange` retain the ordinary CLI's sample units, using
the **low band's sampling speed**. Rate corrections and `--rrange` are Hz at
the joint reference RF. `--frange` and `--rfi MIN,MAX` use MHz relative to the
low-band RF edge (X starts at 1592 MHz here). `--rfi` ranges have an exclusive
upper bound. Other workflows, including rate/acel polynomial fitting, ACF
normalization, bandpass correction, frequency-rate heatmaps/FITS, and COR
rewriting, currently reject MBCOR input with an explicit message.

Outputs always go under the input's `frinZ/mbcor/` directory:

* `*_joint.tsv`: joint delay/rate, phase, amplitude, occupied/usable bandwidth,
  per-band complex mean amplitude and phase, and the noise proxy below.
* `*_wNNNN_spectrum.tsv`: RF and complex spectral means after joint search.
* `*_wNNNN_time.tsv`: timestamps and per-band/joint complex continuum means.
* `*_wNNNN_phase.png`, `*_spectrum.png`, `*_delay.png`, `*_rate.png`:
  phase stability, RF spectrum, sparse-band delay sidelobes, common rate profile.
* `--add-plot`: per-window phase plot, referenced to each window midpoint.
* `--npz` or `--spectrum`: named NumPy arrays containing physical RFs, the
  complex spectrum/time series, and delay/rate amplitude profiles. The rate
  profile is the maximum over searched delay; it is not a 2-D plane.
* `--header`: print both bands' metadata and stop.

`sigma_diff` estimates quadrature noise of the joint mean from adjacent
time differences of occupied-channel visibilities:
`sqrt(mean(|V[n+1]-V[n]|^2) / (4 * usable_sample_count))`.
It assumes independent channel/time noise; phase wander and source variations
contribute to the estimate. `snr_diff = amplitude / sigma_diff` is a diagnostic
proxy, **not** the legacy delay-plane SNR estimator. Do not compare these
two SNR columns as if they use the same definition. Statistics on the same
calibrator used to fit phase are not an independent faint-target sensitivity
measurement. Unequal band responses/noise also mean the measured gain need
not be sqrt(2). Amplitudes remain uncalibrated COR units.
