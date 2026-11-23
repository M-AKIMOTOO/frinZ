#!/usr/bin/env python3
"""Independent wall-clock bins for single-baseline fringe measurements.

Requires numpy and matplotlib, plus a built frinZ CLI. Values remain in frinZ's
uncalibrated amplitude-percent convention, not Jy. Off-fringe error estimates
assume circular Gaussian noise; calibration systematics are not included.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def utc(ns):
    return datetime.fromtimestamp(int(ns) / 1e9, timezone.utc).isoformat()


def normalized_integrations(values):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values) | (values <= 0)] = 1.0
    for scale in [1.0, .1, .01, .001, .0001, .00001, .000001]:
        values[np.abs(values - scale) <= .1 * scale] = scale
    return values


@dataclass
class Cor:
    path: Path
    raw: bytes
    fs: int
    frequency: float
    fft: int
    size: int
    records: np.ndarray
    times_ns: np.ndarray
    integration_s: np.ndarray

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        if len(raw) < 256 or raw[:4] != bytes.fromhex('83f9a23e'):
            raise ValueError(f'Invalid .cor header: {path}')
        fs, frequency, fft, count = struct.unpack_from('<idii', raw, 12)
        if fs <= 0 or frequency <= 0 or fft < 4 or fft % 4 or count <= 0:
            raise ValueError(f'Invalid .cor dimensions: {path}')
        size = 128 + 4 * fft
        if len(raw) != 256 + count * size:
            raise ValueError(f'Truncated or trailing .cor payload: {path}')
        dtype = np.dtype({'names': ['seconds', 'nanos', 'integration', 'visibility'],
                          'formats': ['<i4', '<u4', '<f4', ('<c8', (fft // 2,))],
                          'offsets': [0, 4, 112, 128], 'itemsize': size})
        records = np.frombuffer(raw, dtype=dtype, count=count, offset=256)
        times = records['seconds'].astype(np.int64) * 10**9 + records['nanos']
        if np.any(~np.isfinite(records['integration']) | (records['integration'] <= 0)):
            raise ValueError('Unknown integration duration cannot be used for measurement exposure')
        duration = normalized_integrations(records['integration'])
        if np.any(records['nanos'] >= 10**9) or np.any(np.diff(times) <= 0):
            raise ValueError('Sector epochs must be valid and strictly increasing')
        if np.any(np.diff(times) + 1000 < duration[:-1] * 1e9):
            raise ValueError('Overlapping sector integrations')
        if not np.allclose(duration, duration[0], rtol=1e-6, atol=1e-9):
            raise ValueError('Variable sector durations need a weighted fringe extractor')
        if not np.isfinite(records['visibility']).all():
            raise ValueError('Non-finite visibility samples require explicit flagging')
        return cls(path, raw, fs, frequency, fft, size, records, times, duration)

    def write_subset(self, path, lo, hi):
        header = bytearray(self.raw[:256])
        struct.pack_into('<i', header, 28, hi - lo)
        path.write_bytes(header + self.raw[256 + lo*self.size:256 + hi*self.size])


def wall_bins(times_ns, durations_s, width_s, epoch_ns):
    """Disjoint half-open bins, retaining empty bins and the partial last bin."""
    width_ns = int(width_s) * 10**9
    if width_ns <= 0:
        raise ValueError('Bin width must be positive')
    ids = (np.asarray(times_ns) - int(epoch_ns)) // width_ns
    if ids[0] < 0:
        raise ValueError('Bin epoch must precede the first sample')
    windows = []
    for index in range(int(ids[-1]) + 1):
        lo, hi = np.searchsorted(ids, [index, index + 1])
        start = int(epoch_ns) + index * width_ns
        end = start + width_ns
        if hi > lo and np.any(times_ns[lo:hi] + durations_s[lo:hi]*1e9 > end + 1000):
            raise ValueError('An integration straddles a bin boundary; choose an aligned epoch')
        windows.append((index, int(lo), int(hi), start, end))
    return windows


def fringe_statistics(visibility, times_s, fs, frequency, delay_samples, rate_hz):
    """Reconstruct the searched fringe and estimate off-delay noise separately.

    FFT delay bins two native samples apart are used, since only half of the
    spectrum is recorded. Exclude +/-32 native delay samples around the fringe.
    Robust real/imaginary MAD estimates describe statistical noise, not gain or
    phase-transfer errors. At high S/N their component sigma approximates the
    standard error of the measured amplitude.
    """
    n, channels = visibility.shape
    fft = channels * 2
    f = np.arange(channels) * float(fs) / fft
    phase_cycles = f[None, :] * (delay_samples / fs)
    phase_cycles = phase_cycles + rate_hz * np.asarray(times_s)[:, None] * (1 + f/frequency)
    corrected = np.asarray(visibility, dtype=np.complex128) * np.exp(-2j*np.pi*phase_cycles)
    corrected[:, 0] = 0  # frinZ excludes the DC channel
    scale = 100.0 * 512.0 / (fs / 2 / 1e6)
    sample = corrected.sum(axis=1) * scale
    spectrum = corrected.mean(axis=0)
    lag = np.fft.fft(spectrum, n=fft) * scale
    indices = np.arange(fft)
    signed_lag = np.minimum(indices, fft-indices)
    noise = lag[(indices % 2 == 0) & (signed_lag > 32)]
    if len(noise) < 32:
        raise ValueError('FFT too short to estimate off-fringe noise with this exclusion')
    center = np.median(noise.real) + 1j*np.median(noise.imag)
    delta = noise-center
    sig_re = np.median(abs(delta.real)) / .6744897501960817
    sig_im = np.median(abs(delta.imag)) / .6744897501960817
    sigma = np.sqrt((sig_re**2+sig_im**2)/2)
    fringe = sample.mean()
    return {
        'fringe_real_percent': float(fringe.real),
        'fringe_imag_percent': float(fringe.imag),
        'amplitude_percent': float(abs(fringe)),
        'sigma_stat_percent': float(sigma),
        'sigma_re_percent': float(sig_re),
        'sigma_im_percent': float(sig_im),
        'snr_stat': float(abs(fringe)/sigma),
        'noise_tail_fraction_3sigma': float(np.mean(abs(delta) > 3*sigma)),
        'noise_re_im_correlation': float(np.corrcoef(delta.real, delta.imag)[0, 1]),
        'off_fringe_samples': len(noise),
    }, sample, noise


def coherence_proxy(samples, times_s, sigma_long, block_s=60):
    """Noise-debiased long/short power ratio, a diagnostic rather than a bound.

    Assumes equal-duration independent thermal samples. Amplitude variability
    also reduces this ratio. Estimates above one are retained, not clipped.
    The fitted delay/rate and calibration errors are not propagated here.
    """
    ids = np.floor(np.asarray(times_s)/block_s).astype(int)
    count = len(samples)
    power_short = 0.0
    for block in np.unique(ids):
        z = samples[ids == block]
        power_short += len(z)/count * abs(np.mean(z))**2
    power_short -= 2*sigma_long**2*len(np.unique(ids))
    power_long = abs(np.mean(samples))**2-2*sigma_long**2
    return float(np.sqrt(power_long/power_short)) if power_short > 0 and power_long > 0 else float('nan')


def extract_measurement(binary, source, folder, lo, hi, cpu):
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(source.times_ns[lo]/1e9, timezone.utc)
    subset = folder / f'ANT1_ANT2_{stamp:%Y%j%H%M%S}_measurement.cor'
    source.write_subset(subset, lo, hi)
    command = [str(binary), '--input', str(subset), '--length', '0', '--loop', '1',
               '--search', 'peak', '--iter', '5', '--cpu', str(cpu), '--output']
    with (folder/'frinz.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    files = list((folder/'frinZ').rglob('*delay_rate_search*.tsv'))
    if len(files) != 1:
        raise ValueError(f'Expected one fringe table: {files}')
    lines = [s for s in files[0].read_text().splitlines() if s.strip() and not s.startswith('#')]
    if len(lines) != 1:
        raise ValueError(f'Expected one fringe solution: {files[0]}')
    fields = lines[0].split('\t')
    reported_amp, noise_scale = float(fields[4]), float(fields[7])
    delay, rate = float(fields[8]), float(fields[9])
    times = (source.times_ns[lo:hi] - source.times_ns[lo]).astype(float)/1e9
    stats, samples, noise = fringe_statistics(source.records['visibility'][lo:hi], times,
                                             source.fs, source.frequency, delay, rate)
    relative_error = abs(stats['amplitude_percent']-reported_amp)/max(reported_amp, 1e-30)
    if relative_error > 1e-3:
        raise ValueError(f'Fringe reconstruction differs from frinZ: {relative_error}')
    stats.update({
        'frinz_noise_scale_percent': noise_scale,
        'snr_frinz': reported_amp/noise_scale,
        'res_delay_samples': delay, 'res_rate_hz': rate,
        'amplitude_reconstruction_relative_error': relative_error,
        'coherence_proxy_60s': coherence_proxy(samples, times, stats['sigma_stat_percent']),
    })
    np.savez_compressed(folder/'fringe_diagnostics.npz',
                        elapsed_time_s=times, coherent_sample_percent=samples,
                        off_fringe_complex_percent=noise)
    return stats


def run(args):
    original, corrected = Cor.read(args.original), Cor.read(args.corrected)
    if original.raw[:256] != corrected.raw[:256] or not np.array_equal(original.times_ns, corrected.times_ns):
        raise ValueError('Original and corrected inputs must have matching headers and epochs')
    if not np.array_equal(original.integration_s, corrected.integration_s):
        raise ValueError('Original and corrected exposure times differ')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(args.frinz).resolve()
    epoch = int(original.times_ns[0])
    observation_end = int(original.times_ns[-1] + original.integration_s[-1]*1e9)
    jobs = []
    for width in args.bin_seconds:
        for index, lo, hi, start, end in wall_bins(original.times_ns, original.integration_s, width, epoch):
            for kind, source in [('original', original), ('corrected', corrected)]:
                jobs.append((width, index, lo, hi, start, end, kind, source))

    def one(job):
        width, index, lo, hi, start, end, kind, source = job
        exposure = float(source.integration_s[lo:hi].sum())
        row = {'bin_seconds': width, 'bin_index': index, 'input': kind,
               'bin_start_utc': utc(start), 'bin_end_utc': utc(end),
               'sectors': hi-lo, 'exposure_s': exposure, 'coverage_fraction': exposure/width,
               'partial_last_bin': end > observation_end}
        if hi == lo:
            return row
        weighted_s = float(np.average((source.times_ns[lo:hi]-epoch)/1e9 + source.integration_s[lo:hi]/2,
                                      weights=source.integration_s[lo:hi]))
        row.update({'effective_epoch_utc': utc(epoch+int(weighted_s*1e9)),
                    'effective_elapsed_s': weighted_s,
                    'first_sample_utc': utc(source.times_ns[lo]),
                    'last_sample_end_utc': utc(source.times_ns[hi-1]+source.integration_s[hi-1]*1e9)})
        folder = out/f'bin{width}'/f'{index:03d}'/kind
        row.update(extract_measurement(binary, source, folder, lo, hi, args.cpu))
        print(f'{width}s bin {index:02d} {kind}: exposure={exposure:g}s SNR={row["snr_frinz"]:.2f}', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, jobs))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out/'measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        'bin_anchor_utc': utc(epoch), 'observation_end_utc': utc(observation_end),
        'bin_seconds': args.bin_seconds, 'method': 'disjoint wall-clock bins; independent peak delay/rate search per bin',
        'frinz_version': subprocess.check_output([str(binary), '--version'], text=True).strip(),
        'frinz_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {kind: {'path': str(source.path), 'sha256': hashlib.sha256(source.raw).hexdigest()}
                   for kind, source in [('original', original), ('corrected', corrected)]},
        'frequency_hz': original.frequency, 'sampling_speed_hz': original.fs,
        'fft_point': original.fft, 'cpu_per_job': args.cpu, 'workers': args.workers,
        'amplitude_unit': 'frinZ amplitude percent, not flux-calibrated',
        'uncertainty': 'off-delay real/imag MAD component sigma; high-SNR statistical approximation only',
        'snr_frinz': 'amplitude divided by legacy complex mean-absolute-deviation noise scale',
        'coherence_proxy_60s': 'sqrt(noise-debiased long power / exposure-weighted 60s power); also affected by amplitude variability; estimates above one are retained',
        'limitations': ['No gain or phase-transfer systematic uncertainty included',
                        'Disjoint samples can share calibration errors between bins',
                        'Changing projected baseline and source structure can change correlated amplitude',
                        'No claim of intrinsic source variability or calibrated flux density'],
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    make_plots(rows, out, args.bin_seconds)
    return rows


def make_plots(rows, out, widths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.linewidth': .8})
    fig, axes = plt.subplots(2, len(widths), figsize=(6*len(widths), 6), squeeze=False,
                             sharex='col', constrained_layout=True)
    for col, width in enumerate(widths):
        for kind, color, marker in [('original', '#555555', 's'), ('corrected', '#0072B2', 'o')]:
            selected = [r for r in rows if r['bin_seconds']==width and r['input']==kind and r['sectors']]
            t = [datetime.fromisoformat(r['effective_epoch_utc']) for r in selected]
            x = mdates.date2num(t)
            amp = [r['amplitude_percent']*1e3 for r in selected]
            sigma = [r['sigma_stat_percent']*1e3 for r in selected]
            axes[0, col].errorbar(x, amp, yerr=sigma, fmt=marker, ms=4, capsize=2,
                                  color=color, label=kind)
            axes[1, col].plot(x, [r['snr_frinz'] for r in selected], marker+'-', ms=4,
                              lw=.8, color=color, label=kind)
            for i, row in enumerate(selected):
                if row['partial_last_bin']:
                    axes[0, col].annotate('partial', (x[i], amp[i]), xytext=(0, 10),
                                          textcoords='offset points', ha='center', fontsize=7)
        axes[0, col].set(title=f'{width//60}-min wall-clock bins', ylabel=r'Fringe amplitude [$10^{-3}$ %]')
        axes[1, col].set(xlabel=rows[0]['bin_start_utc'][:10]+' UTC',
                         ylabel='S/N (frinZ convention)')
        axes[0, col].legend(frameon=False)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
    fig.suptitle('Independent single-baseline measurements; error bars: statistical noise only', fontsize=11)
    fig.savefig(out/'independent_measurements.png', dpi=200)
    fig.savefig(out/'independent_measurements.svg')
    fig2, panels = plt.subplots(1, len(widths), figsize=(6*len(widths), 3), squeeze=False,
                               constrained_layout=True)
    for col, width in enumerate(widths):
        ax = panels[0, col]
        for kind, color, marker in [('original', '#555555', 's'), ('corrected', '#0072B2', 'o')]:
            selected = [r for r in rows if r['bin_seconds']==width and r['input']==kind and r['sectors']]
            t = [datetime.fromisoformat(r['effective_epoch_utc']) for r in selected]
            ax.plot(t, [r['coherence_proxy_60s'] for r in selected], marker+'-', color=color,
                    ms=4, lw=.8, label=kind)
        ax.axhline(1, color='black', ls=':', lw=1)
        ax.set(title=f'{width//60}-min bins', ylabel='Long / 60-s coherence proxy', xlabel='UTC')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig2.savefig(out/'coherence_checks.png', dpi=200)
    fig2.savefig(out/'coherence_checks.svg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--corrected', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frinz', default='target/release/frinZ')
    parser.add_argument('--bin-seconds', type=int, nargs='+', default=[1800, 3600])
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if min(args.bin_seconds) <= 0 or args.cpu < 1 or args.workers < 1:
        parser.error('Bin widths, cpu and workers must be positive')
    if len(set(args.bin_seconds)) != len(args.bin_seconds):
        parser.error('Bin widths must be unique')
    run(args)


if __name__ == '__main__':
    main()
