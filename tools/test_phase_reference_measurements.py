"""Tests for time-bin integrity and statistical error scaling."""
import unittest
import numpy as np
from phase_reference_measurements import wall_bins, fringe_statistics, coherence_proxy


class MeasurementsTests(unittest.TestCase):
    def test_coherence_diagnostic_detects_phase_cancellation(self):
        t = np.arange(16*60, dtype=float)
        phase = np.repeat([0.0, np.pi/2]*8, 60)
        z = np.exp(1j*phase)
        self.assertAlmostEqual(coherence_proxy(z, t, 0), np.sqrt(.5))
        rng = np.random.default_rng(349)
        noise = .5*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        self.assertLess(abs(coherence_proxy(np.ones(len(t))+noise, t, .5/np.sqrt(len(t)))-1), .02)

    def test_disjoint_wall_bins_preserve_gaps_and_boundaries(self):
        t = np.array([10, 11, 50, 70], dtype=np.int64)*10**9
        bins = wall_bins(t, np.ones(4), 60, 10*10**9)
        self.assertEqual([r[1:3] for r in bins], [(0, 3), (3, 4)])
        self.assertEqual(sum(hi-lo for _, lo, hi, _, _ in bins), len(t))
        bins = wall_bins(np.array([0, 180], dtype=np.int64)*10**9, np.ones(2), 60, 0)
        self.assertEqual([hi-lo for _, lo, hi, _, _ in bins], [1, 0, 0, 1])

    def test_rejects_records_that_straddle_boundaries(self):
        with self.assertRaises(ValueError):
            wall_bins(np.array([59], dtype=np.int64)*10**9, np.array([2.0]), 60, 0)

    def test_off_fringe_sigma_matches_gaussian_monte_carlo(self):
        rng = np.random.default_rng(3917)
        n, fft, fs, nu0 = 64, 256, 64e6, 6.6e9
        channels = fft//2
        times = np.arange(n, dtype=float)
        times[32:] += 20  # real timestamps include a gap
        f = np.arange(channels)*fs/fft
        delay, rate, phi, amplitude, sigma = 1.3, .012, .3, .002, .02
        model = 2*np.pi*(f[None, :]*delay/fs + rate*times[:, None]*(1+f/nu0))
        signal = amplitude*np.exp(1j*(phi+model))
        scale = 100*512/(fs/2/1e6)
        expected_sigma = sigma*np.sqrt((channels-1)/n)*scale
        expected_real = amplitude*(channels-1)*np.cos(phi)*scale
        real, estimates = [], []
        for _ in range(64):
            noise = sigma*(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))
            stats, _, _ = fringe_statistics(signal+noise, times, fs, nu0, delay, rate)
            real.append(stats['fringe_real_percent'])
            estimates.append(stats['sigma_stat_percent'])
        self.assertLess(abs(np.mean(estimates)/expected_sigma-1), .08)
        self.assertLess(abs(np.std(real, ddof=1)/expected_sigma-1), .25)
        self.assertLess(abs(np.mean(real)-expected_real), 3*expected_sigma/8)


if __name__ == '__main__':
    unittest.main()
