use super::{rate_search_bounds, window_bounds, DeepSearchContext};
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Local group delay of a cross-baseline fringe at the first sector epoch.
#[derive(Debug, Clone, Copy)]
pub struct GroupDelayEstimate {
    pub delay_samples: f64,
    pub delay_seconds: f64,
    /// Independent-visibility, high-S/N standard error conditional on this fringe peak.
    pub sigma_seconds: Option<f64>,
    /// Rate at the header observing frequency, in cycles/s.
    pub rate_hz: f64,
    /// False for one epoch or an explicitly fixed fringe rate.
    pub rate_fitted: bool,
    /// Coherent amplitude divided by estimated noise in its phase quadrature.
    pub snr_quadrature: Option<f64>,
    /// RMS spread of retained frequencies (equal weight per contributing visibility).
    pub bandwidth_rms_hz: f64,
    /// Exact repetition interval in delay at fixed rate on the retained channel grid.
    pub ambiguity_seconds: f64,
    pub visibilities: usize,
}

#[derive(Default)]
struct Moments {
    sum: Complex64,
    first: [Complex64; 2],
    second: [Complex64; 3], // xx, xy, yy
    n: usize,
    sum_x: f64,
    sum_xx: f64,
    min_channel: usize,
    max_channel: usize,
}

struct Profile {
    rotation: Complex64,
    mean: [f64; 2],
    inverse: [f64; 3],
    step: [f64; 2],
}

impl Moments {
    // Profile out an arbitrary constant phase by maximizing |sum V exp(-i phi)|.
    // The real parts give the observed curvature, the imaginary parts the score.
    fn profile(&self, fit_rate: bool) -> Result<Profile, &'static str> {
        let amplitude = self.sum.norm();
        if self.n == 0 || !amplitude.is_finite() || amplitude <= 0.0 {
            return Err("no_coherent_signal");
        }
        if self.min_channel == self.max_channel {
            return Err("one_frequency");
        }
        let rotation = self.sum.conj() / amplitude;
        let first = self.first.map(|v| v * rotation);
        let mean = first.map(|v| v.re / amplitude);
        let [xx, xy, yy] = self.second.map(|v| (v * rotation).re);
        let a = xx - first[0].re * mean[0];
        let b = xy - first[0].re * mean[1];
        let c = yy - first[1].re * mean[1];
        if !a.is_finite() || a <= 0.0 {
            return Err("unresolved_delay");
        }
        let inverse = if fit_rate {
            let det = a * c - b * b;
            if !det.is_finite() || c <= 0.0 || det <= 1e-12 * a * c {
                return Err("unresolved_delay_rate");
            }
            [c / det, -b / det, a / det]
        } else {
            [1.0 / a, 0.0, 0.0]
        };
        let [u, v, w] = inverse;
        let step = [
            u * first[0].im + v * first[1].im,
            v * first[0].im + w * first[1].im,
        ];
        Ok(Profile {
            rotation,
            mean,
            inverse,
            step,
        })
    }
}

impl DeepSearchContext<'_> {
    // x and y are phase derivatives in rad/sample and rad/(rate * span).
    // A constant offset in x is absorbed by the free phase. The physical
    // phase rotation still uses the unshifted channel and window-start epoch.
    fn visit_group_delay_samples(
        &self,
        delay: f64,
        rate: f64,
        span: f64,
        mut visit: impl FnMut(usize, f64, f64, Complex64),
    ) {
        let channels = self.rfi_mask.len();
        let dx = TAU / self.effective_fft_point as f64;
        let rf_increment = self.channel_rf_increment();
        for (i, (&time, row)) in self
            .phase_times
            .iter()
            .zip(self.complex_vec.chunks_exact(channels))
            .enumerate()
        {
            let (mut phase, step) = self.row_rotation(i, delay, rate, rf_increment);
            let time_angle = TAU * (time / span);
            for (k, &sample) in row.iter().enumerate().skip(1) {
                let factor = phase;
                phase *= step;
                if self.rfi_mask[k] || (sample.re == 0.0 && sample.im == 0.0) {
                    continue;
                }
                let mut value = Complex64::new(sample.re.into(), sample.im.into()) * factor;
                if let Some(gain) = self.bandpass_gains.as_ref().and_then(|g| g.get(k)) {
                    value *= *gain;
                }
                if value != Complex64::default() {
                    let frequency_bin = k as f64 + self.args.channel_origin_bins;
                    visit(
                        k,
                        dx * k as f64 - TAU / 4.0,
                        time_angle * (1.0 + frequency_bin * rf_increment),
                        value,
                    );
                }
            }
        }
    }

    fn group_delay_moments(&self, delay: f64, rate: f64, span: f64) -> Moments {
        let mut m = Moments {
            min_channel: usize::MAX,
            ..Moments::default()
        };
        self.visit_group_delay_samples(delay, rate, span, |k, x, y, z| {
            let zx = z * x;
            let zy = z * y;
            m.sum += z;
            m.first[0] += zx;
            m.first[1] += zy;
            m.second[0] += zx * x;
            m.second[1] += zx * y;
            m.second[2] += zy * y;
            m.n += 1;
            m.min_channel = m.min_channel.min(k);
            m.max_channel = m.max_channel.max(k);
            m.sum_x += x;
            m.sum_xx += x * x;
        });
        m
    }

    pub(super) fn fit_group_delay(
        &self,
        mut delay: f64,
        mut rate: f64,
    ) -> Result<GroupDelayEstimate, &'static str> {
        let fs = f64::from(self.header.sampling_speed);
        if fs <= 0.0
            || !self.header.observing_frequency.is_finite()
            || self.header.observing_frequency <= 0.0
        {
            return Err("invalid_frequency");
        }
        let span = self
            .phase_times
            .iter()
            .fold(0.0_f64, |m, &t| m.max(t.abs()));
        let delay_bounds = window_bounds(&self.args.drange);
        let (r_low, r_high) = rate_search_bounds(self.effective_integ_time, &self.args.rrange);
        let fit_rate = span > 0.0 && r_low != r_high;
        if r_low == r_high {
            rate = r_low.into();
        } else if !fit_rate {
            rate = 0.0;
        }
        let span = if span > 0.0 { span } else { 1.0 };
        let interior = |d: f64, r: f64| {
            delay_bounds.is_none_or(|(lo, hi)| f64::from(lo) < d && d < f64::from(hi))
                && (!fit_rate || (f64::from(r_low) < r && r < f64::from(r_high)))
                && !self.candidate_is_masked(d as f32, r as f32)
        };
        let mut moments = self.group_delay_moments(delay, rate, span);
        for _ in 0..16 {
            let profile = moments.profile(fit_rate)?;
            let [dd, dr] = profile.step;
            let tolerance = 2e-12 + 16.0 * f64::EPSILON * (delay.abs() + (rate * span).abs());
            if dd.abs().max(dr.abs()) <= tolerance {
                if !interior(delay, rate) {
                    return Err("at_search_limit");
                }
                let [u, v, _] = profile.inverse;
                let [mx, my] = profile.mean;
                let mut variance = 0.0;
                let mut noise_power = 0.0;
                let mut channel_stride = 0;
                // Independent visibility scores give H^-1 (sum score score') H^-1.
                // Projecting out phase and rate includes their covariance with delay.
                self.visit_group_delay_samples(delay, rate, span, |k, x, y, z| {
                    // The gcd of the surviving channel separations gives their
                    // common frequency spacing, including regularly flagged combs.
                    if channel_stride != 1 {
                        let mut gap = k - moments.min_channel;
                        while gap != 0 {
                            (channel_stride, gap) = (gap, channel_stride % gap);
                        }
                    }
                    let q = (z * profile.rotation).im;
                    variance += (q * (u * (x - mx) + v * (y - my))).powi(2);
                    noise_power += q * q;
                });
                let n = moments.n as f64;
                let p = if fit_rate { 3.0 } else { 2.0 }; // includes the phase intercept
                let dof_factor = (n > p).then(|| n / (n - p));
                return Ok(GroupDelayEstimate {
                    delay_samples: delay,
                    delay_seconds: delay / fs,
                    rate_hz: rate,
                    rate_fitted: fit_rate,
                    sigma_seconds: dof_factor.map(|f| (variance * f).sqrt() / fs),
                    snr_quadrature: dof_factor
                        .filter(|_| noise_power > 0.0)
                        .map(|f| moments.sum.norm() / (noise_power * f).sqrt()),
                    bandwidth_rms_hz: (moments.sum_xx / n - (moments.sum_x / n).powi(2))
                        .max(0.0)
                        .sqrt()
                        * fs
                        / TAU,
                    ambiguity_seconds: self.effective_fft_point as f64
                        / (channel_stride as f64 * fs),
                    visibilities: moments.n,
                });
            }
            // Stay on the selected fringe lobe. A step can approach a search
            // boundary, but the uncertainty is only defined for an interior maximum.
            let mut fraction = 1.0 / (2.0 * dd.abs()).max(4.0 * dr.abs()).max(1.0);
            let mut accepted = None;
            for _ in 0..12 {
                let next_delay = delay + fraction * dd;
                let next_rate = rate + fraction * dr / span;
                if interior(next_delay, next_rate) {
                    let next = self.group_delay_moments(next_delay, next_rate, span);
                    if next.sum.norm() >= moments.sum.norm() * (1.0 - 1e-12) {
                        accepted = Some((next_delay, next_rate, next));
                        break;
                    }
                }
                fraction *= 0.5;
            }
            (delay, rate, moments) = accepted.ok_or("no_interior_peak")?;
        }
        Err("fit_did_not_converge")
    }
}
