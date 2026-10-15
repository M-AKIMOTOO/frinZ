use serde::Serialize;

/// Ordinary (not circular) statistics of finite numeric cells. Sample variance
/// uses n-1; a singleton has no defined standard deviation or standard error.
#[derive(Clone, Debug, Serialize)]
pub struct Statistics {
    pub count: usize,
    pub missing: usize,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub sample_stddev: Option<f64>,
    pub standard_error: Option<f64>,
    pub rms: f64,
}

impl Statistics {
    pub fn calculate(values: impl IntoIterator<Item = Option<f64>>) -> Option<Self> {
        let mut data = Vec::new();
        let mut missing = 0;
        let (mut mean, mut m2, mut sum) = (0.0, 0.0, 0.0);
        for value in values {
            if let Some(value) = value.filter(|v| v.is_finite()) {
                data.push(value);
                let delta = value - mean;
                mean += delta / data.len() as f64;
                m2 += delta * (value - mean);
                sum += value;
            } else {
                missing += 1;
            }
        }
        if data.is_empty() {
            return None;
        }
        data.sort_by(f64::total_cmp);
        let n = data.len();
        let median = if n % 2 == 0 {
            data[n / 2 - 1] / 2.0 + data[n / 2] / 2.0
        } else {
            data[n / 2]
        };
        let sample_stddev = (n > 1).then(|| (m2.max(0.0) / (n - 1) as f64).sqrt());
        // Scale before squaring to avoid overflow for otherwise finite data.
        let scale = data.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let rms = if scale == 0.0 {
            0.0
        } else {
            scale * (data.iter().map(|v| (v / scale).powi(2)).sum::<f64>() / n as f64).sqrt()
        };
        Some(Self {
            count: n,
            missing,
            sum,
            mean,
            median,
            min: data[0],
            max: data[n - 1],
            sample_stddev,
            standard_error: sample_stddev.map(|std| std / (n as f64).sqrt()),
            rms,
        })
    }
}

/// Equal-width bins including the maximum in the final bin.
pub fn histogram(values: &[f64], bins: usize) -> Vec<(f64, f64, usize)> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || bins == 0 {
        return Vec::new();
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        return vec![(min, (min.abs() * 0.05).max(1e-6), finite.len())];
    }
    let width = (max - min) / bins as f64;
    let mut counts = vec![0; bins];
    for value in finite {
        let index = (((value - min) / width) as usize).min(bins - 1);
        counts[index] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, n)| (min + (i as f64 + 0.5) * width, width, n))
        .collect()
}

pub fn display(value: f64) -> String {
    if !value.is_finite() {
        return "—".into();
    }
    if value != 0.0 && (value.abs() < 0.001 || value.abs() >= 1e7) {
        format!("{value:.6e}")
    } else {
        format!("{value:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_statistics_and_missing_cells() {
        let s =
            Statistics::calculate([Some(1.0), Some(2.0), Some(3.0), None, Some(f64::NAN)]).unwrap();
        assert_eq!((s.count, s.missing), (3, 2));
        assert_eq!(
            (s.mean, s.median, s.sum, s.min, s.max),
            (2.0, 2.0, 6.0, 1.0, 3.0)
        );
        assert_eq!(s.sample_stddev, Some(1.0));
        assert!((s.standard_error.unwrap() - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
        assert!(Statistics::calculate([None]).is_none());
        assert!(
            Statistics::calculate([Some(3.0)])
                .unwrap()
                .sample_stddev
                .is_none()
        );
    }
    #[test]
    fn histogram_keeps_endpoints_and_constants() {
        assert_eq!(
            histogram(&[0.0, 1.0, 2.0], 2)
                .iter()
                .map(|v| v.2)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(histogram(&[4.0; 3], 10)[0].2, 3);
    }
}
