use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitModel {
    #[default]
    Linear,
    Polynomial,
    Sinusoidal,
    Exponential,
    Logarithmic,
}

impl FitModel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Polynomial => "Polynomial",
            Self::Sinusoidal => "Sinusoid",
            Self::Exponential => "Exponential",
            Self::Logarithmic => "Logarithmic",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FitConfig {
    pub model: FitModel,
    pub degree: usize,
    pub source: Option<String>,
    pub x_min: String,
    pub x_max: String,
    pub initial_offset: String,
    pub initial_amplitude: String,
    pub initial_period: String,
    pub initial_phase: String,
    pub initial_rate: String,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            model: FitModel::Linear,
            degree: 2,
            source: None,
            x_min: String::new(),
            x_max: String::new(),
            initial_offset: "0".into(),
            initial_amplitude: "1".into(),
            initial_period: "1".into(),
            initial_phase: "0".into(),
            initial_rate: "0.1".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitResult {
    pub model: FitModel,
    pub source: Option<String>,
    pub parameters: Vec<f64>,
    pub x_center: f64,
    pub x_scale: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub count: usize,
    pub rmse: f64,
    pub r_squared: Option<f64>,
}

impl FitResult {
    pub fn is_well_formed(&self) -> bool {
        let parameter_count_valid = match self.model {
            FitModel::Linear | FitModel::Logarithmic => self.parameters.len() == 2,
            FitModel::Polynomial => (2..=9).contains(&self.parameters.len()),
            FitModel::Sinusoidal => self.parameters.len() == 4,
            FitModel::Exponential => self.parameters.len() == 3,
        };
        parameter_count_valid
            && self.parameters.iter().all(|value| value.is_finite())
            && self.x_center.is_finite()
            && self.x_scale.is_finite()
            && self.x_scale > 0.0
            && self.x_min.is_finite()
            && self.x_max.is_finite()
            && self.x_min <= self.x_max
            && self.count > 0
            && self.rmse.is_finite()
            && self.r_squared.is_none_or(f64::is_finite)
    }

    pub fn value(&self, x: f64) -> Option<f64> {
        if !self.is_well_formed() {
            return None;
        }
        let u = (x - self.x_center) / self.x_scale;
        let value = match self.model {
            FitModel::Linear | FitModel::Polynomial => self
                .parameters
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * u + coefficient),
            FitModel::Logarithmic if x > 0.0 => self.parameters[0] + self.parameters[1] * x.ln(),
            FitModel::Logarithmic => return None,
            FitModel::Sinusoidal => {
                self.parameters[0]
                    + self.parameters[1]
                        * (std::f64::consts::TAU * u / self.parameters[2] + self.parameters[3])
                            .sin()
            }
            FitModel::Exponential => {
                self.parameters[0]
                    + self.parameters[1] * (self.parameters[2] * u).clamp(-700.0, 700.0).exp()
            }
        };
        value.is_finite().then_some(value)
    }

    pub fn named_parameters(&self) -> Vec<(&'static str, f64)> {
        match self.model {
            FitModel::Linear => vec![
                ("intercept at x center", self.parameters[0]),
                ("slope per x", self.parameters[1] / self.x_scale),
            ],
            FitModel::Polynomial => self
                .parameters
                .iter()
                .enumerate()
                .map(|(index, &value)| match index {
                    0 => ("c0", value),
                    1 => ("c1", value),
                    2 => ("c2", value),
                    3 => ("c3", value),
                    4 => ("c4", value),
                    5 => ("c5", value),
                    6 => ("c6", value),
                    7 => ("c7", value),
                    _ => ("c8", value),
                })
                .collect(),
            FitModel::Sinusoidal => vec![
                ("offset", self.parameters[0]),
                ("amplitude", self.parameters[1]),
                ("period", self.parameters[2] * self.x_scale),
                ("phase [rad]", self.parameters[3]),
            ],
            FitModel::Exponential => vec![
                ("offset", self.parameters[0]),
                ("amplitude", self.parameters[1]),
                ("rate per x", self.parameters[2] / self.x_scale),
            ],
            FitModel::Logarithmic => vec![
                ("a", self.parameters[0]),
                ("b (ln coefficient)", self.parameters[1]),
            ],
        }
    }

    pub fn equation_basis(&self) -> String {
        match self.model {
            FitModel::Linear
            | FitModel::Polynomial
            | FitModel::Sinusoidal
            | FitModel::Exponential => {
                format!(
                    "u = (x − {}) / {}",
                    short(self.x_center),
                    short(self.x_scale)
                )
            }
            FitModel::Logarithmic => "x > 0; model uses ln(x)".into(),
        }
    }

    pub fn legend_label(&self) -> String {
        match &self.source {
            Some(source) => format!("{} fit [{}]", self.model.label(), source),
            None => format!("{} fit", self.model.label()),
        }
    }
}

pub fn least_squares(points: &[(f64, f64)], config: &FitConfig) -> Result<FitResult, String> {
    let points: Vec<_> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    if points.is_empty() {
        return Err("No finite data points in the selected X range.".into());
    }
    let x_min = points
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let x_max = points
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_center = x_min * 0.5 + x_max * 0.5;
    let x_scale = ((x_max - x_min) * 0.5).abs().max(1e-12);
    let mut distinct_x: Vec<_> = points.iter().map(|point| point.0).collect();
    distinct_x.sort_by(f64::total_cmp);
    distinct_x.dedup_by(|left, right| *left == *right);
    let required_distinct = match config.model {
        FitModel::Linear | FitModel::Logarithmic => 2,
        FitModel::Polynomial => config.degree.clamp(1, 8) + 1,
        FitModel::Sinusoidal => 4,
        FitModel::Exponential => 3,
    };
    if distinct_x.len() < required_distinct {
        return Err(format!(
            "This model requires at least {required_distinct} distinct X values."
        ));
    }
    let normalized: Vec<_> = points
        .iter()
        .map(|&(x, y)| ((x - x_center) / x_scale, x, y))
        .collect();

    let parameters = match config.model {
        FitModel::Linear => linear_fit(&normalized, 1)?,
        FitModel::Polynomial => {
            let degree = config.degree.clamp(1, 8);
            linear_fit(&normalized, degree)?
        }
        FitModel::Logarithmic => {
            if points.iter().any(|(x, _)| *x <= 0.0) {
                return Err("Logarithmic fitting requires positive X values.".into());
            }
            let design: Vec<Vec<f64>> = points.iter().map(|(x, _)| vec![1.0, x.ln()]).collect();
            solve_linear(
                &design,
                &points.iter().map(|point| point.1).collect::<Vec<_>>(),
            )?
        }
        FitModel::Sinusoidal => {
            let initial = [
                parse_initial(&config.initial_offset, "offset")?,
                parse_initial(&config.initial_amplitude, "amplitude")?,
                parse_initial(&config.initial_period, "period")? / x_scale,
                parse_initial(&config.initial_phase, "phase")?,
            ];
            if initial[2] <= 0.0 {
                return Err("Initial period must be positive.".into());
            }
            nonlinear_fit(&normalized, FitModel::Sinusoidal, &initial)?
        }
        FitModel::Exponential => {
            let initial = [
                parse_initial(&config.initial_offset, "offset")?,
                parse_initial(&config.initial_amplitude, "amplitude")?,
                parse_initial(&config.initial_rate, "rate")? * x_scale,
            ];
            nonlinear_fit(&normalized, FitModel::Exponential, &initial)?
        }
    };

    let mut sse = 0.0;
    let mut mean = 0.0;
    for &(_, _, y) in &normalized {
        mean += y / normalized.len() as f64;
    }
    let mut total = 0.0;
    for &(u, x, y) in &normalized {
        let predicted = predict(config.model, &parameters, u, x)
            .ok_or_else(|| "The fitted function produced a non-finite value.".to_owned())?;
        sse += (y - predicted).powi(2);
        total += (y - mean).powi(2);
    }
    if !sse.is_finite() {
        return Err("The fit diverged; review the initial values and range.".into());
    }
    Ok(FitResult {
        model: config.model,
        source: config.source.clone(),
        parameters,
        x_center,
        x_scale,
        x_min,
        x_max,
        count: points.len(),
        rmse: (sse / points.len() as f64).sqrt(),
        r_squared: (total > 0.0).then_some(1.0 - sse / total),
    })
}

fn linear_fit(points: &[(f64, f64, f64)], degree: usize) -> Result<Vec<f64>, String> {
    let design: Vec<Vec<f64>> = points
        .iter()
        .map(|&(u, _, _)| (0..=degree).map(|power| u.powi(power as i32)).collect())
        .collect();
    solve_linear(
        &design,
        &points.iter().map(|point| point.2).collect::<Vec<_>>(),
    )
}

fn solve_linear(design: &[Vec<f64>], values: &[f64]) -> Result<Vec<f64>, String> {
    let columns = design.first().map_or(0, Vec::len);
    if design.len() < columns {
        return Err(format!(
            "At least {columns} points are required for this model."
        ));
    }
    let matrix = DMatrix::from_fn(design.len(), columns, |row, column| design[row][column]);
    let values = DVector::from_column_slice(values);
    matrix
        .svd(true, true)
        .solve(&values, 1e-12)
        .map(|solution| solution.iter().copied().collect())
        .map_err(|_| "The fit is singular; check the X range and data.".into())
}

fn nonlinear_fit(
    points: &[(f64, f64, f64)],
    model: FitModel,
    initial: &[f64],
) -> Result<Vec<f64>, String> {
    if points.len() < initial.len() {
        return Err(format!(
            "At least {} points are required for this model.",
            initial.len()
        ));
    }
    let mut parameters = initial.to_vec();
    let mut lambda = 1e-3;
    let mut loss = objective_loss(points, model, &parameters)
        .ok_or_else(|| "Initial values do not define a finite function.".to_owned())?;
    for _ in 0..300 {
        let n = points.len();
        let m = parameters.len();
        let jacobian = DMatrix::from_fn(n, m, |row, column| {
            let step = 1e-5 * parameters[column].abs().max(1.0);
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus[column] += step;
            minus[column] -= step;
            match (
                predict(model, &plus, points[row].0, points[row].1),
                predict(model, &minus, points[row].0, points[row].1),
            ) {
                (Some(plus), Some(minus)) => (plus - minus) / (2.0 * step),
                _ => 0.0,
            }
        });
        let residual = DVector::from_iterator(
            n,
            points.iter().map(|point| {
                point.2 - predict(model, &parameters, point.0, point.1).unwrap_or(point.2)
            }),
        );
        let normal = jacobian.transpose() * &jacobian;
        let gradient = jacobian.transpose() * residual;
        let mut damped = normal.clone();
        for index in 0..m {
            damped[(index, index)] += lambda * normal[(index, index)].abs().max(1.0);
        }
        let Some(delta) = damped.lu().solve(&gradient) else {
            lambda *= 10.0;
            continue;
        };
        let candidate: Vec<_> = parameters
            .iter()
            .zip(delta.iter())
            .map(|(value, change)| value + change)
            .collect();
        if let Some(candidate_loss) = objective_loss(points, model, &candidate) {
            if candidate_loss < loss {
                parameters = candidate;
                let improvement = loss - candidate_loss;
                loss = candidate_loss;
                lambda = (lambda * 0.3).max(1e-12);
                if delta.norm() < 1e-8 * (DVector::from_column_slice(&parameters).norm() + 1.0)
                    || improvement < 1e-12 * (loss + 1.0)
                {
                    break;
                }
            } else {
                lambda = (lambda * 10.0).min(1e12);
            }
        } else {
            lambda = (lambda * 10.0).min(1e12);
        }
    }
    Ok(parameters)
}

fn objective_loss(points: &[(f64, f64, f64)], model: FitModel, parameters: &[f64]) -> Option<f64> {
    let mut total = 0.0;
    for &(u, x, y) in points {
        let residual = y - predict(model, parameters, u, x)?;
        total += residual * residual;
    }
    total.is_finite().then_some(total)
}

fn predict(model: FitModel, parameters: &[f64], u: f64, x: f64) -> Option<f64> {
    let value = match model {
        FitModel::Linear | FitModel::Polynomial => parameters
            .iter()
            .rev()
            .fold(0.0, |value, coefficient| value * u + coefficient),
        FitModel::Logarithmic if x > 0.0 => parameters[0] + parameters[1] * x.ln(),
        FitModel::Logarithmic => return None,
        FitModel::Sinusoidal if parameters[2] > 1e-12 => {
            parameters[0]
                + parameters[1] * (std::f64::consts::TAU * u / parameters[2] + parameters[3]).sin()
        }
        FitModel::Sinusoidal => return None,
        FitModel::Exponential => {
            parameters[0] + parameters[1] * (parameters[2] * u).clamp(-700.0, 700.0).exp()
        }
    };
    value.is_finite().then_some(value)
}

fn parse_initial(text: &str, name: &str) -> Result<f64, String> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("Initial {name} must be a finite number."))
}

fn short(value: f64) -> String {
    format!("{value:.6}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_and_polynomial_least_squares_recover_coefficients() {
        let points: Vec<_> = (-4..=4)
            .map(|x| {
                let x = x as f64;
                (x, 3.0 + 2.0 * x - 0.5 * x * x)
            })
            .collect();
        let linear = least_squares(&points, &FitConfig::default()).unwrap();
        assert!(linear.rmse > 0.0);

        let polynomial = least_squares(
            &points,
            &FitConfig {
                model: FitModel::Polynomial,
                degree: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(polynomial.rmse < 1e-10);
        for &(x, y) in &points {
            assert!((polynomial.value(x).unwrap() - y).abs() < 1e-9);
        }
    }

    #[test]
    fn sinusoid_and_exponential_fits_converge_from_initial_values() {
        let sine: Vec<_> = (0..60)
            .map(|index| {
                let x = index as f64 * 0.1;
                (x, 2.0 + 3.0 * (std::f64::consts::TAU * x / 2.0 + 0.4).sin())
            })
            .collect();
        let sine_fit = least_squares(
            &sine,
            &FitConfig {
                model: FitModel::Sinusoidal,
                initial_offset: "1.8".into(),
                initial_amplitude: "2.5".into(),
                initial_period: "1.8".into(),
                initial_phase: "0.2".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(sine_fit.rmse < 1e-5, "{}", sine_fit.rmse);

        let exponential: Vec<_> = (0..30)
            .map(|index| {
                let x = index as f64 * 0.1;
                (x, 1.5 + 2.0 * (0.7 * x).exp())
            })
            .collect();
        let exponential_fit = least_squares(
            &exponential,
            &FitConfig {
                model: FitModel::Exponential,
                initial_offset: "1".into(),
                initial_amplitude: "1.8".into(),
                initial_rate: "0.5".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(exponential_fit.rmse < 1e-5, "{}", exponential_fit.rmse);
    }

    #[test]
    fn logarithmic_fit_requires_positive_x_and_recovers_parameters() {
        let points: Vec<_> = (1..=12)
            .map(|x| (x as f64, 4.0 - 1.25 * (x as f64).ln()))
            .collect();
        let fit = least_squares(
            &points,
            &FitConfig {
                model: FitModel::Logarithmic,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fit.rmse < 1e-10);
        assert!(
            least_squares(
                &[(0.0, 1.0), (1.0, 2.0)],
                &FitConfig {
                    model: FitModel::Logarithmic,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn linear_fit_rejects_a_range_with_only_one_distinct_x() {
        let error = least_squares(&[(2.0, 1.0), (2.0, 3.0)], &FitConfig::default()).unwrap_err();
        assert!(error.contains("distinct X values"));
    }
}
