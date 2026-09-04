use nalgebra::{DMatrix, DVector};
use std::error::Error; // Import Error trait
use std::f64;
use std::fmt; // Import fmt for custom error display

// Custom error type for fitting
#[derive(Debug)]
struct FittingError(String);

impl fmt::Display for FittingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fitting Error: {}", self.0)
    }
}

impl Error for FittingError {}

// C++版の QuadraticFitResult 構造体に対応
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct QuadraticFitResult {
    pub peak_x: f64, // 2次関数の頂点のx座標
    // pub success: bool,
    pub a: f64, // y = ax^2 + bx + c の係数 a
    pub b: f64, // y = ax^2 + bx + c の係数 b
    pub c: f64, // y = ax^2 + bx + c の係数 c
                // pub message: String,
}

/// Fits a quadratic function y = ax^2 + bx + c to N points using least squares.
/// x_coords and y_values must be of the same size and contain at least 3 points.
/// Returns the x-coordinate of the vertex (-b / 2a). Success only if 'a' is negative.
pub fn fit_quadratic_least_squares(
    x_coords: &[f64],
    y_values: &[f64],
) -> Result<QuadraticFitResult, Box<dyn Error>> {
    // Changed return type

    let n = x_coords.len();
    let epsilon = 1e-9;

    if n < 3 || n != y_values.len() {
        return Err(Box::new(FittingError("Input vectors must be of the same size and contain at least 3 points for least squares.".to_string())));
        // Changed error return
    }

    // Shift x-coordinates by the central value to improve numerical stability.
    let x_center = x_coords[n / 2];
    let shifted_x_coords: Vec<f64> = x_coords.iter().map(|&x| x - x_center).collect();

    let s0 = n as f64;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;
    let mut t0 = 0.0;
    let mut t1 = 0.0;
    let mut t2 = 0.0;

    for i in 0..n {
        let x = shifted_x_coords[i]; // Use shifted coordinates
        let y = y_values[i];
        let x_sq = x * x;
        s1 += x;
        s2 += x_sq;
        s3 += x_sq * x;
        s4 += x_sq * x_sq;
        t0 += y;
        t1 += x * y;
        t2 += x_sq * y;
    }

    // 連立方程式を解くための行列式
    // S0*c + S1*b + S2*a = T0
    // S1*c + S2*b + S3*a = T1
    // S2*c + S3*b + S4*a = T2

    // Denominator D
    let d = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);

    if d.abs() < epsilon {
        return Err(Box::new(FittingError(format!(
            "Denominator D ({}) is almost zero. Matrix is singular or ill-conditioned.",
            d
        )))); // Changed error return
    }

    // Numerators for c, b, a (using Cramer's rule implicitly)
    let dc_num = t0 * (s2 * s4 - s3 * s3) - s1 * (t1 * s4 - t2 * s3) + s2 * (t1 * s3 - t2 * s2);
    let db_num = s0 * (t1 * s4 - t2 * s3) - t0 * (s1 * s4 - s2 * s3) + s2 * (s1 * t2 - s2 * t1);
    let da_num = s0 * (s2 * t2 - s3 * t1) - s1 * (s1 * t2 - s2 * t1) + t0 * (s1 * s3 - s2 * s2);

    let a = da_num / d;
    let b = db_num / d;
    let c = dc_num / d;

    if a.abs() < epsilon {
        return Err(Box::new(FittingError(
            "Coefficient 'a' is almost zero. Quadratic function is degenerate.".to_string(),
        ))); // Changed error return
    }

    if a > 0.0 {
        return Err(Box::new(FittingError("Coefficient 'a' is positive. Quadratic function is convex downwards, no maximum exists.".to_string())));
        // Changed error return
    }

    let peak_x_shifted = -b / (2.0 * a);
    let peak_x = peak_x_shifted + x_center; // Add the offset back

    Ok(QuadraticFitResult { peak_x, a, b, c })
}

#[allow(dead_code)]
pub fn fit_linear_least_squares(
    x_coords: &[f64],
    y_values: &[f64],
) -> Result<(f64, f64), Box<dyn Error>> {
    if x_coords.len() != y_values.len() || x_coords.len() < 2 {
        return Err("Input vectors must be of the same size and contain at least 2 points.".into());
    }

    let n = x_coords.len() as f64;
    let sum_x: f64 = x_coords.iter().sum();
    let sum_y: f64 = y_values.iter().sum();
    let sum_xy: f64 = x_coords
        .iter()
        .zip(y_values.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_x_sq: f64 = x_coords.iter().map(|x| x * x).sum();

    let denominator = n * sum_x_sq - sum_x * sum_x;
    if denominator.abs() < 1e-9 {
        return Err("Denominator is zero, cannot fit a line.".into());
    }

    let m = (n * sum_xy - sum_x * sum_y) / denominator;
    let c = (sum_y * sum_x_sq - sum_x * sum_xy) / denominator;

    Ok((m, c)) // slope, intercept
}

/// Fits a polynomial function to data points using least squares.
/// Returns the coefficients of the polynomial in ascending order (c0, c1, ..., cn).
pub fn fit_polynomial_least_squares(
    x_coords: &[f64],
    y_values: &[f64],
    degree: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let n = x_coords.len();
    if n != y_values.len() {
        return Err(Box::new(FittingError(
            "Input vectors must have the same length.".to_string(),
        )));
    }
    if n <= degree {
        return Err(Box::new(FittingError(format!(
            "Not enough data points ({}) for a polynomial of degree {}. Need at least {} points.",
            n,
            degree,
            degree + 1
        ))));
    }

    // Center and scale the abscissa before building the Vandermonde matrix.
    // Fringe fits commonly use seconds from an absolute epoch; forming normal
    // equations directly from those values squares an already large condition
    // number and noticeably biases small quadratic coefficients.
    let x_center = x_coords.iter().sum::<f64>() / n as f64;
    let x_scale = x_coords
        .iter()
        .map(|x| (x - x_center).abs())
        .fold(0.0_f64, f64::max);
    if !x_center.is_finite() || !x_scale.is_finite() || x_scale == 0.0 {
        return Err(Box::new(FittingError(
            "X coordinates must be finite and span a non-zero range.".to_string(),
        )));
    }

    let mut a_data = Vec::with_capacity(n * (degree + 1));
    for &x in x_coords {
        let normalized_x = (x - x_center) / x_scale;
        for i in 0..=degree {
            a_data.push(normalized_x.powi(i as i32));
        }
    }
    let a = DMatrix::from_row_slice(n, degree + 1, &a_data);
    let y = DVector::from_vec(y_values.to_vec());
    let normalized_coeffs = a
        .svd(true, true)
        .solve(&y, f64::EPSILON * n as f64)
        .map_err(|message| Box::new(FittingError(message.to_string())))?;

    // Convert coefficients in z=(x-x_center)/x_scale back to powers of x.
    let mut coefficients = vec![0.0; degree + 1];
    for power in 0..=degree {
        let scaled = normalized_coeffs[power] / x_scale.powi(power as i32);
        let mut binomial = 1.0;
        for (x_power, coefficient) in coefficients.iter_mut().enumerate().take(power + 1) {
            *coefficient += scaled * binomial * (-x_center).powi((power - x_power) as i32);
            if x_power < power {
                binomial *= (power - x_power) as f64 / (x_power + 1) as f64;
            }
        }
    }

    Ok(coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_quadratic_least_squares_basic() {
        // y = -x^2 + 2x + 3 の頂点は x = 1, y = 4
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_values = vec![3.0, 4.0, 3.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_ok());
        let fit_result = result.unwrap();

        // assert!(fit_result.success);
        assert!((fit_result.peak_x - 1.0).abs() < 1e-9);
        assert!((fit_result.a - (-1.0)).abs() < 1e-9);
        assert!((fit_result.b - 0.0).abs() < 1e-9);
        assert!((fit_result.c - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_fit_quadratic_least_squares_more_points() {
        // y = -2x^2 + 8x + 1 の頂点は x = 2, y = 9
        let x_coords = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_values = vec![1.0, 7.0, 9.0, 7.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_ok());
        let fit_result = result.unwrap();

        // assert!(fit_result.success);
        assert!((fit_result.peak_x - 2.0).abs() < 1e-9);
        assert!((fit_result.a - (-2.0)).abs() < 1e-9);
        assert!((fit_result.b - 0.0).abs() < 1e-9);
        assert!((fit_result.c - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_fit_quadratic_least_squares_positive_a() {
        // y = x^2 の頂点は x = 0, y = 0 (下に凸)
        let x_coords = vec![-1.0, 0.0, 1.0];
        let y_values = vec![1.0, 0.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result {
            // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("Coefficient 'a' is positive"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }

    #[test]
    fn test_fit_quadratic_least_squares_degenerate() {
        // 直線の場合 (a=0)
        let x_coords = vec![0.0, 1.0, 2.0];
        let y_values = vec![0.0, 1.0, 2.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result {
            // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("Coefficient 'a' is almost zero"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }

    #[test]
    fn test_fit_quadratic_least_squares_insufficient_points() {
        let x_coords = vec![0.0, 1.0];
        let y_values = vec![0.0, 1.0];

        let result = fit_quadratic_least_squares(&x_coords, &y_values);
        assert!(result.is_err());
        if let Err(e) = result {
            // Changed error handling
            let err_msg = e.to_string();
            assert!(err_msg.contains("at least 3 points"));
        } else {
            panic!("予期しないエラータイプ: {:?}", result);
        }
    }

    #[test]
    fn test_fit_polynomial_least_squares_linear() {
        // y = 2x + 1
        let x_coords = vec![0.0, 1.0, 2.0, 3.0];
        let y_values = vec![1.0, 3.0, 5.0, 7.0];
        let degree = 1;

        let result = fit_polynomial_least_squares(&x_coords, &y_values, degree);
        assert!(result.is_ok());
        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[0] - 1.0).abs() < 1e-9); // intercept
        assert!((coeffs[1] - 2.0).abs() < 1e-9); // slope
    }

    #[test]
    fn test_fit_polynomial_least_squares_quadratic() {
        // y = x^2 - 2x + 3
        let x_coords = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_values = vec![3.0, 2.0, 3.0, 6.0, 11.0];
        let degree = 2;

        let result = fit_polynomial_least_squares(&x_coords, &y_values, degree);
        assert!(result.is_ok());
        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 3);
        assert!((coeffs[0] - 3.0).abs() < 1e-9); // c
        assert!((coeffs[1] - (-2.0)).abs() < 1e-9); // b
        assert!((coeffs[2] - 1.0).abs() < 1e-9); // a
    }

    #[test]
    fn test_fit_polynomial_least_squares_with_large_time_offset() {
        let x_coords: Vec<f64> = vec![1.0e9, 1.0e9 + 60.0, 1.0e9 + 120.0, 1.0e9 + 180.0];
        let x0: f64 = 1.0e9;
        let y_values: Vec<f64> = x_coords
            .iter()
            .map(|x| 3.0 + 0.25 * (x - x0) + 2.0e-6 * (x - x0).powi(2))
            .collect();

        let coefficients = fit_polynomial_least_squares(&x_coords, &y_values, 2).unwrap();
        assert!((coefficients[2] - 2.0e-6).abs() < 1.0e-12);
    }

    #[test]
    fn test_fit_polynomial_least_squares_insufficient_points() {
        let x_coords = vec![0.0, 1.0];
        let y_values = vec![0.0, 1.0];
        let degree = 2; // Need at least 3 points for degree 2

        let result = fit_polynomial_least_squares(&x_coords, &y_values, degree);
        assert!(result.is_err());
        if let Err(e) = result {
            let err_msg = e.to_string();
            assert!(err_msg.contains("Not enough data points"));
        }
    }
}
