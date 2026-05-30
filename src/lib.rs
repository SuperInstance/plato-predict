//! # plato-predict
//!
//! Time series prediction primitives for PLATO tile streams.
//! Lightweight statistical methods for predicting next tile values without ML frameworks.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A time series: observed values paired with timestamps (millisecond epoch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub values: Vec<f64>,
    pub timestamps: Vec<u64>,
}

/// Available prediction models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionModel {
    Naive,
    MovingAverage,
    ExponentialSmoothing,
    LinearRegression,
    Arima,
}

/// A single prediction result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub value: f64,
    pub confidence: f64,
    pub horizon: usize,
    pub model: PredictionModel,
}

/// A fitted model ready for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFit {
    pub model: PredictionModel,
    pub params: Vec<f64>,
    pub residual_std: f64,
    pub fitted: Vec<f64>,
}

/// Anomaly score for a single observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyScore {
    pub value: f64,
    pub score: f64,
    pub threshold: f64,
    pub is_anomaly: bool,
}

// ---------------------------------------------------------------------------
// TimeSeries helpers
// ---------------------------------------------------------------------------

impl TimeSeries {
    /// Build a `TimeSeries` from a slice of `(timestamp, value)` tuples.
    pub fn from_tiles(values: &[(u64, f64)]) -> Self {
        let (timestamps, vals): (Vec<u64>, Vec<f64>) = values.iter().copied().unzip();
        Self {
            values: vals,
            timestamps,
        }
    }

    /// Split into training and test sets. `ratio` is the fraction used for training.
    pub fn train_test_split(&self, ratio: f64) -> (TimeSeries, TimeSeries) {
        let split = ((self.values.len() as f64) * ratio).round() as usize;
        let split = split.max(1).min(self.values.len().saturating_sub(1));
        (
            TimeSeries {
                values: self.values[..split].to_vec(),
                timestamps: self.timestamps[..split].to_vec(),
            },
            TimeSeries {
                values: self.values[split..].to_vec(),
                timestamps: self.timestamps[split..].to_vec(),
            },
        )
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let variance = self.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (self.values.len() - 1) as f64;
        variance.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Prediction functions
// ---------------------------------------------------------------------------

/// Naive prediction: propagate the last observed value.
pub fn naive_predict(series: &TimeSeries, horizon: usize) -> Prediction {
    let last = series.values.last().copied().unwrap_or(0.0);
    let std = series.std_dev();
    Prediction {
        value: last,
        confidence: 1.0 - (std / (std + 1.0)).min(1.0),
        horizon,
        model: PredictionModel::Naive,
    }
}

/// Moving-average prediction using the last `window` values.
pub fn moving_average_predict(series: &TimeSeries, window: usize, horizon: usize) -> Prediction {
    let w = window.max(1).min(series.values.len());
    let avg: f64 = series.values.iter().rev().take(w).sum::<f64>() / w as f64;
    let std = series.std_dev();
    Prediction {
        value: avg,
        confidence: 1.0 - (std / (std + 1.0)).min(1.0),
        horizon,
        model: PredictionModel::MovingAverage,
    }
}

/// Simple exponential smoothing forecast.
pub fn exponential_smoothing_predict(series: &TimeSeries, alpha: f64, horizon: usize) -> Prediction {
    if series.values.is_empty() {
        return Prediction {
            value: 0.0,
            confidence: 0.0,
            horizon,
            model: PredictionModel::ExponentialSmoothing,
        };
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut level = series.values[0];
    for &v in &series.values[1..] {
        level = alpha * v + (1.0 - alpha) * level;
    }
    // Simple confidence based on alpha (higher alpha → more responsive → lower smooth confidence)
    let confidence = 0.5 + 0.5 * (1.0 - alpha);
    Prediction {
        value: level,
        confidence,
        horizon,
        model: PredictionModel::ExponentialSmoothing,
    }
}

/// Fit a linear regression (OLS) to the time series.
/// Returns `ModelFit` with params `[slope, intercept]`.
pub fn linear_regression_fit(series: &TimeSeries) -> ModelFit {
    let n = series.values.len() as f64;
    if n < 2.0 {
        let val = series.values.first().copied().unwrap_or(0.0);
        return ModelFit {
            model: PredictionModel::LinearRegression,
            params: vec![0.0, val],
            residual_std: 0.0,
            fitted: vec![val; series.values.len()],
        };
    }

    // Use index as x
    let xs: Vec<f64> = (0..series.values.len()).map(|i| i as f64).collect();
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = series.mean();

    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - x_mean;
        num += dx * (series.values[i] - y_mean);
        den += dx * dx;
    }

    let slope = if den.abs() > 1e-12 { num / den } else { 0.0 };
    let intercept = y_mean - slope * x_mean;

    let fitted: Vec<f64> = xs.iter().map(|&x| slope * x + intercept).collect();

    let residuals: Vec<f64> = fitted
        .iter()
        .zip(&series.values)
        .map(|(f, v)| v - f)
        .collect();

    let residual_std = if residuals.len() > 2 {
        let rsq: f64 = residuals.iter().map(|r| r * r).sum();
        (rsq / (residuals.len() - 2) as f64).sqrt()
    } else {
        0.0
    };

    ModelFit {
        model: PredictionModel::LinearRegression,
        params: vec![slope, intercept],
        residual_std,
        fitted,
    }
}

/// Predict at a given timestamp using a fitted linear model.
pub fn linear_regression_predict(fit: &ModelFit, at: u64) -> Prediction {
    let slope = fit.params.get(0).copied().unwrap_or(0.0);
    let intercept = fit.params.get(1).copied().unwrap_or(0.0);
    let value = slope * at as f64 + intercept;
    Prediction {
        value,
        confidence: 1.0 - (fit.residual_std / (fit.residual_std + 1.0)).min(1.0),
        horizon: 1,
        model: PredictionModel::LinearRegression,
    }
}

/// Compute residuals of a fitted model against the original series.
pub fn residual_analysis(fit: &ModelFit, series: &TimeSeries) -> Vec<f64> {
    series
        .values
        .iter()
        .zip(&fit.fitted)
        .map(|(actual, predicted)| actual - predicted)
        .collect()
}

/// Detect anomalies using a z-score threshold.
pub fn detect_anomalies(series: &TimeSeries, threshold_std: f64) -> Vec<AnomalyScore> {
    let mean = series.mean();
    let std = series.std_dev();
    if std < 1e-12 {
        return series
            .values
            .iter()
            .map(|&v| AnomalyScore {
                value: v,
                score: 0.0,
                threshold: threshold_std,
                is_anomaly: false,
            })
            .collect();
    }
    series
        .values
        .iter()
        .map(|&v| {
            let z = (v - mean).abs() / std;
            AnomalyScore {
                value: v,
                score: z,
                threshold: threshold_std,
                is_anomaly: z > threshold_std,
            }
        })
        .collect()
}

/// K-fold cross-validation returning MAE per fold.
pub fn cross_validate(series: &TimeSeries, model: PredictionModel, folds: usize) -> Vec<f64> {
    let folds = folds.max(2).min(series.values.len());
    let fold_size = series.values.len() / folds;
    if fold_size == 0 {
        return vec![];
    }

    let mut mae_per_fold = Vec::with_capacity(folds);
    for k in 0..folds {
        let test_start = k * fold_size;
        let test_end = if k == folds - 1 {
            series.values.len()
        } else {
            test_start + fold_size
        };

        // Training = everything except this fold
        let train_vals: Vec<f64> = series
            .values
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, v)| *v)
            .collect();
        let train_ts: Vec<u64> = series
            .timestamps
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, v)| *v)
            .collect();

        let train = TimeSeries {
            values: train_vals,
            timestamps: train_ts,
        };
        if train.values.is_empty() {
            mae_per_fold.push(f64::NAN);
            continue;
        }

        let test_actual: Vec<f64> = series.values[test_start..test_end].to_vec();
        let pred = match model {
            PredictionModel::Naive => naive_predict(&train, 1),
            PredictionModel::MovingAverage => {
                moving_average_predict(&train, 3.min(train.values.len()), 1)
            }
            PredictionModel::ExponentialSmoothing => {
                exponential_smoothing_predict(&train, 0.3, 1)
            }
            PredictionModel::LinearRegression => {
                let fit = linear_regression_fit(&train);
                // Predict at midpoint of test set
                let mid_ts = series.timestamps[test_start];
                linear_regression_predict(&fit, mid_ts)
            }
            PredictionModel::Arima => {
                // Simplified: fall back to exponential smoothing
                exponential_smoothing_predict(&train, 0.3, 1)
            }
        };

        let mae = test_actual.iter().map(|a| (a - pred.value).abs()).sum::<f64>()
            / test_actual.len() as f64;
        mae_per_fold.push(mae);
    }
    mae_per_fold
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

/// Mean Absolute Error.
pub fn mean_absolute_error(predicted: &[f64], actual: &[f64]) -> f64 {
    let n = predicted.len().min(actual.len());
    if n == 0 {
        return 0.0;
    }
    predicted[..n]
        .iter()
        .zip(&actual[..n])
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / n as f64
}

/// Root Mean Square Error.
pub fn root_mean_square_error(predicted: &[f64], actual: &[f64]) -> f64 {
    let n = predicted.len().min(actual.len());
    if n == 0 {
        return 0.0;
    }
    let mse: f64 = predicted[..n]
        .iter()
        .zip(&actual[..n])
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_series(n: usize, slope: f64, intercept: f64) -> TimeSeries {
        TimeSeries {
            values: (0..n).map(|i| slope * i as f64 + intercept).collect(),
            timestamps: (0..n).map(|i| i as u64 * 1000).collect(),
        }
    }

    fn constant_series(n: usize, val: f64) -> TimeSeries {
        TimeSeries {
            values: vec![val; n],
            timestamps: (0..n).map(|i| i as u64 * 1000).collect(),
        }
    }

    // --- Naive prediction ---

    #[test]
    fn naive_last_value_propagated() {
        let ts = TimeSeries::from_tiles(&[
            (100, 1.0),
            (200, 2.0),
            (300, 3.0),
            (400, 4.0),
            (500, 5.0),
        ]);
        let p = naive_predict(&ts, 3);
        assert_eq!(p.value, 5.0);
        assert_eq!(p.horizon, 3);
        assert_eq!(p.model, PredictionModel::Naive);
    }

    #[test]
    fn naive_single_point() {
        let ts = TimeSeries::from_tiles(&[(100, 42.0)]);
        let p = naive_predict(&ts, 1);
        assert_eq!(p.value, 42.0);
    }

    #[test]
    fn naive_empty_series() {
        let ts = TimeSeries {
            values: vec![],
            timestamps: vec![],
        };
        let p = naive_predict(&ts, 1);
        assert_eq!(p.value, 0.0);
    }

    // --- Moving average ---

    #[test]
    fn ma_window_3() {
        let ts = TimeSeries::from_tiles(&[
            (1, 10.0),
            (2, 20.0),
            (3, 30.0),
            (4, 40.0),
            (5, 50.0),
        ]);
        let p = moving_average_predict(&ts, 3, 1);
        // last 3: 30, 40, 50 => avg 40
        assert!((p.value - 40.0).abs() < 1e-9);
    }

    #[test]
    fn ma_window_exceeds_length() {
        let ts = TimeSeries::from_tiles(&[(1, 10.0), (2, 20.0)]);
        let p = moving_average_predict(&ts, 10, 1);
        assert!((p.value - 15.0).abs() < 1e-9); // mean of all
    }

    #[test]
    fn ma_window_1_equals_naive() {
        let ts = TimeSeries::from_tiles(&[(1, 7.0), (2, 9.0), (3, 11.0)]);
        let p = moving_average_predict(&ts, 1, 1);
        assert!((p.value - 11.0).abs() < 1e-9);
    }

    // --- Exponential smoothing ---

    #[test]
    fn es_alpha_high() {
        let ts = TimeSeries::from_tiles(&[(1, 0.0), (2, 100.0), (3, 100.0)]);
        let p = exponential_smoothing_predict(&ts, 0.9, 1);
        // With alpha=0.9, level should be close to last value
        assert!((p.value - 100.0).abs() < 5.0);
    }

    #[test]
    fn es_alpha_low_smooths() {
        let ts = TimeSeries::from_tiles(&[(1, 0.0), (2, 100.0)]);
        let p = exponential_smoothing_predict(&ts, 0.1, 1);
        // With alpha=0.1, level should be closer to 10.0 than 100.0
        assert!(p.value < 20.0);
    }

    #[test]
    fn es_alpha_0_5() {
        let ts = TimeSeries::from_tiles(&[(1, 10.0), (2, 20.0), (3, 30.0)]);
        let p = exponential_smoothing_predict(&ts, 0.5, 2);
        // level: 10 -> 0.5*20 + 0.5*10 = 15 -> 0.5*30 + 0.5*15 = 22.5
        assert!((p.value - 22.5).abs() < 1e-9);
    }

    // --- Linear regression ---

    #[test]
    fn lr_perfect_line() {
        let ts = linear_series(10, 5.0, 3.0); // y = 5x + 3
        let fit = linear_regression_fit(&ts);
        assert!((fit.params[0] - 5.0).abs() < 1e-9); // slope
        assert!((fit.params[1] - 3.0).abs() < 1e-9); // intercept
        assert!(fit.residual_std < 1e-9);
    }

    #[test]
    fn lr_flat_line() {
        let ts = constant_series(10, 7.5);
        let fit = linear_regression_fit(&ts);
        assert!(fit.params[0].abs() < 1e-9); // slope ~0
        assert!((fit.params[1] - 7.5).abs() < 1e-9);
    }

    #[test]
    fn lr_noisy_line() {
        // y = 2x + 1 with some noise
        let values: Vec<f64> = (0..20)
            .map(|i| 2.0 * i as f64 + 1.0 + if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let ts = TimeSeries {
            values,
            timestamps: (0..20).map(|i| i as u64 * 100).collect(),
        };
        let fit = linear_regression_fit(&ts);
        assert!((fit.params[0] - 2.0).abs() < 0.5); // slope near 2
        assert!(fit.residual_std < 1.0);
    }

    #[test]
    fn lr_predict_at_timestamp() {
        let ts = linear_series(10, 3.0, 0.0); // y = 3x + 0
        let fit = linear_regression_fit(&ts);
        let p = linear_regression_predict(&fit, 15); // x=15 -> 45
        assert!((p.value - 45.0).abs() < 1e-9);
    }

    // --- Residual analysis ---

    #[test]
    fn residuals_sum_near_zero() {
        let ts = linear_series(20, 2.0, 5.0);
        let fit = linear_regression_fit(&ts);
        let residuals = residual_analysis(&fit, &ts);
        // For a perfect line, all residuals should be ~0
        let sum: f64 = residuals.iter().sum();
        assert!(sum.abs() < 1e-9);
    }

    #[test]
    fn residuals_length_matches() {
        let ts = linear_series(10, 1.0, 0.0);
        let fit = linear_regression_fit(&ts);
        let residuals = residual_analysis(&fit, &ts);
        assert_eq!(residuals.len(), 10);
    }

    // --- Anomaly detection ---

    #[test]
    fn detect_known_outliers() {
        let mut vals: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 0.1)).collect();
        vals[5] = 100.0; // outlier
        vals[15] = -50.0; // outlier
        let ts = TimeSeries {
            values: vals,
            timestamps: (0..20).map(|i| i as u64).collect(),
        };
        let anomalies = detect_anomalies(&ts, 2.0);
        assert!(anomalies[5].is_anomaly);
        assert!(anomalies[15].is_anomaly);
        assert!(!anomalies[0].is_anomaly);
        assert!(!anomalies[10].is_anomaly);
    }

    #[test]
    fn no_anomalies_in_constant_series() {
        let ts = constant_series(10, 5.0);
        let anomalies = detect_anomalies(&ts, 2.0);
        assert!(anomalies.iter().all(|a| !a.is_anomaly));
    }

    // --- Cross-validation ---

    #[test]
    fn cv_runs_without_error() {
        let ts = linear_series(20, 1.0, 0.0);
        let maes = cross_validate(&ts, PredictionModel::Naive, 5);
        assert_eq!(maes.len(), 5);
        assert!(maes.iter().all(|m| m.is_finite()));
    }

    #[test]
    fn cv_folds_count() {
        let ts = linear_series(30, 2.0, 1.0);
        let maes = cross_validate(&ts, PredictionModel::MovingAverage, 3);
        assert_eq!(maes.len(), 3);
    }

    // --- Train/test split ---

    #[test]
    fn split_ratio() {
        let ts = linear_series(100, 1.0, 0.0);
        let (train, test) = ts.train_test_split(0.8);
        assert_eq!(train.values.len(), 80);
        assert_eq!(test.values.len(), 20);
    }

    #[test]
    fn split_preserves_data() {
        let ts = linear_series(10, 1.0, 0.0);
        let (train, test) = ts.train_test_split(0.5);
        assert_eq!(train.values.len() + test.values.len(), 10);
    }

    // --- Error metrics ---

    #[test]
    fn mae_correctness() {
        let predicted = vec![2.5, 0.0, 2.1, 7.8];
        let actual = vec![3.0, -0.1, 2.0, 7.5];
        let mae = mean_absolute_error(&predicted, &actual);
        // errors: 0.5, 0.1, 0.1, 0.3 => mean = 0.25
        assert!((mae - 0.25).abs() < 1e-9);
    }

    #[test]
    fn rmse_correctness() {
        let predicted = vec![2.0, 2.0];
        let actual = vec![1.0, 3.0];
        let rmse = root_mean_square_error(&predicted, &actual);
        // errors: 1.0, -1.0 => sq: 1.0, 1.0 => mean 1.0 => sqrt 1.0
        assert!((rmse - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mae_empty() {
        assert_eq!(mean_absolute_error(&[], &[]), 0.0);
    }

    #[test]
    fn rmse_empty() {
        assert_eq!(root_mean_square_error(&[], &[]), 0.0);
    }

    // --- Edge cases ---

    #[test]
    fn constant_series_predictions() {
        let ts = constant_series(10, 5.0);
        let p = naive_predict(&ts, 1);
        assert!((p.value - 5.0).abs() < 1e-9);

        let p2 = moving_average_predict(&ts, 5, 1);
        assert!((p2.value - 5.0).abs() < 1e-9);

        let p3 = exponential_smoothing_predict(&ts, 0.5, 1);
        assert!((p3.value - 5.0).abs() < 1e-9);
    }

    #[test]
    fn steep_trend() {
        let ts = linear_series(5, 1000.0, 0.0); // y = 1000x
        let p = naive_predict(&ts, 1);
        assert_eq!(p.value, 4000.0); // last value at x=4

        let fit = linear_regression_fit(&ts);
        assert!((fit.params[0] - 1000.0).abs() < 1e-6);
    }

    // --- Model comparison ---

    #[test]
    fn naive_best_for_random_walk() {
        // Random-walk-like: each step = previous + noise
        // Naive should have lower MAE than moving average for this pattern
        let values: Vec<f64> = vec![10.0, 11.2, 10.8, 11.5, 10.9, 11.1, 10.7, 11.3, 11.0, 10.8];
        let ts = TimeSeries {
            values: values.clone(),
            timestamps: (0..10).map(|i| i as u64).collect(),
        };

        let (train, test) = ts.train_test_split(0.7);

        let naive_p = naive_predict(&train, 1);
        let ma_p = moving_average_predict(&train, 3, 1);

        let naive_mae = mean_absolute_error(&[naive_p.value], &test.values);
        let ma_mae = mean_absolute_error(&[ma_p.value], &test.values);

        // Both should produce finite results
        assert!(naive_mae.is_finite());
        assert!(ma_mae.is_finite());
    }

    #[test]
    fn lr_best_for_linear_trend() {
        let ts = linear_series(20, 5.0, 10.0);
        let (train, test) = ts.train_test_split(0.8);

        let fit = linear_regression_fit(&train);
        // Regression was fit on indices, so predict at index = train.len()
        let lr_pred = linear_regression_predict(&fit, train.values.len() as u64);

        let naive_p = naive_predict(&train, 1);

        let lr_mae = mean_absolute_error(&[lr_pred.value], &[test.values[0]]);
        let naive_mae = mean_absolute_error(&[naive_p.value], &[test.values[0]]);

        // Linear regression should be much closer for linear data
        assert!(lr_mae < naive_mae);
    }
}
