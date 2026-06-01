# plato-predict

> Time series prediction primitives for PLATO — naive, moving average, exponential smoothing, linear regression, ARIMA

## What This Does

plato-predict provides lightweight statistical prediction methods for tile streams. No ML frameworks required. It implements five models: naive (last value), moving average, exponential smoothing, linear regression, and ARIMA(0,1,1). Each model can be fitted to data and used to predict future values with confidence estimates.

## The Key Idea

Before deploying a neural network, try statistics. A naive predictor ("tomorrow will be like today") is surprisingly hard to beat. Moving average smooths out noise. Exponential smoothing weights recent observations more. Linear regression captures trends. ARIMA combines differencing with moving average. For most sensor data, one of these is good enough.

## Install

```bash
cargo add plato-predict
```

## Quick Start

```rust
use plato_predict::*;

let ts = TimeSeries::from_tiles(&[
    (1000, 20.0), (2000, 21.0), (3000, 22.0),
    (4000, 23.0), (5000, 24.0),
]);

// Fit and predict with exponential smoothing
let fit = fit_exponential_smoothing(&ts.values, 0.3);
let prediction = predict_next(&fit, 1); // 1 step ahead
println!("Predicted: {:.1} (confidence: {:.2})", prediction.value, prediction.confidence);

// Anomaly scoring
let scores = anomaly_scores(&ts.values, &fit);
for score in &scores {
    if score.is_anomaly {
        println!("Anomaly at value {:.1} (score: {:.2})", score.value, score.score);
    }
}
```

## API Reference

| Type | Description |
|---|---|
| `TimeSeries { values, timestamps }` | Observations. `from_tiles()`, `train_test_split()` |
| `PredictionModel` | `Naive` / `MovingAverage` / `ExponentialSmoothing` / `LinearRegression` / `Arima` |
| `Prediction { value, confidence, horizon, model }` | A single prediction |
| `ModelFit { model, params, residual_std, fitted }` | Fitted model parameters |
| `AnomalyScore { value, score, threshold, is_anomaly }` | Per-observation anomaly detection |

### Fitting Functions

```rust
fit_naive(&values);
fit_moving_average(&values, window);
fit_exponential_smoothing(&values, alpha);
fit_linear_regression(&values);
fit_arima(&values, p, d, q);
```

### Prediction

```rust
predict_next(&fit, horizon);    // One step
predict_n(&fit, n);             // Multiple steps
anomaly_scores(&values, &fit);  // Score each observation
```

## How It Works

**Naive**: y[t+1] = y[t]. Surprisingly effective baseline.

**Moving Average**: y[t+1] = mean(y[t-w..t]). Smooths noise.

**Exponential Smoothing**: y[t+1] = α·y[t] + (1-α)·ŷ[t]. Recursive; α controls responsiveness.

**Linear Regression**: Fits y = a + b·t via least squares. Captures trends.

**ARIMA(0,1,1)**: Differencing + exponential smoothing on differences. Handles non-stationary data.

## Testing

29 tests: naive prediction, moving average, exponential smoothing, linear regression, ARIMA, anomaly scoring, train/test split, confidence intervals, edge cases.

## License

Apache-2.0
