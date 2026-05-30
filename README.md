# plato-predict

Time series prediction primitives for PLATO tile streams.

## Overview

Lightweight statistical prediction methods (no ML frameworks):

- **Naive** — propagate last observed value
- **Moving Average** — average of last N values
- **Exponential Smoothing** — weighted smoothing with configurable α
- **Linear Regression** — OLS fit with slope, intercept, and residual analysis
- **ARIMA** — placeholder (falls back to exponential smoothing)

Also provides:
- **Anomaly detection** — z-score based outlier flagging
- **Cross-validation** — k-fold CV with MAE per fold
- **Error metrics** — MAE, RMSE

## Usage

```rust
use plato_predict::*;

let ts = TimeSeries::from_tiles(&[(100, 1.0), (200, 2.0), (300, 3.0)]);

let p = naive_predict(&ts, 1);
let p2 = moving_average_predict(&ts, 3, 1);
let p3 = exponential_smoothing_predict(&ts, 0.5, 1);

let fit = linear_regression_fit(&ts);
let anomalies = detect_anomalies(&ts, 2.0);
```

## License

Apache-2.0
