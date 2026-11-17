# TimesFM vs Actual: Forecasting 2020–2025 from 2005–2019 Data

A 10-cell Jupyter notebook that tests how well Google's **TimesFM** (a pretrained time-series foundation model) forecasts a real market index it has never seen. It trains on nothing: the model gets only 2005–2019 monthly prices, forecasts 2020–2025, and the forecast is compared with what actually happened.

## What it does

1. Downloads daily prices from Yahoo Finance (`yfinance`) for 2005–2025.
2. Resamples to monthly closes.
3. Splits the data at 2020-01-01. Everything before is context, everything after is held out.
4. Runs TimesFM zero-shot, forecasting the whole 2020–2025 horizon in a single pass.
5. Computes the difference between predicted and actual prices.
6. Reports KPIs, charts, and a year-by-year breakdown.

## Why monthly data

A 6-year daily horizon is about 1,500 steps, far beyond TimesFM's practical horizon. Monthly closes give about 72 steps, which fits in one forecast call. Using monthly data also avoids feeding the model's own predictions back in as input, which would compound errors.

## Requirements

- Python 3.10+
- Internet access (Yahoo Finance and the Hugging Face model download)
- About 2 GB of disk for the model weights
- CPU is enough. A GPU is optional.

Packages (installed in cell 1): `timesfm[torch]`, `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## Quick start

1. Open the notebook in Jupyter, Colab, or VS Code.
2. Run the cells in order, 1 to 10.
3. Outputs are saved in the working directory: `timesfm_vs_actual.csv` and `timesfm_kpis.csv`.

## Configuration (cell 2)

| Variable | Default | Meaning |
|---|---|---|
| `TICKER` | `^GSPC` | Any Yahoo Finance symbol |
| `START` | `2005-01-01` | Start of the training window |
| `SPLIT` | `2020-01-01` | Train/forecast boundary |
| `END` | `2025-12-31` | End of the evaluation window |

Ticker notes:
- `^NSEI` (Nifty 50) works, but Yahoo's data only starts around Sept 2007.
- `^NSEBANK` (BankNifty) has a similarly shorter history.
- The `SPLIT` and `END` dates must leave at least a few years of actual data, or the KPIs are not meaningful.

## Cell-by-cell guide

| Cell | Purpose |
|---|---|
| 1 | Install packages and import libraries |
| 2 | Configure ticker and dates, download raw daily data |
| 3 | Resample to monthly, split into `train` and `actual`, set `HORIZON` |
| 4 | Load TimesFM 2.5 (200M) and compile with a forecast config |
| 5 | Forecast the full horizon using only pre-2020 data |
| 6 | Build the comparison table (actual, predicted, q10, q90, errors) |
| 7 | Compute and display the KPIs |
| 8 | Chart: history, actual vs predicted, 80% interval, gap shading |
| 9 | Four error diagnostic charts |
| 10 | Yearly summary table, bar chart, CSV export |

## Model configuration (cell 4)

- `max_context=1024`: more than the roughly 180 training months, so all history is used.
- `max_horizon=128`: above the roughly 72 months needed.
- `normalize_inputs=True`: scales the series internally, which helps with large price levels.
- `use_continuous_quantile_head=True`: produces quantile forecasts, used for the 80% interval.
- `infer_is_positive=True`: keeps predicted prices non-negative.
- `fix_quantile_crossing=True`: ensures q10 ≤ median ≤ q90.

## Output columns (`cmp` dataframe)

| Column | Meaning |
|---|---|
| `actual` | Real month-end close |
| `predicted` | TimesFM point forecast |
| `q10`, `q90` | Lower and upper bound of the 80% interval |
| `error` | `predicted − actual` (positive = over-predicted) |
| `abs_error` | Absolute error |
| `pct_error` | Error as a percentage of the actual price |
| `cum_abs_pct` | Running mean absolute error as a % of running mean price |

## KPIs explained (cell 7)

| KPI | How to read it |
|---|---|
| **MAE** | Average miss in index points |
| **RMSE** | Like MAE but penalizes large misses more. A big gap between RMSE and MAE means a few months were far off. |
| **MAPE %** | Average miss as a percent of price. Easiest to compare across assets. |
| **sMAPE %** | Symmetric version of MAPE, less distorted when prices are low |
| **Bias** | Mean signed error. Positive means the model over-predicted overall, negative means it under-predicted. |
| **R²** | 1 is perfect. Near 0 means no better than predicting the average. Negative means worse than predicting the average. |
| **Correlation** | Whether predicted and actual move together, ignoring level differences |
| **Directional accuracy %** | How often the predicted month-to-month direction matched reality. About 50% is coin-flip level. |
| **80% interval coverage %** | How often the actual price fell inside q10–q90. A well-calibrated model lands near 80%. |
| **Actual CAGR / Predicted CAGR** | Compound annual growth rate implied by the end points, from the last training price |
| **Final-month gap %** | How far off the forecast was at the end of the horizon |

## Visualizations

- **Chart 1 (cell 8):** full price history with the forecast, actual path, uncertainty band, and the shaded gap between them. The dotted line marks the split date.
- **Chart 2 (cell 9):**
  - Monthly % error bars (green = over-predicted, red = under-predicted)
  - Cumulative absolute error over time
  - Error distribution histogram
  - Predicted-vs-actual scatter, colored by time, with a perfect-forecast diagonal
- **Chart 3 (cell 10):** average yearly actual vs predicted prices.

## How to interpret the results

- Expect the forecast to be smooth. TimesFM captures trend and seasonality patterns well, but it cannot anticipate shocks.
- The 2020 COVID crash and the 2022 drawdown were unforeseeable from 2005–2019 data. Large errors in those years are expected and say little about the model's quality.
- Error typically grows with horizon. Check `cum_abs_pct` and the yearly table to see where accuracy degrades.
- Low interval coverage means the model's uncertainty was too narrow for the regime change.
- A single run on one asset is an anecdote, not a benchmark.

## Limitations

- **Single series, single split.** Results depend heavily on where the split falls.
- **One-shot long-horizon forecast.** This is deliberately hard. Rolling forecasts (re-forecasting each month with updated data) would score much better but answer a different question.
- **Price levels, not returns.** Errors on levels are dominated by trend. For trading use, evaluate returns or volatility instead.
- **No transaction costs or strategy logic.** This is a forecast evaluation only, not a backtest of a trading system.
- **Not financial advice.**

## Possible extensions

- Rolling-origin evaluation: forecast 1 to 12 months ahead repeatedly across 2020–2025.
- Compare against baselines such as naive (last value), drift, seasonal naive, and ARIMA/ETS.
- Forecast log returns instead of prices.
- Add covariates (e.g. VIX, rates) using TimesFM's covariate support.
- Run across multiple tickers and compare KPIs in one table.
- Try daily data with chunked, autoregressive forecasting.

## Troubleshooting

| Problem | Fix |
|---|---|
| `resample("ME")` raises an error | Older pandas. The notebook already falls back to `"M"`. |
| `module 'timesfm' has no attribute 'TimesFM_2p5_200M_torch'` | Outdated package. Upgrade with `pip install -U timesfm[torch]`, or install from the official GitHub repo. |
| Model download fails | Check the network and Hugging Face access. If offline, download the weights first and load them from a local path. |
| `yf.download` returns an empty frame | Rate limit or bad ticker. Retry, or check the symbol on Yahoo Finance. |
| Column or MultiIndex errors | Newer `yfinance` returns multi-level columns. Cell 2 handles this with `.squeeze()`. |
| Plots do not appear | Add `%matplotlib inline` at the top of cell 1. |

## Files

- `timesfm_vs_actual.ipynb`: the notebook
- `timesfm_vs_actual.csv`: monthly actual, predicted, and error table (generated)
- `timesfm_kpis.csv`: KPI summary (generated)

## Credits

- **TimesFM**: Google Research, a decoder-only foundation model for time-series forecasting
- **Market data**: Yahoo Finance via `yfinance`
