# German Power Market Forecasting & BESS Arbitrage

This repository contains an end-to-end **electricity market analytics and machine learning project** using German/Luxembourg day-ahead market data from **SMARD (Bundesnetzagentur)**.

The project combines exploratory market analysis, **24-hour-ahead electricity price forecasting**, and a simplified **Battery Energy Storage System (BESS) arbitrage simulation**.

The main questions are:

- What market conditions are associated with negative electricity prices?
- Can machine learning improve on a simple 24-hour persistence forecast?
- Does improved forecasting accuracy also improve a simplified battery-trading decision?

---

## 📂 Dataset

The project uses hourly electricity-market data from **SMARD**, the market-data platform of the German Federal Network Agency.

- **Period:** 2022–2025
- **Bidding zone:** Germany/Luxembourg
- **Frequency:** Hourly
- **Prepared observations:** 35,064

The analysis uses:

- Day-ahead electricity prices
- Grid load
- Residual load
- Onshore wind generation
- Offshore wind generation
- Photovoltaic generation

The raw CSV files are not included in the repository because they can be downloaded directly from SMARD.

Official source:

https://www.smard.de

---

## 🕒 Time-Series Preparation and Daylight-Saving Time

SMARD timestamps are reported in German local time. This creates repeated clock hours during the autumn daylight-saving transition.

The project keeps:

- **Europe/Berlin time** for calendar interpretation
- **UTC** as the technical time-series and merge key

### Time-zone conversion

```python
prices["timestamp_berlin"] = (
    prices["start_time_local"]
    .dt.tz_localize("Europe/Berlin", ambiguous="infer")
)

prices["timestamp_utc"] = (
    prices["timestamp_berlin"]
    .dt.tz_convert("UTC")
)
```

The prepared time series was checked for duplicates and gaps:

```text
Rows: 35064
Duplicate local timestamps: 4
Duplicate UTC timestamps: 0
Non-hourly UTC gaps: 0
```

The four duplicated local timestamps are expected during the autumn daylight-saving transition. After conversion to UTC, the series contains no duplicate timestamps and no non-hourly gaps.

---

## 📊 Day-Ahead Electricity Price Analysis

Electricity prices varied strongly across the analysed years.

![Average Day-Ahead Electricity Price by Year](outputs/average_price_by_year.png)

The hourly profile also shows a clear intraday pattern, with lower average prices around midday and stronger evening prices.

![Average Day-Ahead Electricity Price by Hour](outputs/average_price_by_hour.png)

Across 2022–2025, the dataset contains:

- **1,400 negative-price hours**
- **3.99% of all observations**

Negative-price hours increased across the analysed years:

| Year | Negative-price hours |
|---:|---:|
| 2022 | 69 |
| 2023 | 301 |
| 2024 | 457 |
| 2025 | 573 |

The heatmap below shows when negative prices occurred most frequently by month and hour.

![Negative Electricity Prices by Month and Hour](outputs/negative_price_heatmap.png)

---

## ⚡ Market Conditions During Negative Prices

Price data was merged with electricity consumption and renewable-generation data using UTC timestamps.

Average market conditions were then compared between negative and non-negative price hours.

| Variable | Non-negative hours | Negative hours | Difference |
|---|---:|---:|---:|
| Grid load (MWh) | 54,155.27 | 48,821.20 | -9.85% |
| Residual load (MWh) | 32,522.62 | 4,259.67 | **-86.90%** |
| Solar generation (MWh) | 6,366.88 | 25,081.69 | **+293.94%** |
| Total wind generation (MWh) | 15,265.77 | 19,479.83 | +27.60% |
| Wind + solar (MWh) | 21,632.65 | 44,561.52 | **+105.99%** |

![Average Market Conditions: Negative vs Non-Negative Price Hours](outputs/negative_vs_nonnegative_market_conditions.png)

Residual load showed the strongest positive linear relationship with electricity price among the selected variables:

```text
Residual load correlation with price:  0.581
Wind + solar correlation with price:  -0.456
```

Grouping observations by residual-load decile shows how average electricity prices rise as residual load increases.

![Average Electricity Price by Residual Load Decile](outputs/price_by_residual_load_decile.png)

These findings describe **association rather than causation**.

---

## 🤖 24-Hour-Ahead Electricity Price Forecasting

The forecasting task predicts the electricity price **24 hours ahead**.

The feature set includes:

- Target hour
- Day of week
- Month
- Price lags: 24h, 48h and 168h
- Residual-load lags
- Grid-load lag
- Solar lag
- Wind lag
- Rolling price mean
- Rolling price volatility

### Example feature engineering

```python
model["price_lag_24h"] = (
    model["price_eur_mwh"].shift(24)
)
```

The data is split chronologically:

- **2022–2023:** Training
- **2024:** Validation and model comparison
- **2025:** Final held-out test period

A chronological split is used because random train/test splitting would mix earlier and later market observations.

---

## 📈 Model Comparison

Four forecasting approaches were compared:

- 24-hour Persistence
- Linear Regression
- Random Forest
- XGBoost

The persistence model acts as the main baseline and uses the electricity price **24 hours earlier**.

### 2024 validation results

| Model | MAE (€/MWh) | RMSE (€/MWh) | MAE improvement vs baseline |
|---|---:|---:|---:|
| 24h Persistence | 27.82 | 44.30 | 0.00% |
| Linear Regression | 26.11 | 39.29 | 6.13% |
| Random Forest | 26.28 | 41.68 | 5.53% |
| **XGBoost** | **25.20** | **38.46** | **9.42%** |

XGBoost produced the lowest validation MAE and RMSE, so it was selected for the final test.

### XGBoost feature importance

![XGBoost Top 15 Feature Importances](outputs/xgboost_feature_importance.png)

The 24-hour price lag is the strongest feature in the fitted model, showing the importance of recent daily price behaviour.

---

## 🎯 Final 2025 Evaluation

After model selection, XGBoost was retrained using the available 2022–2024 development data and evaluated on the held-out 2025 period.

| Model | MAE (€/MWh) | RMSE (€/MWh) |
|---|---:|---:|
| 2025 Persistence | 25.97 | 40.45 |
| **2025 XGBoost** | **22.61** | **33.81** |

**Final MAE improvement vs persistence: 12.93%**

A two-week sample shows that the model captures many of the broader movements in electricity prices while still missing some of the sharper price changes.

![Actual vs Predicted Day-Ahead Prices](outputs/actual_vs_predicted_2025.png)

---

## 🔋 Simplified BESS Arbitrage Simulation

The final section tests whether the forecasting improvement also has value in a simplified battery decision problem.

The simulation assumes:

- **5 MWh** trade size
- **90% round-trip efficiency**
- Maximum of one charge/discharge cycle per day
- Charge must occur before discharge
- No trade is selected when the signal does not indicate positive expected P&L

For each day, all valid charge-before-discharge pairs are checked.

### Trade-selection logic

```python
def select_best_trade(day, signal_col):
    best_charge = None
    best_discharge = None
    best_signal_pnl = 0.0

    for charge in range(len(day) - 1):
        for discharge in range(charge + 1, len(day)):
            buy_signal = day.loc[charge, signal_col]
            sell_signal = day.loc[discharge, signal_col]

            signal_pnl = (
                sell_signal * energy * efficiency
                - buy_signal * energy
            )

            if signal_pnl > best_signal_pnl:
                best_signal_pnl = signal_pnl
                best_charge = charge
                best_discharge = discharge

    return best_charge, best_discharge
```

Three signals are compared under the same trading rule:

1. **24h Persistence**
2. **XGBoost Forecast**
3. **Actual-Price Reference**

Realised P&L is always calculated using the actual electricity prices.

### BESS results

| Strategy | Total simulated P&L | Trading days | Loss-making days |
|---|---:|---:|---:|
| 24h Persistence | €168,013.28 | 365 | 8 |
| **XGBoost Forecast** | **€173,344.64** | **365** | **9** |
| Actual-Price Reference | €193,534.72 | 365 | 0 |

XGBoost generated approximately **€5,331 more simulated P&L than persistence**, equivalent to about **3.17% improvement** under the same simplified trading rule.

The result also shows that lower forecasting error does not guarantee a better trading decision every day: XGBoost produced 9 loss-making days compared with 8 for persistence.

![Cumulative Simplified BESS Arbitrage P&L](outputs/bess_cumulative_pnl_2025.png)

---

## 📁 Repository Structure

```text
german-power-market-forecasting-bess/
├── Power_Market_BESS_Project.ipynb
├── outputs/
│   ├── actual_vs_predicted_2025.png
│   ├── average_price_by_hour.png
│   ├── average_price_by_year.png
│   ├── bess_cumulative_pnl_2025.png
│   ├── daily_price_spread.png
│   ├── negative_price_heatmap.png
│   ├── negative_vs_nonnegative_market_conditions.png
│   ├── price_by_residual_load_decile.png
│   └── xgboost_feature_importance.png
└── README.md
```

The notebook contains the complete data-preparation, exploratory-analysis, forecasting and BESS-simulation workflow.

---

## 🔁 Reproducing the Project

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

Download the following hourly datasets from SMARD and place them in the same directory as the notebook:

```text
Day-ahead_prices_202201010000_202401010000_Hour.csv
Day-ahead_prices_202401010000_202601010000_Hour.csv

Actual_consumption_202201010000_202401010000_Hour.csv
Actual_consumption_202401010000_202601010000_Hour.csv

Actual_generation_202201010000_202401010000_Hour.csv
Actual_generation_202401010000_202601010000_Hour.csv
```

Then open:

```text
Power_Market_BESS_Project.ipynb
```

and run the notebook from top to bottom.

---

## ⚠️ Project Limitations

This project was developed as an **MSc Data Analytics learning and portfolio project**, not as a production electricity-trading system.

Important limitations include:

- The forecasting setup is a rolling **24-hour-ahead experiment**, not an exact reconstruction of the German day-ahead auction process.
- Exact publication timing of every realised load and generation observation is not reconstructed.
- Hyperparameters were selected manually rather than through extensive optimisation.
- Residual load overlaps with information already represented by grid load, wind and solar features.
- The BESS simulation does not include detailed state-of-charge dynamics, degradation, transaction costs, power limits or market-execution effects.
- The actual-price reference uses information unavailable in advance and is included only as a simplified reference case.

These limitations are intentionally documented rather than treating the project as a production-ready trading model.

---

## 🛠️ Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- Jupyter Notebook

---

## 📚 Data Source

**SMARD — Bundesnetzagentur**

German electricity-market data platform:

https://www.smard.de

The project uses data from the Germany/Luxembourg bidding zone covering **2022–2025**.

---

## 📌 Note

This project is intended for educational and portfolio purposes. The BESS results are outputs of a simplified simulation and should not be interpreted as achievable battery revenue, a trading recommendation or financial advice.
