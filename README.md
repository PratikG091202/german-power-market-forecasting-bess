# German Electricity Price Forecasting & BESS Arbitrage

An end-to-end data analytics and machine learning project using German/Luxembourg electricity market data from **2022–2025**.

The project explores day-ahead electricity price behaviour, investigates the market conditions associated with negative prices, builds a **24-hour-ahead price forecasting model**, and tests whether the forecasts can support a simplified Battery Energy Storage System (BESS) arbitrage strategy.

> This is a learning project developed as part of my MSc Data Analytics journey. The forecasting and BESS sections are simplified experiments and are not intended to represent a production electricity-trading system.

## Project Questions

This project was built around three main questions:

1. What market conditions are associated with negative electricity prices?
2. Can a machine-learning model improve on a simple 24-hour persistence forecast?
3. Does better forecasting accuracy also improve decisions in a simplified battery-arbitrage simulation?

## Data

The data comes from **SMARD**, the electricity-market data platform of the German Federal Network Agency (Bundesnetzagentur).

Source: https://www.smard.de

The project uses hourly data for the Germany/Luxembourg bidding zone covering **2022–2025**:

- Day-ahead electricity prices
- Actual grid load and residual load
- Onshore wind generation
- Offshore wind generation
- Photovoltaic generation

After cleaning, the price time series contains **35,064 hourly observations**.

### Time-zone handling

SMARD timestamps are reported in German local time. Because daylight-saving time creates repeated and missing local clock hours, the project keeps:

- **Berlin local time** for calendar features such as hour, weekday and month
- **UTC** as the technical time-series and merge key

After conversion:

- Duplicate UTC timestamps: **0**
- Non-hourly UTC gaps: **0**

This check is important because lag-based forecasting assumes that consecutive observations represent consecutive hours.

## Project Workflow

The notebook follows this sequence:

1. Load and inspect SMARD price data
2. Clean timestamps and handle daylight-saving time
3. Explore day-ahead price behaviour and negative-price periods
4. Add electricity consumption and renewable-generation data
5. Analyse market conditions during negative-price hours
6. Engineer lag, rolling and calendar features
7. Split the data chronologically
8. Compare forecasting models
9. Evaluate the selected model on 2025
10. Use the forecasts in a simplified BESS arbitrage simulation

## Exploratory Analysis

The analysis identified **1,400 negative-price hours**, representing approximately **3.99%** of all observations between 2022 and 2025.

Negative-price periods were associated with very different system conditions compared with non-negative-price periods:

| Variable | Difference during negative-price hours |
|---|---:|
| Residual load | **86.9% lower** |
| Solar generation | **293.9% higher** |
| Total wind generation | **27.6% higher** |
| Wind + solar generation | **106.0% higher** |

Residual load had the strongest positive linear relationship with electricity price in the analysed data (`r ≈ 0.58`), while combined wind and solar generation showed a negative relationship (`r ≈ -0.46`).

These findings describe **association rather than causation**.

## Forecasting Approach

The target is the day-ahead electricity price **24 hours ahead**.

The feature set includes:

- Calendar features: hour, weekday and month
- Price lags: 24h, 48h and 168h
- Historical load and residual-load lags
- Historical wind and solar generation
- Rolling price mean and volatility features

The lag and rolling features are constructed so that information from inside the forecast horizon is not used.

### Chronological split

Because this is a time-series problem, the data is split chronologically instead of randomly:

- **2022–2023:** training
- **2024:** validation and model comparison
- **2025:** final held-out test period

The models compared are:

- 24-hour Persistence
- Linear Regression
- Random Forest
- XGBoost

## Model Results

### 2024 validation

| Model | MAE (€/MWh) | RMSE (€/MWh) | MAE improvement vs persistence |
|---|---:|---:|---:|
| 24h Persistence | 27.82 | 44.30 | 0.00% |
| Linear Regression | 26.11 | 39.29 | 6.13% |
| Random Forest | 26.28 | 41.68 | 5.53% |
| **XGBoost** | **25.20** | **38.46** | **9.42%** |

XGBoost produced the lowest validation MAE and RMSE, so it was selected for the final evaluation.

### 2025 held-out test

| Model | MAE (€/MWh) | RMSE (€/MWh) |
|---|---:|---:|
| 24h Persistence | 25.97 | 40.45 |
| **XGBoost** | **22.61** | **33.81** |

XGBoost improved MAE by **12.93%** relative to the 24-hour persistence benchmark.

The result suggests that the historical market and calendar features contain useful information beyond simply using the price 24 hours earlier.

## Simplified BESS Arbitrage Simulation

The final section connects forecasting performance to a simple economic decision problem.

For each day in 2025, the strategy searches all valid **charge-before-discharge** hour pairs and selects the pair with the highest expected value according to its signal.

The simulation uses:

- Trade size: **5 MWh**
- Round-trip efficiency: **90%**
- Maximum of one charge/discharge cycle per day
- No trade when the signal does not indicate positive expected P&L

Three signals are compared using the same trading mechanics:

1. **24h Persistence**
2. **XGBoost Forecast**
3. **Actual-Price Reference**

Realised P&L is calculated using the actual electricity prices.

### BESS results

| Strategy | Total simulated P&L | Trading days | Loss-making days |
|---|---:|---:|---:|
| 24h Persistence | €168,013.28 | 365 | 8 |
| **XGBoost Forecast** | **€173,344.64** | **365** | **9** |
| Actual-Price Reference | €193,534.72 | 365 | 0 |

Under this simplified rule, the XGBoost signal generated approximately **€5,331 more simulated P&L than persistence**, an improvement of about **3.17%**.

XGBoost still produced more loss-making days than persistence, showing that lower forecasting error does not guarantee a better trading decision every day.

The actual-price reference uses information that would not be available in advance and is included only as an upper reference for the simplified one-cycle strategy.

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- Jupyter Notebook

## How to Run the Project

Clone the repository and install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

Download the required hourly CSV files from SMARD and place them in the same working directory as the notebook.

The notebook expects the following files:

```text
Day-ahead_prices_202201010000_202401010000_Hour.csv
Day-ahead_prices_202401010000_202601010000_Hour.csv

Actual_consumption_202201010000_202401010000_Hour.csv
Actual_consumption_202401010000_202601010000_Hour.csv

Actual_generation_202201010000_202401010000_Hour.csv
Actual_generation_202401010000_202601010000_Hour.csv
```

Then open the notebook and run the cells from top to bottom.

## Repository Structure

```text
.
├── README.md
└── Power_Market_BESS_Project.ipynb
```

The raw SMARD CSV files can be downloaded separately using the file periods listed above.

## Limitations

This project intentionally keeps the modelling scope manageable.

- The forecast is a **rolling 24-hour-ahead experiment**, not an exact recreation of German day-ahead auction timing.
- The exact publication time of every realised SMARD load and generation observation is not reconstructed.
- Model hyperparameters were selected manually rather than through extensive optimisation.
- Residual load overlaps with information already represented by load, wind and solar features.
- The BESS model does not include detailed state-of-charge dynamics, degradation, transaction costs, power limits or market-execution effects.
- The actual-price BESS strategy is a reference case, not a realistic forecast.

Possible future improvements include using published day-ahead load and renewable forecasts, testing reduced feature sets, applying formal time-series cross-validation, and developing a more detailed battery model.

## What I Learned

The main goal of this project was not only to obtain the lowest forecasting error, but to understand the complete workflow behind a real time-series problem.

The project helped me practise:

- Cleaning and combining real market datasets
- Handling daylight-saving time in hourly data
- Avoiding leakage when creating lag and rolling features
- Using chronological validation for time-series forecasting
- Comparing machine-learning models with a simple baseline
- Interpreting model performance beyond a single metric
- Connecting forecasting results to a simplified business/economic decision

## Note

This project is for educational and portfolio purposes. The BESS simulation is simplified and should not be interpreted as a real trading strategy, financial recommendation or estimate of achievable battery revenue.
