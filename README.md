# Portfolio Monte Carlo Lab

A local Python research simulator with three modes: the original BTC / S&P 500 / Gold experiment, sequential memecoin trading, and arbitrary-asset spot portfolios. The original command-line interface (CLI), calculations, JSON/CSV/PNG output, tests, and GitHub Actions remain available. **Educational research tool only; not financial advice or a real-world forecast.** No accounts, external APIs, keys, or hosted service are needed.

## Start in Windows PowerShell

Install Python 3.12 or newer and Git. In PowerShell:

```powershell
git clone https://github.com/ariefzzz5421/Monte-Carlo.git
cd Monte-Carlo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

Streamlit prints the local URL, normally `http://127.0.0.1:8501`; open it in your browser. If PowerShell blocks script activation, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run dashboard.py` directly. In VS Code or Cursor, open the repository folder and select `.venv\Scripts\python.exe` as the interpreter. On macOS/Linux, use `python3 -m venv .venv`, `source .venv/bin/activate`, then the same `python -m streamlit run dashboard.py` command.

## Choose a dashboard mode

After launching `python -m streamlit run dashboard.py`, use **Research mode** at the top:

- **BTC / S&P 500 / Gold (original)** keeps the existing three-asset simulator and its PNG/CSV/JSON exports.
- **Memecoin trading** simulates a bankroll through sequential realized exits. It offers a user-defined hypothetical outcome table or an empirical bootstrap of your own realized trades.
- **Spot portfolio** starts at BTC 50%, HYPE 20%, USDT 15%, XAUT 15%. You can add or remove assets and change weights. It offers correlated GBM or an empirical block bootstrap of your imported daily USD prices.

All modes start idle. **Computation progress** measures completed trades or months while the model runs. **Playback** only animates paths already calculated. At most 100 paths are drawn; all scenarios contribute to metrics and exports. Each new-mode run saves `simulation_summary.json`, `final_portfolio_values.csv`, and `percentile_paths.csv` in its own timestamped folder under `results/runs/`. Download these or the reusable JSON configuration from the dashboard. Historical raw CSV data stays local and is not copied into the result export.

### Original portfolio mode

1. Set capital, the three weights, annual expected returns, volatility, pairwise correlations, 100–100,000 scenarios, selected horizon, rebalancing, and seed in the sidebar. Weights must total 100%, and the correlation matrix must be mathematically valid.
2. Select **Run Simulation**. The computation bar reports completed monthly steps of the actual 10-year path calculation. The 1Y/5Y/10Y comparison uses that one shared calculation. After it finishes, the app writes the normal JSON, CSV, and PNG outputs.
3. Use **Play**, **Pause**, **Reset**, speed, and the timeline slider to replay *completed* scenario paths. Playback does not run more simulations. At most 100 paths are drawn; metrics and distributions use every scenario.
4. Inspect the outcome, risk, and horizon tabs. Download an individual JSON, CSV, or PNG, or a ZIP containing the full original output structure.

The dashboard starts idle and computes only after you press the button. If a configuration is invalid, it explains the problem and leaves the last successful result available. Its dark layout uses full-width charts and a sidebar that can collapse on narrow screens.

### Keep using the CLI

```powershell
python simulation.py
```

The first run writes `results/simulation_summary.json`, `results/final_portfolio_values.csv`, `results/percentile_paths.csv`, and five PNGs in `results/charts/`. Later unnamed runs use a new UTC timestamped folder under `results/runs/`, so prior results are retained. Dashboard and CLI outputs follow this same rule. Results are ignored by Git except for the empty folder marker.

## Run an original CLI experiment

```powershell
python simulation.py --help
python simulation.py --config configs/default.json
python simulation.py --years 10 --simulations 10000
python simulation.py --years 1 --simulations 2000 --rebalancing monthly --seed 42 --run-name monthly-study
python -m pytest -q
```

`--run-name` saves in `results/runs/<name>` and rejects a name that already exists. To reproduce an experiment, keep the same JSON file, seed, scenario count, horizon, and rebalancing rule. The summary records every effective assumption, including CLI overrides. The same inputs and software versions yield the same random scenarios. Version ranges are in `requirements.txt`; for strict archival reproduction, also save `python -m pip freeze` alongside your results.

## Change assumptions

Copy `configs/default.json` to another JSON file and pass it with `--config`. Edit initial capital, all three weights, annual expected returns, annual volatility, correlation matrix, horizon, scenario count, rebalancing, and seed there. Asset keys and the matrix row/column order are **BTC, S&P 500, Gold**. Weights must be nonnegative and sum to 1. Correlations must form a symmetric positive semidefinite matrix with 1s on the diagonal. The CLI reports a useful error if these rules fail.

The default inputs are **illustrative educational assumptions, not current estimates or predictions**:

| Asset | Weight | Expected simple 1Y return | Annual volatility |
|---|---:|---:|---:|
| BTC | 30% | 12% | 60% |
| S&P 500 | 40% | 8% | 20% |
| Gold | 30% | 5% | 16% |

The example pairwise correlations are BTC/S&P 500 = 0.30, BTC/Gold = 0.10, and S&P 500/Gold = 0.05. Initial capital is $10,000; horizon is 5 years; 10,000 scenarios use seed 42 with annual rebalancing.

## Memecoin trading research

The included `configs/memecoin.json` defines **hypothetical examples**, not observed trading frequencies. The six example outcomes include 0x total loss, 0.5x partial loss, and net realized exits of 2x, 10x, 50x, and 100x. Probabilities must sum to 1. A realized multiple is **net proceeds after fees divided by entry cost**. A token touching 100x at its all-time high says nothing about whether you sold at 100x.

Each modeled trade stakes either a fixed dollar position or, with reinvestment enabled, the initial position fraction of the *current* bankroll. Stake is capped at cash available. One trade settles before the next. Ruin is the share of runs that ever touch the configured bankroll threshold (default 10% of starting capital); probability of loss checks the final bankroll against the start. Mean, median, P5/P95/P99, maximum drawdown, terminal-loss VaR and expected shortfall describe the modeled runs, not your real trading edge.

In Windows PowerShell, run the dashboard as above and select **Memecoin trading**, or use the new optional CLI:

```powershell
python research.py memecoin --config configs/memecoin.json
```

For empirical bootstrap, make a UTF-8 CSV with this exact header and **your own realized trades**:

```text
timestamp,entry_cost,realized_net_proceeds,fees
```

Use ISO timestamps in chronological order. `realized_net_proceeds` is the dollar amount you actually received **after** the listed fees; fees are retained for audit and never deducted a second time. A zero exit is allowed. Timestamps validate and describe the input order; this model counts sequential trades and does not infer trading frequency or calendar time. Change `"model"` in a copy of `configs/memecoin.json` to `"empirical_bootstrap"`, then run:

```powershell
Copy-Item configs/memecoin.json configs/my-trades.json
# Edit my-trades.json and set model to empirical_bootstrap.
python research.py memecoin --config configs/my-trades.json --trades-csv C:\path\to\my-trades.csv
```

The bootstrap samples those observed realized multiples with replacement. It preserves the empirical outcome distribution, but it does not prove that sample is representative or model changing market conditions, trade selection, liquidity, slippage, or execution delay.

## Spot portfolio research

The included `configs/spot.json` supplies the requested example allocation and **explicit hypothetical future** annual simple expected returns, annual GBM volatilities, and a correlation matrix. These are not market estimates or predictions. Edit the asset table and correlation matrix in the dashboard, or copy and edit JSON for any 1–30 uniquely named assets. Weights must sum to 1; correlation must be symmetric and positive semidefinite. The model can rebalance monthly, annually, or never.

Windows PowerShell commands:

```powershell
python research.py spot --config configs/spot.json
```

For historical block bootstrap, supply **your own** UTF-8 daily USD price CSV. It must have `date` followed by one column per configured asset, in the same asset set as JSON. Dates must be unique, ascending, and contiguous calendar days; every asset needs a positive finite price on every date. At least 31 aligned dates are required. For the default asset set, the header is:

```text
date,BTC,HYPE,USDT,XAUT
```

Change `"method"` in a copy of `configs/spot.json` to `"block_bootstrap"`, then run:

```powershell
Copy-Item configs/spot.json configs/my-spot.json
# Edit my-spot.json and set method to block_bootstrap.
python research.py spot --config configs/my-spot.json --prices-csv C:\path\to\daily-prices.csv
```

The historical panel computes daily log returns, their annualized mean and volatility, and annualized covariance and correlation (365-day convention). Correlation with a constant-price asset is mathematically undefined; those assets are listed in `undefined_correlation_assets`, and off-diagonal display entries are set to zero by convention. These **descriptive historical estimates** appear in a separate JSON field; they never silently overwrite future GBM assumptions. Block bootstrap samples contiguous blocks of your historical daily *joint* returns, preserving within-block serial behavior and same-day cross-asset movements. It stitches independently sampled blocks into modeled months, so long regime dependence is not preserved. No price history is bundled or fetched.

Optional one-time stablecoin depeg and crypto crash shocks multiply the chosen holdings at the configured month; a multiplier of 0.8 means a 20% loss at that point. They are stress assumptions, not empirical event probabilities. Stress is applied before that month's optional rebalance. The output gives paths, P5/median/P95, final distribution, drawdown, and terminal-loss VaR/expected shortfall under the same USD loss convention as the original mode.

## What the model means

Monte Carlo simulation makes many random *possible outcomes under a chosen model*. It describes the spread of those modeled outcomes, not the chance of an actual future event. Here each asset follows **geometric Brownian motion (GBM)**: its monthly growth is positive and its log return has a drift plus a normally distributed shock. The annual input return is an **arithmetic expected one-year simple return**, not an assumed annual compounded rate. The annual volatility is the GBM diffusion parameter: the standard deviation of a one-year log-return shock. The code uses `ln(1 + expected_return)` in the drift so the model's expected one-year simple return matches that input. Each scenario's realized compound annual growth rate (CAGR) is calculated afterward from its final portfolio value.

Correlated shocks make assets tend to rise or fall together to the extent specified by the matrix. Higher positive correlation can weaken diversification in this model; low or negative correlation can improve it. The same seed generates all ten years once, and the 1Y, 5Y, and 10Y comparison slices the same scenarios. Monthly, annual, and no rebalancing mean that holdings are returned to target weights at the end of each month, each 12th month, or never. No contributions, withdrawals, fees, taxes, inflation, or dividends are modeled separately; the chosen return assumptions would have to incorporate any intended effects.

## Read the results

- `simulation_paths.png` shows up to 100 individual scenarios, a median, and initial capital.
- `percentile_band.png` shows the median with P5–P95 **simulation percentile band**. It is not a statistical confidence interval or a prediction guarantee.
- `final_values_histogram.png` shows the spread of final dollar values up to P99; an annotation counts outcomes beyond the axis, and all values remain in the CSV.
- `drawdown_distribution.png` shows each scenario's largest peak-to-trough portfolio loss over the selected horizon.
- `horizon_comparison.png` compares median and P5–P95 final values at 1Y, 5Y, and 10Y.

`simulation_summary.json` also gives the mean and median final value, P5/P95, the share finishing below initial capital or at least double, distributions of CAGR and maximum drawdown, and 95% Value at Risk (VaR) and expected shortfall (ES). **Loss convention:** terminal dollar loss = initial capital − final value over the selected full horizon. A negative loss is a gain. VaR is the 95th percentile of these losses; ES is the mean loss among scenarios at or above that percentile. Both can be negative if even the modeled lower tail gained money. They are *terminal* risk measures, not one-month or intrahorizon peak-loss measures. `final_portfolio_values.csv` contains one row per scenario; `percentile_paths.csv` contains each monthly P5/median/P95.

Sample output from the default 10,000-scenario experiment: ![Five-year simulated percentile band](docs/sample_percentile_band.png)

After running the project, open the other generated PNG files in `results/charts/`.

## GitHub Actions

Every push to `main`, or a manual **Actions → Portfolio simulation → Run workflow**, installs dependencies, runs pytest, and executes a small 500-scenario experiment. Its JSON, CSV, and charts are uploaded as a downloadable workflow artifact. The workflow does not deploy anything.

## Model limits

GBM assumes normally distributed log returns with fixed return, volatility, and correlation parameters. Real markets can have **fat tails** (more extreme moves than a normal model), abrupt regime changes, changing correlations, liquidity gaps, and estimation error. BTC, equity, and gold histories do not reliably determine future parameters. Rebalancing can require trades and costs absent from this model. The spot block bootstrap is limited by the coverage and quality of the uploaded dates. The memecoin bootstrap is limited by the realized trades you include and cannot represent unrecorded failures. More scenarios reduce Monte Carlo sampling noise **inside a model** but do not fix model risk or turn hypothetical probabilities into reliable forecasts.

## Code map

`dashboard.py` selects the local Streamlit mode; `src/legacy_dashboard.py` and `src/dashboard_charts.py` retain the original interactive experiment. `simulation.py`, `src/experiment.py`, `src/config.py`, `src/engine.py`, `src/portfolio.py`, `src/analytics.py`, and `src/visualization.py` retain the original CLI/model/export flow. `src/memecoin.py` and `src/spot.py` implement the new research models; `src/research_dashboard.py`, `src/research_experiment.py`, and `research.py` provide controls, exports, and optional new-mode CLI. `tests/` contains pytest checks. The MIT license is in `LICENSE`.
