# CONTENUIX — Regime-Based Market Analytics

An interactive equity analytics dashboard built with Dash and Plotly, featuring hierarchical Markov regime detection, regime-conditional Monte Carlo simulation, CAPM analysis, and AI-generated insights.

## Example
![Dashboard Example](./example.png)

---

## Methodology

### Yang–Zhang Volatility
Volatility is estimated using the **Yang–Zhang estimator**, which accounts for overnight gaps and intraday price ranges. It is more robust than close-to-close volatility and forms the basis for vol regime classification.

### Trend Detection
Trend strength is measured using a rolling OLS slope t-statistic over a configurable window. This captures how significant a directional move is relative to noise, and is used to classify each day into one of three trend states: **Bear**, **Neutral**, or **Bull**.

### Hierarchical Regime Model
The model is two-layered. The trend state is the master driver, estimated via a hysteresis-gated t-stat threshold to prevent rapid state flickering. The volatility state — **Low** or **High** — is estimated conditionally within each trend state, using a rolling quantile cutoff that adapts in calendar time via EWM smoothing. The combination of three trend states and two vol states produces **six distinct market regimes**.

### Markov Transition Matrices
A first-order Markov trend transition matrix (3×3) is estimated from the historical regime sequence. Separately, three vol transition matrices (2×2) are estimated conditionally per trend state, with cross-boundary transitions excluded to avoid fabricating regime changes that never occurred.

### Regime-Conditional Monte Carlo
Each simulated path evolves by first sampling the next trend state from the trend transition matrix, then sampling the vol state from the corresponding conditional vol matrix. Returns are then drawn from the six-regime return distribution for the resolved regime. Running 20,000 paths this way captures volatility clustering, regime persistence, and asymmetric risk dynamics that standard GBM models miss.

### CAPM
Alpha and beta are estimated via OLS regression of the stock's log returns against SPY over the selected window.

### AI Summary
A locally-run LLM (Gemma 2B via Ollama) generates a plain-language technical insight from the simulation output and CAPM metrics, displayed inline on the dashboard.

---

## Features
- **Historical price chart** with adjusted OHLCV data
- **Markov regime state overlay** — six color-coded regimes over price history
- **Transition matrix panel** — tabbed view of trend and per-trend vol matrices
- **Monte Carlo simulation** — 20,000 paths with 5th/95th percentile bands
- **Simulation metrics** — probability of profit, average gain/loss, expected return
- **CAPM metrics** — alpha and beta vs SPY
- **AI-generated insight** — compact technical summary from a local LLM

---

## Stack
- [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/)
- [dash-mantine-components](https://www.dash-mantine-components.com/)
- [Tiingo API](https://www.tiingo.com/) for price data
- [Ollama](https://ollama.com/) + Gemma 2B for AI summaries

---

Jose Reyes — Data Science  
📧 jstunner55@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jose-reyes-634768264)