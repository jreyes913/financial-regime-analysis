# 📈 Financial Data Analytics Dashboard

An interactive financial analytics dashboard built with Dash and Plotly, featuring regime-based Monte Carlo simulation, CAPM analysis, and real-time market data.

## Example
![Dashboard Example](./example.png)

---

## Methodology

### Regime Detection

Daily log returns are computed from historical price data. Volatility is estimated using the **Yang–Zhang estimator**, which accounts for overnight gaps and intraday price ranges — making it more robust than simple close-to-close volatility. Trend strength is measured using a rolling OLS slope t-statistic, which captures how significant a directional move is relative to the noise around it.

These two signals — volatility and trend — are combined to classify each trading day into one of **six market regimes**, ranging from high-volatility bull to high-volatility bear.

### Markov Transition Matrix

A first-order Markov transition matrix is estimated from the historical regime sequence. This captures how likely the market is to stay in its current regime or shift to another — for example, whether a low-volatility bull run tends to persist or quickly revert.

### Regime-Conditional Parameters

For each regime, the mean and standard deviation of returns are computed separately. This means both drift and volatility vary depending on which regime is active, rather than assuming a single constant value across all market conditions.

### Monte Carlo Simulation

Each simulated path evolves day-by-day by first sampling the next regime from the transition matrix, then drawing a return from that regime's distribution. Running 1,000+ paths this way captures volatility clustering, regime persistence, and non-constant risk dynamics that standard GBM models miss.

---

## Features

- **Historical price chart** with interactive zoom
- **Markov regime state overlay** showing detected regimes over time
- **State transition matrix** showing regime persistence probabilities
- **Monte Carlo simulation** with probability of profit, average gain/loss, and expected return
- **CAPM metrics** — alpha and beta computed against SPY

---

## Stack

- [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/)
- [dash-mantine-components](https://www.dash-mantine-components.com/)
- [MarketStack API](https://marketstack.com/) for price data

---

Jose Reyes — Data Science  
📧 jstunner55@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jose-reyes-634768264)