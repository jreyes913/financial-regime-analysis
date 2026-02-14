## Example
![Dashboard Example](./example.png)

## 📈 Methodology

This dashboard uses a **Markov-Switching lognormal diffusion model** to simulate forward price paths over a 63-day (quarterly) horizon.

Instead of assuming constant volatility like standard GBM, the model allows both **volatility and trend** to change depending on the current market regime.

---

### 1️⃣ Regime Detection

Daily log returns are computed as:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
$$

Volatility is estimated using the **Yang–Zhang estimator**, which captures both overnight jumps and intraday price ranges:

$$
\sigma^2_{YZ,t} = \sigma_o^2 + k \sigma_c^2 + (1-k) \sigma_{RS}^2
$$

where:

- $\sigma_o^2$ is the variance of overnight returns ($O_t / C_{t-1}$)  
- $\sigma_c^2$ is the variance of open-to-close returns ($C_t / O_t$)  
- $\sigma_{RS}^2$ is the Rogers–Satchell component capturing intraday range  
- $k = \frac{0.34}{1.34 + \frac{n+1}{n-1}}$  

Trends are estimated using a **rolling OLS slope t-statistic** over log returns:

$$
t_t = \frac{\hat{\beta}_t}{\mathrm{SE}(\hat{\beta}_t)}
$$

which measures the significance of the trend relative to volatility, normalizing drift by noise.

Volatility and trend are then combined into **six market regimes** using quantiles:

| Volatility | Trend (t-stat) | State |
|------------|----------------|-------|
| Low        | Up             | 3     |
| Low        | Down           | 4     |
| Medium     | Up             | 2     |
| Medium     | Down           | 5     |
| High       | Up             | 1     |
| High       | Down           | 6     |

- Low, medium, and high volatility are defined by historical percentiles of $\sigma^2_{YZ}$ (e.g., 30th and 85th).  
- Up/down trends are determined by the sign of the rolling t-stat.

---

### 2️⃣ Markov Transition Matrix

A first-order Markov transition matrix $P$ is estimated from historical state changes:

$$
P_{ij} = \mathbb{P}(S_{t+1}=j \mid S_t=i)
$$

This matrix governs how likely the system is to remain in the same regime or transition to another during simulation.

---

### 3️⃣ Regime-Conditional Parameters

For each state $i$, the conditional mean and standard deviation of returns are computed:

$$
\mu_i = \mathbb{E}[r_t \mid S_t = i]
$$

$$
\sigma_i = \mathrm{Std}(r_t \mid S_t = i)
$$

Returns are modeled as:

$$
r_t \mid S_t=i \sim \mathcal{N}(\mu_i, \sigma_i^2)
$$

So both drift and volatility vary depending on the active regime.

---

### 4️⃣ Monte Carlo Simulation

For each simulated path:

1. The next regime is sampled using the transition matrix.
2. A random shock $Z \sim \mathcal{N}(0,1)$ is drawn.
3. Price evolves according to a regime-dependent lognormal diffusion:

$$
S_{t+1} = S_t \exp\left( \mu_i - \frac{\sigma_i^2}{2} + \sigma_i Z \right)
$$

Over 1,000+ simulated paths, this approach captures:

- Volatility clustering  
- Trend-significance filtering  
- Regime persistence  
- State-dependent dispersion  
- Non-constant risk dynamics  

---

Jose Reyes  
Interested in data science

## Contact

📧 jstunner55@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jose-reyes-634768264)
