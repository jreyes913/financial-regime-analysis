# Hierarchical Hidden Markov Modeling by Analysis Strategy

## Legend

| Symbol | State |
|--------|-------|
| D | Downward Trend |
| U | Upward Trend |
| H | High Volatility |
| L | Low Volatility |

---

## Philosophy

The market is modeled as a hierarchy of latent regimes. We distinguish two analysis strategies that differ only in which variable governs the hierarchy:

- **Technical:** Trend is the parent, Volatility is the child.
- **Quantitative:** Volatility is the parent, Trend is the child.

Let $Z_t$ denote the parent regime and $Y_t$ the child regime at time $t$. Observations are conditionally generated from $(Z_t, Y_t)$.

The core thesis is that both investors observe the same market and use the same trend signal, but differ fundamentally in how they **prioritize and measure risk**. The technical analyst treats trend as the structural backdrop against which volatility is assessed. The quant treats volatility as the governing risk environment within which trend is interpreted. This hierarchy difference — not the trend signal itself — is the object of study.

---

## Model Comparison

| Metric | Technical (Trend → Volatility) | Quantitative (Volatility → Trend) |
|--------|-------------------------------|-----------------------------------|
| Basis | Heuristic / Chart-Based | Statistical / Model-Based |
| Parent Variable | Trend: $Z_t \in \{D, U\}$ | Volatility: $Z_t \in \{H, L\}$ |
| Child Variable | Volatility: $Y_t \in \{H, L\}$ | Trend: $Y_t \in \{D, U\}$ |
| State Order | D → {H(D), L(D)}, U → {H(U), L(U)} | H → {D(H), U(H)}, L → {D(L), U(L)} |
| Trend Derivation | Z-score of price relative to 200-day SMA | Z-score of price relative to 200-day SMA |
| Volatility Derivation | Log-scaled ATR relative to 200-day SMA | Log Yang–Zhang realized volatility |

**Note on trend states:** Both models use exactly two trend states (Down, Up). A three-state specification (Down, Neutral, Up) was evaluated and rejected — the Neutral state was statistically indistinguishable from Down, confirming that trend is genuinely bimodal in the data.

---

## Feature Definitions

### Shared Trend Signal

Both models use the identical trend measure — a 200-day z-score of price:

$$\text{trend} = \frac{\text{close} - \mu_{200}}{\sigma_{200}}$$

where $\mu_{200}$ is the 200-day rolling mean and $\sigma_{200}$ is the 200-day rolling standard deviation (population, $\text{ddof}=0$). This is dimensionless, symmetric around zero, stationary across tickers, and directly comparable across both models. Using the same trend signal in both models ensures that any disagreement in the agreement matrix is attributable **purely to the volatility measurement method and hierarchy**, not to differences in trend methodology.

### Technical Volatility

The technical analyst observes volatility visually through Average True Range (ATR) scaled to price level:

$$\text{vol\_technical} = \log\left(\frac{\text{ATR}_{14}}{\text{SMA}_{200}}\right)$$

where $\text{ATR}_{14}$ is the 14-day Average True Range and $\text{SMA}_{200}$ is the 200-day simple moving average used purely as a price-level scaling factor. The log transform is applied to satisfy the Gaussian assumption of the HMM, consistent with how technical analysts perceive volatility on a logarithmic scale visually.

### Quantitative Volatility

The quant measures volatility through the Yang–Zhang estimator, which accounts for overnight gaps, intraday range, and close-to-close returns:

$$r_t^{\text{open}} = \log\left(\frac{\text{open}_t}{\text{close}_{t-1}}\right), \quad r_t^{\text{close}} = \log\left(\frac{\text{close}_t}{\text{open}_t}\right)$$

$$\sigma^2_{\text{RS}} = \frac{1}{14}\sum_{t}\left[r_t^{\text{high}}\left(r_t^{\text{high}} - r_t^{\text{close}}\right) + r_t^{\text{low}}\left(r_t^{\text{low}} - r_t^{\text{close}}\right)\right]$$

$$k = \frac{0.34}{1.34 + \frac{14+1}{14-1}}$$

$$\sigma^2_{\text{YZ}} = \sigma^2_{\text{open}} + k\,\sigma^2_{\text{close}} + (1-k)\,\sigma^2_{\text{RS}}$$

$$\text{vol\_quant} = \log\left(\sqrt{\sigma^2_{\text{YZ}}}\right) = \frac{1}{2}\log\left(\sigma^2_{\text{YZ}}\right)$$

All rolling variances use population variance ($\text{ddof}=0$) to match the Yang–Zhang estimator's assumptions. The log transform places both volatility measures in the same units and scale, making the controlled comparison valid.

---

## Why the Hierarchy Differs

The technical analyst conditions volatility on trend because chart-based practice establishes trend direction first — *is the stock above or below its moving average?* — and then assesses whether the current move is large or small relative to recent history. Volatility is a qualifier within a trend context.

The quant conditions trend on volatility because quantitative risk management prioritizes the volatility regime first — *are we in a high-risk environment?* — and interprets trend only within that context. High volatility periods are associated with drawdown risk, which the quant monitors independently of direction. This is consistent with institutional risk frameworks based on VaR, CVaR, and maximum drawdown thresholds, where the high-volatility regime empirically occupies roughly 7–20% of trading days depending on the stock.

---

## HHMM Transition Structure

### State Spaces

**Technical model** joint states: $\{H(D),\ L(D),\ H(U),\ L(U)\}$

**Quantitative model** joint states: $\{D(H),\ U(H),\ D(L),\ U(L)\}$

### Transition Probability Formula

For any joint state transition:

$$\pi\left((z,y) \to (z',y')\right) = \pi_Z(z' \mid z) \times \begin{cases} Q_{y,y'}^{(z)} & z' = z \\ M_{y,y'}^{z \to z'} & z' \neq z \end{cases}$$

where $Q^{(z)}$ is the child transition matrix when the parent stays the same, and $M^{z \to z'}$ is the child mapping distribution when the parent switches.

### Technical Model Full Transition Matrix

Parent states: $\{D, U\}$, Child states within each parent: $\{H, L\}$

| From / To | H(D) | L(D) | H(U) | L(U) |
|-----------|------|------|------|------|
| **H(D)** | $\pi_Z(D\|D)\,Q_{H,H}^{(D)}$ | $\pi_Z(D\|D)\,Q_{H,L}^{(D)}$ | $\pi_Z(U\|D)\,M_{H,H}^{D\to U}$ | $\pi_Z(U\|D)\,M_{H,L}^{D\to U}$ |
| **L(D)** | $\pi_Z(D\|D)\,Q_{L,H}^{(D)}$ | $\pi_Z(D\|D)\,Q_{L,L}^{(D)}$ | $\pi_Z(U\|D)\,M_{L,H}^{D\to U}$ | $\pi_Z(U\|D)\,M_{L,L}^{D\to U}$ |
| **H(U)** | $\pi_Z(D\|U)\,M_{H,H}^{U\to D}$ | $\pi_Z(D\|U)\,M_{H,L}^{U\to D}$ | $\pi_Z(U\|U)\,Q_{H,H}^{(U)}$ | $\pi_Z(U\|U)\,Q_{H,L}^{(U)}$ |
| **L(U)** | $\pi_Z(D\|U)\,M_{L,H}^{U\to D}$ | $\pi_Z(D\|U)\,M_{L,L}^{U\to D}$ | $\pi_Z(U\|U)\,Q_{L,H}^{(U)}$ | $\pi_Z(U\|U)\,Q_{L,L}^{(U)}$ |

### Quantitative Model Full Transition Matrix

Parent states: $\{H, L\}$, Child states within each parent: $\{D, U\}$

| From / To | D(H) | U(H) | D(L) | U(L) |
|-----------|------|------|------|------|
| **D(H)** | $\pi_Z(H\|H)\,Q_{D,D}^{(H)}$ | $\pi_Z(H\|H)\,Q_{D,U}^{(H)}$ | $\pi_Z(L\|H)\,M_{D,D}^{H\to L}$ | $\pi_Z(L\|H)\,M_{D,U}^{H\to L}$ |
| **U(H)** | $\pi_Z(H\|H)\,Q_{U,D}^{(H)}$ | $\pi_Z(H\|H)\,Q_{U,U}^{(H)}$ | $\pi_Z(L\|H)\,M_{U,D}^{H\to L}$ | $\pi_Z(L\|H)\,M_{U,U}^{H\to L}$ |
| **D(L)** | $\pi_Z(H\|L)\,M_{D,D}^{L\to H}$ | $\pi_Z(H\|L)\,M_{D,U}^{L\to H}$ | $\pi_Z(L\|L)\,Q_{D,D}^{(L)}$ | $\pi_Z(L\|L)\,Q_{D,U}^{(L)}$ |
| **U(L)** | $\pi_Z(H\|L)\,M_{U,D}^{L\to H}$ | $\pi_Z(H\|L)\,M_{U,U}^{L\to H}$ | $\pi_Z(L\|L)\,Q_{U,D}^{(L)}$ | $\pi_Z(L\|L)\,Q_{U,U}^{(L)}$ |

---

## Historical Agreement Matrix

With both HHMMs fully specified, we measure how similarly they classify the same historical data.

Let each model produce a sequence of joint states:

$$S_t^{\text{tech}} \in \{H(D),\ L(D),\ H(U),\ L(U)\}$$

$$S_t^{\text{quant}} \in \{D(H),\ U(H),\ D(L),\ U(L)\}$$

Define the agreement proportion:

$$A_{i,j} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\left[S_t^{\text{tech}} = i \ \wedge \ S_t^{\text{quant}} = j\right]$$

### Agreement / Disagreement Matrix

| Technical ↓ / Quantitative → | D(H) | U(H) | D(L) | U(L) |
|------------------------------|------|------|------|------|
| **H(D)** | $A_{H(D),D(H)}$ | $A_{H(D),U(H)}$ | $A_{H(D),D(L)}$ | $A_{H(D),U(L)}$ |
| **L(D)** | $A_{L(D),D(H)}$ | $A_{L(D),U(H)}$ | $A_{L(D),D(L)}$ | $A_{L(D),U(L)}$ |
| **H(U)** | $A_{H(U),D(H)}$ | $A_{H(U),U(H)}$ | $A_{H(U),D(L)}$ | $A_{H(U),U(L)}$ |
| **L(U)** | $A_{L(U),D(H)}$ | $A_{L(U),U(H)}$ | $A_{L(U),D(L)}$ | $A_{L(U),U(L)}$ |

Cells on the **semantic diagonal** — where both models agree on the direction and volatility regime — represent agreement. Off-diagonal cells represent disagreement and are the primary object of analysis: when the models disagree, what market conditions prevail, and which model's classification proves more predictive?

---

## Conclusion

Together, these components form a unified HHMM framework that cleanly contrasts trend-led and volatility-led market views. The controlled design — identical trend signal, identical window lengths, log-transformed volatility measures in matching units — ensures that the agreement matrix isolates the effect of **hierarchy and volatility measurement philosophy** rather than confounding feature differences. This enables empirical comparison of historical state assignments between the two philosophies and provides a principled quantitative basis for studying how technical and quantitative investors perceive and prioritize market risk.