import pandas as pd
import numpy as np

def ols_slope_t(y_window):
    n = len(y_window)
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = y_window.mean()
    denominator = np.sum((x - x_mean)**2)
    beta = np.sum((x - x_mean) * (y_window - y_mean)) / denominator
    residuals = y_window - (y_mean + beta * (x - x_mean))
    s_squared = np.sum(residuals**2) / (n - 2)
    se_beta = np.sqrt(s_squared / denominator)
    return beta / se_beta

class StockData:
    def __init__(self):
        self.symbol = None
        self.start = None
        self.end = None
        self.interval = None
        self.history = None
        self.has_data = False
        self.calculations = None
        self.calculations_states = None
        self.transition_matrix = None
        self.has_states = False
        self.pred_price_runs = None
        self.pred_state_runs = None
        self.drift = None
        self.volatility = None
        self.has_pred = False
        self.days = None
        self.beta = None
        self.alpha = None
        self.capm_df = None
        self.has_capm = False
    def transform_data(
            self,
            symbol: str = "AAPL",
            raw_history: pd.DataFrame = pd.DataFrame()
    ):
        history = raw_history.copy()

        # --- Parse and sort ascending for clean shift() operations ---
        history['Date'] = pd.to_datetime(history['date'])
        history = history.sort_values('Date', ascending=True).reset_index(drop=True)

        # --- Adjustment factor (backward from most recent) ---
        # Iterate descending: each day's factor reflects all corporate actions
        # that occurred on strictly later dates
        n = len(history)
        adj_factors = np.ones(n)
        cum_factor = 1.0

        for i in range(n - 1, -1, -1):
            adj_factors[i] = cum_factor
            row_split = history.at[i, 'split_factor'] if history.at[i, 'split_factor'] > 0 else 1.0
            raw_close = history.at[i, 'close']
            row_div   = history.at[i, 'dividend'] if history.at[i, 'dividend'] > 0 else 0.0
            div_ratio = (raw_close - row_div) / raw_close if row_div > 0 and raw_close > 0 else 1.0
            cum_factor *= (div_ratio / row_split)

        history['adj_factor'] = adj_factors

        # --- Adjusted OHLCV ---
        history['Open']   = history['open']   * history['adj_factor']
        history['High']   = history['high']   * history['adj_factor']
        history['Low']    = history['low']    * history['adj_factor']
        history['Close']  = history['close']  * history['adj_factor']
        history['Volume'] = history['volume'] / history['adj_factor']

        # --- Set index and trim ---
        history = (
            history[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            .set_index('Date')
            .sort_index()
        )

        # --- Derived columns ---
        history['Upper'] = np.where(history['Close'] >= history['Open'], history['Close'], history['Open'])
        history['Lower'] = np.where(history['Close'] <  history['Open'], history['Close'], history['Open'])

        history['Log Return']   = np.log(history['Close'] / history['Close'].shift(1))
        history['Avg Price']    = (history['Close'] + history['Low'] + history['Upper']) / 3
        history['Spread']       = history['High'] - history['Low']
        history['Gap']          = np.log(history['Open'] / history['Close'].shift(1))
        history['Total Return'] = np.cumsum(history['Log Return'])

        history.dropna(inplace=True)

        # --- Assign ---
        self.symbol   = symbol
        self.history  = history
        self.has_data = True

        return self.history
    def simulation_metrics(self):
        if not self.has_pred:
            print("No Monte Carlo results available.")
            return None
        S0 = self.history["Close"].iloc[-1]
        final_prices = np.array(self.pred_price_runs)[:, -1]
        returns = final_prices / S0 - 1
        pop = (returns > 0).mean()
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        avg_gain = gains.mean() if gains.size > 0 else 0.0
        avg_loss = losses.mean() if losses.size > 0 else 0.0
        expected_return = returns.mean()
        self.sim_pop = pop
        self.sim_avg_gain = avg_gain
        self.sim_avg_loss = avg_loss
        self.sim_expected_return = expected_return
        return {
            "probability_of_profit": pop,
            "average_gain": avg_gain,
            "average_loss": avg_loss,
            "expected_return": expected_return
        }
    def markov_states(self, days: int = 63):
        if not self.has_data:
            print("No data to analyze.")
            return

        calculations = self.history.copy()

        # --- Normalized price components ---
        calculations["Normal Open"] = np.log(calculations["Open"] / calculations["Close"].shift(1))
        calculations["Normal High"] = np.log(calculations["High"] / calculations["Open"])
        calculations["Normal Low"]  = np.log(calculations["Low"]  / calculations["Open"])
        calculations["Normal Close"]= np.log(calculations["Close"]/ calculations["Open"])

        # --- Yang-Zhang Variance & Volatility ---
        calculations["Normal Close Var"] = calculations["Normal Close"].rolling(window=days).var()
        calculations["Normal Open Var"]  = calculations["Normal Open"].rolling(window=days).var()
        calculations["RS Var"] = (
            calculations["Normal High"] * (calculations["Normal High"] - calculations["Normal Close"]) +
            calculations["Normal Low"]  * (calculations["Normal Low"]  - calculations["Normal Close"])
        ).rolling(window=days).mean()
        k = 0.34 / (1.34 + ((days + 1) / (days - 1)))
        calculations["YZ Var"] = (
            calculations["Normal Open Var"] +
            k * calculations["Normal Close Var"] +
            (1 - k) * calculations["RS Var"]
        )
        calculations["YZ Vol"] = np.sqrt(calculations["YZ Var"])

        # --- Trend State via OLS t-statistic ---
        calculations["Trend t-stat"] = (
            calculations["Log Return"]
            .rolling(window=days)
            .apply(ols_slope_t, raw=True)
        )

        enter = 1.5
        exit_ = 0.75
        trend_states = []
        prev_trend_state = 0
        for tau in calculations["Trend t-stat"]:
            if np.isnan(tau):
                trend_states.append(prev_trend_state)
                continue
            if prev_trend_state == 0:
                if tau >= enter:
                    trend_state = 1
                elif tau <= -enter:
                    trend_state = -1
                else:
                    trend_state = 0
            else:
                trend_state = 0 if abs(tau) <= exit_ else prev_trend_state
            trend_states.append(trend_state)
            prev_trend_state = trend_state
        calculations["Trend State"] = trend_states

        # --- Vol Cutoff (calendar-time correct) ---
        unique_trends = [-1, 0, 1]
        cutoff_window = int(days * 4)
        min_periods   = int(days * 0.5)

        for trend_val in unique_trends:
            calculations[f"_yz_masked_{trend_val}"] = calculations["YZ Vol"].where(
                calculations["Trend State"] == trend_val
            )
            calculations[f"_cutoff_raw_{trend_val}"] = (
                calculations[f"_yz_masked_{trend_val}"]
                .rolling(window=cutoff_window, min_periods=min_periods)
                .quantile(0.65)
                .ffill()
                .ewm(span=days, adjust=False)
                .mean()
            )

        calculations["Vol Cutoff"] = np.select(
            [calculations["Trend State"] == t for t in unique_trends],
            [calculations[f"_cutoff_raw_{t}"] for t in unique_trends]
        )

        # cleanup intermediates
        calculations.drop(
            columns=[c for c in calculations.columns if c.startswith("_")],
            inplace=True
        )

        # --- Vol State ---
        enter_vol = 1.05
        exit_vol  = 0.95
        vol_states = []
        prev_vol_state = 0
        for vol, cutoff in zip(calculations["YZ Vol"], calculations["Vol Cutoff"]):
            if np.isnan(vol) or np.isnan(cutoff):
                vol_states.append(prev_vol_state)
                continue
            if prev_vol_state == 0:
                vol_state = 1 if vol > cutoff * enter_vol else 0
            else:
                vol_state = 0 if vol < cutoff * exit_vol else 1
            vol_states.append(vol_state)
            prev_vol_state = vol_state
        calculations["Vol State"] = vol_states

        # --- Regime State (1–6) ---
        calculations["Regime State"] = (
            (calculations["Vol State"] * 3) +
            (calculations["Trend State"] + 1) + 1
        )

        # --- Drop NaNs before transition matrix computation ---
        calculations.dropna(
            subset=["Trend State", "Vol State", "Regime State", "Log Return"],
            inplace=True
        )

        # --- Lag columns for transition matrices ---
        calculations["Trend State Lag"] = calculations["Trend State"].shift(1)
        calculations["Vol State Lag"]   = calculations["Vol State"].shift(1)
        calculations.dropna(subset=["Trend State Lag", "Vol State Lag"], inplace=True)

        # --- Trend Transition Matrix (3x3) ---
        trend_matrix = np.zeros((3, 3))
        t0_arr = calculations["Trend State Lag"].to_numpy()
        tf_arr = calculations["Trend State"].to_numpy()
        for i, t0 in enumerate(unique_trends):
            total = (t0_arr == t0).sum()
            for j, tf in enumerate(unique_trends):
                count = ((t0_arr == t0) & (tf_arr == tf)).sum()
                if total > 0:
                    trend_matrix[i, j] = count / total

        trend_labels = {-1: "Bear", 0: "Neutral", 1: "Bull"}
        self.trend_transition_matrix = trend_matrix
        self.trend_tm_df = pd.DataFrame(
            trend_matrix,
            index=[trend_labels[s] for s in unique_trends],
            columns=[trend_labels[s] for s in unique_trends]
        ) 

        # --- Vol Transition Matrices (2x2 per trend, cross-boundary safe) ---
        vol_transition_dict = {}
        for trend_val in unique_trends:
            sub = calculations[
                (calculations["Trend State"] == trend_val) &
                (calculations["Trend State Lag"] == trend_val)
            ]
            v0_arr = sub["Vol State Lag"].to_numpy()
            vf_arr = sub["Vol State"].to_numpy()
            vol_matrix = np.zeros((2, 2))
            for i, v0 in enumerate([0, 1]):
                total = (v0_arr == v0).sum()
                for j, vf in enumerate([0, 1]):
                    count = ((v0_arr == v0) & (vf_arr == vf)).sum()
                    if total > 0:
                        vol_matrix[i, j] = count / total
            vol_transition_dict[trend_val] = pd.DataFrame(
                vol_matrix,
                index=["low", "high"],
                columns=["low", "high"]
            )
        self.vol_transition_dict = vol_transition_dict

        # --- Per-state return statistics ---
        self.calculations_states = (
            calculations.groupby("Trend State")["Log Return"]
            .agg(['mean', 'std'])
            .rename(columns={'mean': 'Daily Mean', 'std': 'Daily Std'})
        )
        self.calculations_states_6 = (
            calculations.groupby("Regime State")["Log Return"]
            .agg(['mean', 'std'])
            .rename(columns={'mean': 'Daily Mean', 'std': 'Daily Std'})
            .sort_index()
        )

        # --- Metadata ---
        self.calculations = calculations
        self.days = days
        self.has_states = True
    
    def monte_carlo(
        self,
        days: int = 63,
        num_runs: int = int(3e4)
    ):
        if not self.has_states:
            print("No Markov states to make predictions.")
            return

        days = int(days / 3)

        # --- Starting conditions ---
        start_price   = self.history['Close'].iloc[-1]
        start_trend   = int(self.calculations['Trend State'].iloc[-1])
        start_vol     = int(self.calculations['Vol State'].iloc[-1])

        # --- Precompute return stats indexed by Regime State (1-6) ---
        # calculations_states_6 is indexed by Regime State
        regime_means = self.calculations_states_6['Daily Mean'].to_dict()
        regime_stds  = self.calculations_states_6['Daily Std'].to_dict()

        # --- Transition matrices ---
        unique_trends  = [-1, 0, 1]
        trend_tm       = self.trend_transition_matrix          # (3,3) numpy array
        trend_idx      = {t: i for i, t in enumerate(unique_trends)}

        # vol_transition_dict: {trend_val: 2x2 DataFrame, index/cols = ["low","high"]}
        vol_tm = {
            t: self.vol_transition_dict[t].values              # (2,2) numpy array
            for t in unique_trends
        }

        # --- Regime state encoding (must match markov_states) ---
        def regime_state(trend, vol):
            return (vol * 3) + (trend + 1) + 1

        # --- Storage ---
        prices      = np.zeros((days, num_runs))
        trend_states= np.zeros((days, num_runs), dtype=int)
        vol_states  = np.zeros((days, num_runs), dtype=int)

        prices[0, :]       = start_price
        trend_states[0, :] = start_trend
        vol_states[0, :]   = start_vol

        # --- Simulation loop ---
        for t in range(1, days):
            prev_trends = trend_states[t - 1, :]   # (num_runs,)
            prev_vols   = vol_states[t - 1, :]     # (num_runs,)

            # 1. Transition trend state
            u_trend    = np.random.rand(num_runs)
            trend_rows = np.array([trend_idx[tr] for tr in prev_trends])  # row index into trend_tm
            cum_trend  = np.cumsum(trend_tm[trend_rows], axis=1)           # (num_runs, 3)
            new_trend_idx = np.argmax(cum_trend > u_trend[:, None], axis=1)
            new_trends    = np.array(unique_trends)[new_trend_idx]         # map back to -1/0/1

            # 2. Transition vol state conditional on new trend
            u_vol     = np.random.rand(num_runs)
            new_vols  = np.empty(num_runs, dtype=int)
            for trend_val in unique_trends:
                mask = new_trends == trend_val
                if not mask.any():
                    continue
                vtm      = vol_tm[trend_val]                               # (2,2)
                pv       = prev_vols[mask]                                 # vol state for this subset
                cum_vol  = np.cumsum(vtm[pv], axis=1)                     # (subset, 2)
                new_vols[mask] = np.argmax(cum_vol > u_vol[mask, None], axis=1)

            trend_states[t, :] = new_trends
            vol_states[t, :]   = new_vols

            # 3. Sample returns from 6-regime stats
            regimes = regime_state(new_trends, new_vols)                   # (num_runs,)
            mu      = np.array([regime_means[r] for r in regimes])
            sigma   = np.array([regime_stds[r]  for r in regimes])
            z       = np.random.randn(num_runs)
            prices[t, :] = prices[t - 1, :] * np.exp((mu - (sigma ** 2) / 2) + sigma * z)

        # --- Store results ---
        self.pred_price_runs  = prices.T.tolist()
        self.pred_trend_runs  = trend_states.T.tolist()
        self.pred_vol_runs    = vol_states.T.tolist()

        final_prices          = prices[-1, :]
        sim_log_returns       = np.log(final_prices / start_price)
        self.drift            = sim_log_returns.mean()
        self.volatility       = sim_log_returns.std()
        self.has_pred         = True
        self.simulation_metrics()

    def compute_capm_metrics(
            self,
            benchmark_df: pd.DataFrame= pd.DataFrame()
            ):
        if not self.has_data:
            print("No data to analyze.")
        else:
            df = self.history[['Log Return']].merge(
            benchmark_df[['Log Return']],
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=(f'_{self.symbol}', '_SPY')
            )
            df.columns = [self.symbol, 'SPY']
            beta, alpha = np.polyfit(df["SPY"], df[self.symbol], 1)
            self.beta = beta
            self.alpha = alpha
            self.capm_df = df
            self.has_capm = True