import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import httpx

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

class stockData:
    def __init__(self, api_key):
        self.api_key = api_key
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
    def ticker_data(
            self,
            symbol: str = "AAPL",
            interval: str = "1d",
            start: datetime = datetime.today() - timedelta(days=365.25*5),
            end: datetime = datetime.today(),
            limit: int = 1500
    ):
        key = self.api_key
        base_url = "https://api.marketstack.com/v1/eod"
        start_date = start.strftime(format="%Y-%m-%d")
        end_date = end.strftime(format="%Y-%m-%d")
        params = {
            "access_key" : key,
            "symbols" : symbol,
            "date_from" : start_date,
            "date_to" : end_date,
            "limit" : limit
        }
        start_date = start.strftime(format="%Y-%m-%d")
        end_date = end.strftime(format="%Y-%m-%d")
        with httpx.Client(http2=False, timeout=10.0) as client:
            r = client.get(base_url, params=params)
            r.raise_for_status()
            data = r.json()["data"]
        history = pd.json_normalize(data)
        history['date'] = pd.to_datetime(history['date'])
        history['Date'] = pd.to_datetime(history['date'])
        history = history.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        history = history[['Date', 'Open', 'High', 'Low', 'Close']]
        history = history.set_index('Date').sort_index()
        set_upper = lambda row: row["Close"] if row["Close"] >= row["Open"] else row["Open"]
        set_lower = lambda row: row["Close"] if row["Close"] < row["Open"] else row["Open"]
        history["Upper"] = history.apply(set_upper, axis=1)
        history["Lower"] = history.apply(set_lower, axis=1)
        history["Log Return"] = np.log(history["Close"]/history["Close"].shift(1))
        history["Avg Price"] = (history["Close"] + history["Low"] + history["Upper"]) / 3
        history["Spread"] = history["High"] - history["Low"]
        history["Gap"] = np.log(history["Open"] / history["Close"].shift(1))
        history["Total Return"] = np.cumsum(history["Log Return"])
        history.dropna(inplace=True)
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.history = history
        self.has_data = True
    def markov_states(
            self,
            days:int=63
    ):
        if not self.has_data:
            print("No data to analyze.")
        else:
            calculations = self.history.copy()
            calculations["Normal Open"] = np.log(
                calculations["Open"] / calculations["Close"].shift(1)
            )
            calculations["Normal High"] = np.log(
                calculations["High"] / calculations["Open"]
            )
            calculations["Normal Low"] = np.log(
                calculations["Low"] / calculations["Open"]
            )
            calculations["Normal Close"] = np.log(
                calculations["Close"] / calculations["Open"]
            )
            calculations["Normal Close Var"] = calculations["Normal Close"].rolling(window=days).var()
            calculations["Normal Open Var"] = calculations["Normal Open"].rolling(window=days).var()
            calculations["RS Var"] = (calculations["Normal High"]*(calculations["Normal High"] - calculations["Normal Close"]) + calculations["Normal Low"]*(calculations["Normal Low"] - calculations["Normal Close"])).rolling(window=days).mean()
            k = 0.34 / (1.34 + ((days+1)/(days-1)))
            calculations["YZ Var"] = calculations["Normal Open Var"] + k*calculations["Normal Close Var"] + (1-k)*calculations["RS Var"]
            calculations["Trend t-stat"] = calculations["Log Return"].rolling(window=days).apply(ols_slope_t, raw=True)
            trend_quantiles = calculations["Trend t-stat"].quantile([0.5]).values
            low_trend_quantiles = calculations[calculations["Trend t-stat"] < trend_quantiles[0]]["YZ Var"].quantile([0.3,0.6]).values
            high_trend_quantiles = calculations[calculations["Trend t-stat"] >= trend_quantiles[0]]["YZ Var"].quantile([0.3,0.6]).values
            def separator(row):
                vol = row["YZ Var"]
                mean = row["Trend t-stat"]
                if mean < trend_quantiles[0]:
                    if vol < low_trend_quantiles[0]:
                        return 4
                    elif vol < low_trend_quantiles[1]:
                        return 5
                    else:
                        return 6
                else:
                    if vol < high_trend_quantiles[0]:
                        return 3
                    elif vol < high_trend_quantiles[1]:
                        return 2
                    else:
                        return 1
            calculations["State"] = calculations.apply(separator, axis=1)
            states = calculations["State"].to_numpy()
            day_0 = states[:-1]
            day_f = states[1:]
            transition_matrix = np.zeros((6, 6))
            for i, state_0 in enumerate((1, 2, 3, 4, 5, 6)):
                total_occurrences_of_i = (day_0 == state_0).sum()
                for j, state_f in enumerate((1, 2, 3, 4, 5, 6)):
                    mask = (day_0 == state_0) & (day_f == state_f)
                    count_i_to_j = mask.sum()
                    if total_occurrences_of_i > 0:
                        transition_matrix[i, j] = count_i_to_j / total_occurrences_of_i
            calculations_states = calculations.groupby(by=["State"])["Log Return"].agg(['mean', 'std']).reset_index()
            calculations_states['Daily Mean'] = calculations_states['mean']
            calculations_states['Daily Std'] = calculations_states['std']
            calculations_states.index = calculations_states['State']
            calculations_states.drop(columns=['mean', 'std', 'State'], inplace=True)
            calculations.dropna(inplace=True)
            self.calculations_states = calculations_states
            self.transition_matrix = transition_matrix
            self.days = days
            self.calculations = calculations
            self.has_states = True
    def monte_carlo(
            self,
            days:int=63,
            num_runs:int=int(1e4)
    ):
        if self.has_states:
            start_price = self.history['Close'].iloc[-1]
            prices = np.zeros((days, num_runs))
            states = np.zeros((days, num_runs), dtype=int)
            current_state = int(self.calculations['State'].iloc[-1])
            prices[0, :] = start_price
            states[0, :] = current_state
            means = self.calculations_states['Daily Mean'].values
            stds = self.calculations_states['Daily Std'].values
            trans_mat = np.array(self.transition_matrix)
            for t in range(1, days):
                prev_states = states[t-1, :]
                u = np.random.rand(num_runs)
                probs = trans_mat[prev_states - 1]
                cum_probs = np.cumsum(probs, axis=1)
                new_states = np.argmax(cum_probs > u[:, None], axis=1) + 1
                states[t, :] = new_states
                mu = means[new_states - 1]
                sigma = stds[new_states - 1]
                z = np.random.randn(num_runs)
                prices[t, :] = prices[t-1, :] * np.exp((mu - (sigma**2)/2) + sigma * z)

            self.pred_price_runs = prices.T.tolist()
            self.pred_state_runs = states.T.tolist()
            self.drift = self.history["Log Return"].mean() * days
            self.volatility = self.history['Log Return'].std() * np.sqrt(days)
            self.has_pred = True
        else:
            print("No Markov states to make predictions.")