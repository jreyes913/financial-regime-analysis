import yfinance as yf
import numpy as np

class stockData:
    def __init__(self):
        self.symbol = ""
        self.period = ""
        self.interval = ""
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
    def ticker_data(
            self,
            symbol: str,
            period: str = "3y",
            interval: str = "1d"
    ):
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        set_upper = lambda row: row["Close"] if row["Close"] >= row["Open"] else row["Open"]
        set_lower = lambda row: row["Close"] if row["Close"] < row["Open"] else row["Open"]
        history["Upper"] = history.apply(set_upper, axis=1)
        history["Lower"] = history.apply(set_lower, axis=1)
        history["Log Return"] = np.log(history["Close"]/history["Close"].shift(1))
        history["Avg Price"] = (history["Close"] + history["Low"] + history["Upper"]) / 3
        history["Spread"] = history["High"] - history["Low"]
        history["Gap"] = np.log(history["Open"] / history["Close"].shift(1))
        history["Total Return"] = np.cumsum(history["Log Return"])
        #history.index = history.index.date
        history.dropna(inplace=True)
        history.drop(columns=["Dividends", "Stock Splits", "Volume"], inplace=True)
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.history = history
        self.has_data = True
    def markov_states(self):
        if not self.has_data:
            print("No data to analyze.")
        else:
            calculations = self.history.copy()
            calculations["Rolling Var"] = calculations["Log Return"].rolling(window=63).var()
            quantiles = calculations["Rolling Var"].quantile([0.3,0.85]).values
            separator = lambda x: 1 if x < quantiles[0] else 2 if x < quantiles[1] else 3
            calculations["State"] = calculations["Rolling Var"].apply(separator)
            states = calculations["State"].to_numpy()
            day_0 = states[:-1]
            day_f = states[1:]
            transition_matrix = np.zeros((3, 3))
            for i, state_0 in enumerate((1, 2, 3)):
                total_occurrences_of_i = (day_0 == state_0).sum()
                for j, state_f in enumerate((1, 2, 3)):
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
            self.calculations = calculations
            self.has_states = True
    def monte_carlo(
            self,
            days:int=63,
            num_runs:int=int(5e3)
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
            self.drift = self.history["Log Return"].mean() * 63
            self.volatility = self.history['Log Return'].std() * np.sqrt(63)
            self.has_pred = True
        else:
            print("No Markov states to make predictions.")