import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from .engine import StockData
import pandas as pd

streamlit_light = {
    "primary": "#FF4B4B",
    "background": "#FFFFFF",
    "secondary_background": "#F0F2F6",
    "text": "#31333F",
    # Default categorical colors for Streamlit charts
    "palette": ["#0068C9", "#83C9FF", "#FF2B2B", "#FFABAB", "#29B09D", "#7DEFA1"]
}

streamlit_style = {
    "figure.facecolor": streamlit_light["background"],
    "axes.facecolor": streamlit_light["background"],
    "savefig.facecolor": streamlit_light["background"],
    "axes.edgecolor": "#D5DAE5",       # Light gray border
    "axes.labelcolor": streamlit_light["text"],
    "xtick.color": "#555867",
    "ytick.color": "#555867",
    "grid.color": streamlit_light["secondary_background"],
    "text.color": streamlit_light["text"],
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.prop_cycle": plt.cycler(color=streamlit_light["palette"]),
}

plt.style.use(streamlit_style)
figheight = 3.2
figsize = (figheight*4, figheight)

colors = {
    1: "#14532D",  # dark green
    2: "#16A34A",  # green
    3: "#BBF7D0",  # light green
    4: "#FECACA",  # light red
    5: "#DC2626",  # red
    6: "#7F1D1D"   # darker red
}

def format_mpf_axis(ax, df):
    unique_years = sorted(df.index.year.unique())
    tick_indices = []
    for year in unique_years:
        ts = pd.Timestamp(year, 1, 1, tz='UTC')
        idx_array = np.where(df.index >= ts)[0]
        if len(idx_array) > 0:
            tick_indices.append(idx_array[0])
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([df.index[i].year for i in tick_indices])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=0)

class StockPlots:
    def __init__(self, stock: StockData):
        self.stock = stock
    def plot_historical_price(self):
        fig, axes = plt.subplots(figsize=figsize, dpi=300)
        
        mpf.plot(
            self.stock.history,
            type='candle',
            volume=False,
            style='yahoo',
            ax=axes,
            warn_too_much_data=int(2e3),
            scale_width_adjustment=dict(candle=10)
        )
        axes.set_xlim((-63, len(self.stock.history) + 63))
        axes.set_ylim((self.stock.history["Lower"].min()*0.9, self.stock.history["Upper"].max()*1.1))
        axes.yaxis.tick_left()
        axes.yaxis.set_label_position("left")
        axes.set_ylabel("Price (USD)")
        axes.grid(visible=True, axis='y')
        format_mpf_axis(ax=axes, df=self.stock.history)
        return fig
    def plot_markov_states(self):
        if self.stock.has_states:
            fig, axes = plt.subplots(figsize=figsize, dpi=300)
            for state, color in colors.items():
                masked_price = np.where(
                    self.stock.calculations['State'] == state,
                    self.stock.calculations['Close'],
                    np.nan
                )
                axes.plot(
                    np.arange(len(self.stock.calculations))+self.stock.days,
                    masked_price,
                    color=color,
                    linewidth=2
                )
            axes.set_xlim((-63, len(self.stock.history) + 63))
            axes.set_ylim((self.stock.history["Lower"].min()*0.9, self.stock.history["Upper"].max()*1.1))
            axes.set_ylabel("Price (USD)")
            axes.grid(visible=True, axis='y')
            format_mpf_axis(ax=axes, df=self.stock.history)
            return fig
        else:
            print("No data to analyze.")
    # def plot_simulation(self):
    #     if self.stock.has_pred:
    #         fig, axes = plt.subplots(figsize=figsize, dpi=300)
    #         expected_value = np.mean(self.stock.pred_price_runs, axis=0)
    #         for idx, run in enumerate(self.stock.pred_price_runs):
    #             for state, color in colors.items():
    #                 masked_price = np.where(
    #                     np.array(self.stock.pred_state_runs[idx]) == state,
    #                     np.array(run),
    #                     np.nan
    #                 )
    #                 axes.plot(
    #                     np.arange(len(run)),
    #                     masked_price,
    #                     color=color,
    #                     linewidth=0.8,
    #                     alpha=0.5,
    #                 )
    #         axes.plot(
    #             np.arange(len(self.stock.pred_price_runs[0])),
    #             expected_value,
    #             linewidth=2,
    #             color='white',
    #         )
    #         axes.spines['top'].set_visible(False)
    #         axes.spines['right'].set_visible(False)
    #         axes.xaxis.set_visible(False)
    #         axes.grid(visible=True, axis='y')
    #         axes.set_ylabel("Price (USD)")
    #         return fig
    #     else:
    #         print("No Markov states to make predictions.")
    def plot_simulation(self):
        if not self.stock.has_pred:
            print("No Markov states to make predictions.")
            return
        fig, axes = plt.subplots(figsize=figsize, dpi=100)
        runs = np.array(self.stock.pred_price_runs)  # shape: (n_runs, n_points)
        lower = np.percentile(runs, 5, axis=0)
        upper = np.percentile(runs, 95, axis=0)
        expected_value = np.mean(runs, axis=0)
        axes.plot(
            np.arange(runs.shape[1]),
            upper,
            color='green',
            linewidth=2,
            label='Lower'
        )
        axes.plot(
            np.arange(runs.shape[1]),
            lower,
            color='red',
            linewidth=2,
            label='Lower'
        )
        axes.fill_between(
            np.arange(runs.shape[1]),
            lower,
            upper,
            color='black',  # choose whatever you like
            alpha=0.2     # semi-transparent
        )
        axes.plot(
            np.arange(runs.shape[1]),
            expected_value,
            color='black',
            linewidth=2,
            linestyle='--',
            label='Expected'
        )
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.xaxis.set_visible(False)
        axes.grid(visible=True, axis='y')
        axes.set_ylabel("Price (USD)")
        axes.legend()
        return fig