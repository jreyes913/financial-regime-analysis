import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from .engine import stockData

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
def format_mpf_axis(ax, df):
    years = df.index.year
    unique_years = sorted(years.unique())
    tick_indices = [np.where(years == y)[0][0] for y in unique_years]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(unique_years)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=0)

class stockPlots:
    def __init__(self, stock: stockData):
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
        axes.set_xlim((-63, 252*3 + 64))
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
            colors = {1: 'green', 2: 'orange', 3: 'red'}
            for state, color in colors.items():
                masked_price = np.where(
                    self.stock.calculations['State'] == state,
                    self.stock.calculations['Close'],
                    np.nan
                )
                axes.plot(
                    np.arange(len(self.stock.calculations))+63,
                    masked_price,
                    color=color,
                    linewidth=2
                )
            axes.set_xlim((-63, 252*3 + 63))
            axes.set_ylim((self.stock.history["Lower"].min()*0.9, self.stock.history["Upper"].max()*1.1))
            axes.set_ylabel("Price (USD)")
            axes.grid(visible=True, axis='y')
            format_mpf_axis(ax=axes, df=self.stock.history)
            return fig
        else:
            print("No data to analyze.")
    def plot_simulation(self):
        if self.stock.has_pred:
            fig, axes = plt.subplots(figsize=figsize, dpi=300)
            expected_value = np.mean(self.stock.pred_price_runs, axis=0)
            colors = {1: 'green', 2: 'orange', 3: 'red'}
            for idx, run in enumerate(self.stock.pred_price_runs):
                for state, color in colors.items():
                    masked_price = np.where(
                        np.array(self.stock.pred_state_runs[idx]) == state,
                        np.array(run),
                        np.nan
                    )
                    axes.plot(
                        np.arange(len(run)),
                        masked_price,
                        color=color,
                        linewidth=1,
                        alpha=0.7,
                    )
            axes.plot(
                np.arange(len(self.stock.pred_price_runs[0])),
                expected_value,
                linewidth=2,
                color='white',
            )
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.xaxis.set_visible(False)
            axes.grid(visible=True, axis='y')
            axes.set_ylabel("Price (USD)")
            return fig
        else:
            print("No Markov states to make predictions.")
