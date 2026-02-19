import plotly.graph_objects as go
import numpy as np

# =========================
# DESIGN TOKENS
# =========================
DARK_BG = "#050A0E"
PANEL_BG = "#0C1318"
PANEL_BORDER = "#1A2530"
ACCENT_GREEN = "#00FFA3"
ACCENT_CYAN = "#00D4FF"
ACCENT_RED = "#FF4D6A"
TEXT_PRIMARY = "#E8F4F8"
TEXT_MUTED = "#4A6572"
TEXT_DIM = "#243540"
FONT_DISPLAY = "Syne, sans-serif"
FONT_MONO = "'Space Mono', monospace"

colors = {
    1: "#14532D",
    2: ACCENT_GREEN,
    3: "#BBF7D0",
    4: "#FECACA",
    5: ACCENT_RED,
    6: "#7F1D1D"
}


class StockPlots:
    def __init__(self, stock):
        self.stock = stock

    # =========================
    # Historical Candlestick
    # =========================
    def plot_historical_price(self):

        df = self.stock.history

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    increasing_line_color=ACCENT_GREEN,
                    decreasing_line_color=ACCENT_RED,
                    showlegend=False,
                )
            ]
        )

        fig.update_layout(
            #height=350,
            margin=dict(l=40, r=20, t=20, b=20),
            yaxis_title="",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
        )

        return fig

    # =========================
    # Markov State Overlay
    # =========================
    def plot_markov_states(self):

        if not self.stock.has_states:
            return go.Figure()

        df = self.stock.calculations

        fig = go.Figure()

        for state, color in colors.items():

            masked = np.where(df["State"] == state, df["Close"], np.nan)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=masked,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                )
            )

        fig.update_layout(
            #height=350,
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=20),
        )

        return fig

    # =========================
    # Monte Carlo Simulation
    # =========================
    def plot_simulation(self):

        if not self.stock.has_pred:
            return go.Figure()

        runs = np.array(self.stock.pred_price_runs)

        lower = np.percentile(runs, 5, axis=0)
        upper = np.percentile(runs, 95, axis=0)
        expected = np.mean(runs, axis=0)

        x = np.arange(len(expected))

        fig = go.Figure()

        # Upper bound
        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                line=dict(color=ACCENT_GREEN),
                name="95th percentile",
            )
        )

        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                fill="tonexty",
                line=dict(color=ACCENT_RED),
                name="5th percentile",
                fillcolor="rgba(255,255,255,0.15)",
            )
        )

        # Expected path
        fig.add_trace(
            go.Scatter(
                x=x,
                y=expected,
                line=dict(color=TEXT_PRIMARY, dash="dash", width=1),
                name="Expected",
            )
        )

        fig.update_layout(
            yaxis_title="",
            xaxis_visible=False,
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=20),
        )

        return fig
    
    def plot_greek(self):
        if not self.stock.has_capm:
            return go.Figure()

        df = self.stock.capm_df

        x = df["SPY"].values
        y = df[self.stock.symbol].values

        # regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = self.stock.alpha + self.stock.beta * x_line

        fig = go.Figure()

        # scatter
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Returns",
                opacity=0.6,
                marker=dict(color=ACCENT_CYAN)
            )
        )

        # regression
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                marker=dict(color=TEXT_PRIMARY)
            )
        )

        fig.update_layout(
            yaxis_title="",
            xaxis_visible=False,
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=20),
        )

        return fig
