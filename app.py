import dash
from dash import html, dcc, Input, Output, State, dash_table, _dash_renderer
from dotenv import dotenv_values
from datetime import datetime, timedelta
from warnings import filterwarnings

import dash_mantine_components as dmc

from src.engine import StockData
from src.visuals import StockPlots
from src.data_loader import DataLoader
from src.summary import stockSummary

filterwarnings("ignore")

# Required before Dash() when using dash-mantine-components
_dash_renderer._set_react_version("18.2.0")

# =========================
# CONFIG
# =========================
config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["MARKETSTACK_KEY"]
SPY = "SPY"

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap"
    ],
)
server = app.server

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

# =========================
# STATE COLORS / LABELS
# =========================
state_colors = {
    1: ACCENT_RED,
    2: ACCENT_CYAN,
    3: ACCENT_GREEN,
    4: "#FF8C42",
    5: TEXT_MUTED,
    6: "#7B61FF",
}

state_labels = {
    1: "Bear + Low Vol",
    2: "Neutral + Low Vol",
    3: "Bull + Low Vol",
    4: "Bear + High Vol",
    5: "Neutral + High Vol",
    6: "Bull + High Vol",
}


# =========================
# HELPER COMPONENTS
# =========================
def metric_card(title, value, accent=ACCENT_GREEN):
    return html.Div(
        [
            html.Div(
                title.upper(),
                style={
                    "fontSize": "0.6vw",
                    "color": TEXT_MUTED,
                    "letterSpacing": "0.12em",
                    "fontFamily": FONT_MONO,
                    "marginBottom": "0.6vh",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "1.4vw",
                    "fontWeight": 700,
                    "color": accent,
                    "fontFamily": FONT_MONO,
                },
            ),
        ],
        style={
            "padding": "1.2vh 1.2vw",
            "border": f"1px solid {PANEL_BORDER}",
            "borderTop": f"2px solid {accent}",
            "borderRadius": "0 0 6px 6px",
            "background": PANEL_BG,
            "flex": 1,
        },
    )


def legend_pill(color, label, number):
    return html.Div(
        [
            html.Div(
                f"S{number}",
                style={
                    "fontFamily": FONT_MONO,
                    "fontSize": "0.55vw",
                    "color": DARK_BG,
                    "background": color,
                    "borderRadius": "3px",
                    "padding": "0.2vh 0.4vw",
                    "fontWeight": 700,
                    "marginRight": "0.5vw",
                    "flexShrink": 0,
                },
            ),
            html.Div(
                label,
                style={
                    "fontSize": "0.65vw",
                    "color": TEXT_PRIMARY,
                    "fontFamily": FONT_MONO,
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "2.35vh"},
    )

def panel(title, children, flex=1, extra_style=None):
    s = {
        "flex": flex,
        "display": "flex",
        "flexDirection": "column",
        "background": PANEL_BG,
        "border": f"1px solid {PANEL_BORDER}",
        "borderRadius": "6px",
        "overflow": "hidden",
    }
    if extra_style:
        s.update(extra_style)
    return html.Div(
        [
            html.Div(
                title.upper(),
                style={
                    "fontSize": "0.6vw",
                    "color": TEXT_MUTED,
                    "letterSpacing": "0.15em",
                    "fontFamily": FONT_MONO,
                    "padding": "0.8vh 1vw",
                    "borderBottom": f"1px solid {PANEL_BORDER}",
                    "background": "#080F14",
                },
            ),
            html.Div(
                children,
                style={
                    "flex": 1,
                    "display": "flex",
                    "flexDirection": "column",
                    "overflow": "hidden",
                },
            ),
        ],
        style=s,
    )


# Shared styles dict for dmc.DatePickerInput so both pickers are identical
DATE_PICKER_STYLES = {
    "input": {
        "backgroundColor": "#080F14",
        "border": f"1px solid {PANEL_BORDER}",
        "color": ACCENT_CYAN,
        "fontFamily": FONT_MONO,
        "fontSize": "0.65vw",
        "height": "28px",
        "minHeight": "28px",
        "padding": "0 8px",
        "borderRadius": "4px",
    },
}


def labeled_date_picker(label, picker_id, date_value):
    """Label + dmc.DatePickerInput, styled to match the rest of the navbar."""
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize": "0.5vw",
                    "color": TEXT_MUTED,
                    "letterSpacing": "0.12em",
                    "fontFamily": FONT_MONO,
                    "marginBottom": "0.3vh",
                },
            ),
            dmc.DatePickerInput(
                id=picker_id,
                value=date_value,
                valueFormat="YYYY-MM-DD",
                w=130,
                styles=DATE_PICKER_STYLES,
            ),
        ]
    )

def _tm_table(df, cell_style, header_style):
    return html.Div(
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[
                dict(name=i, id=i, format=percentage, type="numeric") if i != "" else dict(name=i, id=i)
                for i in df.columns
            ],
            style_table={"width": "100%", "overflowX": "hidden", "background": PANEL_BG},
            style_cell=cell_style,
            style_header=header_style,
            style_data={"border": f"1px solid {PANEL_BORDER}"},
        ),
        style={"padding": "0.5vh 0.5vw"},
    )

# =========================
# LAYOUT
# Wrap everything in dmc.MantineProvider so the calendar popup
# inherits dark mode automatically — no CSS overrides needed.
# =========================
app.layout = dmc.MantineProvider(
    forceColorScheme="dark",
    theme={
        "fontFamily": FONT_MONO,
        "primaryColor": "teal",
        # Map our accent green onto Mantine's teal scale so selected
        # days, focus rings, and hover states all use #00FFA3.
        "colors": {
            "teal": [
                "#e6fff5", "#b3ffe0", "#80ffcc", "#4dffb8",
                "#1affa3", "#00FFA3", "#00cc82", "#009962",
                "#007041", "#004721",
            ],
        },
    },
    children=[
        html.Div(
            [
                # ---- TOP NAV BAR ----
                html.Div(
                    [
                        # Brand
                        html.Div(
                            [
                                html.Span("▲", style={"color": ACCENT_GREEN, "marginRight": "0.5vw", "fontSize": "1vw"}),
                                html.Span(
                                    "CONTENUIX",
                                    style={
                                        "fontFamily": FONT_DISPLAY,
                                        "fontSize": "1.1vw",
                                        "fontWeight": 800,
                                        "color": TEXT_PRIMARY,
                                        "letterSpacing": "0.1em",
                                    },
                                )
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),

                        # Controls
                        html.Div(
                            [
                                # Ticker
                                html.Div(
                                    [
                                        html.Label(
                                            "TICKER",
                                            style={
                                                "fontSize": "0.5vw",
                                                "color": TEXT_MUTED,
                                                "letterSpacing": "0.12em",
                                                "fontFamily": FONT_MONO,
                                                "marginBottom": "0.3vh",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Input(
                                            id="symbol",
                                            value="AAPL",
                                            debounce=True,
                                            style={
                                                "width": "5vw",
                                                "background": "#080F14",
                                                "border": f"1px solid {PANEL_BORDER}",
                                                "color": ACCENT_GREEN,
                                                "fontFamily": FONT_MONO,
                                                "fontSize": "0.8vw",
                                                "padding": "0.4vh 0.5vw",
                                                "borderRadius": "4px",
                                                "outline": "none",
                                                "textTransform": "uppercase",
                                            },
                                        ),
                                    ]
                                ),
                                # Window
                                html.Div(
                                    [
                                        html.Label(
                                            "WINDOW (DAYS)",
                                            style={
                                                "fontSize": "0.5vw",
                                                "color": TEXT_MUTED,
                                                "letterSpacing": "0.12em",
                                                "fontFamily": FONT_MONO,
                                                "marginBottom": "0.3vh",
                                                "display": "block",
                                            },
                                        ),
                                        dcc.Input(
                                            id="window",
                                            type="number",
                                            value=63,
                                            style={
                                                "width": "4vw",
                                                "background": "#080F14",
                                                "border": f"1px solid {PANEL_BORDER}",
                                                "color": ACCENT_CYAN,
                                                "fontFamily": FONT_MONO,
                                                "fontSize": "0.8vw",
                                                "padding": "0.4vh 0.5vw",
                                                "borderRadius": "4px",
                                                "outline": "none",
                                            },
                                        ),
                                    ]
                                ),
                                # Date pickers — clean, dark, no CSS hacks
                                labeled_date_picker(
                                    "START DATE",
                                    "start-date",
                                    (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d"),
                                ),
                                labeled_date_picker(
                                    "END DATE",
                                    "end-date",
                                    datetime.today().strftime("%Y-%m-%d"),
                                ),
                                # Run button
                                html.Button(
                                    [
                                        html.Span("▶ ", style={"color": DARK_BG}),
                                        html.Span("RUN ANALYSIS", style={"letterSpacing": "0.1em"}),
                                    ],
                                    id="run",
                                    n_clicks=0,
                                    style={
                                        "background": ACCENT_GREEN,
                                        "color": DARK_BG,
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "0.8vh 1.2vw",
                                        "fontFamily": FONT_MONO,
                                        "fontSize": "0.65vw",
                                        "fontWeight": 700,
                                        "cursor": "pointer",
                                        "alignSelf": "flex-end",
                                        "letterSpacing": "0.08em",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "1.2vw",
                                "alignItems": "flex-end",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "padding": "1vh 1.5vw",
                        "borderBottom": f"1px solid {PANEL_BORDER}",
                        "background": "#080F14",
                    },
                ),

                # ---- MAIN CONTENT ----
                dcc.Loading(
                    html.Div(
                        id="main-content",
                        style={
                            "flex": 1,
                            "padding": "1.2vh 1.2vw",
                            "overflow": "hidden",
                        },
                    ),
                    color=ACCENT_GREEN,
                    type="circle",  # or "dot", "default"
                )
            ],
            style={
                "background": DARK_BG,
                "fontFamily": FONT_DISPLAY,
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
                "color": TEXT_PRIMARY,
            },
        )
    ],
)

percentage = dash_table.FormatTemplate.percentage(2)


# =========================
# CALLBACK
# dmc.DatePickerInput exposes its value via "value" (not "date"),
# which is consistent with every other Dash input component.
# =========================
@app.callback(
    Output("main-content", "children"),
    Input("run", "n_clicks"),
    State("symbol", "value"),
    State("window", "value"),
    State("start-date", "value"),
    State("end-date", "value"),
)
def run_sim(n, symbol, window, start, end):

    if n == 0:
        return html.Div(
            [
                html.Div(
                    [
                        html.Div("▲", style={"fontSize": "3vw", "color": ACCENT_GREEN, "marginBottom": "1vh", "opacity": "0.3"}),
                        html.Div("Enter a ticker symbol and click", style={"fontFamily": FONT_MONO, "fontSize": "0.8vw", "color": TEXT_MUTED}),
                        html.Div("RUN ANALYSIS", style={"fontFamily": FONT_MONO, "fontSize": "0.8vw", "color": ACCENT_GREEN}),
                        html.Div("to begin.", style={"fontFamily": FONT_MONO, "fontSize": "0.8vw", "color": TEXT_MUTED}),
                    ],
                    style={"textAlign": "center"},
                )
            ],
            style={"height": "100%", "display": "flex", "alignItems": "center", "justifyContent": "center"},
        )

    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)

    with DataLoader(api_key=api_key) as loader:
        data = StockData()
        spy_data = StockData()

        raw = loader.load_ticker(symbol=symbol, start=start, end=end)
        raw_spy_history = loader.load_ticker(symbol=SPY, start=start, end=end)
        spy_df = spy_data.transform_data(symbol=SPY, raw_history=raw_spy_history)

        data.transform_data(symbol=symbol, raw_history=raw)
        data.markov_states(days=window)
        data.monte_carlo(days=window)
        data.compute_capm_metrics(benchmark_df=spy_df)

        plots = StockPlots(data)

        def dark_fig(fig):
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=9),
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(gridcolor=PANEL_BORDER, linecolor=PANEL_BORDER, tickcolor=TEXT_MUTED),
                yaxis=dict(gridcolor=PANEL_BORDER, linecolor=PANEL_BORDER, tickcolor=TEXT_MUTED),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8)),
            )
            return fig

        fig_price  = dark_fig(plots.plot_historical_price())
        fig_states = dark_fig(plots.plot_markov_states())
        fig_sim    = dark_fig(plots.plot_simulation())
        fig_greek  = dark_fig(plots.plot_greek())

        summary = {
            "ticker": symbol,
            "horizon_days": int(data.days / 3),
            "rolling_window": data.days,
            "drift": data.drift,
            "volatility": data.volatility,
            "alpha": data.alpha,
            "beta": data.beta,
            "probability_of_profit": data.sim_pop,
            "average_gain": data.sim_avg_gain,
            "average_loss": data.sim_avg_loss,
            "expected_return": data.sim_expected_return,
        }
        data_summary = stockSummary(summary=summary)
        data_summary.generate_summary()

    pop_color = ACCENT_GREEN if data.sim_pop > 0.5 else ACCENT_RED
    exp_color = ACCENT_GREEN if data.sim_expected_return > 0 else ACCENT_RED

    table_style_cell = {
        "fontSize": "0.5vw",
        "textAlign": "center",
        "padding": "0.3vh 0.4vw",
        "fontFamily": FONT_MONO,
        "background": PANEL_BG,
        "color": TEXT_PRIMARY,
        "border": f"1px solid {PANEL_BORDER}",
    }
    table_style_header = {
        "fontWeight": "600",
        "backgroundColor": "#080F14",
        "color": TEXT_MUTED,
        "border": f"1px solid {PANEL_BORDER}",
        "fontFamily": FONT_MONO,
        "fontSize": "0.45vw",
        "letterSpacing": "0.1em",
    }

    def stat_block(items):
        """Renders a list of (label, value, color) tuples as stacked stat rows."""
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(lbl, style={"fontSize": "0.5vw", "color": TEXT_MUTED, "fontFamily": FONT_MONO, "letterSpacing": "0.1em", "marginBottom": "0.3vh"}),
                        html.Div(val, style={"fontSize": "0.9vw", "color": col, "fontFamily": FONT_MONO, "fontWeight": 700, "marginBottom": "1.2vh"}),
                    ]
                )
                for lbl, val, col in items
            ],
            style={"padding": "1vh 1vw"},
        )

    return html.Div(
        [
            # Symbol header + summary row
            html.Div(
                [
                    # Left: symbol title
                    html.Div(
                        [
                            html.Span(symbol.upper(), style={"fontFamily": FONT_DISPLAY, "fontSize": "1.1vw", "fontWeight": 800, "color": ACCENT_GREEN, "letterSpacing": "0.05em"}),
                            html.Span(" / COMPREHENSIVE ANALYSIS", style={"fontFamily": FONT_MONO, "fontSize": "0.6vw", "color": TEXT_MUTED, "letterSpacing": "0.15em", "marginLeft": "0.5vw"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "flexShrink": 0},
                    ),
                    # Right: AI summary (wraps to more lines if long)
                    html.Div(
                        data_summary.message,
                        style={
                            "fontFamily": FONT_DISPLAY,
                            "fontSize": "0.8vw",
                            "color": TEXT_PRIMARY,
                            "lineHeight": "1.4",
                            "padding": "0.4vh 1vw",
                            "borderLeft": f"2px solid {PANEL_BORDER}",
                            "marginLeft": "1.5vw",
                            "flex": 1,
                            "whiteSpace": "normal",
                            "wordBreak": "break-word",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "1vh"},
            ),

            # Row 1
            html.Div(
                [
                    panel("Historical Price",      dcc.Graph(figure=fig_price,  style={"flex": 1.3}, config={"displayModeBar": False})),
                    panel("Markov Regime States",  dcc.Graph(figure=fig_states, style={"flex": 1.3}, config={"displayModeBar": False})),
                    panel("Transition Matrices",
                        dmc.Tabs(
                            [
                                dmc.TabsList(
                                    [
                                        dmc.TabsTab("TREND", value="trend", style={"fontSize": "0.45vw", "fontFamily": FONT_MONO, "color": TEXT_MUTED}),
                                        dmc.TabsTab("BEAR VOL", value="bear", style={"fontSize": "0.45vw", "fontFamily": FONT_MONO, "color": TEXT_MUTED}),
                                        dmc.TabsTab("NEUTRAL VOL", value="neutral", style={"fontSize": "0.45vw", "fontFamily": FONT_MONO, "color": TEXT_MUTED}),
                                        dmc.TabsTab("BULL VOL", value="bull", style={"fontSize": "0.45vw", "fontFamily": FONT_MONO, "color": TEXT_MUTED}),
                                    ],
                                    style={"borderBottom": f"1px solid {PANEL_BORDER}"},
                                ),
                                dmc.TabsPanel(
                                    _tm_table(data.trend_tm_df.reset_index().rename(columns={"index": ""}), table_style_cell, table_style_header),
                                    value="trend"
                                ),
                                dmc.TabsPanel(
                                    _tm_table(data.vol_transition_dict[-1].reset_index().rename(columns={"index": ""}), table_style_cell, table_style_header),
                                    value="bear"
                                ),
                                dmc.TabsPanel(
                                    _tm_table(data.vol_transition_dict[0].reset_index().rename(columns={"index": ""}), table_style_cell, table_style_header),
                                    value="neutral"
                                ),
                                dmc.TabsPanel(
                                    _tm_table(data.vol_transition_dict[1].reset_index().rename(columns={"index": ""}), table_style_cell, table_style_header),
                                    value="bull"
                                ),
                            ],
                            value="trend",
                            style={"display": "flex", "flexDirection": "column", "height": "100%"},
                        )
                    ),
                    panel("Regime Legend",
                        html.Div(
                            [legend_pill(state_colors[i], state_labels[i], i) for i in range(1, 7)],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "justifyContent": "space-evenly",
                                "height": "100%",
                                "padding": "0.8vh 1vw",
                            },
                        ),
                        flex=0.4
                    ),
                ],
                style={"display": "flex", "gap": "0.8vw", "height": "34vh", "marginBottom": "0.8vw"},
            ),

            # Row 2
            html.Div(
                [
                    panel("Monte Carlo Simulation", dcc.Graph(figure=fig_sim,   style={"flex": 1}, config={"displayModeBar": False}), flex=2),
                    panel("GBM Parameters",         stat_block([
                        ("DRIFT (μ)",     f"{data.drift:.4%}",      ACCENT_GREEN),
                        ("VOLATILITY (σ)",f"{data.volatility:.4%}", ACCENT_CYAN),
                        ("HORIZON",       f"{int(data.days/3)} days",      TEXT_PRIMARY),
                    ])),
                    panel("Factor Chart",  dcc.Graph(figure=fig_greek, style={"flex": 1}, config={"displayModeBar": False}), flex=2),
                    panel("CAPM Metrics",           stat_block([
                        ("ALPHA (α)", f"{data.alpha:.8f}", ACCENT_GREEN if data.alpha > 0 else ACCENT_RED),
                        ("BETA (β)",  f"{data.beta:.8f}",  ACCENT_CYAN),
                        ("VS BENCHMARK", "SPY",            TEXT_MUTED),
                    ])),
                ],
                style={"display": "flex", "gap": "0.8vw", "height": "28vh", "marginBottom": "0.8vw"},
            ),

            # Metrics row
            html.Div(
                [
                    metric_card("Probability of Profit", f"{data.sim_pop:.2%}",             pop_color),
                    metric_card("Average Gain",          f"{data.sim_avg_gain:.2%}",         ACCENT_GREEN),
                    metric_card("Average Loss",          f"{abs(data.sim_avg_loss):.2%}",    ACCENT_RED),
                    metric_card("Expected Return",       f"{data.sim_expected_return:.2%}",  exp_color),
                ],
                style={"display": "flex", "gap": "0.8vw", "height": "12vh"},
            ),
        ],
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False)