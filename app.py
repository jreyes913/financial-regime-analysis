import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from pathlib import Path
import os
import certifi
from dotenv import dotenv_values
from src.engine import stockData
from src.visuals import stockPlots
from datetime import datetime
from datetime import timedelta
from warnings import filterwarnings
filterwarnings("ignore")
config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["MARKETSTACK_KEY"]

st.set_page_config(page_title="Contenuix | Stock Analytics", layout="wide")
st.markdown("""
<style>
hr {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
st.title("📈 Financial Data Analytics")
st.markdown("---")
with st.sidebar:
    with st.form("ticker_form"):
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        window = st.number_input("Analyzing Window", value=63)
        start_date = st.date_input("Start Date", value=datetime.today()-timedelta(days=365.25*5))
        end_date = st.date_input("End Date", value=datetime.today())
        submit_button = st.form_submit_button("Simulate")

if submit_button:
    stock = stockData(api_key=api_key)
    stock.ticker_data(symbol=symbol, start=start_date, end=end_date)
    stock.markov_states(days=window)
    stock.monte_carlo(days=window)
    plots = stockPlots(stock)
    fig1 = plots.plot_markov_states()
    fig2 = plots.plot_historical_price()
    fig3 = plots.plot_simulation()
    st.markdown(f"### Analysis for **{symbol}**")
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("Historical Price Trend")
        if fig2: st.pyplot(fig2)
    with row1_col2:
        st.subheader("Markov State Transitions")
        if fig1: st.pyplot(fig1)
    st.markdown("---")
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.subheader("Monte Carlo Next Quarter Simulation")
        if fig3: st.pyplot(fig3)
    with row2_col2:
        st.markdown("### Forecast Methodology")
        st.write(f"""
        This forecast for **{symbol}** is generated using a Geometric Brownian Motion (GBM) model. 
        By iterating through thousands of potential price paths, we can visualize the 
        statistical probability of future price targets.
        **Key Quarterly Foreacast Parameters:**
        * **Drift ($\mu$):** {stock.drift:.2%}
        * **Volatility ($\sigma$):** {stock.volatility:.2%}
        * **Time Horizon:** 1Q (63 Days).
        """)
    with row2_col3:
        components.html("""
        <div style="
            background-color: #F9FAFB;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            font-family: sans-serif;
            font-size: 14px;
        ">

            <h4 style="margin-top:0;">Markov Regime Meanings</h4>

            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="width:18px; height:18px; background:#14532D; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 1</b> — High Volatility Profit</div>
            </div>

            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="width:18px; height:18px; background:#16A34A; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 2</b> — Medium Volatility Profit</div>
            </div>

            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="width:18px; height:18px; background:#BBF7D0; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 3</b> — Low Volatility Profit</div>
            </div>

            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="width:18px; height:18px; background:#FECACA; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 4</b> — Low Volatility Loss</div>
            </div>

            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <div style="width:18px; height:18px; background:#DC2626; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 5</b> — Medium Volatility Loss</div>
            </div>

            <div style="display:flex; align-items:center;">
                <div style="width:18px; height:18px; background:#7F1D1D; border-radius:5px; margin-right:12px;"></div>
                <div><b>State 6</b> — High Volatility Loss</div>
            </div>

        </div>
        """, height=330)
    st.markdown("### Summary Metrics")
    st.markdown("---")
    S0 = stock.history["Close"].iloc[-1]
    final_prices = np.array(stock.pred_price_runs)[:, -1]
    returns = final_prices / S0 - 1
    pop = (returns > 0).mean()
    avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    expected_return = returns.mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability of Profit", f"{pop:.2%}")
    col2.metric("Average Gain", f"{avg_gain:.2%}")
    col3.metric("Average Loss", f"{abs(avg_loss):.2%}")
    delta_color = "normal"
    col4.metric(
        "Expected Return",
        f"{expected_return:.2%}"
    )

else:
    st.info("Enter a ticker symbol in the sidebar and click 'Simulate' to begin.")