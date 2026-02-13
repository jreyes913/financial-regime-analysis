import streamlit as st
import numpy as np
from src.engine import stockData
from src.visuals import stockPlots

st.set_page_config(page_title="Contenuix | Stock Analytics", layout="wide")
st.title("📈 Financial Data Analytics")
st.markdown("---")
with st.sidebar:
    with st.form("ticker_form"):
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        submit_button = st.form_submit_button("Simulate")
if submit_button:
    stock = stockData()
    stock.ticker_data(symbol=symbol)
    stock.markov_states()
    stock.monte_carlo()
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
        st.markdown("### Markov State Meanings")
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width:18px; height:18px; background-color:green; border-radius:4px; margin-right:10px;"></div>
            <span><b>State 1 — Low Volatility Regime</b><br>
            Volatility below the 30th percentile.<br>
        </div>

        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width:18px; height:18px; background-color:gold; border-radius:4px; margin-right:10px;"></div>
            <span><b>State 2 — Moderate Volatility Regime</b><br>
            Volatility between the 30th and 85th percentiles.<br>
        </div>

        <div style="display: flex; align-items: center;">
            <div style="width:18px; height:18px; background-color:red; border-radius:4px; margin-right:10px;"></div>
            <span><b>State 3 — High Volatility Regime</b><br>
            Volatility above the 85th percentile.<br>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    final_prices = np.array(stock.pred_price_runs)[:, -1]
    pop = (final_prices > stock.history["Close"].iloc[-1]).mean()
    var_95 = np.log(np.percentile(final_prices, 5) / stock.history["Close"].iloc[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric("Prob. of Profit", f"{pop:.2%}")
    col2.metric("95% VaR (%)", f"{var_95:.2%}")
    col3.metric("Expected Return", f"{stock.drift:.2%}")
else:
    st.info("Enter a ticker symbol in the sidebar and click 'Simulate' to begin.")