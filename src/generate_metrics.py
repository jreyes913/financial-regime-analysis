from dotenv import dotenv_values
from engine import StockData
from data_loader import DataLoader
from datetime import datetime, timedelta
from warnings import filterwarnings
import pandas as pd
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
data_file = project_root / "data" / "out.txt"

filterwarnings("ignore")

config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["TIINGO_KEY"]

start_date_5y = datetime.today() - timedelta(days=365.25 * 5)
start_date_1y = datetime.today() - timedelta(days=365.25 * 1)
end_date = datetime.today()
window = 63
SPY = "SPY"

with data_file.open("w") as file:
    print("Starting...", file=file)
    start = time.perf_counter()

    with DataLoader(api_key=api_key) as loader:
        symbol_df = loader.load_symbols()

        # Load SPY once — pass spy_data StockData object to compute_capm_metrics
        spy_data = StockData()
        raw_spy = loader.load_ticker(symbol=SPY, start=start_date_1y, end=end_date)
        spy_data.transform_data(symbol=SPY, raw_history=raw_spy)

        for idx, row in symbol_df.iterrows():
            symbol = row["Ticker"]
            sector = row["Sector"]
            industry = row["Industry"]
            company_name = row["Company"]

            try:
                data = StockData()
                raw = loader.load_ticker(symbol=symbol, start=start_date_5y, end=end_date)
                data.transform_data(symbol=symbol, raw_history=raw)
                data.markov_states(days=window)
                data.monte_carlo(days=window)
                data.compute_capm_metrics(benchmark_df=spy_data)

                print("-" * 50, file=file)
                print(f"{symbol} Summary", file=file)
                print("-" * 50, file=file)

                print("Fundamental Information", file=file)
                print(f"    Ticker               -> {symbol}", file=file)
                print(f"    Company              -> {company_name}", file=file)
                print(f"    Sector               -> {sector}", file=file)
                print(f"    Industry             -> {industry}", file=file)
                print(file=file)

                print("CAPM Metrics", file=file)
                print(f"    Alpha                -> {data.alpha:.6f}", file=file)
                print(f"    Beta                 -> {data.beta:.4f}", file=file)
                print(file=file)

                print("Simulation Parameters", file=file)
                print(f"    Drift (mu)           -> {data.drift:.4%}", file=file)
                print(f"    Volatility (sigma)   -> {data.volatility:.4%}", file=file)
                print(f"    Time Horizon         -> {data.sim_days} days / {data.sim_hours} hours", file=file)
                print(file=file)

                print("Monte Carlo Results", file=file)
                print(f"    Probability of Profit-> {data.sim_pop:.2%}", file=file)
                print(f"    Average Gain         -> {data.sim_avg_gain:.2%}", file=file)
                print(f"    Average Loss         -> {abs(data.sim_avg_loss):.2%}", file=file)
                print(f"    Expected Return      -> {data.sim_expected_return:.2%}", file=file)
                print(file=file)

                print("Markov Regime Statistics", file=file)
                print("    Trend Transition Matrix:", file=file)
                print(data.trend_tm_df.round(3).to_string(), file=file)
                print(file=file)

                print("    Regime Means / Stds (hourly):", file=file)
                print(data.calculations_states_6.to_string(), file=file)
                print("-" * 50, file=file)

                # Respect rate limit: 2 calls per ticker, 50/hour = ~25 tickers/hour
                # Sleep 145s between tickers to stay safely under the hourly cap
                time.sleep(145)

            except Exception as e:
                print(f"{symbol} failed: {e}")
                continue

    end = time.perf_counter()
    print(f"Complete in {end - start:.2f} seconds.", file=file)