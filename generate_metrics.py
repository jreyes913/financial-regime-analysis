from dotenv import dotenv_values
from src.engine import StockData
from src.data_loader import DataLoader
from datetime import datetime, timedelta
from warnings import filterwarnings
from pathlib import Path
import pandas as pd
import time

filterwarnings("ignore")

# --------------------------------------------------
# Paths
# --------------------------------------------------
project_root = Path(__file__).parent
data_dir = project_root / "data"
cache_dir = project_root / "cache"
data_dir.mkdir(exist_ok=True)
cache_dir.mkdir(exist_ok=True)

today = datetime.today().date()
log_file = data_dir / f"metrics-{today}.txt"
resume_file = data_dir / "completed.txt"

# --------------------------------------------------
# Config
# --------------------------------------------------
config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["TIINGO_KEY"]

start_date_5y = datetime.today() - timedelta(days=365.25 * 5)
start_date_1y = datetime.today() - timedelta(days=365.25 * 1)
end_date = datetime.today()

window = 63
SPY = "SPY"

SECONDS_PER_TICKER = 145  # 2 requests per ticker → 50/hour

# --------------------------------------------------
# Resume support
# --------------------------------------------------
completed = set()

# --------------------------------------------------
# Logging
# --------------------------------------------------
with log_file.open("w") as file:

    print("Starting...", file=file, flush=True)
    start = time.perf_counter()

    with DataLoader(api_key=api_key) as loader:

        symbol_df = loader.load_symbols()

        # --------------------------------------------------
        # SPY (cached)
        # --------------------------------------------------
        spy_cache = cache_dir / "SPY.parquet"

        if spy_cache.exists():
            raw_spy = pd.read_parquet(spy_cache)
        else:
            raw_spy = loader.load_ticker(symbol=SPY, start=start_date_1y, end=end_date)
            raw_spy.to_parquet(spy_cache)

        spy_data = StockData()
        spy_data.transform_data(symbol=SPY, raw_history=raw_spy)

        # --------------------------------------------------
        # Main loop
        # --------------------------------------------------
        for idx, row in symbol_df.iterrows():

            symbol = row["Ticker"]
            sector = row["Sector"]
            industry = row["Industry"]
            company_name = row["Company"]

            if symbol in completed:
                continue

            t0 = time.time()

            try:
                print(f"{idx+1}/{len(symbol_df)} : {symbol}", file=file, flush=True)

                cache_file = cache_dir / f"{symbol}.parquet"

                if cache_file.exists():
                    raw = pd.read_parquet(cache_file)
                else:
                    raw = loader.load_ticker(symbol=symbol, start=start_date_5y, end=end_date)
                    raw.to_parquet(cache_file)

                data = StockData()
                data.transform_data(symbol=symbol, raw_history=raw)
                data.markov_states(days=window)
                data.monte_carlo(days=window)
                data.compute_capm_metrics(benchmark_df=spy_data)

                # --------------------------------------------------
                # Output
                # --------------------------------------------------
                print("-" * 50, file=file)
                print(f"{symbol} Summary", file=file)
                print("-" * 50, file=file)

                print("Fundamental Information", file=file)
                print(f"Ticker   -> {symbol}", file=file)
                print(f"Company  -> {company_name}", file=file)
                print(f"Sector   -> {sector}", file=file)
                print(f"Industry -> {industry}", file=file)
                print(file=file)

                print("CAPM Metrics", file=file)
                print(f"Alpha -> {getattr(data,'alpha',None)}", file=file)
                print(f"Beta  -> {getattr(data,'beta',None)}", file=file)
                print(file=file)

                print("Simulation Parameters", file=file)
                print(f"Drift      -> {getattr(data,'drift',None)}", file=file)
                print(f"Volatility -> {getattr(data,'volatility',None)}", file=file)
                print(file=file)

                print("Monte Carlo", file=file)
                print(f"PoP   -> {getattr(data,'sim_pop',None)}", file=file)
                print(f"Gain  -> {getattr(data,'sim_avg_gain',None)}", file=file)
                print(f"Loss  -> {getattr(data,'sim_avg_loss',None)}", file=file)
                print(f"ExpR  -> {getattr(data,'sim_expected_return',None)}", file=file)
                print(file=file)

                if hasattr(data, "trend_tm_df"):
                    print("Trend Transition Matrix", file=file)
                    print(data.trend_tm_df.round(3).to_string(), file=file)

                if hasattr(data, "calculations_states_6"):
                    print("Regime Means / Stds", file=file)
                    print(data.calculations_states_6.to_string(), file=file)

                print("-" * 50, file=file, flush=True)

                # --------------------------------------------------
                # Mark complete
                # --------------------------------------------------
                with open(resume_file, "a") as r:
                    r.write(f"{symbol}\n")

                completed.add(symbol)

            except Exception as e:
                print(f"{symbol} FAILED: {e}", file=file, flush=True)

            # --------------------------------------------------
            # Rate limiting
            # --------------------------------------------------
            elapsed = time.time() - t0
            sleep_time = max(0, SECONDS_PER_TICKER - elapsed)
            time.sleep(sleep_time)

    end = time.perf_counter()
    print(f"Complete in {end - start:.2f} seconds.", file=file, flush=True)