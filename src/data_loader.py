import requests
import pandas as pd
from tiingo import TiingoClient
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from warnings import filterwarnings
filterwarnings("ignore")


class DataLoader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = self.project_root / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.client = TiingoClient({'session': True, 'api_key': api_key})

    def __enter__(self):
        self.session = requests.Session()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.session is not None:
            self.session.close()

    def load_symbols(self, to_csv: bool = False):
        URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")
        data = []
        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) >= 4:
                data.append({
                    "Ticker": cols[0].text.strip(),
                    "Company": cols[1].text.strip(),
                    "Sector": cols[2].text.strip(),
                    "Industry": cols[3].text.strip()
                })
        sp500_df = pd.DataFrame(data)
        if to_csv:
            sp500_df.to_csv(self.data_path / 'sp500_symbols.csv', index=False)
        return sp500_df

    def _get_split_factors(self, symbol: str, start: datetime, end: datetime) -> pd.Series:
        """
        Fetch daily EOD data and compute a split adjustment factor per day.
        factor = adjClose / close — apply this to intraday prices to adjust for splits.
        """
        eod = self.client.get_dataframe(
            symbol,
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
            frequency='daily'
        )
        eod.index = pd.to_datetime(eod.index).date
        eod['split_factor'] = eod['adjClose'] / eod['close']
        return eod['split_factor']

    def load_ticker(
            self,
            symbol: str = "AAPL",
            start: datetime = datetime.today() - timedelta(days=365.25 * 5),
            end: datetime = datetime.today(),
            frequency: str = "1hour",
            to_csv: bool = False
    ) -> pd.DataFrame:
        # --- Intraday prices ---
        df = self.client.get_dataframe(
            symbol,
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
            frequency=frequency
        )

        # --- Feature columns ---
        df["day"] = df.index.date
        df["time"] = df.index.time
        df["hour"] = df.groupby("day")["time"].rank(method="dense").astype(int)

        # --- Apply split adjustment ---
        split_factors = self._get_split_factors(symbol, start, end)
        df["split_factor"] = df["day"].map(split_factors)

        # Fill any missing factors (e.g. holidays/gaps) forward then backward
        df["split_factor"] = df["split_factor"].ffill().bfill()

        price_cols = ["close", "high", "low", "open"]
        df[price_cols] = df[price_cols].multiply(df["split_factor"], axis=0)
        df.drop(columns=["split_factor"], inplace=True)

        if to_csv:
            df.to_csv(self.data_path / f'{symbol}.csv')

        return df