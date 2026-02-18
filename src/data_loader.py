import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = None
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = self.project_root / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)
    def __enter__(self):
        self.session = requests.Session()
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.session is not None:
            self.session.close()
    def load_symbols(self, to_csv: bool =False):
        URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")
        data = []
        for row in rows[1:]:  # skip header
            cols = row.find_all("td")
            if len(cols) >= 4:
                ticker = cols[0].text.strip()
                company = cols[1].text.strip()
                sector = cols[2].text.strip()
                industry = cols[3].text.strip()
                data.append({
                    "Ticker": ticker,
                    "Company": company,
                    "Sector": sector,
                    "Industry": industry
                })
        sp500_df = pd.DataFrame(data)
        if to_csv:
            file_path = self.data_path / 'sp500_symbols.csv'
            sp500_df.to_csv(file_path, index=False)
        return sp500_df
    def load_ticker(
            self,
            symbol: str = "AAPL",
            start: datetime = datetime.today() - timedelta(days=365.25*5),
            end: datetime = datetime.today(),
            limit: int = 1500,
            to_csv: bool = False
    ):
        key = self.api_key
        base_url = "https://api.marketstack.com/v2/eod"
        start_date = start.strftime(format="%Y-%m-%d")
        end_date = end.strftime(format="%Y-%m-%d")
        params = {
            "access_key": key,
            "symbols": symbol,
            "date_from": start_date,
            "date_to": end_date,
            "sort": "ASC",
            "limit": limit
        }
        all_data = []
        offset = 0
        params["offset"] = offset
        r = self.session.get(base_url, params=params, timeout=15)
        response_json = r.json()
        batch_data = response_json.get("data",)
        all_data.extend(batch_data)
        total_available = response_json["pagination"]["total"]
        history = pd.json_normalize(all_data)
        if to_csv:
            file_path = self.data_path / f'{symbol}.csv'
            history.to_csv(file_path, index=False)
        return history