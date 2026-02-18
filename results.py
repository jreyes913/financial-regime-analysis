from dotenv import dotenv_values
from src.engine import stockData
from src.visuals import stockPlots
from src.data_loader import DataLoader
from datetime import datetime
from datetime import timedelta
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filterwarnings("ignore")
config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["MARKETSTACK_KEY"]
symbol = "AAPL"
start_date = datetime.today() - timedelta(days=365.25*5)
end_date = datetime.today()
window = 63

loader = DataLoader()
loader.load_symbols(to_csv=True)

exit()
with stockData(api_key=api_key) as data:
    aapl = data.ticker_data(symbol=symbol, start=start_date, end=end_date)
    sp500 = data.ticker_data(symbol="SPY", start=start_date, end=end_date)


df = aapl[['Log Return']].merge(
    sp500[['Log Return']],
    left_index=True,
    right_index=True,
    how='inner',
    suffixes=(f'_{symbol}', '_SPY')
)
df.columns = [symbol, 'SPY']
beta, alpha = np.polyfit(df["SPY"], df[symbol], 1)
x = np.linspace(df["SPY"].min(), df["SPY"].max(), 100)
y = alpha + beta * x
plt.figure()
plt.plot(x, y)
plt.scatter(df["SPY"], df[symbol])
plt.show()