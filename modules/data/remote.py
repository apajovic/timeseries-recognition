import pandas as pd
import numpy as np
from typing import Optional

def get_binance_historical_data(trading_pair:str, interval: str, start_date: str, end_date:Optional[str] = None) -> pd.DataFrame:
    from binance.client import Client
    from binance.enums import HistoricalKlinesType

    client = Client()
    
    historical_klines = client.get_historical_klines(trading_pair, interval, start_date, end_date, klines_type=HistoricalKlinesType.FUTURES)
    kline_columns = ["Open_time",
        "Open", "High", "Low", "Close", "Volume",  # OHCL
        "Close_time", "Quote_Asset_volume", "Number_of_trades", "Taker_buy_base_asset_volume","Taker_buy_quote_asset_volume", # Others
        "Unused_field"]

    # Create a Dataframe from binance data
    klines_df = pd.DataFrame(columns = kline_columns)
    klines_df.set_index("Open_time", inplace=True)

    for kline in historical_klines:
        index = kline[0]
        klines_df.loc[index] = kline[1:]

    # Format it properly
    klines_df.index = np.array(klines_df.index).astype('datetime64[ms]')
    klines_df['Open'] = klines_df['Open'].astype(float)
    klines_df['High'] = klines_df['High'].astype(float)
    klines_df['Low'] = klines_df['Low'].astype(float)
    klines_df['Close'] = klines_df['Close'].astype(float)
    klines_df['Volume'] = klines_df['Volume'].astype(float)

    return klines_df
