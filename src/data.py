from datetime import datetime
import pandas_datareader.data as web
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def download_data(start=datetime(1980, 1, 1), end=datetime.today()):
    indicators = {
        "GDP": "GDP",
        "Unemployment": "UNRATE",
        "Interest Rate": "FEDFUNDS",
        "Inflation": "CPIAUCSL"
    }
    df = web.DataReader(list(indicators.values()), "fred", start, end)
    df.columns = indicators.keys()
    df = df.ffill().dropna()
    df["GDP Growth"] = df["GDP"].pct_change()
    df["Inflation Rate"] = df["Inflation"].pct_change()
    df["Recession"] = (df["GDP Growth"] < 0).astype(int)
    df.drop(columns=["GDP", "Inflation"], inplace=True)
    return df.dropna()

def prepare_data(df):
    X = df.drop(columns=["Recession"])
    y = df["Recession"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler
