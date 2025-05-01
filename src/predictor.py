import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
import pandas_datareader.data as web

def descargar_datos():
    indicators = {
        "GDP": "GDP",
        "Unemployment": "UNRATE",
        "Interest Rate": "FEDFUNDS",
        "Inflation": "CPIAUCSL"
    }
    start = datetime(1980, 1, 1)
    end = datetime.today()
    df = web.DataReader(list(indicators.values()), "fred", start, end)
    df.columns = indicators.keys()
    df = df.ffill().dropna()
    df["GDP Growth"] = df["GDP"].pct_change()
    df["Inflation Rate"] = df["Inflation"].pct_change()
    df["Recession"] = (df["GDP Growth"] < 0).astype(int)
    df = df.drop(columns=["GDP", "Inflation"])
    return df.dropna()

def preparar_datos(df):
    X = df.drop(columns=["Recession"])
    y = df["Recession"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler

def construir_modelo(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
