import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEQ_LEN = 30

categorical_cols = [
    "day_of_week",
    "time_of_day",
    "location_type",
    "weather_condition"
]

features = [
    'total_withdrawals',
    'total_deposits',
    'previous_day_cash_level',
    'nearby_competitor_atms',
    'holiday_flag',
    'special_event_flag',
    'day',
    'month',
    'day_of_week',
    'rolling_7',
    'rolling_30'
]


def load_data(path):
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("date")

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def encode_categorical(df):

    le_dict = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    return df, le_dict


def add_rolling_features(df):

    df["rolling_7"] = df["total_withdrawals"].rolling(7).mean()
    df["rolling_30"] = df["total_withdrawals"].rolling(30).mean()

    df.fillna(method="bfill", inplace=True)

    return df


def scale_features(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def create_sequences(X, seq_len=SEQ_LEN):

    sequences = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i + seq_len])

    return np.array(sequences)
