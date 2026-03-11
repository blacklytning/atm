import numpy as np
import joblib
from tensorflow.keras.models import load_model

from utils.preprocessing import features, create_sequences


# Load models
lr = joblib.load("models/linear_regression_model.joblib")
rf = joblib.load("models/random_forest_model.joblib")
xgb = joblib.load("models/xgboost_model.joblib")

lstm_model = load_model("models/lstm_model.keras")
cnn_model = load_model("models/cnn_model.keras")


def predict_single(X):

    lr_pred = lr.predict(X)
    rf_pred = rf.predict(X)
    xgb_pred = xgb.predict(X)

    return lr_pred, rf_pred, xgb_pred


def predict_sequence(X_seq):

    lstm_pred = lstm_model.predict(X_seq)
    cnn_pred = cnn_model.predict(X_seq)

    return lstm_pred.flatten(), cnn_pred.flatten()


def ensemble_prediction(lr_pred, rf_pred, xgb_pred, lstm_pred, cnn_pred):

    pred = (
        lr_pred +
        rf_pred +
        xgb_pred +
        lstm_pred +
        cnn_pred
    ) / 5

    return pred


def forecast_next_days(df, days=7):

    history = df.copy()

    predictions = []

    for i in range(days):

        X = history[features].values[-1].reshape(1, -1)

        lr_pred, rf_pred, xgb_pred = predict_single(X)

        seq_data = history[features].values
        seq_data = seq_data[-30:]

        seq_data = seq_data.reshape(1, 30, len(features))

        lstm_pred, cnn_pred = predict_sequence(seq_data)

        final_pred = ensemble_prediction(
            lr_pred,
            rf_pred,
            xgb_pred,
            lstm_pred,
            cnn_pred
        )

        predictions.append(final_pred[0])

    return predictions
