from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from all_algorithms_runner.common import Result, ratio_series, robust_threshold


SEQ_LENGTH = 8


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_layer_size: int = 200, output_size: int = 1, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def _sequences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= SEQ_LENGTH:
        return np.array([]), np.array([])
    x = np.array([values[idx : idx + SEQ_LENGTH] for idx in range(len(values) - SEQ_LENGTH)])
    y = np.array(values[SEQ_LENGTH:])
    return x, y


def load_model(model_dir: Path) -> tuple[LSTMModel, object, object]:
    model = LSTMModel()
    model.load_state_dict(torch.load(model_dir / "lstm_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    return (
        model,
        joblib.load(model_dir / "lstm_scaler_X.pkl"),
        joblib.load(model_dir / "lstm_scaler_y.pkl"),
    )


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str, model_bundle=None) -> list[Result]:
    if model_bundle is None:
        raise ValueError("lstm requires a loaded model bundle")
    model, scaler_x, scaler_y = model_bundle
    rows = []
    for serial, serial_df in df.groupby(level="serialno"):
        values = ratio_series(serial_df).to_numpy()
        x_test, y_test = _sequences(values)
        if x_test.size == 0:
            continue
        x_scaled = scaler_x.transform(x_test.reshape(-1, 1)).reshape(x_test.shape)
        tensor = torch.tensor(x_scaled.reshape(-1, SEQ_LENGTH, 1), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(tensor).numpy()
        pred = scaler_y.inverse_transform(pred_scaled)
        rows.append((str(serial), mean_squared_error(y_test, pred)))

    threshold = robust_threshold([mse for _, mse in rows], sigma=2.0)
    return [
        Result(prefix, day, "lstm", rg_id, serial)
        for serial, mse in rows
        if mse > threshold
    ]
